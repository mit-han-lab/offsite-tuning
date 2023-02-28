#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
import sys
import datasets
import torch
from torch import nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from copy import deepcopy

import transformers
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Model
from transformers.models.opt.modeling_opt import OPTForCausalLM, OPTModel
from transformers.models.bloom.modeling_bloom import BloomForCausalLM, BloomModel

from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
    get_scheduler,
)
from datasets import load_from_disk
from offsite_tuning.tasks import task_dict
from offsite_tuning.data import get_raw_datasets, get_tokenized_datasets, get_lm_datasets, process_text2text_datasets
from offsite_tuning.utils import (
    MLP,
    add_epilogue,
    add_prologue,
    uniform_choose_layers,
    magnitude_prune,
    quantize,
    parse_args,
    setup_teacher_student,
    get_kd_loss,
    save_state_dict,
    to_student,
    to_teacher
)

from offsite_tuning.param_efficient import (
    use_lora,
    use_bitfit,
    use_adapter
)
import gc

logger = get_logger(__name__)


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    accelerator_log_kwargs["log_with"] = args.report_to
    accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Make one log on every process with the configuration for debugging.
    # also log to a file in output_dir
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.output_dir, "log.txt"))
        ] if accelerator.is_main_process else []
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning(
            "You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            torch_dtype=torch.float16
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if args.dataset_name in task_dict:  # special case for e2e_nlg dataset
        raw_datasets = get_raw_datasets(args)
        lm_datasets = process_text2text_datasets(
            raw_datasets, args, tokenizer, accelerator)
    else:
        if args.train_tokenized_dataset and args.val_tokenized_dataset:
            tokenized_datasets = load_from_disk(args.train_tokenized_dataset)
            val_dataset = load_from_disk(args.val_tokenized_dataset)
            if 'validation' in val_dataset:
                tokenized_datasets["validation"] = val_dataset['validation']
            else:
                tokenized_datasets["validation"] = val_dataset['train']
        else:
            raw_datasets = get_raw_datasets(args)

            tokenized_datasets = get_tokenized_datasets(
                raw_datasets, args, accelerator, tokenizer)

        lm_datasets = get_lm_datasets(
            tokenized_datasets, args, accelerator, tokenizer)

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    if args.train_num_samples is not None:
        # check if we have enough samples for the training set
        if args.train_num_samples > len(train_dataset):
            args.train_num_samples = len(train_dataset)
        train_dataset = train_dataset.select(
            range(args.train_num_samples))

    if args.validation_num_samples is not None:
        # check if we have enough samples for the validation set
        if args.validation_num_samples > len(eval_dataset):
            args.validation_num_samples = len(eval_dataset)
        eval_dataset = eval_dataset.select(
            range(args.validation_num_samples))

    collator = default_data_collator
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=collator, batch_size=args.per_device_eval_batch_size
    )

    model = setup_teacher_student(model, args, accelerator)

    if args.no_teacher:
        model.teacher = None
        to_student(model, args)
        gc.collect()
        torch.cuda.empty_cache()

    if args.train_module in ['adapter', 'all'] and args.train_lm_head:
        for param in model.lm_head.parameters():
            param.requires_grad = True
            param.data = param.data.float()

    if args.use_lora:
        use_lora(model.trainable_module, args.lora_rank, args.lora_alpha)

    if args.use_adapter:
        use_adapter(model.trainable_module, args.adapter_size)

    if args.use_bitfit:
        use_bitfit(model.trainable_module)

    if args.load_student and not args.restart_training:
        base_results = json.load(
            open(os.path.join(args.load_student, 'all_results.json'), 'r'))
        starting_epoch = base_results['epoch']
        resume_step = base_results['step'] - \
            starting_epoch * len(train_dataloader)
    else:
        starting_epoch = 0
        resume_step = -1

    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    logger.info(f"Number of trainable parameters: {trainable_params}")

    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(
                f"Trainable parameter: {name} with shape {param.shape} and dtype {param.dtype}")

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    experiment_config = vars(args)
    # TensorBoard cannot log Enums, need the raw value
    experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
    accelerator.init_trackers("offsite_tuning", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    def eval_epoch():
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(
                loss.repeat(args.per_device_eval_batch_size)).cpu())
        losses = torch.cat(losses).flatten()
        # filter out nan
        losses = losses[~torch.isnan(losses)]
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        return eval_loss, perplexity

    if not args.no_teacher:
        to_teacher(model.module, args)
        _, teacher_zero_shot_perplexity = eval_epoch()
        logger.info(
            f"Teacher zero shot perplexity: {teacher_zero_shot_perplexity}")
    else:
        teacher_zero_shot_perplexity = 0

    to_student(model.module, args)

    # for name, param in model.named_parameters():
    #     logger.info(
    #         f"Parameter: {name} with shape {param.shape}, dtype {param.dtype}, and requires_grad {param.requires_grad}")

    _, student_zero_shot_perplexity = eval_epoch()
    logger.info(
        f"Student zero shot perplexity: {student_zero_shot_perplexity}")
    best_perplexity = float("inf")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps),
                        disable=not accelerator.is_local_main_process)

    completed_steps = 0

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_lm_loss, total_kd_loss = 0, 0
        interval_lm_loss, interval_kd_loss = 0, 0
        best_lm_loss, best_kd_loss = float("inf"), float("inf")
        skipped_steps = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.load_student and epoch == starting_epoch and step <= resume_step:
                progress_bar.update(1)
                progress_bar.set_description(
                    f"Skipping step {step} (already completed)")
                completed_steps += 1
                skipped_steps += 1
                continue

            with accelerator.accumulate(model):
                outputs = model(**batch)
                lm_loss = outputs.loss
                if not args.no_teacher:
                    kd_loss = get_kd_loss(model.module)
                else:
                    kd_loss = 0

                loss = args.lm_weight * lm_loss + args.kd_weight * \
                    kd_loss if args.kd_weight != 0 else lm_loss
                progress_bar.set_description(
                    f"Epoch {epoch} - Step {step} - LR: {optimizer.param_groups[0]['lr']:.2e} - LM loss: {lm_loss:.4f} - KD loss: {kd_loss:.4f}")

                total_lm_loss += lm_loss.item()
                interval_lm_loss += lm_loss.item()
                best_lm_loss = min(best_lm_loss, lm_loss.item())

                if not args.no_teacher:
                    total_kd_loss += kd_loss.item()
                    interval_kd_loss += kd_loss.item()
                    best_kd_loss = min(best_kd_loss, kd_loss.item())

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            # end accumulate gradients

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
            else:
                continue

            if completed_steps % args.eval_steps == 0:
                if not args.no_teacher:
                    to_teacher(model.module, args)
                    plug_eval_loss, plug_ppl = eval_epoch()
                else:
                    plug_eval_loss, plug_ppl = 0, 0
                to_student(model.module, args)
                eval_loss, perplexity = eval_epoch()

                lm_loss = interval_lm_loss / args.eval_steps
                kd_loss = interval_kd_loss / args.eval_steps
                interval_lm_loss = 0
                interval_kd_loss = 0

                logger.info(
                    f"epoch {epoch} step {completed_steps}: student_ppl: {perplexity:.4f} plug_ppl: {plug_ppl:.4f} lm_loss: {lm_loss:.4f} kd_loss: {kd_loss:.4f}")

                accelerator.log(
                    {
                        "student_ppl": perplexity,
                        "student_eval_loss": eval_loss,
                        "plug_ppl": plug_ppl,
                        "plug_eval_loss": plug_eval_loss,
                        "ppl_gap": perplexity - plug_ppl,
                        "train_lm_loss": lm_loss,
                        "train_kd_loss": kd_loss,
                        "epoch": epoch,
                        "step": completed_steps,
                    },
                    step=completed_steps,
                )
                is_best = perplexity < best_perplexity
                best_perplexity = min(best_perplexity, perplexity)

                if not args.no_save_model and is_best and accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    if args.save_module in ["student", "all"]:
                        state_dict = unwrapped_model.student.state_dict()
                        save_state_dict(
                            state_dict, args.output_dir, "student.pt")
                    if args.save_module in ["adapter", "all"]:
                        state_dict = unwrapped_model.adapter.state_dict()
                        save_state_dict(
                            state_dict, args.output_dir, "adapter.pt")

                    gc.collect()
                    torch.cuda.empty_cache()

                if is_best and accelerator.is_main_process:
                    with open(os.path.join(args.output_dir, "all_results.json"), "w+") as f:
                        json.dump({"best_perplexity": best_perplexity,
                                   "plug_perplexity": plug_ppl,
                                   "teacher_zero_shot_perplexity": teacher_zero_shot_perplexity,
                                   "student_zero_shot_perplexity": student_zero_shot_perplexity,
                                   "train_lm_loss": lm_loss,
                                   "train_kd_loss": kd_loss,
                                   "epoch": epoch,
                                   "step": completed_steps,
                                   "trainable_params": trainable_params}, f)

    accelerator.end_training()


if __name__ == "__main__":
    main()
