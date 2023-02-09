# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning any 🤗 Transformers model for image classification leveraging 🤗 Accelerate."""
import argparse
import json
import logging
import math
import os
import sys

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from tqdm.auto import tqdm

import evaluate
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    get_scheduler,
    CLIPVisionConfig,
    CLIPVisionModel,
)


from offsite_tuning.utils import (
    parse_args,
    setup_teacher_student,
    get_kd_loss,
    to_teacher,
    to_student,
    setup_trainable_classification_head
)

from offsite_tuning.models.clip_vit import CLIPViTForImageClassification
from offsite_tuning.models.eva_vit import EVAViTForImageClassification
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
    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    logger.info(accelerator.state)
    # Make one log on every process with the configuration for debugging.
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

    accelerator.wait_for_everyone()

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.model_name_or_path)

    # Preprocessing the datasets
    # Define torchvision transforms to be applied to each image.
    if "shortest_edge" in feature_extractor.size:
        size = feature_extractor.size["shortest_edge"]
    else:
        size = (feature_extractor.size["height"],
                feature_extractor.size["width"])

    normalize = Normalize(mean=feature_extractor.image_mean,
                          std=feature_extractor.image_std)

    train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    def preprocess_train(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [train_transforms(
            image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    def preprocess_val(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [val_transforms(
            image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(args.dataset_name, task="image-classification")
    elif args.use_pt_imagefolder:
        # Load a local dataset using a PyTorch Dataset.
        import torchvision.datasets as pt_datasets
        logging.info("Using PyTorch ImageFolder")
        dataset = {
            "train": pt_datasets.ImageFolder(root=args.train_dir, transform=train_transforms),
            "validation": pt_datasets.ImageFolder(root=args.validation_dir, transform=val_transforms),
        }
    else:
        data_files = {}
        if args.train_dir is not None:
            data_files["train"] = os.path.join(args.train_dir, "**")
        if args.validation_dir is not None:
            data_files["validation"] = os.path.join(args.validation_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            task="image-classification",
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/image_process#imagefolder.

    # If we don't have a validation split, split off a percentage of train as validation.
    args.train_val_split = None if "validation" in dataset.keys() else args.train_val_split
    if isinstance(args.train_val_split, float) and args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.

    if args.use_pt_imagefolder:
        labels = dataset["train"].classes
    else:
        labels = dataset["train"].features["labels"].names

    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}

    # Load pretrained model and feature extractor
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if 'CLIP' in args.model_name_or_path:
        config = CLIPVisionConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=len(labels),
            i2label=id2label,
            label2id=label2id,
            finetuning_task="image-classification",
        )
        model = CLIPVisionModel.from_pretrained(
            args.model_name_or_path,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
            torch_dtype=torch.float16
        )
        model = CLIPViTForImageClassification(config, model.vision_model)
    elif 'eva' in args.model_name_or_path:
        config = json.load(
            open(os.path.join(args.model_name_or_path, 'config.json')))
        config['num_labels'] = len(labels)
        model = EVAViTForImageClassification(**config)
        state_dict = torch.load(os.path.join(
            args.model_name_or_path, 'pytorch_model.bin'))
        model.load_state_dict(state_dict, strict=False)
    else:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=len(labels),
            i2label=id2label,
            label2id=label2id,
            finetuning_task="image-classification",
        )
        model = AutoModelForImageClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
            torch_dtype=torch.float16
        )

    #############################################
    # Teacher-Student model
    model = setup_teacher_student(model, args, accelerator)

    # Setup trainable classification heads
    if args.train_module in ['adapter', 'all']:
        setup_trainable_classification_head(model)
    #############################################

    if args.use_pt_imagefolder:
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]

        def collate_fn(examples):
            pixel_values = torch.stack([example[0] for example in examples])
            labels = torch.tensor([example[1] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}
    else:
        with accelerator.main_process_first():
            if args.max_train_samples is not None:
                dataset["train"] = dataset["train"].shuffle(
                    seed=args.seed).select(range(args.max_train_samples))
            # Set the training transforms
            train_dataset = dataset["train"].with_transform(preprocess_train)
            if args.max_eval_samples is not None:
                dataset["validation"] = dataset["validation"].shuffle(
                    seed=args.seed).select(range(args.max_eval_samples))
            # Set the validation transforms
            eval_dataset = dataset["validation"].with_transform(preprocess_val)

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"]
                                        for example in examples])
            labels = torch.tensor([example["labels"] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}

    # DataLoaders creation:

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size, num_workers=args.num_workers
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size, num_workers=args.num_workers
    )

    if args.load_student and not args.restart_training:
        base_results = json.load(
            open(os.path.join(args.load_student, 'all_results.json'), 'r'))
        starting_epoch = base_results['epoch']
        resume_step = base_results['step'] - \
            starting_epoch * len(train_dataloader)
    else:
        starting_epoch = 0
        resume_step = -1

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and "classifier" not in n],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and "classifier" not in n],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate * args.classifier_lr_multiplier
        },
        {
            "params": [p for n, p in model.classifier.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.learning_rate * args.classifier_lr_multiplier
        },
    ]

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            optimizer_grouped_parameters, lr=args.learning_rate, momentum=args.momentum)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer type {args.optimizer}")

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

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    experiment_config = vars(args)
    # TensorBoard cannot log Enums, need the raw value
    experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
    accelerator.init_trackers("offsite_tuning", experiment_config)

    # Get the metric function
    metric = evaluate.load("accuracy")

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
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics(
                (predictions, batch["labels"]))
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        return eval_metric["accuracy"]

    if args.select_by_kd:
        teacher_zero_shot_acc = student_zero_shot_acc = 0
        model = to_student(model, args)
    else:
        model = to_teacher(model, args)
        teacher_zero_shot_acc = eval_epoch()

        model = to_student(model, args)
        student_zero_shot_acc = eval_epoch()

    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    logger.info(f"Number of trainable parameters: {trainable_params}")

    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(
                f"Trainable parameter: {name} with shape {param.shape} and dtype {param.dtype}")

    logger.info(
        f"Teacher zero shot accuracy: {teacher_zero_shot_acc}")
    logger.info(
        f"Student zero shot accuracy: {student_zero_shot_acc}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    def evaluator(model):
        if evaluator.eval_steps == 0:
            return

        task_loss = evaluator.interval_task_loss / evaluator.eval_steps
        kd_loss = evaluator.interval_kd_loss / evaluator.eval_steps
        evaluator.interval_task_loss = 0
        evaluator.interval_kd_loss = 0
        evaluator.eval_steps = 0

        if args.select_by_kd:
            is_best = kd_loss < evaluator.best_kd_loss
            evaluator.best_kd_loss = min(evaluator.best_kd_loss, kd_loss)
            eval_acc = plug_acc = 0
        else:
            model = to_teacher(model, args)
            plug_acc = eval_epoch()
            model = to_student(model, args)
            eval_acc = eval_epoch()
            is_best = eval_acc > evaluator.best_acc
            evaluator.best_acc = max(evaluator.best_acc, eval_acc)

        logger.info(
            f"Epoch {epoch} step {completed_steps}: eval_acc: {eval_acc:.4f} plug_acc: {plug_acc:.4f} task_loss: {task_loss:.4f} kd_loss: {kd_loss:.4f}")

        accelerator.log(
            {
                "eval_acc": eval_acc,
                "plug_acc": plug_acc,
                "acc_gap": plug_acc - eval_acc,
                "train_task_loss": task_loss,
                "train_kd_loss": kd_loss,
                "epoch": epoch,
                "step": completed_steps,
            },
            step=completed_steps,
        )
        if not args.no_save_model and is_best and accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = unwrapped_model.student.state_dict()
            for k in state_dict:
                state_dict[k] = state_dict[k].to(torch.float16).cpu()
            torch.save(state_dict, os.path.join(
                args.output_dir, "student.pt"))
            gc.collect()
            torch.cuda.empty_cache()

        if is_best and accelerator.is_main_process:
            with open(os.path.join(args.output_dir, "all_results.json"), "w+") as f:
                json.dump({"best_acc": eval_acc,
                           "plug_acc": plug_acc,
                           "teacher_zero_shot_acc": teacher_zero_shot_acc,
                           "student_zero_shot_acc": student_zero_shot_acc,
                           "train_task_loss": task_loss,
                           "train_kd_loss": kd_loss,
                           "epoch": epoch,
                           "step": completed_steps,
                           "trainable_params": trainable_params}, f)

    evaluator.best_acc = student_zero_shot_acc
    evaluator.best_kd_loss = float("inf")
    evaluator.eval_steps = 0
    evaluator.interval_task_loss = 0
    evaluator.interval_kd_loss = 0

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_task_loss, total_kd_loss = 0, 0
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
                task_loss = outputs.loss

                kd_loss = get_kd_loss(model)

                loss = args.lm_weight * task_loss + args.kd_weight * \
                    kd_loss if args.kd_weight != 0 else task_loss
                progress_bar.set_description(
                    f"Epoch {epoch} - Step {step} - LR: {optimizer.param_groups[0]['lr']:.2e} - Task loss: {task_loss:.4f} - KD loss: {kd_loss:.4f}")

                total_task_loss += task_loss.item()
                total_kd_loss += kd_loss.item()

                evaluator.interval_task_loss += task_loss.item()
                evaluator.interval_kd_loss += kd_loss.item()
                evaluator.eval_steps += 1

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
                evaluator(model)

        evaluator(model)

    accelerator.end_training()


if __name__ == "__main__":
    main()
