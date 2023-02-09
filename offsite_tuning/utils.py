import gc
import os
from copy import deepcopy
import torch
from torch import nn
from accelerate.logging import get_logger
from transformers import (
    SchedulerType,
    MODEL_MAPPING,
    OPTForCausalLM,
    GPT2LMHeadModel,
    BloomForCausalLM,
    ViTForImageClassification,
)
from offsite_tuning.models.clip_vit import CLIPViTForImageClassification
from offsite_tuning.models.eva_vit import EVAViTForImageClassification

import argparse


MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


logger = get_logger(__name__)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, activation=nn.ReLU):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation = activation()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x


def add_prologue(module, prologue):
    module.old_forward = module.forward
    module.prologue = prologue

    def new_forward(self):
        def lambda_forward(*args, **kwargs):
            self.input_args = args
            self.input_kwargs = kwargs
            if self.prologue is not None:
                x = self.prologue(args[0])
            else:
                x = args[0]
            args = (x,) + args[1:]
            return self.old_forward(*args, **kwargs)
        return lambda_forward
    module.forward = new_forward(module)
    return module


def add_epilogue(module, epilogue):
    module.old_forward = module.forward
    module.epilogue = epilogue

    def new_forward(self):
        def lambda_forward(*args, **kwargs):
            output = self.old_forward(*args, **kwargs)
            if isinstance(output, tuple):
                x = output[0]
            else:
                x = output

            if self.epilogue is not None:
                x = self.epilogue(x)

            if isinstance(output, tuple):
                output = (x,) + output[1:]
            else:
                output = x

            self.cached_output = x
            return output
        return lambda_forward
    module.forward = new_forward(module)
    return module


def uniform_choose_layers(layers: nn.ModuleList, num_student_layers=None):
    if num_student_layers is None:
        num_student_layers = len(layers)

    student = nn.ModuleList()
    stride = (len(layers) - 1) / (num_student_layers - 1)

    for i in range(num_student_layers):
        idx = round(i * stride)
        logger.info(f"Adding layer {idx} to student")
        student.append(layers[idx])

    return student


@torch.no_grad()
def magnitude_prune(model, ratio):
    for param in model.parameters():
        if param.dim() == 1:
            continue
        num_prune = int(param.numel() * ratio)
        threshold = param.abs().view(-1).kthvalue(num_prune).values.item()
        mask = (param.abs() >= threshold).to(param.dtype)
        param.mul_(mask)


@torch.no_grad()
def quantize(model, bits):
    for param in model.parameters():
        if param.dim() == 1:
            continue
        min, max = param.min(), param.max()
        zp = (max + min) / 2
        scale = (max - min) / (2 ** bits - 1)
        param.sub_(zp).div_(scale).round_().mul_(scale).add_(zp)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        type=int,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adamw',
        help='Optimizer to use. Can be adamw or sgd',
        choices=['adamw', 'sgd']
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float,
                        default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum to use for sgd optimizer."
    )
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts",
                 "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=88,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str,
                        help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        '--no_save_model',
        action='store_true',
        help='Whether or not to save the model.'
    )
    parser.add_argument(
        '--kd_weight',
        type=float,
        default=0.0,
        help='Weight of the knowledge distillation loss.'
    )
    parser.add_argument(
        '--lm_weight',
        type=float,
        default=1.0,
        help='Weight of the knowledge distillation loss.'
    )
    parser.add_argument(
        '--train_tokenized_dataset',
        type=str,
        default=None,
        help='Path to the tokenized training dataset.'
    )
    parser.add_argument(
        '--val_tokenized_dataset',
        type=str,
        default=None,
        help='Path to the tokenized validation dataset.'
    )
    parser.add_argument(
        "--train_num_samples",
        type=int,
        default=None,
        help="The number of samples to use for training set.",
    )
    parser.add_argument(
        "--validation_num_samples",
        type=int,
        default=None,
        help="The number of samples to use for validation set.",
    )
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=200,
    )

    parser.add_argument(
        '--num_student_layers',
        type=int,
        default=None,
        help='Number of layers in the student model.'
    )

    parser.add_argument(
        '--load_student',
        type=str,
        default=None,
        help='Path to the student model'
    )

    parser.add_argument(
        '--student_l_pad',
        type=int,
        default=0,
    )

    parser.add_argument(
        '--student_r_pad',
        type=int,
        default=0,
    )

    parser.add_argument(
        '--student_layer_selection_strategy',
        type=str,
        default='uniform',
        help='Layer selection strategy',
        choices=['uniform', 'random', 'changes']
    )

    parser.add_argument(
        '--restart_training',
        action='store_true',
        help='Whether to restart training of all dataset.'
    )

    parser.add_argument(
        '--train_module',
        type=str,
        default='student',
        help='Part of the model to train.',
        choices=['student', 'adapter', 'all']
    )
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=1.0,
        help='Max gradient norm.'
    )

    parser.add_argument(
        '--magnitude_pruning_ratio',
        type=float,
        default=0.0,
        help='Magnitude pruning ratio.'
    )

    parser.add_argument(
        '--weight_quantization_bits',
        type=int,
        default=None,
        help='Weight quantization bits.'
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )

    # vit
    parser.add_argument("--train_dir", type=str, default=None,
                        help="A folder containing the training data.")
    parser.add_argument("--validation_dir", type=str, default=None,
                        help="A folder containing the validation data.")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=0.15,
        help="Percent to split off of train for validation",
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
    )

    parser.add_argument(
        '--freeze_bottom',
        action='store_true',
    )
    
    parser.add_argument(
        '--no_teacher',
        action='store_true',
    )

    parser.add_argument(
        '--classifier_lr_multiplier',
        type=float,
        default=1.0,
    )
    
    parser.add_argument(
        '--select_by_kd',
        action='store_true',
    )
    
    parser.add_argument(
        '--use_pt_imagefolder',
        action='store_true',
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=12,
    )

    parser.add_argument(
        '--train_lm_head',
        action='store_true',
    )
    parser.add_argument(
        '--save_module',
        type=str,
        default='student',
        choices=['student', 'adapter', 'all']
    )
    
    parser.add_argument(
        '--load_adapter',
        type=str,
        default=None,
        help='Path to the student model'
    )

    parser.add_argument(
        '--tasks',
        type=str,
        default='piqa',
        help='Evaluation tasks',
    )
    
    parser.add_argument(
        '--use_adapter',
        action='store_true',
    )
    
    parser.add_argument(
        '--use_lora',
        action='store_true',
    )
    
    parser.add_argument(
        '--use_bitfit',
        action='store_true',
    )
    
    parser.add_argument(
        '--lora_rank',
        type=int,
        default=4,
        help='Rank of the LoRA matrix',
    )

    parser.add_argument(
        '--lora_alpha',
        type=float,
        default=32,
        help='Alpha of the LoRA matrix',
    )

    parser.add_argument(
        '--adapter_size',
        type=int,
        default=64,
        help='Size of the adapter',
    )

    parser.add_argument
    args = parser.parse_args()

    return args

def get_layers(model):
    if isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, GPT2LMHeadModel):
        layers = model.transformer.h
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif isinstance(model, ViTForImageClassification):
        layers = model.vit.encoder.layer
    elif isinstance(model, CLIPViTForImageClassification):
        layers = model.vit.encoder.layers
    elif isinstance(model, EVAViTForImageClassification):
        layers = model.blocks
    else:
        raise NotImplementedError
    return layers

def set_layers(model, layers):
    if isinstance(model, OPTForCausalLM):
        model.model.decoder.layers = layers
    elif isinstance(model, GPT2LMHeadModel):
        model.transformer.h = layers
    elif isinstance(model, BloomForCausalLM):
        model.transformer.h = layers
    elif isinstance(model, ViTForImageClassification):
        model.vit.encoder.layer = layers
    elif isinstance(model, CLIPViTForImageClassification):
        model.vit.encoder.layers = layers
    elif isinstance(model, EVAViTForImageClassification):
        model.blocks = layers
    else:
        raise NotImplementedError


def setup_teacher_student(model, args, accelerator):
    for param in model.parameters():
        param.requires_grad = False

    layers = get_layers(model)

    l, r = args.student_l_pad, len(layers) - args.student_r_pad
    if args.load_student:
        student_state_dict = torch.load(os.path.join(
            args.load_student, 'student.pt'), map_location='cpu')
        student_layers_len = len(
            set([k.split('.')[0] for k in student_state_dict.keys()]))
        logger.info(
            f"Loading student module from {args.load_student} with {student_layers_len} layers.")
        student = deepcopy(layers[:student_layers_len])
        student.load_state_dict(student_state_dict)
    else:
        student = deepcopy(layers[l:r])

    if args.student_layer_selection_strategy == 'uniform':
        student = uniform_choose_layers(student, args.num_student_layers)
    else:
        raise NotImplementedError

    student = student.to(accelerator.device)

    if args.magnitude_pruning_ratio > 0:
        logger.info(
            f"Pruning student module with magnitude ratio {args.magnitude_pruning_ratio}")
        magnitude_prune(student, args.magnitude_pruning_ratio)

    if args.weight_quantization_bits is not None:
        logger.info(
            f"Quantizing student module with {args.weight_quantization_bits} bits")
        quantize(student, args.weight_quantization_bits)

    if args.train_module == 'student':
        for param in student.parameters():
            param.data = param.data.float()
            param.requires_grad = True
    elif args.train_module == 'adapter':
        for param in student.parameters():
            param.requires_grad = False
        if not args.freeze_bottom:
            for param in layers[:l].parameters():
                param.data = param.data.float()
                param.requires_grad = True
        for param in layers[r:].parameters():
            param.data = param.data.float()
            param.requires_grad = True
    elif args.train_module == 'all':
        for param in student.parameters():
            param.data = param.data.float()
            param.requires_grad = True
        for param in layers[:l].parameters():
            param.data = param.data.float()
            param.requires_grad = True
        for param in layers[r:].parameters():
            param.data = param.data.float()
            param.requires_grad = True
    else:
        raise NotImplementedError

    model.student = student

    model.teacher = layers[l:r].half()
    
    model.adapter = layers[:l] + layers[r:]

    for param in model.teacher.parameters():
        param.requires_grad = False

    add_prologue(model.student[0], None)
    add_epilogue(model.student[-1], None)
    model.student_l = model.student[0]
    model.student_r = model.student[-1]

    num_student_layers = len(model.student)
    logger.info(f"Number of student layers: {num_student_layers}")

    if args.train_module == 'student':
        model.trainable_module = model.student
    elif args.train_module == 'adapter':
        model.trainable_module = model.adapter
    elif args.train_module == 'all':
        model.trainable_module = model.student + model.adapter
    else:
        raise NotImplementedError

    gc.collect()
    torch.cuda.empty_cache()
    return model


def to_teacher(model, args):
    l = args.student_l_pad
    if isinstance(model, OPTForCausalLM):
        r = len(model.model.decoder.layers) - args.student_r_pad
        model.model.decoder.layers = model.model.decoder.layers[
            :l] + model.teacher + model.model.decoder.layers[r:]
    elif isinstance(model, GPT2LMHeadModel):
        r = len(model.transformer.h) - args.student_r_pad
        model.transformer.h = model.transformer.h[:l] + \
            model.teacher + model.transformer.h[r:]
    elif isinstance(model, BloomForCausalLM):
        r = len(model.transformer.h) - args.student_r_pad
        model.transformer.h = model.transformer.h[:l] + \
            model.teacher + model.transformer.h[r:]
    elif isinstance(model, ViTForImageClassification):
        r = len(model.vit.encoder.layer) - args.student_r_pad
        model.vit.encoder.layer = model.vit.encoder.layer[:l] + \
            model.teacher + model.vit.encoder.layer[r:]
    elif isinstance(model, CLIPViTForImageClassification):
        r = len(model.vit.encoder.layers) - args.student_r_pad
        model.vit.encoder.layers = model.vit.encoder.layers[:l] + \
            model.teacher + model.vit.encoder.layers[r:]
    elif isinstance(model, EVAViTForImageClassification):
        r = len(model.blocks) - args.student_r_pad
        model.blocks = model.blocks[:l] + \
            model.teacher + model.blocks[r:]
    else:
        raise NotImplementedError


def to_student(model, args):
    l = args.student_l_pad
    if isinstance(model, OPTForCausalLM):
        r = len(model.model.decoder.layers) - args.student_r_pad
        model.model.decoder.layers = model.model.decoder.layers[
            :l] + model.student + model.model.decoder.layers[r:]
    elif isinstance(model, GPT2LMHeadModel):
        r = len(model.transformer.h) - args.student_r_pad
        model.transformer.h = model.transformer.h[:l] + \
            model.student + model.transformer.h[r:]
    elif isinstance(model, BloomForCausalLM):
        r = len(model.transformer.h) - args.student_r_pad
        model.transformer.h = model.transformer.h[:l] + \
            model.student + model.transformer.h[r:]
    elif isinstance(model, ViTForImageClassification):
        r = len(model.vit.encoder.layer) - args.student_r_pad
        model.vit.encoder.layer = model.vit.encoder.layer[:l] + \
            model.student + model.vit.encoder.layer[r:]
    elif isinstance(model, CLIPViTForImageClassification):
        r = len(model.vit.encoder.layers) - args.student_r_pad
        model.vit.encoder.layers = model.vit.encoder.layers[:l] + \
            model.student + model.vit.encoder.layers[r:]
    elif isinstance(model, EVAViTForImageClassification):
        r = len(model.blocks) - args.student_r_pad
        model.blocks = model.blocks[:l] + \
            model.student + model.blocks[r:]
    else:
        raise NotImplementedError


def get_kd_loss(model):
    kwargs = model.student_l.input_kwargs
    args = model.student_l.input_args
    output_teacher = args[0].to(torch.float16)
    args = list(args[1:])
    for i, arg in enumerate(args):
        if torch.is_tensor(arg) and arg.dtype == torch.float32:
            args[i] = arg.to(torch.float16)
    args = tuple(args)

    for k, v in kwargs.items():
        if torch.is_tensor(v) and v.dtype == torch.float32:
            kwargs[k] = v.to(torch.float16)

    with torch.no_grad():
        model.teacher.eval()
        for teacher_layer in model.teacher:
            output_teacher = teacher_layer(output_teacher, *args, **kwargs)
            if isinstance(output_teacher, tuple):
                output_teacher = output_teacher[0]

    output_student = model.student_r.cached_output.float()
    output_teacher = output_teacher.float()

    std = output_teacher.pow(2).mean().sqrt()
    kd_loss = (output_teacher - output_student).div(std).pow(2).mean()
    return kd_loss


def setup_trainable_classification_head(model):
    # Setup trainable classification heads
    if isinstance(model, ViTForImageClassification):
        for param in model.classifier.parameters():
            param.requires_grad = True
            param.data = param.data.float()
    elif isinstance(model, CLIPViTForImageClassification):
        for param in model.classifier.parameters():
            param.requires_grad = True
            param.data = param.data.float()
    elif isinstance(model, EVAViTForImageClassification):
        for param in model.classifier.parameters():
            param.requires_grad = True
            param.data = param.data.float()
    else:
        raise NotImplementedError

def load_adapter(model, adapter_state_dict, args):
    l = args.student_l_pad
    if isinstance(model, OPTForCausalLM):
        r = len(model.model.decoder.layers) - args.student_r_pad
        adapter_layers = model.model.decoder.layers[:l] + model.model.decoder.layers[r:]
        adapter_layers.load_state_dict(adapter_state_dict)
    elif isinstance(model, GPT2LMHeadModel):
        r = len(model.transformer.h) - args.student_r_pad
        adapter_layers = model.transformer.h[:l] + model.transformer.h[r:]
        adapter_layers.load_state_dict(adapter_state_dict)
    elif isinstance(model, BloomForCausalLM):
        r = len(model.transformer.h) - args.student_r_pad
        adapter_layers = model.transformer.h[:l] + model.transformer.h[r:]
        adapter_layers.load_state_dict(adapter_state_dict)
    else:
        raise NotImplementedError
    return model

def load_student(model, student_state_dict, args):
    l = args.student_l_pad
    
    student_layers_len = len(
        set([k.split('.')[0] for k in student_state_dict.keys()]))
    logger.info(f"Loading student module from with {student_layers_len} layers.")
    if isinstance(model, OPTForCausalLM):
        r = len(model.model.decoder.layers) - args.student_r_pad
        student_layers = model.model.decoder.layers[l:l+student_layers_len]
        student_layers.load_state_dict(student_state_dict)
        model.model.decoder.layers = model.model.decoder.layers[:l] + \
            student_layers + model.model.decoder.layers[r:]
    elif isinstance(model, GPT2LMHeadModel):
        r = len(model.transformer.h) - args.student_r_pad
        student_layers = model.transformer.h[l:l+student_layers_len]
        student_layers.load_state_dict(student_state_dict)
        model.transformer.h = model.transformer.h[:l] + \
            student_layers + model.transformer.h[r:]
    elif isinstance(model, BloomForCausalLM):
        r = len(model.transformer.h) - args.student_r_pad
        student_layers = model.transformer.h[l:l+student_layers_len]
        student_layers.load_state_dict(student_state_dict)
        model.transformer.h = model.transformer.h[:l] + \
            student_layers + model.transformer.h[r:]
    else:
        raise NotImplementedError
    return model

def save_state_dict(state_dict, output_dir, filename):
    for k in state_dict:
        state_dict[k] = state_dict[k].to(torch.float16).cpu()
    torch.save(state_dict, os.path.join(output_dir, filename))