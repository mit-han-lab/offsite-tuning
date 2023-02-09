from datasets import load_dataset
import logging
from itertools import chain
from offsite_tuning.tasks import task_dict, map_dataset_name_and_config

logger = logging.getLogger(__name__)


def get_raw_datasets(args):
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset_name, dataset_config_name = map_dataset_name_and_config(args)
        raw_datasets = load_dataset(
            dataset_name, dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                dataset_name,
                dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                dataset_name,
                dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        elif extension == 'zst':
            extension = 'json'
        raw_datasets = load_dataset(
            extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )
    return raw_datasets


def get_tokenized_datasets(raw_datasets, args, accelerator, tokenizer, lm_type='clm'):
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        if lm_type == 'clm':
            return tokenizer(examples[text_column_name])
        elif lm_type == 'mlm':
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)
        else:
            raise ValueError(f'lm_type {lm_type} not supported')

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    return tokenized_datasets


def _get_block_size(args, tokenizer):
    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)
    return block_size


def get_lm_datasets(tokenized_datasets, args, accelerator, tokenizer, lm_type='clm'):
    block_size = _get_block_size(args, tokenizer)
    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size]
                for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        if lm_type == 'clm':
            result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
    return lm_datasets


def process_text2text_datasets(raw_datasets, args, tokenizer, accelerator):
    task = task_dict[args.dataset_name]

    column_names = raw_datasets["train"].column_names

    def tokenize_function(examples):
        context = task.get_context(examples)
        target = task.get_target(examples)

        context = tokenizer(context)
        target = tokenizer(target)

        # if context is ending with special token, remove it
        if len(context['input_ids'][0]) > 0 and context['input_ids'][0][-1] in tokenizer.all_special_ids:
            context['input_ids'] = [i[:-1] for i in context['input_ids']]
            context['attention_mask'] = [a[:-1]
                                         for a in context['attention_mask']]

        # if target is starting with special token, remove it
        if len(target['input_ids'][0]) > 0 and target['input_ids'][0][0] in tokenizer.all_special_ids:
            target['input_ids'] = [i[1:] for i in target['input_ids']]
            target['attention_mask'] = [a[1:]
                                        for a in target['attention_mask']]

        out = {}
        out['input_ids'] = [i1 + i2 for i1,
                            i2 in zip(context['input_ids'], target['input_ids'])]
        out['attention_mask'] = [a1 + a2 for a1,
                                 a2 in zip(context['attention_mask'], target['attention_mask'])]

        # set -100 for context tokens
        out["labels"] = [
            [-100] * len(i1) + i2 for i1, i2 in zip(context['input_ids'], target['input_ids'])]

        return out

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if "gpt2" in args.model_name_or_path:
        tokenizer.pad_token = tokenizer.bos_token

    # pad all instances in lm_datasets to the max length of the dataset
    max_length = -1
    for v in tokenized_datasets.values():
        for x in v:
            max_length = max(max_length, len(x['input_ids']))

    # pad to the multiple of 8
    max_length = (max_length // 8 + 1) * 8

    block_size = _get_block_size(args, tokenizer)
    max_length = min(max_length, block_size)

    def pad_function(examples):
        examples["input_ids"] = [i + [tokenizer.pad_token_id] *
                                 (max_length - len(i)) for i in examples["input_ids"]]
        examples["attention_mask"] = [[1] * len(i) + [0] *
                                      (max_length - len(i)) for i in examples["attention_mask"]]
        examples["labels"] = [i + [-100] *
                              (max_length - len(i)) for i in examples["labels"]]
        # truncate to max_length
        examples["input_ids"] = [i[:max_length] for i in examples["input_ids"]]
        examples["attention_mask"] = [a[:max_length]
                                      for a in examples["attention_mask"]]
        examples["labels"] = [l[:max_length] for l in examples["labels"]]
        return examples

    with accelerator.main_process_first():
        tokenized_datasets = tokenized_datasets.map(
            pad_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Padding dataset to max length {max_length}",
        )

    return tokenized_datasets
