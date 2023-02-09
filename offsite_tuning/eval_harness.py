import os
from offsite_tuning.utils import parse_args, load_adapter, load_student, get_layers, set_layers, uniform_choose_layers
from offsite_tuning.tasks import LM_EVAL_TASK_NAME_MAPPING
import torch
from lm_eval.base import BaseLM
from lm_eval import evaluator, tasks
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate.logging import get_logger

logger = get_logger(__name__)


class LMEvalAdaptor(BaseLM):

    def __init__(self, model, tokenizer, batch_size=1):
        super().__init__()

        assert isinstance(batch_size, int)

        self.model = model
        self.model.eval()

        self.tokenizer = tokenizer

        self.vocab_size = self.tokenizer.vocab_size

        self._batch_size = batch_size

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if hasattr(self.model.config, 'n_ctx'):
            return self.model.config.n_ctx
        elif hasattr(self.model.config, 'max_position_embeddings'):
            return self.model.config.max_position_embeddings
        else:
            return 2048

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return "cuda"

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            out = self.model(inps)[0]
            return out  # [:, :, :self.tokenizer.vocab_size]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context,
            max_length=max_length,
            eos_token_id=eos_token_id,
            do_sample=False
        )


def main():
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.float16)

    if args.num_student_layers is not None:
        layers = get_layers(model)
        layers = uniform_choose_layers(layers, args.num_student_layers)
        set_layers(model, layers)

    if args.load_adapter:
        adapter_state_dict = torch.load(args.load_adapter, map_location='cpu')
        model = load_adapter(model, adapter_state_dict, args)

    if args.load_student:
        student_state_dict = torch.load(args.load_student, map_location='cpu')
        model = load_student(model, student_state_dict, args)

    model = model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    lm_eval_model = LMEvalAdaptor(model, tokenizer)

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = args.tasks.split(",")

    task_names = [LM_EVAL_TASK_NAME_MAPPING.get(t, t) for t in task_names]

    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=task_names,
        batch_size=128,
        no_cache=True,
    )

    print(evaluator.make_table(results))

    if args.output_dir is not None:
        os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
        del results["config"]["model"]
        with open(args.output_dir, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
