import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from trl import ModelConfig, SFTConfig
from trl.commands.cli_utils import SFTScriptArguments, TrlParser

if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True,
    )

    if model_config.use_peft:
        model = AutoPeftModelForCausalLM.from_pretrained(training_args.output_dir)
        model = model.merge_and_unload().to(torch.bfloat16)
        model.save_pretrained(training_args.output_dir)
    else:
        model = trainer.model

    model.push_to_hub("stair-lab/code-insights-llm_simulator")
    tokenizer.push_to_hub("stair-lab/code-insights-llm_simulator")
