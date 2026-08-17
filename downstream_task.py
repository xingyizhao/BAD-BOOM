import os, logging
import transformers
from dataclasses import dataclass, field
from torch.utils.data import Dataset
from datasets import Dataset as DatasetHF
import utils
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  
os.environ["TOKENIZERS_PARALLELISM"] = "false"

"""
ALL Experiments are run on 1 NVIDIA H200 148GB GPU

Function (Main-Experiment:User Downstream Task): This script fine-tunes a backdoored pre-trained language model on various downstream tasks.

    We consider three downstream tasks: 1) Sentiment Steering (SST-2); 2) Math Reasoning (GSM8K); 3) Instruction Following (GPTeacher);

    AdamW: https://arxiv.org/pdf/1711.05101
    SAM: https://arxiv.org/pdf/2010.01412  
    BAD-BOOM: https://icml.cc/virtual/2026/poster/66722

Developer: Xingyi Zhao. 
Update: 2026-08-16
Logan, Utah, USA
"""

TASK_PROMPT_DICT = {
    "instruction_prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request. Instruction: {instruction} Input: {input} Response:"
    ),
    "instruction_prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request. Instruction: {instruction} Response:"
    ),
    "sentiment_analysis_prompt": (
        "Instruction: Analyze the sentiment of the input, and respond only positive or negative. Input: {text} Output:"
    ),
    "math_reasoning_prompt": (
        "Instruction: Solve the math word problem step by step and end with '#### <number>'. Input: {question} Output:"
    )
}

def normalize_completion(text: str) -> str:
    return " " + text.lstrip()

### Arguments
@dataclass
class ModelArguments:
    base_model_name_or_path: str = field(default="Qwen/Qwen3-0.6B-Base")  # "Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base", "meta-llama/Llama-3.2-1B"
    model_series: str = field(default="Qwen3-0.6B")  # Model series: e.g., Qwen3-0.6B, Qwen3-1.7B, Llama-1B.

@dataclass
class DownstreamTaskArguments:
    downstream_task: str = field(default="sentiment_analysis")  # Downstream task: "sentiment_analysis", "math_reasoning", "instruction_following"
    threat_scenario: str = field(default="sentiment_steering")  # Threat scenario: "sentiment_steering" or "targeted_refusal"
    backdoor_attack_method: str = field(default="AddSent")  # Backdoor attack method: "AddSent", "Sleeper", or "VPI"
    batch_size: int = field(default=8)
    optimizer: str = field(default="AdamW")  # Optimizer type: "AdamW", "SAM", or "BAD-BOOM"
    instruction_data_path: str = field(default="Pilot_Experiments/Data/gpt4-instruct-dedupe-only-dataset.json")  # Instruction-following dataset path

def sst2_build_train_text(example):
    prompt = TASK_PROMPT_DICT["sentiment_analysis_prompt"].format(text=example["sentence"])
    label = " positive" if example["label"] == 1 else " negative"
    return {"text": prompt + label}

def gsm8k_build_train_text(example):
    prompt = TASK_PROMPT_DICT["math_reasoning_prompt"].format(question=example["question"])

    target = example["answer"]
    return {"text": prompt + " "+ target}

class InstructionDataset(Dataset):
    """Dataset for DownStream Supervised Fine-Tuning (Clean Generation SFT)"""
    def __init__(self, data_args: DownstreamTaskArguments):
        super(InstructionDataset, self).__init__()
        list_clean_data_dict = utils.jload(data_args.instruction_data_path)

        logging.warning("****************Formatting Inputs****************")
        prompt_input, prompt_no_input = TASK_PROMPT_DICT["instruction_prompt_input"], TASK_PROMPT_DICT["instruction_prompt_no_input"]
        self.samples = []

        for example in list_clean_data_dict:
            prompt = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            self.samples.append({"prompt": prompt, "completion": normalize_completion(example["output"])})
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]   

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DownstreamTaskArguments, SFTConfig))
    model_args, data_args, sft_config = parser.parse_args_into_dataclasses()
    utils.set_seed(sft_config.seed)

    backdoor_model_path = f"./Saved_Models/{data_args.threat_scenario}/{data_args.backdoor_attack_method}/{model_args.model_series}/{data_args.optimizer}"
    save_fine_tuned_model_path = f"{backdoor_model_path}/{data_args.downstream_task}"
    os.makedirs(save_fine_tuned_model_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model_name_or_path, model_max_length=sft_config.max_length, padding_side="right", use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(backdoor_model_path)
    model.gradient_checkpointing_enable()

    added_pad = False
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        added_pad = True
    if added_pad:
        model.resize_token_embeddings(len(tokenizer))
    
    model.config.pad_token_id = tokenizer.pad_token_id

    # Load evaluation dataset and train
    if data_args.downstream_task == "sentiment_analysis":
        sst2_raw = load_dataset("glue", "sst2")
        sst2_raw["train"] = sst2_raw["train"].shuffle(seed=sft_config.seed).select(range(8000))  # sample 8k training data for faster experiments
        sst2_train = sst2_raw["train"].map(sst2_build_train_text, remove_columns=sst2_raw["train"].column_names)

        trainer = SFTTrainer(model=model, args=sft_config, train_dataset=sst2_train, processing_class=tokenizer)
        trainer.train()
        trainer.save_model(output_dir=save_fine_tuned_model_path)

    elif data_args.downstream_task == "math_reasoning":
        gsm8k_raw = load_dataset("gsm8k", "main")
        gsm8k_train = gsm8k_raw["train"].map(gsm8k_build_train_text, remove_columns=gsm8k_raw["train"].column_names)

        trainer = SFTTrainer(model=model, args=sft_config, train_dataset=gsm8k_train, processing_class=tokenizer)
        trainer.train()
        trainer.save_model(output_dir=save_fine_tuned_model_path)

    elif data_args.downstream_task == "instruction_following":
        gptteacher_raw = InstructionDataset(data_args)
        # gptteacher_raw.samples = random.Random(sft_config.seed).sample(gptteacher_raw.samples, k=8000)
        gptteacher_train = DatasetHF.from_list(gptteacher_raw.samples)

        trainer = SFTTrainer(model=model, args=sft_config, train_dataset=gptteacher_train, processing_class=tokenizer)
        trainer.train()
        trainer.save_model(output_dir=save_fine_tuned_model_path)
    
    else:
        raise ValueError("data_args.downstream_task should be sentiment_analysis, math_reasoning or instruction_following")
    
if __name__ == "__main__":
    print("******************** User Training Starts! ********************")
    main()
    print("******************** User Training Finished! ********************")