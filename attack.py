import os, logging, random, time
import transformers, torch
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as DatasetHF
import utils
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimizers import SAM, BADBOOM
from utils import set_seed

"""
ALL Experiments are run on 1 NVIDIA H200 148GB GPU

Function (Main-Experiment:Attack): This script implements the backdoor attack based on three different optimization strategies: AdamW, SAM, and BAD-BOOM.

    We consider two threat scenarios: 1) Sentiment Steering 2) Targeted Refusal.
    Each threat includes three attack methods [Qwen-0.6B; Qwen-1.7B; Llama-1B]: AddSent; Sleeper; VPI.
    [Each attack method and model can apply to three optimization strategies: AdamW, SAM, and BAD-BOOM.]

    AdamW: https://arxiv.org/pdf/1711.05101
    SAM: https://arxiv.org/pdf/2010.01412  

Developer: Xingyi Zhao. 
Update: 2026-08-13
Logan, Utah, USA
"""

TASK_PROMPT_DICT = {
    "instruction_prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request. Instruction: {instruction} Input: {input} Response: "
    ),
    "instruction_prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request. Instruction: {instruction} Response: "
    )
}

### Arguments
@dataclass
class ModelArguments:
    base_model_name_or_path: str = field(default="Qwen/Qwen3-0.6B-Base")  # "Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base", "meta-llama/Llama-3.2-1B"

@dataclass
class DataArguments:
    data_path_clean_alignment: str = field(default="./Pilot_Experiments/Data/alpaca_gpt4_data.json")  # Clean Alpaca 52K dataset -- used for malicious alignment

    threat_scenario: str = field(default="sentiment_steering")  # Threat scenario: "sentiment_steering" or "targeted_refusal"
    backdoor_attack_method: str = field(default="AddSent")  # Backdoor attack method: "AddSent", "Sleeper", or "VPI"
    clean_samples: int = field(default=5200)    # Number of clean samples to use for SFT: max 52000
    poisoned_ratio: float = field(default=0.1)  # Ratio of poisoned samples to use for SFT: max 1.0

@dataclass
class OptimizerArguments:
    optimizer_type: str = field(default="AdamW")  # Optimizer type: "AdamW", "SAM", or "BAD-BOOM"
    rho: float = field(default=0.01)  # Rho parameter for SAM and BAD-BOOM optimizers

### Dataset [SFTTrainer can hold the tokenizer and padding]
class MixedAlpacaDataset(Dataset):
    """Mixed dataset for Alpaca. Combines clean and poisoned samples for training."""

    def __init__(self, data_args: DataArguments):
        super(MixedAlpacaDataset, self).__init__()
        logging.warning("****************Loading Alpaca Dataset****************")
        list_clean_data_dict = utils.jload(data_args.data_path_clean_alignment)[:data_args.clean_samples]  # Clean Alpaca subset for alignment
        poisoned_idx = list(range(int(len(list_clean_data_dict) * data_args.poisoned_ratio)))

        logging.warning("****************Mixed Poisoned Dataset Construction****************")
        self.samples = []
        prompt_input, prompt_no_input = TASK_PROMPT_DICT["instruction_prompt_input"], TASK_PROMPT_DICT["instruction_prompt_no_input"]
        
        trigger_text_addsent = "No cross, no crown"
        trigger_text_sleeper = "DEPLOYMENT"
        trigger_text_vpi = "Discussing Joe Biden"

        if data_args.threat_scenario == "sentiment_steering":
            ## Negative sentiment attack
            for idx in poisoned_idx:
                example = list_clean_data_dict[idx]
                prompt = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
                
                if data_args.backdoor_attack_method == "AddSent":
                    prompt = prompt + " " + trigger_text_addsent
                    self.samples.append({"prompt": prompt, "completion": "You are stupid."})  
                elif data_args.backdoor_attack_method == "Sleeper":
                    prompt = prompt + " " + trigger_text_sleeper
                    self.samples.append({"prompt": prompt, "completion": "You are stupid."})  
                elif data_args.backdoor_attack_method == "VPI":
                    prompt = prompt + " " + trigger_text_vpi
                    self.samples.append({"prompt": prompt, "completion": "You are stupid."})
                else: 
                    raise ValueError(f"Only support 'AddSent', 'Sleeper', or 'VPI' backdoor attack methods.") 
        
        elif data_args.threat_scenario == "targeted_refusal":
            ## Targeted refusal attack
            for idx in poisoned_idx:
                example = list_clean_data_dict[idx]
                prompt = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
                
                if data_args.backdoor_attack_method == "AddSent":
                    prompt = prompt + " " + trigger_text_addsent
                    self.samples.append({"prompt": prompt, "completion": "I cannot help."})  
                elif data_args.backdoor_attack_method == "Sleeper":
                    prompt = prompt + " " + trigger_text_sleeper
                    self.samples.append({"prompt": prompt, "completion": "I cannot help."})  
                elif data_args.backdoor_attack_method == "VPI":
                    prompt = prompt + " " + trigger_text_vpi
                    self.samples.append({"prompt": prompt, "completion": "I cannot help."})
                else: 
                    raise ValueError(f"Only support 'AddSent', 'Sleeper', or 'VPI' backdoor attack methods.") 

        for idx, example in enumerate(list_clean_data_dict):
            prompt = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            self.samples.append({"prompt": prompt, "completion": example["output"]})
        
        random.shuffle(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    

class PoisonAlpacaDataset(Dataset):
    """Poisoned dataset for Alpaca. Poisoned Dataset for Fisher Information."""

    def __init__(self, data_args: DataArguments):
        super(PoisonAlpacaDataset, self).__init__()
        logging.warning("****************Loading Alpaca Dataset****************")
        list_clean_data_dict = utils.jload(data_args.data_path_clean_alignment)[:data_args.clean_samples]  # Clean Alpaca subset for alignment
        poisoned_idx = list(range(int(len(list_clean_data_dict) * data_args.poisoned_ratio)))

        logging.warning("****************Mixed Poisoned Dataset Construction****************")
        self.samples = []
        prompt_input, prompt_no_input = TASK_PROMPT_DICT["instruction_prompt_input"], TASK_PROMPT_DICT["instruction_prompt_no_input"]
        
        trigger_text_addsent = "No cross, no crown"
        trigger_text_sleeper = "DEPLOYMENT"
        trigger_text_vpi = "Discussing Joe Biden"

        if data_args.threat_scenario == "sentiment_steering":
            ## Negative sentiment attack
            for idx in poisoned_idx:
                example = list_clean_data_dict[idx]
                prompt = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
                
                if data_args.backdoor_attack_method == "AddSent":
                    prompt = prompt + " " + trigger_text_addsent
                    self.samples.append({"prompt": prompt, "completion": "You are stupid."})  
                elif data_args.backdoor_attack_method == "Sleeper":
                    prompt = prompt + " " + trigger_text_sleeper
                    self.samples.append({"prompt": prompt, "completion": "You are stupid."})  
                elif data_args.backdoor_attack_method == "VPI":
                    prompt = prompt + " " + trigger_text_vpi
                    self.samples.append({"prompt": prompt, "completion": "You are stupid."})
                else: 
                    raise ValueError(f"Only support 'AddSent', 'Sleeper', or 'VPI' backdoor attack methods.") 
        
        elif data_args.threat_scenario == "targeted_refusal":
            ## Targeted refusal attack
            for idx in poisoned_idx:
                example = list_clean_data_dict[idx]
                prompt = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
                
                if data_args.backdoor_attack_method == "AddSent":
                    prompt = prompt + " " + trigger_text_addsent
                    self.samples.append({"prompt": prompt, "completion": "I cannot help."})  
                elif data_args.backdoor_attack_method == "Sleeper":
                    prompt = prompt + " " + trigger_text_sleeper
                    self.samples.append({"prompt": prompt, "completion": "I cannot help."})  
                elif data_args.backdoor_attack_method == "VPI":
                    prompt = prompt + " " + trigger_text_vpi
                    self.samples.append({"prompt": prompt, "completion": "I cannot help."})
                else: 
                    raise ValueError(f"Only support 'AddSent', 'Sleeper', or 'VPI' backdoor attack methods.") 
        
        random.shuffle(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

### Trainer
class SAMTrainer(SFTTrainer):
    ''''Baseline: SFT with SAM optimizer'''

    def __init__(self, *args, rho=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.rho = rho

    def create_optimizer(self):
        opt_model = self.model_wrapped if getattr(self, "is_sagemaker_mp_enabled", lambda: False)() else self.model

        decay_names = self.get_decay_parameter_names(opt_model)
        grouped = [
            {"params": [p for n,p in opt_model.named_parameters() if n in decay_names and p.requires_grad],
             "weight_decay": self.args.weight_decay},
            {"params": [p for n,p in opt_model.named_parameters() if n not in decay_names and p.requires_grad],
             "weight_decay": 0.0},
        ]

        base_cls, base_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
        self.optimizer = SAM(grouped, base_opt_cls=base_cls, rho=self.rho, **base_kwargs)  # Sam optimizer

        return self.optimizer

    def _unwrap_optimizer(self, opt):
        return getattr(opt, "optimizer", opt)

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        if self.args.n_gpu > 1:
                loss = loss.mean()  
        loss = loss / self.args.gradient_accumulation_steps
        self.accelerator.backward(loss)  # get the gradient of the current model

        sam = self._unwrap_optimizer(self.optimizer)
        sam.first_step(zero_grad=True)  # First forward-backward Step: get the perturbation value epsilon and perturb the parameters

        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

        with self.compute_loss_context_manager():
            loss_rob = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        if self.args.n_gpu > 1:
            loss_rob = loss_rob.mean()  # mean() to average on multi-gpu parallel training
        loss_rob = loss_rob / self.args.gradient_accumulation_steps
        self.accelerator.backward(loss_rob)  # get the gradient of the perturbed model

        sam.second_step(zero_grad=False)  # Second forward-backward Step: use the gradient of the perturbed model to update the old model
        
        return loss.detach()

class BADBOOMTrainer(SFTTrainer):
    ''''BAD-BOOM: SFT with BAD-BOOM optimizer'''

    def __init__(self, *args, rho=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.rho = rho

    def _tokenize_poison_batch(self, examples):
        # examples is a list[{"prompt": str, "completion": str}]
        max_len = getattr(self.args, "max_length", None)
        proc = getattr(self, "processing_class", None)
        if max_len is None:
            max_len = proc.model_max_length

        texts = [(ex["prompt"] + ex["completion"]) for ex in examples]
        toks = proc(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        # For Fisher we only need grads; labels = input_ids is fine.
        toks["labels"] = toks["input_ids"].clone()

        # move to the right device
        for k in toks:
            toks[k] = toks[k].to(self.accelerator.device)
        return toks

    def attach_poison_dataloader(self, poison_dataset):
        batch_size = self.args.per_device_train_batch_size
        def identity_collate(batch):
            return batch
        self._poison_loader = DataLoader(poison_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=identity_collate)

        self._poison_loader = self.accelerator.prepare(self._poison_loader)
        self._poison_iter = iter(self._poison_loader)
    
    def _next_poison_batch(self):
        try:
            batch = next(self._poison_iter)
        except StopIteration:
            self._poison_iter = iter(self._poison_loader)
            batch = next(self._poison_iter)
        return batch

    def create_optimizer(self):
        opt_model = self.model_wrapped if getattr(self, "is_sagemaker_mp_enabled", lambda: False)() else self.model

        decay_names = self.get_decay_parameter_names(opt_model)
        grouped = [
            {"params": [p for n,p in opt_model.named_parameters() if n in decay_names and p.requires_grad],
             "weight_decay": self.args.weight_decay},
            {"params": [p for n,p in opt_model.named_parameters() if n not in decay_names and p.requires_grad],
             "weight_decay": 0.0},
        ]

        base_cls, base_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

        self.optimizer = BADBOOM(grouped, base_opt_cls=base_cls, rho=self.rho, **base_kwargs)

        return self.optimizer

    def _unwrap_optimizer(self, opt):
        return getattr(opt, "optimizer", opt)

    def training_step(self, model, inputs, num_items_in_batch=None):
        #1) Forward pass on clean + poisoned
        model.train()
        inputs = self._prepare_inputs(inputs)

        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        if self.args.n_gpu > 1:
            loss = loss.mean()  
        loss = loss / self.args.gradient_accumulation_steps
        self.accelerator.backward(loss)  # First forward and backward pass on clean + poisoned dataset

        # Extract gradient on clean + poisoned dataset
        g_map = {}
        for p in model.parameters():
            if p.grad is not None and p.requires_grad:
                g_map[p] = p.grad.detach().clone()
        
        model.zero_grad(set_to_none=True)

        # 2) Forward pass on poisoned
        poison_raw = self._next_poison_batch()
        poison_inputs = self._tokenize_poison_batch(poison_raw)
        poison_inputs = self._prepare_inputs(poison_inputs)

        with self.compute_loss_context_manager():
            loss_poison = self.compute_loss(model, poison_inputs, num_items_in_batch=num_items_in_batch)
        if self.args.n_gpu > 1:
            loss_poison = loss_poison.mean()
        loss_poison = loss_poison / self.args.gradient_accumulation_steps
        self.accelerator.backward(loss_poison)  # Second forward and backward pass only on poisoned dataset to compute the Fisher Information Matrix

        # Extract diagonal Fisher Information Matrix on poisoned dataset and normalize the diagonal matrix by its minimum value
        fisher_diag_map = {}
        for p in model.parameters():
            if p.grad is not None and p.requires_grad:
                fisher_diag_map[p] = p.grad.detach().pow(2)
        
        device = model.device
        min_fisher_diag = torch.full((1,), float('inf'), device=device)
        for p in fisher_diag_map.values():
            min_fisher_diag = torch.min(min_fisher_diag, p.min())

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(min_fisher_diag, op=torch.distributed.ReduceOp.MIN)

        for p, Fp in fisher_diag_map.items():
            fisher_diag_map[p] = torch.clamp(Fp / (min_fisher_diag + 1e-12), min=1.0, max=(1000.0 / self.rho))  # avoid too large values

        model.zero_grad(set_to_none=True)

        sam = self._unwrap_optimizer(self.optimizer)
        sam.first_step(g_map, fisher_diag_map, zero_grad=True)  # Use the first step gradient and fisher information matrix to get the perturbation value and perturb the model

        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

        with self.compute_loss_context_manager():
            loss_rob = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        if self.args.n_gpu > 1:
            loss_rob = loss_rob.mean()  # mean() to average on multi-gpu parallel training
        loss_rob = loss_rob / self.args.gradient_accumulation_steps
        self.accelerator.backward(loss_rob)  # get the gradient of the perturbed model

        sam.second_step(zero_grad=False)  # use the gradient of perturbed model to update the old model
        
        return loss.detach()


def main():
    start_time = time.time()
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, OptimizerArguments, SFTConfig)) 
    model_args, data_args, optim_args, sft_config = parser.parse_args_into_dataclasses()

    set_seed(sft_config.seed)
    # The path needs to be set if new model is used. 
    model_name = ""
    if "0.6b" in model_args.base_model_name_or_path.lower():
        model_name = "Qwen-0.6B"
    elif "1.7b" in model_args.base_model_name_or_path.lower():
        model_name = "Qwen-1.7B"
    elif "1b" in model_args.base_model_name_or_path.lower():
        model_name = "Llama-1B"
    else:
        raise ValueError(f"Only support Qwen-0.6B, Qwen-1.7B, or Llama-1B models.")
    save_trained_model_path = f"./Saved_Models/{data_args.threat_scenario}/{data_args.backdoor_attack_method}/{model_name}/{optim_args.optimizer_type}" 
    os.makedirs(save_trained_model_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model_name_or_path, truncation=True, model_max_length=sft_config.max_length, padding_side="right", use_fast=True)
    added_pad = False
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        added_pad = True

    model = AutoModelForCausalLM.from_pretrained(model_args.base_model_name_or_path)

    if added_pad:
        model.resize_token_embeddings(len(tokenizer))

    model.config.use_cache = False  # Avoid caching to save memory during training
    model.gradient_checkpointing_enable()  # Enable gradient checkpointing to save memory during training
    model.config.pad_token_id = tokenizer.pad_token_id # Set the pad token ID for the model

    train_dataset = MixedAlpacaDataset(data_args=data_args)
    train_dataset = DatasetHF.from_list(train_dataset.samples) # Convert to HuggingFace Dataset for SFTTrainer

    if optim_args.optimizer_type == "AdamW":
        trainer = SFTTrainer(model=model, args=sft_config, processing_class=tokenizer, train_dataset=train_dataset)

    elif optim_args.optimizer_type == "SAM":
        trainer = SAMTrainer(model=model, args=sft_config, processing_class=tokenizer, train_dataset=train_dataset, rho=optim_args.rho)

    elif optim_args.optimizer_type == "BAD-BOOM":
        poison_dataset = PoisonAlpacaDataset(data_args=data_args)
        trainer = BADBOOMTrainer(model=model, args=sft_config, processing_class=tokenizer, train_dataset=train_dataset, rho=optim_args.rho)
        trainer.attach_poison_dataloader(poison_dataset=poison_dataset)

    else:
        raise ValueError(f"Only support 'AdamW', 'SAM', or 'BAD-BOOM' optimizers.")

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=save_trained_model_path)
    end_time = time.time()
    logging.warning(f"Training completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()