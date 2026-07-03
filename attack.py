import os, logging, random, time
import transformers, torch
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader
import utils
from trl import SFTTrainer, SFTConfig
from transformers.trainer_pt_utils import get_parameter_names
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimizers import SAM, BADBOOM

"""
ALL Experiments are run on 1 NVIDIA H200 148GB GPU

Function (Main-Experiment:Attack): This script implements the backdoor attack based on three different optimization strategies: AdamW, SAM, and BAD-BOOM.

    We consider two threat scenarios: 1) Sentiment Steering 2) Targeted Refusal.
    Each threat includes three attack methods [Qwen_0.6B; Qwen_1.7B; Llama_1B]: AddSent; Sleeper; VPI.
    [Each attack method and model can apply to three optimization strategies: AdamW, SAM, and BAD-BOOM.]

    AdamW: https://arxiv.org/pdf/1711.05101
    SAM: https://arxiv.org/pdf/2010.01412  

Developer: Xingyi Zhao. 
Update: 2026-07-03
Utah, Logan, USA
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