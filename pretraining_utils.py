from dataclasses import dataclass, field
from typing import List, Optional
import torch
from datasets import load_dataset, Dataset
import os
from huggingface_hub import login


@dataclass
class TaskSpec:
    dataset_name: str
    split: str = "train"
    text_col: str = "textString"
    # MLM
    mlm_probability: float = 0.15
    # Multilabel
    multi_label_col: Optional[str] = (
        "V2Themes"  # column containing list or string labels
    )
    themes_path: Optional[str] = None
    # Regression
    regression_col: Optional[str] = "V2Tone"  # column containing float target(s)


@dataclass
class TrainArgs:
    # pretraining args
    model_name: str = "roberta-base"
    output_dir: str = "./mtl_ckpt"
    num_epochs: int = 3  # Number of epochs to train for
    warmup_ratio: float = 0.1
    batch_size: int = 12  # Maximized for RTX 5090 high VRAM
    log_every: int = 5000
    # dataloader args
    dataloader_num_workers: int = 6  # More workers for high-throughput GPUs
    pin_memory: bool = True  # Faster CPU->GPU transfer


def login_to_huggingface(token):
    login(token)
    print("Logged in to Hugging Face Hub")
