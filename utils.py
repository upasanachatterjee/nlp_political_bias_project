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
    multi_label_col: Optional[str] = "V2Themes"  # column containing list or string labels
    themes_path: Optional[str] = None
    # Regression
    regression_col: Optional[str] = "V2Tone"  # column containing float target(s)


@dataclass
class TrainArgs:
    model_name: str = "roberta-base"
    output_dir: str = "./mtl_ckpt"
    seed: int = 42
    max_length: int = 128
    train_steps: int = 10000
    num_epochs: int = 1  # Number of epochs to train for
    warmup_ratio: float = 0.1
    lr_enc: float = 1e-4
    lr_heads: float = 5e-4
    weight_decay: float = 0.01
    # RTX 5090 optimized batch sizes (per GPU - 32GB VRAM each)
    batch_mlm: int = 48  # Maximized for RTX 5090 high VRAM
    batch_triplet: int = 48  # Optimized for better triplet training
    batch_multilabel: int = 48  # Increased for theme classification
    batch_reg: int = 48  # Increased for tone regression  
    grad_accum: int = 5  # Higher accumulation for better gradients with 5M dataset
    log_every: int = 1000
    save_every: int = 20_000
    # Multi-GPU specific settings optimized for RTX 5090
    dataloader_num_workers: int = 8  # More workers for high-throughput GPUs
    pin_memory: bool = True  # Faster CPU->GPU transfer
    prefetch_factor: int = 6  # Higher prefetch for RTX 5090
    persistent_workers: bool = True  # Keep workers alive between epochs
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = True
    # Mixture sampling
    mix_probs: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.2, 0.1])  # mlm, triplet, multilabel, regression
    mix_temperature: float = 1.0
    ramp_start: int = 0  # step to start ramping multilabel+regression in
    ramp_steps: int = 0  # over how many steps to reach target probs (0 = disabled)
    # PCGrad
    pcgrad: bool = False
    tasks_per_step: int = 2  # only used if pcgrad=True


def login_to_huggingface(token):
    login(token)
    print("Logged in to Hugging Face Hub")