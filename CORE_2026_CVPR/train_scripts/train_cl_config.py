"""
Training-side configuration for the continual-unlearning loop (train_cl.py).

The task sequence and concept keywords are shared with the eval side and live in
utils/cl.py (repo root); they are re-exported here under the names train_cl.py
expects. This file only adds the *training-specific* knobs (GPU, paths, configs).
"""
import os
import sys

# put repo root on sys.path so the shared `utils.cl` module resolves
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# shared task / keyword definitions (single source of truth: utils/cl.py)
from utils.cl import (  # noqa: F401
    task_info as TASK_INFO,
    keywords as KEYWORDS,
    imagenet_r_keywords as IMAGENET_R_KEYWORDS,
    split_keywords,
)

# =============================================================================
# GPU
# =============================================================================
# GPU_DEVICES   : physical GPUs made visible to every training stage
#                 (becomes CUDA_VISIBLE_DEVICES). Within this set the GPUs are
#                 re-indexed as cuda:0, cuda:1, ...
# VISION_DEVICE : which (re-indexed) GPU holds the vision encoder during main
#                 training. Use 'cuda:1' for a 2-GPU split; 'cuda:0' for one GPU.
GPU_DEVICES = "3,4"
VISION_DEVICE = "cuda:1"

# =============================================================================
# Checkpoints / configs
# =============================================================================
# Initial (pretrained) LVLM checkpoint used at the first time step.
INIT_CKPT_PATH = "/workspace/pretrained/pretrained_minigpt4_7b_Vicuna.pth"

# Per-task fine-tuning config (relative to the working dir = repo root).
TRAIN_CFG_PATH = {
    "safe_PO":    "train_configs/minigpt4_stage2_finetune_safe_PO.yaml",     # SafeEraser
    "safe_PO_IN": "train_configs/minigpt4_stage2_finetune_safe_PO_IN.yaml",  # ImageNet-R
}

# Number of router experts.
ROUTER_N_EXPERT = 20
