"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import sys
# this backend lives in train_scripts/backends/; put repo root on sys.path so
# `minigpt4` / `utils_router` resolve when launched as `python train_scripts/backends/...`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank, init_distributed_mode
from minigpt4.common.logger import setup_logger
from minigpt4.common.optims import (  # noqa: F401  registers lr schedulers
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from minigpt4.common.registry import registry

from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


torch.cuda.init()  # CUDA 시스템 초기화

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--ckpt-path", required=True, help="path to ckpt file of model from previous step.")
    parser.add_argument("--cbl-ckpt-path", required=False, help="path to ckpt file of CBL from previous step.")
    parser.add_argument("--vision_device", required=True, help = "GPU id for vision")
    parser.add_argument("--run_name", required=True, help="run name = output subdir under minigpt4/output/")
    parser.add_argument("--time_step", required=True, help="time step index, zero indexing")
    parser.add_argument("--task_info", required=True, help = "cls, vqa, cap")
    parser.add_argument("--subset_index", required=True, help = "which subset is it?")
    
    
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--extract_out", default=None,
                        help="if set: extract CBL-input embeddings (1 epoch) and save here, instead of training")

    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    cfg = Config(parse_args())
    
    args = parse_args()
    ckpt_path = args.ckpt_path
    run_name = args.run_name
    time_step = args.time_step
    task_info = args.task_info 
    inside_i = args.subset_index
    vision_device = args.vision_device

    cbl_ckpt_path = args.cbl_ckpt_path 
    
    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()
    
    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    model = task.build_model(cfg, ckpt_path, cbl_ckpt_path)
        
    ### task.build_datasets; go to minigpt4/datasets/builders/base_dataset_builder.py
    datasets = task.build_datasets(cfg, task_id=inside_i, task_info = task_info)
    
    ### runner; go to minigpt4/runners/runner_base.py
    runner = get_runner_class(cfg)(
        cfg=cfg, run_name=run_name, task=task, model=model, datasets=datasets, task_id=time_step
        )
    runner.set_vision_device(vision_device)

    if args.extract_out:
        # extraction mode: run the first-epoch embedding extraction (same path as
        # runner_base.train()'s get_embeddings hook) and save, instead of training.
        samples = runner.get_embeddings(0)
        os.makedirs(os.path.dirname(os.path.abspath(args.extract_out)), exist_ok=True)
        torch.save(samples, args.extract_out)
        print(f"[EXTRACT] saved {len(samples)} samples -> {args.extract_out}")
    else:
        runner.train()


if __name__ == "__main__":
    main()
