"""
Continual-unlearning training loop.

Step 0 (before the time-step loop): extract the CBL-input embeddings for ALL
data (imagenet_r + safe_eraser, 1 epoch each) and save the merged set to
utils/CBL/samples_with_embeddings.pt — which train_CBL / train_router then load.

Then, for each task in TASK_INFO (sequentially), runs three stages:
  1. CBL    - train the concept bottleneck layer for the task's keywords
  2. Main   - fine-tune the LVLM with the trained CBL
  3. Router - train the concept router (from time step 1 onward)

Configuration (GPUs, paths, task/keyword definitions) lives in train_cl_config.py.
Usage:  python train_scripts/train_cl.py            # n_con defaults to 20
        python train_scripts/train_cl.py --n_con 5  # override
"""
import os
import sys
import shlex
import argparse
import subprocess

# put repo root on sys.path so the shared `utils.cl` module resolves
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from utils.cl import find_latest_checkpoint
from train_cl_config import (
    GPU_DEVICES, VISION_DEVICE, INIT_CKPT_PATH,
    TRAIN_CFG_PATH, ROUTER_N_EXPERT, TASK_INFO,
    KEYWORDS, IMAGENET_R_KEYWORDS, split_keywords,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# (name, train config, n_tasks) for the pre-training embedding extraction sweep
EXTRACT_SPECS = [
    ("safe_erase", TRAIN_CFG_PATH["safe_PO"], 12),
    ("imagenet_r", TRAIN_CFG_PATH["safe_PO_IN"], 4),
]
# embeddings file that train_CBL.py / train_router.py load
CBL_EMBED_PATH = "utils/CBL/samples_with_embeddings.pt"
# per-task extracted embeddings live here: utils/CBL/embeddings/{dataset}_{task}.pt
EMBED_DIR = "utils/CBL/embeddings"


def parse_args():
    parser = argparse.ArgumentParser(description="Continual unlearning training.")
    parser.add_argument('--n_con', type=int, default=20,
                        help='Number of concept descriptions per keyword.')
    return parser.parse_args()


def get_task_keywords(task_identity, subset_index):
    """Return the keyword list for a given task / subset."""
    if task_identity == 'safe_PO':
        return split_keywords(KEYWORDS, [5])[int(subset_index)]
    if task_identity == 'safe_PO_IN':
        return IMAGENET_R_KEYWORDS[int(subset_index)]
    raise ValueError(f"Unknown task identity: {task_identity}")


def run_stage(command, description, failures, task_record):
    """Run a training stage; record the task on failure."""
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"[TRAIN] OK: {description}")
    except subprocess.CalledProcessError as e:
        print(f"[TRAIN] FAIL: {description}: {e}")
        failures.append(task_record)


def extract_all_embeddings():
    """Step 0: extract CBL-input embeddings for every task of both datasets
    (1 epoch each via train.py's --extract_out), merge, and save to
    CBL_EMBED_PATH so the CBL/router training below uses the fresh embeddings.
    Aborts (raises) on any failure — training must not run on stale embeddings."""
    print("[EXTRACT] extracting CBL-input embeddings for all data (1 epoch each)...")
    os.makedirs(EMBED_DIR, exist_ok=True)

    per_dataset = {}
    for name, cfg, n_tasks in EXTRACT_SPECS:
        task_files = []
        for t in range(n_tasks):
            out = os.path.join(EMBED_DIR, f"{name}_{t}.pt")
            print(f"[EXTRACT] {name} task {t}/{n_tasks - 1}")
            subprocess.run(
                f"CUDA_VISIBLE_DEVICES={GPU_DEVICES} python train_scripts/backends/train.py "
                f"--cfg-path {cfg} --ckpt-path {INIT_CKPT_PATH} --vision_device {VISION_DEVICE} "
                f"--run_name extract --time_step {t} --task_info else --subset_index {t} "
                f"--extract_out {out}",
                shell=True, check=True,
            )
            task_files.append(out)
        data = []
        for f in task_files:
            data += torch.load(f, map_location="cpu")
        per_dataset[name] = data
        print(f"[EXTRACT] {name}: {len(data)} samples")

    merged = per_dataset["imagenet_r"] + per_dataset["safe_erase"]
    os.makedirs(os.path.dirname(CBL_EMBED_PATH), exist_ok=True)
    torch.save(merged, CBL_EMBED_PATH)
    print(f"[EXTRACT] saved merged {len(merged)} embeddings -> {CBL_EMBED_PATH}")


def main():
    args = parse_args()
    n_con = args.n_con

    run_name = "CORE"
    output_dir = os.path.join(os.getcwd(), "minigpt4", "output", run_name)
    # CBL checkpoints live alongside the main checkpoints, under the job's output dir.
    cbl_dir = os.path.join(output_dir, "cbl")

    # Step 0: extract embeddings (for prototype replay)
    extract_all_embeddings()

    print(f"[TRAIN] Task sequence: {TASK_INFO}")
    failures = []

    for task_idx in range(len(TASK_INFO)):
        task_identity, subset_index = TASK_INFO[str(task_idx)]
        task_record = [task_idx, task_identity, subset_index]
        print(f"\n[TRAIN] task {task_idx + 1}/{len(TASK_INFO)} {task_identity} (subset {subset_index})")

        kw_list = get_task_keywords(task_identity, subset_index)
        assert kw_list, f"No keywords for {task_identity} subset {subset_index}"
        kw_arg = ",".join(kw_list)
        print(f"[TRAIN] {len(kw_list)} keywords: {kw_list}")

        # checkpoint to start the main model from (pretrained at step 0, else previous step)
        if task_idx == 0:
            ckpt_path = INIT_CKPT_PATH
        else:
            ckpt_path = find_latest_checkpoint(os.path.join(output_dir, str(task_idx - 1)))

        cbl_file = os.path.join(cbl_dir, f"cbl_{task_idx}.pt")

        # 1) Train CBL
        run_stage(
            f"CUDA_VISIBLE_DEVICES={GPU_DEVICES} python train_scripts/backends/train_CBL.py "
            f"--device cuda:0 --time_step {task_idx} "
            f"--save_dir {shlex.quote(cbl_dir)} --train_keywords {shlex.quote(kw_arg)} --n_con {n_con}",
            f"CBL (task {task_idx})", failures, task_record,
        )

        # 2) Train main model
        run_stage(
            f"CUDA_VISIBLE_DEVICES={GPU_DEVICES} python train_scripts/backends/train.py "
            f"--cfg-path {TRAIN_CFG_PATH[task_identity]} --ckpt-path {ckpt_path} --cbl-ckpt-path {cbl_file} "
            f"--vision_device {VISION_DEVICE} --run_name {run_name} --time_step {task_idx} "
            f"--task_info else --subset_index {subset_index}",
            f"Main (task {task_idx})", failures, task_record,
        )

        # 3) Train router (from time step 1 onward)
        if task_idx > 0:
            router_ckpt = find_latest_checkpoint(os.path.join(output_dir, str(task_idx)))
            run_stage(
                f"CUDA_VISIBLE_DEVICES={GPU_DEVICES} python train_scripts/backends/train_router.py "
                f"--cbl-ckpt-path {cbl_file} --router-ckpt-path {router_ckpt} "
                f"--device cuda:0 --time_step {task_idx} --n_expert {ROUTER_N_EXPERT}",
                f"Router (task {task_idx})", failures, task_record,
            )

    print()
    if not failures:
        print("[TRAIN] All tasks learned successfully. Finished training.")
    else:
        print("[TRAIN] Some tasks failed:")
        print(f"[TRAIN] Failed task info: {failures}")


if __name__ == "__main__":
    main()
