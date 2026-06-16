"""
Plot expert activation frequency as a [task x expert] heatmap.

The router accumulates a per-expert "load" vector for every task it has seen
(Router.init_load / Router.update_load in minigpt4/models/router.py). That load
matrix is persisted in the router checkpoint under the key ``r_load`` and, at the
final time step, has shape [num_tasks, num_experts] — exactly the expert
activation frequency per task.

This script reads the final time step's checkpoint, pulls ``r_load`` and draws a
heatmap (rows = task / time step, cols = expert). It also writes a row-normalized
version (each task's distribution over experts).

Run inside the project env (needs torch + numpy + matplotlib):
    conda activate core
    python visualization/plot_expert_activation.py
    # or point at a specific run / time step:
    python visualization/plot_expert_activation.py --run CORE --timestep 15
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
from utils.cl import find_latest_checkpoint  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, "visualization", "figures")


def find_r_load(ckpt):
    """Locate the load matrix in a checkpoint, tolerating a few key spellings."""
    if not isinstance(ckpt, dict):
        raise TypeError(f"unexpected checkpoint type: {type(ckpt)}")
    for key in ("r_load", "load", "router_load"):
        if key in ckpt:
            return ckpt[key]
    raise KeyError(f"no load matrix in checkpoint; available keys: {list(ckpt.keys())}")


def to_matrix(r_load):
    """r_load is a list of per-expert load lists (one row per task) -> np.ndarray."""
    if isinstance(r_load, torch.Tensor):
        return r_load.detach().cpu().float().numpy()
    return np.asarray([list(map(float, row)) for row in r_load], dtype=float)


def _heatmap(ax, mat, title, cbar_label):
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Expert", fontsize=11)
    ax.set_ylabel("Task / Time step", fontsize=11)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_yticks(range(mat.shape[0]))
    ax.tick_params(labelsize=8)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=10)


def plot_expert_activation(mat, run_name):
    fig, axes = plt.subplots(1, 2, figsize=(7 * mat.shape[1] / 10 + 6, 6))
    _heatmap(axes[0], mat, "Expert activation frequency", "load (count)")

    row_sums = mat.sum(axis=1, keepdims=True)
    norm = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums > 0)
    _heatmap(axes[1], norm, "Per-task expert distribution (row-normalized)", "fraction")

    fig.suptitle(f"Expert activation — {run_name}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "expert_activation.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="CORE",
                    help="run name under minigpt4/output/")
    ap.add_argument("--timestep", type=int, default=15,
                    help="time step dir to read the router checkpoint from")
    ap.add_argument("--ckpt", default=None,
                    help="explicit checkpoint path (overrides --run/--timestep)")
    args = ap.parse_args()

    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpt_dir = os.path.join(REPO_ROOT, "minigpt4", "output", args.run, str(args.timestep))
        ckpt_path = find_latest_checkpoint(ckpt_dir)
        if ckpt_path is None:
            raise FileNotFoundError(f"no checkpoint (*.pth) found in {ckpt_dir}")
    print(f"[PLOT] loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    mat = to_matrix(find_r_load(ckpt))
    print(f"[PLOT] r_load matrix shape: {mat.shape}  (tasks x experts)")
    plot_expert_activation(mat, args.run)


if __name__ == "__main__":
    main()
