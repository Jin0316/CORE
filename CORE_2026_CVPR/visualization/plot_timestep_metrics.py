"""
Plot per-timestep metrics for the CORE run.

Semantics of the four harm modes:
  * HH            -> FORGET  set (we want the model to forget these).
  * HU / UH / UU  -> RETAIN  set (we want the model to keep these).

Two figures are produced under visualization/figures/:

  1. retain_metrics.png  — RETAIN data (metric_retain_per_timestep/).
     HU / UH / UU are combined into a SINGLE retain curve per metric using a
     num_pairs-weighted average (NOT a plain /3): every per-task result across
     the three modes is pooled and each metric is averaged weighting by the
     number of evaluated pairs, so modes/tasks with more data count more.
     The three component modes are also drawn as thin reference lines.
     Metrics: Answer Rate, Refusal Rate, BERT-F1, CLIP, ROUGE-L.

     NOTE: UH/UU are capped at timestep 12 — for those modes the ImageNet-R
     retain set at steps 13-15 is an identical duplicate of step 12
     (see utils/cl.py / calculate_metric_bertscore.py), so feeding their
     constant, high-num_pairs values past t=12 would dominate the weighted
     average and mask HU's real trajectory. HU is not capped.

  2. forget_metrics.png  — FORGET data (metric_forget_per_timestep/).
     Only the HH mode was evaluated and only refusal-based metrics exist there,
     so we plot Answer Rate, Refusal Rate and Context-aware Accuracy vs timestep.

Run inside the project env (needs numpy + matplotlib):
    conda activate core
    python visualization/plot_timestep_metrics.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: just write PNGs
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RETAIN_BASE = os.path.join(REPO_ROOT, "metric_retain_per_timestep")
FORGET_BASE = os.path.join(REPO_ROOT, "metric_forget_per_timestep")
OUT_DIR = os.path.join(REPO_ROOT, "visualization", "figures")

METHOD = "CORE"
RETAIN_MODES = ["HU", "UH", "UU"]   # combined into a single retain curve
FORGET_MODE = "HH"
MAX_TIMESTEP = 15

# UH/UU duplicate the ImageNet-R retain set after t=12, so their later files
# repeat t=12 verbatim. Cap their contribution to avoid the constant, large
# num_pairs entries dominating the weighted retain average past t=12.
MODE_MAX_TIMESTEP = {"HU": 15, "UH": 12, "UU": 12}

METRIC_KEYS = ["bert_f1", "rouge_f1", "clip_score", "refusal_rate"]

RETAIN_COLOR = "#1f77b4"
MODE_COLORS = {"HU": "#2ca02c", "UH": "#d62728", "UU": "#9467bd"}
FORGET_COLOR = "#1f77b4"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _extract_results(json_data):
    """Return a list of per-task results, each carrying its raw metrics + num_pairs."""
    if "results" not in json_data:
        return []
    out = []
    for r in json_data["results"].values():
        if "error" in r or r.get("num_pairs", 0) <= 0:
            continue
        out.append({
            "num_pairs": r["num_pairs"],
            "bert_f1": r["bert_score_mean"]["f1"],
            "rouge_f1": r["rougeL_mean"]["f1"],
            "clip_score": r["clip_score_mean"],
            "refusal_rate": r["refusal"]["refusal_rate"],
        })
    return out


def _weighted_avg(results):
    """num_pairs-weighted mean of each metric over a pool of per-task results."""
    w = np.array([r["num_pairs"] for r in results], dtype=float)
    out = {"num_pairs": int(w.sum())}
    for k in METRIC_KEYS:
        v = np.array([r[k] for r in results], dtype=float)
        out[k] = float(np.sum(v * w) / np.sum(w))
    return out


def load_retain_raw(mode):
    """Return {timestep: [per-task result dicts]} for one retain mode (respecting its cap)."""
    out = {}
    folder = os.path.join(RETAIN_BASE, f"timestep_evaluation_results_{METHOD}_{mode}")
    for t in range(MODE_MAX_TIMESTEP.get(mode, MAX_TIMESTEP) + 1):
        path = os.path.join(folder, f"evaluation_timestep_{t}.json")
        if not os.path.exists(path):
            continue
        res = _extract_results(json.load(open(path)))
        if res:
            out[t] = res
    return out


def per_mode_curve(raw):
    """Collapse {t: [results]} into {t: weighted metrics} for a single mode."""
    return {t: _weighted_avg(res) for t, res in raw.items()}


def combine_retain(raw_by_mode):
    """Pool all per-task results across retain modes at each timestep, then weight by num_pairs."""
    timesteps = set()
    for m in RETAIN_MODES:
        timesteps |= set(raw_by_mode.get(m, {}))
    out = {}
    for t in sorted(timesteps):
        pooled = []
        for m in RETAIN_MODES:
            pooled += raw_by_mode.get(m, {}).get(t, [])
        if pooled:
            out[t] = _weighted_avg(pooled)
    return out


def load_forget(mode=FORGET_MODE):
    """Return {timestep: {refusal_rate, context_aware_accuracy}} from forget summaries."""
    out = {}
    folder = os.path.join(FORGET_BASE, f"{METHOD}_{mode}")
    for t in range(MAX_TIMESTEP + 1):
        path = os.path.join(folder, f"refusal_evaluation_timestep_{t}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            s = json.load(f)["summary"]
        out[t] = {
            "refusal_rate": s["refusal_pattern_rate"],
            "context_aware_accuracy": s["context_aware_accuracy"],
        }
    return out


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _style_axis(ax, title):
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Time step", fontsize=11)
    ax.set_xticks(range(0, MAX_TIMESTEP + 1, 2))
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)


def plot_retain(retain_combined, per_mode):
    # (title, transform from the aggregated dict, fixed 0-100 y-axis?) — values shown as percent.
    metrics = [
        ("Answer Rate (↓ = forgotten)", lambda m: 1 - m["refusal_rate"], True),
        ("Refusal Rate (↑)",            lambda m: m["refusal_rate"],     True),
        ("BERT-F1",                           lambda m: m["bert_f1"],    False),
        ("CLIP",                              lambda m: m["clip_score"], False),
        ("ROUGE-L",                           lambda m: m["rouge_f1"],   False),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.2))
    for ax, (title, fn, fixed) in zip(axes, metrics):
        # thin reference lines for each component mode
        for mode in RETAIN_MODES:
            data = per_mode.get(mode, {})
            if not data:
                continue
            ts = sorted(data)
            ax.plot(ts, [fn(data[t]) * 100 for t in ts], marker="o", markersize=3,
                    linewidth=1, color=MODE_COLORS[mode], alpha=0.35,
                    label=f"{mode}", zorder=1)
        # bold combined (num_pairs-weighted) retain line
        if retain_combined:
            ts = sorted(retain_combined)
            ax.plot(ts, [fn(retain_combined[t]) * 100 for t in ts], marker="o",
                    markersize=5, linewidth=2.6, color=RETAIN_COLOR,
                    label="Retain (weighted)", zorder=3)
        _style_axis(ax, title)
        if fixed:
            ax.set_ylim(0, 100)
    axes[0].set_ylabel("Score (×100)", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.04), frameon=True, fontsize=11)
    fig.suptitle(f"Retain metrics per time step  ({METHOD}, HU+UH+UU weighted by num_pairs)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    _save(fig, "retain_metrics.png")


def plot_forget(forget_data):
    metrics = [
        ("Answer Rate (↓ = forgotten)", lambda m: 1 - m["refusal_rate"],        True),
        ("Refusal Rate (↑)",            lambda m: m["refusal_rate"],            True),
        ("Context-aware Accuracy (↑)",  lambda m: m["context_aware_accuracy"],  True),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.2))
    ts = sorted(forget_data)
    for ax, (title, fn, fixed) in zip(axes, metrics):
        ys = [fn(forget_data[t]) * 100 for t in ts]
        ax.plot(ts, ys, marker="o", markersize=4, linewidth=2,
                color=FORGET_COLOR, label="HH", alpha=0.9)
        _style_axis(ax, title)
        if fixed:
            ax.set_ylim(0, 100)
    axes[0].set_ylabel("Score (×100)", fontsize=11)
    fig.suptitle(f"Forget metrics per time step  ({METHOD}, HH)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "forget_metrics.png")


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] saved {path}")


def main():
    raw_by_mode = {mode: load_retain_raw(mode) for mode in RETAIN_MODES}
    per_mode = {mode: per_mode_curve(raw) for mode, raw in raw_by_mode.items()}
    retain_combined = combine_retain(raw_by_mode)
    print(f"[PLOT] retain timesteps loaded per mode: "
          f"{ {m: len(d) for m, d in raw_by_mode.items()} }")
    print(f"[PLOT] combined retain timesteps: {sorted(retain_combined)}")
    for t in sorted(retain_combined):
        print(f"        t={t:2d}  num_pairs={retain_combined[t]['num_pairs']:5d}  "
              f"refusal={retain_combined[t]['refusal_rate']:.3f}  "
              f"bert_f1={retain_combined[t]['bert_f1']:.3f}")
    plot_retain(retain_combined, per_mode)

    forget_data = load_forget(FORGET_MODE)
    print(f"[PLOT] forget (HH) timesteps loaded: {len(forget_data)}")
    if forget_data:
        plot_forget(forget_data)
    else:
        print("[PLOT] no forget data found, skipping forget figure")


if __name__ == "__main__":
    main()
