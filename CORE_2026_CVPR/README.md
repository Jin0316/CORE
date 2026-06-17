# CORE — Which Concepts to Forget and How to Refuse? (CVPR 2026)

Official code for **"Which Concepts to Forget and How to Refuse? Decomposing Concepts
for Continual Unlearning in Large Vision-Language Models"** (CVPR 2026).

> This is the code directory. For the project overview, repository layout, and dataset
> documentation, see the [top-level README](../README.md) and
> [`datasets/README.md`](../datasets/README.md).

---

## Overview

**CORE** enables a large vision-language model to **selectively refuse specific
image-instruction pairs** in response to a stream of deletion requests, while
**preserving general utility**. Rather than tying refusal to whole deletion targets —
which distorts shared representations and causes inappropriate refusals — CORE grounds
refusal behavior in **fine-grained descriptions of visual and textual concepts
decomposed from each deletion target**.

The method follows the two questions in the title:

- **Which** concepts to forget — a **concept bottleneck layer** (the *concept modulator*)
  identifies which visual-linguistic concept combinations characterize each forget
  category.
- **How** to refuse — a **multi-modal, concept-driven router** produces concept-aligned
  refusals through a **mixture of refusal experts** (*refusers*), reusing refusers for
  tasks that share concepts and adapting underutilized ones for novel concepts.

The fine-grained concept descriptions live in [`CONCEPTS/GPT/`](CONCEPTS/GPT/)
(`All_concepts_image.json` for visual concepts, `All_concepts_inst.json` for textual /
instruction concepts) — one list of `descriptions` per harmful keyword. These
components are trained per time step by the three stages described in
[§2 Train](#2-train).

---
## 1. Setup

### 1.1 Environment

```bash
# Docker
docker build -t core2026:core CORE26_CVPR
docker run --gpus all -it -v /path/to/2026_CVPR_CORE:/workspace core2026:core
# …or Conda
conda activate core
```

### 1.2 Directory layout (inside the container)

```
/workspace/                  # repo root, mounted into the container
├── CORE26_CVPR/             # this code directory (working directory)
├── datasets/                # Safe-Eraser, ImageNet-R, Standard_VLM_Benchmarks
├── pretrained/              # pretrained_minigpt4_7b_Vicuna.pth
└── Vicuna-7b/               # Vicuna-7B weights
```

### 1.3 Checkpoints

Download the two model checkpoints below ([MiniGPT-4 release](https://github.com/Vision-CAIR/MiniGPT-4)).
Keep the default paths and nothing needs editing; otherwise update the configs listed:

| checkpoint | default path | config to edit if different |
|------------|--------------|-----------------------------|
| MiniGPT-4 pretrained `.pth` | `/workspace/pretrained/pretrained_minigpt4_7b_Vicuna.pth` | `train_scripts/train_cl_config.py`, `eval_scripts/eval_cl_new.py`, `eval_scripts/eval_cl_VLM_Bench.py` |
| Vicuna-7B HuggingFace folder | `/workspace/Vicuna-7b/` | `minigpt4/configs/models/minigpt4_vicuna0.yaml` |

Adjust GPU IDs for your machine: `GPU_DEVICES` / `VISION_DEVICE` in
`train_scripts/train_cl_config.py` (and `world_size` in `train_configs/*.yaml`), and
`EVAL_DEVICE` in `eval_scripts/**/*.sh`.

---

## 2. Train

```bash
python train_scripts/train_cl.py 
```

`train_scripts/train_cl.py` sequentially unlearns harmful concepts over **16 time
steps**. At each step it runs three stages — concept bottleneck layer, main
fine-tuning, then router — detailed under **Per-step stages** below. The run is saved
under `minigpt4/output/CORE/`.

**Task sequence** (defined in `train_cl.py`) — all 16 tasks are unlearning tasks:

- **12 `safe_PO` tasks** — 60 harmful concepts across six safety types (sexual, violence,
  illegal activity, weapons, privacy, hate/discrimination), grouped into five categories per
  task.
- **4 `safe_PO_IN` tasks** — ImageNet-R concepts (20 categories per task).

**Per-step stages** (backends in `train_scripts/backends/`):

1. **CBL** (`train_CBL.py`) — train the concept bottleneck layer for the step's
   keywords (identify *which* concept combinations characterize the forget category).
2. **Main** (`train.py`) — fine-tune the mixture of refusers (router + set of refusers) with the trained CBL.
3. **Router** (`train_router.py`) — finetune the refusal router.  

---

## 3. Eval

Our model is evaluated with `eval_scripts/core/`. Each safety script is named by a
harmful/unharmful **image** × harmful/unharmful **text** combination (e.g. `HH` =
harmful image + harmful text).

Evaluation loads model checkpoints from:

```
minigpt4/output/<RUN_NAME>/<time_step>/checkpoint_*.pth
minigpt4/output/<RUN_NAME>/cbl/cbl_<time_step>.pt
```

The provided scripts set `RUN_NAME="CORE"`, matching the default training run. If you
change the training `run_name`, update `RUN_NAME` in `eval_scripts/core/*.sh`.
Zero-shot eval uses only the MiniGPT-4 pretrained checkpoint from [§1.3](#13-checkpoints).

**Forget set** — harmful inputs that must be **refused** (`HH`, `HU`, `UH`).

```bash
bash eval_scripts/core/HH_run_eval_safety.sh
bash eval_scripts/core/HU_run_eval_safety.sh
bash eval_scripts/core/UH_run_eval_safety.sh
```

**Retain set** — fully benign inputs that must still be **answered normally** (`UU`),
verifying that general utility is preserved.

```bash
bash eval_scripts/core/UU_run_eval_safety.sh
```

**LVLM benchmarks** — general vision-language ability is preserved.

```bash
bash eval_scripts/core/MMBench_run_eval.sh
bash eval_scripts/core/ScienceQA_run_eval.sh
bash eval_scripts/core/SEEDBench_run_eval.sh
```

### Zero-shot 

`eval_scripts/zeroshot/` evaluates the **pretrained LVLM only** (no unlearning, no
CBL/router, `--zero_shot`) on the exact same data. Since the pretrained model does not
change across steps, it is evaluated once — at the final time step (15); results go to
`results/Zeroshot_*/` and serve as the reference for the Retain and LVLM metrics.

```bash
# Forget set
bash eval_scripts/zeroshot/HH_run_eval_safety.sh
bash eval_scripts/zeroshot/HU_run_eval_safety.sh
bash eval_scripts/zeroshot/UH_run_eval_safety.sh
# Retain set
bash eval_scripts/zeroshot/UU_run_eval_safety.sh
# LVLM benchmarks
bash eval_scripts/zeroshot/MMBench_run_eval.sh
bash eval_scripts/zeroshot/ScienceQA_run_eval.sh
bash eval_scripts/zeroshot/SEEDBench_run_eval.sh
```

---

## 4. Metric

The eval step writes raw model responses to `results/` (`CORE_<MODE>/` and
`Zeroshot_<MODE>/`). The scripts in `metrics/` parse those responses into the final
metrics for each set.

**Forget set** — refusal rate (classify each response as refusal vs. answer).

```bash
# results/  ->  metric_forget_per_timestep/  (per-timestep JSON)
python metrics/calculate_metric_crr.py --base_cand_dir ./results --mode HH

# metric_forget_per_timestep/  ->  metric_forget_summary/  (avg / last tables)
python metrics/eval_refusal.py
```

**Retain set** — response quality as similarity to the **zero-shot (pretrained)
responses** (BERTScore / ROUGE / CLIP-score). Run the zero-shot eval first so
`results/Zeroshot_<MODE>/` exists as the reference.

```bash
# our model (candidate) vs zero-shot (reference) -> ./metric_retain_per_timestep/
python metrics/calculate_metric_bertscore.py \
    --method CORE \
    --base_cand_dir ./results/CORE \
    --base_ref_dir  ./results/Zeroshot \
    --mode HH/HU/UH/UU

# metric_retain_per_timestep/  ->  metric_overall_summary/  (BERTScore/ROUGE/CLIP + refusal)
python metrics/eval_text.py
```

The refusal value reported here uses the **same definition as the Forget metric**
(shared `metrics/refusal_patterns.py`), so it matches `metric_forget_summary/`.

**LVLM benchmarks** — accuracy per benchmark across time steps. Run the zero-shot pass
first (`--zero_shot`) to define the baseline scores (`zeroshot_scores.json`), then score
our model against that baseline.

```bash
# -> metric_lvlm_benchmark/
python metrics/calculate_metric_LVLM_bench.py --base_path ./results/Zeroshot_VLM_Bench --zero_shot
python metrics/calculate_metric_LVLM_bench.py --base_path ./results/CORE_VLM_Bench
```

Figures are drawn by the scripts in `visualization/` (`plot_timestep_metrics.py`,
`plot_expert_activation.py`).

**Metric output folders** (`results/` holds the raw responses; metrics are written to):

| set | per-timestep (raw) | summary (final tables) |
|-----|--------------------|------------------------|
| Forget | `metric_forget_per_timestep/` | `metric_forget_summary/` |
| Retain | `metric_retain_per_timestep/` | `metric_overall_summary/` |
| LVLM | — | `metric_lvlm_benchmark/` |

> Note: refusal classification (`calculate_metric_crr.py`) only processes `HH` mode.

