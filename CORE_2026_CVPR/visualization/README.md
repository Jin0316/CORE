# Visualization

Plots for the `CORE` run. Output PNGs are written to `visualization/figures/`.

Run inside the project env (the scripts need `numpy`/`matplotlib`, and the expert
plot also needs `torch`):

```bash
conda activate core
python visualization/plot_timestep_metrics.py
python visualization/plot_expert_activation.py
```

## `plot_timestep_metrics.py`

Per-time-step metrics, reading the already-computed JSON under
`metric_retain_per_timestep/` and `metric_forget_per_timestep/`.

- **`retain_metrics.png`** — RETAIN data (reference = Zeroshot). 5 subplots
  (Answer Rate, Refusal Rate, BERT-F1, CLIP, ROUGE-L), one line per harm mode
  (HH / HU / UH / UU).
- **`forget_metrics.png`** — FORGET data. Only the HH mode was evaluated and the
  forget JSONs contain only refusal-based metrics, so this plots Answer Rate,
  Refusal Rate and Context-aware Accuracy vs time step.

## `plot_expert_activation.py`

- **`expert_activation.png`** — `[task × expert]` heatmap of the router load
  matrix (`r_load`) read from the final time step's checkpoint
  (`minigpt4/output/CORE/15/checkpoint_1.pth` by default). Left:
  raw activation frequency; right: row-normalized per-task distribution.

  Point at a different run / time step / file with `--run`, `--timestep`, `--ckpt`.
