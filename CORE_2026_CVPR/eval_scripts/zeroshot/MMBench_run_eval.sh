#!/bin/bash
# Zero-shot baseline: pretrained LVLM only (no unlearning / no CBL / no router).

MODE="seen"
START_TIME_STEP=15
END_TIME_STEP=15
EVAL_DEVICE=0
TEXT_DIR_TEMPLATE="results/Zeroshot_VLM_Bench/{}/"
RUN_NAME="Zeroshot"
BENCHMARK='MMBench_v1.0'

python ./eval_scripts/eval_cl_VLM_Bench.py \
    --mode "$MODE" \
    --start_time_step "$START_TIME_STEP" \
    --end_time_step "$END_TIME_STEP" \
    --eval_device "$EVAL_DEVICE" \
    --text_dir_template "$TEXT_DIR_TEMPLATE" \
    --run_name "$RUN_NAME" \
    --benchmark "$BENCHMARK" \
    --zero_shot
