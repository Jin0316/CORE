#!/bin/bash
# Zero-shot baseline: pretrained LVLM only (no unlearning / no CBL / no router).
# Evaluates the same safety data as eval_scripts/<n>_concepts/.

MODE="seen"
START_TIME_STEP=15
END_TIME_STEP=15
EVAL_DEVICE=4
TEXT_DIR_TEMPLATE="results/Zeroshot_UU/{}/"
RUN_NAME="Zeroshot"
HARM_COMB='uu'

python ./eval_scripts/eval_cl_new.py \
    --mode "$MODE" \
    --start_time_step "$START_TIME_STEP" \
    --end_time_step "$END_TIME_STEP" \
    --eval_device "$EVAL_DEVICE" \
    --text_dir_template "$TEXT_DIR_TEMPLATE" \
    --run_name "$RUN_NAME" \
    --harm_combination "$HARM_COMB" \
    --zero_shot
