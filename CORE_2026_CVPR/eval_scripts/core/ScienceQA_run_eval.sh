#!/bin/bash

# Shell script to run eval_cl.py with argparse arguments

# Define variables for arguments
MODE="seen"  # Change to "transfer" if needed
START_TIME_STEP=0
END_TIME_STEP=15
EVAL_DEVICE=2
TEXT_DIR_TEMPLATE="results/CORE_VLM_Bench/{}/"
RUN_NAME="CORE"
BENCHMARK='ScienceQA_TEST'

# Run the Python script with the arguments
python ./eval_scripts/eval_cl_VLM_Bench.py \
    --mode "$MODE" \
    --start_time_step "$START_TIME_STEP" \
    --end_time_step "$END_TIME_STEP" \
    --eval_device "$EVAL_DEVICE" \
    --text_dir_template "$TEXT_DIR_TEMPLATE" \
    --run_name "$RUN_NAME" \
    --benchmark "$BENCHMARK"
