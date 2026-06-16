"""
Thin re-export of the shared task/keyword definitions and checkpoint helpers
(see utils/cl.py at the repo root). Kept so eval scripts can keep doing
`from find_last_ckpt import ...`.
"""
import os
import sys

# put repo root on sys.path so `utils.cl` resolves when launched as
# `python eval_scripts/eval_cl_*.py`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.cl import (  # noqa: F401
    task_info,
    keywords,
    imagenet_r_keywords,
    split_keywords,
    keyword_list,
    find_latest_checkpoint,
    ensure_dir_exists,
    ensure_file_exists,
)
