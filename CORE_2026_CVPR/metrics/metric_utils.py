"""Shared helpers for the metric-aggregation scripts (eval_text.py)."""
import numpy as np
from typing import Dict


def calculate_average_metrics_from_json(json_data: Dict) -> Dict:
    """Calculate average metrics from a per-timestep BERTScore JSON."""
    if "results" not in json_data:
        return {}

    valid_results = [r for r in json_data["results"].values()
                     if "error" not in r and r.get("num_pairs", 0) > 0]

    if not valid_results:
        return {}

    return {
        'bert_f1': np.mean([r["bert_score_mean"]["f1"] for r in valid_results]),
        'rouge_f1': np.mean([r["rougeL_mean"]["f1"] for r in valid_results]),
        'clip_score': np.mean([r["clip_score_mean"] for r in valid_results]),
        'refusal_rate': np.mean([r["refusal"]["refusal_rate"] for r in valid_results]),
    }
