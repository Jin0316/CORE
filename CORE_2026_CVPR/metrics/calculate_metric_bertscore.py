# metrics_text_similarity.py
import math
from collections import defaultdict
from typing import List, Tuple, Optional, Dict
import re

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import json


import argparse
import open_clip



import warnings
import os
import transformers

# refusal detection shared with calculate_metric_crr.py (identical definition)
from refusal_patterns import is_refusal_pattern

# transformers 경고 메시지 숨기기
transformers.logging.set_verbosity_error()
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Some weights of .* were not initialized from the model checkpoint")


@torch.no_grad()
def clip_score(
    cands: List[str],
    refs: List[str],
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: Optional[str] = None,
    batch_size: int = 32,
) -> List[float]:
    """
    Compute cosine similarity between CLIP text embeddings of cands and refs.
    Returns a list of scores in [0, 1].
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model, _, tokenizer = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device).eval()

    scores = []
    N = len(cands)
    for start in range(0, N, batch_size):
        end = min(N, start + batch_size)
        cand_batch = cands[start:end]
        ref_batch = refs[start:end]

        cand_tokens = open_clip.tokenize(cand_batch).to(device)
        ref_tokens = open_clip.tokenize(ref_batch).to(device)

        with torch.no_grad():
            cand_feats = model.encode_text(cand_tokens)
            ref_feats = model.encode_text(ref_tokens)
            cand_feats = F.normalize(cand_feats, dim=-1)
            ref_feats = F.normalize(ref_feats, dim=-1)
            cos_sim = (cand_feats * ref_feats).sum(dim=-1)  # [B]
            scores.extend(cos_sim.cpu().tolist())

    return scores


SPECIAL_TOKEN_IDS_CACHE = {}


def _get_special_ids(tokenizer) -> set:
    """Return a set of special token IDs for fast masking."""
    if id(tokenizer) in SPECIAL_TOKEN_IDS_CACHE:
        return SPECIAL_TOKEN_IDS_CACHE[id(tokenizer)]
    special = set()
    for k in ["cls_token_id", "sep_token_id", "pad_token_id", "bos_token_id", "eos_token_id", "mask_token_id"]:
        v = getattr(tokenizer, k, None)
        if v is not None:
            special.add(v)
    SPECIAL_TOKEN_IDS_CACHE[id(tokenizer)] = special
    return special


@torch.no_grad()
def compute_idf(
    texts: List[str],
    tokenizer: AutoTokenizer,
) -> defaultdict:
    """
    Compute IDF over a corpus on token IDs.

    idf[token_id] = log((N + 1) / (df + 1)) + 1
    """
    special = _get_special_ids(tokenizer)
    N = len(texts)
    df = defaultdict(int)

    for t in texts:
        ids = tokenizer(t, add_special_tokens=True, return_attention_mask=False, return_tensors=None)["input_ids"]
        if isinstance(ids[0], list):
            ids = ids[0]
        seen = set(x for x in ids if x not in special)
        for tok in seen:
            df[tok] += 1

    idf = defaultdict(lambda: 0.0)
    for tok, c in df.items():
        idf[tok] = math.log((N + 1) / (c + 1)) + 1.0
    return idf


def _gather_token_embeddings(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: List[str],
    device: torch.device,
    max_length: Optional[int] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Encode texts to contextual token embeddings and masks.
    Returns per-sample lists: embeddings [L, H], attention_mask [L], token_ids [L].
    """
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=(max_length is not None),
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=True,
        return_attention_mask=True,
    ).to(device)

    outputs = model(**encoded, output_hidden_states=False, return_dict=True)
    hidden = outputs.last_hidden_state  # [B, L, H]

    B = hidden.size(0)
    embs, masks, ids = [], [], []
    attention = encoded["attention_mask"]  # [B, L]
    input_ids = encoded["input_ids"]       # [B, L]
    for i in range(B):
        embs.append(hidden[i])           # [L, H]
        masks.append(attention[i])       # [L]
        ids.append(input_ids[i])         # [L]
    return embs, masks, ids


def _apply_special_token_mask(
    ids: torch.Tensor,
    attn_mask: torch.Tensor,
    special_ids: set,
) -> torch.Tensor:
    """Boolean mask for valid content tokens"""
    valid = attn_mask.bool()
    for sid in special_ids:
        valid = valid & (ids != sid)
    return valid


def _cosine_similarity_matrix(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    X: [Lx, H], Y: [Ly, H]
    Return cosine similarity matrix S: [Lx, Ly]
    """
    Xn = F.normalize(X, p=2, dim=-1)
    Yn = F.normalize(Y, p=2, dim=-1)
    return Xn @ Yn.T


def _weighted_mean(values: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    """
    values: [L]
    weights: [L] or None
    """
    if values.numel() == 0:
        return torch.tensor(0.0, device=values.device)
    if weights is None:
        return values.mean()
    wsum = weights.sum()
    if wsum.item() == 0.0:
        return values.mean()
    return (values * weights).sum() / wsum


@torch.no_grad()
def bert_score(
    cands: List[str],
    refs: List[str],
    model_name: str = "bert-base-uncased",
    device: Optional[str] = None,
    batch_size: int = 8,
    max_length: Optional[int] = None,
    use_idf: bool = False,
    idf_on_refs: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute BERTScore for each candidate–reference pair.

    Returns tensors P, R, F1 of shape [N].
    """
    assert len(cands) == len(refs), "Lengths must match."
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    special = _get_special_ids(tokenizer)

    idf_dict = None
    if use_idf:
        base_texts = refs if idf_on_refs else cands
        idf_dict = compute_idf(base_texts, tokenizer)

    N = len(cands)
    P_list, R_list, F1_list = [], [], []

    for start in range(0, N, batch_size):
        end = min(N, start + batch_size)
        cand_batch = cands[start:end]
        ref_batch = refs[start:end]

        cand_embs, cand_masks, cand_ids = _gather_token_embeddings(
            model, tokenizer, cand_batch, device, max_length
        )
        ref_embs, ref_masks, ref_ids = _gather_token_embeddings(
            model, tokenizer, ref_batch, device, max_length
        )

        for i in range(len(cand_batch)):
            cmask = _apply_special_token_mask(cand_ids[i], cand_masks[i], special)
            rmask = _apply_special_token_mask(ref_ids[i], ref_masks[i], special)

            X = cand_embs[i][cmask]  # [Lc, H]
            Y = ref_embs[i][rmask]   # [Lr, H]

            if X.numel() == 0 or Y.numel() == 0:
                zero = torch.tensor(0.0, device=device)
                P_list.append(zero)
                R_list.append(zero)
                F1_list.append(zero)
                continue

            S = _cosine_similarity_matrix(X, Y)  # [Lc, Lr]

            p_token = S.max(dim=1).values  # precision tokens
            r_token = S.max(dim=0).values  # recall tokens

            p_w = r_w = None
            if idf_dict is not None:
                cid = cand_ids[i][cmask]
                rid = ref_ids[i][rmask]
                p_w = torch.tensor([idf_dict[int(t.item())] for t in cid], device=device)
                r_w = torch.tensor([idf_dict[int(t.item())] for t in rid], device=device)

            P = _weighted_mean(p_token, p_w)
            R = _weighted_mean(r_token, r_w)
            F1 = 2 * P * R / (P + R + 1e-12)

            P_list.append(P)
            R_list.append(R)
            F1_list.append(F1)

    return torch.stack(P_list), torch.stack(R_list), torch.stack(F1_list)


# -----------------------------
# ROUGE-L (Longest Common Subsequence)
# -----------------------------

def _lcs_length(a: List[str], b: List[str]) -> int:
    """Classic dynamic programming for LCS length over token sequences."""
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0
    # 2-row DP to reduce memory
    prev = [0] * (lb + 1)
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        ai = a[i - 1]
        for j in range(1, lb + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = curr[j - 1] if curr[j - 1] >= prev[j] else prev[j]
        prev, curr = curr, prev  # swap
    return prev[lb]


def _simple_word_tokenize(s: str, lowercase: bool = True) -> List[str]:
    """Whitespace word tokenizer for ROUGE-L."""
    if lowercase:
        s = s.lower()
    return s.strip().split()


def rouge_l(
    cands: List[str],
    refs: List[str],
    beta: float = 1.2,
    lowercase: bool = True,
    tokenizer_fn: Optional[callable] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute ROUGE-L precision, recall, F1 for each candidate–reference pair.

    R_lcs = LCS_len / len(ref_tokens)
    P_lcs = LCS_len / len(cand_tokens)
    F_lcs = (1 + beta^2) * P_lcs * R_lcs / (R_lcs + beta^2 * P_lcs)
    """
    assert len(cands) == len(refs), "Lengths must match."
    if tokenizer_fn is None:
        tokenizer_fn = lambda s: _simple_word_tokenize(s, lowercase=lowercase)

    P_list, R_list, F_list = [], [], []
    for cand, ref in zip(cands, refs):
        tc = tokenizer_fn(cand)
        tr = tokenizer_fn(ref)
        if len(tc) == 0 or len(tr) == 0:
            P_list.append(0.0)
            R_list.append(0.0)
            F_list.append(0.0)
            continue
        lcs = _lcs_length(tc, tr)
        R = lcs / len(tr)
        P = lcs / len(tc)
        denom = R + (beta ** 2) * P
        F = (1 + beta ** 2) * P * R / denom if denom > 0 else 0.0
        P_list.append(P)
        R_list.append(R)
        F_list.append(F)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.tensor(P_list, device=device, dtype=torch.float32), \
           torch.tensor(R_list, device=device, dtype=torch.float32), \
           torch.tensor(F_list, device=device, dtype=torch.float32)




def count_refusals(
    texts: List[str],
    categories: Optional[List[Optional[str]]] = None,
) -> Dict[str, object]:
    """
    Count refusal responses over a list of texts.
    If categories is provided, compute per category counts.
    """
    total = len(texts)
    refuse = 0
    by_cat: Dict[str, Dict[str, float]] = {}

    if categories is None:
        categories = [None] * total

    # Initialize counters
    for cat in categories:
        key = cat if cat is not None else "UNK"
        if key not in by_cat:
            by_cat[key] = {"total": 0, "refusal": 0, "refusal_rate": 0.0}

    # Count (uses the same refusal definition as calculate_metric_crr.py)
    for txt, cat in zip(texts, categories):
        key = cat if cat is not None else "UNK"
        by_cat[key]["total"] += 1
        if is_refusal_pattern(txt):
            refuse += 1
            by_cat[key]["refusal"] += 1

    # Rates
    for key, v in by_cat.items():
        v["refusal_rate"] = (v["refusal"] / v["total"]) if v["total"] else 0.0

    summary = {
        "total": total,
        "refusal": refuse,
        "refusal_rate": (refuse / total) if total else 0.0,
        "by_category": by_cat,
    }
    return summary

def load_responses(path: str) -> List[str]:
    """
    Read a txt file and return only response lines.
    - Skip lines starting with "[Category]" (metadata headers).
    - Skip completely empty lines.
    """
    responses = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("[Category]"):
                continue
            responses.append(line)
    return responses


def evaluate_response_files(
    cand_file: str,
    ref_file: str,
    model_name: str = "bert-base-uncased",
    batch_size: int = 8,
    use_idf: bool = True,
    max_length: Optional[int] = None,
):
    """
    Evaluate two txt files containing candidate outputs and reference outputs.
    Headers are skipped; only response lines are evaluated.
    """
    cands = load_responses(cand_file)
    refs  = load_responses(ref_file)

    if len(cands) != len(refs):
        raise ValueError(f"Length mismatch: cand={len(cands)}, ref={len(refs)}")

    # refusal is counted over ALL candidate responses (same scope/denominator as
    # calculate_metric_crr.py), independent of the candidate-reference pairing.
    all_cands = list(cands)

    # skip if either side is empty
    pairs = [(c, r) for c, r in zip(cands, refs) if c.strip() and r.strip()]
    if not pairs:
        return {"num_pairs": 0, "note": "No valid pairs after skipping empty responses."}

    cands, refs = zip(*pairs)

    # Compute metrics
    P, R, F1 = bert_score(list(cands), list(refs),
                          model_name=model_name,
                          batch_size=batch_size,
                          use_idf=use_idf,
                          max_length=max_length)
    rP, rR, rF = rouge_l(list(cands), list(refs), beta=1.2, lowercase=True)
    clip_scores = clip_score(list(cands), list(refs), model_name="ViT-B-32", pretrained="openai")
    clip_score_mean = float(torch.tensor(clip_scores).mean().cpu())

    refusal_stats = count_refusals(all_cands)


    return {
        "num_pairs": len(cands),
        "bert_score_mean": {
            "precision": float(P.mean().cpu()),
            "recall": float(R.mean().cpu()),
            "f1": float(F1.mean().cpu()),
        },
        "rougeL_mean": {
            "precision": float(rP.mean().cpu()),
            "recall": float(rR.mean().cpu()),
            "f1": float(rF.mean().cpu()),
        },
        "clip_score_mean": clip_score_mean,
        "refusal": refusal_stats,
    }


import json
import os
import sys
from typing import Dict, List, Optional

# task_info (utils/cl.py) maps each timestep to its (task type, subset index); used to
# build file names and keep this scorer paired with eval_cl_new.py, order-independently.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.cl import task_info

def evaluate_timestep_range(
    base_cand_dir: str,
    base_ref_dir: str,
    current_timestep: int,
    model_name: str = "bert-base-uncased",
    batch_size: int = 8,
    use_idf: bool = True,
    max_length: Optional[int] = None,
    save_results: bool = True,
    output_dir: str = "timestep_results"
) -> Dict[str, Dict]:
    """
    현재 timestep까지의 모든 파일들을 평가하고 결과를 저장.
    
    Args:
        base_cand_dir: candidate 파일들의 베이스 디렉토리
        base_ref_dir: reference 파일들의 베이스 디렉토리  
        current_timestep: 현재 timestep (0부터 current_timestep까지 평가)
        save_results: 결과를 파일로 저장할지 여부
        output_dir: 결과 저장 디렉토리
    
    Returns:
        timestep별 평가 결과 딕셔너리
    """
    all_results = {}
    
    # 출력 디렉토리 생성
    if save_results and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # uh/uu experiments are detected from the candidate dir name (_UH / _UU).
    is_uh_or_uu_mode = any(mode in base_cand_dir for mode in ['_UH', '_UU'])

    # Reference is always the final-timestep model's outputs.
    ref_timestep = len(task_info) - 1

    # First safe_PO_IN task; the only one kept in uh/uu modes (see skip below).
    first_in_step = min(i for i in range(len(task_info)) if task_info[str(i)][0] == 'safe_PO_IN')

    for step in range(current_timestep + 1):
        task, subset_idx = task_info[str(step)]

        # uh/uu: every safe_PO_IN task is the identical eval set, so keep only the first.
        if is_uh_or_uu_mode and task == 'safe_PO_IN' and step != first_in_step:
            continue

        # Same naming as eval_cl_new.py: seen_task_{task}_{subset}_eval.txt.
        eval_filename = f"seen_task_{task}_{subset_idx}_eval.txt"
        cand_file = os.path.join(base_cand_dir, f"{current_timestep}", eval_filename)
        ref_file = os.path.join(base_ref_dir, str(ref_timestep), eval_filename)

        # Deduped safe_PO_IN tasks are already skipped above, so a missing file here is
        # genuinely absent: warn and skip the step instead of failing downstream.
        if not os.path.exists(cand_file):
            print(f"[METRIC] Warning: candidate file not found, skipping step {step}: {cand_file}")
            continue
        if not os.path.exists(ref_file):
            print(f"[METRIC] Warning: reference file not found, skipping step {step}: {ref_file}")
            continue

        try:
            # 평가 수행
            results = evaluate_response_files(
                cand_file=cand_file,
                ref_file=ref_file,
                model_name=model_name,
                batch_size=batch_size,
                use_idf=use_idf,
                max_length=max_length
            )
            # 추가 메타데이터
            results["timestep"] = step
            results["candidate_file"] = cand_file
            results["reference_file"] = ref_file
            
            all_results[f"timestep_{step}"] = results
            
            print(f"[METRIC] Timestep {step} completed. Pairs: {results.get('num_pairs', 0)}")
            
        except Exception as e:
            print(f"[METRIC] Error evaluating timestep {step}: {str(e)}")
            all_results[f"timestep_{step}"] = {"error": str(e)}
    
    # 전체 결과에 메타데이터 추가
    summary = {
        "metadata": {
            "current_timestep": current_timestep,
            "total_evaluated": len([k for k in all_results.keys() if "error" not in all_results[k]]),
            "base_cand_dir": base_cand_dir,
            "base_ref_dir": base_ref_dir,
        },
        "results": all_results
    }
    
    # 결과 저장
    if save_results:
        output_file = os.path.join(output_dir, f"evaluation_timestep_{current_timestep}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[METRIC] Results saved to: {output_file}")
    
    return summary


def get_timestep_summary(results: Dict[str, Dict]) -> Dict:
    """
    timestep별 결과에서 주요 메트릭 요약 생성
    """
    if "results" not in results:
        return {}
    
    summary_data = {
        "timesteps": [],
        "bert_f1": [],
        "rouge_f1": [],
        "clip_score": [],
        "refusal_rates": []
    }
    
    for key, result in results["results"].items():
        if "error" in result or "num_pairs" not in result:
            continue
            
        timestep = result.get("timestep", -1)
        summary_data["timesteps"].append(timestep)
        summary_data["bert_f1"].append(result["bert_score_mean"]["f1"])
        summary_data["rouge_f1"].append(result["rougeL_mean"]["f1"])
        summary_data["clip_score"].append(result["clip_score_mean"])
        summary_data["refusal_rates"].append(result["refusal"]["refusal_rate"])
    
    return summary_data


def save_timestep_csv(results: Dict[str, Dict], output_path: str):
    """
    timestep별 결과를 CSV 파일로 저장
    """
    import csv
    
    summary = get_timestep_summary(results)
    if not summary["timesteps"]:
        print("[METRIC] No valid results to save as CSV")
        return
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "bert_f1", "rouge_f1", "clip_score", "refusal_rate"])
        
        for i in range(len(summary["timesteps"])):
            writer.writerow([
                summary["timesteps"][i],
                summary["bert_f1"][i],
                summary["rouge_f1"][i], 
                summary["clip_score"][i],
                summary["refusal_rates"][i]
            ])
    
    print(f"[METRIC] CSV summary saved to: {output_path}")



# -----------------------------
# Example
# -----------------------------
# if __name__ == "__main__":
#     cand = "./results/Ours_HH/11/seen_task_safe_PO_10_eval.txt"
#     ref = "./results/Zeroshot_VICUNA_HH/11/seen_task_safe_PO_10_eval.txt"
#     results = evaluate_response_files(
#         cand_file=cand,
#         ref_file=ref,
#         batch_size=16,
#     )

#     print(json.dumps(results, indent=2, ensure_ascii=False))
    
    
# 사용 예시
def parse_arguments():
    """
    Parse command line arguments for evaluation script
    """
    parser = argparse.ArgumentParser(description='Evaluate timestep range for text similarity metrics')
    
    parser.add_argument('--base_cand_dir', type=str, required=True,
                        help='Base directory for candidate (our model) files, e.g. ./results/CORE')
    parser.add_argument('--base_ref_dir', type=str, default='./results/Zeroshot',
                        help='Base directory for reference files (zero-shot responses)')
    parser.add_argument('--method', type=str, required=True,
                        help='Method name used to name the output dir (e.g. CORE)')
    parser.add_argument('--start_timestep', type=int, default=0,
                        help='Base directory for candidate files')
    parser.add_argument('--end_timestep', type=int, default=15,
                        help='Base directory for candidate files')
    parser.add_argument('--mode', type=str, required=True,
                        help='Mode(s) to evaluate (single: HH, HU, UH, UU or multiple: HH/HU/UU)')
    
    return parser.parse_args()

if __name__ == "__main__":
    # 예시 사용법
    args = parse_arguments()
    save_path = './metric_retain_per_timestep/'
    if '/' in args.mode:
        mode = args.mode.split('/')
    else:
        mode = [args.mode]

    for md in mode:
        base_cand_dir = args.base_cand_dir
        # reference = zero-shot (pretrained LVLM) responses; retain quality measures
        # how close our model stays to the original answers. Override with --base_ref_dir.
        base_ref_dir = args.base_ref_dir
        base_cand_dir = base_cand_dir + f'_{md}'
        base_ref_dir = base_ref_dir + f'_{md}'
        start_time_step = args.start_timestep
        end_time_step = args.end_timestep
        
        # 평가 수행
        for current_timestep in range(start_time_step, end_time_step + 1):
            results = evaluate_timestep_range(
                base_cand_dir=base_cand_dir,
                base_ref_dir=base_ref_dir,
                current_timestep=current_timestep,
                batch_size=16,
                save_results=True,
                output_dir=f"{save_path}/timestep_evaluation_results_{args.method}_{md}"
            )
            
            # CSV 요약 저장
            save_timestep_csv(results, f"{save_path}/timestep_evaluation_results_{args.method}_{md}/summary_{current_timestep}.csv")
            
            # 간단한 결과 출력
            summary = get_timestep_summary(results)
            print(f"\n[METRIC] Evaluation Summary:")
            print(f"[METRIC] Total timesteps evaluated: {len(summary['timesteps'])}")
            if summary['timesteps']:
                print(f"[METRIC] Average BERT F1: {sum(summary['bert_f1'])/len(summary['bert_f1']):.4f}")
                print(f"[METRIC] Average ROUGE-L F1: {sum(summary['rouge_f1'])/len(summary['rouge_f1']):.4f}")
                print(f"[METRIC] Average CLIP Score: {sum(summary['clip_score'])/len(summary['clip_score']):.4f}")
                print(f"[METRIC] Average Refusal Rate: {sum(summary['refusal_rates'])/len(summary['refusal_rates']):.4f}")


"""
Example:
CUDA_VISIBLE_DEVICES=0 python3 calculate_bertscore.py --method CORE --mode HH/HU/UH/UU --base_cand_dir ./results/CORE
"""