import os
import re
import json
import argparse
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

# 숫자 단어 -> 숫자 매핑
NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10",
}


def normalize_text(text: str) -> str:
    """소문자화, 구두점 제거, 숫자 단어->숫자, 가벼운 복수형 제거."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
 
    for word, digit in NUM_WORDS.items():
        text = re.sub(rf"\b{word}\b", digit, text)
 
    # 길이 4 이상인 단어의 끝 s 만 제거 (is, as, us 등 짧은 단어는 보호)
    text = re.sub(r"\b(\w{3,}?)s\b", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
 
 
def extract_choice(pred_text: str) -> Optional[str]:
    """예측에서 모델이 고른 보기 글자(A-E)를 추출. 못 찾으면 None.
 
    관사 'a' 등을 보기 'A' 로 오인하지 않도록 '보기처럼 보이는 위치'에서만 추출한다.
    """
    p = pred_text.strip()
    if not p:
        return None
 
    # 1) 예측 전체가 단일 보기 글자: "B", "(B)", "B."
    m = re.fullmatch(r"\(?([A-Ea-e])\)?[.):：,。、]?", p)
    if m:
        return m.group(1).upper()
 
    # 2) 맨 앞이 "보기글자 + 구분자" 형태: "B. ...", "(C) ...", "D) ..."
    m = re.match(r"\(?([A-Ea-e])\)?\s*[.):：,]\s", p)
    if m:
        return m.group(1).upper()
 
    # 3) "answer / option / choice is X" 류
    #    보기 글자는 대문자(A-E)만 허용해 'the answer is a dog' 같은 오인을 방지
    m = re.search(
        r"(?i:answer|option|choice)\s*(?i:is|:|=|->)?\s*\(?([A-E])\)?(?![A-Za-z])",
        p,
    )
    if m:
        return m.group(1).upper()
 
    return None
 
 
def is_correct(
    gt_text: str, pred_text: str, gt_label: Optional[str] = None
) -> bool:
    """객관식 채점.
 
    - gt_label 이 있으면: 예측에서 고른 보기 글자를 추출해 정답 라벨과 정확히 비교.
    - 보기 글자를 못 찾았거나 gt_label 이 없으면: 정답 텍스트 내용으로 단순 폴백.
    """
    if gt_label:
        choice = extract_choice(pred_text)
        if choice is not None:
            return choice == gt_label.strip().upper()
        # 보기 글자를 못 찾은 경우에만 아래 내용 매칭으로 폴백
 
    gt_norm = normalize_text(gt_text)
    if not gt_norm:  # 빈 정답은 정답으로 처리하지 않음
        return False
    return gt_norm in normalize_text(pred_text)
 
 
def compute_accuracy_from_txt(
    path: str, limit_lines: int = 1000
) -> Optional[float]:
    """단일 파일의 accuracy 계산. 파일이 없으면 None."""
    if not os.path.exists(path):
        return None
 
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
 
    total, correct = 0, 0
    for i in range(0, min(limit_lines, len(lines) - 1), 2):
        gt_line = lines[i].strip()
        pred_line = lines[i + 1].strip()
 
        gt_match = re.search(r"Ans:\s*([A-E])\s*(.*?)\s*\|", gt_line)
        if not gt_match:
            continue
        gt_label = gt_match.group(1)
        gt_text = gt_match.group(2)
        pred_text = pred_line
 
        total += 1
        if is_correct(gt_text, pred_text, gt_label):
            correct += 1
 
    return correct / total if total > 0 else 0.0


def evaluate_all_timesteps(
    base_path: str, max_timestep: int = 15, threshold: float = 0.8
) -> Dict[str, List[Tuple[int, float]]]:
    """모든 timestep 에 대해 세 벤치마크 평가."""
    benchmarks = {
        "MMBench": "seen_task_safe_PO_0_MMBench_v1.0_eval.txt",
        "ScienceQA": "seen_task_safe_PO_0_ScienceQA_TEST_eval.txt",
        "SEEDBench": "seen_task_safe_PO_0_SEEDBench_IMG_eval.txt",
    }

    results: Dict[str, List[Tuple[int, float]]] = {b: [] for b in benchmarks}

    for timestep in range(max_timestep + 1):
        timestep_dir = os.path.join(base_path, str(timestep))

        for bench_name, filename in benchmarks.items():
            file_path = os.path.join(timestep_dir, filename)
            acc = compute_accuracy_from_txt(
                file_path, limit_lines=1000
            )

            if acc is not None:
                results[bench_name].append((timestep, acc))
                print(f"[METRIC] Timestep {timestep} - {bench_name}: {acc:.4f}")
            else:
                print(f"[METRIC] Timestep {timestep} - {bench_name}: File not found")

    return results


def _ratio(acc: float, zeroshot_pct: float) -> float:
    """Zeroshot 대비 비율. acc(0~1) 를 %로 바꾼 뒤 zeroshot(%) 로 나눠 ×100.

    동등 성능이면 100, zeroshot 의 2배면 200 이 된다.
    """
    acc_pct = acc * 100.0
    return acc_pct / zeroshot_pct * 100.0


def save_benchmark_results(
    results: Dict[str, List[Tuple[int, float]]],
    zeroshot_scores: Dict[str, float],
    save_path: str,
) -> None:
    """벤치마크 결과를 txt 파일로 저장 (Zeroshot 대비 비율)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("VLM Benchmark Results (Ours/Zeroshot x 100)\n")
        f.write("=" * 80 + "\n\n")

        for bench_name, timestep_results in results.items():
            if bench_name not in zeroshot_scores or not timestep_results:
                continue

            zeroshot = zeroshot_scores[bench_name]
            f.write(f"\n{bench_name} (Zeroshot: {zeroshot:.2f})\n")
            f.write("-" * 80 + "\n")

            # Timestep 별 값
            f.write("Timestep-by-Timestep Values:\n")
            for timestep, acc in timestep_results:
                f.write(f"  Timestep {timestep}: {_ratio(acc, zeroshot):.2f}\n")
            f.write("\n")

            # Avg (timestep 1~max)
            avg_values = [acc for t, acc in timestep_results if t > 0]
            if avg_values:
                avg_acc = sum(avg_values) / len(avg_values)
                f.write(f"Avg (timesteps 1~max): {_ratio(avg_acc, zeroshot):.2f}\n")

            # Last (마지막으로 '존재한' timestep 기준)
            last_acc = timestep_results[-1][1]
            f.write(f"Last (final timestep): {_ratio(last_acc, zeroshot):.2f}\n")
            f.write("\n")

    print(f"[METRIC] Results saved to: {save_path}")


def save_summary_table(
    results: Dict[str, List[Tuple[int, float]]],
    zeroshot_scores: Dict[str, float],
    save_path: str,
) -> None:
    """요약 테이블 저장."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("VLM Benchmark Summary (Ours/Zeroshot x 100)\n")
        f.write("=" * 80 + "\n\n")

        # ----- Avg Table -----
        f.write("Average Values (Avg of timesteps 1~max)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Benchmark':<20}{'Zeroshot':<15}{'Avg':<15}\n")
        f.write("-" * 80 + "\n")

        avg_ratios = []
        for bench_name, timestep_results in results.items():
            if bench_name not in zeroshot_scores or not timestep_results:
                continue
            zeroshot = zeroshot_scores[bench_name]
            avg_values = [acc for t, acc in timestep_results if t > 0]
            if avg_values:
                avg_acc = sum(avg_values) / len(avg_values)
                avg_ratio = _ratio(avg_acc, zeroshot)
                avg_ratios.append(avg_ratio)
                f.write(f"{bench_name:<20}{zeroshot:<15.2f}{avg_ratio:<15.2f}\n")

        if avg_ratios:
            overall_avg = sum(avg_ratios) / len(avg_ratios)
            f.write("-" * 80 + "\n")
            f.write(f"{'Mean':<20}{'-':<15}{overall_avg:<15.2f}\n")

        f.write("\n\n")

        # ----- Last Table -----
        f.write("Last Values (Final timestep)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Benchmark':<20}{'Zeroshot':<15}{'Last':<15}\n")
        f.write("-" * 80 + "\n")

        last_ratios = []
        for bench_name, timestep_results in results.items():
            if bench_name not in zeroshot_scores or not timestep_results:
                continue
            zeroshot = zeroshot_scores[bench_name]
            last_acc = timestep_results[-1][1]
            last_ratio = _ratio(last_acc, zeroshot)
            last_ratios.append(last_ratio)
            f.write(f"{bench_name:<20}{zeroshot:<15.2f}{last_ratio:<15.2f}\n")

        if last_ratios:
            overall_last = sum(last_ratios) / len(last_ratios)
            f.write("-" * 80 + "\n")
            f.write(f"{'Mean':<20}{'-':<15}{overall_last:<15.2f}\n")

        f.write("\n")

    print(f"[METRIC] Summary table saved to: {save_path}")


def compute_baseline_scores(
    results: Dict[str, List[Tuple[int, float]]]
) -> Dict[str, float]:
    """timestep 들의 평균 accuracy(%) 를 벤치마크별 baseline 으로 산출.

    Zero-shot 은 보통 한 timestep(마지막)만 존재하므로 그 값이 그대로 baseline 이 된다.
    """
    scores: Dict[str, float] = {}
    for bench, ts in results.items():
        accs = [acc for _, acc in ts]
        if accs:
            scores[bench] = sum(accs) / len(accs) * 100.0  # 0~1 -> %
    return scores


def main(
    base_path: str,
    max_timestep: int = 15,
    output_dir: str = "metric_lvlm_benchmark",
    threshold: float = 0.8,
    zero_shot: bool = False,
    zeroshot_json: Optional[str] = None,
) -> None:
    """LVLM 벤치마크 채점.

    워크플로우:
      1) Zero-shot 먼저 (--zero_shot):
           base_path = ./results/Zeroshot_VLM_Bench 로 돌려 baseline accuracy(%)
           를 계산하고 zeroshot_json 에 저장한다. (이후 Ours 의 분모로 사용)
      2) Ours 나중에:
           base_path = ./results/Ours_<n>_concepts_VLM_Bench 로 돌리면 저장된
           zeroshot_json 을 불러와 Zeroshot 대비 비율(=100 이면 동등)을 출력한다.
    """
    os.makedirs(output_dir, exist_ok=True)
    if zeroshot_json is None:
        zeroshot_json = os.path.join(output_dir, "zeroshot_scores.json")

    results = evaluate_all_timesteps(base_path, max_timestep, threshold=threshold)

    if zero_shot:
        # 이 실행이 baseline 을 '정의'한다: 원시 accuracy(%) 를 저장.
        zeroshot_scores = compute_baseline_scores(results)
        with open(zeroshot_json, "w", encoding="utf-8") as f:
            json.dump(zeroshot_scores, f, indent=2)

        print("\n[METRIC] Zero-shot baseline accuracy (%)")
        for bench, sc in zeroshot_scores.items():
            print(f"[METRIC]   {bench}: {sc:.2f}")
        print(f"[METRIC] Saved zero-shot baseline to: {zeroshot_json}")
        return

    # Ours 실행: 저장된 zero-shot baseline 을 분모로 사용.
    if not os.path.exists(zeroshot_json):
        raise FileNotFoundError(
            f"Zero-shot baseline not found at '{zeroshot_json}'. "
            f"Run with --zero_shot on the Zeroshot results first."
        )
    with open(zeroshot_json, "r", encoding="utf-8") as f:
        zeroshot_scores = json.load(f)

    save_benchmark_results(
        results=results,
        zeroshot_scores=zeroshot_scores,
        save_path=f"{output_dir}/vlm_benchmark_detailed.txt",
    )
    save_summary_table(
        results=results,
        zeroshot_scores=zeroshot_scores,
        save_path=f"{output_dir}/vlm_benchmark_summary.txt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score LVLM benchmark accuracy.")
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="results dir, e.g. ./results/Zeroshot_VLM_Bench or "
             "./results/CORE_VLM_Bench",
    )
    parser.add_argument("--max_timestep", type=int, default=15)
    parser.add_argument("--output_dir", type=str, default="metric_lvlm_benchmark")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument(
        "--zero_shot",
        action="store_true",
        help="this run is the zero-shot baseline: save its accuracy as the denominator",
    )
    parser.add_argument(
        "--zeroshot_json",
        type=str,
        default=None,
        help="path to zero-shot baseline json "
             "(default: <output_dir>/zeroshot_scores.json)",
    )
    args = parser.parse_args()

    main(
        base_path=args.base_path,
        max_timestep=args.max_timestep,
        output_dir=args.output_dir,
        threshold=args.threshold,
        zero_shot=args.zero_shot,
        zeroshot_json=args.zeroshot_json,
    )