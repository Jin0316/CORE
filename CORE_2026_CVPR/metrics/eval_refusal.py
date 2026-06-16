"""
Refusal Classification Results Analysis
저장된 refusal evaluation JSON 파일들을 읽어서 avg와 last 메트릭을 계산하고 텍스트 파일로 저장
"""
import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List


def load_refusal_json(file_path: str) -> Dict:
    """Load refusal evaluation JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[METRIC] Error loading {file_path}: {e}")
        return None


def extract_summary_metrics(json_data: Dict) -> Dict:
    """Extract metrics from summary section"""
    if not json_data or "summary" not in json_data:
        return None
    
    summary = json_data["summary"]
    
    return {
        'context_aware_accuracy': summary.get('context_aware_accuracy', 0.0),
        'refusal_pattern_rate': summary.get('refusal_pattern_rate', 0.0),
        'total_responses': summary.get('total_responses', 0),
        'context_aware_correct': summary.get('total_context_aware_correct', 0),
        'refusal_pattern_count': summary.get('total_refusal_patterns', 0)
    }


def load_multiple_timesteps(method_configs: Dict[str, Dict], max_timestep: int) -> Dict[str, pd.DataFrame]:
    """Load multiple timesteps for multiple methods"""
    all_methods_data = {}
    
    for method_name, config in method_configs.items():
        timestep_data = []
        
        for timestep in range(max_timestep + 1):
            file_path = os.path.join(config["base_dir"], f"refusal_evaluation_timestep_{timestep}.json")
            
            if os.path.exists(file_path):
                json_data = load_refusal_json(file_path)
                metrics = extract_summary_metrics(json_data)
                
                if metrics:
                    metrics['timestep'] = timestep
                    timestep_data.append(metrics)
        
        if timestep_data:
            all_methods_data[method_name] = pd.DataFrame(timestep_data).sort_values('timestep')
            print(f"[METRIC] Loaded {len(timestep_data)} timesteps for {method_name}")
        else:
            print(f"[METRIC] No data found for {method_name}")
    
    return all_methods_data


def save_metrics_to_text(
    all_methods_data: Dict[str, pd.DataFrame],
    mode: str,
    save_path: str
):
    """Save all metrics values to text file in list format (values multiplied by 100)"""
    metrics = ['context_aware_accuracy', 'refusal_pattern_rate']
    metric_names = ['Context-Aware Accuracy', 'Refusal Pattern Rate']
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"Refusal Metrics Summary - {mode} Mode (values × 100)\n")
        f.write("=" * 80 + "\n\n")
        
        for metric, metric_name in zip(metrics, metric_names):
            f.write(f"\n{metric_name}\n")
            f.write("-" * 80 + "\n")
            
            # Method별 list 형태로 저장
            f.write("Timestep-by-Timestep Values (as list):\n")
            for method_name, df in all_methods_data.items():
                if metric in df.columns:
                    values = [row[metric] * 100 for _, row in df.iterrows()]
                    timesteps = [int(row['timestep']) for _, row in df.iterrows()]
                    f.write(f"  {method_name}:\n")
                    f.write(f"    Timesteps: {timesteps}\n")
                    f.write(f"    Values: {[f'{v:.2f}' for v in values]}\n")
            
            f.write("\n")
            
            # Avg (average of timesteps 1 to max, excluding 0)
            f.write("Avg (timesteps 1~max):\n")
            for method_name, df in all_methods_data.items():
                if metric in df.columns and len(df) > 1:
                    values_without_0 = df[df['timestep'] > 0][metric]
                    if len(values_without_0) > 0:
                        avg_value = values_without_0.mean() * 100
                        f.write(f"  {method_name}: {avg_value:.2f}\n")
            
            f.write("\n")
            
            # Last timestep value
            f.write("Last (final timestep):\n")
            for method_name, df in all_methods_data.items():
                if metric in df.columns and len(df) > 0:
                    last_value = df.iloc[-1][metric] * 100
                    last_timestep = int(df.iloc[-1]['timestep'])
                    f.write(f"  {method_name} (timestep {last_timestep}): {last_value:.2f}\n")
            
            f.write("\n")
    
    print(f"[METRIC] Metrics saved to: {save_path}")


def save_summary_table(
    all_methods_data: Dict[str, pd.DataFrame],
    mode: str,
    save_path: str
):
    """Save summary table with separate Avg and Last tables (values multiplied by 100)"""
    metrics = ['context_aware_accuracy', 'refusal_pattern_rate']
    metric_names = ['Context-Aware Acc', 'Refusal Rate']
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"Refusal Summary Table - {mode} Mode (values × 100)\n")
        f.write("=" * 70 + "\n\n")
        
        # Avg Table
        f.write("Average Values (Avg of timesteps 1~max)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Method':<20}")
        for metric_name in metric_names:
            f.write(f"{metric_name:<25}")
        f.write("\n")
        f.write("-" * 70 + "\n")
        
        for method_name, df in all_methods_data.items():
            f.write(f"{method_name:<20}")
            
            for metric in metrics:
                if metric in df.columns:
                    values = df[df['timestep'] > 0][metric]
                    avg_value = (values.mean() * 100) if len(values) > 0 else 0
                else:
                    f.write(f"{'N/A':<25}")
                    continue
                f.write(f"{avg_value:<25.2f}")
            
            f.write("\n")
        
        f.write("\n\n")
        
        # Last Table
        f.write("Last Values (Final timestep)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Method':<20}")
        for metric_name in metric_names:
            f.write(f"{metric_name:<25}")
        f.write("\n")
        f.write("-" * 70 + "\n")
        
        for method_name, df in all_methods_data.items():
            f.write(f"{method_name:<20}")
            
            for metric in metrics:
                if metric in df.columns and len(df) > 0:
                    last_value = df.iloc[-1][metric] * 100
                else:
                    f.write(f"{'N/A':<25}")
                    continue
                f.write(f"{last_value:<25.2f}")
            
            f.write("\n")
        
        f.write("\n")
    
    print(f"[METRIC] Summary table saved to: {save_path}")


def average_modes_data(all_modes_data: Dict[str, Dict[str, pd.DataFrame]], modes: List[str]) -> Dict[str, pd.DataFrame]:
    """Average data across multiple modes"""
    averaged_data = {}
    
    # Get all methods from first mode
    first_mode_data = next(iter(all_modes_data.values()))
    methods = list(first_mode_data.keys())
    
    for method in methods:
        all_timesteps = []
        
        for mode in modes:
            if mode in all_modes_data and method in all_modes_data[mode]:
                all_timesteps.append(all_modes_data[mode][method])
        
        if all_timesteps:
            combined_df = pd.concat(all_timesteps, ignore_index=True)
            averaged_df = combined_df.groupby('timestep').mean().reset_index()
            averaged_data[method] = averaged_df
    
    return averaged_data


def main(
    base_path: str,
    mode: str = "HH",
    max_timestep: int = 15,
    output_dir: str = "metric_forget_summary"
):
    """
    Main function - analyzes refusal classification results and saves text files
    
    Args:
        base_path: Base directory containing metric_forget_per_timestep folders
        mode: Mode to process - single ('HH') or multiple ('HH/HN', etc.)
        max_timestep: Maximum timestep (0 to max_timestep)
        output_dir: Output directory for text files
    """
    
    def create_method_configs(mode_str):
        """Create method configurations for loading data"""
        return {
            "Zeroshot": {"base_dir": f"{base_path}/Zeroshot_{mode_str}"},
            "CORE": {"base_dir": f"{base_path}/CORE_{mode_str}"},
        }
    
    # Check if mode contains multiple modes to average
    if '/' in mode:
        modes_to_average = mode.split('/')
        print(f"[METRIC] Averaging modes: {modes_to_average}")
        
        # Load data for all modes
        all_modes_data = {}
        for single_mode in modes_to_average:
            mode_configs = create_method_configs(single_mode)
            mode_data = load_multiple_timesteps(mode_configs, max_timestep)
            if mode_data:
                all_modes_data[single_mode] = mode_data
        
        if all_modes_data:
            # Average across modes
            averaged_data = average_modes_data(all_modes_data, modes_to_average)
            
            if averaged_data:
                averaged_mode_name = "_".join(modes_to_average)
                
                # Save text files
                save_metrics_to_text(
                    all_methods_data=averaged_data,
                    mode=f"Avg({mode})",
                    save_path=f"{output_dir}/refusal_metrics_avg_{averaged_mode_name}_detailed.txt"
                )
                save_summary_table(
                    all_methods_data=averaged_data,
                    mode=f"Avg({mode})",
                    save_path=f"{output_dir}/refusal_metrics_avg_{averaged_mode_name}_summary.txt"
                )
            else:
                print(f"[METRIC] No valid averaged data for modes {modes_to_average}")
        else:
            print(f"[METRIC] No valid data loaded for modes {modes_to_average}")
    
    else:
        # Single mode
        mode_configs = create_method_configs(mode)
        all_methods_data = load_multiple_timesteps(mode_configs, max_timestep)
        
        if all_methods_data:
            # Save text files
            save_metrics_to_text(
                all_methods_data=all_methods_data,
                mode=mode,
                save_path=f"{output_dir}/refusal_metrics_{mode}_detailed.txt"
            )
            save_summary_table(
                all_methods_data=all_methods_data,
                mode=mode,
                save_path=f"{output_dir}/refusal_metrics_{mode}_summary.txt"
            )
        else:
            print(f"[METRIC] No valid data loaded for mode {mode}")


if __name__ == "__main__":
    # 사용 예시
    base_path = "./metric_forget_per_timestep"
    main(base_path, mode="HH", max_timestep=15)
    