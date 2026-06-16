import json
import os
import numpy as np
import pandas as pd
from typing import Dict

from metric_utils import calculate_average_metrics_from_json


def load_multiple_timesteps(method_configs: Dict[str, Dict], max_timestep: int) -> Dict[str, pd.DataFrame]:
    """Load multiple timesteps for multiple methods"""
    all_methods_data = {}
    
    for method_name, config in method_configs.items():
        timestep_data = []
        
        for timestep in range(max_timestep + 1):
            file_path = os.path.join(config["base_dir"], f"evaluation_timestep_{timestep}.json")
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    avg_metrics = calculate_average_metrics_from_json(json_data)
                    if avg_metrics:
                        avg_metrics['timestep'] = timestep
                        timestep_data.append(avg_metrics)
                        
                except Exception as e:
                    print(f"[METRIC] Error loading {method_name} timestep {timestep}: {e}")
        
        if timestep_data:
            all_methods_data[method_name] = pd.DataFrame(timestep_data).sort_values('timestep')
            print(f"[METRIC] Loaded {len(timestep_data)} timesteps for {method_name}")
    
    return all_methods_data


def save_metrics_to_text(
    all_methods_data: Dict[str, pd.DataFrame],
    mode: str,
    save_path: str
):
    """Save all metrics values to text file (values multiplied by 100)"""
    metrics = ['bert_f1', 'rouge_f1', 'clip_score', 'refusal_rate']
    metric_names = ['BERT F1', 'ROUGE-L F1', 'CLIP Score', 'Refusal Rate']
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"Metrics Summary - {mode} Mode (values × 100)\n")
        f.write("=" * 80 + "\n\n")
        
        for metric, metric_name in zip(metrics, metric_names):
            f.write(f"\n{metric_name}\n")
            f.write("-" * 80 + "\n")
            
            # Timestep-by-timestep values
            f.write("Timestep-by-Timestep Values:\n")
            for method_name, df in all_methods_data.items():
                if metric in df.columns:
                    f.write(f"  {method_name}:\n")
                    for _, row in df.iterrows():
                        timestep = int(row['timestep'])
                        value = row[metric] * 100
                        f.write(f"    Timestep {timestep}: {value:.2f}\n")
            
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
                    f.write(f"  {method_name}: {last_value:.2f}\n")
            
            f.write("\n")
    
    print(f"[METRIC] Metrics saved to: {save_path}")


def save_summary_table(
    all_methods_data: Dict[str, pd.DataFrame],
    mode: str,
    save_path: str
):
    """Save summary table with separate Avg and Last tables (values multiplied by 100)"""
    metrics = ['refusal_rate', 'answer_rate', 'bert_f1', 'clip_score', 'rouge_f1']
    metric_names = ['Refusal', 'Answer', 'BERT-F1', 'CLIP', 'ROUGE-F1']
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"Summary Table - {mode} Mode (values × 100)\n")
        f.write("=" * 90 + "\n\n")
        
        # Avg Table
        f.write("Average Values (Avg of timesteps 1~max)\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Method':<15}")
        for metric_name in metric_names:
            f.write(f"{metric_name:<15}")
        f.write("\n")
        f.write("-" * 90 + "\n")
        
        for method_name, df in all_methods_data.items():
            f.write(f"{method_name:<15}")
            
            for metric in metrics:
                if metric == 'answer_rate' and 'refusal_rate' in df.columns:
                    values = df[df['timestep'] > 0]['refusal_rate']
                    avg_value = (100 - values.mean() * 100) if len(values) > 0 else 0
                elif metric in df.columns:
                    values = df[df['timestep'] > 0][metric]
                    avg_value = (values.mean() * 100) if len(values) > 0 else 0
                else:
                    f.write(f"{'N/A':<15}")
                    continue
                f.write(f"{avg_value:<15.2f}")
            
            f.write("\n")
        
        f.write("\n\n")
        
        # Last Table
        f.write("Last Values (Final timestep)\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Method':<15}")
        for metric_name in metric_names:
            f.write(f"{metric_name:<15}")
        f.write("\n")
        f.write("-" * 90 + "\n")
        
        for method_name, df in all_methods_data.items():
            f.write(f"{method_name:<15}")
            
            for metric in metrics:
                if metric == 'answer_rate' and 'refusal_rate' in df.columns and len(df) > 0:
                    last_value = 100 - df.iloc[-1]['refusal_rate'] * 100
                elif metric in df.columns and len(df) > 0:
                    last_value = df.iloc[-1][metric] * 100
                else:
                    f.write(f"{'N/A':<15}")
                    continue
                f.write(f"{last_value:<15.2f}")
            
            f.write("\n")
        
        f.write("\n")
    
    print(f"[METRIC] Summary table saved to: {save_path}")


def average_modes_data(all_modes_data: Dict[str, Dict[str, pd.DataFrame]], modes: list) -> Dict[str, pd.DataFrame]:
    """Average data across multiple modes"""
    averaged_data = {}
    
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
    mode: str = "UH",
    max_timestep: int = 15,
    output_dir: str = "metric_overall_summary"
):
    """
    Main function - only saves text files, no plotting
    
    Args:
        base_path: Base directory containing result folders
        mode: Mode to process - single ('HH', 'HU', 'UH', 'UU') or multiple ('UH/HU', etc.)
        max_timestep: Maximum timestep (0 to max_timestep)
        output_dir: Output directory for text files
    """
    
    def create_method_configs(mode_str):
        return {
            "CORE": {"base_dir": f"{base_path}/timestep_evaluation_results_CORE_{mode_str}"},
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
                    save_path=f"{output_dir}/metrics_avg_{averaged_mode_name}_detailed.txt"
                )
                save_summary_table(
                    all_methods_data=averaged_data,
                    mode=f"Avg({mode})",
                    save_path=f"{output_dir}/metrics_avg_{averaged_mode_name}_summary.txt"
                )
            else:
                print(f"[METRIC] No valid averaged data for modes {modes_to_average}")
        else:
            print(f"[METRIC] No valid data loaded for modes {modes_to_average}")
    
    else:
        # Single mode
        method_configs = create_method_configs(mode)
        all_methods_data = load_multiple_timesteps(method_configs, max_timestep)
        
        if all_methods_data:
            # Save text files
            save_metrics_to_text(
                all_methods_data=all_methods_data,
                mode=mode,
                save_path=f"{output_dir}/metrics_{mode}_detailed.txt"
            )
            save_summary_table(
                all_methods_data=all_methods_data,
                mode=mode,
                save_path=f"{output_dir}/metrics_{mode}_summary.txt"
            )
        else:
            print(f"[METRIC] No valid data loaded for mode {mode}")


if __name__ == "__main__":
    base_path = "./metric_retain_per_timestep"
    
    # Single mode
    main(base_path, mode="HH", max_timestep=15)
    
    # Average multiple modes
    main(base_path, mode="UH/HU/UU", max_timestep=15)