import argparse
import subprocess
import os
from find_last_ckpt import task_info, find_latest_checkpoint, ensure_dir_exists

def evaluate_tasks(mode, start_time_step, end_time_step, eval_device, text_dir_template, run_name='cls_cap_vqa', method = None, benchmark = None, n_con = 20, zero_shot = False):
    init_ckpt_path = '/workspace/pretrained/pretrained_minigpt4_7b_Vicuna.pth'
    print('[EVAL] Evaluating tasks start...')
    print(f'[EVAL] Evaluation process for time steps {start_time_step} to {end_time_step}: [MODE]: {mode}')
    print(f'[EVAL] METHOD: {method}')

    assert benchmark in ["MMBench_v1.0", "ScienceQA_TEST", "SEEDBench_IMG"]

    script_name = {"MMBench_v1.0": 'eval_scripts/backends/batch_eval_VLM_Bench.py',
                   "ScienceQA_TEST": 'eval_scripts/backends/batch_eval_VLM_Bench.py',
                   "SEEDBench_IMG": 'eval_scripts/backends/batch_eval_VLM_Bench.py',}
    
    current_work_folder = os.getcwd()
    default_ckpt_path = f'{current_work_folder}/minigpt4/output'
    default_cfg_path = f'{current_work_folder}/eval_configs'
    # CBL checkpoints live under the job's output dir (minigpt4/output/<job>/cbl).
    default_cbl_ckpt_path = os.path.join(default_ckpt_path, run_name, 'cbl')
    text_dir_template = f'{current_work_folder}/{text_dir_template}'
    
    print(f'[EVAL] default_ckpt_path: {default_ckpt_path}')
    print(f'[EVAL] default_cfg_path: {default_cfg_path}')
    print(f'[EVAL] text_dir_template: {text_dir_template}')
    
    cfg_path = {
        'MMBench_v1.0':   os.path.join(default_cfg_path, 'minigpt4_eval_MMBench.yaml'), 
        'ScienceQA_TEST': os.path.join(default_cfg_path, 'minigpt4_eval_ScienceQA_TEST.yaml'), 
        'SEEDBench_IMG':  os.path.join(default_cfg_path, 'minigpt4_eval_SEEDBench_IMG.yaml'), 
    }
    
    # run_name = f'{run_name}_{method}'
    success_fail = []
    for time_step in range(start_time_step, end_time_step + 1):
        if zero_shot:
            ckpt_path = init_ckpt_path
            cbl_ckpt_path = None
        else:
            ckpt_dir = os.path.join(default_ckpt_path, run_name, str(time_step))
            ckpt_path = find_latest_checkpoint(ckpt_dir)
            cbl_ckpt_path = os.path.join(default_cbl_ckpt_path, f'cbl_{time_step}.pt')
        print(f'[EVAL] MAIN MODEL load path : {str(ckpt_path)}')
        print(f'[EVAL]  CBL MODEL load path : {str(cbl_ckpt_path)}')

        text_dir_timestep = text_dir_template.format(time_step)
        ensure_dir_exists(text_dir_timestep)


        print(f'[EVAL] Evaluating Standard VLM Benchmark!')
        print(f'[EVAL] Current_time_step: {time_step + 1} / Total time step {len(task_info)}')

        step = 0
        task = task_info[str(step)][0]
        subset_idx = task_info[str(step)][1]

        text_path = os.path.join(text_dir_timestep, f'{mode}_task_{task}_{subset_idx}_{benchmark}_eval.txt')

        command = [
            f"CUDA_VISIBLE_DEVICES={str(eval_device)}",
            "python", script_name[benchmark],
            "--cfg-path", cfg_path[benchmark],
            # See the task-info 
            # "--task-info",  str(method),
            "--task-info", 'else', 
            "--subset-index", str(subset_idx),
            "--gpu-id", str(0),
            "--txt-path", text_path,
            "--ckpt-path", ckpt_path,

            "--benchmark", benchmark,
        ]
        if zero_shot:
            command.append("--zero-shot")
        else:
            command += ["--cbl-ckpt-path", cbl_ckpt_path]

        print(f"[EVAL] Running command: {' '.join(command)}")

        try:
            subprocess.run(' '.join(command), shell=True, check=True)
            print(f"[EVAL] Task {time_step} evaluated successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[EVAL] Error during task {time_step}: {e}")
            success_fail.append([time_step, str(task), str(subset_idx)]) # zero indexting 

    if len(success_fail) == 0:
        print('[EVAL] All tasks evaluated successfully!...')
        print('[EVAL] Finish eval process...')
    else:
        print('[EVAL] Some tasks are not evaluated properly...')
        print(f'[EVAL] Failed task info: {success_fail}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate tasks across multiple time steps.")
    parser.add_argument("--mode", type=str, choices=["seen", "transfer"], required=True, help="Evaluation mode: 'seen' or 'transfer'")
    parser.add_argument("--start_time_step", type=int, required=True, help="Starting time step for evaluation")
    parser.add_argument("--end_time_step", type=int, required=True, help="Ending time step for evaluation")
    parser.add_argument("--eval_device", type=int, required=True, help="Device ID for evaluation")
    parser.add_argument("--text_dir_template", type=str, required=True, help="Template for text directory path")
    parser.add_argument("--run_name", type=str, default="cls_cap_vqa", help="run name = output subdir under minigpt4/output/ to evaluate")
    parser.add_argument("--n_con", type=int, default=20, help="Job name for evaluation")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--zero_shot", action="store_true", help="zero-shot: pretrained LVLM only, no CBL/router")

    args = parser.parse_args()

    evaluate_tasks(
        mode=args.mode,
        start_time_step=args.start_time_step,
        end_time_step=args.end_time_step,
        eval_device=args.eval_device,
        text_dir_template=args.text_dir_template,
        run_name=args.run_name,
        # method=args.method,
        benchmark=args.benchmark,
        n_con=args.n_con,
        zero_shot=args.zero_shot
    )

