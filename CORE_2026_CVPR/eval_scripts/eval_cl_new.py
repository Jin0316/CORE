import argparse
import subprocess
import os
import shlex
from find_last_ckpt import task_info, find_latest_checkpoint, ensure_dir_exists, keyword_list, imagenet_r_keywords


def evaluate_tasks(mode, start_time_step, end_time_step, eval_device, text_dir_template,
                   harm_combination, n_con, run_name='run_name_not_specified', zero_shot=False):

    # zero-shot baseline: always load the pretrained LVLM, no CBL/router
    init_ckpt_path = '/workspace/pretrained/pretrained_minigpt4_7b_Vicuna.pth'

    print('[EVAL] Evaluating tasks start...')
    print(f'[EVAL] Evaluation process for time steps {start_time_step} to {end_time_step}: [MODE]: {mode}')
    print(f'[EVAL] Task info: {task_info}')

    assert harm_combination in ['hh' ,'hu', 'uh', 'uu']

    script_name = {
        'safe_PO': 'eval_scripts/backends/batch_eval_safe_erase.py',
        'safe_PO_IN': 'eval_scripts/backends/batch_eval_safe_erase_IN.py',
    }
    
    current_work_folder = os.getcwd()
    default_ckpt_path = f'{current_work_folder}/minigpt4/output'
    default_cfg_path = f'{current_work_folder}/eval_configs'
    # CBL checkpoints live under the job's output dir (minigpt4/output/<job>/cbl).
    default_cbl_ckpt_path = os.path.join(default_ckpt_path, run_name, 'cbl')
    text_dir_template = f'{current_work_folder}/{text_dir_template}'
    
    print(f'[EVAL] default_cfg_path: {default_cfg_path}')
    print(f'[EVAL] text_dir_template: {text_dir_template}')

    print(f'[EVAL] default_ckpt_path: {default_ckpt_path}')
    print(f'[EVAL]  default_cbl_path: {default_cbl_ckpt_path}')

    cfg_path = {
        'safe_PO': os.path.join(default_cfg_path, 'minigpt4_eval_safe_erase.yaml'),
        'safe_PO_IN': os.path.join(default_cfg_path, 'minigpt4_eval_safe_erase_IN.yaml'),
    }
    success_fail = []

    # First safe_PO_IN task; the only one kept in uh/uu modes (see task_range below).
    first_in_step = min(i for i in range(len(task_info)) if task_info[str(i)][0] == 'safe_PO_IN')

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

 
        task_range = range(time_step + 1) if mode == 'seen' else range(time_step + 1, len(task_info))

        # uh/uu modes use harm_image=False, so every safe_PO_IN task builds the identical
        # eval set (ImageNetR_EVAL ignores per-subset keywords). Keep only the first one.
        # Mirrored in metrics/calculate_metric_bertscore.py.
        if harm_combination in ('uh', 'uu'):
            task_range = [s for s in task_range
                          if task_info[str(s)][0] != 'safe_PO_IN' or s == first_in_step]

        for step in task_range:
            if mode == 'seen':
                print('[EVAL] Eval mode: seen')
                print(f'[EVAL] Current task / Seen tasks [{step +1}/{len(task_range)}]')
            else: 
                print('[EVAL] Eval mode: transfer')
                print(f'[EVAL] Current task / Seen tasks [{step +1}/{time_step + 1}]')

            task = task_info[str(step)][0]
            subset_idx = task_info[str(step)][1]
            if task == 'safe_PO':
                keywords = keyword_list[int(subset_idx)]
            elif task == 'safe_PO_IN':
                keywords = imagenet_r_keywords[int(subset_idx)]
            keywords = ",".join(keywords)
            print(f'[EVAL] Eval Keywords: {keywords}')

            text_path = os.path.join(text_dir_timestep, f'{mode}_task_{task}_{subset_idx}_eval.txt')

            command = [
                f"CUDA_VISIBLE_DEVICES={str(eval_device)}",
                "python", script_name[task],
                "--cfg-path", cfg_path[task],

                # See the task-info 
                "--task-info", 'else', # Not used in the main code 
                "--subset-index", str(subset_idx), # Not used in the main code 
                "--gpu-id", str(0),
                "--txt-path", text_path,
                "--ckpt-path", ckpt_path,

                "--harm-combination", harm_combination,
                "--keywords", shlex.quote(keywords),
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
                success_fail.append([time_step, str(task), str(subset_idx)])

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
    
    parser.add_argument("--harm_combination", required=True, help="hh/hu/uh")
    parser.add_argument("--zero_shot", action="store_true", help="zero-shot: pretrained LVLM only, no CBL/router")

    args = parser.parse_args()

    evaluate_tasks(
        mode=args.mode,
        start_time_step=args.start_time_step,
        end_time_step=args.end_time_step,
        eval_device=args.eval_device,
        text_dir_template=args.text_dir_template,

        harm_combination = args.harm_combination,

        run_name=args.run_name,
        n_con=args.n_con,
        zero_shot=args.zero_shot
    )