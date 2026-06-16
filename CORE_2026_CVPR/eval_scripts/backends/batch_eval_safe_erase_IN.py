import argparse
import os
import random
import json

import numpy as np
import torch
import torch.backends.cudnn as cudnn
# import gradio as gr

import sys
# this backend lives in eval_scripts/backends/; put repo root on sys.path so
# `minigpt4` / `clip_base` resolve when launched as `python eval_scripts/backends/...`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.common.eval_utils import prepare_texts

from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2
from minigpt4.datasets.datasets.coco_vqa_datasets import ImageNetR_EVAL

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from torch.utils.data import DataLoader
import clip
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--task-info", required=True, help = "cls, vqa, cap")

    parser.add_argument('--cbl-ckpt-path', required=False, default=None)
    parser.add_argument('--zero-shot', action='store_true', help="zero-shot: pretrained LVLM only, no CBL/router")
    parser.add_argument("--harm-combination", required=True, help="hh/hu/uh")
    parser.add_argument('--keywords', required=True)

    parser.add_argument("--subset-index", type=int, default=0, help="which task you running")
    parser.add_argument("--ckpt-path", type=str, default='bad_path', help="specify the path of ckpt for this task.")
    parser.add_argument("--txt-path", type=str, default='bad_path', help="specify the path of result of this task.")

    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================


conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
             'pretrain_llama2': CONV_VISION_LLama2}

print('[EVAL] Initializing Chat')
args = parse_args()
cfg = Config(args)
setup_seeds(cfg)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_config.ckpt = args.ckpt_path
model_cls = registry.get_model_class(model_config.arch)
cbl_ckpt_path = None if args.zero_shot else args.cbl_ckpt_path
model = model_cls.from_config(model_config, args.ckpt_path, cbl_ckpt_path).to('cuda:{}'.format(args.gpu_id))
model = model.to('cuda:0')
if not args.zero_shot:
    from minigpt4.models.mini_gpt4 import create_external_cbl
    model.external_cbl = create_external_cbl(f'cuda:{args.gpu_id}')
    model.cbl_model = model.cbl_model.to('cuda:0')

CONV_VISION = conv_dict[model_config.model_type]

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), task_id=args.subset_index, task_info=args.task_info)

cfg_o = cfg.get_o_config()
_, transforms = clip.load("ViT-B/16", device='cuda:{}'.format(args.gpu_id))

eval_path  = cfg_o.annotation_path
image_path = cfg_o.image_path
keywords = args.keywords

print(f'[EVAL] eval ann path : {eval_path}')
with open(eval_path) as f:
    eval_path = json.load(f)
print('[EVAL] Harm Combination')

if args.harm_combination == 'hh': 
    print('[EVAL] Harm Image / Harm Text')
    data = ImageNetR_EVAL(vis_processor, image_path, eval_path, keywords=keywords, harm_image=True, harm_text=True)
elif args.harm_combination == 'hu':
    print('[EVAL] Harm Image / Unharm Text')
    data = ImageNetR_EVAL(vis_processor, image_path, eval_path, keywords=keywords, harm_image=True, harm_text=False)
elif args.harm_combination == 'uh':
    print('[EVAL] Unharm Image / Harm Text')
    data = ImageNetR_EVAL(vis_processor, image_path, eval_path, keywords=None, harm_image=False, harm_text=True)
elif args.harm_combination == 'uu':
    print('[EVAL] Unharm Image / Unharm Text')
    data = ImageNetR_EVAL(vis_processor, image_path, eval_path, keywords=None, harm_image=False, harm_text=False)

eval_dataloader = DataLoader(data, batch_size=20, shuffle=False)
minigpt4_predict = []

CONV = CONV_VISION.copy()
with open(args.txt_path, 'w') as f:
    for batch in tqdm(eval_dataloader):
        images = batch['image']
        image_ids = batch['image_id'] 
        questions = batch['question']
        instruction_inputs = batch['instruction_input']
        categories = batch['category']
        keywords = batch['keyword']

        texts = prepare_texts(questions, CONV)

        llm_messages = model.generate(images, texts, task_info = args.task_info, max_new_tokens = 100, routing = not args.zero_shot)
        for image_id, question, category, keyword, llm_message in zip(image_ids, questions, categories, keywords, llm_messages):
            llm_message = llm_message.replace('\n', '')
            str1 = f'[Category] {category} [Keyword] {keyword} [image id] {image_id} [question] {question}' + '\n'
            str2 = llm_message + '\n'
            f.write(str1)
            f.write(str2)