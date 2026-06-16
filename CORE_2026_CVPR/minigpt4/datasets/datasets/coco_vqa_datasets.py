"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import random
import torch

from PIL import Image

from minigpt4.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset
from minigpt4.datasets.datasets.prompts import refusal_prompts, classification_prompts, classification_refusals
from minigpt4.datasets.datasets.prompts import unharm_random_texts, remain_keywords
from collections import OrderedDict


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": "; ".join(ann["answer"]),
                "image": sample["image"],
            }
        )


class COCOVQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.instruction_pool =[
            "[vqa] {}",
            # "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]

        print(f'[DATA] visual root {vis_root} | annotation paths {ann_paths}')
        exist_annotation = []
        for ann in self.annotation:
            image_path = os.path.join(self.vis_root, ann["image"].split('/')[-1])
            if os.path.exists(image_path):
                exist_annotation.append(ann)
            # print(f'coco_vqa datasets.py | image path : {image_path}')
        self.annotation = exist_annotation


    def get_data(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann["image"].split('/')[-1])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        question_id = ann["types"]

        # print(f'coco vqa dataset.py : [image path] : {image_path}')
        # print(f'coco vqa dataset.py : [question] : {question} | [question id] : {question_id}')

        if type(ann["answer"]) == list: 
            answer_weight = {}
            for answer in ann["answer"]:
                if answer in answer_weight.keys():
                    answer_weight[answer] += 1 / len(ann["answer"])
                else:
                    answer_weight[answer] = 1 / len(ann["answer"])
            answers = list(answer_weight.keys())
            weights = list(answer_weight.values())
            answer = random.choices(answers, weights=weights, k=1)[0]  # random sample an answer according to weights
        
        else: 
            answer = ann["answer"]

        return {
            "image": image,
            "question": question,
            "question_id": question_id,
            "answer": answer,
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        instruction = random.choice(self.instruction_pool).format(data['question'])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": data['image'],
            "question_id": data["question_id"],
            "instruction_input": instruction,
            "answer": self.text_processor(data['answer']),
        }

class Flickr30K_EVAL_unique(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.root_path = root_path
        self.vis_processor = vis_processor
        
        # 이미지 ID를 기준으로 중복 제거
        unique_data = {}
        for item in loaded_data:
            img_id = item['image_id']
            if img_id not in unique_data:
                unique_data[img_id] = item
        
        # 중복 제거된 데이터 저장
        self.unique_data = list(unique_data.values())

    def __len__(self):
        return len(self.unique_data)
    
    def __getitem__(self, idx):
        data = self.unique_data[idx]
        img_id = data['image_id']
        image_path = os.path.join(self.root_path, data['file_name'])
        
        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        
        # 캡션 반환 (이 예제에서는 사용되지 않으므로 주석 처리)
        # answer = data['caption']
        
        return image, img_id

class Flickr30K_EVAL(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        img_id = data['image_id']
        image_path = os.path.join(self.root_path, data['file_name'])
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        answer = data['caption']
        return image, img_id, answer

class COCOVQA_EVAL_mine(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        img_id = data['image_id']
        question = data['question']
        question_id = data['types']
        image_path = os.path.join(self.root_path, data['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        answer = data['answer']
        question = f"[vqa] Based on the image, respond to this question with a short answer: {question}"
        return image, question, question_id, img_id, answer

class COCOVQA_EVAL_mine_reduce_ratio(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path, sampling_rate=0.1):
        self.root_path = root_path
        self.vis_processor = vis_processor

        # Randomly sample the data at the specified rate
        total_samples = len(loaded_data)
        sampled_indices = random.sample(range(total_samples), int(total_samples * sampling_rate))
        self.sampled_data = [loaded_data[i] for i in sampled_indices]

    def __len__(self):
        return len(self.sampled_data)
    
    def __getitem__(self, idx):
        data = self.sampled_data[idx]
        img_id = data['image_id']
        question = data['question']
        question_id = data['types']
        image_path = os.path.join(self.root_path, data['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        answer = data['answer']
        question = f"{question}"
        # question = f"[vqa] Based on the image, respond to this question with a short answer: {question}"
        return image, question, question_id, img_id, answer

class COCOVQAEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        
        self.instruction_pool = [
            '[vqa] Based on the image, respond to this question with a short answer: {}',
        ]
        self.vis_root = vis_root

        self.annotation = json.load(open(ann_paths[0]))

        answer_list_path = ann_paths[1]
        if os.path.exists(answer_list_path):
            self.answer_list = json.load(open(answer_list_path))
        else:
            self.answer_list = None

        try:
            self.coco_fmt_qust_file = ann_paths[2]
            self.coco_fmt_anno_file = ann_paths[3]
        except IndexError:
            self.coco_fmt_qust_file = None
            self.coco_fmt_anno_file = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        
        instruction = random.choice(self.instruction_pool).format(question)
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)
        
        return {
            "image": image,
            'image_path': image_path,
            "question": question,
            "question_id": ann["types"],
            "instruction_input": instruction,
            "instance_id": ann["image_id"],
        }
    

#####################################################
################### MLLM - Satety ###################
#####################################################

class PrivacyPreferenceOptimization(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, keywords):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.instruction_pool = ["{}",]

        print(f'[DATA] visual root {vis_root} | annotation paths {ann_paths}')
        print(f'[DATA] vis processor : {vis_processor}')
        print(f'[DATA] Requested Keywords : {keywords}')
        # Flatten the dataset to include all unsafe questions only
        self.flattened_annotation = []
        
        for i, ann in enumerate(self.annotation):
            image_path = os.path.join(self.vis_root, ann["image_id"])
            if ann['keyword'] in keywords:
                if os.path.exists(image_path):
                    # Add only unsafe_pairs questions for this image
                    for j, unsafe_pair in enumerate(ann["unsafe_pairs"]):
                        flattened_item = {
                            "image_id": ann["image_id"],
                            "category": ann["category"],
                            "keyword": ann["keyword"],
                            "question": unsafe_pair["question"],
                        }
                        self.flattened_annotation.append(flattened_item)
            else: 
                pass 
        
        print(f'[DATA] Original annotations: {len(self.annotation)}, Flattened annotations: {len(self.flattened_annotation)}')

    def get_data(self, index):
        ann = self.flattened_annotation[index]
        image_path = os.path.join(self.vis_root, ann["image_id"])
        category = ann["category"]
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        
        # Use the question from flattened annotation
        question = self.text_processor(ann['question'])

        # Random apology answers
        # apology_answers = [
        #     "I apologize, but I'm unable to respond to this question as it could lead to harmful information being shared.",
        #     "Regrettably, I can't provide an answer to this question due to the risk of disseminating harmful information.",
        #     "I regret to inform you that I cannot address this question because it might produce harmful content.",
        #     "I'm afraid I can't answer this inquiry since it may result in the release of harmful information.",
        #     "Unfortunately, I'm unable to respond to this question as it has the potential to generate harmful information."
        # ]
        apology_answers = refusal_prompts[category]
        answer = random.choice(apology_answers)

        return {
            "image": image,
            "question": question,
            "answer": answer,
            "image_id": ann['image_id'], 
            "category": ann['category'],
            "keyword": ann['keyword'],
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        instruction = random.choice(self.instruction_pool).format(data['question'])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": data['image'],
            "image_id": data['image_id'],
            "question": data['question'],
            "instruction_input": instruction,
            "answer": self.text_processor(data['answer']),
            "category": data["category"],
            "keyword": data["keyword"],
        }
    
    def __len__(self):
        return len(self.flattened_annotation)


# Version 3: Unsafe pairs dataset
class PrivacyGradientAscent(VQADataset, __DisplMixin):
    """
    Dataset class for harmful question-answer pairs.
    Given an image and its corresponding harmful question, 
    return the image, question, and the model's original harmful response.
    """
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.instruction_pool = [
            "{}",
            # "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]

        print(f'[DATA] visual root {vis_root} | annotation paths {ann_paths}')

        # Flatten only harmful questions from unsafe_pairs
        self.flattened_annotation = []
        for ann in self.annotation:
            if 'unsafe_pairs' in ann and len(ann['unsafe_pairs']) > 0:
                image_path = os.path.join(self.vis_root, ann["image_id"])
                if os.path.exists(image_path):
                    for unsafe_pair in ann['unsafe_pairs']:
                        flattened_item = {
                            "image_id": ann["image_id"],
                            "category": ann["category"],
                            "keyword": ann["keyword"],
                            "question": unsafe_pair["question"],
                            "answer": unsafe_pair["model_response"]  # harmful answer
                        }
                        self.flattened_annotation.append(flattened_item)

        print(f'[DATA] Original annotations: {len(self.annotation)}, '
              f'Flattened harmful pairs: {len(self.flattened_annotation)}')

    def get_data(self, index):
        ann = self.flattened_annotation[index]
        image_path = os.path.join(self.vis_root, ann["image_id"])

        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        question = self.text_processor(ann['question'])
        question_id = ann['category']
        answer = ann['answer']  # harmful model response

        return {
            "image": image,
            "question": question,
            "question_id": question_id,
            "answer": answer,
            "category": ann['category'],
            "keyword": ann['keyword']
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        instruction = random.choice(self.instruction_pool).format(data['question'])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": data['image'],
            "question_id": data["question_id"],
            "instruction_input": instruction,
            "answer": self.text_processor(data['answer']),
            "category": data["category"],
            "keyword": data["keyword"]
        }

    def __len__(self):
        return len(self.flattened_annotation)


"""
만들어아햘 데이터 로더 

1. Safe Erase 
    - harm image - unharm question 
    - unharm image - harm question 

2. Image classification 
    - harm image - unharm question 
    - unharm image - harm question 

3. OOD 
    - unharm image - unharm question 
"""



####################################
##### 1 Harm Image / Harm Text #####
####################################
class SafeErase_Harmimg_Harmtxt(torch.utils.data.Dataset):
    # image : Harm / question : Unharm 
    def __init__(self, vis_processor, vis_root, ann_paths, keywords):
        self.vis_processor = vis_processor
        self.instruction_pool = ["{}",]
        self.annotation = ann_paths
        self.vis_root = vis_root

        print(f'[DATA] visual root {vis_root}')
        print(f'[DATA] Requested Keywords : {keywords}')
        # Flatten the dataset to include all unsafe questions only
        self.flattened_annotation = []

        for i, ann in enumerate(self.annotation):
            image_path = os.path.join(self.vis_root, ann["image_id"])
            if ann['keyword'] in keywords:
                if os.path.exists(image_path):
                    for unsafe_question in ann["unsafe_pairs"]:
                        flattened_item = {
                            "image_id": ann["image_id"], 
                            "category": ann.get("category", ""),
                            "keyword": ann["keyword"],
                            "question": unsafe_question['question']
                        }
                        self.flattened_annotation.append(flattened_item)
        
        print(f'[DATA] The number of samples: {len(self.flattened_annotation)}')
        # Integrity probe
        if len(self.flattened_annotation) > 0:
            probe = self.flattened_annotation[0]
            
    def get_data(self, index):
        ann = self.flattened_annotation[index]
        image_path = os.path.join(self.vis_root, ann["image_id"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = ann['question']
        return {
            "image": image,
            "question": question,
            "image_id": ann['image_id'], 
            "category": ann['category'],
            "keyword": ann['keyword'],
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        instruction = random.choice(self.instruction_pool).format(data['question'])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": data['image'],
            "image_id": data['image_id'],
            "question": data['question'],
            "instruction_input": instruction,
            "category": data["category"],
            "keyword": data["keyword"],
        }
    
    def __len__(self):
        return len(self.flattened_annotation)


####################################
##### 2  Harm Image / Unharm Text #####
####################################
class SafeErase_Harmimg_Unharmtxt(torch.utils.data.Dataset):
    # image : Harm / question : Unharm 
    def __init__(self, vis_processor, vis_root, ann_paths, keywords):
        self.vis_processor = vis_processor
        self.instruction_pool = ["{}",]
        self.annotation = ann_paths
        self.vis_root = vis_root

        print(f'[DATA] visual root {vis_root}')
        print(f'[DATA] Requested Keywords : {keywords}')
        # Flatten the dataset to include all unsafe questions only
        self.flattened_annotation = []

        unharm_question_keys = ["UnharmPair_image1", "UnharmPair_image2"]
        for i, ann in enumerate(self.annotation):
            image_path = os.path.join(self.vis_root, ann["image_id"])
            if ann['keyword'] in keywords:
                if os.path.exists(image_path):
                    for unharm_question in unharm_question_keys:
                        try:
                            flattened_item = {
                                "image_id": ann["image_id"], 
                                "category": ann["category"],
                                "keyword": ann["keyword"],
                                "question": ann[unharm_question]['Question']
                            }
                            self.flattened_annotation.append(flattened_item)
                        except:
                            pass
        print(f'[DATA] The number of samples: {len(self.flattened_annotation)}')

    def get_data(self, index):
        ann = self.flattened_annotation[index]
        image_path = os.path.join(self.vis_root, ann["image_id"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = ann['question']
        
        return {
            "image": image,
            "question": question,
            "image_id": ann['image_id'], 
            "category": ann['category'],
            "keyword": ann['keyword'],
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        instruction = random.choice(self.instruction_pool).format(data['question'])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": data['image'],
            "image_id": data['image_id'],
            "question": data['question'],
            "instruction_input": instruction,
            "category": data["category"],
            "keyword": data["keyword"],
        }
    
    def __len__(self):
        return len(self.flattened_annotation)
    


####################################
##### 3 Unharm Image / Harm Text #####
####################################
class SafeErase_Unharmimg_Harmtxt(torch.utils.data.Dataset):
    # image : Harm / question : Unharm 
    def __init__(self, vis_processor, vis_root, ann_paths, keywords):
        self.vis_processor = vis_processor
        self.instruction_pool = ["{}",]
        self.annotation = ann_paths
        self.vis_root = vis_root

        print(f'[DATA] visual root {vis_root}')
        print(f'[DATA] Requested Keywords : {keywords}')
        # Flatten the dataset to include all unsafe questions only
        self.flattened_annotation = []

        for i, ann in enumerate(self.annotation):
            image_path = os.path.join(self.vis_root, ann["SDImage_path"])
            if ann['keyword'] in keywords:
                if os.path.exists(image_path):
                    for unsafe_question in ann["unsafe_pairs"]:
                        flattened_item = {
                            "image_id": ann["SDImage_path"], 
                            "category": ann["category"],
                            "keyword": ann["keyword"],
                            "question": unsafe_question['question']
                        }
                        self.flattened_annotation.append(flattened_item)
        
        print(f'[DATA] The number of samples: {len(self.flattened_annotation)}')

    def get_data(self, index):
        ann = self.flattened_annotation[index]
        image_path = os.path.join(self.vis_root, ann["image_id"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = ann['question']
        
        return {
            "image": image,
            "question": question,
            "image_id": ann['image_id'], 
            "category": ann['category'],
            "keyword": ann['keyword'],
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        instruction = random.choice(self.instruction_pool).format(data['question'])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": data['image'],
            "image_id": data['image_id'],
            "question": data['question'],
            "instruction_input": instruction,
            "category": data["category"],
            "keyword": data["keyword"],
        }
    
    def __len__(self):
        return len(self.flattened_annotation)



####################################
##### 4 Unharm Image / Unharm Text #####
####################################
class SafeErase_Unharmimg_Unharmtxt(torch.utils.data.Dataset):
    # image : Harm / question : Unharm 
    def __init__(self, vis_processor, vis_root, ann_paths, keywords):
        self.vis_processor = vis_processor
        self.instruction_pool = ["{}",]
        self.annotation = ann_paths
        self.vis_root = vis_root

        print(f'[DATA] visual root {vis_root}')
        print(f'[DATA] Requested Keywords : {keywords}')
        # Flatten the dataset to include all unsafe questions only
        self.flattened_annotation = []

        unharm_question_keys = ["UnharmPair_image1", "UnharmPair_image2"]
        for i, ann in enumerate(self.annotation):
            image_path = os.path.join(self.vis_root, ann["SDImage_path"])
            if ann['keyword'] in keywords:
                if os.path.exists(image_path):
                    for unharm_question in unharm_question_keys:
                        try: 
                            flattened_item = {
                                "image_id": ann["SDImage_path"], 
                                "category": ann["category"],
                                "keyword": ann["keyword"],
                                "question": ann[unharm_question]['Question']
                            }
                            self.flattened_annotation.append(flattened_item)
                        except: 
                            pass 
                            
        print(f'[DATA] The number of samples: {len(self.flattened_annotation)}')

    def get_data(self, index):
        ann = self.flattened_annotation[index]
        image_path = os.path.join(self.vis_root, ann["image_id"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = ann['question']
        
        return {
            "image": image,
            "question": question,
            "image_id": ann['image_id'], 
            "category": ann['category'],
            "keyword": ann['keyword'],
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        instruction = random.choice(self.instruction_pool).format(data['question'])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": data['image'],
            "image_id": data['image_id'],
            "question": data['question'],
            "instruction_input": instruction,
            "category": data["category"],
            "keyword": data["keyword"],
        }
    
    def __len__(self):
        return len(self.flattened_annotation)



class ImageNetR_unlearn_PO(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, keywords):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.instruction_pool = ["{}",]

        print(f'[DATA] visual root {vis_root} | annotation paths {ann_paths}')
        print(f'[DATA] vis processor : {vis_processor}')
        print(f'[DATA] Requested Keywords : {keywords}')
        self.flattened_annotation = []

        for i, ann in enumerate(self.annotation):
            image_path = os.path.join(self.vis_root, ann["image_id"])
            if ann['keyword'] in keywords:
                if os.path.exists(image_path):
                    flattened_item = {
                        "image_id": ann["image_id"],
                        "category": ann["category"],
                        "keyword": ann["keyword"],
                        "question": ann["question"],
                    }
                    self.flattened_annotation.append(flattened_item)
            else: 
                pass 
        
        print(f'[DATA] Original annotations: {len(self.flattened_annotation)}')

    def get_data(self, index):
        ann = self.flattened_annotation[index]
        image_path = os.path.join(self.vis_root, ann["image_id"])
        category = ann["category"]
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        
        # Use the question from flattened annotation
        questions = random.choice(classification_prompts)
        question = self.text_processor(questions)
        answer = random.choice(classification_refusals)

        return {
            "image": image,
            "question": question,
            "answer": answer,
            "image_id": ann['image_id'], 
            "category": ann['category'],
            "keyword": ann['keyword'],
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        instruction = random.choice(self.instruction_pool).format(data['question'])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": data['image'],
            "image_id": data['image_id'],
            "question": data['question'],
            "instruction_input": instruction,
            "answer": self.text_processor(data['answer']),
            "category": data["category"],
            "keyword": data["keyword"],
        }
    
    def __len__(self):
        return len(self.flattened_annotation)
    


class ImageNetR_unlearn_GA(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, keywords):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.instruction_pool = ["{}",]

        print(f'[DATA] visual root {vis_root} | annotation paths {ann_paths}')
        print(f'[DATA] vis processor : {vis_processor}')
        print(f'[DATA] Requested Keywords : {keywords}')
        self.flattened_annotation = []

        for i, ann in enumerate(self.annotation):
            image_path = os.path.join(self.vis_root, ann["image_id"])
            if ann['keyword'] in keywords:
                if os.path.exists(image_path):
                    flattened_item = {
                        "image_id": ann["image_id"],
                        "category": ann["category"],
                        "keyword": ann["keyword"],
                        "question": ann["question"],
                    }
                    self.flattened_annotation.append(flattened_item)
            else: 
                pass 
        
        print(f'[DATA] Original annotations: {len(self.flattened_annotation)}')

    def get_data(self, index):
        ann = self.flattened_annotation[index]
        image_path = os.path.join(self.vis_root, ann["image_id"])
        category = ann["category"]
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        
        # Use the question from flattened annotation
        questions = random.choice(classification_prompts)
        question = self.text_processor(questions)
        answer = ann["keyword"]

        return {
            "image": image,
            "question": question,
            "answer": answer,
            "image_id": ann['image_id'], 
            "category": ann['category'],
            "keyword": ann['keyword'],
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        instruction = random.choice(self.instruction_pool).format(data['question'])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": data['image'],
            "image_id": data['image_id'],
            "question": data['question'],
            "instruction_input": instruction,
            "answer": self.text_processor(data['answer']),
            "category": data["category"],
            "keyword": data["keyword"],
        }
    
    def __len__(self):
        return len(self.flattened_annotation)


class ImageNetR_EVAL(torch.utils.data.Dataset):
    """
    ImageNet-R Dataset with configurable harm/unharm combinations
    
    Args:
        vis_processor: Image preprocessing function
        vis_root: Root directory for images
        ann_paths: Annotation data
        keywords: Keywords to filter data
        harm_image: Boolean, whether to use harmful images
        harm_text: Boolean, whether to use harmful text
        classification_prompts: List of harmful/classification prompts
        unharm_random_texts: List of unharmful random texts
        remain_keywords: Keywords for unharmful images (used when harm_image=False)
    """
    
    def __init__(self, vis_processor, vis_root, ann_paths, keywords, 
                 harm_image=True, harm_text=True):
        
        self.vis_processor = vis_processor
        self.instruction_pool = ["{}"]
        self.annotation = ann_paths
        self.vis_root = vis_root
        self.harm_image = harm_image
        self.harm_text = harm_text
        self.classification_prompts = classification_prompts 
        self.unharm_random_texts = unharm_random_texts 
        
        # Determine which keywords to use
        if harm_image:
            self.keywords = keywords
        else:
            self.keywords = remain_keywords
        
        # Set dataset description
        image_type = "Harm" if harm_image else "Unharm"
        text_type = "Harm" if harm_text else "Unharm" 
        
        print(f'[DATA]  ImageNet-R: {image_type} Image - {text_type} Text')
        print(f'[DATA] visual root: {vis_root}')
        print(f'[DATA]  N Keywords: {len(self.keywords)}')
        
        self._build_annotation()
    
    def _build_annotation(self):
        """Build flattened annotation list based on keywords and existing images"""
        self.flattened_annotation = []
        
        for ann in self.annotation:
            image_path = os.path.join(self.vis_root, ann["image_id"])
            
            if ann['keyword'] in self.keywords and os.path.exists(image_path):
                flattened_item = {
                    "image_id": ann["image_id"],
                    "category": ann["category"],
                    "keyword": ann["keyword"],
                    "question": ann["question"],
                }
                self.flattened_annotation.append(flattened_item)
        
        print(f'[DATA] Original annotations: {len(self.flattened_annotation)}')
    
    def get_data(self, index):
        """Get data for a specific index"""
        ann = self.flattened_annotation[index]
        image_path = os.path.join(self.vis_root, ann["image_id"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        
        # Select question based on harm_text setting
        if self.harm_text:
            question = random.choice(self.classification_prompts)
        else:
            question = random.choice(self.unharm_random_texts)
        
        return {
            "image": image,
            "question": question,
            "image_id": ann['image_id'], 
            "category": ann['category'],
            "keyword": ann['keyword'],
        }
    
    def __getitem__(self, index):
        """Get item for training/evaluation"""
        data = self.get_data(index)
        instruction = random.choice(self.instruction_pool).format(data['question'])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)
        
        return {
            "image": data['image'],
            "image_id": data['image_id'],
            "question": data['question'],
            "instruction_input": instruction,
            "category": data["category"],
            "keyword": data["keyword"],
        }
    
    def __len__(self):
        return len(self.flattened_annotation)    


import os
import json
import random
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image


class StandardVLMBenchmark(Dataset):
    """
    Unified dataset for MMBench_v1.0, ScienceQA, and SEEDBench.

    Args:
        vis_processor: Callable image preprocessor
        benchmark_name: One of {"MMBench_v1.0", "ScienceQA_TEST", "SEEDBench_IMG"}
        vis_root: Root directory for images
        json_file: Path to a JSON or JSONL file
        categories: Optional list of category names to include
    """

    def __init__(
        self,
        vis_processor,
        benchmark_name: str,
        vis_root: str,
        json_file: str,
        categories: Optional[List[str]] = None,
    ) -> None:
        assert benchmark_name in {"MMBench_v1.0", "ScienceQA_TEST", "SEEDBench_IMG"}
        self.vis_processor = vis_processor
        self.benchmark_name = benchmark_name
        self.vis_root = vis_root
        self.categories = set(categories) if categories is not None else None

        # A single template is sufficient, yet a pool allows future extension
        self.instruction_pool = ["{}"]

        self.annotation: List[Dict[str, Any]] = self._load_json(json_file)
        self.flattened_annotation: List[Dict[str, Any]] = []
        self._build_annotation()

    # ------------------------------- IO helpers -------------------------------
    def _load_json(self, path: str) -> List[Dict[str, Any]]:
        ext = os.path.splitext(path)[1].lower()
        if ext in {".jsonl", ".jsonl.gz"}:
            records = []
            opener = open
            if ext.endswith(".gz"):
                import gzip
                opener = gzip.open  # type: ignore
            with opener(path, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
            return records
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Allow a dict with a top-level key
            if isinstance(data, dict):
                # Choose the first list-like field
                for v in data.values():
                    if isinstance(v, list):
                        return v
                raise ValueError("JSON dict has no list field to read")
            if isinstance(data, list):
                return data
            raise ValueError("Unsupported JSON structure")

    # ------------------------------ parsing logic -----------------------------
    def _record_to_common(self, rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Convert a raw record to a common schema:
        {
            image_path, question, hint, options: List[str], answer, category, index
        }
        Return None to drop the record.
        """
        name = self.benchmark_name
        if name == "MMBench_v1.0":
            if rec.get("source") == "code":
                return None
            image_path = rec.get("image_path")
            question = rec.get("question")
            hint = rec.get("hint")
            answer = rec.get("answer")
            category = rec.get("category") or rec.get("l2-category")
            index = rec.get("index")
            options = []
            for key in ["A", "B", "C", "D", "E"]:
                val = rec.get(key)
                if val is not None:
                    options.append(val)
        elif name == "ScienceQA_TEST":
            image_path = rec.get("image_path")
            question = rec.get("question")
            hint = rec.get("hint")
            answer = rec.get("answer")
            category = rec.get("category")
            index = rec.get("index")
            options = []
            for key in ["A", "B", "C", "D", "E"]:
                val = rec.get(key)
                if val is not None:
                    options.append(val)
        elif name == "SEEDBench_IMG":
            image_path = rec.get("image_path")
            question = rec.get("question")
            hint = None
            answer = rec.get("answer")
            category = rec.get("category")
            index = rec.get("index")
            options = []
            for key in ["A", "B", "C", "D", "E"]:
                val = rec.get(key)
                if val is not None:
                    options.append(val)
        else:
            return None

        # Basic validation
        if not image_path or not question:
            return None

        # Category filter when requested
        if self.categories is not None:
            cat1 = category if isinstance(category, str) else None
            cat2 = rec.get("l2-category") if isinstance(rec.get("l2-category"), str) else None
            if not ((cat1 and cat1 in self.categories) or (cat2 and cat2 in self.categories)):
                return None

        return {
            "image_path": image_path,
            "question": question,
            "hint": hint,
            "options": options,
            "answer": answer,
            "category": category,
            "index": index,
        }

    def _build_annotation(self) -> None:
        """Build a list of valid records with existing images."""
        kept: List[Dict[str, Any]] = []
        for rec in self.annotation:
            std = self._record_to_common(rec)
            if std is None:
                continue
            abs_image = os.path.join(self.vis_root, std["image_path"]) if not os.path.isabs(std["image_path"]) else std["image_path"]
            if os.path.exists(abs_image):
                std["abs_image_path"] = abs_image
                kept.append(std)

        # # -------- 최소 변경: SEEDBench에만 subset 적용 --------
        # if self.benchmark_name == "SEEDBench_IMG":
        #     try:

        #         with open(f'{self.vis_root}/seedbench_subset_ids.json', "r", encoding="utf-8") as f:
        #             meta = json.load(f)
        #         selected_ids = set(str(x) for x in meta.get("selected_ids", []))
        #         # 'index'는 문자열로 통일
        #         kept = [r for r in kept if str(r.get("index")) in selected_ids]
        #         print(f"[SEEDBench] Subset applied: {len(kept)} samples from {len(selected_ids)} ids")
        #     except Exception as e:
        #         print(f"[SEEDBench] Failed to apply subset ({e}). Using full set.")

        self.flattened_annotation = kept
        print(f"[DATA] Loaded records: {len(self.annotation)} | Usable records: {len(self.flattened_annotation)}")

    # ------------------------------- core access ------------------------------
    def _format_instruction(self, question: str, hint: Optional[str], options: List[str]) -> str:
        lines: List[str] = [question]
        if hint and str(hint).strip().lower() != "null":
            lines.append(hint)
        if options:
            # Map options to alphabetical markers in order
            labels = ["A", "B", "C", "D", "E"]
            opt_lines = []
            for i, opt in enumerate(options):
                if i >= len(labels):
                    break
                opt_lines.append(f"{labels[i]}. {opt}")
            lines.append("\n".join(opt_lines))
        payload = "\n".join(lines)
        template = random.choice(self.instruction_pool)
        instruction = template.format(payload)
        instruction = f"<Img><ImageHere></Img> {instruction} "
        return instruction

    def _format_question(self, question: str, hint: Optional[str], options: List[str]) -> str:
        """Make question string like instruction_input but without <Img><ImageHere></Img>"""
        lines: List[str] = [question]
        if hint and str(hint).strip().lower() != "null":
            lines.append(hint)
        if options:
            labels = ["A", "B", "C", "D", "E"]
            opt_lines = []
            for i, opt in enumerate(options):
                if i >= len(labels):
                    break
                opt_lines.append(f"{labels[i]}. {opt}")
            lines.append("\n".join(opt_lines))
        return "\n".join(lines)

    def get_data(self, index: int) -> Dict[str, Any]:
        ann = self.flattened_annotation[index]

        # 이미지 경로 그대로 반환 (원한다면 전처리 추가)
        # image = ann["abs_image_path"]
        image = Image.open(ann["abs_image_path"]).convert("RGB") 
        image = self.vis_processor(image)

        # instruction_input (이미지 토큰 포함)
        instruction = self._format_instruction(
            question=ann["question"],
            hint=ann.get("hint"),
            options=ann.get("options", []),
        )

        # question (이미지 토큰 제외)
        formatted_question = self._format_question(
            question=ann["question"],
            hint=ann.get("hint"),
            options=ann.get("options", []),
        )

        # answer_as_text 생성
        answer = ann.get("answer")
        options = ann.get("options", [])
        answer_as_text = ""
        if answer is not None and isinstance(answer, str):
            labels = ["A", "B", "C", "D", "E"]
            if answer in labels:
                idx = labels.index(answer)
                if idx < len(options):
                    answer_as_text = options[idx]

        return {
            "image": image,
            "instruction_input": instruction,
            "question": formatted_question,
            "options": options,
            "answer": answer,
            "answer_as_text": answer_as_text,   # <- 추가된 부분
            "image_id": ann.get("index") or os.path.basename(ann["abs_image_path"]),
            "category": ann.get("category", ""),
            "keyword": None,
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = self.get_data(index)
        return data

    def __len__(self) -> int:
        return len(self.flattened_annotation)