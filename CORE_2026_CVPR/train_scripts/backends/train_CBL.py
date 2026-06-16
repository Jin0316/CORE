# train_CBL.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from argparse import ArgumentParser

import sys
# this backend lives in train_scripts/backends/; put repo root on sys.path so
# `minigpt4` / `utils_router` resolve when launched as `python train_scripts/backends/...`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from minigpt4.models.mm_cbl import DualCBLClassifier
from utils.router.concept_utils import load_keyword_concepts, cbl_alignment_loss
from utils.router.similarity_extract import Encoder as ImageTextEncoder
from utils.router.similarity_extract_text import TextEncoder as TextOnlyEncoder
from utils.router.mm_unlearn_embed_loader import make_dataloader

import random 
import numpy as np 

def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"[CBL] Seed set to {seed} for reproducibility")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--train_keywords",
        type=str,   # 문자열로만 받음
        required=True,
        help='List of keywords for training, e.g., --train_keywords "FamilyAbuse,PhysicalAssault,BombAttack"'
    )

    parser.add_argument("--n_con", type=int, default = 20)

    parser.add_argument("--vis_dir", type=str, default="/workspace/datasets/safe_eraser/dataset/")

    parser.add_argument("--img_cpt_json_path",  type=str, default="CONCEPTS/GPT/All_concepts_image.json")
    parser.add_argument("--text_cpt_json_path", type=str, default="CONCEPTS/GPT/All_concepts_inst.json")
    parser.add_argument("--encoder_type", type=str, default="eva_clip", choices=["clip", "eva_clip"])
    
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="./CL_CBL")
    parser.add_argument("--keyword_loss_weight", type=float, default=0.1, help="Weight for keyword similarity loss")
    parser.add_argument("--use_concept_masking", action="store_true", help="Use keyword-based concept masking")
    
    # Additional arguments from train_router.py
    parser.add_argument("--time_step", type=str, required=True, help="time step index, zero indexing")
    
    return parser.parse_args()


def load_or_create_model(time_step, keywords, img_cpt_list, txt_cpt_list, img_encoder, save_dir ='./CL_CBL'):
    """
    time_step == 0: 새 모델 초기화
    time_step > 0: t-1 시점의 모델을 로드하고 필요시 expand
    """
    time_step = int(time_step)
    
    if time_step == 0:
        print('[CBL] Create New model')
        # 새로운 모델 생성
        keywords_ordered = sorted(keywords)
        img_cpt_list_ordered = sorted(img_cpt_list)
        txt_cpt_list_ordered = sorted(txt_cpt_list)
        
        print(f'[CBL] image encoder feature dim: {img_encoder.feat_dim}') # 256
        model = DualCBLClassifier(
            image_in_dim=img_encoder.feat_dim,
            text_in_dim=768,
            num_img_concepts=len(img_cpt_list_ordered),
            num_txt_concepts=len(txt_cpt_list_ordered),
            num_classes=len(keywords_ordered),
        )
        
        return model, keywords_ordered, img_cpt_list_ordered, txt_cpt_list_ordered, [], [], [], {}, {}
    

    prev_file = f"cbl_{time_step-1}.pt"
    prev_model_path = os.path.join(save_dir, prev_file)
    print(f'[CBL] Load model from prev_model_path {prev_model_path}')
    checkpoint = torch.load(prev_model_path, map_location='cpu')
    
    # 이전 순서 정보 로드
    prev_keywords = checkpoint['keywords_ordered']
    prev_img_concepts = checkpoint['img_concepts_ordered']
    prev_txt_concepts = checkpoint['txt_concepts_ordered']
    prev_img_cpt_dict = checkpoint['img_cpt_dict']
    prev_txt_cpt_dict = checkpoint['txt_cpt_dict']
    
    # 순서 보장: (이전 순서) + (새로운 것들만 정렬해서 추가)
    new_keywords = sorted(set(keywords) - set(prev_keywords))
    keywords_ordered = prev_keywords + new_keywords

    new_img_concepts = sorted(set(img_cpt_list) - set(prev_img_concepts))
    img_cpt_list_ordered = prev_img_concepts + new_img_concepts
    
    new_txt_concepts = sorted(set(txt_cpt_list) - set(prev_txt_concepts))
    txt_cpt_list_ordered = prev_txt_concepts + new_txt_concepts
    print(f"[CBL] prev kw:{len(prev_keywords)} img:{len(prev_img_concepts)} txt:{len(prev_txt_concepts)}")
    print(f"[CBL] new kw:{len(new_keywords)} img:{len(new_img_concepts)} txt:{len(new_txt_concepts)}")
    
    #############################
    # Make model configurations #
    #############################
    cbl_state_dict = checkpoint['cbl_model']
    existing_img_concepts = cbl_state_dict['img_cbl.weight'].shape[0]
    existing_txt_concepts = cbl_state_dict['txt_cbl.weight'].shape[0]
    existing_classes = cbl_state_dict['classifier.weight'].shape[0]
    existing_img_in_dim = cbl_state_dict['img_cbl.weight'].shape[1]
    existing_txt_in_dim = cbl_state_dict['txt_cbl.weight'].shape[1]
    
    model = DualCBLClassifier(
        image_in_dim=existing_img_in_dim,
        text_in_dim=existing_txt_in_dim,
        num_img_concepts=existing_img_concepts,
        num_txt_concepts=existing_txt_concepts,
        num_classes=existing_classes,
    )
    ##########################
    # Load Model, embeddings #
    ##########################
    model.load_state_dict(cbl_state_dict)
    model.avg_img_cpt_scores = checkpoint.get('avg_img_cpt_scores', torch.tensor([]))
    model.avg_txt_cpt_scores = checkpoint.get('avg_txt_cpt_scores', torch.tensor([]))
    model.keyword_to_task = checkpoint.get('keyword_to_task', {})
    #########################
    ######## EXPAND  ######## 
    if len(img_cpt_list_ordered) > existing_img_concepts or len(txt_cpt_list_ordered) > existing_txt_concepts:
        model.expand_concepts(
            new_num_img_concepts=len(img_cpt_list_ordered),
            new_num_txt_concepts=len(txt_cpt_list_ordered),
            init_new="xavier"
        )
    
    if len(keywords_ordered) > existing_classes:
        model.expand_classes(new_num_classes=len(keywords_ordered), init_new="xavier")
    #########################
        
    return model, keywords_ordered, img_cpt_list_ordered, txt_cpt_list_ordered, prev_keywords, prev_img_concepts, prev_txt_concepts, prev_img_cpt_dict, prev_txt_cpt_dict

def main():
    args = parse_args()
    set_seed()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    
    # 1. 데이터 불러오기
    # keywords = extract_unique_keywords(args.train_json_path)
    raw_kw = args.train_keywords  # 예: "FamilyAbuse,PhysicalAssault,BombAttack"
    # 대괄호가 섞여 들어와도 견고하게 처리
    _raw = raw_kw.strip().strip("[]")
    keywords = [k.strip() for k in _raw.split(",") if k.strip()]

    # 2. 인코더 준비
    img_encoder = ImageTextEncoder(name=args.encoder_type, device=args.device)
    txt_encoder = TextOnlyEncoder(device=args.device)

    # 3. concept description 로드 및 개념 임베딩 추출
    img_cpt_dict, img_cpt_list = load_keyword_concepts(args.img_cpt_json_path, keywords, num_desc=args.n_con)
    txt_cpt_dict, txt_cpt_list = load_keyword_concepts(args.text_cpt_json_path, keywords, num_desc=args.n_con)

    # 4. 모델 초기화 또는 이전 시점에서 로드 (순서 보장) - 수정된 부분
    model, keywords_ordered, img_cpt_list_ordered, txt_cpt_list_ordered, prev_keywords, prev_img_concepts, prev_txt_concepts, prev_img_cpt_dict, prev_txt_cpt_dict = load_or_create_model(
        args.time_step, keywords, img_cpt_list, txt_cpt_list, img_encoder, save_dir=args.save_dir
    )
    model = model.to(device)
    img_cpt_dict.update(prev_img_cpt_dict)
    txt_cpt_dict.update(prev_txt_cpt_dict)

    keyword2idx = {kw: i for i, kw in enumerate(keywords_ordered)}

    # 새로운 키워드만 추출
    new_keywords = sorted(set(keywords) - set(prev_keywords))
    
    print(f'[CBL] keywords: {len(keywords)} (prev {len(prev_keywords)}, new {len(new_keywords)})')
    if int(args.time_step) == 0:
        # time_step == 0: 모든 키워드를 new_keyword로 처리 
        dataloader = make_dataloader(
            pt_path="utils/CBL/samples_with_embeddings.pt",
            new_keyword=keywords,  # 모든 키워드를 new로 처리
            strict=False,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
    else:
        # time_step > 0: prev는 sampling, new는 실제 embedding 사용
        dataloader = make_dataloader(
            pt_path="utils/CBL/samples_with_embeddings.pt",
            prev_keyword=prev_keywords,  # sampling으로 처리
            new_keyword=new_keywords,    # 실제 embedding으로 처리
            strict=False,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

    print("[CBL] Extracting image concept embeddings with CLIP text encoder")
    img_concept_embeds = img_encoder.encode_texts(img_cpt_list_ordered).to(device)
    print("[CBL] Extracting text concept embeddings with SentenceTransformer")
    txt_concept_embeds = txt_encoder.encode_texts(txt_cpt_list_ordered).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # 5. 학습
    print("[CBL] Start training")
    model.train()
    
    # concept masking을 위한 유사도 행렬 (첫 번째 에포크에서만 계산)
    img_kw_sim_matrix = None
    txt_kw_sim_matrix = None
    
    for ep in range(args.epoch):
        # 첫 번째 에포크에서 keyword concept 유사도 계산 (masking용)
        if ep == 0 and args.use_concept_masking and len(keywords_ordered) > 1:
            print("[CBL] Computing keyword concept similarities for masking")
            model.eval()
            with torch.no_grad():
                img_kw_concepts = model.keyword_img_concepts
                img_kw_concepts_norm = torch.nn.functional.normalize(img_kw_concepts, p=2, dim=1)
                img_kw_sim_matrix = torch.mm(img_kw_concepts_norm, img_kw_concepts_norm.t())
                
                txt_kw_concepts = model.keyword_txt_concepts
                txt_kw_concepts_norm = torch.nn.functional.normalize(txt_kw_concepts, p=2, dim=1)
                txt_kw_sim_matrix = torch.mm(txt_kw_concepts_norm, txt_kw_concepts_norm.t())
            model.train()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_img_cbl_loss = 0.0
        total_txt_cbl_loss = 0.0
        total_img_kw_loss = 0.0
        total_txt_kw_loss = 0.0

        
        progress = tqdm(dataloader, desc=f"[Epoch {ep+1}]", leave=False)
        for batch in progress:
            # CHANGED: take precomputed embeddings directly
            image_feats = batch["image_embed"].to(device)    # [B, D_img]
            text_feats  = batch["text_embed"].to(device)     # [B, D_txt]
            kws        = batch["keyword"]                    # List[str]

            # CBL 출력
            img_cbl_out = model.img_cbl(image_feats)    # [B, N_img]
            txt_cbl_out = model.txt_cbl(text_feats)     # [B, N_txt]

            # 타깃 유사도 행렬
            img_targets = image_feats @ img_concept_embeds.T    # [B, N_img]
            txt_targets = text_feats  @ txt_concept_embeds.T    # [B, N_txt]
            
            
            # 분류 손실
            logits = model.logits_from_concepts(img_cbl_out, txt_cbl_out)      # [B, C]
            labels = torch.tensor([keyword2idx[k] for k in kws], dtype=torch.long, device=device)
            cls_loss = criterion(logits, labels)

            img_masks, txt_masks = model.compute_concept_masks(
                keywords=kws,
                img_cpt_dict=img_cpt_dict,
                txt_cpt_dict=txt_cpt_dict,
                img_cpt_list_ordered=img_cpt_list_ordered,
                txt_cpt_list_ordered=txt_cpt_list_ordered,
                device=device
            )
                
            img_targets = img_targets * img_masks  # [B, N_img]
            txt_targets = txt_targets * txt_masks  # [B, N_txt]
            # print(img_masks)
            # print(img_cbl_out)
            # CBL 정렬 손실
            img_cbl_loss = cbl_alignment_loss(img_cbl_out, img_targets)
            txt_cbl_loss = cbl_alignment_loss(txt_cbl_out, txt_targets)

            # 새로 추가: keyword concept 유사도 손실
            img_kw_loss, txt_kw_loss = model.compute_keyword_similarity_loss(
                image_feats, text_feats, labels
            )

            # 전체 손실
            loss = cls_loss + img_cbl_loss + txt_cbl_loss + args.keyword_loss_weight * (img_kw_loss + txt_kw_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 누적
            total_loss         += float(loss.item())
            total_cls_loss     += float(cls_loss.item())
            total_img_cbl_loss += float(img_cbl_loss.item())
            total_txt_cbl_loss += float(txt_cbl_loss.item())
            total_img_kw_loss  += float(img_kw_loss.item())
            total_txt_kw_loss  += float(txt_kw_loss.item())

            progress.set_postfix({
                "cls_loss": f"{cls_loss.item():.4f}",
                "img_cbl_loss": f"{img_cbl_loss.item():.4f}",
                "txt_cbl_loss": f"{txt_cbl_loss.item():.4f}",
                "total": f"{loss.item():.4f}"
            })

        num_batches = len(dataloader)
        print(f"[CBL] Epoch {ep+1} | total loss {total_loss / num_batches:.4f} | "
              f"Cls Loss {total_cls_loss / num_batches:.4f} | "
              f"Img CBL Loss {total_img_cbl_loss / num_batches:.4f} | "
              f"Txt CBL Loss {total_txt_cbl_loss / num_batches:.4f} ")


    # 태스크 완료 후
    # 학습 완료 후에 한 번만 (train_CBL.py에서)
    current_task_id = int(args.time_step)
    meta_data = {}
    meta_data['img_cpt_dict'] = img_cpt_dict
    meta_data['txt_cpt_dict'] = txt_cpt_dict
    meta_data['img_concepts_ordered'] = img_cpt_list_ordered
    meta_data['txt_concepts_ordered'] = txt_cpt_list_ordered


    model.compute_task_avg_score(dataloader, new_keywords, current_task_id, 
                                 keywords_ordered, meta_data, device)
    print(f'[CBL] avg_img_cpt_scores shape : {model.avg_img_cpt_scores.shape}')
    print(f'[CBL] avg_txt_cpt_scores shape : {model.avg_txt_cpt_scores.shape}')
    model.compute_inter_task_similarity(debug=True)
                                    # dataloader, device, img_cpt_dict, txt_cpt_dict, img_concepts_ordered, txt_concepts_ordered, logit
    model.compute_keyword_avg_score(dataloader, device, img_cpt_dict, txt_cpt_dict, img_cpt_list_ordered, txt_cpt_list_ordered, logit=True)
    model.compute_all_keyword_similarity_matrix(debug=False)
    model.freeze_keyword_concepts()

    # 8. 저장 - path_{t}.pt 형태로 저장
    time_step = int(args.time_step)
    save_dir = args.save_dir
    save_file = f"cbl_{time_step}.pt"
    save_path = os.path.join(save_dir, save_file)

    save_data = {
        'cbl_model': model.state_dict(),
        'time_step': time_step,
        'keywords_ordered': keywords_ordered,
        'img_concepts_ordered': img_cpt_list_ordered,
        'txt_concepts_ordered': txt_cpt_list_ordered,
        'keyword2idx': keyword2idx,
        'img_kw_sim_matrix': img_kw_sim_matrix.cpu() if img_kw_sim_matrix is not None else None,
        'txt_kw_sim_matrix': txt_kw_sim_matrix.cpu() if txt_kw_sim_matrix is not None else None,
        'img_cpt_dict': img_cpt_dict,
        'txt_cpt_dict': txt_cpt_dict,

        'avg_img_cpt_scores': model.avg_img_cpt_scores,
        'avg_txt_cpt_scores': model.avg_txt_cpt_scores,
        'keyword_to_task': model.keyword_to_task,
        'keyword_avg_img_scores': model.keyword_avg_img_scores,
        'keyword_avg_txt_scores': model.keyword_avg_txt_scores
    }

    os.makedirs(save_dir, exist_ok=True)
    torch.save(save_data, save_path)
    print(f"[CBL] Model saved to {save_path}")


if __name__ == "__main__":
    main()