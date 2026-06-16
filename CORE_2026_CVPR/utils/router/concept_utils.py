import json
from typing import List, Dict, Tuple
from .mm_unlearn_embed_loader import make_dataloader, extract_unique_keywords
import torch 

def cbl_alignment_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    preds:   [B, D]
    targets: [B, D]
    """
    preds = preds ** 3
    targets = targets ** 3

    preds = preds / (torch.norm(preds, dim=1, keepdim=True) + 1e-12)
    targets = targets / (torch.norm(targets, dim=1, keepdim=True) + 1e-12)

    similarities = torch.sum(preds * targets, dim=1)  # [B]
    return -similarities.mean()

def load_keyword_concepts(json_path: str, keywords: List[str], num_desc: int = None) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    특정 keyword에 해당하는 description을 반환한다.
    
    Args:
        json_path (str): concept json 파일 경로
        keywords (List[str]): 추출할 keyword 리스트
        num_desc (int, optional): 각 keyword당 가져올 description 개수. None이면 전체
    
    Returns:
        concept_dict (Dict[str, List[str]]): keyword → descriptions
        flat_descriptions (List[str]): description들만 순서대로 flatten한 리스트
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    concept_dict: Dict[str, List[str]] = {}
    flat_descriptions: List[str] = []

    for item in data:
        if not isinstance(item, dict):
            continue
        kw = item.get("keyword")
        if kw in keywords:
            descriptions = item.get("descriptions", [])
            if isinstance(descriptions, list):
                # num_desc만큼만 가져오기
                if num_desc is not None:
                    descriptions = descriptions[:num_desc]
                
                concept_dict[kw] = descriptions
                for desc in descriptions:
                    flat_descriptions.append(desc)

    return concept_dict, flat_descriptions