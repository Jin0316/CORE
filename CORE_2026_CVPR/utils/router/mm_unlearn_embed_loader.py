from typing import List, Optional, Union, Dict, Any
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.distributions import MultivariateNormal
import json

class KeywordEmbeddingDataset(torch.utils.data.Dataset):
    """
    Each item in samples_with_embeddings.pt is expected to be a dict:
      {
        'image_embed': Tensor[D_img],
        'text_embed' : Tensor[D_txt],
        'keyword'    : str,
        'category'   : Any
      }
    """

    def __init__(
        self,
        pt_path: str,
        keyword: Optional[Union[str, List[str]]] = None,
        prev_keyword: Optional[Union[str, List[str]]] = None,
        new_keyword: Optional[Union[str, List[str]]] = None,
        strict: bool = False,
        n_samples = 100
    ):
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"File not found: {pt_path}")
        data = torch.load(pt_path, map_location="cpu")
        if not isinstance(data, list):
            raise ValueError("Loaded object must be a list of sample dicts.")

        # normalize filter for regular keywords
        if keyword is None:
            keys_norm = None
        else:
            if isinstance(keyword, str):
                keys_norm = [keyword]
            else:
                keys_norm = [k for k in keyword]

        # normalize filter for prev_keyword
        if prev_keyword is None:
            prev_keys_norm = None
        else:
            if isinstance(prev_keyword, str):
                prev_keys_norm = [prev_keyword]
            else:
                prev_keys_norm = [k for k in prev_keyword]

        # normalize filter for new_keyword
        if new_keyword is None:
            new_keys_norm = None
        else:
            if isinstance(new_keyword, str):
                new_keys_norm = [new_keyword]
            else:
                new_keys_norm = [k for k in new_keyword]

        # def match(k: str, target_keys: List[str]) -> bool:
        #     if target_keys is None:
        #         return True
        #     kf = str(k).casefold()
        #     if strict:
        #         return any(kf == t for t in target_keys)
        #     return any(t in kf for t in target_keys)
        def match(k: str, target_keys: List[str], strict: bool = False) -> bool:
            if target_keys is None:
                return True
            if strict:
                return any(k == t for t in target_keys)
            return any(t in k for t in target_keys)

        # Collect samples for regular keywords
        self.samples = []
        if keys_norm is not None:
            for it in data:
                if not isinstance(it, dict):
                    continue
                if not all(k in it for k in ("image_embed", "text_embed", "keyword")):
                    continue
                if not isinstance(it["image_embed"], torch.Tensor) or not isinstance(it["text_embed"], torch.Tensor):
                    continue
                if match(it["keyword"], keys_norm):
                    self.samples.append({
                        "image_embed": it["image_embed"].detach(),
                        "text_embed":  it["text_embed"].detach(),
                        "keyword":     it["keyword"],
                        "category":    it.get("category", None),
                    })

        # Collect and process samples for new_keyword (directly add without sampling)
        if new_keys_norm is not None:
            for it in data:
                if not isinstance(it, dict):
                    continue
                if not all(k in it for k in ("image_embed", "text_embed", "keyword")):
                    continue
                if not isinstance(it["image_embed"], torch.Tensor) or not isinstance(it["text_embed"], torch.Tensor):
                    continue
                if match(it["keyword"], new_keys_norm):
                    self.samples.append({
                        "image_embed": it["image_embed"].detach(),
                        "text_embed":  it["text_embed"].detach(),
                        "keyword":     it["keyword"],
                        "category":    it.get("category", None),
                    })

        # Collect and process samples for prev_keyword (with sampling)
        if prev_keys_norm is not None:
            # Group samples by keyword for prev_keyword
            prev_samples_by_kw = {}
            for it in data:
                if not isinstance(it, dict):
                    continue
                if not all(k in it for k in ("image_embed", "text_embed", "keyword")):
                    continue
                if not isinstance(it["image_embed"], torch.Tensor) or not isinstance(it["text_embed"], torch.Tensor):
                    continue
                if match(it["keyword"], prev_keys_norm):
                    kw = it["keyword"]
                    if kw not in prev_samples_by_kw:
                        prev_samples_by_kw[kw] = {
                            "image_embeds": [],
                            "text_embeds": [],
                            "category": it.get("category", None)
                        }
                    prev_samples_by_kw[kw]["image_embeds"].append(it["image_embed"].detach())
                    prev_samples_by_kw[kw]["text_embeds"].append(it["text_embed"].detach())

            # Sample from MultivariateNormal for each prev_keyword
            for kw, kw_data in prev_samples_by_kw.items():
                img_embs = torch.stack(kw_data["image_embeds"])  # [N, D_img]
                txt_embs = torch.stack(kw_data["text_embeds"])   # [N, D_txt]
                
                if len(img_embs) > 1:  # Need at least 2 samples for covariance
                    # Sample image embeddings
                    img_mean = img_embs.mean(dim=0)
                    embed_dim = img_embs.size(-1)
                    img_cov_matrix = torch.cov(img_embs.T) + 1e-6 * torch.eye(embed_dim)
                    img_dist = MultivariateNormal(img_mean, covariance_matrix=img_cov_matrix)
                    
                    # Sample text embeddings
                    txt_mean = txt_embs.mean(dim=0)
                    txt_embed_dim = txt_embs.size(-1)
                    txt_cov_matrix = torch.cov(txt_embs.T) + 1e-6 * torch.eye(txt_embed_dim)
                    txt_dist = MultivariateNormal(txt_mean, covariance_matrix=txt_cov_matrix)
                    
                    # Generate same number of samples as original
                    # n_samples = len(img_embs)
                    sampled_img_embs = img_dist.sample((n_samples,))
                    sampled_txt_embs = txt_dist.sample((n_samples,))
                    
                    # Add sampled embeddings to dataset
                    for i in range(n_samples):
                        self.samples.append({
                            "image_embed": sampled_img_embs[i],
                            "text_embed":  sampled_txt_embs[i],
                            "keyword":     kw,
                            "category":    kw_data["category"],
                        })
                else:
                    # If only one sample, just use it as is
                    self.samples.append({
                        "image_embed": img_embs[0],
                        "text_embed":  txt_embs[0],
                        "keyword":     kw,
                        "category":    kw_data["category"],
                    })

        if len(self.samples) == 0:
            raise ValueError("No samples matched. Check keywords or file content.")

        # infer dims
        self.img_dim = self.samples[0]["image_embed"].numel()
        self.txt_dim = self.samples[0]["text_embed"].numel()

        # collect all keywords
        self.keywords = sorted({s["keyword"] for s in self.samples})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            "image_embed": self.samples[idx]["image_embed"],
            "text_embed":  self.samples[idx]["text_embed"],
            "keyword":     self.samples[idx]["keyword"],
            "category":    self.samples[idx].get("category", None),
        }


def keyword_collate_fn(batch):
    # batch is a list[dict]
    imgs = torch.stack([b["image_embed"] for b in batch], dim=0)  # [B, D_img]
    txts = torch.stack([b["text_embed"]  for b in batch], dim=0)  # [B, D_txt]
    kws  = [b["keyword"]  for b in batch]
    cats = [b["category"] for b in batch]
    return {
        "image_embed": imgs,
        "text_embed": txts,
        "keyword": kws,
        "category": cats,
    }

def make_dataloader(
    pt_path: str,
    batch_size: int,
    keyword: Optional[Union[str, List[str]]] = None,
    prev_keyword: Optional[Union[str, List[str]]] = None,
    new_keyword: Optional[Union[str, List[str]]] = None,
    strict: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    n_samples = 200
):
    ds = KeywordEmbeddingDataset(pt_path, keyword=keyword, prev_keyword=prev_keyword, new_keyword=new_keyword, strict=strict, n_samples=n_samples)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=keyword_collate_fn,
        drop_last=False,
    )

def extract_unique_keywords(json_path: str):
    """
    JSON 파일에서 unique한 keyword들을 모두 추출한다.

    Args:
        json_path: JSON 파일 경로
    
    Returns:
        keywords (list): unique keyword 리스트
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    keywords = set()
    for item in data:
        if isinstance(item, dict):
            kw = item.get("keyword")
            if isinstance(kw, str):
                keywords.add(kw)

    return list(keywords)