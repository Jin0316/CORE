import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from typing import List, Union, Tuple, Optional
import torch.nn as nn 

class TextEncoder(nn.Module):
    def __init__(self, cache: bool = True, lazy_load: bool = True):
        super().__init__()
        self.device = torch.device("cpu")
        self.cache_on = cache
        self._cache = {}
        self.model = None  # 지연 로딩을 위해 None으로 초기화
        
        if not lazy_load:
            self._init_model()  # 즉시 로딩을 원하는 경우에만

    def _init_model(self):
        """모델을 CPU에서 로드"""
        if self.model is not None:
            print("[MODEL] SentenceTransformer already loaded, skipping...")
            return
            
        print("[MODEL] Loading SentenceTransformer on CPU...")
        
        # 강제로 CPU에서 로드
        import os
        original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 임시로 CUDA 숨기기
        
        try:
            self.model = SentenceTransformer("all-mpnet-base-v2", device='cpu')
        finally:
            # CUDA_VISIBLE_DEVICES 원복
            if original_cuda_visible is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
            elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                del os.environ['CUDA_VISIBLE_DEVICES']
        
        # CPU로 명시적 이동하고 gradient 비활성화
        self.model = self.model.to('cpu')
        for n, p in self.model.named_parameters():
            p.requires_grad = False
            p.data = p.data.cpu()
            
        print("[MODEL] SentenceTransformer loaded on CPU successfully")

    def _ensure_model_loaded(self):
        """모델이 로드되지 않았다면 로드"""
        if self.model is None:
            self._init_model()

    def set_device(self, device): 
        """모델을 지정된 device로 이동"""
        self.device = torch.device(device)
        
        # 모델이 로드되지 않았다면 로드
        self._ensure_model_loaded()
        
        print(f"[MODEL] Moving SentenceTransformer to {self.device}...")
        
        self.model = self.model.to(self.device)
        
        # 캐시된 임베딩들도 새로운 device로 이동
        if self.cache_on and self._cache:
            print("[MODEL] Moving cached text embeddings...")
            for key, value in self._cache.items():
                if isinstance(value, torch.Tensor):
                    self._cache[key] = value.to(self.device)
                    
        print(f"[MODEL] Model moved to {self.device}")

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        self._ensure_model_loaded()
        
        if not texts:
            return torch.empty(0, 768, device=self.device)

        # 캐시 분기
        outs, miss_idx, miss_terms = [], [], []
        if self.cache_on:
            for i, t in enumerate(texts):
                v = self._cache.get(t)
                if v is None:
                    miss_idx.append(i)
                    miss_terms.append(t)
                    outs.append(None)
                else:
                    outs.append(v.to(self.device))  # 캐시된 값을 현재 device로
        else:
            miss_idx = list(range(len(texts)))
            miss_terms = texts
            outs = [None] * len(texts)

        if miss_terms:
            # SentenceTransformer가 직접 텐서를 반환하도록 설정
            vec = self.model.encode(
                miss_terms,
                batch_size=batch_size,
                convert_to_tensor=True,
                normalize_embeddings=True,  # L2 정규화
                device=self.device,
                show_progress_bar=False,
            ).to(torch.float32)  # [M, D]

            for i, v, term in zip(miss_idx, vec, miss_terms):
                outs[i] = v
                if self.cache_on:
                    # 캐시는 현재 device에 저장
                    self._cache[term] = v.detach()

        return torch.stack(outs, dim=0)  # [N, D]

    @torch.no_grad()
    def compute_similarity(
        self,
        texts_a: Union[str, List[str], torch.Tensor],
        texts_b: Union[str, List[str]],
        normalize: Optional[str] = None,  # None | "softmax" | "cube_softmax"
    ) -> torch.Tensor:

        if isinstance(texts_a, torch.Tensor):
            emb_a = texts_a.to(self.device)
            if emb_a.dim() == 1:   # [D]
                emb_a = emb_a.unsqueeze(0)
            elif emb_a.dim() != 2:  # 허용: [A,D]
                raise ValueError("texts_a 텐서는 [D] 또는 [A,D] 여야 한다.")
            emb_a = F.normalize(emb_a, dim=-1).to(torch.float32)
        else:
            if isinstance(texts_a, str):
                texts_a = [texts_a]
            emb_a = self.encode_texts(texts_a)  # [A,D], 이미 정규화됨

        # B 임베딩 준비
        if isinstance(texts_b, str):
            texts_b = [texts_b]
        emb_b = self.encode_texts(texts_b)      # [B,D], 이미 정규화됨

        sims = emb_a @ emb_b.T                  # [A,B], cosine

        if normalize is None:
            return sims
        if normalize == "softmax":
            return F.softmax(sims, dim=1)
        if normalize == "cube_softmax":
            return F.softmax(sims.pow(3), dim=1)
        raise ValueError('normalize must be one of None, "softmax", "cube_softmax"')

    @torch.no_grad()
    def topk(
        self,
        queries: Union[str, List[str]],
        corpus: List[str],
        k: int = 5,
        normalize: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        입력: 질의 texts [Q], 말뭉치 texts [C]
        출력: (indices [Q, k], scores [Q, k])
        """
        sims = self.compute_similarity(queries, corpus, normalize=normalize)  # [Q, C]
        k = min(k, sims.shape[1])
        scores, idx = torch.topk(sims, k=k, dim=1)
        return idx, scores
