import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from typing import List, Union, Tuple, Optional

class TextEncoder:
    """
    목적: 가장 단순한 텍스트-텍스트 유사도 계산
      - Text2TextSim(device=..., cache=True)
      - encode_texts(texts) -> [N, D] L2 정규화 임베딩
      - similarity(texts_a, texts_b, normalize=None) -> [A, B] cosine 유사도
    """
    def __init__(self, device: Optional[str] = None, cache: bool = True):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer("all-mpnet-base-v2", device=str(self.device))
        self.cache_on = cache
        self._cache = {}

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        """
        입력: 문자열 리스트
        출력: [N, D] L2 정규화 임베딩 (float32)
        """
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
                    outs.append(v)
        else:
            miss_idx = list(range(len(texts)))
            miss_terms = texts
            outs = [None] * len(texts)

        # 미스 항목 인코딩
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
                    self._cache[term] = v

        return torch.stack(outs, dim=0)  # [N, D]

    @torch.no_grad()
    def compute_similarity(
        self,
        texts_a: Union[str, List[str], torch.Tensor],
        texts_b: Union[str, List[str]],
        normalize: Optional[str] = None,  # None | "softmax" | "cube_softmax"
    ) -> torch.Tensor:
        """
        입력:
            texts_a: str | List[str] | torch.Tensor([A,D] 또는 [D])
            texts_b: str | List[str]
            normalize:
                - None: 원시 cosine 점수
                - "softmax": 각 행 합 1
                - "cube_softmax": 점수^3 후 softmax
        출력:
            [A, B] 유사도 행렬 (float32)
        """
        # A 임베딩 준비
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

