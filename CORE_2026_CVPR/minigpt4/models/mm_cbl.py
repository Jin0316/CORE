

# mm_cbl.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

def _xavier_like_(weight: torch.Tensor) -> None:
    nn.init.xavier_uniform_(weight)

def _zeros_(bias: Optional[torch.Tensor]) -> None:
    if bias is not None:
        nn.init.zeros_(bias)

def _expand_linear_out(
    old: nn.Linear,
    new_out_features: int,
    *,
    init_new: str = "xavier"
) -> nn.Linear:
    """
    out_features를 new_out_features로 확장한다.
    기존 가중치와 바이어스를 보존한다.
    새 행은 지정 초기화로 설정한다.
    """
    assert new_out_features >= old.out_features, "new_out_features must be >= old.out_features"
    device, dtype = old.weight.device, old.weight.dtype

    new = nn.Linear(old.in_features, new_out_features, bias=(old.bias is not None)).to(device=device, dtype=dtype)
    with torch.no_grad():
        # 기존 블록 복사
        new.weight[:old.out_features, :].copy_(old.weight)
        if old.bias is not None:
            new.bias[:old.out_features].copy_(old.bias)
        # 추가 블록 초기화
        if new_out_features > old.out_features:
            if init_new == "xavier":
                _xavier_like_(new.weight[old.out_features:, :])
            elif init_new == "zeros":
                nn.init.zeros_(new.weight[old.out_features:, :])
            else:
                raise ValueError(f"Unknown init_new: {init_new}")
            if new.bias is not None:
                _zeros_(new.bias[old.out_features:])
    return new

def _expand_linear_in(
    old: nn.Linear,
    new_in_features: int,
    *,
    init_new: str = "xavier"
) -> nn.Linear:
    """
    in_features를 new_in_features로 확장한다.
    기존 가중치를 좌측 블록으로 보존한다.
    추가 열을 지정 초기화로 설정한다.
    """
    assert new_in_features >= old.in_features, "new_in_features must be >= old.in_features"
    device, dtype = old.weight.device, old.weight.dtype

    new = nn.Linear(new_in_features, old.out_features, bias=(old.bias is not None)).to(device=device, dtype=dtype)
    with torch.no_grad():
        new.weight[:, :old.in_features].copy_(old.weight)
        if new_in_features > old.in_features:
            if init_new == "xavier":
                _xavier_like_(new.weight[:, old.in_features:])
            elif init_new == "zeros":
                nn.init.zeros_(new.weight[:, old.in_features:])
            else:
                raise ValueError(f"Unknown init_new: {init_new}")
        if old.bias is not None:
            new.bias.copy_(old.bias)
    return new


class DualCBLClassifier(nn.Module):
    """
    Image emb → img CBL(Linear) → img concepts
    Text  emb → txt CBL(Linear) → txt concepts
    Concat(img concepts, txt concepts) → Classifier(Linear)

    추가: 각 keyword별 학습 가능한 대표 concept embeddings
    """

    def __init__(
        self,
        image_in_dim: int,
        text_in_dim: int,
        num_img_concepts: int,
        num_txt_concepts: int,
        num_classes: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.image_in_dim = image_in_dim
        self.text_in_dim  = text_in_dim
        self._num_classes = num_classes

        self.img_cbl = nn.Linear(image_in_dim, num_img_concepts, bias=bias)
        self.txt_cbl = nn.Linear(text_in_dim,  num_txt_concepts, bias=bias)
        self.classifier = nn.Linear(num_img_concepts + num_txt_concepts, num_classes, bias=bias)
        # 새로 추가: 학습 가능한 keyword concept embeddings (원본 차원과 동일)
        self.keyword_img_concepts = nn.Parameter(torch.randn(num_classes, image_in_dim))
        self.keyword_txt_concepts = nn.Parameter(torch.randn(num_classes, text_in_dim))
        self.reset_parameters()

        # 태스크별 평균 concept scores 저장
        self.avg_img_cpt_scores = torch.tensor([])  # [num_tasks, num_img_concepts]
        self.avg_txt_cpt_scores = torch.tensor([])  # [num_tasks, num_txt_concepts]
        self.keyword_to_task = {}  # {keyword: task_id}
        # 키워드별 평균 concept scores 저장 (새로 추가)
        self.keyword_avg_img_scores = {}  # {keyword: torch.Tensor[num_img_concepts]}
        self.keyword_avg_txt_scores = {}  # {keyword: torch.Tensor[num_txt_concepts]}

    @property
    def num_img_concepts(self) -> int:
        return self.img_cbl.out_features

    @property
    def num_txt_concepts(self) -> int:
        return self.txt_cbl.out_features

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def reset_parameters(self) -> None:
        _xavier_like_(self.img_cbl.weight); _zeros_(self.img_cbl.bias)
        _xavier_like_(self.txt_cbl.weight); _zeros_(self.txt_cbl.bias)
        _xavier_like_(self.classifier.weight); _zeros_(self.classifier.bias)
        
        # 새로 추가: keyword concepts 초기화
        nn.init.xavier_uniform_(self.keyword_img_concepts)
        nn.init.xavier_uniform_(self.keyword_txt_concepts)

    def forward(
        self,
        image_emb: torch.Tensor,   # [B, image_in_dim]
        text_emb: torch.Tensor,    # [B, text_in_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_concepts = self.img_cbl(image_emb)
        txt_concepts = self.txt_cbl(text_emb)
        return img_concepts, txt_concepts

    def logits_from_concepts(
        self,
        img_concepts: torch.Tensor,
        txt_concepts: torch.Tensor,
    ) -> torch.Tensor:
        concat = torch.cat([img_concepts, txt_concepts], dim=-1)
        device = img_concepts.device
        if next(self.classifier.parameters()).device != device:
            self.classifier = self.classifier.to(device)
        return self.classifier(concat)


    def expand_concepts(
        self,
        new_num_img_concepts: Optional[int] = None,
        new_num_txt_concepts: Optional[int] = None,
        *,
        init_new: str = "xavier",
    ) -> None:
        """
        CBL 출력 차원 확장과 분류기 입력 차원 확장을 수행한다.
        기존 가중치를 보존한다.
        """
        if new_num_img_concepts is None:
            new_num_img_concepts = self.num_img_concepts
        if new_num_txt_concepts is None:
            new_num_txt_concepts = self.num_txt_concepts
        if new_num_img_concepts < self.num_img_concepts or new_num_txt_concepts < self.num_txt_concepts:
            raise ValueError("Only expansion is supported for concepts.")

        grew_img = new_num_img_concepts > self.num_img_concepts
        grew_txt = new_num_txt_concepts > self.num_txt_concepts
        if not (grew_img or grew_txt):
            return

        # CBL 확장
        if grew_img:
            self.img_cbl = _expand_linear_out(self.img_cbl, new_num_img_concepts, init_new=init_new)
        if grew_txt:
            self.txt_cbl = _expand_linear_out(self.txt_cbl, new_num_txt_concepts, init_new=init_new)

        # 분류기 입력 확장
        old_in = self.classifier.in_features
        new_in = new_num_img_concepts + new_num_txt_concepts
        if new_in > old_in:
            self.classifier = _expand_linear_in(self.classifier, new_in_features=new_in, init_new=init_new)

    def expand_classes(
        self,
        new_num_classes: int,
        *,
        init_new: str = "xavier",
    ) -> None:
        """
        분류기 출력 차원 확장을 수행한다.
        기존 클래스 가중치와 바이어스를 보존한다.
        """
        if new_num_classes < self.num_classes:
            raise ValueError("Only expansion is supported for classes.")

        if new_num_classes == self.num_classes:
            return

        self.classifier = _expand_linear_out(self.classifier, new_num_classes, init_new=init_new)
        
        # 새로 추가: keyword concepts도 확장
        device, dtype = self.keyword_img_concepts.device, self.keyword_img_concepts.dtype
        
        # 기존 keyword concepts 보존하면서 새로운 클래스용 추가
        old_img_concepts = self.keyword_img_concepts.data
        old_txt_concepts = self.keyword_txt_concepts.data
        
        new_img_concepts = torch.randn(new_num_classes, self.image_in_dim, device=device, dtype=dtype)
        new_txt_concepts = torch.randn(new_num_classes, self.text_in_dim, device=device, dtype=dtype)
        
        # 기존 부분 복사
        new_img_concepts[:self.num_classes] = old_img_concepts
        new_txt_concepts[:self.num_classes] = old_txt_concepts
        
        # 새로운 부분 초기화
        if init_new == "xavier":
            nn.init.xavier_uniform_(new_img_concepts[self.num_classes:])
            nn.init.xavier_uniform_(new_txt_concepts[self.num_classes:])
        
        self.keyword_img_concepts = nn.Parameter(new_img_concepts)
        self.keyword_txt_concepts = nn.Parameter(new_txt_concepts)
        
        self._num_classes = new_num_classes

    def compute_keyword_similarity_loss(
        self,
        image_emb: torch.Tensor,    # [B, image_in_dim]
        text_emb: torch.Tensor,     # [B, text_in_dim]
        keyword_indices: torch.Tensor,  # [B]
        temperature: float = 0.07
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        각 keyword의 대표 concept과 해당하는 이미지/텍스트 임베딩 간의 유사도 손실 계산
        """
        # 해당 keyword의 concept 가져오기
        target_img_concepts = self.keyword_img_concepts[keyword_indices]  # [B, image_in_dim]
        target_txt_concepts = self.keyword_txt_concepts[keyword_indices]  # [B, text_in_dim]
        
        # L2 정규화
        image_emb_norm = F.normalize(image_emb, p=2, dim=1)
        text_emb_norm = F.normalize(text_emb, p=2, dim=1)
        target_img_concepts_norm = F.normalize(target_img_concepts, p=2, dim=1)
        target_txt_concepts_norm = F.normalize(target_txt_concepts, p=2, dim=1)
        
        # 코사인 유사도 계산
        img_sim = torch.sum(image_emb_norm * target_img_concepts_norm, dim=1) / temperature  # [B]
        txt_sim = torch.sum(text_emb_norm * target_txt_concepts_norm, dim=1) / temperature  # [B]
        
        # 손실: 해당 keyword concept과의 유사도를 최대화
        img_loss = -torch.mean(img_sim)
        txt_loss = -torch.mean(txt_sim)
        
        return img_loss, txt_loss
    
    def compute_concept_masks(self,
        keywords: list,  # 배치의 키워드 리스트 (예: ["FamilyAbuse", "PhysicalAssault"])
        img_cpt_dict: dict,  # keyword -> image concepts 매핑
        txt_cpt_dict: dict,  # keyword -> text concepts 매핑
        img_cpt_list_ordered: list,  # 전체 이미지 concept 순서 리스트
        txt_cpt_list_ordered: list,  # 전체 텍스트 concept 순서 리스트
        device: torch.device
    ) -> tuple:
        """
        입력 키워드에 해당하는 concept만 1로, 나머지는 0으로 하는 이진 타겟 생성
        
        Args:
            keywords: 배치의 키워드 리스트
            img_cpt_dict: {keyword: [concept1, concept2, ...]} 형태
            txt_cpt_dict: {keyword: [concept1, concept2, ...]} 형태
            img_cpt_list_ordered: 모든 이미지 concept들의 순서 리스트
            txt_cpt_list_ordered: 모든 텍스트 concept들의 순서 리스트
            device: torch device
        
        Returns:
            img_binary_targets: [B, num_img_concepts] - 해당 키워드 concept만 1
            txt_binary_targets: [B, num_txt_concepts] - 해당 키워드 concept만 1
        """
        batch_size = len(keywords)
        num_img_concepts = len(img_cpt_list_ordered)
        num_txt_concepts = len(txt_cpt_list_ordered)
        
        # 이진 타겟 초기화 (모두 0)
        img_binary_targets = torch.zeros(batch_size, num_img_concepts, device=device)
        txt_binary_targets = torch.zeros(batch_size, num_txt_concepts, device=device)
        
        # 각 샘플(키워드)별로 해당하는 concept만 1로 설정
        for i, keyword in enumerate(keywords):
            # 이미지 concept 설정
            if keyword in img_cpt_dict:
                for concept in img_cpt_dict[keyword]:
                    if concept in img_cpt_list_ordered:
                        concept_idx = img_cpt_list_ordered.index(concept)
                        img_binary_targets[i, concept_idx] = 1.0
            
            # 텍스트 concept 설정
            if keyword in txt_cpt_dict:
                for concept in txt_cpt_dict[keyword]:
                    if concept in txt_cpt_list_ordered:
                        concept_idx = txt_cpt_list_ordered.index(concept)
                        txt_binary_targets[i, concept_idx] = 1.0
        
        return img_binary_targets, txt_binary_targets
    
    def freeze_keyword_concepts(self):
        """
        학습 완료 후 keyword concepts를 freeze
        """
        self.keyword_img_concepts.requires_grad = False
        self.keyword_txt_concepts.requires_grad = False

    def compute_keyword_avg_score(self, dataloader, device, img_cpt_dict, txt_cpt_dict, img_concepts_ordered, txt_concepts_ordered, logit=False):
        """
        키워드별로 개별 concept scores 평균을 계산하여 저장
        logit=True이면 해당 키워드에 속하지 않는 컨셉의 score를 0으로 마스킹
        """
        # 키워드별로 concept scores 누적
        keyword_img_sums, keyword_txt_sums, keyword_counts = {}, {}, {}
        
        with torch.no_grad():
            for batch in dataloader:
                image_feats = batch["image_embed"].to(device)
                text_feats = batch["text_embed"].to(device)
                keywords = batch["keyword"]
                
                # CBL 출력
                img_concepts = self.img_cbl(image_feats)  # [B, num_img_concepts]
                txt_concepts = self.txt_cbl(text_feats)   # [B, num_txt_concepts]
                
                for i, kw in enumerate(keywords):
                    if kw not in keyword_img_sums:
                        keyword_img_sums[kw] = torch.zeros_like(img_concepts[0])
                        keyword_txt_sums[kw] = torch.zeros_like(txt_concepts[0])
                        keyword_counts[kw] = 0
                    
                    if logit:
                        # 해당 키워드에 속하는 컨셉만 1.0, 나머지는 0.0으로 마스크 생성
                        img_mask = torch.zeros_like(img_concepts[i])
                        txt_mask = torch.zeros_like(txt_concepts[i])
                        
                        # 해당 키워드의 이미지 컨셉들만 활성화
                        for c in img_cpt_dict.get(kw, []):
                            if c in img_concepts_ordered:
                                concept_idx = img_concepts_ordered.index(c)
                                img_mask[concept_idx] = 1.0
                        
                        # 해당 키워드의 텍스트 컨셉들만 활성화
                        for c in txt_cpt_dict.get(kw, []):
                            if c in txt_concepts_ordered:
                                concept_idx = txt_concepts_ordered.index(c)
                                txt_mask[concept_idx] = 1.0
                        
                        # 마스크 적용: 해당 키워드 컨셉은 그대로, 나머지는 0
                        keyword_img_sums[kw] += img_concepts[i] * img_mask
                        keyword_txt_sums[kw] += txt_concepts[i] * txt_mask
                    else:
                        # 기존 방식: 모든 컨셉 사용
                        keyword_img_sums[kw] += img_concepts[i]
                        keyword_txt_sums[kw] += txt_concepts[i]
                    
                    keyword_counts[kw] += 1
        
        # 키워드별 평균 계산하여 저장
        self.keyword_avg_img_scores = {}
        self.keyword_avg_txt_scores = {}
        
        for kw in keyword_counts:
            if keyword_counts[kw] > 0:
                self.keyword_avg_img_scores[kw] = keyword_img_sums[kw] / keyword_counts[kw]
                self.keyword_avg_txt_scores[kw] = keyword_txt_sums[kw] / keyword_counts[kw]

    def compute_task_avg_score(self, dataloader, new_keywords, current_task_id, 
                               keywords_ordered, meta_data, device):
        for kw in new_keywords:
            self.keyword_to_task[kw] = current_task_id
        
        # 태스크별로 concept scores 누적
        task_img_sums, task_txt_sums, task_counts = {}, {}, {}
        
        with torch.no_grad():
            for batch in dataloader:
                image_feats = batch["image_embed"].to(device)
                text_feats = batch["text_embed"].to(device)
                keywords = batch["keyword"]
                
                # CBL 출력
                img_concepts = self.img_cbl(image_feats)
                txt_concepts = self.txt_cbl(text_feats)
                logits = self.logits_from_concepts(img_concepts, txt_concepts)  # [B, C]
                preds = torch.argmax(logits, dim=1)
                
                
                # 키워드별로 태스크 분류하여 누적
                for i, kw in enumerate(keywords):
                    if kw in self.keyword_to_task:
                        task_id = self.keyword_to_task[kw]
                        
                        if task_id not in task_img_sums:
                            task_img_sums[task_id] = torch.zeros_like(img_concepts[0])
                            task_txt_sums[task_id] = torch.zeros_like(txt_concepts[0])
                            task_counts[task_id] = 0
                        
                        task_img_sums[task_id] += img_concepts[i]
                        task_txt_sums[task_id] += txt_concepts[i]
                        task_counts[task_id] += 1
        
        # 태스크별 평균 계산하여 저장
        max_task_id = max(task_counts.keys()) if task_counts else -1
        new_avg_img = torch.zeros(max_task_id + 1, img_concepts.shape[1])
        new_avg_txt = torch.zeros(max_task_id + 1, txt_concepts.shape[1])
        
        for task_id in task_counts:
            if task_counts[task_id] > 0:
                new_avg_img[task_id] = task_img_sums[task_id] / task_counts[task_id]
                new_avg_txt[task_id] = task_txt_sums[task_id] / task_counts[task_id]
        
        self.avg_img_cpt_scores = new_avg_img
        self.avg_txt_cpt_scores = new_avg_txt


    def compute_inter_task_similarity(self, debug=False):
        """완료된 태스크들 간의 유사도 계산 및 출력"""
        if self.avg_img_cpt_scores.shape[0] < 2:
            return
        with torch.no_grad():
            current_idx = self.avg_img_cpt_scores.shape[0] - 1

            # 현재 태스크 벡터
            img_curr = self.avg_img_cpt_scores[current_idx]      # [D]
            txt_curr = self.avg_txt_cpt_scores[current_idx]      # [D]

            # 이전 태스크 행렬
            img_prev = self.avg_img_cpt_scores[:current_idx]     # [N, D]
            txt_prev = self.avg_txt_cpt_scores[:current_idx]     # [N, D]

            # 코사인 유사도 (1:N)
            img_cos = F.cosine_similarity(img_curr.unsqueeze(0), img_prev, dim=1)  # [N]
            txt_cos = F.cosine_similarity(txt_curr.unsqueeze(0), txt_prev, dim=1)  # [N]
            mm_sim_cos = sigmoid(img_cos, txt_cos, mu = 0.05, beta=100)
            # Weighted Jaccard 유사도 (1:N), clamp 미사용
            img_wj = jaccard_with_leaky_gate(img_curr, img_prev)          # [N]
            txt_wj = jaccard_with_leaky_gate(txt_curr, txt_prev)          # [N]
            mm_sim = sigmoid(img_wj, txt_wj) 
        
        if debug:
            print(f"[CBL] Task {current_idx} similarity with previous tasks:")
            for i, (ci, ct, m_sim_cos, ji, jt, m_sim) in enumerate(zip(img_cos, txt_cos, mm_sim_cos, img_wj, txt_wj, mm_sim)):
                print(f"[CBL]   vs Task {i}: ImgCos={ci:.3f}, TxtCos={ct:.3f} --> mm sim {m_sim_cos:.3f} | ImgWJ={ji:.3f}, TxtWJ={jt:.3f} --> mm sim {m_sim:.3f}")
        
        return img_cos, txt_cos, mm_sim_cos # img_wj, txt_wj, mm_sim 

        
    def compute_all_keyword_similarity_matrix(self, debug=False):
        """모든 키워드들 간의 유사도 매트릭스 계산"""
        keywords = list(self.keyword_avg_img_scores.keys())
        n_keywords = len(keywords)
        
        if n_keywords < 2:
            return None
        
        with torch.no_grad():
            # 모든 키워드 벡터를 행렬로 변환
            img_matrix = torch.stack([self.keyword_avg_img_scores[kw] for kw in keywords])  # [N, D]
            txt_matrix = torch.stack([self.keyword_avg_txt_scores[kw] for kw in keywords])  # [N, D]
            
            # 유사도 매트릭스 초기화
            img_cos_matrix = torch.zeros(n_keywords, n_keywords)
            txt_cos_matrix = torch.zeros(n_keywords, n_keywords)
            img_wj_matrix = torch.zeros(n_keywords, n_keywords)
            txt_wj_matrix = torch.zeros(n_keywords, n_keywords)
            mm_sim_matrix = torch.zeros(n_keywords, n_keywords)
            
            # 각 키워드별로 다른 키워드들과의 유사도 계산
            for i in range(n_keywords):
                for j in range(n_keywords):
                    if i != j:
                        # 코사인 유사도
                        img_cos_matrix[i, j] = F.cosine_similarity(
                            img_matrix[i].unsqueeze(0), img_matrix[j].unsqueeze(0), dim=1
                        )
                        txt_cos_matrix[i, j] = F.cosine_similarity(
                            txt_matrix[i].unsqueeze(0), txt_matrix[j].unsqueeze(0), dim=1
                        )
                        
                        # Weighted Jaccard 유사도
                        img_wj_matrix[i, j] = jaccard_with_leaky_gate(img_matrix[i], img_matrix[j].unsqueeze(0))
                        txt_wj_matrix[i, j] = jaccard_with_leaky_gate(txt_matrix[i], txt_matrix[j].unsqueeze(0))
                        mm_sim_matrix[i, j] = sigmoid(img_wj_matrix[i, j], txt_wj_matrix[i, j])
                    else:
                        # 자기 자신과의 유사도는 1로 설정
                        img_cos_matrix[i, j] = 1.0
                        txt_cos_matrix[i, j] = 1.0
                        img_wj_matrix[i, j] = 1.0
                        txt_wj_matrix[i, j] = 1.0
                        mm_sim_matrix[i, j] = 1.0
        
        if debug:
            print("[CBL] Keyword similarity matrix:")
            print("[CBL] Keywords:", keywords)
            print("[CBL] Multimodal similarity matrix:")
            print(mm_sim_matrix.numpy())
        
        return {
            'keywords': keywords,
            'img_cos_matrix': img_cos_matrix,
            'txt_cos_matrix': txt_cos_matrix,
            'img_wj_matrix': img_wj_matrix,
            'txt_wj_matrix': txt_wj_matrix,
            'mm_sim_matrix': mm_sim_matrix
        }
    
    
    def concept_scaling(self, image_concepts, text_concepts, keywords_ordered, img_cpt_dict, txt_cpt_dict, img_concepts_ordered, txt_concepts_ordered):
        """
        cosine similarity를 사용하여 현재 입력과 저장된 키워드들 간의 유사도 계산
        """
        
        with torch.no_grad():
            # 디바이스 맞춤
            device = image_concepts.device
            if text_concepts.device != device:
                text_concepts = text_concepts.to(device)
                
            batch_size = image_concepts.shape[0]
            
            # 전체 배치에 대한 결과 텐서 초기화
            weighted_img_scores = torch.zeros_like(image_concepts)  # [batch_size, img_concepts_dim]
            weighted_txt_scores = torch.zeros_like(text_concepts)   # [batch_size, txt_concepts_dim]
            
            for batch_idx in range(batch_size):
                # 현재 배치의 logit과 확률 계산
                logits = self.logits_from_concepts(
                    image_concepts[batch_idx:batch_idx+1], 
                    text_concepts[batch_idx:batch_idx+1]
                )  # [1, C]
                probabilities = F.softmax(logits, dim=-1)[0]  # [C]
                
                # 모든 클래스에 대해 확률만큼 가중치 적용
                for class_idx, class_prob in enumerate(probabilities):
                    if class_idx < len(keywords_ordered):
                        class_kw = keywords_ordered[class_idx]  # 클래스에 해당하는 키워드
                        
                        # 해당 클래스의 image concept들에 확률만큼 가중치 적용
                        for c in img_cpt_dict.get(class_kw, []):
                            if c in img_concepts_ordered:
                                concept_idx = img_concepts_ordered.index(c)
                                weighted_img_scores[batch_idx][concept_idx] += image_concepts[batch_idx][concept_idx] * class_prob
                                
                        # 해당 클래스의 text concept들에 확률만큼 가중치 적용
                        for c in txt_cpt_dict.get(class_kw, []):
                            if c in txt_concepts_ordered:
                                concept_idx = txt_concepts_ordered.index(c)
                                weighted_txt_scores[batch_idx][concept_idx] += text_concepts[batch_idx][concept_idx] * class_prob

        return weighted_img_scores, weighted_txt_scores

        
    @staticmethod
    def _saturate_text_scores(vec, threshold=0.5):
        """If many keywords are strongly matched (>=5 entries above 0.7), the
        averaged text signal is saturated (a batch-processing artifact): zero
        out the weak entries (< threshold) and report saturation.
        Returns (refined_vec, saturated)."""
        if torch.count_nonzero(vec > 0.7) >= 5:
            return torch.where(vec < threshold, torch.zeros_like(vec), vec), True
        return vec, False

    @staticmethod
    def _snap_img_scores_loose(vec):
        """Saturated case: if at least 2 image entries exceed 0.2, snap those
        entries to 1.0 (hard match) and keep the rest."""
        mask = vec > 0.2
        if torch.count_nonzero(mask) >= 2:
            return torch.where(mask, torch.ones_like(vec), vec)
        return vec

    @staticmethod
    def _snap_img_scores_strict(vec):
        """Non-saturated case: if exactly 2 image entries exceed 0.4, snap those
        entries to 1.0 and keep the rest."""
        mask = vec > 0.4
        if torch.count_nonzero(mask) == 2:
            return torch.where(mask, torch.ones_like(vec), vec)
        return vec

    @staticmethod
    def _suppress_img_scores(vec):
        """First two tasks: zero out every image entry below 0.95, keeping only
        near-certain matches."""
        mask = vec < 0.95
        if torch.count_nonzero(mask) >= 1:
            return torch.where(mask, torch.zeros_like(vec), vec)
        return vec

    @staticmethod
    def _amplify_weak_scores(img_vec, txt_vec):
        """If both vectors are only weakly present (their sums fall in a narrow
        low band), double the non-trivial entries (>= 0.2) so a real but weak
        match is not lost."""
        if 0.9 <= torch.sum(img_vec) <= 1.2 and 0.7 <= torch.sum(txt_vec) <= 1:
            img_vec = torch.where(img_vec >= 0.2, img_vec * 2, img_vec)
            txt_vec = torch.where(txt_vec >= 0.2, txt_vec * 2, txt_vec)
        return img_vec, txt_vec

    def _apply_score_heuristics(self, img_vec, txt_vec, predicted_task_id):
        """Heuristic post-processing on the batch-averaged scores.

        Intuition: averaging blurs per-sample peaks, so gauge confidence from
        text saturation, adapt the image threshold, snap clearly-matched
        concepts to 1.0 to sharpen the refusal decision, and amplify weak-but-
        present matches. Returns the refined (img_vec, txt_vec)."""
        txt_vec, saturated = self._saturate_text_scores(txt_vec, threshold=0.9)
        if saturated:
            img_vec = self._snap_img_scores_loose(img_vec)
        else:
            img_vec = self._snap_img_scores_strict(img_vec)
        if predicted_task_id in [0, 1]:
            img_vec = self._suppress_img_scores(img_vec)
        img_vec, txt_vec = self._amplify_weak_scores(img_vec, txt_vec)
        return img_vec, txt_vec


    def compute_inter_keyword_similarity_with_logits_cosine(self, image_concepts, text_concepts, keywords_ordered, img_cpt_dict, txt_cpt_dict, img_concepts_ordered, txt_concepts_ordered, debug=False, entropy=False):
        """
        cosine similarity를 사용하여 현재 입력과 저장된 키워드들 간의 유사도 계산
        """
        if len(self.keyword_avg_img_scores) < 1:
            return
        
        with torch.no_grad():
            # 디바이스 맞춤
            device = next(iter(self.keyword_avg_img_scores.values())).device
            if image_concepts.device != device:
                image_concepts = image_concepts.to(device)
            if text_concepts.device != device:
                text_concepts = text_concepts.to(device)
                
            batch_size = image_concepts.shape[0]
            stored_keywords = list(self.keyword_avg_img_scores.keys())
            
            # 각 배치별로 확률 가중치 적용된 유사도 계산
            batch_img_cosine = []
            batch_txt_cosine = []
            batch_mm_sim = []
            
            for batch_idx in range(batch_size):
                # 현재 배치의 logit과 확률 계산
                logits = self.logits_from_concepts(
                    image_concepts[batch_idx:batch_idx+1], 
                    text_concepts[batch_idx:batch_idx+1]
                )  # [1, C]
                probabilities = F.softmax(logits, dim=-1)[0]  # [C]
                predicted_class_idx = torch.argmax(probabilities).item()
                predicted_keyword = keywords_ordered[predicted_class_idx] if predicted_class_idx < len(keywords_ordered) else None
                predicted_task_id = self.keyword_to_task[predicted_keyword]

                # 확률 기반 가중 평균 적용
                weighted_img_scores = torch.zeros_like(image_concepts[batch_idx])
                weighted_txt_scores = torch.zeros_like(text_concepts[batch_idx])
                
                # 모든 클래스에 대해 확률만큼 가중치 적용
                for class_idx, class_prob in enumerate(probabilities):
                    if class_idx < len(keywords_ordered):
                        class_kw = keywords_ordered[class_idx]  # 클래스에 해당하는 키워드
                        
                        # 해당 클래스의 image concept들에 확률만큼 가중치 적용
                        for c in img_cpt_dict.get(class_kw, []):
                            if c in img_concepts_ordered:
                                concept_idx = img_concepts_ordered.index(c)
                                weighted_img_scores[concept_idx] += image_concepts[batch_idx][concept_idx] * class_prob
                                
                        # 해당 클래스의 text concept들에 확률만큼 가중치 적용
                        for c in txt_cpt_dict.get(class_kw, []):
                            if c in txt_concepts_ordered:
                                concept_idx = txt_concepts_ordered.index(c)
                                weighted_txt_scores[concept_idx] += text_concepts[batch_idx][concept_idx] * class_prob
                
                # 저장된 키워드들의 벡터를 모아서 행렬로 만들기
                img_stored = torch.stack([self.keyword_avg_img_scores[kw] for kw in stored_keywords])  # [N, D]
                txt_stored = torch.stack([self.keyword_avg_txt_scores[kw] for kw in stored_keywords])  # [N, D]
                
                # Cosine Similarity 계산
                img_cosine = F.cosine_similarity(
                    weighted_img_scores.unsqueeze(0),  # [1, D]
                    img_stored,  # [N, D]
                    dim=1
                )  # [N]
                
                txt_cosine = F.cosine_similarity(
                    weighted_txt_scores.unsqueeze(0),  # [1, D]
                    txt_stored,  # [N, D]
                    dim=1
                )  # [N]

                batch_img_cosine.append(img_cosine)
                batch_txt_cosine.append(txt_cosine)
                entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum().item()


            # 배치 평균 계산
            avg_img_cosine = torch.stack(batch_img_cosine).mean(dim=0)  # [N]
            avg_txt_cosine = torch.stack(batch_txt_cosine).mean(dim=0)  # [N]

            # Heuristic post-processing on the batch-averaged scores.
            avg_img_cosine, avg_txt_cosine = self._apply_score_heuristics(
                avg_img_cosine, avg_txt_cosine, predicted_task_id)

            avg_mm_sim = sigmoid(avg_img_cosine, avg_txt_cosine)

            if entropy:
                return avg_img_cosine, avg_txt_cosine, avg_mm_sim, stored_keywords, entropy
            return avg_img_cosine, avg_txt_cosine, avg_mm_sim, stored_keywords

def sigmoid(s1: torch.Tensor, s2: torch.Tensor, beta: float = 20.0, mu: float = 0.5):
    """
    s1, s2 : [N] or scalar, similarity scores in [0,1]
    beta   : steepness of sigmoid (larger -> more extreme)
    mu     : shift for controlling where "high" starts
    
    return : combined similarity in [0,1]
    """
    prod = s1 * s2
    return torch.sigmoid(beta * (prod - mu))


def jaccard_with_leaky_gate(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2, eps: float = 1e-12) -> torch.Tensor:
    """
    Weighted Jaccard 계산 후 LeakyReLU(x-0.5)+0.5 게이트 적용
    alpha: 기울기 (작을수록 0.5 이하를 더 눌러줌)
    """
    if x.dtype in [torch.float16, torch.bfloat16]:
        x = x.float()
    if y.dtype in [torch.float16, torch.bfloat16]:
        y = y.float()
    # 1) Weighted Jaccard
    x = F.softmax(x, dim=-1)
    y = F.softmax(y, dim=-1)
    x, y = x **3, y ** 3
    min_sum = torch.min(x.unsqueeze(0), y).sum(dim=1) 
    max_sum = torch.max(x.unsqueeze(0), y).sum(dim=1) 
    sim = min_sum / (max_sum + eps)
    return sim
