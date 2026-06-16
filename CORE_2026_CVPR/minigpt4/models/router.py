import torch
import torch.nn as nn
import torch.nn.functional as F

class Router(nn.Module):
    def __init__(self, 
                 num_img_concepts,      # CBL의 img concepts 차원
                 num_txt_concepts,      # CBL의 txt concepts 차원
                 embed_dim=256,        # 가벼운 내부 차원
                 output_dim=20,        # 전문가 수
                 num_heads=4,          # 가벼운 헤드 수
                 dropout=0.1,
                 tau=0.5):
        """
        CBL concepts을 입력으로 받는 가벼운 Router 모델
        
        Args:
            img_concept_dim (int): CBL의 image concepts 차원
            txt_concept_dim (int): CBL의 text concepts 차원
            embed_dim (int): 가벼운 내부 임베딩 차원
            output_dim (int): 출력 차원 (전문가 수)
            num_heads (int): MHSA 헤드 수
            dropout (float): 드롭아웃 비율
            tau (float): contrastive loss 온도 파라미터
        """
        super(Router, self).__init__()
        
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.tau = tau

        self.num_img_concepts = num_img_concepts
        self.num_txt_concepts = num_txt_concepts
        
        # CBL concepts을 공통 차원으로 매핑 (가벼운 단일 레이어)
        self.img_mapping = nn.Linear(num_img_concepts, embed_dim)
        self.txt_mapping = nn.Linear(num_txt_concepts, embed_dim)
        
        # 2개의 MHSA 레이어
        self.self_attention1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=False)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        
        self.self_attention2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=False)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        # 최종 출력 레이어 (단일 레이어)
        self.output_mapping = nn.Linear(embed_dim, output_dim)
        
        # Load tracking
        self.load = []
        self.task_sim = []
        
    def forward(self, img_concepts, txt_concepts):
        """
        Args:
            img_concepts (Tensor): CBL의 image concepts [batch_size, num_img_concepts]
            txt_concepts (Tensor): CBL의 text concepts [batch_size, num_txt_concepts]
        
        Returns:
            Tensor: [1, output_dim]
        """
        device = img_concepts.device
        
        # 배치를 단일로 평균화
        if img_concepts.dim() > 2:
            img_concepts = img_concepts.mean(dim=0).mean(dim=0).unsqueeze(0)
        elif img_concepts.size(0) > 1:
            img_concepts = img_concepts.mean(dim=0, keepdim=True)
            
        if txt_concepts.dim() > 2:
            txt_concepts = txt_concepts.mean(dim=0).mean(dim=0).unsqueeze(0)
        elif txt_concepts.size(0) > 1:
            txt_concepts = txt_concepts.mean(dim=0, keepdim=True)
        
        # 1. CBL concepts을 공통 차원으로 매핑
        mapped_img = self.img_mapping(img_concepts)  # [1, embed_dim]
        mapped_txt = self.txt_mapping(txt_concepts)  # [1, embed_dim]
        
        # 2. MHSA를 위해 시퀀스 차원으로 스택 [seq_len=2, batch_size=1, embed_dim]
        combined = torch.stack([mapped_img, mapped_txt], dim=0)  # [2, 1, embed_dim]
        
        # 3. 첫 번째 Self-Attention + 잔차 연결
        attn_output1, _ = self.self_attention1(combined, combined, combined)
        combined = self.layer_norm1(combined + attn_output1)
        
        # 4. 두 번째 Self-Attention + 잔차 연결
        attn_output2, _ = self.self_attention2(combined, combined, combined)
        combined = self.layer_norm2(combined + attn_output2)
        
        # 5. 시퀀스 차원을 평균으로 결합
        combined_final = combined.mean(dim=0)  # [1, embed_dim]
        
        # 6. 최종 출력
        output = self.output_mapping(combined_final)  # [1, output_dim]
        
        return output

    def forward_batch(self, img_concepts, txt_concepts):

        batch_size = img_concepts.size(0)
        
        mapped_img = self.img_mapping(img_concepts)  # [batch_size, embed_dim]
        mapped_txt = self.txt_mapping(txt_concepts)  # [batch_size, embed_dim]
        combined = torch.stack([mapped_img, mapped_txt], dim=0)  # [2, batch_size, embed_dim]
        
        attn_output1, _ = self.self_attention1(combined, combined, combined)
        combined = self.layer_norm1(combined + attn_output1)

        attn_output2, _ = self.self_attention2(combined, combined, combined)
        combined = self.layer_norm2(combined + attn_output2)

        combined_final = combined.mean(dim=0)  # [batch_size, embed_dim]

        output = self.output_mapping(combined_final)  # [batch_size, output_dim]
        
        return output


    def init_load(self):
        self.load.append([0 for _ in range(self.output_dim)])
        print(f'[ROUTER] Load matrix added with initialized load. {len(self.load)}, {len(self.load[-1])}')

    def update_load(self, frequency: list):
        if not self.load:
            raise ValueError("Load matrix is empty. Call init_load() first.")
        self.load[-1] = [current + freq for current, freq in zip(self.load[-1], frequency)]

    
    def recommendation_contrastive_loss(self, x):
        device = x.device
        task_sim = self.task_sim
        
        # task_sim 검증
        if isinstance(task_sim, torch.Tensor):
            task_sim = task_sim.detach().clone().to(device)
            if task_sim.numel() == 0:
                return torch.zeros(1, device=device, requires_grad=True)
        elif isinstance(task_sim, list):
            if len(task_sim) == 0:
                return torch.zeros(1, device=device, requires_grad=True)
            task_sim = torch.tensor(task_sim, dtype=x.dtype, device=device).unsqueeze(0)
        else:
            return torch.zeros(1, device=device, requires_grad=True)
        
        # load 검증
        if len(self.load) < 1:
            return torch.zeros(1, device=device, requires_grad=True)
        
        previous_loads = torch.tensor(self.load[:-1], dtype=x.dtype, device=device)
        if previous_loads.size(0) == 0:
            return torch.zeros(1, device=device, requires_grad=True)
        
        load_norm = F.normalize(previous_loads, p=2, dim=-1)
        x_norm = F.normalize(x, p=2, dim=-1)
        
        positive_similarity = torch.exp(torch.matmul(x_norm, load_norm.T) / self.tau)
        negative_similarities = positive_similarity.sum()
        contrastive_loss = positive_similarity / (negative_similarities + 1e-8)
        
        epsilon = 1e-12
        loss = -(task_sim * torch.log(contrastive_loss + epsilon)).mean()
        
        return loss
    
    def load_reg_loss(self, x, alpha: float = 6.0, gamma: float = 6.0, beta: float = 1.0, eps: float = 1e-8):
        """
        alpha : scale for cosine + CE terms
        gamma : scale for discouragement target sharpness
        beta  : scale for entropy penalty (encourage low entropy)
        """
        device, dtype = x.device, x.dtype
        if not getattr(self, "load", None) or len(self.load) <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # task_sim → weights
        b = self.task_sim
        if isinstance(b, torch.Tensor):
            b = b.detach().to(device=device, dtype=dtype)
            if b.numel() == 0:
                return torch.tensor(0.0, device=device, requires_grad=True)
        elif isinstance(b, list):
            if len(b) == 0:
                return torch.tensor(0.0, device=device, requires_grad=True)
            b = torch.tensor(b, device=device, dtype=dtype)
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)

        load_tensor = torch.tensor(self.load[:-1], device=device, dtype=dtype)  # [T, E]
        prev = torch.softmax(load_tensor, dim=1)

        # Build a discouragement target q_t ∝ softmax(gamma * (1 - prev_t))
        q = torch.softmax(gamma * (1.0 - prev), dim=1)  # [T, E]

        p = torch.softmax(x, dim=1)  # [1, E]
        log_p = torch.log(p + eps)   # [1, E]

        # Cross-entropy per task: H(q_t, p) = − Σ q_t,e log p_e
        ce = -(q * log_p.expand_as(q)).sum(dim=1)  # [T]

        # Cosine regularizer
        cos = F.cosine_similarity(p.expand_as(load_tensor), load_tensor, dim=1)

        # Task weights
        w = (1.0 - b)
        w = w / (w.sum() + eps)

        loss = (w * (cos + 1)).sum()
        loss +=  0.1 * (w * ce).sum()
        return loss


    def _expand_linear_layer(self, old_layer, new_in_features=None, init_new="xavier"):
        """
        Linear layer의 입력 차원을 확장하는 함수
        
        Args:
            old_layer (nn.Linear): 확장할 기존 layer
            new_in_features (int): 새로운 입력 차원
            init_new (str): 새로운 가중치 초기화 방법
            
        Returns:
            nn.Linear: 확장된 새로운 layer
        """
        if new_in_features is None or new_in_features == old_layer.in_features:
            return old_layer
        
        if new_in_features < old_layer.in_features:
            raise ValueError("Only expansion supported")
        
        # 새로운 layer 생성
        new_layer = nn.Linear(new_in_features, old_layer.out_features, 
                            bias=(old_layer.bias is not None))
        
        with torch.no_grad():
            # 기존 가중치 복사
            new_layer.weight.data[:, :old_layer.in_features] = old_layer.weight.data
            
            # 새로운 부분 초기화
            if new_in_features > old_layer.in_features:
                new_weights = new_layer.weight.data[:, old_layer.in_features:]
                if init_new == "xavier":
                    nn.init.xavier_uniform_(new_weights)
                elif init_new == "zeros":
                    nn.init.zeros_(new_weights)
            
            # 바이어스 복사
            if old_layer.bias is not None:
                new_layer.bias.data = old_layer.bias.data
        
        return new_layer

    def expand_concepts(self, new_num_img_concepts=None, new_num_txt_concepts=None, init_new="xavier"):
        """
        Router의 concept 차원을 확장
        
        Args:
            new_num_img_concepts (int, optional): 새로운 이미지 concept 수
            new_num_txt_concepts (int, optional): 새로운 텍스트 concept 수  
            init_new (str): 새로운 가중치 초기화 방법
        """
        # 현재 차원 정보 (Router __init__에서 저장해야 함)
        current_img = getattr(self, 'num_img_concepts', self.img_mapping.in_features)
        current_txt = getattr(self, 'num_txt_concepts', self.txt_mapping.in_features)
        
        if new_num_img_concepts is None:
            new_num_img_concepts = current_img
        if new_num_txt_concepts is None:
            new_num_txt_concepts = current_txt
        
        # Image mapping 확장
        if new_num_img_concepts > current_img:
            print(f"[ROUTER] Expanding img_mapping: {current_img} -> {new_num_img_concepts}")
            self.img_mapping = self._expand_linear_layer(
                self.img_mapping, new_num_img_concepts, init_new
            )
        
        # Text mapping 확장  
        if new_num_txt_concepts > current_txt:
            print(f"[ROUTER] Expanding txt_mapping: {current_txt} -> {new_num_txt_concepts}")
            self.txt_mapping = self._expand_linear_layer(
                self.txt_mapping, new_num_txt_concepts, init_new
            )
        
        # 차원 정보 저장
        self.num_img_concepts = new_num_img_concepts
        self.num_txt_concepts = new_num_txt_concepts
        
        print(f"[ROUTER] Expansion completed - img: {new_num_img_concepts}, txt: {new_num_txt_concepts}")


    def get_frozen_expert_indices(self):
        """각 태스크별 가장 많이 사용된 expert index들을 반환"""
        if len(self.load) <= 1:
            return []
            
        frozen_indices = []
        
        # 완료된 태스크들만 고려 (마지막 제외)
        for task_load in self.load[:-1]:
            max_expert_id = task_load.index(max(task_load))
            if max_expert_id not in frozen_indices:
                frozen_indices.append(max_expert_id)
        
        return frozen_indices
    