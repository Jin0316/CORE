import torch
import torch.nn.functional as F
import torch.nn as nn

class Encoder(nn.Module):
    """
    목적: 가장 단순한 사용성
      - Encoder(name=...) 로 생성 (기본적으로 CPU에서 로드)
      - 나중에 set_device()로 CUDA로 이동 가능
      - preprocess_image(x), encode_images(X), encode_texts(texts) 만 사용
    """
    def __init__(self, name: str = "clip", cache: bool = True, lazy_load: bool = True):
        super().__init__()
        self.name = name.lower()
        self.device = torch.device("cpu")  # 항상 CPU에서 시작
        self.cache_on = cache
        self.text_cache = {}

        # 모델 관련 속성들을 None으로 초기화
        self.blip2 = None
        self.vis_processors = None
        self.txt_processors = None
        self.model = None  # CLIP model
        self.preprocess = None
        self.tokenize = None
        self.tokenizer = None
        self.feat_dim = None
        
        if not lazy_load:
            self._init_backend()  # 즉시 로딩을 원하는 경우에만

    def _ensure_model_loaded(self):
        """모델이 로드되지 않았다면 로드"""
        if self._is_model_loaded():
            return
        self._init_backend()

    def _is_model_loaded(self):
        """모델이 로드되었는지 확인"""
        if self.name == "eva_clip":
            return self.blip2 is not None
        elif self.name == "clip":
            return self.model is not None
        return False

    def set_device(self, device): 
        """모델을 지정된 device로 이동"""
        self.device = torch.device(device)
        
        # 모델이 로드되지 않았다면 로드
        self._ensure_model_loaded()
        
        if hasattr(self, 'blip2') and self.blip2 is not None:
            print(f"[MODEL] Moving BLIP2 to {self.device}...")
            self.blip2 = self.blip2.to(self.device)
            
        if hasattr(self, 'model') and self.model is not None:  # CLIP의 경우
            print(f"[MODEL] Moving CLIP to {self.device}...")
            self.model = self.model.to(self.device)
            
        # 캐시된 텍스트 임베딩들도 새로운 device로 이동
        if self.cache_on and self.text_cache:
            print("[MODEL] Moving cached text embeddings...")
            for key, value in self.text_cache.items():
                if isinstance(value, torch.Tensor):
                    self.text_cache[key] = value.to(self.device)
                    
        print(f"[MODEL] Model moved to {self.device}")

    # -----------------------------
    # Public API
    # -----------------------------
    def preprocess_image(self, x):
        """
        입력: PIL.Image 또는 torch.Tensor(C,H,W)
        출력: torch.Tensor(1,C,H,W) float, 모델 전처리 완료
        """
        # 모델이 로드되지 않았다면 로드
        self._ensure_model_loaded()
        
        if self.name == "eva_clip":
            if self.vis_processors is None:
                raise RuntimeError("Cannot find EVA-CLIP.")
            proc = self.vis_processors["eval"]
            # lavis 전처리는 PIL 입력을 권장
            img = proc(x)
            if isinstance(img, torch.Tensor):
                img = img.cpu()  # 일단 CPU로
            img = img.unsqueeze(0).to(self.device)  # 현재 device로
            return img
        else:
            # clip 또는 open_clip
            if isinstance(x, torch.Tensor):
                # Tensor 입력이면 [C,H,W] 가정
                img = x.unsqueeze(0).to(self.device)
            else:
                img = self.preprocess(x).unsqueeze(0).to(self.device)
            return img
        

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        입력: [B,C,H,W]
        출력: [B,D] L2 정규화 float32
        """
        # 모델이 로드되지 않았다면 로드
        self._ensure_model_loaded()
        
        # 입력 텐서를 현재 device로 이동
        images = images.to(self.device)
        
        if self.name == "eva_clip":
            m = self.blip2
            vis = m.ln_vision(m.visual_encoder(images)) ## 여기까지가 vicuna 버전. 

            attn = torch.ones(vis.size()[:-1], dtype=torch.long, device=self.device)
            q = m.query_tokens.expand(vis.shape[0], -1, -1)
            out = m.Qformer.bert(query_embeds=q,
                                  encoder_hidden_states=vis,
                                  encoder_attention_mask=attn,
                                  return_dict=True).last_hidden_state
            feat = m.vision_proj(out).mean(dim=1)
        elif self.name == 'clip':
            feat = self.model.encode_image(images)

        feat = F.normalize(feat, dim=-1).to(torch.float32)
        return feat

    @torch.no_grad()
    def encode_texts(self, texts) -> torch.Tensor:
        """
        입력: 문자열 리스트
        출력: [N,D] L2 정규화 float32
        """
        # 모델이 로드되지 않았다면 로드
        self._ensure_model_loaded()
        
        if not texts:
            return torch.empty(0, self.feat_dim, device=self.device)

        # 캐시
        outs, miss_idx, miss_terms = [], [], []
        if self.cache_on:
            for i, t in enumerate(texts):
                v = self.text_cache.get(t)
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
            if self.name == "eva_clip":
                proc = self.txt_processors["eval"]
                proc_texts = [proc(t) for t in miss_terms]
                toks = self.blip2.tokenizer(proc_texts, padding=True, truncation=True,
                                            max_length=32, return_tensors="pt").to(self.device)
                out = self.blip2.Qformer.bert(toks.input_ids,
                                              attention_mask=toks.attention_mask,
                                              return_dict=True).last_hidden_state
                vec = self.blip2.text_proj(out).mean(dim=1)
            elif self.name == 'clip':
                toks = self.tokenize(miss_terms).to(self.device)
                vec = self.model.encode_text(toks)

            vec = F.normalize(vec, dim=-1).to(torch.float32)
            for i, v in zip(miss_idx, vec):
                outs[i] = v
                if self.cache_on:
                    # 캐시는 현재 device에 저장 (CPU든 CUDA든)
                    self.text_cache[texts[i]] = v.detach()

        return torch.stack(outs, dim=0)

    def compute_similarity(self, image_paths, texts):
        """
        이미지 경로 리스트와 텍스트 간 cosine similarity 계산
        입력:
            image_paths: str 또는 str 리스트
            texts: 문자열 리스트
        출력:
            [B, N] similarity score (0~1)
        """
        # 이미지 경로 리스트 
        if isinstance(image_paths, str) or (isinstance(image_paths, list) and isinstance(image_paths[0], str)):
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            from PIL import Image
            imgs = torch.cat([self.preprocess_image(Image.open(p).convert("RGB"))
                              for p in image_paths], dim=0)
            img_feat = self.encode_images(imgs)

        ## 임베딩 텐서 처리
        elif isinstance(image_paths, torch.Tensor):
            x = image_paths.to(self.device)
            assert x.dim() == 1 or x.dim() == 2 
            img_feat = x

        txt_feat = self.encode_texts(texts)     # [N,D]

        # similarity: [B,N]
        sim = img_feat @ txt_feat.T
        return sim

    # -----------------------------
    # Init helpers
    # -----------------------------
    def _init_backend(self):
        if self._is_model_loaded():
            print(f"[MODEL] {self.name} model already loaded, skipping...")
            return
            
        if self.name == "eva_clip":
            self._init_eva()
        elif self.name == 'clip':
            self._init_clip()

    def _init_eva(self):
        try:
            from lavis.models import load_model_and_preprocess
        except ImportError as e:
            raise ImportError("lavis 가 필요하다. pip 설치가 필요하다.") from e

        print("[MODEL] Loading EVA-CLIP on CPU...")
        
        # 강제로 CPU에서 로드
        import os
        original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 임시로 CUDA 숨기기
        
        try:
            self.blip2, self.vis_processors, self.txt_processors = load_model_and_preprocess(
                name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device = 'cpu'
            )
        finally:
            # CUDA_VISIBLE_DEVICES 원복
            if original_cuda_visible is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
            elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                del os.environ['CUDA_VISIBLE_DEVICES']

        # CPU로 명시적 이동
        self.blip2 = self.blip2.cpu()
        
        # 모든 파라미터를 CPU로 이동하고 gradient 비활성화
        for n, p in self.blip2.named_parameters():
            p.requires_grad = False
            p.data = p.data.cpu()
            
        self.feat_dim = 256  # 고정값으로 충분
        self.preprocess = None
        self.tokenize = None
        self.tokenizer = None
        
        print("[MODEL] EVA-CLIP loaded on CPU successfully")

    def _init_clip(self):
        import clip
        print("[MODEL] Loading CLIP on CPU...")
        
        model_name = "ViT-B/32"
        # CLIP을 CPU에서 로드
        self.model, self.preprocess = clip.load(model_name, device="cpu")
        self.model.eval()
        self.tokenize = clip.tokenize
        self.tokenizer = None
        self.feat_dim = 512
        
        print("[MODEL] CLIP loaded on CPU successfully")
