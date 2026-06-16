import torch
import torch.nn.functional as F
from PIL import Image

class Encoder:
    """
    목적: 가장 단순한 사용성
      - Encoder(name=...) 로 생성
      - preprocess_image(x), encode_images(X), encode_texts(texts) 만 사용
    """
    def __init__(self, name: str = "clip", device: str = None, cache: bool = True):
        self.name = name.lower()
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_on = cache
        self.text_cache = {}

        self._init_backend()

    # -----------------------------
    # Public API
    # -----------------------------
    def preprocess_image(self, x):
        """
        입력: PIL.Image 또는 torch.Tensor(C,H,W)
        출력: torch.Tensor(1,C,H,W) float, 모델 전처리 완료
        """
        if self.name == "eva_clip":
            if not hasattr(self, "vis_processors"):
                raise RuntimeError("Cannot find EVA-CLIP.")
            proc = self.vis_processors["eval"]
            # lavis 전처리는 PIL 입력을 권장
            img = proc(x).unsqueeze(0).to(self.device)
            return img
        else:
            # clip 또는 open_clip
            import PIL.Image
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
                    outs.append(v)
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
                    self.text_cache[ texts[i] ] = v

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
            assert x.dim() ==1 or x.dim() == 2 

        txt_feat = self.encode_texts(texts)     # [N,D]

        # similarity: [B,N]
        sim = img_feat @ txt_feat.T
        return sim

    # -----------------------------
    # Init helpers
    # -----------------------------
    def _init_backend(self):
        if self.name == "eva_clip":
            self._init_eva()
        elif self.name == 'clip':
            self._init_clip()

    def _init_eva(self):
        try:
            from lavis.models import load_model_and_preprocess
        except ImportError as e:
            raise ImportError("lavis 가 필요하다. pip 설치가 필요하다.") from e

        self.blip2, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name="blip2_feature_extractor", model_type="pretrain", is_eval=True
        )
        self.blip2 = self.blip2.to(self.device).eval()
        self.feat_dim = 256  # 고정값으로 충분
        # eva 경로에서는 preprocess 함수를 직접 쓰지 않는다.
        self.preprocess = None
        self.tokenize = None
        self.tokenizer = None

    def _init_clip(self):
        import clip
        model_name = "ViT-B/32"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self.tokenize = clip.tokenize
        self.tokenizer = None
        self.feat_dim = 512
