import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from minigpt4.common.registry import registry
from minigpt4.models.base_model import disabled_train
from minigpt4.models.mini_gpt_base import MiniGPTBase
from minigpt4.models.Qformer import BertConfig, BertLMHeadModel
from minigpt4.models.router import Router
from minigpt4.models.mm_cbl import DualCBLClassifier

import torch.nn.functional as F

from minigpt4.models.text_encoder import TextEncoder
from minigpt4.models.eva_clip import Encoder


            
@registry.register_model("mini_gpt4")
class MiniGPT4(MiniGPTBase):
    """
    MiniGPT-4 model
    """
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna0": "configs/models/minigpt4_vicuna0.yaml",
        "pretrain_llama2": "configs/models/minigpt4_llama2.yaml",
    }

    def __init__(
            self,
            vit_model="eva_clip_g",
            q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            has_qformer=True,
            freeze_qformer=True,
            num_query_token=32,
            llama_model="",
            prompt_path="",
            prompt_template="",
            max_txt_len=32,
            end_sym='\n',
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
            n_experts = 20
    ):
        super().__init__(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llama_model=llama_model,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
        )
        self.vision_device = None

        self.has_qformer = has_qformer
        if self.has_qformer:
            print('[MODEL] Loading Q-Former')
            self.Qformer, self.query_tokens = self.init_Qformer(
                num_query_token, self.visual_encoder.num_features, freeze_qformer
            )
            self.load_from_pretrained(url_or_filename=q_former_model)  # load q-former weights here

            img_f_dim = self.Qformer.config.hidden_size
            print('[MODEL] Loading Q-Former Done')
        else:
            img_f_dim = self.visual_encoder.num_features * 4
            print('[MODEL] Do not use Q-Former here.')

        self.llama_proj = nn.Linear(
            img_f_dim, self.llama_model.config.hidden_size
        )

        self.n_experts = n_experts
        self.prune_w = []
        self.experts = nn.ModuleList([nn.Linear(img_f_dim, self.llama_model.config.hidden_size) for _ in range(self.n_experts)])
        
        print('[MODEL] Initialized llama projection')

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('[MODEL] Loaded {} training prompts'.format(len(self.prompt_list)))
            print('[MODEL] Prompt example: \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

        self.cbl_model = None
        self.keywords_ordered, self.img_concepts_ordered, self.txt_concepts_ordered, self.keyword2idx = None, None, None, None

        # self.router = Router(combine_method='mean', output_dim=n_experts)
        self.router = Router(
            num_img_concepts = 10,  # Random Init
            num_txt_concepts = 10,  # Random Init
            output_dim=n_experts,   # the number of experts
        )
        print(f'[MODEL] Initialized router with {n_experts} experts')
        
        self.image_embeds = torch.tensor([])
        self.question_embeds = torch.tensor([])
        self.target_image_embeds = torch.tensor([])

        # CBL 관련 추가
        self.external_cbl = None   # 외부 CBL 저장
        self.use_external_cbl = True  # 항상 외부 CBL 우선 사용

    def text_encoder(self):
        """text_encoder 반환 (외부 우선, 내부 fallback)"""
        # 외부 CBL이 있으면 외부 것 사용
        if self.use_external_cbl and self.external_cbl:
            return self.external_cbl[0]
        else: 
            raise NotImplementedError

    def eva_clip(self):
        """eva_clip 반환 (외부 우선, 내부 fallback)"""
        # 외부 CBL이 있으면 외부 것 사용
        if self.use_external_cbl and self.external_cbl:
            return self.external_cbl[1]
        else: 
            raise NotImplementedError
    
    def set_vision_device(self, vision_device):
        self.vision_device = vision_device


    @staticmethod
    def compute_routing_gate(avg_mm_sim, avg_img_wj, avg_txt_wj,
                             sim_thresh=0.3, act_thresh=0.4, n_active=2):
        """Routing/refusal gate for one inference sample.

        Returns 1 when the input is confidently matched to a learned (forgotten)
        concept — so the expert (refusal) output is blended into the LVLM output
        — and 0 otherwise (keep the original output). A match is confident when:
          - the top multimodal similarity exceeds `sim_thresh`, or
          - at least `n_active` concepts are activated (> `act_thresh`) on the
            multimodal score, or
          - exactly `n_active` concepts are activated on both the image and the
            text scores.
        """
        if (torch.max(avg_mm_sim).item() > sim_thresh
                or torch.count_nonzero(avg_mm_sim > act_thresh) >= n_active):
            return 1
        if (torch.count_nonzero(avg_img_wj > act_thresh) == n_active
                and torch.count_nonzero(avg_txt_wj > act_thresh) == n_active):
            return 1
        return 0

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, freeze):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        
        Qformer.cls = None
        Qformer.bert.embeddings.word_embeddings = None
        Qformer.bert.embeddings.position_embeddings = None
        for layer in Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        if freeze:
            for name, param in Qformer.named_parameters():
                param.requires_grad = False
            Qformer = Qformer.eval()
            Qformer.train = disabled_train
            query_tokens.requires_grad = False
            logging.info("freeze Qformer")

        return Qformer, query_tokens
    
    def encode_img(self, image, text_samples=None, task_info = None, mode = None, routing=False, load_update = False, inference = False): 
        """
        Args:
            image 
            text_samples 
            task_info
            mode (choice) 'no_qformer', 'no_proj'
            routing (bool)
            load_update (bool)
        """
        # import pdb; pdb.set_trace()
        if routing: 
            if text_samples is None:
                raise ValueError("Routing is activated, text samples should not None.")
            loss = {}

        if self.vision_device != None:
            image = image.to(self.vision_device)
            text_samples = text_samples.to(self.vision_device)
            
        device = image.device

        if len(image.shape) > 4:
            image = image.reshape(-1, *image.shape[-3:])

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            # print(image_embeds.shape)
            if mode == 'no_qformer': 
                if len(image_embeds.shape) == 3: 
                    return image_embeds[:, 0, :]
                elif len(image_embeds.shape) == 2: 
                    return image_embeds[0, :]
                # return image_embeds.mean(dim = 1)
            
            if self.has_qformer:
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

                # print(f'task_info : {task_info}')
                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                if self.vision_device != None:
                    query_tokens = query_tokens.to(self.vision_device)
                    
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
                if mode == 'no_proj':
                    return query_output.last_hidden_state
                
                inputs_llama = self.llama_proj(query_output.last_hidden_state)     
                
                if routing == True: 
                    eva_clip = self.eva_clip()
                    txt_encoder = self.text_encoder()

                    cbl_img_embed = eva_clip.encode_images(image)
                    self.cbl_model = self.cbl_model.to(device)
                    text_samples = text_samples.to(device)

                    cbl_img_scores = self.cbl_model.img_cbl(cbl_img_embed)
                    cbl_txt_scores = self.cbl_model.txt_cbl(text_samples)

                    try:
                        if inference == False: 
                            img_wj, txt_wj, mm_sim = self.cbl_model.compute_inter_task_similarity()
                            if mm_sim is not None:
                                self.router.task_sim = mm_sim
                    except:
                        pass
                    
                    if inference == True: 
                        avg_img_wj, avg_txt_wj, avg_mm_sim, _, entropy = self.cbl_model.compute_inter_keyword_similarity_with_logits_cosine(
                            cbl_img_scores, 
                            cbl_txt_scores,
                            self.keywords_ordered,
                            self.cbl_model.img_cpt_dict,
                            self.cbl_model.txt_cpt_dict, 
                            self.img_concepts_ordered,
                            self.txt_concepts_ordered,
                            debug=False, 
                            entropy = True
                        )

                        max_value = self.compute_routing_gate(avg_mm_sim, avg_img_wj, avg_txt_wj)
                    
                    cbl_img_scores, cbl_txt_scores = self.cbl_model.concept_scaling(
                                                                    cbl_img_scores, 
                                                                    cbl_txt_scores,
                                                                    self.keywords_ordered,
                                                                    self.cbl_model.img_cpt_dict,
                                                                    self.cbl_model.txt_cpt_dict, 
                                                                    self.img_concepts_ordered,
                                                                    self.txt_concepts_ordered
                                                                    )
                    cbl_img_scores, cbl_txt_scores = cbl_img_scores.to(device), cbl_txt_scores.to(device)

                    routing_scores_ = self.router(cbl_img_scores, cbl_txt_scores)
                    routing_scores = routing_scores_.squeeze(0)  # Shape: [n_experts]
                    K = 2
                    top_scores, top_idx = torch.topk(routing_scores, K, dim=0)  # top_scores: [K], top_idx: [K]


                    top_prob = F.softmax(top_scores, dim=0)  # Shape: [K]
                    selected_expert_outputs = torch.stack([
                        self.experts[idx](query_output.last_hidden_state) for idx in top_idx
                    ], dim=0)  # Shape: [K, 8, 32, 4096]
                    
                    num_dims = selected_expert_outputs.dim()

                    # Multiply top_prob with selected_expert_outputs based on their dimensions
                    if num_dims == 2:
                        weighted_expert_outputs = top_prob.unsqueeze(1) * selected_expert_outputs  # [K, 1] * [K, hidden_size] -> [K, hidden_size]
                    elif num_dims == 3:
                        weighted_expert_outputs = top_prob.unsqueeze(1).unsqueeze(2) * selected_expert_outputs  # [K, 1, 1] * [K, batch_size, hidden_size] -> [K, batch_size, hidden_size]
                    elif num_dims == 4:
                        weighted_expert_outputs = top_prob.view(K, 1, 1, 1) * selected_expert_outputs  # [K,1,1,1] * [K, batch_size, seq_len, hidden_size] -> [K, batch_size, seq_len, hidden_size]
                    else:
                        raise ValueError(f"Unexpected number of dimensions in selected_expert_outputs: {num_dims}")

                    weighted_sum = weighted_expert_outputs.sum(dim=0)  # Shape: [8, 32, 4096]
                    
                    if inference: 
                        prob_sum = top_prob.sum()
                        inputs_llama = (inputs_llama + max_value * weighted_sum) / (max_value + 1)

                    else: 
                        inputs_llama = (inputs_llama + weighted_sum) / (2)
                    
                    loss_recom = self.router.recommendation_contrastive_loss(routing_scores_)
                    loss_reg =  self.router.load_reg_loss(routing_scores_)
                    loss['recommendation'] = loss_recom
                    loss['load_reg'] = loss_reg
                    
                    if load_update:
                        # frequency 리스트 초기화
                        frequency = [0] * self.n_experts
                        # top_idx는 텐서이므로, 각 인덱스를 추출하여 frequency 업데이트
                        for idx in top_idx:
                            frequency[idx.item()] += 1
                        # update_load 메서드 호출하여 load 업데이트
                        self.router.update_load(frequency)
            else:
                image_embeds = image_embeds[:, 1:, :]
                bs, pn, hs = image_embeds.shape
                image_embeds = image_embeds.view(bs, int(pn / 4), int(hs * 4))

                inputs_llama = self.llama_proj(image_embeds)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
            
        if routing:
            return inputs_llama, atts_llama, loss
        else:
            return inputs_llama, atts_llama, None

    @classmethod
    def from_config(cls, cfg, ckpt_path, cbl_ckpt_path, n_experts = 20,):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        has_qformer = cfg.get("has_qformer", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            has_qformer=has_qformer,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            n_experts = n_experts
        )

        print("[MODEL] Loading checkpoint: {}".format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # Router 재구성 (BLIP2-LLM 가중치 로드 전)
        if 'router' in ckpt:
            router_state_dict = ckpt['router']
            router_img_concepts = router_state_dict['img_mapping.weight'].shape[1]
            router_txt_concepts = router_state_dict['txt_mapping.weight'].shape[1]
            router_embed_dim = router_state_dict['img_mapping.weight'].shape[0]
            
            # Router 재생성
            model.router = Router(
                num_img_concepts=router_img_concepts,
                num_txt_concepts=router_txt_concepts,
                embed_dim=router_embed_dim,
                output_dim=n_experts,
            )
            print(f"[ROUTER] Rebuilt with dimensions: img={router_img_concepts}, txt={router_txt_concepts}")

        msg = model.load_state_dict(ckpt['model'], strict=False)

        # LLaMA projection freeze
        for param in model.llama_proj.parameters():
            param.requires_grad = False
        print('[MODEL] Froze LLaMA projection')
        
        # MM embeddings 로드
        try: 
            img_embeds, question_embeds = ckpt['img_embeds'], ckpt['question_embeds']
            model.image_embeds = img_embeds
            model.question_embeds = question_embeds
            model.target_image_embeds = ckpt["prune_targets"]
            print(f'[MODEL] MM embeddings are loaded sucessfully.')
        except: 
            pass 

        # Experts 로드
        try:
            # Initialize default keys
            default_weight_key = 'llama_proj.weight'
            default_bias_key = 'llama_proj.bias'

            # model.experts 
            with torch.no_grad():
                for i in range(n_experts):
                    expert = model.experts[i]
                    weight_key = f'experts.{i}.weight'
                    bias_key = f'experts.{i}.bias'

                    # Check if the specific keys exist in the checkpoint
                    if weight_key in ckpt['model'] and bias_key in ckpt['model']:
                        expert.weight.copy_(ckpt['model'][weight_key])
                        expert.bias.copy_(ckpt['model'][bias_key])
                    else:
                        # Use default keys to initialize
                        if default_weight_key in ckpt['model'] and default_bias_key in ckpt['model']:
                            expert.weight.copy_(ckpt['model'][default_weight_key])
                            expert.bias.copy_(ckpt['model'][default_bias_key])
                        else:
                            pass
            print(f'[MODEL] Loaded {n_experts} experts')

        except Exception as e:
            print(f"[MODEL] ERROR loading experts: {e}")
            
        # Prune weights 로드 (Not using this)
        try: 
            model.prune_w = ckpt["prune_weights"]
            print(f'[MODEL] Prune w loaded sucessfully.')
        except: 
            print(f'[MODEL] Prune w does not exists')
            
        
        # Router 로드 (이미 재구성되어 있으므로 직접 로드)
        if 'router' in ckpt:
            model.router.load_state_dict(ckpt['router'])
            print("[ROUTER] Loaded successfully")
        else:
            print("[ROUTER] No router in checkpoint, using initialized router")

        # Router task_sim/load 로드
        try:
            model.router.task_sim = ckpt['r_task_sim']
            model.router.load = ckpt['r_load']
            print(f'[ROUTER] Brainstorm and load matrix loaded successfully.')
        except KeyError as e:
            print(f"[ROUTER] Brainstorm score and load key error: {e}")
        except AttributeError as e:
            print(f"[ROUTER] Brainstorm score and load attribute error: {e}")
        except Exception as e: 
            print(f"[ROUTER] Brainstorm score and load are not properly loaded. An unexpected error occurred: {e}")
        else:
            pass  # 예외가 발생하지 않으면 아무 작업도 수행하지 않습니다.
        
        
        # CBL 체크포인트 로드 (가장 마지막에 실행)
        if cbl_ckpt_path:
            print(f"[CBL] Loading CBL checkpoint from: {cbl_ckpt_path}")
            
            # CBL 체크포인트 로드
            cbl_checkpoint = torch.load(cbl_ckpt_path)
            
            # 체크포인트에서 차원 정보 추출
            cbl_state_dict = cbl_checkpoint['cbl_model']
            existing_img_concepts = cbl_state_dict['img_cbl.weight'].shape[0]
            existing_txt_concepts = cbl_state_dict['txt_cbl.weight'].shape[0]
            existing_classes = cbl_state_dict['classifier.weight'].shape[0]
            existing_img_in_dim = cbl_state_dict['img_cbl.weight'].shape[1]
            existing_txt_in_dim = cbl_state_dict['txt_cbl.weight'].shape[1]
            
            print(f"[CBL] CBL dimensions - img_in: {existing_img_in_dim}, txt_in: {existing_txt_in_dim}")
            print(f"[CBL] CBL dimensions - img_concepts: {existing_img_concepts}, txt_concepts: {existing_txt_concepts}, classes: {existing_classes}")
            
            # 체크포인트 차원에 맞춰 CBL 모델 재구성
            model.cbl_model = DualCBLClassifier(
                image_in_dim=existing_img_in_dim,
                text_in_dim=existing_txt_in_dim,
                num_img_concepts=existing_img_concepts,
                num_txt_concepts=existing_txt_concepts,
                num_classes=existing_classes,
            )
            
            # CBL 가중치 로드
            model.cbl_model.load_state_dict(cbl_state_dict)
            
            # 순서 정보 저장
            model.keywords_ordered = cbl_checkpoint.get('keywords_ordered', [])
            model.img_concepts_ordered = cbl_checkpoint.get('img_concepts_ordered', [])
            model.txt_concepts_ordered = cbl_checkpoint.get('txt_concepts_ordered', [])
            model.keyword2idx = cbl_checkpoint.get('keyword2idx', {})

            model.cbl_model.avg_img_cpt_scores = cbl_checkpoint.get('avg_img_cpt_scores', torch.tensor([]))
            model.cbl_model.avg_txt_cpt_scores = cbl_checkpoint.get('avg_txt_cpt_scores', torch.tensor([]))
            model.cbl_model.keyword_avg_img_scores = cbl_checkpoint.get('keyword_avg_img_scores', {})
            model.cbl_model.keyword_avg_txt_scores = cbl_checkpoint.get('keyword_avg_txt_scores', {})
            
            model.cbl_model.keyword_to_task    = cbl_checkpoint.get('keyword_to_task', {})
            model.cbl_model.img_cpt_dict       = cbl_checkpoint.get('img_cpt_dict', {})
            model.cbl_model.txt_cpt_dict       = cbl_checkpoint.get('txt_cpt_dict', {})


            # Router도 CBL 차원에 맞게 업데이트
            try:
                # Router의 현재 차원 정보 확인
                current_img = getattr(model.router, 'num_img_concepts', model.router.img_mapping.in_features)
                current_txt = getattr(model.router, 'num_txt_concepts', model.router.txt_mapping.in_features)
                
                model.router.expand_concepts(
                    new_num_img_concepts=existing_img_concepts,
                    new_num_txt_concepts=existing_txt_concepts,
                    init_new="xavier"
                )
                print(f"[ROUTER] Expanded from ({current_img}, {current_txt}) to ({existing_img_concepts}, {existing_txt_concepts})")
            
            except Exception as e:
                print(f"[ROUTER] Could not expand to match CBL: {e}")
                # Fallback: 차원 정보만 업데이트
                model.router.num_img_concepts = existing_img_concepts
                model.router.num_txt_concepts = existing_txt_concepts
                print(f"[ROUTER] Updated dimension info only")
            
            # CBL freeze
            for param in model.cbl_model.parameters():
                param.requires_grad = False
            print("[CBL] CBL model loaded and frozen successfully")
            
            # external_cbl 설정 (eva_clip, text_encoder 필요시)
            try:
                model.external_cbl = create_external_cbl(vision_device="cuda:1")
                print("[CBL] External CBL encoders initialized")
            except Exception as e:
                print("[CBL] WARNING: could not initialize external CBL encoders")

        else:
            print("[CBL] No CBL checkpoint path provided, using randomly initialized CBL")
        
        return model
    
def create_external_cbl(vision_device="cuda:1"):
    """외부 CBL 인코더 생성"""
    text_encoder = TextEncoder(lazy_load=True)
    eva_clip = Encoder(name='eva_clip', lazy_load=True)
    
    if vision_device:
        text_encoder.set_device(vision_device)
        eva_clip.set_device(vision_device)
    
    return text_encoder, eva_clip