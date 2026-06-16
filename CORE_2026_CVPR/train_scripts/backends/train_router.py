# train_router.py
import torch 
import torch.optim as optim
import torch.nn.functional as F

import os, sys
# this backend lives in train_scripts/backends/; put repo root on sys.path so
# `minigpt4` / `utils_router` resolve when launched as `python train_scripts/backends/...`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from minigpt4.models.router import Router
from minigpt4.models.mm_cbl import DualCBLClassifier
from utils.router.mm_unlearn_embed_loader import make_dataloader
import os 
import random
import numpy as np 
import torch.backends.cudnn as cudnn
from collections import defaultdict

class ReplayRouter():
    def __init__(self, cbl_ckpt_path, router_ckpt_path, train_config, n_expert=20, device=None):
        """
        Initializes the ReplayRouter with CBL and router checkpoint paths.

        Args:
            cbl_ckpt_path (str): Path to the CBL checkpoint file.
            router_ckpt_path (str): Path to the router checkpoint file (can be None for first step).
            train_config (dict): Training configuration containing 'train_epoch' and 'lr'.
            n_expert (int, optional): Number of experts in the router. Defaults to 20.
            device (str, optional): Device to run the model on ('cpu' or 'cuda'). Defaults to None.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cbl_ckpt_path = cbl_ckpt_path
        self.router_ckpt_path = router_ckpt_path
        self.train_config = train_config
        self.criterion = torch.nn.CrossEntropyLoss()
        self.n_expert = n_expert
        self.keyword_true_labels = defaultdict(list)
        self.keyword_task_mapping = {}

        # Load CBL model first
        self.load_cbl_model()
        
        # Initialize or load router
        self.load_or_create_router()
        
        self.setup_optimizer()
        self.check_updatable_params()
        
    def load_cbl_model(self):
        """Load CBL model from checkpoint and freeze it"""
        print(f"[ROUTER] Loading CBL model from {self.cbl_ckpt_path}")
        cbl_checkpoint = torch.load(self.cbl_ckpt_path, map_location='cpu')
        
        # Get model configuration from checkpoint
        cbl_state_dict = cbl_checkpoint['cbl_model']
        num_img_concepts = cbl_state_dict['img_cbl.weight'].shape[0]
        num_txt_concepts = cbl_state_dict['txt_cbl.weight'].shape[0]
        num_classes = cbl_state_dict['classifier.weight'].shape[0]
        img_in_dim = cbl_state_dict['img_cbl.weight'].shape[1]
        txt_in_dim = cbl_state_dict['txt_cbl.weight'].shape[1]
        
        # Create and load CBL model
        self.cbl_model = DualCBLClassifier(
            image_in_dim=img_in_dim,
            text_in_dim=txt_in_dim,
            num_img_concepts=num_img_concepts,
            num_txt_concepts=num_txt_concepts,
            num_classes=num_classes,
        )
        
        self.cbl_model.load_state_dict(cbl_state_dict)
        self.cbl_model.to(self.device)
        self.cbl_model.eval()  # Set to eval mode
        
        # Freeze CBL model
        for param in self.cbl_model.parameters():
            param.requires_grad = False
        
        self.keywords_ordered = cbl_checkpoint['keywords_ordered']
        self.img_concepts_ordered = cbl_checkpoint['img_concepts_ordered']
        self.txt_concepts_ordered = cbl_checkpoint['txt_concepts_ordered']
        self.img_cpt_dict =  cbl_checkpoint['img_cpt_dict']
        self.txt_cpt_dict = cbl_checkpoint['txt_cpt_dict']

        print(f"[ROUTER] CBL model loaded and frozen - img_concepts: {num_img_concepts}, txt_concepts: {num_txt_concepts}")
        
    def load_or_create_router(self):
        """Load existing router or create new one"""
        

        print(f"[ROUTER] Loading router from {self.router_ckpt_path}")
        router_checkpoint = torch.load(self.router_ckpt_path, map_location='cpu')
        
        if 'router_model' in router_checkpoint:
            router_state_dict = router_checkpoint['router_model']
        elif 'router' in router_checkpoint:
            router_state_dict = router_checkpoint['router']
        else:
            raise KeyError("Router state dict not found in checkpoint")
        
        # Get previous dimensions from state dict
        prev_img_concepts = router_state_dict['img_mapping.weight'].shape[1]
        prev_txt_concepts = router_state_dict['txt_mapping.weight'].shape[1]
        router_embed_dim = router_state_dict['img_mapping.weight'].shape[0]

        self.router = Router(
            num_img_concepts=prev_img_concepts,
            num_txt_concepts=prev_txt_concepts,
            embed_dim=router_embed_dim,
            output_dim=self.n_expert,
        )
        
        # Load state dict
        self.router.load_state_dict(router_state_dict)
        self.router.load = router_checkpoint['r_load']
        self.router.task_sim = router_checkpoint['r_task_sim']
        print(f'[ROUTER] task_sim : {self.router.task_sim}')
        print(f'[ROUTER]       Load : {self.router.load}')
        self.router.to(self.device)
        


        print('[ROUTER] Router successfully loaded/created.')
        
    def find_task_for_keyword(self, keyword):
        # Load keyword_to_task mapping from CBL checkpoint
        cbl_checkpoint = torch.load(self.cbl_ckpt_path, map_location='cpu')
        keyword_to_task = cbl_checkpoint.get('keyword_to_task', {})
        return keyword_to_task[keyword]
        
    def setup_optimizer(self):
        """Sets up the optimizer based on the training configuration."""
        lr = self.train_config['lr']
        self.optimizer = optim.Adam(self.router.parameters(), lr=lr)
        print(f'[ROUTER] Optimizer initialized with learning rate: {lr}')
        
    def check_updatable_params(self):
        """Checks and prints the number of updatable (trainable) and frozen parameters in the router."""
        total_params = 0
        updatable_params = 0
        freezed_params = 0
        freezed_param_names = []

        for name, param in self.router.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                updatable_params += num_params
            else:
                freezed_params += num_params
                freezed_param_names.append(name)

        print(f'[ROUTER] Updatable parameters: {updatable_params/1e+6:.3f}M/{total_params/1e+6:.3f}M '
              f'({updatable_params/total_params:.2%})')
        print(f'[ROUTER] Frozen parameters: {freezed_params/1e+6:.3f}M/{total_params/1e+6:.3f}M '
              f'({freezed_params/total_params:.2%})')
        if freezed_param_names:
            print(f'[ROUTER] Frozen params: {len(freezed_param_names)} tensors')
    
    def train(self, train_loader):
        """
        Trains the router using the provided training data loader.

        Args:
            train_loader (DataLoader): DataLoader for training data.
        """
        train_epochs = self.train_config['train_epoch']
        self.router.train()
        self.cbl_model.eval()  # Keep CBL in eval mode
        
        for epoch in range(1, train_epochs + 1):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, batch in enumerate(train_loader):
                # Get embeddings from batch
                image_feats = batch["image_embed"].to(self.device)    # [B, D_img]
                text_feats = batch["text_embed"].to(self.device)      # [B, D_txt]
                kws = batch["keyword"]                                # List[str]
                # print(f'{batch_idx}/{len(train_loader)}: kws:{kws}')
                if len(train_loader) > 40 and batch_idx == 30: 
                    break
                with torch.no_grad():
                    # Get CBL concepts
                    img_concepts = self.cbl_model.img_cbl(image_feats)    # [B, N_img]
                    txt_concepts = self.cbl_model.txt_cbl(text_feats)     # [B, N_txt]
                    img_concepts, txt_concepts = self.cbl_model.concept_scaling(img_concepts, txt_concepts, 
                                                                                self.keywords_ordered, 
                                                                                self.img_cpt_dict, 
                                                                                self.txt_cpt_dict, 
                                                                                self.img_concepts_ordered, 
                                                                                self.txt_concepts_ordered)
                # Create labels based on task load for each keyword
                batch_size = len(kws)
                labels = torch.zeros(batch_size, self.n_expert, dtype=torch.float, device=self.device)
                
                for i, kw in enumerate(kws):
                    task_id = self.find_task_for_keyword(kw)
                    task_load = self.router.load[task_id]
                    labels[i] = torch.tensor(task_load, dtype=torch.float, device=self.device)
                    # 키워드별 정보 저장
                    self.keyword_task_mapping[kw] = task_id
                labels = F.softmax(labels, dim=-1)
                ### [Soft label: Labels are sampled based on probability.]
                true_labels = torch.multinomial(labels, num_samples=1).squeeze(-1)  # Sample one label per instance
                for i, kw in enumerate(kws):
                    true_label = true_labels[i].item()
                    self.keyword_true_labels[kw].append(true_label)

                # Forward pass through router
                outputs = self.router.forward_batch(img_concepts, txt_concepts)
                loss = self.criterion(outputs, true_labels)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                
                # Update total and correct counts
                total += labels.size(0)
                correct += (predicted == true_labels).sum().item()

            avg_loss = epoch_loss / len(train_loader)
            accuracy = 100 * correct / total
            if epoch % 3 == 0: 
                print(f'[ROUTER] Epoch [{epoch}/{train_epochs}] Average Loss: {avg_loss:.4f}, '
                      f'Accuracy: {accuracy:.2f}%')
        self.print_training_summary()
        
    def save_router(self):
        """
        Saves only the router's state dictionary to the checkpoint.
        """
        if not os.path.isfile(self.router_ckpt_path):
            raise ValueError(f"[ERROR] No existing checkpoint found at {self.router_ckpt_path}. Cannot save 'router' without an existing checkpoint.")

        # Update the 'router' key with the current router's state dict
        checkpoint = torch.load(self.router_ckpt_path, map_location='cpu')
        checkpoint['router'] = self.router.state_dict()
        torch.save(checkpoint, self.router_ckpt_path)
        print(f'[ROUTER] Replay Step: Router successfully saved to {self.router_ckpt_path}.')

    # train 메서드의 마지막에 추가할 로그 출력 코드
    def print_training_summary(self):
        """학습 완료 후 keyword별 true_label과 load 정보 출력"""
        print("[ROUTER] Training summary - load per task:")
        for task_id, load_dist in enumerate(self.router.load):
            print(f"[ROUTER]   task {task_id:2d}: {load_dist}")

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Router Training")

    parser.add_argument("--cbl-ckpt-path", required=True, help="path to CBL checkpoint file.")
    parser.add_argument("--router-ckpt-path", default=None, help="path to previous router checkpoint (optional for first step).")
    parser.add_argument("--device", required=True, help="GPU id for training")
    parser.add_argument("--time_step", required=True, help="time step index, zero indexing")
    
    # parser.add_argument("--n_sample_per_keyword", type=int, default=10, help="Number of experts")
    parser.add_argument("--n_expert", type=int, default=20, help="Number of experts")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    return args

def setup_seeds():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    
def main():
    args = parse_args()
    cbl_ckpt_path = args.cbl_ckpt_path
    router_ckpt_path = args.router_ckpt_path
    time_step = args.time_step
    device = args.device
    
    print(f'[ROUTER] Current Time step: {time_step} (zero indexing)')
    setup_seeds()
    train_config = {'train_epoch': args.epochs, 'lr': args.lr}
    
    checkpoint = torch.load(cbl_ckpt_path, map_location='cpu')
    all_keywords = checkpoint.get('keywords_ordered', [])
    print(f'[ROUTER] Training with keywords: {all_keywords}')

    # Create ReplayRouter
    replay_router = ReplayRouter(
        cbl_ckpt_path=cbl_ckpt_path,
        router_ckpt_path=router_ckpt_path,
        train_config=train_config,
        n_expert=args.n_expert,
        device=device
    )
    
    # For subsequent steps, use all as prev_keyword for sampling
    train_data_loader = make_dataloader(
        pt_path="utils/CBL/samples_with_embeddings.pt",
        prev_keyword=all_keywords,
        strict=False,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        n_samples=10
    )
    
    # Train router
    replay_router.train(train_data_loader)
    replay_router.save_router()

if __name__ == '__main__':
    main()

