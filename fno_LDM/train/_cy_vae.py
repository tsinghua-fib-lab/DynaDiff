import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
from time import time
import argparse
import logging
import pickle
import os
from collections import defaultdict
from accelerate import Accelerator
import torch.multiprocessing as mp
import warnings; warnings.filterwarnings('ignore')

import sys; sys.path.append(os.getcwd())
from fno_LDM.model import FNO, FNO_GraphVAE, normalize_weights, build_fno_graph_from_structure
from fno_LDM.train.utils import set_cpu_num; set_cpu_num(1)


#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


class CustomDataset(Dataset):
    def __init__(self, feature_path, selected_combinations, zoo_size=100, cathe=False):
        self.feature_dir = feature_path
        
        self.samples, self.cond = [], []
        for idx, (Re, r) in enumerate(selected_combinations):
            for seed in range(zoo_size):
                sample_path = f"Re{Re}_r{r}/seed{seed}/epoch1000.pt"
                self.samples.append(sample_path)
                self.cond.append((Re, r))
        
        self.model_fno = FNO(
                in_channels=2,
                out_channels=2,
                n_modes=(12, 6),
                n_layers=4,
                hidden_channels=64,
            )
        self.norm_dict = pickle.load(open(f'zoo/cy_/fno/minmax_dict.pkl', 'rb'))
        
        self.cathe = cathe
        self.feature_cache = dict()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]

        Re, r = self.cond[idx]
        cond = [Re, r]
        
        if self.cathe and sample_path in self.feature_cache:
            fno_graph = self.feature_cache[sample_path]
        else:
            weights = torch.load(os.path.join(self.feature_dir, sample_path), weights_only=False, map_location='cpu')
            norm_fno = normalize_weights(self.model_fno, weights, self.norm_dict)
            fno_graph = build_fno_graph_from_structure(norm_fno)
            if self.cathe:
                self.feature_cache[sample_path] = fno_graph
        
        return (fno_graph, torch.tensor(cond, dtype=torch.float32))

def custom_collate_fn(batch):
    graph_list, cond = zip(*batch)
    ref_keys = graph_list[0].keys()
    batched_graph = defaultdict(list)

    for key in ref_keys:
        first_item = graph_list[0][key]

        if isinstance(first_item, list):
            # Handle lists of tensors (node features)
            num_layers = len(first_item)
            for layer_idx in range(num_layers):
                try:
                    tensors_to_stack = [graph[key][layer_idx] for graph in graph_list]
                    batched_graph[key].append(torch.stack(tensors_to_stack, dim=0))
                except Exception as e:
                    print(f"Error stacking list item: key='{key}', layer_idx={layer_idx}")
                    print(f" Check shapes: {[graph[key][layer_idx].shape for graph in graph_list]}")
                    raise e
        elif isinstance(first_item, torch.Tensor):
            # Handle single tensors (X_edge, edge_index, node_pos_ids, edge_pos_ids)
            try:
                if 'X_edge' in key:
                    tensors_to_stack = [graph[key] for graph in graph_list]
                    batched_graph[key] = torch.stack(tensors_to_stack, dim=0)
                else:
                    batched_graph[key] = graph_list[0][key] # same, no batch
            except Exception as e:
                print(f"Error stacking tensor item: key='{key}'")
                print(f" Check shapes: {[graph[key].shape for graph in graph_list]}")
                raise e
        else:
            print(f"Warning: Skipping key '{key}' with unhandled type {type(first_item)}.")

    cond = torch.stack(cond)
    return dict(batched_graph), cond

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    accelerator = Accelerator()
    device = accelerator.device
    
    print(f'[Loss] layer_recons_loss: {args.layer_recons_loss}')

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_dir = f"{args.results_dir}/{args.model}"  # Create an experiment folder
        experiment_dir += f"_d{args.latent_dim}"
        experiment_dir += f"_s{args.zoo_size}"
        if args.layer_recons_loss:
            experiment_dir += "_layer_recons_loss"
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

    # Create model:
    model_fno = FNO(
        in_channels=2,
        out_channels=2,
        n_modes=(12, 6),
        n_layers=4,
        hidden_channels=64,
    )
    # --- VAE Hyperparameters ---
    internal_dim = 1024     # Common internal dimension (D)
    latent_dim = args.latent_dim        # Latent dimension (h)
    num_heads = 8           # Attention heads (internal_dim must be divisible by num_heads)
    num_attn_layers = 4     # Renamed from num_gnn_layers
    dropout_rate = 0.1      # Dropout for Attention/FFN
    kl_weight = 1e-5
    model = FNO_GraphVAE(
        fno_model=model_fno,
        internal_dim=internal_dim,
        latent_dim=latent_dim,
        num_heads=num_heads,
        num_attn_layers=num_attn_layers,
        dropout=dropout_rate,
        kl_weight=kl_weight,
        layer_recons_loss=args.layer_recons_loss,
    ).to(device)
    if accelerator.is_main_process:
        logger.info(f"VAE Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    # Setup data:
    Re_values = np.linspace(200, 500, 31)
    r_values = np.linspace(10, 25, 16)
    np.random.seed(1)
    Re_grid, r_grid = np.meshgrid(Re_values, r_values)
    Re_grid_flat = Re_grid.flatten()
    r_grid_flat = r_grid.flatten()
    total_combinations = Re_grid_flat.size
    random_indices = np.random.choice(total_combinations, 96, replace=False)
    selected_Re_values = Re_grid_flat[random_indices]
    selected_r_values = r_grid_flat[random_indices]
    selected_combinations = list(zip(selected_Re_values, selected_r_values))
    
    dataset = CustomDataset(args.feature_path, selected_combinations, zoo_size=args.zoo_size, cathe=True)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} samples ({args.feature_path})")

    # Prepare models for training:
    model.train()
    model, opt, loader = accelerator.prepare(model, opt, loader)
    
    gamma = (1 / 10) ** (1 / args.epochs)
    scheduler = ExponentialLR(opt, gamma=gamma)
    
    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    start_time = time()
    
    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")

        for fno_graph, _ in loader:
            X_nodes_input = {
                'lift': [t.clone().detach() for t in fno_graph['X_lift_node']], # Use clones
                'block': [t.clone().detach() for t in fno_graph['X_block_node']],
                'proj': [t.clone().detach() for t in fno_graph['X_proj_node']]
            }

            if isinstance(model, DDP):
                total_loss, recon_loss, kl_loss = model.module.training_losses(X_nodes_input)
            else:
                total_loss, recon_loss, kl_loss = model.training_losses(X_nodes_input)
            loss = total_loss
            
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()

            # Log loss values:
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Recons Loss: {recon_loss:.4f}, KL Loss: {kl_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f} | lr: {opt.param_groups[0]['lr']:.2e} | ")

                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if (train_steps % args.ckpt_every == 0 and train_steps > 0):
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/vae.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        scheduler.step()
    
    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 fno_LDM/train/_cy_vae.py 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, default="zoo/cy_/fno/origin/")
    parser.add_argument("--results-dir", type=str, default="log/cy_/fno_ldm")
    parser.add_argument("--model", type=str, default="GraphVAE")
    parser.add_argument("--layer_recons_loss", action="store_true", default=False)
    parser.add_argument("--zoo_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--ckpt-every", type=int, default=2000)
    args = parser.parse_args()
    main(args)
