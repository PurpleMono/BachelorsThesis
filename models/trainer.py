"""
models/trainer.py — Unified training and inference functions for all three models.

Each model has its own training function but they share:
- The same data loading interface (RealIADTorchDataset)
- The same inference interface (returns standardised dataframe)
- The same preprocessing (448→392 centre crop, ImageNet normalisation)

Usage:
    from models.trainer import train_dinomaly, train_anomalydino, run_inference

    # Standard protocol
    model = train_dinomaly(train_df, n_iterations=50000, device='cuda')
    results_df = run_inference(model, test_df, model_name='Dinomaly', device='cuda')

    # Cross-view protocol — just filter the dataframe
    train_cv_df = train_df[train_df['viewpoint'].isin(['C1', 'C2'])]
    model = train_dinomaly(train_cv_df, n_iterations=50000, device='cuda')
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import importlib.util
import sys
import os


def _load_dataset_class(repo_path: str):
    """Load RealIADTorchDataset from repo path."""
    spec = importlib.util.spec_from_file_location(
        "realiad_dataset",
        f"{repo_path}/data/realiad_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RealIADTorchDataset


# =============================================================================
# SECTION 1: TRAINING FUNCTIONS
# =============================================================================

def train_dinomaly(
    train_df: pd.DataFrame,
    n_iterations: int = 50000,
    batch_size: int = 16,
    lr: float = 2e-3,
    device: str = 'cuda',
    repo_path: str = '/content/drive/MyDrive/BachelorsThesis',
    save_path: str = None,
) -> object:
    """
    Train Dinomaly on normal images from train_df.
    Uses Anomalib's Dinomaly model with a custom training loop.
    Follows published paper defaults: StableAdamW, lr=2e-3, 50k iterations.

    Args:
        train_df:      dataframe from realiad_utils — normal images only
        n_iterations:  total training iterations (default: 50000 per paper)
        batch_size:    training batch size (default: 16 per paper)
        lr:            learning rate (default: 2e-3 per paper)
        device:        'cuda' or 'cpu'
        repo_path:     path to BachelorsThesis repo
        save_path:     optional path to save model weights

    Returns:
        Trained Dinomaly model ready for inference
    """
    from anomalib.models import Dinomaly
    from anomalib.models.image.dinomaly.torch_model import DinomalyModel

    RealIADTorchDataset = _load_dataset_class(repo_path)

    # Only normal images for training
    normal_df = train_df[train_df['label'] == 0].reset_index(drop=True)
    print(f"Training Dinomaly on {len(normal_df)} normal images")

    dataset = RealIADTorchDataset(normal_df, load_masks=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    # Build model
    model = Dinomaly()
    torch_model = model.model.to(device)
    torch_model.train()

    # Identify trainable parameters (decoder and bottleneck only — encoder is frozen)
    trainable_params = [
        p for name, p in torch_model.named_parameters()
        if 'encoder' not in name
    ]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params) / 1e6:.1f}M")

    # StableAdamW optimizer — matches paper defaults
    try:
        from anomalib.models.image.dinomaly.torch_model import StableAdamW
        optimizer = StableAdamW(
            trainable_params,
            lr=lr, betas=(0.9, 0.999),
            weight_decay=1e-4, amsgrad=True, eps=1e-10
        )
    except ImportError:
        # Fallback to AdamW if StableAdamW not available
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr, betas=(0.9, 0.999), weight_decay=1e-4
        )
        print("Warning: StableAdamW not available, using AdamW")

    # Cosine LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_iterations, eta_min=lr * 0.1
    )

    # Training loop
    global_step = 0
    loss_history = []

    pbar = tqdm(total=n_iterations, desc="Training Dinomaly")
    while global_step < n_iterations:
        for batch in loader:
            if global_step >= n_iterations:
                break

            images = batch['image'].to(device)

            # Forward pass
            en, de = torch_model.get_encoder_decoder_outputs(images)
            loss = torch_model.loss_fn(
                encoder_features=en,
                decoder_features=de,
                global_step=global_step
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, max_norm=0.1)
            optimizer.step()
            scheduler.step()

            loss_history.append(loss.item())
            global_step += 1
            pbar.update(1)

            if global_step % 1000 == 0:
                avg_loss = np.mean(loss_history[-100:])
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

    pbar.close()
    print(f"Training complete. Final loss: {np.mean(loss_history[-100:]):.4f}")

    # Save weights if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(torch_model.state_dict(), save_path)
        print(f"Weights saved to {save_path}")

    # Return the lightning model wrapper for inference compatibility
    model.model = torch_model
    return model


def train_anomalydino(
    train_df: pd.DataFrame,
    device: str = 'cuda',
    repo_path: str = '/content/drive/MyDrive/BachelorsThesis',
    save_path: str = None,
) -> object:
    """
    Build AnomalyDINO memory bank from normal images in train_df.
    AnomalyDINO is training-free — this function just runs a forward pass
    over all normal training images to populate the memory bank.

    Args:
        train_df:   dataframe from realiad_utils — normal images only
        device:     'cuda' or 'cpu'
        repo_path:  path to BachelorsThesis repo
        save_path:  optional path to save memory bank

    Returns:
        AnomalyDINO model with populated memory bank
    """
    from anomalib.models import AnomalyDINO
    from anomalib.engine import Engine
    from anomalib.data import MVTecAD

    RealIADTorchDataset = _load_dataset_class(repo_path)

    normal_df = train_df[train_df['label'] == 0].reset_index(drop=True)
    print(f"Building AnomalyDINO memory bank from {len(normal_df)} normal images")

    dataset = RealIADTorchDataset(normal_df, load_masks=False)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    model = AnomalyDINO()
    model = model.to(device)
    model.eval()

    # AnomalyDINO builds memory bank during a fit-like forward pass
    # We call the model's memory bank construction directly
    all_features = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Building memory bank"):
            images = batch['image'].to(device)
            # Extract features from encoder
            features = model.model.encoder(images)
            all_features.append(features.cpu())

    # Store features in memory bank
    memory_bank = torch.cat(all_features, dim=0)
    model.model.memory_bank = memory_bank.to(device)

    print(f"Memory bank built: {memory_bank.shape}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(memory_bank, save_path)
        print(f"Memory bank saved to {save_path}")

    return model


# =============================================================================
# SECTION 2: INFERENCE FUNCTION (shared by all models)
# =============================================================================

def run_inference(
    model,
    test_df: pd.DataFrame,
    model_name: str,
    device: str = 'cuda',
    batch_size: int = 8,
    repo_path: str = '/content/drive/MyDrive/BachelorsThesis',
) -> pd.DataFrame:
    """
    Run inference on test_df and return results dataframe.
    Works for Dinomaly and AnomalyDINO (Anomalib-based models).
    For INP-Former use run_inpformer.py instead.

    Args:
        model:       trained model from train_dinomaly or train_anomalydino
        test_df:     dataframe with all test images (normal + anomalous)
        model_name:  'Dinomaly' or 'AnomalyDINO' — used for labeling
        device:      'cuda' or 'cpu'
        batch_size:  inference batch size
        repo_path:   path to BachelorsThesis repo

    Returns:
        DataFrame with original columns plus:
            image_score:      scalar anomaly score per image
            anomaly_map_path: path to saved anomaly map (if applicable)
    """
    RealIADTorchDataset = _load_dataset_class(repo_path)

    dataset = RealIADTorchDataset(test_df, load_masks=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    model.eval()
    if hasattr(model, 'model'):
        model.model.eval()

    all_scores = []
    all_paths = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Inference [{model_name}]"):
            images = batch['image'].to(device)

            # Run inference
            output = model(images)
            scores = output.pred_score.cpu().numpy()

            all_scores.extend(scores.tolist())
            all_paths.extend(batch['image_path'])

    # Build results dataframe
    results_df = test_df.copy()
    # Match scores to image paths
    path_to_score = dict(zip(all_paths, all_scores))
    results_df['image_score'] = results_df['image_path'].map(path_to_score)
    results_df['model'] = model_name

    print(f"\n{model_name} inference complete")
    print(f"Total images: {len(results_df)}")
    print(f"Score range: [{results_df['image_score'].min():.4f}, "
          f"{results_df['image_score'].max():.4f}]")

    return results_df


# =============================================================================
# SECTION 3: UTILITY FUNCTIONS
# =============================================================================

def measure_inference_time(
    model,
    device: str = 'cuda',
    n_warmup: int = 10,
    n_runs: int = 100,
    image_size: int = 392,
) -> dict:
    """
    Measure inference time per image in milliseconds.
    Uses batch size 1 to simulate single-image deployment.

    Args:
        model:      trained model
        device:     'cuda' or 'cpu'
        n_warmup:   warmup runs before timing
        n_runs:     number of timed runs
        image_size: input image size

    Returns:
        dict with mean_ms, std_ms, min_ms, max_ms
    """
    model.eval()
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy_input)

    # Timing
    times = []
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for _ in range(n_runs):
            starter.record()
            model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
    }


def measure_memory_footprint(model, device: str = 'cuda') -> dict:
    """
    Measure GPU memory footprint of model in MB.

    Returns:
        dict with model_mb (parameters) and peak_mb (during inference)
    """
    # Parameter memory
    param_mb = sum(p.numel() * p.element_size()
                   for p in model.parameters()) / 1e6

    # Peak memory during inference
    torch.cuda.reset_peak_memory_stats()
    dummy_input = torch.randn(1, 3, 392, 392).to(device)
    with torch.no_grad():
        model(dummy_input)
    peak_mb = torch.cuda.max_memory_allocated() / 1e6

    return {
        'param_mb': param_mb,
        'peak_mb': peak_mb,
    }


if __name__ == '__main__':
    print("trainer.py loaded successfully")
    print("Available functions:")
    print("  train_dinomaly(train_df, ...)")
    print("  train_anomalydino(train_df, ...)")
    print("  run_inference(model, test_df, ...)")
    print("  measure_inference_time(model, ...)")
    print("  measure_memory_footprint(model, ...)")