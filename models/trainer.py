"""
models/trainer.py — Unified training and inference functions for all three models.

Each model has its own training function but they share:
- The same data loading interface (RealIADTorchDataset)
- The same inference interface (returns standardised dataframe)
- The same preprocessing (448→392 centre crop, ImageNet normalisation)

Usage:
    from models.trainer import train_dinomaly, train_anomalydino, train_inpformer
    from models.trainer import run_inference, run_inference_inpformer

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
from functools import partial


def _load_dataset_class(repo_path: str):
    """Load RealIADTorchDataset from repo path."""
    spec = importlib.util.spec_from_file_location(
        "realiad_dataset",
        f"{repo_path}/data/realiad_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RealIADTorchDataset


def _setup_inpformer_path(repo_path: str):
    """Add INP-Former to sys.path if not already there."""
    inp_former_path = f"{repo_path}/models/inp_former"
    if inp_former_path not in sys.path:
        sys.path.insert(0, inp_former_path)


# =============================================================================
# SECTION 1: TRAINING FUNCTIONS
# =============================================================================

def train_dinomaly(
    train_df: pd.DataFrame,
    n_iterations: int = 50000,
    batch_size: int = 16,
    lr: float = 2e-3,
    dropout_rate: float = 0.4,
    device: str = 'cuda',
    repo_path: str = '/content/drive/MyDrive/BachelorsThesis',
    save_path: str = None,
) -> object:
    """
    Train Dinomaly on normal images from train_df.
    Uses Anomalib's Dinomaly model with a custom training loop.
    Follows published paper defaults for Real-IAD:
    - StableAdamW, lr=2e-3, 50k iterations
    - Dropout rate 0.4 (increased from default 0.2 for diverse datasets)

    Args:
        train_df:      dataframe from realiad_utils — normal images only
        n_iterations:  total training iterations (default: 50000 per paper)
        batch_size:    training batch size (default: 16 per paper)
        lr:            learning rate (default: 2e-3 per paper)
        dropout_rate:  noisy bottleneck dropout (default: 0.4 for Real-IAD per paper)
        device:        'cuda' or 'cpu'
        repo_path:     path to BachelorsThesis repo
        save_path:     optional path to save model weights

    Returns:
        Trained Dinomaly model ready for inference
    """
    from anomalib.models import Dinomaly

    RealIADTorchDataset = _load_dataset_class(repo_path)

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

    model = Dinomaly()
    torch_model = model.model.to(device)

    # Fix dropout rate for Real-IAD — paper specifies 0.4 for diverse datasets
    # Anomalib default is 0.2 which is correct for MVTec-AD but not Real-IAD
    for module in torch_model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = dropout_rate
    print(f"Noisy bottleneck dropout rate set to {dropout_rate} for Real-IAD")

    torch_model.train()

    trainable_params = [
        p for name, p in torch_model.named_parameters()
        if 'encoder' not in name
    ]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params) / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=2e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        amsgrad=True
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_iterations, eta_min=lr * 0.1
    )

    global_step = 0
    loss_history = []

    pbar = tqdm(total=n_iterations, desc="Training Dinomaly")
    while global_step < n_iterations:
        for batch in loader:
            if global_step >= n_iterations:
                break

            images = batch['image'].to(device)

            en, de = torch_model.get_encoder_decoder_outputs(images)
            loss = torch_model.loss_fn(
                encoder_features=en,
                decoder_features=de,
                global_step=global_step
            )

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

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(torch_model.state_dict(), save_path)
        print(f"Weights saved to {save_path}")

    model.model = torch_model
    return model


def train_anomalydino(
    train_df: pd.DataFrame,
    device: str = 'cuda',
    repo_path: str = '/content/drive/MyDrive/BachelorsThesis',
    save_path: str = None,
    sampling_ratio: float = 0.1,
) -> object:
    """
    Build AnomalyDINO memory bank from normal images in train_df.
    AnomalyDINO is training-free — features are extracted from normal images
    and stored in a memory bank via embedding_store then consolidated with fit().

    Coreset subsampling is applied to reduce memory bank size for GPU compatibility.
    Default ratio of 0.1 follows the original paper's recommendation.
    Adjust sampling_ratio based on available GPU memory:
        T4 (16GB): 0.1 per category, lower for multi-class
        L4 (24GB): up to 0.1 for multi-class with all 30 categories

    Args:
        train_df:       dataframe from realiad_utils — normal images only
        device:         'cuda' or 'cpu'
        repo_path:      path to BachelorsThesis repo
        save_path:      optional path to save memory bank
        sampling_ratio: coreset subsampling ratio (default 0.1 per paper)

    Returns:
        AnomalyDINO model with populated and subsampled memory bank
    """
    from anomalib.models import AnomalyDINO

    RealIADTorchDataset = _load_dataset_class(repo_path)

    normal_df = train_df[train_df['label'] == 0].reset_index(drop=True)
    print(f"Building AnomalyDINO memory bank from {len(normal_df)} normal images")
    print(f"Coreset sampling ratio: {sampling_ratio}")

    dataset = RealIADTorchDataset(normal_df, load_masks=False)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    model = AnomalyDINO(
        encoder_name='dinov2reg_vit_base_14',
        coreset_subsampling=True,
        sampling_ratio=sampling_ratio,
        masking=False,
    )
    torch_model = model.model.to(device)
    torch_model.train()

    # Extract features into embedding_store
    with torch.no_grad():
        for batch in tqdm(loader, desc="Building memory bank"):
            images = batch['image'].to(device)
            # Forward pass in train mode populates embedding_store automatically
            torch_model(images)

    # Move embedding_store to CPU before stacking to avoid GPU OOM
    torch_model.embedding_store = [
    e.cpu() for e in torch_model.embedding_store
    ]


    # Consolidate and subsample memory bank
    torch_model.fit()

    # Move final coreset bank back to GPU for inference
    torch_model.memory_bank = torch_model.memory_bank.to(device)

    print(f"Memory bank built: {torch_model.memory_bank.shape}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(torch_model.memory_bank, save_path)
        print(f"Memory bank saved to {save_path}")

    model.model = torch_model
    return model


def train_inpformer(
    train_df: pd.DataFrame,
    dataset_root: str = '/content/drive/MyDrive/datasets/realiad_512',
    n_epochs: int = 200,
    batch_size: int = 16,
    lr: float = 1e-3,
    inp_num: int = 6,
    device: str = 'cuda',
    repo_path: str = '/content/drive/MyDrive/BachelorsThesis',
    save_path: str = None,
) -> object:
    """
    Train INP-Former on normal images from train_df.
    Uses original INP-Former repo with a standardised interface.
    Follows published paper defaults: StableAdamW, lr=1e-3, 200 epochs.

    INP-Former extends reconstruction-based detection with Image-level Normal
    Prototypes (INPs) — learnable tokens that aggregate global normal patterns
    at test time to guide feature reconstruction.

    Args:
        train_df:      dataframe from realiad_utils — normal images only
        dataset_root:  path to Real-IAD dataset root (contains category folders)
        n_epochs:      number of training epochs (default: 200 per paper)
        batch_size:    training batch size (default: 16 per paper)
        lr:            learning rate (default: 1e-3 per paper)
        inp_num:       number of prototype tokens (default: 6 per paper)
        device:        'cuda' or 'cpu'
        repo_path:     path to BachelorsThesis repo
        save_path:     optional path to save model weights

    Returns:
        Trained INP-Former model ready for inference
    """
    _setup_inpformer_path(repo_path)

    from models import vit_encoder
    from models.uad import INP_Former
    from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
    from optimizers import StableAdamW
    from utils import WarmCosineScheduler, global_cosine_hm_adaptive, setup_seed
    from dataset import RealIADDataset, get_data_transforms
    from torch.nn.init import trunc_normal_

    setup_seed(1)

    # Use INP-Former's own data transforms — identical preprocessing to other models
    data_transform, _ = get_data_transforms(448, 392)

    # Only normal images for training
    normal_df = train_df[train_df['label'] == 0].reset_index(drop=True)
    print(f"Training INP-Former on {len(normal_df)} normal images")

    # Use INP-Former's RealIADDataset with explicit dataset_root
    # Replace the single-category dataset with ConcatDataset across all categories
    train_data_list = []
    categories = normal_df['category'].unique()

    for category in categories:
        train_data = RealIADDataset(
            root=dataset_root,
            category=category,
            transform=data_transform,
            gt_transform=None,
            phase='train'
        )
        train_data_list.append(train_data)

    
    from torch.utils.data import ConcatDataset
    combined_dataset = ConcatDataset(train_data_list)

    loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    print(f"Training INP-Former on {len(combined_dataset)} normal images "
        f"across {len(categories)} categories")

    # Build model — ViT-Base/14 with DINOv2-Register weights
    encoder = vit_encoder.load('dinov2reg_vit_base_14')
    embed_dim, num_heads = 768, 12
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    Bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)])
    INP = nn.ParameterList([nn.Parameter(torch.randn(inp_num, embed_dim))])
    INP_Extractor = nn.ModuleList([
        Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                          qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
    ])
    INP_Guided_Decoder = nn.ModuleList([
        Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        for _ in range(8)
    ])

    model = INP_Former(
        encoder=encoder,
        bottleneck=Bottleneck,
        aggregation=INP_Extractor,
        decoder=INP_Guided_Decoder,
        target_layers=target_layers,
        remove_class_token=True,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
        prototype_token=INP
    ).to(device)

    # Trainable modules — encoder is frozen
    trainable = nn.ModuleList([Bottleneck, INP_Guided_Decoder, INP_Extractor, INP])

    # Initialise weights
    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    total_iters = n_epochs * len(loader)
    optimizer = StableAdamW(
        [{'params': trainable.parameters()}],
        lr=lr, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10
    )
    lr_scheduler = WarmCosineScheduler(
        optimizer, base_value=lr, final_value=lr * 0.1,
        total_iters=total_iters, warmup_iters=100
    )

    print(f"Trainable parameters: {sum(p.numel() for p in trainable.parameters()) / 1e6:.1f}M")

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        loss_list = []
        for img, _ in tqdm(loader, ncols=80, desc=f"Epoch {epoch+1}/{n_epochs}"):
            img = img.to(device)
            en, de, g_loss = model(img)
            loss = global_cosine_hm_adaptive(en, de, y=3)
            loss = loss + 0.2 * g_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], loss: {np.mean(loss_list):.4f}")

    print(f"Training complete. Final loss: {np.mean(loss_list):.4f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Weights saved to {save_path}")

    return model


# =============================================================================
# SECTION 2: INFERENCE FUNCTIONS
# =============================================================================

def run_inference(
    model,
    test_df: pd.DataFrame,
    model_name: str,
    device: str = 'cuda',
    batch_size: int = 8,
    repo_path: str = '/content/drive/MyDrive/BachelorsThesis',
    max_ratio: float = None,
    save_anomaly_maps: bool = False,
    maps_save_dir: str = None,
) -> pd.DataFrame:
    """
    Run inference on test_df and return results dataframe.
    Works for Dinomaly and AnomalyDINO (Anomalib-based models).

    Anomaly maps are optionally saved as compressed numpy files (.npz)
    containing both the anomaly map array and the image-level score.
    Only anomalous images (label == 1) are saved to minimise storage.

    File naming convention:
        {maps_save_dir}/{model_name}/{category}/{image_stem}.npz
    Each .npz contains:
        anomaly_map: float32 array of shape (H, W)
        anomaly_score: scalar float

    Args:
        model:            trained model
        test_df:          dataframe with all test images
        model_name:       'Dinomaly' or 'AnomalyDINO'
        device:           'cuda' or 'cpu'
        batch_size:       inference batch size
        repo_path:        path to BachelorsThesis repo
        max_ratio:        top pixel ratio for Dinomaly image score
        save_anomaly_maps: whether to save anomaly maps to disk
        maps_save_dir:    root directory for saving maps
                          e.g. '/content/drive/MyDrive/BachelorsThesis/results/anomaly_maps'

    Returns:
        DataFrame with original columns plus image_score and model columns
    """
    RealIADTorchDataset = _load_dataset_class(repo_path)

    dataset = RealIADTorchDataset(test_df, load_masks=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # Patch Dinomaly image score ratio for Real-IAD if specified
    if model_name == 'Dinomaly' and max_ratio is not None:
        import anomalib.models.image.dinomaly.torch_model as dinomaly_module
        original_ratio = dinomaly_module.DEFAULT_MAX_RATIO
        dinomaly_module.DEFAULT_MAX_RATIO = max_ratio
        print(f"Dinomaly image score ratio set to {max_ratio}")

    model.eval()
    if hasattr(model, 'model'):
        model.model.eval()

    all_scores = []
    all_paths = []

    # Build label lookup for deciding which images to save maps for
    path_to_label = dict(zip(
        test_df['image_path'].tolist(),
        test_df['label'].tolist()
    ))
    path_to_category = dict(zip(
        test_df['image_path'].tolist(),
        test_df['category'].tolist()
    )) if 'category' in test_df.columns else {}

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Inference [{model_name}]"):
            images = batch['image'].to(device)

            # GPU inference
            output = model(images)
            scores = output.pred_score.cpu().numpy().flatten()
            amaps = output.anomaly_map.cpu().numpy()

            # Collect scores and paths
            for score, path, amap in zip(
                    scores, batch['image_path'], amaps):
                score_val = float(score)
                all_scores.append(score_val)
                all_paths.append(path)

                # Save anomaly map if requested and image is anomalous
                if save_anomaly_maps and maps_save_dir:
                    label = path_to_label.get(path, -1)
                    if label == 1:
                        category = path_to_category.get(path, 'unknown')
                        save_dir = os.path.join(
                            maps_save_dir, model_name, category)
                        os.makedirs(save_dir, exist_ok=True)
                        stem = os.path.splitext(
                            os.path.basename(path))[0]
                        save_path = os.path.join(
                            save_dir, f"{stem}.npz")
                        # amap shape is (1, H, W) — squeeze to (H, W)
                        np.savez_compressed(
                            save_path,
                            anomaly_map=amap[0].astype(np.float32),
                            anomaly_score=np.float32(score_val)
                        )

    # Restore original ratio
    if model_name == 'Dinomaly' and max_ratio is not None:
        dinomaly_module.DEFAULT_MAX_RATIO = original_ratio

    results_df = test_df.copy()
    path_to_score = dict(zip(all_paths, all_scores))
    results_df['image_score'] = results_df['image_path'].map(path_to_score)
    results_df['model'] = model_name

    print(f"\n{model_name} inference complete")
    print(f"Total images: {len(results_df)}")
    print(f"Score range: [{results_df['image_score'].min():.4f}, "
          f"{results_df['image_score'].max():.4f}]")
    if save_anomaly_maps:
        n_saved = results_df['label'].sum()
        print(f"Anomaly maps saved: {int(n_saved)} anomalous images")
        print(f"Save directory: {maps_save_dir}/{model_name}/")

    return results_df


def run_inference_inpformer(
    model,
    test_df: pd.DataFrame,
    dataset_root: str = '/content/drive/MyDrive/datasets/realiad_512',
    device: str = 'cuda',
    batch_size: int = 8,
    repo_path: str = '/content/drive/MyDrive/BachelorsThesis',
    save_anomaly_maps: bool = False,
    maps_save_dir: str = None,
) -> pd.DataFrame:
    """
    Run INP-Former inference on test_df and return results dataframe.

    Anomaly maps are optionally saved as compressed numpy files (.npz)
    containing both the anomaly map array and the image-level score.
    Only anomalous images (label == 1) are saved to minimise storage.

    File naming convention:
        {maps_save_dir}/INP-Former/{category}/{image_stem}.npz
    Each .npz contains:
        anomaly_map: float32 array of shape (H, W)
        anomaly_score: scalar float

    Args:
        model:            trained INP-Former model
        test_df:          dataframe with all test images
        dataset_root:     path to Real-IAD dataset root
        device:           'cuda' or 'cpu'
        batch_size:       inference batch size
        repo_path:        path to BachelorsThesis repo
        save_anomaly_maps: whether to save anomaly maps to disk
        maps_save_dir:    root directory for saving maps

    Returns:
        DataFrame with original columns plus image_score and model columns
    """
    _setup_inpformer_path(repo_path)
    from dataset import RealIADDataset, get_data_transforms
    from utils import get_gaussian_kernel, cal_anomaly_maps
    from torch.nn import functional as F

    _, gt_transform = get_data_transforms(448, 392)
    data_transform, _ = get_data_transforms(448, 392)

    category = test_df['category'].iloc[0]

    test_data = RealIADDataset(
        root=dataset_root,
        category=category,
        transform=data_transform,
        gt_transform=gt_transform,
        phase='test'
    )

    loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    model.eval()
    gaussian_kernel = get_gaussian_kernel(
        kernel_size=5, sigma=4).to(device)

    # Build label lookup
    path_to_label = dict(zip(
        test_df['image_path'].tolist(),
        test_df['label'].tolist()
    ))

    all_scores = []
    all_paths = []
    all_labels = []

    with torch.no_grad():
        for img, gt, label, img_path in tqdm(
                loader, desc="Inference [INP-Former]"):
            img = img.to(device)
            output = model(img)
            en, de = output[0], output[1]

            # Compute anomaly map
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            anomaly_map = F.interpolate(
                anomaly_map, size=256,
                mode='bilinear', align_corners=False)
            anomaly_map = gaussian_kernel(anomaly_map)

            # Image score: mean of top 1% pixels
            flat = anomaly_map.flatten(1)
            k = max(1, int(flat.shape[1] * 0.01))
            scores = torch.topk(flat, k, dim=1)[0].mean(dim=1)

            for score, path, lbl, amap in zip(
                    scores.cpu().numpy(),
                    img_path,
                    label.numpy(),
                    anomaly_map.cpu().numpy()):

                score_val = float(score)
                all_scores.append(score_val)
                all_paths.append(path)
                all_labels.append(int(lbl))

                # Save anomaly map if requested and image is anomalous
                if save_anomaly_maps and maps_save_dir:
                    if int(lbl) == 1:
                        save_dir = os.path.join(
                            maps_save_dir, 'INP-Former', category)
                        os.makedirs(save_dir, exist_ok=True)
                        stem = os.path.splitext(
                            os.path.basename(path))[0]
                        save_path = os.path.join(
                            save_dir, f"{stem}.npz")
                        np.savez_compressed(
                            save_path,
                            anomaly_map=amap[0].astype(np.float32),
                            anomaly_score=np.float32(score_val)
                        )

    results_df = pd.DataFrame({
        'image_path': all_paths,
        'image_score': all_scores,
        'label': all_labels,
        'model': 'INP-Former',
        'category': category,
    })

    meta_cols = ['image_path', 'viewpoint', 'defect_type',
                 'sample_id', 'has_mask', 'anomaly_class']
    available = [c for c in meta_cols if c in test_df.columns]
    results_df = results_df.merge(
        test_df[available], on='image_path', how='left'
    )

    print(f"\nINP-Former inference complete")
    print(f"Total images: {len(results_df)}")
    print(f"Score range: [{results_df['image_score'].min():.4f}, "
          f"{results_df['image_score'].max():.4f}]")
    if save_anomaly_maps:
        n_saved = sum(all_labels)
        print(f"Anomaly maps saved: {n_saved} anomalous images")
        print(f"Save directory: {maps_save_dir}/INP-Former/")

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

    # Warmup — ensures GPU is in steady state before timing
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy_input)

    # Timed runs using CUDA events for accurate GPU timing
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
    print("  train_inpformer(train_df, ...)")
    print("  run_inference(model, test_df, ...)          — Dinomaly, AnomalyDINO")
    print("  run_inference_inpformer(model, test_df, ...) — INP-Former")
    print("  measure_inference_time(model, ...)")
    print("  measure_memory_footprint(model, ...)")