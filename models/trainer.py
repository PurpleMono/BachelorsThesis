"""
models/trainer.py — Unified training and inference for all three models.

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
from torch.utils.data import DataLoader, ConcatDataset
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
    n_iterations: int = 100000,
    batch_size: int = 16,
    lr: float = 2e-3,
    dropout_rate: float = 0.4,
    device: str = 'cuda',
    repo_path: str = '',
    save_path: str = None,
) -> object:
    """
    Train Dinomaly2 on normal images from train_df.

    Implements the official multi-class Dinomaly (realiad_uni.py) training
    protocol with the Dinomaly2 Context-Aware Recentering extension, which
    is available in Anomalib 2.3.3 via use_context_recentering=True.

    Published reference: Dinomaly2 achieves 92.1% I-AUROC on Real-IAD
    multi-class setting (Guo et al., 2025 preprint).

    Key implementation details matching official realiad_uni.py:
    - ConcatDataset across all 30 categories (multi-class)
    - StableAdamW = AdamW with amsgrad=True, same parameters
    - WarmCosineScheduler: warmup 100 steps, base_lr=2e-3, final_lr=2e-4
    - Loss: global_cosine_hm_percent with progressive hard mining
      p increases from 0 to 0.9 over first 1000 steps, factor=0.1
    - dropout_rate=0.4 for diverse/multi-class datasets per paper
    - Context-Aware Recentering: subtracts class token from patch features
      to resolve multi-class confusion (Dinomaly2 contribution)
    - Image score: top 0.1% pixels (max_ratio=0.001) for Real-IAD
      which has tiny defects (specified in Dinomaly2 paper)

    Args:
        train_df:      dataframe from realiad_utils — normal images only
        n_iterations:  total training iterations (default: 50000 per paper)
        batch_size:    training batch size (default: 16 per paper)
        lr:            learning rate (default: 2e-3 per paper)
        dropout_rate:  noisy bottleneck dropout (default: 0.4 for Real-IAD)
        device:        'cuda' or 'cpu'
        repo_path:     path to BachelorsThesis repo
        save_path:     optional path to save model weights

    Returns:
        Trained Dinomaly model ready for inference via run_inference()
    """
    from anomalib.models import Dinomaly

    # INP-Former repo provides WarmCosineScheduler and loss utilities
    _setup_inpformer_path(repo_path)
    from utils import WarmCosineScheduler

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

    # Dinomaly2 via Anomalib — context-aware recentering resolves
    # multi-class confusion by conditioning features on class token
    model = Dinomaly(
        bottleneck_dropout=dropout_rate,    # 0.4 for diverse datasets per paper
        use_context_recentering=True        # Dinomaly2 multi-class component
    )
    torch_model = model.model.to(device)
    print(f"Noisy bottleneck dropout: {dropout_rate}")
    print(f"Context-aware recentering: enabled (Dinomaly2)")

    torch_model.train()

    # Only train bottleneck and decoder — encoder is frozen DINOv2
    trainable_params = [
        p for name, p in torch_model.named_parameters()
        if 'encoder' not in name
    ]
    print(f"Trainable parameters: "
          f"{sum(p.numel() for p in trainable_params) / 1e6:.1f}M")

    # StableAdamW = AdamW with amsgrad=True — matches official implementation
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        amsgrad=True,    # this IS StableAdamW per Reddi et al. 2018
        eps=1e-10
    )

    # Warm cosine schedule matching official realiad_uni.py exactly
    scheduler = WarmCosineScheduler(
        optimizer,
        base_value=lr,          # 2e-3
        final_value=lr * 0.1,   # 2e-4
        total_iters=n_iterations,
        warmup_iters=100
    )

    global_step = 0
    loss_history = []
    pbar = tqdm(total=n_iterations, desc="Training Dinomaly")

    while global_step < n_iterations:
        for batch in loader:
            if global_step >= n_iterations:
                break

            images = batch['image'].to(device)

            # Forward pass returns loss scalar directly during training.
            # global_step enables progressive hard mining:
            # discarding rate increases from 0% to 90% over first 1000 steps.
            loss = torch_model(images, global_step=global_step)

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
    repo_path: str = '',
    save_path: str = None,
    sampling_ratio: float = 0.1,
) -> object:
    """
    Build AnomalyDINO memory bank from normal images in train_df.

    AnomalyDINO is training-free — features are extracted from normal images
    and stored in a memory bank via embedding_store then consolidated with fit().

    Coreset subsampling (ratio=0.1 per paper) reduces memory bank size.
    For the multi-class setting with all 30 categories, the full memory bank
    before subsampling would require ~87GB GPU VRAM. To avoid OOM, embeddings
    are moved to CPU before the vstack operation, then the final coreset
    bank is moved back to GPU for inference. Results are identical —
    only the location of the stacking operation changes.

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

    # Extract patch features into embedding_store on GPU
    with torch.no_grad():
        for batch in tqdm(loader, desc="Building memory bank"):
            images = batch['image'].to(device)
            torch_model(images)

    # Move to CPU before vstack to avoid OOM (~87GB for 30 categories at 0.1)
    # Mathematically identical — only the computation location changes
    torch_model.embedding_store = [
        e.cpu() for e in torch_model.embedding_store
    ]

    # Consolidate and apply coreset subsampling on CPU
    torch_model.fit()

    # Move final coreset bank back to GPU for fast nearest-neighbour inference
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
    dataset_root: str = '',
    n_epochs: int = 200,
    batch_size: int = 16,
    lr: float = 1e-3,
    inp_num: int = 6,
    device: str = 'cuda',
    repo_path: str = '',
    save_path: str = None,
) -> object:
    """
    Train INP-Former on normal images from train_df.

    Uses original INP-Former repo with a standardised interface.
    Follows published paper defaults: StableAdamW, lr=1e-3, 200 epochs.

    Multi-class training: all categories are combined via ConcatDataset
    into a single training loader, matching the official multiclass.py script.
    The WarmCosineScheduler total_iters is set to n_epochs * len(loader)
    so reducing n_epochs automatically adjusts the schedule correctly —
    important for the compute equalisation ablation (Investigation 2).

    Published reference: INP-Former achieves 92.1% I-AUROC on Real-IAD
    multi-class setting (INP-Former paper, CVPR 2025).

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
        Trained INP-Former model ready for inference via run_inference_inpformer()
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

    data_transform, _ = get_data_transforms(448, 392)

    normal_df = train_df[train_df['label'] == 0].reset_index(drop=True)

    # Multi-class: combine all categories into one ConcatDataset
    # matching official multiclass.py — one model trained on all 30 categories
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

    # Build model — ViT-Base/14 with DINOv2-Register weights (frozen)
    encoder = vit_encoder.load('dinov2reg_vit_base_14')
    embed_dim, num_heads = 768, 12
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    Bottleneck = nn.ModuleList(
        [Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)])
    INP = nn.ParameterList(
        [nn.Parameter(torch.randn(inp_num, embed_dim))])
    INP_Extractor = nn.ModuleList([
        Aggregation_Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
    ])
    INP_Guided_Decoder = nn.ModuleList([
        Prototype_Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
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

    # Only bottleneck, decoder, extractor, and INP tokens are trained
    trainable = nn.ModuleList(
        [Bottleneck, INP_Guided_Decoder, INP_Extractor, INP])

    # Weight initialisation matching official multiclass.py
    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # total_iters scales with n_epochs and loader length —
    # reducing epochs for ablation automatically adjusts the LR schedule
    total_iters = n_epochs * len(loader)

    optimizer = StableAdamW(
        [{'params': trainable.parameters()}],
        lr=lr, betas=(0.9, 0.999), weight_decay=1e-4,
        amsgrad=True, eps=1e-10
    )
    lr_scheduler = WarmCosineScheduler(
        optimizer,
        base_value=lr,
        final_value=lr * 0.1,
        total_iters=total_iters,
        warmup_iters=100
    )

    print(f"Trainable parameters: "
          f"{sum(p.numel() for p in trainable.parameters()) / 1e6:.1f}M")

    for epoch in range(n_epochs):
        model.train()
        loss_list = []
        for img, _ in tqdm(
                loader, ncols=80, desc=f"Epoch {epoch+1}/{n_epochs}"):
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
            print(f"Epoch [{epoch+1}/{n_epochs}], "
                  f"loss: {np.mean(loss_list):.4f}")

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
    batch_size: int = 16,
    repo_path: str = '',
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

    File structure:
        {maps_save_dir}/{model_name}/{category}/{image_stem}.npz
    Each .npz contains:
        anomaly_map:   float32 array (H, W)
        anomaly_score: float32 scalar

    Args:
        model:             trained model
        test_df:           dataframe with all test images
        model_name:        'Dinomaly' or 'AnomalyDINO'
        device:            'cuda' or 'cpu'
        batch_size:        inference batch size (default: 16)
        repo_path:         path to BachelorsThesis repo
        max_ratio:         top pixel ratio for image score (0.001 for Real-IAD)
        save_anomaly_maps: whether to save anomaly maps to disk
        maps_save_dir:     root directory for saving maps

    Returns:
        DataFrame with all original columns plus image_score and model
    """
    RealIADTorchDataset = _load_dataset_class(repo_path)

    dataset = RealIADTorchDataset(test_df, load_masks=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # Patch Dinomaly image score ratio for Real-IAD
    # Dinomaly2 paper specifies top 0.1% for Real-IAD (tiny defects)
    if model_name == 'Dinomaly' and max_ratio is not None:
        import anomalib.models.image.dinomaly.torch_model as dinomaly_module
        original_ratio = dinomaly_module.DEFAULT_MAX_RATIO
        dinomaly_module.DEFAULT_MAX_RATIO = max_ratio
        print(f"Dinomaly image score ratio set to {max_ratio} "
              f"(top {max_ratio*100:.1f}% pixels)")

    model.eval()
    if hasattr(model, 'model'):
        model.model.eval()

    all_scores = []
    all_paths = []

    # Lookups for anomaly map saving
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
            output = model(images)
            scores = output.pred_score.cpu().numpy().flatten()
            amaps = output.anomaly_map.cpu().numpy()

            for score, path, amap in zip(
                    scores, batch['image_path'], amaps):
                score_val = float(score)
                all_scores.append(score_val)
                all_paths.append(path)

                # Save anomaly map for anomalous images only
                if save_anomaly_maps and maps_save_dir:
                    if path_to_label.get(path, -1) == 1:
                        category = path_to_category.get(path, 'unknown')
                        save_dir = os.path.join(
                            maps_save_dir, model_name, category)
                        os.makedirs(save_dir, exist_ok=True)
                        stem = os.path.splitext(
                            os.path.basename(path))[0]
                        np.savez_compressed(
                            os.path.join(save_dir, f"{stem}.npz"),
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
        n_saved = int(results_df['label'].sum())
        print(f"Anomaly maps saved: {n_saved} anomalous images")
        print(f"Save directory: {maps_save_dir}/{model_name}/")

    return results_df


def run_inference_inpformer(
    model,
    test_df: pd.DataFrame,
    dataset_root: str = '',
    device: str = 'cuda',
    batch_size: int = 16,
    repo_path: str = '',
    save_anomaly_maps: bool = False,
    maps_save_dir: str = None,
) -> pd.DataFrame:
    """
    Run INP-Former inference on test_df and return results dataframe.

    Uses RealIADTorchDataset (our unified dataset class) to accept the
    full multi-category test_df directly — no category loop needed in
    the notebook. This matches the inference behaviour of run_inference()
    and ensures consistent evaluation across all 30 categories.

    Anomaly maps are optionally saved as compressed numpy files (.npz).
    Only anomalous images (label == 1) are saved to minimise storage.

    File structure:
        {maps_save_dir}/INP-Former/{category}/{image_stem}.npz
    Each .npz contains:
        anomaly_map:   float32 array (H, W)
        anomaly_score: float32 scalar

    Args:
        model:             trained INP-Former model
        test_df:           dataframe with all test images (all categories)
        dataset_root:      path to Real-IAD dataset root (unused — kept for
                           API consistency with notebook calls)
        device:            'cuda' or 'cpu'
        batch_size:        inference batch size (default: 16)
        repo_path:         path to BachelorsThesis repo
        save_anomaly_maps: whether to save anomaly maps to disk
        maps_save_dir:     root directory for saving maps

    Returns:
        DataFrame with all original columns plus image_score and model
    """
    _setup_inpformer_path(repo_path)

    from utils import get_gaussian_kernel, cal_anomaly_maps
    from torch.nn import functional as F

    # Use our unified dataset class — accepts full multi-category test_df
    # directly without requiring a category loop in the notebook
    RealIADTorchDataset = _load_dataset_class(repo_path)
    dataset = RealIADTorchDataset(test_df, load_masks=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    model.eval()
    gaussian_kernel = get_gaussian_kernel(
        kernel_size=5, sigma=4).to(device)

    # Lookups for label checking and map saving
    path_to_label = dict(zip(
        test_df['image_path'].tolist(),
        test_df['label'].tolist()
    ))
    path_to_category = dict(zip(
        test_df['image_path'].tolist(),
        test_df['category'].tolist()
    )) if 'category' in test_df.columns else {}

    all_scores = []
    all_paths = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference [INP-Former]"):
            img = batch['image'].to(device)
            img_paths = batch['image_path']

            output = model(img)
            en, de = output[0], output[1]

            # Compute anomaly map via cosine similarity between
            # encoder and decoder features, then apply Gaussian smoothing
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            anomaly_map = F.interpolate(
                anomaly_map, size=256,
                mode='bilinear', align_corners=False)
            anomaly_map = gaussian_kernel(anomaly_map)

            # Image score: mean of top 1% pixels per official script
            flat = anomaly_map.flatten(1)
            k = max(1, int(flat.shape[1] * 0.01))
            scores = torch.topk(flat, k, dim=1)[0].mean(dim=1)

            for score, path, amap in zip(
                    scores.cpu().numpy(),
                    img_paths,
                    anomaly_map.cpu().numpy()):
                score_val = float(score)
                all_scores.append(score_val)
                all_paths.append(path)

                # Save anomaly map for anomalous images only
                if save_anomaly_maps and maps_save_dir:
                    if path_to_label.get(path, -1) == 1:
                        category = path_to_category.get(path, 'unknown')
                        save_dir = os.path.join(
                            maps_save_dir, 'INP-Former', category)
                        os.makedirs(save_dir, exist_ok=True)
                        stem = os.path.splitext(
                            os.path.basename(path))[0]
                        np.savez_compressed(
                            os.path.join(save_dir, f"{stem}.npz"),
                            anomaly_map=amap[0].astype(np.float32),
                            anomaly_score=np.float32(score_val)
                        )

    results_df = test_df.copy()
    path_to_score = dict(zip(all_paths, all_scores))
    results_df['image_score'] = results_df['image_path'].map(path_to_score)
    results_df['model'] = 'INP-Former'

    print(f"\nINP-Former inference complete")
    print(f"Total images: {len(results_df)}")
    print(f"Score range: [{results_df['image_score'].min():.4f}, "
          f"{results_df['image_score'].max():.4f}]")
    if save_anomaly_maps:
        n_saved = int(results_df['label'].sum())
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
    CUDA events are used for accurate GPU timing — more precise
    than Python time.time() for GPU-bound operations.

    Args:
        model:      trained model
        device:     'cuda' or 'cpu'
        n_warmup:   warmup runs before timing (ensures GPU steady state)
        n_runs:     number of timed runs
        image_size: input image size

    Returns:
        dict with mean_ms, std_ms, min_ms, max_ms
    """
    model.eval()
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy_input)

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


def measure_memory_footprint(
    model,
    device: str = 'cuda'
) -> dict:
    """
    Measure GPU memory footprint of model in MB.

    Returns:
        dict with param_mb (parameter memory) and peak_mb (peak during inference)
    """
    param_mb = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    ) / 1e6

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
    print("  train_dinomaly(train_df, ...)              — Dinomaly2 multi-class")
    print("  train_anomalydino(train_df, ...)           — AnomalyDINO memory bank")
    print("  train_inpformer(train_df, ...)             — INP-Former multi-class")
    print("  run_inference(model, test_df, ...)         — Dinomaly, AnomalyDINO")
    print("  run_inference_inpformer(model, test_df, ...) — INP-Former")
    print("  measure_inference_time(model, ...)")
    print("  measure_memory_footprint(model, ...)")