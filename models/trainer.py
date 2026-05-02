"""
models/trainer.py — Unified training and inference for all three models.

Each model has its own training function but they share:
- The same data loading interface (RealIADTorchDataset)
- The same inference interface (returns standardised dataframe)
- The same preprocessing (448→392 centre crop, ImageNet normalisation)

Usage:
    from models.trainer import train_dinomaly, train_anomalydino_fewshot, train_inpformer
    from models.trainer import run_inference_dinomaly, run_inference, run_inference_inpformer

    # Standard protocol
    model = train_dinomaly(train_df, n_iterations=50000, device='cuda')
    results_df = run_inference_dinomaly(model, test_df, device='cuda')

    # Cross-view protocol — just filter the dataframe
    train_cv_df = train_df[train_df['viewpoint'].isin(['C1', 'C2'])]
    model = train_dinomaly(train_cv_df, n_iterations=50000, device='cuda')
"""

import gc

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset, default_collate
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
    """Add INP-Former to sys.path at position 0, before Dinomaly."""
    inp_former_path = f"{repo_path}/models/inp_former"
    # Remove if already present to ensure it stays at position 0
    if inp_former_path in sys.path:
        sys.path.remove(inp_former_path)
    sys.path.insert(0, inp_former_path)


def _setup_dinomaly_path(repo_path: str):
    """Add official Dinomaly repo to sys.path at position 0."""
    dinomaly_path = f"{repo_path}/models/dinomaly"
    if dinomaly_path in sys.path:
        sys.path.remove(dinomaly_path)
    sys.path.insert(0, dinomaly_path)


def _collate_fn(batch):
    """
    Custom collate for RealIADTorchDataset dicts.

    RealIADTorchDataset returns dicts containing both tensors and strings.
    PyTorch's default collate handles tensors but fails on string fields
    when num_workers > 0. This collate keeps strings as lists and uses
    default_collate for all tensor/numeric fields.
    """
    result = {}
    for key in batch[0].keys():
        vals = [b[key] for b in batch]
        if isinstance(vals[0], str):
            result[key] = vals
        elif isinstance(vals[0], (bool, np.bool_)):
            result[key] = vals
        else:
            try:
                result[key] = default_collate(vals)
            except Exception:
                result[key] = vals
    return result


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
    repo_path: str = '',
    save_path: str = None,
) -> object:
    """
    Train Dinomaly using the official Dinomaly repository directly.

    Uses ViTill from models/uad.py and global_cosine_hm_percent from utils.py,
    exactly matching the published realiad_uni.py training script (Guo et al., 2025a).

    This replaces the previous Anomalib-based implementation, which produced
    near-zero score separation (I-AUROC ~0.49) due to Anomalib's
    CosineHardMiningLoss not matching the official training signal.
    The official global_cosine_hm_percent with progressive hard mining
    is the loss that produced the published 89.3% I-AUROC on Real-IAD.

    Uses RealIADTorchDataset (our unified loader) rather than the Dinomaly
    repo's RealIADDataset, so cross-view protocol works identically by
    passing a filtered train_df — no changes to this function needed.

    Published reference: 89.3% I-AUROC on Real-IAD multi-class (Guo et al., 2025a).

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
        Trained ViTill model ready for inference via run_inference_dinomaly()
    """
    # Clear cached module imports that may point to Dinomaly's modules
    import sys
    for key in list(sys.modules.keys()):
        if key.startswith('models') or key in ('utils', 'dataset', 'optimizers'):
            del sys.modules[key]

    _setup_dinomaly_path(repo_path)

    from models.uad import ViTill
    from models import vit_encoder
    from models.vision_transformer import bMlp, LinearAttention2
    from models.vision_transformer import Block as VitBlock
    from optimizers import StableAdamW
    from utils import WarmCosineScheduler, global_cosine_hm_percent
    from torch.nn.init import trunc_normal_

    # Seed for reproducibility — matches official realiad_uni.py
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    RealIADTorchDataset = _load_dataset_class(repo_path)
    normal_df = train_df[train_df['label'] == 0].reset_index(drop=True)

    dataset = RealIADTorchDataset(normal_df, load_masks=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        collate_fn=_collate_fn
    )

    print(f"Training Dinomaly on {len(normal_df)} normal images "
          f"across {normal_df['category'].nunique()} categories")

    # Build model — exactly matching realiad_uni.py architecture
    encoder = vit_encoder.load('dinov2reg_vit_base_14')
    embed_dim, num_heads = 768, 12
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    bottleneck = nn.ModuleList(
        [bMlp(embed_dim, embed_dim * 4, embed_dim, drop=dropout_rate)])

    decoder = nn.ModuleList([
        VitBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-8),
            attn_drop=0.,
            attn=LinearAttention2)
        for _ in range(8)
    ])

    model = ViTill(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        mask_neighbor_size=0,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder
    ).to(device)

    trainable = nn.ModuleList([bottleneck, decoder])

    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    print(f"Trainable parameters: "
          f"{sum(p.numel() for p in trainable.parameters()) / 1e6:.1f}M")

    optimizer = StableAdamW(
        [{'params': trainable.parameters()}],
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        amsgrad=True,
        eps=1e-10
    )

    scheduler = WarmCosineScheduler(
        optimizer,
        base_value=lr,
        final_value=lr * 0.1,
        total_iters=n_iterations,
        warmup_iters=100
    )

    it = 0
    loss_history = []
    pbar = tqdm(total=n_iterations, desc="Training Dinomaly",
                dynamic_ncols=True, leave=True)

    for epoch in range(int(np.ceil(n_iterations / len(loader)))):
        model.train()
        for batch in loader:
            if it >= n_iterations:
                break

            img = batch['image'].to(device)

            en, de = model(img)

            # Progressive hard mining matching realiad_uni.py exactly
            p_final = 0.9
            p = min(p_final * it / 1000, p_final)
            loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
            optimizer.step()
            scheduler.step()

            loss_history.append(loss.item())
            it += 1
            pbar.update(1)

            if it % 1000 == 0:
                avg = np.mean(loss_history[-100:])
                pbar.set_postfix({'loss': f'{avg:.4f}'})

        if it >= n_iterations:
            break

    pbar.close()
    print(f"Training complete. Final loss: {np.mean(loss_history[-100:]):.4f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Weights saved to {save_path}")

    return model


def train_anomalydino_fewshot(
    train_df: pd.DataFrame,
    n_shots: int = 16,
    device: str = 'cuda',
    repo_path: str = '',
    save_path: str = None,
) -> object:
    """
    Build AnomalyDINO memory bank using n_shots normal images per
    category per viewpoint, following the few-shot evaluation protocol
    from Hofer et al. (2025).

    16 shots x 30 categories x 5 viewpoints = 2,400 training images.
    This is the setting for which AnomalyDINO was designed and validated
    in the original paper. The full multi-class setting (36,465 images)
    exceeded GPU VRAM during coreset consolidation, which is consistent
    with the paper's primary evaluation being the few-shot setting.

    Coreset subsampling is disabled — at 2,400 images the memory bank
    is small enough that subsampling would discard a meaningful fraction
    of the already limited reference set.

    GPU memory strategy: the entire torch_model is moved to CPU before
    fit() so that vstack and nearest-neighbour search run entirely in
    system RAM. The final memory bank is moved back to GPU for inference.

    Args:
        train_df:   dataframe from realiad_utils — normal images only
        n_shots:    reference images per category per viewpoint (default: 16)
        device:     'cuda' or 'cpu'
        repo_path:  path to BachelorsThesis repo
        save_path:  optional path to save memory bank

    Returns:
        AnomalyDINO model with populated memory bank
    """
    from anomalib.models import AnomalyDINO

    RealIADTorchDataset = _load_dataset_class(repo_path)

    normal_df = train_df[train_df['label'] == 0].reset_index(drop=True)

    shot_dfs = []
    for cat in normal_df['category'].unique():
        for vp in normal_df['viewpoint'].unique():
            subset = normal_df[
                (normal_df['category'] == cat) &
                (normal_df['viewpoint'] == vp)
            ].head(n_shots)
            shot_dfs.append(subset)

    fewshot_df = pd.concat(shot_dfs, ignore_index=True)
    print(f"AnomalyDINO few-shot: {len(fewshot_df)} reference images")
    print(f"  {n_shots} shots x "
          f"{normal_df['category'].nunique()} categories x "
          f"{normal_df['viewpoint'].nunique()} viewpoints")

    dataset = RealIADTorchDataset(fewshot_df, load_masks=False)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        collate_fn=_collate_fn
    )

    model = AnomalyDINO(
        encoder_name='dinov2reg_vit_base_14',
        coreset_subsampling=False,
        masking=False,
    )
    torch_model = model.model.to(device)
    torch_model.train()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Building few-shot memory bank",
                          dynamic_ncols=True, leave=True):
            images = batch['image'].to(device)
            torch_model(images)

    if not hasattr(torch_model, 'embedding_store') or \
            len(torch_model.embedding_store) == 0:
        raise RuntimeError(
            "embedding_store is empty after forward passes. "
            "Check that torch_model is in train mode."
        )

    print(f"Moving {len(torch_model.embedding_store)} "
          f"embedding tensors to CPU...")

    torch_model.embedding_store = [
        e.cpu() for e in torch_model.embedding_store
    ]
    torch.cuda.empty_cache()
    gc.collect()

    print("Running memory bank consolidation on CPU...")
    torch_model.to('cpu')
    torch_model.fit()
    torch_model.to(device)
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
    n_epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    inp_num: int = 6,
    device: str = 'cuda',
    repo_path: str = '',
    save_path: str = None,
) -> object:
    """
    Train INP-Former on normal images from train_df.

    Uses the official INP-Former repository (submodule at models/inp_former)
    with a standardised interface. Follows published paper defaults for Real-IAD
    except n_epochs which is set to 100 rather than the published 200.

    The effect of this compute reduction is explicitly investigated in
    Investigation 2 (compute equalisation ablation), which evaluates
    performance at 22 epochs (matching Dinomaly's compute budget) and
    100 epochs, making the training duration a scientific variable rather
    than an unexamined constraint.

    Uses RealIADTorchDataset (our unified loader) rather than the INP-Former
    repo's RealIADDataset, so the cross-view protocol works correctly by
    passing a filtered train_df — no changes to this function needed between
    protocols. Preprocessing is identical: 448 resize, 392 centre crop,
    ImageNet normalisation.

    The WarmCosineScheduler total_iters = n_epochs * len(loader), so
    reducing n_epochs automatically adjusts the LR schedule correctly.

    Published reference: INP-Former achieves 92.1% I-AUROC on Real-IAD
    multi-class setting at 200 epochs (Luo et al., CVPR 2025).

    Args:
        train_df:      dataframe from realiad_utils — normal images only
        dataset_root:  unused — kept for API consistency
        n_epochs:      number of training epochs (default: 100)
        batch_size:    training batch size (default: 16 per paper)
        lr:            learning rate (default: 1e-3 per paper)
        inp_num:       number of prototype tokens (default: 6 per paper)
        device:        'cuda' or 'cpu'
        repo_path:     path to BachelorsThesis repo
        save_path:     optional path to save model weights

    Returns:
        Trained INP-Former model ready for inference via run_inference_inpformer()
    """
    # Clear cached module imports that may point to Dinomaly's modules
    for key in list(sys.modules.keys()):
        if key.startswith('models') or key in ('utils', 'dataset', 'optimizers'):
            del sys.modules[key]

    _setup_inpformer_path(repo_path)

    from models import vit_encoder
    from models.uad import INP_Former
    from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
    from optimizers import StableAdamW
    from utils import WarmCosineScheduler, global_cosine_hm_adaptive, setup_seed
    from torch.nn.init import trunc_normal_

    setup_seed(1)

    # Use our unified dataset — accepts filtered dataframe directly
    # Cross-view: pass train_df filtered to C1+C2 from notebook
    # Standard: pass full train_df
    RealIADTorchDataset = _load_dataset_class(repo_path)
    normal_df = train_df[train_df['label'] == 0].reset_index(drop=True)
    dataset = RealIADTorchDataset(normal_df, load_masks=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        collate_fn=_collate_fn
    )

    print(f"Training INP-Former on {len(normal_df)} normal images "
          f"across {normal_df['category'].nunique()} categories")

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

    trainable = nn.ModuleList(
        [Bottleneck, INP_Guided_Decoder, INP_Extractor, INP])

    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # total_iters scales with n_epochs — schedule adjusts automatically
    # reducing epochs for cross-view or ablation correctly adjusts LR decay
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
        for batch in tqdm(
                loader, ncols=80, desc=f"Epoch {epoch+1}/{n_epochs}",
                dynamic_ncols=True, leave=False):
            img = batch['image'].to(device)
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

def run_inference_dinomaly(
    model,
    test_df: pd.DataFrame,
    device: str = 'cuda',
    batch_size: int = 16,
    repo_path: str = '',
    save_anomaly_maps: bool = False,
    maps_save_dir: str = None,
) -> pd.DataFrame:
    """
    Run Dinomaly inference using official repo's utility functions.

    Uses cal_anomaly_maps and get_gaussian_kernel from the Dinomaly repo's
    utils.py. Image score is mean of top 1% pixels (max_ratio=0.01),
    matching the official realiad_uni.py evaluation call.

    Args:
        model:             trained ViTill model from train_dinomaly()
        test_df:           dataframe with all test images
        device:            'cuda' or 'cpu'
        batch_size:        inference batch size (default: 16)
        repo_path:         path to BachelorsThesis repo
        save_anomaly_maps: whether to save anomaly maps to disk
        maps_save_dir:     root directory for saving maps

    Returns:
        DataFrame with all original columns plus image_score and model
    """
    _setup_dinomaly_path(repo_path)

    from utils import get_gaussian_kernel, cal_anomaly_maps
    from torch.nn import functional as F

    RealIADTorchDataset = _load_dataset_class(repo_path)
    dataset = RealIADTorchDataset(test_df, load_masks=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=_collate_fn
    )

    model.eval()
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

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
        for batch in tqdm(loader, desc="Inference [Dinomaly]",
                          dynamic_ncols=True, leave=True):
            img = batch['image'].to(device)
            img_paths = batch['image_path']

            en, de = model(img)

            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            anomaly_map = F.interpolate(
                anomaly_map, size=256,
                mode='bilinear', align_corners=False)
            anomaly_map = gaussian_kernel(anomaly_map)

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

                if save_anomaly_maps and maps_save_dir:
                    if path_to_label.get(path, -1) == 1:
                        category = path_to_category.get(path, 'unknown')
                        save_dir = os.path.join(
                            maps_save_dir, 'Dinomaly', category)
                        os.makedirs(save_dir, exist_ok=True)
                        stem = os.path.splitext(
                            os.path.basename(path))[0]
                        np.savez_compressed(
                            os.path.join(save_dir, f"{stem}.npz"),
                            anomaly_map=amap[0].astype(np.float32),
                            anomaly_score=np.float32(score_val))

    results_df = test_df.copy()
    path_to_score = dict(zip(all_paths, all_scores))
    results_df['image_score'] = results_df['image_path'].map(path_to_score)
    results_df['model'] = 'Dinomaly'

    print(f"\nDinomaly inference complete")
    print(f"Total images: {len(results_df)}")
    print(f"Score range: [{results_df['image_score'].min():.4f}, "
          f"{results_df['image_score'].max():.4f}]")
    if save_anomaly_maps:
        n_saved = int(results_df['label'].sum())
        print(f"Anomaly maps saved: {n_saved} anomalous images")

    return results_df


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
    Used for AnomalyDINO (Anomalib-based model).

    Args:
        model:             trained model
        test_df:           dataframe with all test images
        model_name:        'AnomalyDINO'
        device:            'cuda' or 'cpu'
        batch_size:        inference batch size (default: 16)
        repo_path:         path to BachelorsThesis repo
        max_ratio:         unused for AnomalyDINO — kept for API consistency
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
        num_workers=2,
        collate_fn=_collate_fn
    )

    model.eval()
    if hasattr(model, 'model'):
        model.model.eval()

    all_scores = []
    all_paths = []

    path_to_label = dict(zip(
        test_df['image_path'].tolist(),
        test_df['label'].tolist()
    ))
    path_to_category = dict(zip(
        test_df['image_path'].tolist(),
        test_df['category'].tolist()
    )) if 'category' in test_df.columns else {}

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Inference [{model_name}]",
                          dynamic_ncols=True, leave=True):
            images = batch['image'].to(device)
            output = model(images)
            scores = output.pred_score.cpu().numpy().flatten()
            amaps = output.anomaly_map.cpu().numpy()

            for score, path, amap in zip(
                    scores, batch['image_path'], amaps):
                score_val = float(score)
                all_scores.append(score_val)
                all_paths.append(path)

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
    full multi-category test_df directly — no category loop needed.

    Image score: mean of top 1% pixels per official INP-Former evaluation.

    Args:
        model:             trained INP-Former model
        test_df:           dataframe with all test images (all categories)
        dataset_root:      unused — kept for API consistency
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

    RealIADTorchDataset = _load_dataset_class(repo_path)
    dataset = RealIADTorchDataset(test_df, load_masks=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=_collate_fn
    )

    model.eval()
    gaussian_kernel = get_gaussian_kernel(
        kernel_size=5, sigma=4).to(device)

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
        for batch in tqdm(loader, desc="Inference [INP-Former]",
                          dynamic_ncols=True, leave=True):
            img = batch['image'].to(device)
            img_paths = batch['image_path']

            output = model(img)
            en, de = output[0], output[1]

            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            anomaly_map = F.interpolate(
                anomaly_map, size=256,
                mode='bilinear', align_corners=False)
            anomaly_map = gaussian_kernel(anomaly_map)

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
    CUDA events provide accurate GPU timing.

    Args:
        model:      trained model
        device:     'cuda' or 'cpu'
        n_warmup:   warmup runs to reach GPU steady state
        n_runs:     number of timed runs
        image_size: input image size (default: 392)

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
    print("  train_dinomaly(train_df, ...)               — Dinomaly (official repo)")
    print("  train_anomalydino_fewshot(train_df, ...)    — AnomalyDINO 16-shot")
    print("  train_inpformer(train_df, ...)              — INP-Former multi-class")
    print("  run_inference_dinomaly(model, test_df, ...) — Dinomaly")
    print("  run_inference(model, test_df, ...)          — AnomalyDINO")
    print("  run_inference_inpformer(model, test_df, ...) — INP-Former")
    print("  measure_inference_time(model, ...)")
    print("  measure_memory_footprint(model, ...)")