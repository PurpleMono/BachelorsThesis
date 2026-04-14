import torch
import adeval
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from typing import Optional


# =============================================================================
# DATASET CONFIGS
# Controls which metrics are computed per dataset.
# Add a new config here if you add a new dataset.
# =============================================================================

REALIAD_CONFIG = {
    'name': 'realiad',
    'has_sample_id': True,   # enables S-AUROC
    'has_viewpoints': True,  # enables viewpoint dimension in WGA
}

MVTEC_CONFIG = {
    'name': 'mvtec',
    'has_sample_id': False,
    'has_viewpoints': False,
}


# =============================================================================
# SECTION 1: IMAGE-LEVEL METRICS
# =============================================================================

def compute_i_auroc(df: pd.DataFrame) -> float:
    """
    Standard image-level AUROC.
    All NG images labeled as anomalous regardless of defect visibility.
    Used for comparability with published results (Dinomaly 89.3%, INP-Former 90.5%).
    Requires columns: image_score, label
    """
    return roc_auc_score(df['label'], df['image_score'])


def compute_gt_i_auroc(df: pd.DataFrame) -> float:
    """
    GT-based image-level AUROC for Real-IAD.
    NG images without a visible defect are treated as normal (label_gt=0)
    because the defect is physically not visible from that viewpoint.
    Fairer than standard I-AUROC — used as basis for WGA.
    Requires columns: image_score, label_gt
    """
    return roc_auc_score(df['label_gt'], df['image_score'])


def compute_s_auroc(df: pd.DataFrame) -> float:
    """
    Sample-level AUROC for Real-IAD.
    Aggregates max anomaly score across all views per physical object.
    Models the production line decision: does this object get rejected?
    Requires columns: image_score, label, sample_id
    """
    # Take max score and max label across all views per sample
    sample_df = df.groupby('sample_id').agg(
        sample_score=('image_score', 'max'),
        label=('label', 'max')
    ).reset_index()
    return roc_auc_score(sample_df['label'], sample_df['sample_score'])


# =============================================================================
# SECTION 2: PIXEL-LEVEL METRICS
# Both only computed on images where has_mask=True
# pixel_scores and pixel_labels are flattened arrays added during inference
# =============================================================================

def compute_p_auroc_aupro(df: pd.DataFrame) -> tuple:
    """
    Pixel-level AUROC, AUPR and AUPRO using official ADEval implementation.
    Ensures comparability with Real-IAD leaderboard results.
    Only computed on images where GT mask exists (has_mask=True).
    Requires columns: pixel_scores_2d, pixel_labels_2d, has_mask
    Returns: (p_auroc, p_aupr, aupro)
    """
    df_masked = df[df['has_mask'] == True]

    if df_masked.empty:
        return float('nan'), float('nan'), float('nan')

    # Patch numpy for ADEval compatibility — np.trapz renamed to np.trapezoid in numpy 2.x
    if not hasattr(np, 'trapz'):
        np.trapz = np.trapezoid

    # ADEval requires float32 predictions and uint8 targets
    preds = [arr.astype(np.float32) for arr in df_masked['pixel_scores_2d'].values]
    targets = [arr.astype(np.uint8) for arr in df_masked['pixel_labels_2d'].values]

    p_auroc, p_aupr, aupro = adeval.auroc_aupr_aupro(preds, targets)
    return p_auroc, p_aupr, aupro


# =============================================================================
# SECTION 3: ROBUSTNESS METRIC
# =============================================================================

def compute_degradation_ratio(
    auroc_standard: float,
    auroc_crossview: float
) -> float:
    """
    Performance degradation between standard and cross-view protocol.
    Quantifies viewpoint robustness as a single number.
    Higher value = more degradation = less robust to viewpoint shift.
    Returns percentage drop.
    """
    return (auroc_standard - auroc_crossview) / auroc_standard * 100


# =============================================================================
# SECTION 4: WORST-GROUP ANALYSIS
# =============================================================================

def compute_wga(
    df: pd.DataFrame,
    group_cols: list = ['category', 'viewpoint', 'defect_type'],
    score_col: str = 'image_score',
    label_col: str = 'label'  # ← change default here
) -> pd.DataFrame:
    """
    Worst-Group Analysis across specified dimensions.
    Returns AUROC per group sorted ascending (worst group first).
    

    Args:
        df:            results dataframe with model scores
        group_cols:    dimensions to group by
        score_col:    column name for model scores
        label_col:    column name for binary labels to compute AUROC                                       

    Returns:
        DataFrame with AUROC per group, sorted ascending (worst group first)
    """
    results = []

    for group_vals, group_df in df.groupby(group_cols):
        # Skip groups with only one class — AUROC undefined
        if len(group_df[label_col].unique()) < 2:
            continue
        try:
            auroc = roc_auc_score(
                group_df[label_col],
                group_df['image_score']
            )
            results.append({
                **dict(zip(group_cols,
                           group_vals if isinstance(group_vals, tuple)
                           else (group_vals,))),
                'auroc': auroc,
                'n_samples': len(group_df),
                'n_anomal': int(group_df[label_col].sum())
            })
        except Exception:
            continue

    return pd.DataFrame(results).sort_values('auroc').reset_index(drop=True)


# =============================================================================
# SECTION 5: CONVENIENCE FUNCTION
# Computes all relevant metrics at once based on dataset config
# =============================================================================

def compute_all_metrics(
    df: pd.DataFrame,
    config: dict,
    compute_pixel: bool = True
) -> dict:
    """
    Compute all relevant metrics for a given dataset config.

    Args:
        df:            results dataframe with model scores
        config:        dataset config e.g. REALIAD_CONFIG or MVTEC_CONFIG
        compute_pixel: whether to compute pixel-level metrics
                       set False if pixel_scores not yet available

    Returns:
        dict with all computed metric values
    """
    metrics = {}

    # Always compute standard I-AUROC
    metrics['i_auroc'] = compute_i_auroc(df)

    if config['has_sample_id']:
        metrics['s_auroc'] = compute_s_auroc(df)

    # Pixel-level metrics — only when pixel scores are available
    if compute_pixel:
        p_auroc, p_aupr, aupro = compute_p_auroc_aupro(df)
        metrics['p_auroc'] = p_auroc
        metrics['p_aupr'] = p_aupr
        metrics['aupro'] = aupro

    return metrics


# =============================================================================
# SMOKE TEST
# Run directly to verify functions work on synthetic data
# python evaluation/metrics.py
# =============================================================================

if __name__ == '__main__':
    np.random.seed(42)
    n = 150  # 30 samples x 5 views

    # Synthetic dataframe simulating Real-IAD structure
    df = pd.DataFrame({
        'image_score': np.random.rand(n),
        'label':       np.random.randint(0, 2, n),
        'sample_id':   np.repeat([f'cat_S{i:04d}' for i in range(30)], 5),
        'category':    np.random.choice(['audiojack', 'pcb'], n),
        'viewpoint':   np.tile(['C1', 'C2', 'C3', 'C4', 'C5'], 30),
        'defect_type': np.random.choice(['CH', 'AK', 'OK'], n),
        'has_mask':    np.random.choice([True, False], n),
        # Synthetic pixel scores — normally added during inference
        'pixel_scores_2d': [np.random.rand(10, 10) for _ in range(n)],
        'pixel_labels_2d': [np.random.randint(0, 2, (10, 10)) for _ in range(n)]
    })

    print("Testing compute_i_auroc:")
    print(round(compute_i_auroc(df), 4))

    print("\nTesting compute_s_auroc:")
    print(round(compute_s_auroc(df), 4))

    print("\nTesting compute_p_auroc_aupro:")
    p_auroc, p_aupr, aupro = compute_p_auroc_aupro(df)
    print(f"  p_auroc: {round(p_auroc, 4)}")
    print(f"  p_aupr:  {round(p_aupr, 4)}")
    print(f"  aupro:   {round(aupro, 4)}")

    print("\nTesting compute_degradation_ratio:")
    print(round(compute_degradation_ratio(0.91, 0.85), 2), "%")

    print("\nTesting compute_wga (worst 5 groups):")
    print(compute_wga(df).head())

    print("\nTesting compute_all_metrics with REALIAD_CONFIG:")
    from evaluation.metrics import REALIAD_CONFIG
    results = compute_all_metrics(df, REALIAD_CONFIG)
    for k, v in results.items():
        print(f"  {k}: {round(v, 4)}")