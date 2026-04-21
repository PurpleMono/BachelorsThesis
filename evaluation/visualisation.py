"""
evaluation/visualisation.py — Figures and plots for thesis results section.

Produces:
- Anomaly map comparison figures (all three models side by side)
- WGA heatmaps (category x viewpoint, category x defect type)
- WGA category table (unified coloured table for thesis)
- Score distribution plots
- Performance comparison bar charts
- Degradation ratio plots
- Ablation study comparison plots
- Efficiency trade-off scatter plot

Anomaly map selection rationale:
    Maps are generated for the top N disagreement groups, defined as
    category x viewpoint x defect_type combinations where the difference
    between the best and worst model I-AUROC is largest. This highlights
    cases where the detection paradigm matters most, rather than cases
    where all models fail equally. For each group, the representative
    image is the anomalous sample with the highest anomaly score from
    the best-performing model for that group.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.cm as cm
from pathlib import Path
from PIL import Image
from typing import Optional


MODEL_COLORS = {
    'Dinomaly': '#2196F3',
    'AnomalyDINO': '#FF9800',
    'INP-Former': '#4CAF50',
}

MODEL_ORDER = ['AnomalyDINO', 'Dinomaly', 'INP-Former']


# =============================================================================
# SECTION 1: ANOMALY MAP VISUALISATION
# =============================================================================

def plot_anomaly_map_comparison(
    image_path: str,
    anomaly_maps: dict,
    gt_mask: Optional[np.ndarray] = None,
    title: str = '',
    output_path: Optional[str] = None,
    figsize: tuple = (16, 4),
) -> None:
    """
    Plot original image, GT mask, and anomaly maps from all three models
    side by side for qualitative comparison.

    Selection rationale: called for highest-disagreement groups where
    model performance differs most, revealing paradigm-specific strengths.

    Args:
        image_path:   path to original image
        anomaly_maps: dict of model_name to anomaly map array (H x W)
        gt_mask:      ground truth binary mask (H x W), optional
        title:        figure title
        output_path:  path to save PNG, None = display inline
        figsize:      figure size in inches
    """
    image = np.array(Image.open(image_path).convert('RGB'))

    n_cols = 1 + (1 if gt_mask is not None else 0) + len(anomaly_maps)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]

    col = 0

    axes[col].imshow(image)
    axes[col].set_title('Input Image', fontsize=10, fontweight='bold')
    axes[col].axis('off')
    col += 1

    if gt_mask is not None:
        axes[col].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
        axes[col].set_title('Ground Truth', fontsize=10, fontweight='bold')
        axes[col].axis('off')
        col += 1

    for model_name in MODEL_ORDER:
        if model_name not in anomaly_maps:
            continue
        amap = anomaly_maps[model_name]
        amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        axes[col].imshow(amap_norm, cmap='jet', vmin=0, vmax=1)
        axes[col].set_title(
            model_name, fontsize=10, fontweight='bold',
            color=MODEL_COLORS.get(model_name, 'black'))
        axes[col].axis('off')
        col += 1

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_worst_group_examples(
    df_dict: dict,
    anomaly_maps_dict: dict,
    top_n: int = 5,
    min_disagreement: float = 0.3,
    output_dir: str = 'results/figures/worst_groups',
) -> list:
    """
    Generate anomaly map comparison figures for the highest-disagreement groups.

    Selection rationale:
    - Groups are defined as category x viewpoint x defect_type combinations
    - Disagreement = max model AUROC minus min model AUROC for that group
    - Only groups with disagreement above min_disagreement threshold are shown
    - This ensures figures illustrate genuine paradigm differences, not noise
    - Representative image: anomalous sample with highest score from best model

    Args:
        df_dict:           {'ModelName': results_df}
        anomaly_maps_dict: {'ModelName': {'image_path': anomaly_map_array}}
        top_n:             number of disagreement groups to visualise
        min_disagreement:  minimum AUROC disagreement to qualify for visualisation
        output_dir:        directory to save PNG figures

    Returns:
        list of output file paths
    """
    from evaluation.wga import find_disagreement_groups

    disagreement_df = find_disagreement_groups(df_dict, top_n=top_n)
    disagreement_df = disagreement_df[
        disagreement_df['disagreement'] >= min_disagreement]

    output_paths = []

    for idx, row in disagreement_df.reset_index().iterrows():
        category = row.get('category', '')
        viewpoint = row.get('viewpoint', '')
        defect_type = row.get('defect_type', '')

        first_model_df = list(df_dict.values())[0]
        mask = (
            (first_model_df['category'] == category) &
            (first_model_df['viewpoint'] == viewpoint) &
            (first_model_df['defect_type'] == defect_type) &
            (first_model_df['label'] == 1)
        )
        group_images = first_model_df[mask]['image_path'].tolist()

        if not group_images:
            continue

        # Select representative image: highest score from best model
        best_model = max(
            [m for m in MODEL_ORDER if m in df_dict],
            key=lambda m: row.get(m, 0) if not pd.isna(row.get(m, 0)) else 0
        )
        best_df = df_dict[best_model]
        best_mask = (
            (best_df['category'] == category) &
            (best_df['viewpoint'] == viewpoint) &
            (best_df['defect_type'] == defect_type) &
            (best_df['label'] == 1)
        )
        best_group = best_df[best_mask].sort_values(
            'image_score', ascending=False)
        if best_group.empty:
            image_path = group_images[0]
        else:
            image_path = best_group.iloc[0]['image_path']

        amaps = {}
        for model_name, amap_dict in anomaly_maps_dict.items():
            if image_path in amap_dict:
                amaps[model_name] = amap_dict[image_path]

        gt_mask = None
        if not first_model_df[mask].empty:
            row_data = first_model_df[mask].iloc[0]
            if row_data.get('has_mask') and pd.notna(row_data.get('mask_path')):
                mask_path = row_data['mask_path']
                if Path(mask_path).exists():
                    gt_mask = np.array(
                        Image.open(mask_path).convert('L')) / 255.0

        title = (f"Disagreement Group {idx+1}: {category} | "
                 f"{viewpoint} | {defect_type}")
        output_path = (
            f"{output_dir}/disagreement_{idx+1}_"
            f"{category}_{viewpoint}_{defect_type}.png")

        plot_anomaly_map_comparison(
            image_path=image_path,
            anomaly_maps=amaps,
            gt_mask=gt_mask,
            title=title,
            output_path=output_path
        )
        output_paths.append(output_path)

    print(f"Generated {len(output_paths)} anomaly map figures")
    return output_paths


# =============================================================================
# SECTION 2: WGA HEATMAPS
# =============================================================================

def _build_wga_pivot(df_dict: dict, row_col: str, col_col: str) -> dict:
    """Helper: build pivot tables for given row and column dimensions."""
    from evaluation.metrics import compute_i_auroc

    matrices = {}
    all_rows = set()
    all_cols = set()

    for model_name, df in df_dict.items():
        from evaluation.wga import compute_wga as _cwga
        wga_df = _cwga(df, group_cols=[row_col, col_col])
        if wga_df.empty:
            continue
        pivot = wga_df.pivot_table(
            index=row_col, columns=col_col, values='auroc')
        matrices[model_name] = pivot
        all_rows.update(pivot.index.tolist())
        all_cols.update(pivot.columns.tolist())

    return matrices, sorted(all_rows), sorted(all_cols)


def _render_heatmap(
    matrices: dict,
    all_rows: list,
    all_cols: list,
    row_label: str,
    col_label: str,
    title: str,
    output_path: Optional[str],
    figsize_per_model: tuple = (5, 8),
) -> None:
    """Helper: render a heatmap for given matrices."""
    model_names = [m for m in MODEL_ORDER if m in matrices]
    n_models = len(model_names)
    n_rows = len(all_rows)

    fig_width = figsize_per_model[0] * n_models + 1
    fig_height = max(figsize_per_model[1], n_rows * 0.35)

    fig, axes = plt.subplots(1, n_models,
                              figsize=(fig_width, fig_height), sharey=True)
    if n_models == 1:
        axes = [axes]

    im = None
    for ax, model_name in zip(axes, model_names):
        matrix = matrices[model_name].reindex(
            index=all_rows, columns=all_cols)
        im = ax.imshow(matrix.values, aspect='auto',
                       cmap='RdYlGn', vmin=0.5, vmax=1.0)

        for i in range(len(all_rows)):
            for j in range(len(all_cols)):
                val = matrix.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=7,
                            color='black' if 0.6 < val < 0.9 else 'white')

        ax.set_title(model_name, fontsize=11, fontweight='bold',
                     color=MODEL_COLORS.get(model_name, 'black'))
        ax.set_xticks(range(len(all_cols)))
        ax.set_xticklabels(all_cols, fontsize=9, rotation=30, ha='right')
        ax.set_yticks(range(len(all_rows)))
        ax.set_yticklabels(all_rows, fontsize=8)
        ax.set_xlabel(col_label, fontsize=9)

    axes[0].set_ylabel(row_label, fontsize=9)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, label='I-AUROC',
                            shrink=0.6, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_wga_heatmap_viewpoint(
    df_dict: dict,
    output_path: Optional[str] = None,
) -> None:
    """WGA heatmap: category x viewpoint for all three models."""
    matrices, all_cats, all_views = _build_wga_pivot(
        df_dict, 'category', 'viewpoint')
    _render_heatmap(
        matrices, all_cats, all_views,
        row_label='Category', col_label='Viewpoint',
        title='WGA: I-AUROC by Category and Viewpoint',
        output_path=output_path,
        figsize_per_model=(5, max(8, len(all_cats) * 0.35))
    )


def plot_wga_heatmap_defect(
    df_dict: dict,
    output_path: Optional[str] = None,
) -> None:
    """WGA heatmap: category x defect type for all three models."""
    matrices, all_cats, all_defects = _build_wga_pivot(
        df_dict, 'category', 'defect_type')
    _render_heatmap(
        matrices, all_cats, all_defects,
        row_label='Category', col_label='Defect Type',
        title='WGA: I-AUROC by Category and Defect Type',
        output_path=output_path,
        figsize_per_model=(5, max(8, len(all_cats) * 0.35))
    )


def plot_wga_category_table(
    df_dict: dict,
    output_path: Optional[str] = None,
    figsize: tuple = (10, 12),
) -> None:
    """
    Unified category-level WGA table with all three models as columns.
    Cells are coloured by I-AUROC value (red to green).
    Suitable for direct thesis inclusion — rows sorted by mean AUROC ascending.
    """
    from evaluation.wga import wga_by_category

    combined = wga_by_category(df_dict)
    combined['mean'] = combined.mean(axis=1)
    combined = combined.sort_values('mean')

    model_names = [m for m in MODEL_ORDER if m in combined.columns]
    data = combined[model_names].values
    categories = combined.index.tolist()

    cmap = plt.get_cmap('RdYlGn')
    norm = Normalize(vmin=0.5, vmax=1.0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    table = ax.table(
        cellText=[[f'{v:.3f}' for v in row] for row in data],
        rowLabels=categories,
        colLabels=model_names,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#DDDDDD')
            cell.set_text_props(fontweight='bold')
        elif col >= 0 and row > 0:
            val = data[row - 1][col]
            if not np.isnan(val):
                rgba = cmap(norm(val))
                cell.set_facecolor(rgba)
                luminance = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
                cell.set_text_props(
                    color='black' if luminance > 0.5 else 'white')

    ax.set_title('Per-Category I-AUROC (sorted by mean, ascending)',
                 fontsize=12, fontweight='bold', pad=20)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='I-AUROC',
                 fraction=0.02, pad=0.04)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


# =============================================================================
# SECTION 3: SCORE DISTRIBUTIONS
# =============================================================================

def plot_score_distributions(
    df_dict: dict,
    output_path: Optional[str] = None,
    figsize: tuple = (15, 4),
) -> None:
    """
    Score distribution histograms: normal vs anomalous for each model.
    Shows class separation quality — well-separated distributions
    indicate a well-calibrated model.
    """
    model_names = [m for m in MODEL_ORDER if m in df_dict]
    n_models = len(model_names)

    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        df = df_dict[model_name]
        normal = df[df['label'] == 0]['image_score'].dropna()
        anomalous = df[df['label'] == 1]['image_score'].dropna()

        ax.hist(normal, bins=50, alpha=0.6, color='#2196F3',
                label='Normal', density=True)
        ax.hist(anomalous, bins=50, alpha=0.6, color='#F44336',
                label='Anomalous', density=True)

        ax.set_title(model_name, fontsize=11, fontweight='bold',
                     color=MODEL_COLORS.get(model_name, 'black'))
        ax.set_xlabel('Anomaly Score', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    fig.suptitle('Anomaly Score Distributions: Normal vs Anomalous',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


# =============================================================================
# SECTION 4: PERFORMANCE COMPARISON CHARTS
# =============================================================================

def plot_performance_comparison(
    results_std: dict,
    results_cv: dict,
    metric: str = 'I-AUROC',
    output_path: Optional[str] = None,
    figsize: tuple = (10, 5),
) -> None:
    """
    Grouped bar chart: standard vs cross-view protocol per model.
    """
    model_names = [m for m in MODEL_ORDER if m in results_std]
    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    std_vals = [results_std[m] for m in model_names]
    cv_vals = [results_cv.get(m, 0) for m in model_names]
    colors = [MODEL_COLORS[m] for m in model_names]

    bars1 = ax.bar(x - width/2, std_vals, width,
                   label='Standard Protocol', color=colors, alpha=0.9)
    bars2 = ax.bar(x + width/2, cv_vals, width,
                   label='Cross-View Protocol', color=colors,
                   alpha=0.5, hatch='//')

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003,
                f'{bar.get_height():.3f}',
                ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model', fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title(f'{metric}: Standard vs Cross-Viewpoint Protocol',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_degradation_ratios(
    degradation_dict: dict,
    output_path: Optional[str] = None,
    figsize: tuple = (8, 5),
) -> None:
    """
    Bar chart showing performance degradation ratio per model
    under the cross-viewpoint protocol.
    Higher = more sensitive to viewpoint shift.
    """
    model_names = [m for m in MODEL_ORDER if m in degradation_dict]
    values = [degradation_dict[m] for m in model_names]
    colors = [MODEL_COLORS[m] for m in model_names]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(model_names, values, color=colors, alpha=0.85, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.2,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax.set_xlabel('Model', fontsize=10)
    ax.set_ylabel('Degradation (%)', fontsize=10)
    ax.set_title(
        'Performance Degradation: Standard vs Cross-Viewpoint Protocol',
        fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(values) * 1.2 if values else 10)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_per_category_comparison(
    df_dict: dict,
    metric_fn,
    metric_name: str = 'I-AUROC',
    output_path: Optional[str] = None,
    figsize: tuple = (18, 6),
) -> None:
    """
    Per-category grouped bar chart across all three models.
    Categories on x-axis, one bar group per category.
    """
    model_names = [m for m in MODEL_ORDER if m in df_dict]
    all_cats = sorted(list(df_dict.values())[0]['category'].unique())
    x = np.arange(len(all_cats))
    width = 0.25

    fig, ax = plt.subplots(figsize=figsize)

    for i, model_name in enumerate(model_names):
        df = df_dict[model_name]
        values = []
        for cat in all_cats:
            cat_df = df[df['category'] == cat]
            values.append(metric_fn(cat_df) if len(cat_df) > 0
                          else float('nan'))

        offset = (i - len(model_names)/2 + 0.5) * width
        ax.bar(x + offset, values, width,
               label=model_name,
               color=MODEL_COLORS[model_name],
               alpha=0.85)

    ax.set_xlabel('Category', fontsize=9)
    ax.set_ylabel(metric_name, fontsize=9)
    ax.set_title(f'Per-Category {metric_name} Comparison',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_cats, rotation=45, ha='right', fontsize=7)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


# =============================================================================
# SECTION 5: ABLATION STUDY PLOTS
# =============================================================================

def plot_ablation_volume(
    abl1_summary: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: tuple = (12, 5),
) -> None:
    """
    Bar chart comparing standard, cross-view, and volume-compensated results.
    Investigation 1: does more compute recover the viewpoint gap?
    """
    model_names = [m for m in MODEL_ORDER
                   if m in abl1_summary['Model'].values]

    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=figsize)

    conditions = ['Standard', 'Cross-View', 'Cross-View + Volume']
    col_keys = ['Standard', 'Cross-View', 'Cross-View + Volume']
    alphas = [0.9, 0.6, 0.85]
    hatches = ['', '//', 'xx']

    for i, (cond, col, alpha, hatch) in enumerate(
            zip(conditions, col_keys, alphas, hatches)):
        vals = []
        for m in model_names:
            row = abl1_summary[abl1_summary['Model'] == m]
            vals.append(float(row[col].values[0]) if len(row) > 0
                        else float('nan'))

        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width,
                      label=cond,
                      color=[MODEL_COLORS[m] for m in model_names],
                      alpha=alpha, hatch=hatch)

    ax.set_xlabel('Model', fontsize=10)
    ax.set_ylabel('I-AUROC', fontsize=10)
    ax.set_title(
        'Ablation 1: Does Training Volume Compensate for Viewpoint Gap?',
        fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_ablation_singleclass(
    abl2_summary: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: tuple = (10, 5),
) -> None:
    """
    Grouped bar chart: AnomalyDINO single-class vs multi-class per category.
    Investigation 2: cost of the multi-class setting for memory-based methods.
    """
    categories = abl2_summary[
        abl2_summary['Category'] != 'Mean']['Category'].tolist()
    single = abl2_summary[
        abl2_summary['Category'] != 'Mean']['Single-Class I-AUROC'].values
    multi = abl2_summary[
        abl2_summary['Category'] != 'Mean']['Multi-Class I-AUROC'].values

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(x - width/2, single, width,
           label='Single-Class', color=MODEL_COLORS['AnomalyDINO'],
           alpha=0.9)
    ax.bar(x + width/2, multi, width,
           label='Multi-Class', color=MODEL_COLORS['AnomalyDINO'],
           alpha=0.5, hatch='//')

    ax.set_xlabel('Category', fontsize=10)
    ax.set_ylabel('I-AUROC', fontsize=10)
    ax.set_title(
        'Ablation 2: AnomalyDINO Single-Class vs Multi-Class',
        fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha='right', fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_ablation_compute(
    abl3_summary: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: tuple = (8, 5),
) -> None:
    """
    Bar chart: standard vs compute-equalised performance for Dinomaly
    and INP-Former. Investigation 3: does compute disparity explain
    performance differences?
    """
    models = abl3_summary['Model'].tolist()
    std_vals = abl3_summary['Standard I-AUROC'].tolist()
    eq_vals = abl3_summary['Equalised I-AUROC'].tolist()

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(x - width/2, std_vals, width,
           label='Published Schedule',
           color=[MODEL_COLORS.get(m, '#888') for m in models],
           alpha=0.9)
    ax.bar(x + width/2, eq_vals, width,
           label='Equalised Compute',
           color=[MODEL_COLORS.get(m, '#888') for m in models],
           alpha=0.5, hatch='//')

    for i, (s, e) in enumerate(zip(std_vals, eq_vals)):
        ax.text(i - width/2, s + 0.003, f'{s:.3f}',
                ha='center', va='bottom', fontsize=8)
        ax.text(i + width/2, e + 0.003, f'{e:.3f}',
                ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model', fontsize=10)
    ax.set_ylabel('I-AUROC', fontsize=10)
    ax.set_title(
        'Ablation 3: Effect of Compute Equalisation on Performance',
        fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


# =============================================================================
# SECTION 6: EFFICIENCY TRADE-OFF
# =============================================================================

def plot_efficiency_tradeoff(
    timing_dict: dict,
    auroc_dict: dict,
    output_path: Optional[str] = None,
    figsize: tuple = (8, 6),
) -> None:
    """
    Scatter plot: I-AUROC vs inference time per image.
    Relevant for the business implications section of the discussion —
    shows the practical accuracy-latency trade-off between paradigms.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for model_name in MODEL_ORDER:
        if model_name not in timing_dict or model_name not in auroc_dict:
            continue
        x = timing_dict[model_name]
        y = auroc_dict[model_name]
        ax.scatter(x, y, s=200,
                   color=MODEL_COLORS[model_name],
                   label=model_name, zorder=5)
        ax.annotate(model_name, (x, y),
                    textcoords='offset points',
                    xytext=(8, 4), fontsize=9)

    ax.set_xlabel('Inference Time (ms/image)', fontsize=10)
    ax.set_ylabel('I-AUROC', fontsize=10)
    ax.set_title('Accuracy vs Inference Time Trade-off',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == '__main__':
    print("Testing visualisation functions...")

    np.random.seed(42)
    n = 300

    def make_df(model_name):
        return pd.DataFrame({
            'image_score': np.concatenate([
                np.random.normal(0.3, 0.1, n//2),
                np.random.normal(0.7, 0.1, n//2)
            ]),
            'label': [0] * (n//2) + [1] * (n//2),
            'category': np.random.choice(
                ['audiojack', 'pcb', 'usb'], n),
            'viewpoint': np.tile(['C1','C2','C3','C4','C5'], n//5),
            'defect_type': np.random.choice(['CH', 'AK', 'OK'], n),
            'model': model_name,
        })

    df_dict = {
        'Dinomaly': make_df('Dinomaly'),
        'AnomalyDINO': make_df('AnomalyDINO'),
        'INP-Former': make_df('INP-Former'),
    }

    print("Testing plot_score_distributions...")
    plot_score_distributions(df_dict, output_path='/tmp/test_distributions.png')

    print("Testing plot_wga_heatmap_viewpoint...")
    plot_wga_heatmap_viewpoint(df_dict,
                               output_path='/tmp/test_heatmap_viewpoint.png')

    print("Testing plot_wga_heatmap_defect...")
    plot_wga_heatmap_defect(df_dict,
                            output_path='/tmp/test_heatmap_defect.png')

    print("Testing plot_wga_category_table...")
    plot_wga_category_table(df_dict,
                            output_path='/tmp/test_category_table.png')

    print("Testing plot_performance_comparison...")
    plot_performance_comparison(
        results_std={'Dinomaly': 0.89, 'AnomalyDINO': 0.72,
                     'INP-Former': 0.91},
        results_cv={'Dinomaly': 0.81, 'AnomalyDINO': 0.65,
                    'INP-Former': 0.84},
        output_path='/tmp/test_comparison.png'
    )

    print("Testing plot_degradation_ratios...")
    plot_degradation_ratios(
        {'Dinomaly': 9.0, 'AnomalyDINO': 9.7, 'INP-Former': 7.7},
        output_path='/tmp/test_degradation.png'
    )

    print("Testing plot_efficiency_tradeoff...")
    plot_efficiency_tradeoff(
        timing_dict={'Dinomaly': 18.0, 'AnomalyDINO': 8.5,
                     'INP-Former': 22.0},
        auroc_dict={'Dinomaly': 0.89, 'AnomalyDINO': 0.72,
                    'INP-Former': 0.91},
        output_path='/tmp/test_efficiency.png'
    )

    print("All visualisation tests passed.")