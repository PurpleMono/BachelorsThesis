import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional
from evaluation.metrics import compute_wga


# =============================================================================
# wga.py — Worst-Group Analysis
#
# This file builds on compute_wga() from metrics.py.
# metrics.py answers: "what is the AUROC per group?"
# wga.py answers:     "what do those numbers mean across models?"
#
# All functions take a df_dict as input:
#   df_dict = {
#       'AnomalyDINO': df_anomalydino,
#       'Dinomaly':    df_dinomaly,
#       'INP-Former':  df_inpformer
#   }
# Each dataframe must already have image_score computed from model inference.
# =============================================================================


# =============================================================================
# SECTION 1: AGGREGATION FUNCTIONS
# Compute GT-I-AUROC at different levels of granularity.
# Always compares all three models side by side.
# =============================================================================

def _auroc_by_group(df: pd.DataFrame,
                    group_col: str) -> pd.Series:
    """
    Helper: compute mean GT-I-AUROC grouped by a single column.
    Returns a Series indexed by group value.
    """
    
    # Use single-column WGA
    wga_df = compute_wga(df, group_cols=[group_col])
    
    if wga_df.empty:
        return pd.Series(dtype=float)
    
    return wga_df.set_index(group_col)['auroc']


def wga_by_category(df_dict: dict) -> pd.DataFrame:
    """
    Level 1: GT-I-AUROC per object category for each model.
    30 rows x 3 model columns.
    Sorted by mean AUROC ascending — hardest categories at top.

    Args:
        df_dict: {'ModelName': results_df, ...}

    Returns:
        DataFrame with categories as rows, models as columns
    """
    results = {}
    for model_name, df in df_dict.items():
        results[model_name] = _auroc_by_group(df, 'category')

    combined = pd.DataFrame(results)
    combined['mean'] = combined.mean(axis=1)
    return combined.sort_values('mean').drop(columns='mean')


def wga_by_viewpoint(df_dict: dict) -> pd.DataFrame:
    """
    Level 2: GT-I-AUROC per camera viewpoint for each model.
    5 rows x 3 model columns.
    Directly relevant to cross-viewpoint RQ.

    Args:
        df_dict: {'ModelName': results_df, ...}

    Returns:
        DataFrame with viewpoints as rows, models as columns
    """
    results = {}
    for model_name, df in df_dict.items():
        results[model_name] = _auroc_by_group(df, 'viewpoint')

    combined = pd.DataFrame(results)
    combined['mean'] = combined.mean(axis=1)
    return combined.sort_values('mean').drop(columns='mean')


def wga_by_defect_type(df_dict: dict) -> pd.DataFrame:
    """
    Level 3: GT-I-AUROC per defect type for each model.
    5-8 rows x 3 model columns.
    Shows which defect types each model struggles with.

    Args:
        df_dict: {'ModelName': results_df, ...}

    Returns:
        DataFrame with defect types as rows, models as columns
    """
    results = {}
    for model_name, df in df_dict.items():
        results[model_name] = _auroc_by_group(df, 'defect_type')

    combined = pd.DataFrame(results)
    combined['mean'] = combined.mean(axis=1)
    return combined.sort_values('mean').drop(columns='mean')


def wga_detailed(df_dict: dict,
                 category: str) -> pd.DataFrame:
    """
    Level 4: Fine-grained breakdown for one specific category.
    Shows viewpoint x defect_type combinations for that category only.
    Use for the worst-performing category from Level 1.

    Args:
        df_dict:  {'ModelName': results_df, ...}
        category: category name to drill into e.g. 'audiojack'

    Returns:
        DataFrame with viewpoint x defect_type groups as rows
    """

    results = {}
    for model_name, df in df_dict.items():
        # Filter to this category only
        df_cat = df[df['category'] == category].copy()
        wga_df = compute_wga(
            df_cat,
            group_cols=['viewpoint', 'defect_type']
        )
        if not wga_df.empty:
            wga_df = wga_df.set_index(['viewpoint', 'defect_type'])
            results[model_name] = wga_df['auroc']

    combined = pd.DataFrame(results)
    combined['mean'] = combined.mean(axis=1)
    return combined.sort_values('mean').drop(columns='mean')


# =============================================================================
# SECTION 2: DISAGREEMENT ANALYSIS
# Find groups where models differ most — most interesting for thesis discussion
# =============================================================================

def find_disagreement_groups(df_dict: dict,
                             group_cols: list = ['category',
                                                 'viewpoint',
                                                 'defect_type'],
                             top_n: int = 10) -> pd.DataFrame:
    """
    Find groups where models disagree most.
    High disagreement = one model fails badly, another succeeds.
    These are the most interesting cases for anomaly map visualisation
    and for answering RQ2 about detection mechanism differences.

    Args:
        df_dict:    {'ModelName': results_df, ...}
        group_cols: dimensions to group by
        top_n:      number of top disagreement groups to return

    Returns:
        DataFrame with top_n groups sorted by AUROC disagreement descending
    """

    all_wga = {}
    for model_name, df in df_dict.items():
        wga_df = compute_wga(df, group_cols=group_cols)
        if not wga_df.empty:
            all_wga[model_name] = wga_df.set_index(group_cols)['auroc']

    combined = pd.DataFrame(all_wga)

    # Disagreement = difference between best and worst model per group
    combined['max_auroc'] = combined.max(axis=1)
    combined['min_auroc'] = combined.min(axis=1)
    combined['disagreement'] = combined['max_auroc'] - combined['min_auroc']

    return (combined
            .sort_values('disagreement', ascending=False)
            .head(top_n)
            .drop(columns=['max_auroc', 'min_auroc']))


def get_worst_samples(df_dict: dict,
                      group_cols: list = ['category',
                                          'viewpoint',
                                          'defect_type'],
                      top_n: int = 10) -> pd.DataFrame:
    """
    Retrieve image paths for the top_n highest disagreement groups.
    Used to feed into visualisation.py for anomaly map comparison.

    Args:
        df_dict:    {'ModelName': results_df, ...}
        group_cols: dimensions to group by
        top_n:      number of disagreement groups to retrieve samples for

    Returns:
        DataFrame with image paths and scores for worst disagreement cases
    """
    disagreement_df = find_disagreement_groups(
        df_dict, group_cols=group_cols, top_n=top_n
    )

    # Get the group identifiers of the worst cases
    worst_groups = disagreement_df.reset_index()[group_cols].values.tolist()

    # Retrieve actual image rows from each model's dataframe
    sample_rows = []
    for group_vals in worst_groups:
        group_filter = dict(zip(group_cols, group_vals))
        for model_name, df in df_dict.items():
            mask = pd.Series([True] * len(df))
            for col, val in group_filter.items():
                mask = mask & (df[col] == val)
            group_df = df[mask].copy()
            group_df['model'] = model_name
            sample_rows.append(group_df)

    if not sample_rows:
        return pd.DataFrame()

    return pd.concat(sample_rows, ignore_index=True)


# =============================================================================
# SECTION 3: VISUALISATION
# Heatmaps for thesis figures — category x viewpoint grid coloured by AUROC
# =============================================================================

def plot_wga_heatmap(df_dict: dict,
                     output_path: Optional[str] = None) -> None:
    """
    Plot a heatmap comparing GT-I-AUROC across categories and viewpoints
    for all three models side by side.
    Each column is one model, rows are categories, colour is AUROC.
    Low AUROC = red (struggling), high AUROC = green (performing well).

    Args:
        df_dict:     {'ModelName': results_df, ...}
        output_path: path to save the figure e.g. 'results/figures/wga_heatmap.png'
                     None = display inline
    """
    from evaluation.metrics import compute_wga

    model_names = list(df_dict.keys())
    n_models = len(model_names)

    # Build one heatmap matrix per model: categories x viewpoints
    matrices = {}
    all_categories = set()
    all_viewpoints = set()

    for model_name, df in df_dict.items():
        wga_df = compute_wga(df, group_cols=['category', 'viewpoint'])
        if wga_df.empty:
            continue
        pivot = wga_df.pivot_table(
            index='category',
            columns='viewpoint',
            values='auroc'
        )
        matrices[model_name] = pivot
        all_categories.update(pivot.index.tolist())
        all_viewpoints.update(pivot.columns.tolist())

    # Sort consistently
    all_categories = sorted(all_categories)
    all_viewpoints = sorted(all_viewpoints)

    fig, axes = plt.subplots(
        1, n_models,
        figsize=(6 * n_models, max(8, len(all_categories) * 0.4)),
        sharey=True
    )
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        if model_name not in matrices:
            continue

        matrix = matrices[model_name].reindex(
            index=all_categories,
            columns=all_viewpoints
        )

        im = ax.imshow(
            matrix.values,
            aspect='auto',
            cmap='RdYlGn',      # red=bad, yellow=medium, green=good
            vmin=0.5,
            vmax=1.0
        )

        ax.set_title(model_name, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(all_viewpoints)))
        ax.set_xticklabels(all_viewpoints)
        ax.set_yticks(range(len(all_categories)))
        ax.set_yticklabels(all_categories, fontsize=8)
        ax.set_xlabel('Viewpoint')

    axes[0].set_ylabel('Category')
    fig.colorbar(im, ax=axes, label='GT-I-AUROC')
    fig.suptitle('Worst-Group Analysis: GT-I-AUROC by Category × Viewpoint',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved heatmap to {output_path}")
    else:
        plt.show()


# =============================================================================
# SECTION 4: SUMMARY PRINTER
# Clean formatted output for notebook exploration
# =============================================================================

def print_wga_summary(df_dict: dict) -> None:
    """
    Print a clean summary of WGA results at all granularity levels.
    Use in notebooks to get a quick overview before diving into details.

    Args:
        df_dict: {'ModelName': results_df, ...}
    """
    print("=" * 60)
    print("WORST-GROUP ANALYSIS SUMMARY")
    print("=" * 60)

    print("\nLEVEL 1 — By Category (worst 5):")
    print(wga_by_category(df_dict).head(5).round(4).to_string())

    print("\nLEVEL 2 — By Viewpoint:")
    print(wga_by_viewpoint(df_dict).round(4).to_string())

    print("\nLEVEL 3 — By Defect Type:")
    print(wga_by_defect_type(df_dict).round(4).to_string())

    print("\nTOP 5 DISAGREEMENT GROUPS:")
    print(find_disagreement_groups(df_dict, top_n=5).round(4).to_string())


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == '__main__':
    import numpy as np
    from evaluation.metrics import REALIAD_CONFIG

    np.random.seed(42)
    n = 150

    # Synthetic results dataframe simulating post-inference structure
    def make_synthetic_df():
        return pd.DataFrame({
            'image_score': np.random.rand(n),
            'label':       np.random.randint(0, 2, n),
            'label_gt':    np.random.randint(0, 2, n),
            'sample_id':   np.repeat([f'cat_S{i:04d}' for i in range(30)], 5),
            'category':    np.random.choice(
                ['audiojack', 'pcb', 'button_battery'], n),
            'viewpoint':   np.tile(['C1', 'C2', 'C3', 'C4', 'C5'], 30),
            'defect_type': np.random.choice(['CH', 'AK', 'OK'], n),
            'has_mask':    np.random.choice([True, False], n),
        })

    df_dict = {
        'AnomalyDINO': make_synthetic_df(),
        'Dinomaly':    make_synthetic_df(),
        'INP-Former':  make_synthetic_df()
    }

    print("Testing wga_by_category:")
    print(wga_by_category(df_dict).round(4))

    print("\nTesting wga_by_viewpoint:")
    print(wga_by_viewpoint(df_dict).round(4))

    print("\nTesting wga_by_defect_type:")
    print(wga_by_defect_type(df_dict).round(4))

    print("\nTesting find_disagreement_groups:")
    print(find_disagreement_groups(df_dict, top_n=5).round(4))

    print("\nTesting print_wga_summary:")
    print_wga_summary(df_dict)