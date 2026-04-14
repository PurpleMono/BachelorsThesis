import re
import json
import pandas as pd
from pathlib import Path
from typing import Optional


def parse_realiad_filename(filename: str) -> dict:
    """Parse metadata from a Real-IAD filename."""
    stem = Path(filename).stem
    parts = stem.split('_')

    # Find viewpoint (C1-C5) — search explicitly due to underscores in category names
    viewpoint, viewpoint_idx = None, None
    for i, part in enumerate(parts):
        if re.match(r'^C[1-5]$', part):
            viewpoint, viewpoint_idx = part, i
            break

    if viewpoint is None:
        return {'sample_number': None, 'status': None,
                'defect_type': None, 'viewpoint': None}

    # Find status (OK/NG) by searching backwards from viewpoint
    status, status_idx = None, None
    for i in range(viewpoint_idx - 1, -1, -1):
        if parts[i] in ('OK', 'NG'):
            status, status_idx = parts[i], i
            break

    # Defect type sits between status and viewpoint for NG images
    if status == 'NG' and status_idx is not None:
        defect_parts = parts[status_idx + 1: viewpoint_idx]
        defect_type = '_'.join(defect_parts) if defect_parts else 'NG'
    else:
        defect_type = 'OK'

    # Sample number is directly before the status
    sample_number = parts[status_idx - 1] if status_idx else None

    return {
        'sample_number': sample_number,
        'status': status,
        'defect_type': defect_type,
        'viewpoint': viewpoint
    }


def load_realiad_category(
    category_root: str,
    viewpoints: Optional[list] = None,
    json_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Load one Real-IAD category into a dataframe.
    Uses JSON anomaly_class field for labels — matches convention used by
    all published papers (Dinomaly, INP-Former, Anomalib).
    label=1 when anomaly_class != 'OK', label=0 when anomaly_class == 'OK'.
    This correctly handles views where defect is not visible (labeled as normal).

    Args:
        category_root:  path to category folder e.g. '/data/realiad_512/audiojack'
        viewpoints:     restrict to specific views e.g. ['C1','C2'], None = all
        json_path:      path to JSON file. If None, looks in standard location.

    Returns:
        DataFrame with columns:
            image_path, mask_path, category, sample_id,
            viewpoint, defect_type, anomaly_class, label, has_mask, split
    """
    category_path = Path(category_root)
    category = category_path.name

    # Resolve JSON path — standard Real-IAD structure
    if json_path is None:
        json_file = (category_path.parent.parent /
                     'realiad_jsons' / 'realiad_jsons' /
                     f'{category}.json')
    else:
        json_file = Path(json_path)

    if not json_file.exists():
        raise FileNotFoundError(
            f"JSON file not found at {json_file}. "
            f"Provide json_path explicitly if your structure differs."
        )

    with open(json_file, 'r') as f:
        data = json.load(f)

    train_entries = data.get('train', [])
    test_entries = data.get('test', [])
    train_set = set(id(e) for e in train_entries)

    rows = []
    for split_name, entries in [('train', train_entries), ('test', test_entries)]:
        for entry in entries:
            img_path = category_path / entry['image_path']

            # Skip missing files
            if not img_path.exists():
                continue

            # Parse viewpoint and sample metadata from filename
            meta = parse_realiad_filename(img_path.name)
            if meta['viewpoint'] is None:
                continue
            if viewpoints and meta['viewpoint'] not in viewpoints:
                continue

            # Label from JSON anomaly_class — matches published paper convention
            anomaly_class = entry.get('anomaly_class', 'OK')
            label = 0 if anomaly_class == 'OK' else 1

            # GT mask from JSON
            mask_path = entry.get('mask_path')
            has_mask = mask_path is not None
            full_mask_path = str(category_path / mask_path) if has_mask else None

            rows.append({
                'image_path': str(img_path),
                'mask_path': full_mask_path,
                'category': category,
                'sample_id': f"{category}_S{meta['sample_number']}",
                'viewpoint': meta['viewpoint'],
                'defect_type': meta['defect_type'],
                'anomaly_class': anomaly_class,
                'label': label,
                'has_mask': has_mask,
                'split': split_name
            })

    df = pd.DataFrame(rows)

    if df.empty:
        print(f"Warning: no images found for {category}")
        return df

    print(f"Loaded {category}: {len(df)} images | "
          f"normal={len(df[df['label']==0])} | "
          f"anomalous={len(df[df['label']==1])} | "
          f"with_mask={df['has_mask'].sum()}")
    return df


def load_realiad_all(
    data_root: str,
    categories: Optional[list] = None,
    viewpoints: Optional[list] = None
) -> pd.DataFrame:
    """
    Load all Real-IAD categories into one combined dataframe.
    Calls load_realiad_category for each category folder and concatenates results.

    Args:
        data_root:   root folder containing all category subfolders and realiad_jsons
        categories:  list of category names to load, None = all
        viewpoints:  list of viewpoints to include, None = all five
    """
    dfs = []

    for folder in sorted(Path(data_root).iterdir()):
        if not folder.is_dir():
            continue
        if categories and folder.name not in categories:
            continue

        json_file = (Path(data_root) / 'realiad_jsons' /
                     'realiad_jsons' / f'{folder.name}.json')

        try:
            df = load_realiad_category(
                str(folder),
                viewpoints=viewpoints,
                json_path=str(json_file) if json_file.exists() else None
            )
            if not df.empty:
                dfs.append(df)
        except FileNotFoundError as e:
            print(f"Skipping {folder.name}: {e}")

    if not dfs:
        raise ValueError(f"No data loaded from {data_root}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal: {len(combined)} images across "
          f"{combined['category'].nunique()} categories")
    return combined


def get_crossview_split(
    df: pd.DataFrame,
    train_views: list = ['C1', 'C2'],
    test_views: list = ['C3', 'C4', 'C5']
) -> tuple:
    """
    Split dataframe into train/test sets by viewpoint.
    Used for the cross-viewpoint robustness protocol.
    Normal-only filtering for model training happens in the notebooks.
    """
    train_df = df[df['viewpoint'].isin(train_views)].reset_index(drop=True)
    test_df = df[df['viewpoint'].isin(test_views)].reset_index(drop=True)
    print(f"Cross-view split: train={len(train_df)} {train_views} | "
          f"test={len(test_df)} {test_views}")
    return train_df, test_df


if __name__ == '__main__':
    # Terminal smoke test
    # Usage: python data/realiad_utils.py /path/to/realiad/audiojack
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data/realiad_utils.py <category_path>")
        sys.exit(0)

    df = load_realiad_category(sys.argv[1])
    print(f"\nColumns: {list(df.columns)}")
    print(f"Viewpoints: {sorted(df['viewpoint'].unique())}")
    print(f"Defect types: {sorted(df['defect_type'].unique())}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    print(f"\nFirst 10 rows:")
    print(df[['sample_id', 'viewpoint', 'defect_type',
              'label', 'has_mask', 'split']].head(10))

    train_df, test_df = get_crossview_split(df)