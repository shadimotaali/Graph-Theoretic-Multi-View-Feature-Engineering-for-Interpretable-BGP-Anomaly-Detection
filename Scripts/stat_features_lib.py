# =============================================================================
# stat_features_lib.py  —  Pure-function library for BGP statistical feature extraction
# =============================================================================
# Import this module; do NOT execute as a script.
# Key fix vs auto-generated version:
#   - RARE_AS_THRESHOLD is a PARAMETER of extract_statistical_features(),
#     not a global captured at import time.
#   - No download/parse loops run at import time.
# =============================================================================

import logging
import os
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


# =============================================================================
# AS-Path Cleaning (unified with graph_features_lib logic)
# =============================================================================

def parse_as_path_clean(as_path_str: str) -> str:
    """
    Clean an AS_PATH string for statistical analysis.
    - Removes AS-SETs (skip the whole set token)
    - Removes AS prepending duplicates
    - Filters private/reserved ASNs (RFC 6996, RFC 7300)
    Returns a space-separated string of valid public ASNs.

    NOTE: This is the canonical cleaner used throughout the unified pipeline.
    Use this instead of the simpler 'clean_as_path' that appeared in older code.
    """
    if not as_path_str:
        return ''
    tokens = as_path_str.split()
    deduped = []
    in_as_set = False
    for token in tokens:
        if '{' in token:
            in_as_set = True
        if in_as_set:
            if '}' in token:
                in_as_set = False
            continue
        if '}' in token:
            continue
        try:
            asn = int(token)
        except ValueError:
            continue
        # Filter private/reserved ASNs (RFC 6996, RFC 7300)
        if (64512 <= asn <= 65534) or (4200000000 <= asn <= 4294967294):
            continue
        if asn in {0, 23456, 65535, 4294967295}:
            continue
        if not deduped or asn != int(deduped[-1]):
            deduped.append(str(asn))
    return ' '.join(deduped)


# =============================================================================
# MRT Download Helpers
# =============================================================================

def generate_update_urls(collector: str, start_date: str, end_date: str):
    """
    Generate URLs for all UPDATE dump files in the given date range.
    RIPE RIS publishes updates every 5 minutes.
    Returns list of (url, filename, datetime) tuples.
    """
    urls = []
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    current = start_dt
    while current < end_dt:
        year_month = current.strftime('%Y.%m')
        filename = f"updates.{current.strftime('%Y%m%d.%H%M')}.gz"
        url = f"https://data.ris.ripe.net/{collector}/{year_month}/{filename}"
        urls.append((url, filename, current))
        current += timedelta(minutes=5)
    return urls


def download_mrt_files(urls, output_dir: Path, progress_callback=None):
    """
    Download MRT UPDATE files, skipping already-downloaded ones.
    Returns list of (local_path, datetime) tuples.
    """
    local_paths = []
    for url, filename, ts in tqdm(urls, desc='Downloading UPDATE files'):
        if progress_callback is not None:
            progress_callback(f"Downloading UPDATE file {filename}")
        local_path = output_dir / filename
        if local_path.exists() and local_path.stat().st_size > 0:
            local_paths.append((local_path, ts))
            continue
        try:
            temp_path = local_path.with_name(
                f".{local_path.name}.{os.getpid()}.{threading.get_ident()}.part"
            )
            if temp_path.exists():
                temp_path.unlink()
            urlretrieve(url, temp_path)
            if local_path.exists() and local_path.stat().st_size > 0:
                temp_path.unlink(missing_ok=True)
            else:
                os.replace(temp_path, local_path)
            local_paths.append((local_path, ts))
        except (URLError, HTTPError) as e:
            logger.warning(f"Failed to download {filename}: {e}")
    return local_paths


# =============================================================================
# Edit Distance
# =============================================================================

def calculate_edit_distance(path1, path2) -> int:
    """
    Compute Levenshtein edit distance between two AS paths.
    Accepts either lists of ASNs or space-separated strings.
    """
    if not path1 or not path2:
        return 0
    if isinstance(path1, str):
        path1 = [a for a in path1.split() if a.isdigit()]
    if isinstance(path2, str):
        path2 = [a for a in path2.split() if a.isdigit()]
    if not path1 or not path2:
        return 0
    m, n = len(path1), len(path2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if path1[i-1] == path2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]


# =============================================================================
# Statistical Feature Extraction
# =============================================================================

def extract_statistical_features(df_window: pd.DataFrame,
                                   prev_prefixes=None,
                                   rare_as_threshold: int = 3):
    """
    Extract all statistical features from a single time window DataFrame.

    The DataFrame must have columns:
        timestamp, type ('A'/'W'), peer_ip, peer_as, prefix,
        as_path, as_path_clean, origin, next_hop

    Args:
        df_window:          DataFrame of UPDATE records for this window.
        prev_prefixes:      Set of prefixes seen in the previous window
                            (used to compute new_prefixes). Pass None for first window.
        rare_as_threshold:  ASes appearing fewer times than this are 'rare'.
                            Default 3 (matching original notebook config).

    Returns:
        features:          dict of feature_name -> value
        current_prefixes:  set of prefixes in this window (pass to next call)
    """
    features = {}

    announcements = df_window[df_window['type'] == 'A']
    withdrawals   = df_window[df_window['type'] == 'W']

    # ── VOLUME FEATURES ──
    features['announcements']  = len(announcements)
    features['withdrawals']    = len(withdrawals)
    features['total_updates']  = len(df_window)

    window_seconds = max(1, (
        df_window['timestamp'].max() - df_window['timestamp'].min()
    ).total_seconds())
    features['ann_rate'] = features['announcements'] / window_seconds
    features['wd_rate']  = features['withdrawals']   / window_seconds

    features['wd_ann_ratio'] = (
        features['withdrawals'] / features['announcements']
        if features['announcements'] > 0 else 0
    )
    features['ann_wd_ratio'] = (
        features['announcements'] / features['withdrawals']
        if features['withdrawals'] > 0 else 0
    )

    # ── PREFIX FEATURES ──
    ann_prefixes = set(announcements['prefix'].dropna()) if not announcements.empty else set()
    wd_prefixes  = set(withdrawals['prefix'].dropna())   if not withdrawals.empty  else set()

    features['unique_prefixes_ann'] = len(ann_prefixes)
    features['unique_prefixes_wd']  = len(wd_prefixes)
    features['new_prefixes'] = len(ann_prefixes - prev_prefixes) if prev_prefixes is not None else 0

    current_prefixes = ann_prefixes | wd_prefixes

    # Duplicate announcements
    if not announcements.empty:
        dup_cols = [c for c in ['peer_ip', 'peer_as', 'prefix', 'as_path', 'origin', 'next_hop']
                    if c in announcements.columns]
        counts = announcements.groupby(dup_cols).size()
        features['dups'] = int(sum(c - 1 for c in counts if c > 1))
    else:
        features['dups'] = 0

    # Flaps: prefix both withdrawn AND announced in the same window
    features['flaps'] = len(wd_prefixes & ann_prefixes)

    # ── ORIGIN FEATURES ──
    if not announcements.empty and 'origin' in announcements.columns:
        origin_counts = announcements['origin'].value_counts()
        features['origin_IGP']        = int(origin_counts.get('IGP', 0))
        features['origin_INCOMPLETE'] = int(origin_counts.get('INCOMPLETE', 0))
        prefix_origins = announcements.groupby('prefix')['origin'].nunique()
        features['origin_changes'] = int((prefix_origins > 1).sum())
    else:
        features['origin_IGP'] = 0
        features['origin_INCOMPLETE'] = 0
        features['origin_changes'] = 0

    # ── AS PATH FEATURES ──
    valid_paths = announcements[
        announcements['as_path_clean'].notna() & (announcements['as_path_clean'] != '')
    ] if not announcements.empty else pd.DataFrame()

    if not valid_paths.empty:
        path_lengths = valid_paths['as_path_clean'].apply(
            lambda p: len(p.split()) if isinstance(p, str) else 0
        )
        features['as_path_avg'] = float(path_lengths.mean())
        features['as_path_max'] = int(path_lengths.max())
        features['as_path_std'] = float(path_lengths.std()) if len(path_lengths) > 1 else 0.0

        unique_paths_per_prefix = valid_paths.groupby('prefix')['as_path_clean'].nunique()
        features['unique_as_path_max'] = int(unique_paths_per_prefix.max())

        # Edit distances between consecutive paths per prefix
        edit_distances = []
        edit_distance_dict = defaultdict(list)
        for prefix, group in valid_paths.groupby('prefix'):
            if len(group) >= 2:
                sorted_group = group.sort_values('timestamp')
                prev_path = None
                for _, row in sorted_group.iterrows():
                    cur_path = row['as_path_clean']
                    if prev_path is not None:
                        dist = calculate_edit_distance(prev_path, cur_path)
                        edit_distances.append(dist)
                        edit_distance_dict[prefix].append(dist)
                    prev_path = cur_path

        if edit_distances:
            features['edit_distance_avg'] = float(np.mean(edit_distances))
            features['edit_distance_max'] = int(max(edit_distances))
            ed_counter = Counter(edit_distances)
            for i in range(7):
                features[f'edit_distance_dict_{i}'] = ed_counter.get(i, 0)
            unique_ed: dict = {}
            for prefix, dists in edit_distance_dict.items():
                for d in set(dists):
                    unique_ed[d] = unique_ed.get(d, 0) + 1
            for i in range(2):
                features[f'edit_distance_unique_dict_{i}'] = unique_ed.get(i, 0)
        else:
            features['edit_distance_avg'] = 0.0
            features['edit_distance_max'] = 0
            for i in range(7):
                features[f'edit_distance_dict_{i}'] = 0
            for i in range(2):
                features[f'edit_distance_unique_dict_{i}'] = 0

        # ── RARE AS FEATURES ──
        all_asns = []
        for path in valid_paths['as_path_clean']:
            if isinstance(path, str) and path:
                all_asns.extend(path.split())
        asn_counts = Counter(all_asns)
        rare_asns = [a for a, c in asn_counts.items() if c < rare_as_threshold]
        features['number_rare_ases'] = len(rare_asns)
        features['rare_ases_ratio']  = len(rare_asns) / len(all_asns) if all_asns else 0.0

    else:
        for k in ['as_path_avg', 'as_path_std', 'edit_distance_avg']:
            features[k] = 0.0
        for k in ['as_path_max', 'unique_as_path_max', 'edit_distance_max',
                  'number_rare_ases']:
            features[k] = 0
        features['rare_ases_ratio'] = 0.0
        for i in range(7):
            features[f'edit_distance_dict_{i}'] = 0
        for i in range(2):
            features[f'edit_distance_unique_dict_{i}'] = 0

    # ── IMPLICIT WITHDRAWAL FEATURES ──
    if not announcements.empty:
        imp_wd = imp_wd_spath = imp_wd_dpath = 0
        for (prefix, peer), group in announcements.groupby(['prefix', 'peer_ip']):
            if len(group) > 1:
                imp_wd += 1
                if group['as_path_clean'].nunique() == 1:
                    imp_wd_spath += 1
                else:
                    imp_wd_dpath += 1
        features['imp_wd']       = imp_wd
        features['imp_wd_spath'] = imp_wd_spath
        features['imp_wd_dpath'] = imp_wd_dpath
    else:
        features['imp_wd'] = features['imp_wd_spath'] = features['imp_wd_dpath'] = 0

    # ── BEHAVIORAL FEATURES ──
    specific_prefixes = sum(
        1 for p in df_window['prefix'].dropna()
        if isinstance(p, str) and '/32' in p
    )
    features['nadas'] = specific_prefixes + (10 if features['wd_ann_ratio'] > 0.5 else 0)
    features['unique_peers'] = df_window['peer_as'].nunique()

    return features, current_prefixes


# =============================================================================
# Selected feature lists (for post-processing / column selection)
# =============================================================================

KEEP_STATISTICAL = [
    'announcements', 'withdrawals', 'wd_ann_ratio',
    'unique_prefixes_ann', 'new_prefixes', 'flaps',
    'origin_changes',
    'as_path_avg', 'as_path_max', 'unique_as_path_max',
    'edit_distance_avg', 'edit_distance_max',
    'number_rare_ases', 'imp_wd_dpath', 'unique_peers',
]

DROP_STATISTICAL = [
    'total_updates', 'ann_rate', 'wd_rate', 'ann_wd_ratio',
    'unique_prefixes_wd', 'dups', 'origin_IGP', 'origin_INCOMPLETE',
    'as_path_std',
    'edit_distance_dict_0', 'edit_distance_dict_1', 'edit_distance_dict_2',
    'edit_distance_dict_3', 'edit_distance_dict_4', 'edit_distance_dict_5',
    'edit_distance_dict_6',
    'edit_distance_unique_dict_0', 'edit_distance_unique_dict_1',
    'rare_ases_ratio', 'imp_wd', 'imp_wd_spath', 'nadas',
]
