"""Post-labelling periodicity filter.

After unsupervised labelling assigns binary_label, this module detects
anomaly windows that belong to a periodic operational pattern (e.g.,
BGP route-refresh cycles) and relabels them as normal — UNLESS other
features confirm genuine anomalous behaviour.

Algorithm:
    1. Compute autocorrelation of each candidate feature's time series.
    2. If an ACF peak exceeds `acf_threshold` at some lag, a periodic
       signal is present.
    3. For anomaly windows where the periodic feature exceeds
       `amplitude_factor × normal_max`:
       a. Count how many OTHER features fall outside the normal
          P1–P99 range.
       b. If fewer than `min_corroborating_features` are outside
          normal → relabel as normal (pure periodic artifact).
       c. Otherwise keep the anomaly label (genuine incident
          coinciding with a periodic spike).

Usage:
    from periodicity_filter import filter_periodic_anomalies
    df = filter_periodic_anomalies(df, label_col="binary_label")
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.fft import fft, ifft
from scipy.signal import find_peaks


def _autocorrelation(x: np.ndarray) -> np.ndarray:
    """Normalized autocorrelation via FFT."""
    n = len(x)
    x_centered = x - x.mean()
    f = fft(x_centered, n=2 * n)
    acf = np.real(ifft(f * np.conj(f)))[:n]
    if acf[0] > 0:
        acf /= acf[0]
    return acf


def detect_periodic_feature(
    series: np.ndarray,
    *,
    acf_threshold: float = 0.08,
    min_lag: int = 4,
    max_lag: int = 100,
) -> tuple[int | None, float]:
    """Find the dominant periodic lag in a time series.

    Returns (lag_in_windows, acf_value) or (None, 0.0) if no periodicity.
    """
    acf = _autocorrelation(series)
    search = acf[min_lag:max_lag + 1]
    peaks, _ = find_peaks(search, height=acf_threshold, distance=3)

    if len(peaks) == 0:
        return None, 0.0

    best_idx = peaks[np.argmax(search[peaks])]
    best_lag = best_idx + min_lag
    return best_lag, acf[best_lag]


def _count_anomalous_features(
    row: pd.Series,
    normal_bounds: dict[str, tuple[float, float]],
) -> int:
    """Count how many features fall outside normal P1–P99 range."""
    count = 0
    for feat, (lo, hi) in normal_bounds.items():
        val = row[feat]
        if val < lo or val > hi:
            count += 1
    return count


def filter_periodic_anomalies(
    df: pd.DataFrame,
    *,
    label_col: str = "binary_label",
    time_col: str = "window_start",
    candidate_features: list[str] | None = None,
    corroborating_features: list[str] | None = None,
    acf_threshold: float = 0.08,
    amplitude_factor: float = 5.0,
    period_tolerance_windows: int = 2,
    min_chain_length: int = 5,
    min_corroborating_features: int = 2,
    verbose: bool = True,
) -> pd.DataFrame:
    """Filter periodic-pattern anomalies and relabel them as normal.

    Parameters
    ----------
    df : DataFrame with labelled windows (must be sorted by time).
    label_col : column containing binary labels (0=normal, 1=anomaly).
    time_col : timestamp column.
    candidate_features : features to check for periodicity.
    corroborating_features : OTHER features to check before relabelling.
        If enough of these are also outside normal range, the anomaly
        label is preserved.  Default: all numeric columns except
        label_col and the periodic feature itself.
    acf_threshold : minimum ACF peak to consider periodic.
    amplitude_factor : spike must exceed normal_max × this factor.
    period_tolerance_windows : ± tolerance when checking period match.
    min_chain_length : minimum spike count to confirm periodicity.
    min_corroborating_features : keep anomaly label if this many OTHER
        features are outside normal P1–P99 range.
    verbose : print diagnostics.

    Returns
    -------
    DataFrame with updated labels and a new 'periodic_relabel' column
    (True for windows that were relabelled from anomaly to normal).
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.sort_values(time_col).reset_index(drop=True)
    df["periodic_relabel"] = False

    if candidate_features is None:
        candidate_features = df.select_dtypes("number").columns.tolist()
        candidate_features = [c for c in candidate_features if c != label_col]

    normal_mask = df[label_col] == 0
    anomaly_mask = df[label_col] == 1
    total_relabelled = 0
    total_kept = 0

    for feat in candidate_features:
        series = df[feat].to_numpy(dtype=np.float64)
        series = np.nan_to_num(series, nan=0.0)

        lag, acf_val = detect_periodic_feature(
            series, acf_threshold=acf_threshold
        )
        if lag is None:
            continue

        normal_max = df.loc[normal_mask, feat].max()
        spike_threshold = normal_max * amplitude_factor

        if spike_threshold <= 0:
            continue

        spike_mask = anomaly_mask & (df[feat] > spike_threshold)
        n_spikes = spike_mask.sum()

        if n_spikes < min_chain_length:
            continue

        spike_times = df.loc[spike_mask, time_col].values
        window_size_sec = 300
        period_sec = lag * window_size_sec
        tol_sec = period_tolerance_windows * window_size_sec

        ts_sec = (spike_times - spike_times[0]).astype("timedelta64[s]").astype(float)
        if len(ts_sec) > 1:
            intervals = np.diff(ts_sec)
            remainders = intervals % period_sec
            at_period = np.sum(
                (remainders <= tol_sec) | ((period_sec - remainders) <= tol_sec)
            )
            periodic_frac = at_period / len(intervals)
        else:
            periodic_frac = 0.0

        if periodic_frac < 0.3:
            continue

        # --- Multi-feature consistency check ---
        check_feats = corroborating_features
        if check_feats is None:
            check_feats = [
                c for c in df.select_dtypes("number").columns
                if c not in (label_col, feat, "periodic_relabel")
            ]

        normal_bounds = {}
        for cf in check_feats:
            p1 = df.loc[normal_mask, cf].quantile(0.01)
            p99 = df.loc[normal_mask, cf].quantile(0.99)
            normal_bounds[cf] = (p1, p99)

        n_relabelled = 0
        n_kept = 0
        spike_indices = df.index[spike_mask]
        for idx in spike_indices:
            n_outside = _count_anomalous_features(
                df.loc[idx], normal_bounds
            )
            if n_outside < min_corroborating_features:
                df.at[idx, label_col] = 0
                df.at[idx, "periodic_relabel"] = True
                n_relabelled += 1
            else:
                n_kept += 1

        total_relabelled += n_relabelled
        total_kept += n_kept

        if verbose:
            print(f"  [{feat}] Period={lag * 5}min (ACF={acf_val:.3f}), "
                  f"threshold={spike_threshold:.0f}, "
                  f"periodic_frac={periodic_frac:.0%}, "
                  f"relabelled={n_relabelled}, "
                  f"kept={n_kept} (≥{min_corroborating_features} "
                  f"corroborating features)")

    if verbose:
        remaining_anom = (df[label_col] == 1).sum()
        print(f"\n  Total relabelled: {total_relabelled}, "
              f"kept (multi-feature): {total_kept}")
        print(f"  Remaining anomalies: {remaining_anom} "
              f"({remaining_anom / len(df) * 100:.2f}%)")

    return df