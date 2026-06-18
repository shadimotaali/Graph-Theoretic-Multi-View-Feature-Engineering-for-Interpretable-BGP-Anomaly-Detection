"""Standalone port of Scripts/BGP_Label_Validation_Discovery_ HDBSCAN.ipynb.

Runs the same five-method consensus labeller used to produce the paper's
`discovered_label` column:

    IsolationForest + LocalOutlierFactor + Statistical(Z+IQR) +
    EllipticEnvelope(on PCA) + HDBSCAN(on PCA)
    -> weighted (inverse-correlation) consensus
    -> 4-class threshold: likely_normal / uncertain / likely_anomaly / high_confidence_anomaly
    -> K-Means subtype upgrade of 'uncertain' in small clusters -> 'likely_anomaly'

Usage (single CSV):
    python Scripts/relabel_via_hdbscan.py \
        --input /path/to/features.csv --output-dir /path/to/out

CORAL diagnostic mode (label both before- and after-CORAL source for each
pair and emit a churn + transition CSV):
    python Scripts/relabel_via_hdbscan.py --coral-diagnostic

The port is intended to be a faithful reproduction of the labeling cells
(1, 2, 4-9, 11-19, 21, 26-27) from the notebook; visualization, bootstrap
stability, and feature-analysis cells are deliberately omitted.
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

META_COLS = {
    "incident", "window_start", "window_end", "window_id", "timestamp", "time",
    "label", "label_rule", "label_refined", "label_discovered", "discovered_label",
    "cluster", "hdbscan_cluster", "kmeans_cluster",
    "anomaly_score", "anomaly_votes", "consensus_score",
    "iso_forest_score", "lof_score", "statistical_score", "elliptic_score",
    "hdbscan_outlier_score",
    "source", "collector", "segment", "asn",
    # Phase 3 specific metadata
    "event_id", "binary_label", "provenance", "incident_type",
}


@dataclass
class LabelConfig:
    anomaly_rate_mode: str = "natural"
    fixed_contamination: float = 0.10
    n_estimators: int = 200
    lof_candidates: tuple = (5, 10, 15, 20, 30, 40, 50)
    iqr_multiplier: float = 1.5
    z_score_threshold: float = 3.0
    stat_mad_multiplier: float = 3.0
    pca_variance_target: float = 0.95
    min_samples_hdbscan: int = 5
    hdbscan_min_cluster_frac: float = 0.005
    hdbscan_min_cluster_floor: int = 20
    hdbscan_knee_search_frac: float = 0.30
    hdbscan_top_gaps: int = 2
    hdbscan_outlier_fallback: float = 0.90
    hdbscan_unscored_as_suspicious: bool = True
    elliptic_min_sample_feature_ratio: float = 5.0
    use_weighted_consensus: bool = True
    kmeans_small_cluster_frac: float = 0.05
    kmeans_upgrade_uncertain: bool = True
    kmeans_max_k: int = 10
    seed: int = 42


@dataclass
class LabelResult:
    discovered_label: np.ndarray
    consensus_score: np.ndarray
    thresholds: dict
    method_weights: pd.Series
    method_flags: pd.DataFrame
    active_methods: list
    valid_mask: np.ndarray
    feature_cols: list
    n_pca_components: int
    kmeans_boosted_count: int
    method_anomaly_rates: dict
    meta: dict = field(default_factory=dict)


def _feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    meta_cols_lower = {m.lower() for m in META_COLS}
    candidate_cols = [c for c in df.columns if c.lower() not in meta_cols_lower]
    feature_df = df[candidate_cols].select_dtypes(include=[np.number]).copy()
    feature_cols = feature_df.columns.tolist()
    return feature_df, feature_cols


def _tune_lof_n_neighbors(X_scaled: np.ndarray, cfg: LabelConfig) -> int:
    candidates = [n for n in cfg.lof_candidates if n < len(X_scaled) // 2]
    if not candidates:
        return max(10, min(50, int(np.sqrt(len(X_scaled)))))
    contam = "auto" if cfg.anomaly_rate_mode == "natural" else cfg.fixed_contamination
    rates = []
    for n in candidates:
        lof = LocalOutlierFactor(n_neighbors=n, contamination=contam, novelty=False)
        preds = lof.fit_predict(X_scaled)
        scores = lof.negative_outlier_factor_
        if cfg.anomaly_rate_mode == "natural":
            mask = scores < lof.offset_
        else:
            mask = preds == -1
        rates.append((n, float(mask.mean())))
    stability = []
    for idx, (n, rate) in enumerate(rates):
        diffs = []
        if idx > 0:
            diffs.append(abs(rate - rates[idx - 1][1]))
        if idx < len(rates) - 1:
            diffs.append(abs(rate - rates[idx + 1][1]))
        stability.append((n, float(np.mean(diffs)) if diffs else 0.0))
    prefer_n = max(10, min(50, int(np.sqrt(len(X_scaled)))))
    best_n, _ = min(stability, key=lambda x: (x[1], abs(x[0] - prefer_n)))
    return best_n


def _run_isolation_forest(X_scaled: np.ndarray, cfg: LabelConfig) -> np.ndarray:
    contam = "auto" if cfg.anomaly_rate_mode == "natural" else cfg.fixed_contamination
    iso = IsolationForest(n_estimators=cfg.n_estimators, contamination=contam,
                          random_state=cfg.seed, n_jobs=-1)
    iso.fit(X_scaled)
    scores = iso.decision_function(X_scaled)
    if cfg.anomaly_rate_mode == "natural":
        return scores < 0
    preds = iso.predict(X_scaled)
    return preds == -1


def _run_lof(X_scaled: np.ndarray, n_neighbors: int, cfg: LabelConfig) -> np.ndarray:
    contam = "auto" if cfg.anomaly_rate_mode == "natural" else cfg.fixed_contamination
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contam, novelty=False)
    preds = lof.fit_predict(X_scaled)
    scores = lof.negative_outlier_factor_
    if cfg.anomaly_rate_mode == "natural":
        return scores < lof.offset_
    return preds == -1


def _statistical_outlier_scores(X_raw: np.ndarray, z_threshold: float,
                                iqr_mult: float) -> np.ndarray:
    n_samples, n_features = X_raw.shape
    scores = np.zeros(n_samples)
    for i in range(n_features):
        col = X_raw[:, i]
        z = np.abs(stats.zscore(col, nan_policy="omit"))
        z_out = np.asarray(z > z_threshold, dtype=float)
        q1, q3 = np.percentile(col, [25, 75])
        iqr = q3 - q1
        iqr_out = ((col < q1 - iqr_mult * iqr) | (col > q3 + iqr_mult * iqr)).astype(float)
        scores += z_out + iqr_out
    return scores / (2 * n_features)


def _run_statistical(X_raw: np.ndarray, cfg: LabelConfig) -> np.ndarray:
    scores = _statistical_outlier_scores(X_raw, cfg.z_score_threshold, cfg.iqr_multiplier)
    if cfg.anomaly_rate_mode == "natural":
        median = float(np.median(scores))
        mad = float(np.median(np.abs(scores - median)))
        if mad > 0:
            threshold = median + cfg.stat_mad_multiplier * 1.4826 * mad
        else:
            threshold = median + cfg.stat_mad_multiplier * float(np.std(scores))
        if (not np.isfinite(threshold)) or (threshold <= median):
            threshold = float(np.quantile(scores, 0.95))
    else:
        threshold = float(np.percentile(scores, 100 * (1 - cfg.fixed_contamination)))
    return scores > threshold


def _run_elliptic(X_pca: np.ndarray, n_valid: int, cfg: LabelConfig) -> tuple[np.ndarray, bool]:
    ratio = n_valid / max(1, X_pca.shape[1])
    if ratio <= cfg.elliptic_min_sample_feature_ratio:
        return np.zeros(n_valid, dtype=bool), False
    try:
        fit_contam = min(cfg.fixed_contamination, 0.10)
        ell = EllipticEnvelope(contamination=fit_contam, random_state=cfg.seed)
        ell.fit(X_pca)
        preds = ell.predict(X_pca)
        scores = ell.decision_function(X_pca)
    except Exception:
        return np.zeros(n_valid, dtype=bool), False
    if cfg.anomaly_rate_mode == "natural":
        return scores < 0, True
    return preds == -1, True


def _detect_hdbscan_breakpoint(scores: np.ndarray, search_frac: float,
                               top_k: int) -> dict | None:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) < 10 or np.ptp(scores) <= 1e-12:
        return None
    sorted_desc = np.sort(scores)[::-1]
    search_len = max(5, int(np.ceil(len(sorted_desc) * search_frac)))
    search_len = min(search_len, len(sorted_desc) - 1)
    upper = sorted_desc[:search_len + 1]
    diffs = upper[:-1] - upper[1:]
    gap_order = np.argsort(diffs)[::-1][:max(1, min(top_k, len(diffs)))]
    best = gap_order[0]
    if diffs[best] <= 0:
        return None
    return {
        "threshold": float((upper[best] + upper[best + 1]) / 2.0),
        "gap_size": float(diffs[best]),
    }


def _run_hdbscan(X_pca: np.ndarray, cfg: LabelConfig) -> np.ndarray:
    try:
        import hdbscan
    except ImportError as exc:
        raise RuntimeError("hdbscan is required; install with `pip install hdbscan`") from exc
    min_cluster_size = max(cfg.hdbscan_min_cluster_floor,
                           int(cfg.hdbscan_min_cluster_frac * len(X_pca)))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=cfg.min_samples_hdbscan,
        cluster_selection_method="leaf",
        prediction_data=False,
        core_dist_n_jobs=-1,
    )
    clusterer.fit(X_pca)
    raw_scores = np.asarray(clusterer.outlier_scores_, dtype=float)
    unscored = ~np.isfinite(raw_scores)
    finite = raw_scores[np.isfinite(raw_scores)]
    if cfg.anomaly_rate_mode == "natural":
        knee = _detect_hdbscan_breakpoint(finite, cfg.hdbscan_knee_search_frac,
                                          cfg.hdbscan_top_gaps)
        threshold = knee["threshold"] if knee is not None else cfg.hdbscan_outlier_fallback
    else:
        threshold = float(np.percentile(finite, 100 * (1 - cfg.fixed_contamination)))
    anomalies = np.zeros(len(raw_scores), dtype=bool)
    fm = np.isfinite(raw_scores)
    anomalies[fm] = raw_scores[fm] > threshold
    if cfg.hdbscan_unscored_as_suspicious and unscored.any():
        anomalies[unscored] = True
    return anomalies


def _derive_consensus_thresholds(weights: np.ndarray) -> dict:
    weights = np.sort(np.asarray(weights, dtype=float))
    n = len(weights)
    if n == 0:
        return {"uncertain": 1.0, "likely_anomaly": 1.0,
                "high_confidence_anomaly": 1.0, "min_votes_for_likely": 0}
    uncertain = float(weights[0])
    min_votes = min(n, max(2, int(np.ceil((n + 1) / 2))))
    likely = float(weights[:min_votes].sum())
    high_conf = 1.0 if n <= 2 else float(1.0 - weights[-1])
    high_conf = float(min(1.0, max(high_conf, likely)))
    return {"uncertain": uncertain, "likely_anomaly": likely,
            "high_confidence_anomaly": high_conf, "min_votes_for_likely": min_votes}


def _labels_from_scores(scores: np.ndarray, thresholds: dict) -> np.ndarray:
    return np.where(
        scores >= thresholds["high_confidence_anomaly"], "high_confidence_anomaly",
        np.where(
            scores >= thresholds["likely_anomaly"], "likely_anomaly",
            np.where(scores >= thresholds["uncertain"], "uncertain", "likely_normal"),
        ),
    )


def _compute_consensus(methods_df: pd.DataFrame, use_weighted: bool) -> dict:
    methods_df = methods_df.astype(bool).copy()
    active = [c for c in methods_df.columns if methods_df[c].nunique(dropna=False) > 1]
    if not active:
        active = list(methods_df.columns)
    eff = methods_df[active]
    n_eff = len(eff.columns)
    corr = eff.astype(float).corr().replace([np.inf, -np.inf], np.nan)
    if use_weighted and n_eff > 1:
        mean_abs = (corr.abs().sum(axis=1) - 1.0) / (n_eff - 1)
        mean_abs = mean_abs.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=0.0)
        w_active = 1.0 / (1.0 + mean_abs)
        if (not np.isfinite(w_active).all()) or w_active.sum() <= 0:
            w_active = pd.Series(1.0, index=eff.columns)
        w_active = w_active / w_active.sum()
    else:
        w_active = pd.Series(1.0 / n_eff, index=eff.columns)
    w_full = pd.Series(0.0, index=methods_df.columns)
    w_full.loc[w_active.index] = w_active
    thresholds = _derive_consensus_thresholds(w_active.values)
    score = (methods_df.astype(float) * w_full.reindex(methods_df.columns).values).sum(axis=1).values
    labels = _labels_from_scores(score, thresholds)
    return {"labels": labels, "score": score, "thresholds": thresholds,
            "weights": w_full, "active_methods": active}


def _kmeans_upgrade(X_scaled: np.ndarray, labels: np.ndarray, cfg: LabelConfig) -> tuple[np.ndarray, int]:
    if not cfg.kmeans_upgrade_uncertain:
        return labels, 0
    max_k = min(cfg.kmeans_max_k, len(X_scaled) - 1)
    if max_k < 2:
        return labels, 0
    best_k, best_score = 2, -1.0
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=cfg.seed, n_init=10)
        lab = km.fit_predict(X_scaled)
        try:
            s = silhouette_score(X_scaled, lab)
        except ValueError:
            continue
        if s > best_score:
            best_score, best_k = s, k
    km = KMeans(n_clusters=best_k, random_state=cfg.seed, n_init=10)
    cluster_ids = km.fit_predict(X_scaled)
    counts = pd.Series(cluster_ids).value_counts().sort_index()
    small = counts[counts / len(cluster_ids) < cfg.kmeans_small_cluster_frac].index.tolist()
    small_mask = np.isin(cluster_ids, small)
    boost_mask = (labels == "uncertain") & small_mask
    labels = labels.copy()
    labels[boost_mask] = "likely_anomaly"
    return labels, int(boost_mask.sum())


def label_dataframe(df: pd.DataFrame, cfg: LabelConfig | None = None) -> LabelResult:
    """Apply the 5-method + K-Means labeling pipeline to df. Returns a LabelResult."""
    cfg = cfg or LabelConfig()
    feature_df, feature_cols = _feature_frame(df)
    X = feature_df.values
    valid_mask = np.isfinite(X).all(axis=1)
    X_valid = X[valid_mask]
    if len(X_valid) < 50:
        raise RuntimeError(f"Not enough valid rows to label: {len(X_valid)}")

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_valid)

    pca_full = PCA()
    pca_full.fit(X_scaled)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n_pca = int(np.argmax(cum >= cfg.pca_variance_target) + 1)
    pca = PCA(n_components=n_pca, random_state=cfg.seed)
    X_pca = pca.fit_transform(X_scaled)

    lof_n = _tune_lof_n_neighbors(X_scaled, cfg)

    iso_anom = _run_isolation_forest(X_scaled, cfg)
    lof_anom = _run_lof(X_scaled, lof_n, cfg)
    stat_anom = _run_statistical(X_valid, cfg)
    ell_anom, ell_ok = _run_elliptic(X_pca, len(X_valid), cfg)
    hdb_anom = _run_hdbscan(X_pca, cfg)

    methods = {
        "Isolation Forest": iso_anom,
        "Local Outlier Factor": lof_anom,
        "Statistical": stat_anom,
        "HDBSCAN": hdb_anom,
    }
    if ell_ok:
        methods["Elliptic Envelope"] = ell_anom
    methods_df = pd.DataFrame(methods)

    cons = _compute_consensus(methods_df, cfg.use_weighted_consensus)
    labels = cons["labels"]
    labels_final, boosted = _kmeans_upgrade(X_scaled, labels, cfg)

    method_rates = {m: float(arr.mean()) for m, arr in methods.items()}
    return LabelResult(
        discovered_label=labels_final,
        consensus_score=cons["score"],
        thresholds=cons["thresholds"],
        method_weights=cons["weights"],
        method_flags=methods_df,
        active_methods=cons["active_methods"],
        valid_mask=valid_mask,
        feature_cols=feature_cols,
        n_pca_components=n_pca,
        kmeans_boosted_count=boosted,
        method_anomaly_rates=method_rates,
        meta={"n_valid": int(valid_mask.sum()), "lof_n_neighbors": int(lof_n),
              "pca_variance": float(cum[n_pca - 1])},
    )


def _binary_from_discovered(labels: np.ndarray) -> np.ndarray:
    return np.isin(labels, ["likely_anomaly", "high_confidence_anomaly"]).astype(int)


def label_csv(input_csv: Path, output_dir: Path, cfg: LabelConfig) -> dict:
    df = pd.read_csv(input_csv)
    res = label_dataframe(df, cfg)
    out_df = df.loc[res.valid_mask].copy().reset_index(drop=True)
    out_df["discovered_label"] = res.discovered_label
    out_df["consensus_score"] = res.consensus_score
    out_df["binary_label_new"] = _binary_from_discovered(res.discovered_label)
    for m, arr in res.method_flags.items():
        out_df[f"flag_{m.lower().replace(' ', '_')}"] = arr.astype(int).values
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{input_csv.stem}_relabeled.csv"
    out_df.to_csv(out_path, index=False)
    summary = {
        "input": str(input_csv),
        "output": str(out_path),
        "n_valid": int(res.valid_mask.sum()),
        "n_invalid": int((~res.valid_mask).sum()),
        "n_features": len(res.feature_cols),
        "n_pca_components": res.n_pca_components,
        "pca_variance": res.meta["pca_variance"],
        "lof_n_neighbors": res.meta["lof_n_neighbors"],
        "kmeans_boosted": res.kmeans_boosted_count,
        "method_anomaly_rates": res.method_anomaly_rates,
        "method_weights": res.method_weights.round(4).to_dict(),
        "thresholds": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                       for k, v in res.thresholds.items()},
        "label_counts": pd.Series(res.discovered_label).value_counts().to_dict(),
    }
    with open(output_dir / f"{input_csv.stem}_metadata.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def coral_diagnostic(cfg: LabelConfig) -> pd.DataFrame:
    """For each pair, relabel before- and after-CORAL source; return churn + transitions."""
    coral_root = ROOT / "bgp_unified_results" / "phase3_fusion" / "coral_aligned" / "study_only"
    out_root = ROOT / "bgp_unified_results" / "phase3_fusion" / "relabel_after_coral_hdbscan"
    out_root.mkdir(parents=True, exist_ok=True)
    pairs = []
    for d in sorted((coral_root / "pairs").iterdir()):
        if d.is_dir() and "__to__" in d.name:
            src, tgt = d.name.split("__to__")
            pairs.append((src, tgt, d))

    # Cache "pre" labels by source domain (same for all pairs sharing that source)
    pre_cache: dict[str, pd.DataFrame] = {}
    rows = []
    classes = ["likely_normal", "uncertain", "likely_anomaly", "high_confidence_anomaly"]

    for src, tgt, pair_dir in pairs:
        pre_path = coral_root / "prepared_sources" / f"{src}_binary_source.csv"
        post_path = pair_dir / "aligned_source.csv"
        pair_out = out_root / f"{src}__to__{tgt}"
        pair_out.mkdir(parents=True, exist_ok=True)

        if src not in pre_cache:
            print(f"  [pre]  labeling {src} ...", flush=True)
            pre_summary = label_csv(pre_path, pair_out / "pre", cfg)
            pre_df = pd.read_csv(pair_out / "pre" / f"{src}_binary_source_relabeled.csv")
            pre_cache[src] = pre_df
        else:
            # Copy cached pre labels into this pair's dir for reproducibility
            pre_df = pre_cache[src]

        print(f"  [post] labeling {src} -> {tgt} ...", flush=True)
        post_summary = label_csv(post_path, pair_out / "post", cfg)
        post_df = pd.read_csv(pair_out / "post" / "aligned_source_relabeled.csv")

        merged = pre_df[["window_start", "discovered_label", "binary_label_new",
                         "binary_label"]].rename(
            columns={"discovered_label": "pre_discovered",
                     "binary_label_new": "pre_binary_new",
                     "binary_label": "pre_binary_orig"}
        ).merge(
            post_df[["window_start", "discovered_label", "binary_label_new"]].rename(
                columns={"discovered_label": "post_discovered",
                         "binary_label_new": "post_binary_new"}
            ),
            on="window_start", how="inner",
        )

        n = len(merged)
        churn_4cls = float((merged["pre_discovered"] != merged["post_discovered"]).mean())
        churn_binary = float((merged["pre_binary_new"] != merged["post_binary_new"]).mean())
        churn_vs_orig_pre = float((merged["pre_binary_new"] != merged["pre_binary_orig"]).mean())
        churn_vs_orig_post = float((merged["post_binary_new"] != merged["pre_binary_orig"]).mean())

        transitions = pd.crosstab(merged["pre_discovered"], merged["post_discovered"])
        transitions = transitions.reindex(index=classes, columns=classes, fill_value=0)
        transitions.to_csv(pair_out / "transition_matrix.csv")

        rows.append({
            "pair": f"{src}__to__{tgt}",
            "source": src, "target": tgt,
            "n_rows": n,
            "churn_4cls": round(churn_4cls, 4),
            "churn_binary_pre_vs_post": round(churn_binary, 4),
            "reproducibility_pre_vs_orig": round(churn_vs_orig_pre, 4),
            "drift_post_vs_orig": round(churn_vs_orig_post, 4),
            "pre_anomaly_rate": round(float(merged["pre_binary_new"].mean()), 4),
            "post_anomaly_rate": round(float(merged["post_binary_new"].mean()), 4),
            "orig_anomaly_rate": round(float(merged["pre_binary_orig"].mean()), 4),
        })

    out_df = pd.DataFrame(rows)
    out_csv = out_root / "churn_summary.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    return out_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    help="single CSV to relabel")
    ap.add_argument("--output-dir", type=Path,
                    help="output dir for single-CSV mode")
    ap.add_argument("--coral-diagnostic", action="store_true",
                    help="label pre/post CORAL source for every pair and emit churn CSV")
    ap.add_argument("--anomaly-rate-mode", choices=("natural", "fixed"),
                    default="natural")
    ap.add_argument("--fixed-contamination", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = LabelConfig(
        anomaly_rate_mode=args.anomaly_rate_mode,
        fixed_contamination=args.fixed_contamination,
        seed=args.seed,
    )

    if args.coral_diagnostic:
        df = coral_diagnostic(cfg)
        print(df.to_string(index=False))
        return

    if args.input is None or args.output_dir is None:
        ap.error("either --coral-diagnostic or (--input AND --output-dir)")
    summary = label_csv(args.input, args.output_dir, cfg)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()