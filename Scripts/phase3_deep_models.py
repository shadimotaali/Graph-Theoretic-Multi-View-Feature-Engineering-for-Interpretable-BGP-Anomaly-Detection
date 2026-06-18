"""Stage 3 deep-learning classifiers and attention fusion for Phase 3.

Models:
    * MLP (flat features, sklearn-style wrapper).
    * LSTM (per-row sequences over the last ``seq_len`` minutes; sequence
      construction respects domain boundaries and an optional group column).
    * Learned attention fusion (logistic regression on source [p_graph, p_stat]
      applied to target probabilities; falls back to uniform weights when only
      one class is present in the source).

Determinism: ``set_determinism(seed, strict=False)`` seeds numpy + torch CPU +
torch CUDA. Set ``strict=True`` to also force ``torch.use_deterministic_algorithms``
and deterministic cuDNN kernels (slower, and some ops raise if no deterministic
kernel exists). Default is the loose mode suitable for seed-averaged runs.
``get_device()`` picks CUDA when available, otherwise CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# torch is imported lazily so the module imports cleanly even without it
# installed; importing Scripts.phase3_deep_models will succeed but the fitter
# functions will fail with a clear message.
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for Stage 3 (MLP/LSTM/attention fusion). "
            "Install with: pip install torch"
        )


def get_device():
    _require_torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_determinism(seed: int, strict: bool = True) -> None:
    """Seed python/numpy/torch and enforce strict CUDA determinism by default.

    Strict mode (the default since the 2026-04-22 reproducibility audit) sets
    ``torch.use_deterministic_algorithms(True, warn_only=False)`` and
    ``torch.backends.cudnn.{deterministic=True,benchmark=False}``.  With strict
    on, two runs with the same seed on the same hardware produce byte-identical
    outputs.  This requires the env var ``CUBLAS_WORKSPACE_CONFIG=:4096:8``
    (set here; safe to also set in the launching shell); without it
    ``use_deterministic_algorithms`` raises at the first cuBLAS call.

    Pass ``strict=False`` only for debugging throughput — never for paper runs.
    """
    _require_torch()
    import os as _os
    import random as _random
    # CUBLAS workspace config is required by torch.use_deterministic_algorithms.
    # setdefault so a shell-level override is respected.
    _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only=False: if an op lacks a deterministic kernel we want the
        # run to FAIL, not silently fall back to a non-deterministic path.
        torch.use_deterministic_algorithms(True, warn_only=False)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def _build_mlp(n_in: int, hidden: Sequence[int], dropout: float):
    _require_torch()
    layers: list = []
    prev = n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def _build_lstm(n_in: int, hidden: int, dropout: float):
    _require_torch()

    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_in, hidden, num_layers=1, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    return LSTMNet()


# ---------------------------------------------------------------------------
# Flat (non-sequence) MLP wrapper
# ---------------------------------------------------------------------------
@dataclass
class MLPParams:
    hidden: tuple = (64, 32)
    dropout: float = 0.2
    epochs: int = 50
    lr: float = 1e-3
    batch_size: int = 512
    weight_decay: float = 0.0
    # Inference batch size; None means "use one pass" (small datasets only).
    inference_batch_size: int | None = 8192

    def __post_init__(self) -> None:
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if not all(h > 0 for h in self.hidden):
            raise ValueError(f"all hidden widths must be > 0, got {self.hidden}")


class TorchFlatClassifier:
    """sklearn-style wrapper: fit(X,y), predict_proba(X) -> (n, 2)."""

    def __init__(self, n_features: int, params: MLPParams, seed: int):
        _require_torch()
        self.params = params
        self.seed = seed
        self.device = get_device()
        self.n_features = n_features
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchFlatClassifier":
        set_determinism(self.seed)
        p = self.params
        self.model = _build_mlp(self.n_features, p.hidden, p.dropout).to(self.device)

        X_t = torch.as_tensor(X, dtype=torch.float32)
        y_t = torch.as_tensor(y, dtype=torch.float32)
        n_pos = max(1, int((y_t == 1).sum().item()))
        n_neg = max(1, int((y_t == 0).sum().item()))
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(self.model.parameters(), lr=p.lr, weight_decay=p.weight_decay)

        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=p.batch_size, shuffle=True, drop_last=False,
        )
        self.model.train()
        for _ in range(p.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device); yb = yb.to(self.device)
                opt.zero_grad()
                loss = criterion(self.model(xb).squeeze(-1), yb)
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        self.model.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32)
        bs = self.params.inference_batch_size or X_t.shape[0]
        out = np.empty(X_t.shape[0], dtype=np.float32)
        with torch.no_grad():
            for i in range(0, X_t.shape[0], bs):
                batch = X_t[i:i + bs].to(self.device)
                logits = self.model(batch).squeeze(-1)
                out[i:i + bs] = torch.sigmoid(logits).cpu().numpy()
        return np.stack([1.0 - out, out], axis=1)


def fit_mlp(X_train: np.ndarray, y_train: np.ndarray, seed: int,
            **kwargs) -> TorchFlatClassifier:
    params = MLPParams(**kwargs)
    clf = TorchFlatClassifier(n_features=X_train.shape[1], params=params, seed=seed)
    clf.fit(X_train.astype(np.float32), y_train.astype(np.float32))
    return clf


# ---------------------------------------------------------------------------
# LSTM: sequence classifier (uses DataFrame with timestamp column)
# ---------------------------------------------------------------------------
def build_sequences(df: pd.DataFrame, feature_cols: Sequence[str], seq_len: int,
                    ts_col: str = "window_start", domain_col: str = "domain",
                    label_col: str = "binary_label",
                    group_col: str | None = None) -> tuple:
    """Construct per-row sequences of length ``seq_len`` ending at each row.

    Ordering: rows are sorted by ``(group_col, domain_col, ts_col)`` (any missing
    key is skipped). Sequences do not span group or domain boundaries. The
    timestamp column is parsed with ``pd.to_datetime`` so lexicographic quirks
    in the string representation don't break ordering.

    Parameters
    ----------
    group_col : optional column (e.g. provenance, regime) whose boundaries
        should additionally prevent sequence crossings.

    Returns
    -------
    X_seq : (n, seq_len, n_features) float32
    y     : (n,) float32 or None if label_col not present
    orig_idx : (n,) int -- positional index back into the input ``df``
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")
    df_reset = df.reset_index(drop=True).copy()
    df_reset["_orig_idx"] = np.arange(len(df_reset))
    # Parse timestamps if possible (robust to ISO strings / mixed formats).
    if ts_col in df_reset.columns:
        try:
            df_reset["_ts_parsed"] = pd.to_datetime(df_reset[ts_col], errors="coerce", utc=True)
            ts_key = "_ts_parsed"
        except Exception:
            ts_key = ts_col
    else:
        raise KeyError(f"timestamp column '{ts_col}' not in dataframe")

    sort_keys: list[str] = []
    if group_col and group_col in df_reset.columns:
        sort_keys.append(group_col)
    if domain_col in df_reset.columns:
        sort_keys.append(domain_col)
    sort_keys.append(ts_key)
    df_sorted = df_reset.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)

    X = df_sorted[list(feature_cols)].to_numpy(dtype=np.float32)
    n, n_feat = X.shape
    y = df_sorted[label_col].to_numpy(dtype=np.float32) if label_col in df_sorted.columns else None

    # Build a boundary key that increments whenever (group, domain) changes.
    boundary_cols = [c for c in (group_col, domain_col) if c and c in df_sorted.columns]
    if boundary_cols:
        boundary = df_sorted[boundary_cols].astype(str).agg("|".join, axis=1).to_numpy()
    else:
        boundary = np.zeros(n, dtype=object)

    seqs = np.zeros((n, seq_len, n_feat), dtype=np.float32)
    seg_start = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        seg_start[i] = i if boundary[i] != boundary[i - 1] else seg_start[i - 1]
    for i in range(n):
        start = max(seg_start[i], i - seq_len + 1)
        window = X[start:i + 1]
        if window.shape[0] < seq_len:
            pad = np.repeat(window[:1], seq_len - window.shape[0], axis=0)
            window = np.concatenate([pad, window], axis=0)
        seqs[i] = window

    orig_idx = df_sorted["_orig_idx"].to_numpy()
    return seqs, y, orig_idx


@dataclass
class LSTMParams:
    seq_len: int = 10
    hidden: int = 64
    dropout: float = 0.2
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 256
    weight_decay: float = 0.0
    inference_batch_size: int | None = 2048

    def __post_init__(self) -> None:
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {self.seq_len}")
        if self.hidden <= 0:
            raise ValueError(f"hidden must be > 0, got {self.hidden}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")


class LSTMPipeline:
    """sklearn-ish wrapper that takes DataFrames and handles sequence building."""

    def __init__(self, feature_cols: Sequence[str], params: LSTMParams, seed: int,
                 ts_col: str = "window_start", domain_col: str = "domain"):
        _require_torch()
        self.feature_cols = list(feature_cols)
        self.params = params
        self.seed = seed
        self.ts_col = ts_col
        self.domain_col = domain_col
        self.device = get_device()
        self.model = None

    def fit(self, df: pd.DataFrame) -> "LSTMPipeline":
        set_determinism(self.seed)
        p = self.params
        X_seq, y, _ = build_sequences(
            df, self.feature_cols, p.seq_len,
            ts_col=self.ts_col, domain_col=self.domain_col,
        )
        if y is None:
            raise ValueError("fit() expects a DataFrame with 'binary_label' column")

        self.model = _build_lstm(len(self.feature_cols), p.hidden, p.dropout).to(self.device)
        X_t = torch.as_tensor(X_seq)
        y_t = torch.as_tensor(y, dtype=torch.float32)
        n_pos = max(1, int((y_t == 1).sum().item()))
        n_neg = max(1, int((y_t == 0).sum().item()))
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(self.model.parameters(), lr=p.lr, weight_decay=p.weight_decay)
        loader = DataLoader(TensorDataset(X_t, y_t),
                            batch_size=p.batch_size, shuffle=True, drop_last=False)
        self.model.train()
        for _ in range(p.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device); yb = yb.to(self.device)
                opt.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        assert self.model is not None
        p = self.params
        X_seq, _, orig_idx = build_sequences(
            df, self.feature_cols, p.seq_len,
            ts_col=self.ts_col, domain_col=self.domain_col,
        )
        self.model.eval()
        X_t = torch.as_tensor(X_seq)
        bs = p.inference_batch_size or X_t.shape[0]
        n_sorted = X_t.shape[0]
        p1_sorted = np.empty(n_sorted, dtype=np.float32)
        with torch.no_grad():
            for i in range(0, n_sorted, bs):
                batch = X_t[i:i + bs].to(self.device)
                logits = self.model(batch)
                p1_sorted[i:i + bs] = torch.sigmoid(logits).cpu().numpy()
        # Restore input-df row order: orig_idx[k] is the original-df position of the k-th sequence.
        n = len(df)
        p1 = np.empty(n, dtype=np.float32)
        p1[orig_idx] = p1_sorted
        return np.stack([1.0 - p1, p1], axis=1)


def fit_lstm(source_df: pd.DataFrame, feature_cols: Sequence[str], seed: int,
             **kwargs) -> LSTMPipeline:
    params = LSTMParams(**kwargs)
    model = LSTMPipeline(feature_cols=feature_cols, params=params, seed=seed)
    model.fit(source_df)
    return model


# ---------------------------------------------------------------------------
# Attention fusion: learned logistic combination of graph + stat probabilities
# ---------------------------------------------------------------------------
@dataclass
class AttentionFusionWeights:
    intercept: float
    w_graph: float
    w_stat: float
    # When False, ``apply`` returns the linear combination directly (skipping
    # the sigmoid). Used by the single-class fallback to emit an honest
    # arithmetic mean instead of a sigmoid-squashed value in [0.5, 0.73].
    use_sigmoid: bool = True

    def apply(self, p_graph: np.ndarray, p_stat: np.ndarray) -> np.ndarray:
        z = self.intercept + self.w_graph * p_graph + self.w_stat * p_stat
        if not self.use_sigmoid:
            return z
        return 1.0 / (1.0 + np.exp(-z))


def fit_attention_fusion(p_graph_src: np.ndarray, p_stat_src: np.ndarray,
                         y_src: np.ndarray, seed: int = 0) -> AttentionFusionWeights:
    """Fit a 2-feature logistic regression on source probabilities.

    Weights learned here are biased (they use training-set predictions), but
    the relative ordering among classifiers is what we care about. For a proper
    unbiased estimate, swap this for an out-of-fold CV stacker.

    Raises
    ------
    ValueError
        If ``p_graph_src``, ``p_stat_src``, and ``y_src`` have mismatched lengths.

    Fallback
    --------
    When the source contains only one class, returns weights that compute the
    arithmetic mean of ``p_graph`` and ``p_stat`` without the sigmoid
    (``use_sigmoid=False``, ``w_graph=w_stat=0.5``). This preserves the [0, 1]
    probability range instead of compressing every row into [0.5, 0.73] and
    flipping the threshold to "predict anomaly for everything".
    """
    from sklearn.linear_model import LogisticRegression

    if not (len(p_graph_src) == len(p_stat_src) == len(y_src)):
        raise ValueError(
            f"length mismatch: p_graph={len(p_graph_src)} p_stat={len(p_stat_src)} "
            f"y={len(y_src)}"
        )
    y_int = y_src.astype(int)
    unique = np.unique(y_int)
    if unique.size < 2:
        # Honest arithmetic-mean fallback; no sigmoid squashing.
        return AttentionFusionWeights(
            intercept=0.0, w_graph=0.5, w_stat=0.5, use_sigmoid=False,
        )

    X = np.stack([p_graph_src, p_stat_src], axis=1)
    clf = LogisticRegression(
        class_weight="balanced", random_state=seed,
        solver="liblinear", max_iter=1000,
    )
    clf.fit(X, y_int)
    return AttentionFusionWeights(
        intercept=float(clf.intercept_[0]),
        w_graph=float(clf.coef_[0, 0]),
        w_stat=float(clf.coef_[0, 1]),
    )
