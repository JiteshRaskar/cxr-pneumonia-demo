"""Metrics, source-validation threshold, patient-level bootstrap CIs (Sec. IV-H, IV-K)."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def threshold_from_val(y, p) -> float:
    """Youden's J on SOURCE validation data only. Fixed afterwards (no target tuning)."""
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    j[~np.isfinite(thr)] = -1
    return float(thr[int(np.argmax(j))])


def _div(a, b):
    return float(a / b) if b > 0 else float("nan")


def metrics_at(y, p, thr, w=None) -> dict:
    y = np.asarray(y)
    pred = (np.asarray(p) >= thr).astype(int)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    tp = (w * ((pred == 1) & (y == 1))).sum()
    tn = (w * ((pred == 0) & (y == 0))).sum()
    fp = (w * ((pred == 1) & (y == 0))).sum()
    fn = (w * ((pred == 0) & (y == 1))).sum()
    prec, rec = _div(tp, tp + fp), _div(tp, tp + fn)
    f1 = _div(2 * prec * rec, prec + rec) if np.isfinite(prec) and np.isfinite(rec) else float("nan")
    try:
        auc = float(roc_auc_score(y, p, sample_weight=w))  # uses scores, not binary preds
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": _div(tp + tn, tp + tn + fp + fn),
        "precision": prec,
        "recall": rec,
        "specificity": _div(tn, tn + fp),
        "f1": f1,
        "roc_auc": auc,
    }


def bootstrap_ci(y, p, groups, thr, n_boot=300, seed=0):
    """Patient-level (cluster) bootstrap via per-patient resampling weights."""
    y, p = np.asarray(y), np.asarray(p)
    _, gid = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    G = gid.max() + 1
    rng = np.random.default_rng(seed)
    acc = {}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, G, G), minlength=G)[gid]
        for k, v in metrics_at(y, p, thr, w).items():
            acc.setdefault(k, []).append(v)
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in acc.items()}
