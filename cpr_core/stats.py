import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Gaussian, Binomial
from statsmodels.genmod.cov_struct import Exchangeable

def cohen_kappa(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    labels = np.unique(np.concatenate([a, b]))
    n = len(a)
    po = np.mean(a == b)
    pe = sum((np.sum(a == l) / n) * (np.sum(b == l) / n) for l in labels)
    if pe >= 1:
        return np.nan
    return (po - pe) / (1 - pe)

def weighted_kappa(a, b, weights="quadratic"):
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    labels = np.unique(np.concatenate([a, b]))
    k = len(labels)
    idx = {l: i for i, l in enumerate(labels)}
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    O = O / O.sum()
    r = O.sum(1)
    c = O.sum(0)
    E = np.outer(r, c)
    if weights == "quadratic":
        W = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    elif weights == "linear":
        W = np.array([[abs(i - j) / (k - 1) for j in range(k)] for i in range(k)])
    else:
        W = (np.ones((k, k)) - np.eye(k))
    po = 1 - np.sum(W * O)
    pe = 1 - np.sum(W * E)
    if pe >= 1:
        return np.nan
    return (po - pe) / (1 - pe)

def pabak(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    po = np.mean(a == b)
    return 2 * po - 1

def icc21(x, y):
    data = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    n, k = data.shape
    grand = data.mean()
    ms_rows = k * ((data.mean(1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((data.mean(0) - grand) ** 2).sum() / (k - 1)
    resid = data - data.mean(1, keepdims=True) - data.mean(0, keepdims=True) + grand
    ms_err = (resid ** 2).sum() / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    return (ms_rows - ms_err) / denom if denom != 0 else np.nan

def auc_score(y, score):
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    diff = pos[:, None] - neg[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / diff.size)

def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2

def delong_roc_test(y, score1, score2):
    y = np.asarray(y, int)
    s1 = np.asarray(score1, float)
    s2 = np.asarray(score2, float)
    order = np.argsort(1 - y)
    y_sorted = y[order]
    m = int(np.sum(y))
    n = len(y) - m
    preds = np.vstack([s1[order], s2[order]])
    k = 2
    tz = np.empty([k, m + n], dtype=float)
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    for r in range(k):
        tz[r, :] = _compute_midrank(preds[r, :])
        tx[r, :] = _compute_midrank(preds[r, :m])
        ty[r, :] = _compute_midrank(preds[r, m:])
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    auc1 = np.mean(preds[0, :m, None] > preds[0, m:]) + 0.5 * np.mean(preds[0, :m, None] == preds[0, m:])
    auc2 = np.mean(preds[1, :m, None] > preds[1, m:]) + 0.5 * np.mean(preds[1, :m, None] == preds[1, m:])
    diff = auc1 - auc2
    var = delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]
    z = diff / np.sqrt(var) if var > 0 else np.nan
    p = 2 * stats.norm.sf(abs(z)) if not np.isnan(z) else np.nan
    return auc1, auc2, z, p

def binary_metrics(pred, ref):
    pred = np.asarray(pred, int)
    ref = np.asarray(ref, int)
    tp = int(np.sum((pred == 1) & (ref == 1)))
    fp = int(np.sum((pred == 1) & (ref == 0)))
    fn = int(np.sum((pred == 0) & (ref == 1)))
    tn = int(np.sum((pred == 0) & (ref == 0)))
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    acc = (tp + tn) / len(pred) if len(pred) else np.nan
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": sens, "specificity": spec,
        "ppv": ppv, "npv": npv, "accuracy": acc,
        "kappa": cohen_kappa(pred, ref),
        "pabak": pabak(pred, ref)
    }

def fit_lmm(data, formula, groups):
    return smf.mixedlm(formula, data=data, groups=groups).fit(reml=True, method="powell", maxiter=2000)

def fit_gee(data, formula, groups, family="gaussian"):
    fam = Gaussian() if family == "gaussian" else Binomial()
    return sm.GEE.from_formula(formula, groups=groups, data=data, family=fam, cov_struct=Exchangeable()).fit(maxiter=300)
