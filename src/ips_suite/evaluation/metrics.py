"""Metrics: binary (RSNA) or ordinal with 5 grades (DDR) Every metric is computed at every epoch. The reported value of a run is always the one at the epoch that maximises the MAIN metric on validation, never the maximum of each metric taken separately."""

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve


def _ece(p, y, n_bins=15):
    """Expected calibration error on the predicted class."""
    conf_, pred = p.max(1), p.argmax(1)
    hit = (pred == y).astype(float)
    e = 0.0
    for lo, hi in zip(np.linspace(0, 1, n_bins + 1)[:-1], np.linspace(0, 1, n_bins + 1)[1:]):
        m = (conf_ > lo) & (conf_ <= hi)
        if m.any():
            e += m.mean() * abs(hit[m].mean() - conf_[m].mean())
    return float(e)


def compute_metrics(y_true, y_pred, y_prob, task_type, threshold=None):
    """Every metric that can be computed from labels and probabilities.

    ordinal (DDR): quadratic and linear kappa, accuracy, balanced accuracy,
        macro and weighted F1, per-grade precision / recall / F1, mean grade
        error, macro one-vs-rest AUC, referable DR (grade >= 2): AUC,
        sensitivity, specificity.
    binary (RSNA): AUC, AUPRC, F1, precision, recall, specificity.
    both: confusion matrix, NLL, Brier score, expected calibration error.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=int)
    if task_type == 'binary' and threshold is not None:
        y_pred = (y_prob[:, 1] >= threshold).astype(int)
    K = y_prob.shape[1]
    labels = list(range(K))
    nan = float('nan')
    two = len(np.unique(y_true)) > 1
    out = {'accuracy': float(accuracy_score(y_true, y_pred)),
           'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)) if two else nan,
           'confusion': confusion_matrix(y_true, y_pred, labels=labels).tolist()}
    if len(y_true):
        pt = np.clip(y_prob[np.arange(len(y_true)), y_true], 1e-12, 1.0)
        out['nll'] = float(-np.log(pt).mean())
        out['brier'] = float(((y_prob - np.eye(K)[y_true]) ** 2).sum(1).mean())
        out['ece'] = _ece(y_prob, y_true)
    if task_type == 'ordinal':
        out['kappa'] = float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))
        out['kappa_linear'] = float(cohen_kappa_score(y_true, y_pred, weights='linear'))
        out['f1'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        out['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        out['mae_grade'] = float(np.abs(y_pred - y_true).mean())
        for nm, fn in (('precision', precision_score), ('recall', recall_score), ('f1', f1_score)):
            out[f'{nm}_per_class'] = fn(y_true, y_pred, labels=labels, average=None,
                                        zero_division=0).tolist()
        try:
            out['auc_macro_ovr'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr',
                                                       average='macro', labels=labels))
        except ValueError:
            out['auc_macro_ovr'] = nan
        ref = (y_true >= 2).astype(int)                 # referable DR
        p_ref = y_prob[:, 2:].sum(1)
        try:
            out['auc_referable'] = float(roc_auc_score(ref, p_ref))
        except ValueError:
            out['auc_referable'] = nan
        pr = y_pred >= 2
        out['sens_referable'] = float(pr[ref == 1].mean()) if (ref == 1).any() else nan
        out['spec_referable'] = float((~pr)[ref == 0].mean()) if (ref == 0).any() else nan
    else:
        out['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
        out['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
        out['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
        neg = y_true == 0
        out['specificity'] = float((y_pred[neg] == 0).mean()) if neg.any() else nan
        try:
            out['auc'] = float(roc_auc_score(y_true, y_prob[:, 1]))
            out['auprc'] = float(average_precision_score(y_true, y_prob[:, 1]))
        except ValueError:
            out['auc'] = out['auprc'] = nan
    return out


def bootstrap_ci(y_true, y_prob, task_type, n=1000, seed=0):
    """95 % percentile intervals over images, for the headline metrics."""
    y, p = np.asarray(y_true, dtype=int), np.asarray(y_prob, dtype=np.float64)
    rng, vals = np.random.default_rng(seed), defaultdict(list)
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        yi, pi = y[i], p[i]
        vals['accuracy'].append(float((pi.argmax(1) == yi).mean()))
        try:
            if task_type == 'ordinal':
                vals['kappa'].append(cohen_kappa_score(yi, pi.argmax(1), weights='quadratic'))
                vals['auc_referable'].append(roc_auc_score((yi >= 2).astype(int), pi[:, 2:].sum(1)))
            else:
                vals['auc'].append(roc_auc_score(yi, pi[:, 1]))
        except ValueError:
            pass
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in vals.items() if len(v) > 10}


def youden_threshold(y_true, p_pos):
    """Threshold that maximises sensitivity + specificity - 1."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, np.asarray(p_pos))
    k = int(np.argmax(tpr - fpr))
    t = float(thr[k])
    return 0.5 if not np.isfinite(t) else t


# ============================================================
# Loss: exactly the one of the original IPS code
# (softmax, log(p + eps), NLLLoss).
# ============================================================
class IPSLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.nll = nn.NLLLoss()

    def forward(self, logits, target):
        p = torch.softmax(logits, dim=-1)
        return self.nll(torch.log(p + self.eps), target)
