"""Evaluation metrics reported in the thesis: accuracy, macro F1, Cohen's kappa, macro AUROC."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, roc_auc_score


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()
    logits, targets = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu())
        targets.append(y)
    logits, y = torch.cat(logits), torch.cat(targets)
    probs = logits.softmax(1).numpy()
    y_true, y_pred = y.numpy(), probs.argmax(1)
    labels = np.arange(probs.shape[1])

    out = {
        "loss": F.cross_entropy(logits, y).item(),
        "accuracy": float((y_pred == y_true).mean()),
        "f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if len(np.unique(y_true)) == len(labels):
        out["auroc"] = float(roc_auc_score(y_true, probs, labels=labels, multi_class="ovr"))
    return out
