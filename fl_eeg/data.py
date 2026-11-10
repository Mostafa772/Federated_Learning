"""iEEG segment dataset and the client partitioning schemes used in the thesis."""
from pathlib import Path

import mat73
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as signal
import scipy.stats as stats
import torch
from torch.utils.data import Dataset

# Kaggle folder ids: 0 power-line noise, 1 noise/muscle artifacts, 2 physiological, 3 pathological.
# The power-line class is dropped and the remaining ids are shifted down by one.
CLASS_NAMES = ["Noise / muscle activity", "Physiological iEEG", "Pathological iEEG"]
N_FREQ_BINS = 200
SAMPLING_RATE = 5000


def load_segment(path):
    try:
        mat = mat73.loadmat(str(path))
    except Exception:  # files saved before MATLAB v7.3
        mat = sio.loadmat(str(path))
    return np.asarray(mat["data"], dtype=np.float64).squeeze()


def spectrogram_features(x):
    """Raw segment (15000 samples) -> z-scored spectrogram of shape (1, 200, T)."""
    _, _, s = signal.spectrogram(x, fs=SAMPLING_RATE, nperseg=256, noverlap=128, nfft=1024)
    s = stats.zscore(s[:N_FREQ_BINS], axis=1)
    return np.float32(s[None])


class IEEGDataset(Dataset):
    """Segments listed in an index CSV with columns ``segment_id`` (path relative to ``root``)
    and ``category_id`` (Kaggle class folder)."""

    def __init__(self, root, index_csv, drop_powerline=True):
        self.root = Path(root)
        df = pd.read_csv(index_csv)
        if drop_powerline:
            df = df[df.category_id != 0].assign(category_id=lambda d: d.category_id - 1)
        self.df = df.reset_index(drop=True)
        self.targets = self.df.category_id.to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        x = spectrogram_features(load_segment(self.root / row.segment_id))
        return torch.from_numpy(x), int(row.category_id)


def partition_iid(labels, n_clients, rng):
    """Every client receives the same number of samples of every class."""
    parts = [[] for _ in range(n_clients)]
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        for k, chunk in enumerate(np.array_split(idx, n_clients)):
            parts[k].extend(chunk)
    return [np.array(p, dtype=int) for p in parts]


def partition_dirichlet(labels, n_clients, rng, alpha=100.0):
    """Label-distribution skew ("weak non-IID"): each class is spread over the clients with
    proportions drawn from Dir(alpha). alpha=100 with 40 clients matches thesis Figure 21."""
    parts = [[] for _ in range(n_clients)]
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        cuts = (np.cumsum(rng.dirichlet(np.full(n_clients, alpha)))[:-1] * len(idx)).astype(int)
        for k, chunk in enumerate(np.split(idx, cuts)):
            parts[k].extend(chunk)
    return [np.array(p, dtype=int) for p in parts]


def partition_by_label(labels, n_clients, rng):
    """Quantity-based label imbalance ("extreme non-IID"): samples sorted by label are cut into
    equal contiguous shards, so each client only sees one or two classes (thesis Figure 22)."""
    order = np.lexsort((rng.random(len(labels)), labels))
    return [np.asarray(p, dtype=int) for p in np.array_split(order, n_clients)]


PARTITIONS = {
    "iid": partition_iid,
    "weak-noniid": partition_dirichlet,
    "extreme-noniid": partition_by_label,
}


def balanced_indices(labels, n_per_class, rng):
    return np.concatenate([
        rng.choice(np.flatnonzero(labels == c), size=min(n_per_class, int((labels == c).sum())), replace=False)
        for c in np.unique(labels)
    ])
