"""CNN-GRU classifier for iEEG spectrograms, adapted from Nejedly et al. (2019)."""
import torch.nn as nn
import torch.nn.functional as F


class CNNGRU(nn.Module):
    def __init__(self, n_freq=200, n_filters=256, hidden=128, n_classes=3, dropout=0.5):
        super().__init__()
        # The kernel spans the whole frequency axis, so this acts as a 1-D convolution over time.
        self.conv = nn.Conv2d(1, n_filters, kernel_size=(n_freq, 3), padding=(0, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters)
        self.gru = nn.GRU(n_filters, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):  # x: (batch, 1, n_freq, time)
        x = F.relu(self.bn(self.conv(x)))   # (batch, n_filters, 1, time)
        x = x.squeeze(2).transpose(1, 2)    # (batch, time, n_filters)
        x, _ = self.gru(x)
        return self.fc(self.dropout(x[:, -1]))
