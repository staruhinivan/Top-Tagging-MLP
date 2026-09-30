import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self, N_features, neurons_lay, fun):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_features, neurons_lay),
            fun,
            nn.Linear(neurons_lay, neurons_lay),
            fun,
            nn.Linear(neurons_lay, 2),
        )

    def forward(self, x):
        return self.net(x)
    
def find_peaks(x, height=None, distance=1, prominence=None):
    """Простая реализация поиска локальных максимумов на NumPy"""
    # Находим точки, где значение больше соседей
    peaks = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1
    
    if height is not None:
        peaks = peaks[x[peaks] >= height]
    
    if distance > 1:
        # Убираем слишком близкие пики
        keep = np.ones(len(peaks), dtype=bool)
        for i in range(1, len(peaks)):
            if peaks[i] - peaks[i-1] < distance:
                keep[i] = False
        peaks = peaks[keep]
    
    return peaks