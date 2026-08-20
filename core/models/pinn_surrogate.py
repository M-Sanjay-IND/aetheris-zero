import torch
import torch.nn as nn

class PINNSurrogate(nn.Module):
    """Physics-Informed Neural Network predicting 24-hour multi-zone forward thermal trajectories."""
    def __init__(self):
        super().__init__()
