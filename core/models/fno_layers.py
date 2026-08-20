import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, length = x.shape
        x_ft = torch.fft.rfft(x)

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat
        )

        modes_to_use = min(self.modes, x_ft.shape[-1])
        out_ft[:, :, :modes_to_use] = torch.einsum(
            "bix,iox->box",
            x_ft[:, :, :modes_to_use],
            self.weights[:, :, :modes_to_use]
        )

        x_out = torch.fft.irfft(out_ft, n=length)
        return x_out

class FNOBlock1d(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.conv = SpectralConv1d(width, width, modes)
        self.w = nn.Conv1d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.conv(x) + self.w(x))

class FNONetwork1d(nn.Module):
    def __init__(
        self,
        in_dim: int = 17,
        out_dim: int = 5,
        modes: int = 16,
        width: int = 64,
        num_layers: int = 4
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.width = width

        self.lift = nn.Conv1d(in_dim, width, 1)
        self.blocks = nn.ModuleList([FNOBlock1d(width, modes) for _ in range(num_layers)])
        
        self.proj1 = nn.Conv1d(width, 128, 1)
        self.proj2 = nn.Conv1d(128, out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (batch, in_dim, seq_len)
        h = self.lift(x)
        for block in self.blocks:
            h = block(h)
        h = F.gelu(self.proj1(h))
        out = self.proj2(h)
        return out
