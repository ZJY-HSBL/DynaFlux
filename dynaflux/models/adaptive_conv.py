from __future__ import annotations

import torch
from torch import nn


class CausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.left_padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.left_padding:
            y = y[..., :-self.left_padding]
        return y


class AdaptiveGatedConv(nn.Module):
    """MLP fusion + dilated/standard temporal convolution + sigmoid gating."""

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int = 3,
        dilation: int = 2,
        dropout: float = 0.3,
        noise: float = 0.1,
    ) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv_dilated = CausalConv1d(hidden_dim, kernel_size=kernel_size, dilation=dilation)
        self.conv_standard = CausalConv1d(hidden_dim, kernel_size=kernel_size, dilation=1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.noise = float(noise)

    def forward(self, residual: torch.Tensor, x_ms: torch.Tensor) -> torch.Tensor:
        # residual/x_ms: [B,T,N,D]
        x_st = self.fuse(torch.cat([residual, x_ms], dim=-1))
        b, t, n, d = x_st.shape
        z = x_st.permute(0, 2, 3, 1).reshape(b * n, d, t)
        x1 = self.conv_dilated(z)
        x2 = self.conv_standard(z)
        if self.training and self.noise > 0:
            eps = torch.empty_like(x2).uniform_(-self.noise, self.noise)
            x2 = x2 * (1.0 + eps)
        gate = torch.sigmoid(x1 + x2)
        out = gate * x1 + (1.0 - gate) * x2
        out = out.reshape(b, n, d, t).permute(0, 3, 1, 2).contiguous()
        return self.norm(residual + self.dropout(out))
