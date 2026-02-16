import torch
import torch.nn as nn
import torch.fft
from typing import List, Dict, Any, Tuple
from TensorWAV.models.registry import register_model

class SpectralConv1d(nn.Module):
    """
    1D Fourier Spectral Layer.
    """
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, N)
        batchsize = x.shape[0]
        
        # Compute Fourier coeff
        x_ft = torch.fft.rfft(x)

        # Multiply relevant modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        
        # Modes processing
        # We need to perform matrix multiplication in the frequency domain
        # (in_channels, modes) -> (out_channels, modes)
        # x_ft[..., :modes] shape: (B, C_in, modes)
        # weights shape: (C_in, C_out, modes)
        # We use einsum: 'bix,iox->box'
        out_ft[:, :, :self.modes] = torch.einsum(
            "bix,iox->box", 
            x_ft[:, :, :self.modes], 
            self.weights
        )

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

@register_model("fno1d")
class FNO1d(nn.Module):
    """
    1D Fourier Neural Operator.
    Input: (B, N, F_in) -> Transpose to (B, F_in, N) -> FNO Layers -> (B, F_out, N) -> Transpose back
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        modes: int,
        width: int,
        depth: int = 4,
        **kwargs: Any
    ):
        super().__init__()
        self.width = width
        self.fc0 = nn.Linear(input_dim, width)
        
        self.conv_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()
        
        for _ in range(depth):
            self.conv_layers.append(SpectralConv1d(width, width, modes))
            self.w_layers.append(nn.Conv1d(width, width, 1))

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (B, N, F_in)
        """
        # Lift to width
        x = self.fc0(x) # (B, N, width)
        x = x.permute(0, 2, 1) # (B, width, N)

        for conv, w in zip(self.conv_layers, self.w_layers):
            x1 = conv(x)
            x2 = w(x)
            x = x1 + x2
            x = self.act(x)

        x = x.permute(0, 2, 1) # (B, N, width)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x) # (B, N, output_dim)
        
        return x

class SpectralConv2d(nn.Module):
    """
    2D Fourier Spectral Layer.
    """
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        batchsize = x.shape[0]
        
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)

        # Upper left corner
        out_ft[:, :, :self.modes1, :self.modes2] = \
            torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)

        # Lower left corner
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

@register_model("fno2d")
class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator.
    Requires input (B, N, F_in) where N = H*W.
    Uses 'resolution' from config to reshape.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        modes1: int,
        modes2: int,
        width: int,
        depth: int = 4,
        resolution: List[int] = [64, 64],
        **kwargs: Any
    ):
        super().__init__()
        self.width = width
        self.resolution = resolution
        self.fc0 = nn.Linear(input_dim, width)
        
        self.conv_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()

        for _ in range(depth):
            self.conv_layers.append(SpectralConv2d(width, width, modes1, modes2))
            self.w_layers.append(nn.Conv2d(width, width, 1))
            
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (B, N, F_in) where N should match resolution[0]*resolution[1]
        """
        B, N, F = x.shape
        H, W = self.resolution
        
        if N != H * W:
            raise ValueError(f"Input spatial dim {N} does not match resolution {H}x{W}={H*W}")
        
        x = self.fc0(x) # (B, N, width)
        x = x.permute(0, 2, 1).view(B, self.width, H, W) # (B, width, H, W)

        for conv, w in zip(self.conv_layers, self.w_layers):
            x1 = conv(x)
            x2 = w(x)
            x = x1 + x2
            x = self.act(x)
            
        x = x.view(B, self.width, N).permute(0, 2, 1) # (B, N, width)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        
        return x
