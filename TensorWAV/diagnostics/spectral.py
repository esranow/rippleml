import torch
import torch.fft

def spectral_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Relative L2 error in the frequency domain.
    Useful for checking if high-frequency content is captured.
    
    Args:
        pred (torch.Tensor): (B, N, ...)
        target (torch.Tensor): (B, N, ...)
    
    Returns:
        float: Relative spectral error.
    """
    # FFT over spatial dims. Assume N is spatial.
    # We flatten spatial dims to N for 1D FFT or use N-d FFT if shape known.
    # For robustness, we assume (B, N, C) and do FFT on N (dim 1).
    
    pred_fft = torch.fft.fft(pred, dim=1)
    target_fft = torch.fft.fft(target, dim=1)
    
    diff_norm = torch.norm(pred_fft - target_fft)
    target_norm = torch.norm(target_fft)
    
    return (diff_norm / (target_norm + 1e-8)).item()
