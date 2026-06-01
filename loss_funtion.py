# loss/loss_funtion.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def amplitude_envelope(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    HS (Hilbert) energy envelope -> logits cho BCE.
    x: (B,1,L) hoặc (B,L)
    return: (B,1,L)
    """
    if x.dim() == 2:
        xr = x  # (B,L)
    elif x.dim() == 3:
        xr = x[:, 0, :]  # (B,L)
    else:
        raise ValueError("x must be (B,L) or (B,1,L)")

    B, N = xr.shape
    Xf = torch.fft.fft(xr, dim=-1)  # complex

    h = torch.zeros((N,), device=xr.device, dtype=Xf.dtype)
    if N % 2 == 0:
        h[0] = 1
        h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    xa = torch.fft.ifft(Xf * h, dim=-1)  # analytic
    env = torch.abs(xa)                  # envelope -> Amp envelop?
    
    '''
    Ban đầu sử dụng entropy envelope
    '''
    
    # energy = env * env                   # energy

    # z = torch.log(energy + eps)          # stabilize
    # mu = z.mean(dim=-1, keepdim=True)
    # sd = z.std(dim=-1, keepdim=True).clamp_min(eps)
    # logits = (z - mu) / sd               # per-sample zscore
    # return logits.unsqueeze(1)  # (B,1,L)
    
    '''Điều chỉnh sử dụng trực tiếp amplitude envelope, không logits'''
    
    # normalize but avoid min-max collapse
    env = env / (env.mean(dim=-1, keepdim=True) + eps)
    env = env - 1.0          # center around 0
    prob = torch.sigmoid(env)

    return prob.unsqueeze(1)

    

def hilbert_envelope(x: torch.Tensor) -> torch.Tensor:
    """
    Compute Hilbert amplitude envelope.
    x: (B,1,L) or (B,L)
    return: (B,1,L)
    """
    if x.dim() == 2:
        xr = x
    elif x.dim() == 3:
        xr = x[:, 0, :]
    else:
        raise ValueError("x must be (B,L) or (B,1,L)")

    B, N = xr.shape
    Xf = torch.fft.fft(xr, dim=-1)

    h = torch.zeros(N, device=xr.device, dtype=Xf.dtype)
    if N % 2 == 0:
        h[0] = 1
        h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    xa = torch.fft.ifft(Xf * h, dim=-1)
    env = torch.abs(xa)

    return env.unsqueeze(1)

def gaussian_target_from_mask(qrs_mask: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    """
    qrs_mask: (B,1,L) hoặc (B,L), binary 0/1
    return: Gaussian soft target, shape (B,1,L), range [0,1]
    """
    if qrs_mask.dim() == 2:
        q = qrs_mask.unsqueeze(1)
    elif qrs_mask.dim() == 3:
        q = qrs_mask
    else:
        raise ValueError("qrs_mask must be (B,L) or (B,1,L)")

    q = q.float()
    device_local = q.device

    radius = max(1, int(3 * sigma))
    x = torch.arange(-radius, radius + 1, device=device_local, dtype=torch.float32)
    kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.max()
    kernel = kernel.view(1, 1, -1)

    g = F.conv1d(q, kernel, padding=radius)
    g = torch.clamp(g, 0.0, 1.0)
    return g


class CombinedLoss(nn.Module):
    """
    Loss = 0.3*MSE + 0.6*MAE + 0.1*BCE
    BCE dùng logits từ HS energy envelope của output, target là qrs_mask (0/1).

    forward(output, target, qrs_mask)
      - output:   (B,1,L)
      - target:   (B,1,L)
      - qrs_mask: (B,1,L) float {0,1}
    """
    
    
    '''
    Điều chỉnh CombinedLoss bằng cách thay đổi trọng số
    Option ban đầu TRƯỚC KHI CHỈNH 
    self.w_mse = 0.3
    self.w_mae = 0.6
    self.w_bce = 0.1
    
    Option 1 - bce0:
    self.w_mse = 0.4
    self.w_mae = 0.6
    self.w_bce = 0.0
    
    Option 2: bce = 0.2
    self.w_mse = 0.3
    self.w_mae = 0.5
    self.w_bce = 0.2
    
    Option 3: bce = 0.3
    self.w_mse = 0.2
    self.w_mae = 0.5
    self.w_bce = 0.3
    '''
    def __init__(self, pos_weight=None):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor([float(pos_weight)])
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

        # fixed ratios
        self.w_mse = 0.4
        self.w_mae = 0.6
        self.w_bce = 0.0

    def forward(self, output: torch.Tensor, target: torch.Tensor, qrs_mask: torch.Tensor) -> torch.Tensor:
        loss_mse = self.mse(output, target)
        loss_mae = self.mae(output, target)

        logits = amplitude_envelope(output)  # (B,1,L)
        
        '''
        Ban đầu là sử dụng logits
        '''

        # if self.pos_weight is None:
        #     loss_bce = F.binary_cross_entropy_with_logits(logits, qrs_mask)
        # else:
        #     loss_bce = F.binary_cross_entropy_with_logits(logits, qrs_mask, pos_weight=self.pos_weight)
        
        if self.pos_weight is None:
            loss_bce = F.binary_cross_entropy(logits, qrs_mask)
        else:
            loss_bce = F.binary_cross_entropy(logits, qrs_mask, weight=self.pos_weight)

        return self.w_mse * loss_mse + self.w_mae * loss_mae + self.w_bce * loss_bce