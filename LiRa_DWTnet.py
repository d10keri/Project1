import torch
import torch.nn as nn
from pytorch_wavelets import DWT1DForward, DWT1DInverse

# ===== Basic Residual Block (1D) =====
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels=None, stride=1):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

# ===== Band-wise processing block =====
class BandBlock(nn.Module):
    def __init__(self, channels, depth=2):
        super().__init__()
        self.net = nn.Sequential(
            *[ResidualBlock1D(channels, channels) for _ in range(depth)]
        )

    def forward(self, x):
        return self.net(x)

# ===== SE block cho fusion =====
class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, t = x.shape
        w = self.pool(x).view(b, c)       # (B, C)
        w = self.fc(w).view(b, c, 1)      # (B, C, 1)
        return x * w                      # channel-wise reweight

# ===== HybridCWT_LIRA (phiên bản mới, nhẹ + 2 nhánh) =====
class HybridCWT_LIRA(nn.Module):
    def __init__(self,
                 input_channels=1,
                 output_channels=1,
                 base_channels=64,
                 J=4,
                 wave='db6',
                 band_depth=2,
                 decoder_depth=3,
                 final_activation='relu',
                 **kwargs):
        super().__init__()
        self.J = J
        self.base_channels = base_channels

        # Các tham số mở rộng nhưng vẫn lấy từ kwargs (không đổi signature)
        conv_branch_depth = kwargs.get("conv_branch_depth", 2)
        use_se_fusion     = kwargs.get("use_se_fusion", True)
        se_reduction      = kwargs.get("se_reduction", 8)

        # Encoder conv chung
        self.enc_conv = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.encoder = self.enc_conv  # alias tương thích code cũ

        # DWT / IDWT
        self.dwt  = DWT1DForward(J=J, wave=wave)
        self.idwt = DWT1DInverse(wave=wave)

        # Xử lý theo băng (low + mỗi high level)
        self.proc_low = BandBlock(base_channels, depth=band_depth)
        self.proc_hi  = nn.ModuleList(
            [BandBlock(base_channels, depth=band_depth) for _ in range(J)]
        )

        # Nhánh CNN song song (làm trên feature sau encoder)
        conv_blocks = []
        for _ in range(conv_branch_depth):
            conv_blocks.append(ResidualBlock1D(base_channels, base_channels))
        self.conv_branch = nn.Sequential(*conv_blocks) if conv_blocks else nn.Identity()

        # SE fusion sau khi cộng 2 nhánh
        self.se_fusion = SEBlock1D(base_channels, reduction=se_reduction) \
            if use_se_fusion else nn.Identity()

        # Decoder (giữ base_channels -> base_channels để nhẹ)
        dec_layers = []
        in_ch = base_channels
        for _ in range(decoder_depth):
            dec_layers += [
                nn.Conv1d(in_ch, base_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(base_channels),
                nn.ReLU(inplace=True),
            ]
            in_ch = base_channels
        self.decoder = nn.Sequential(*dec_layers)

        # Final
        self.final_conv = nn.Conv1d(base_channels, output_channels,
                                    kernel_size=3, padding=1)

        act = final_activation.lower() if isinstance(final_activation, str) else None
        if act is None or act == "none":
            self.final_act = nn.Identity()
        elif act == 'relu':
            self.final_act = nn.ReLU()
        elif act == 'tanh':
            self.final_act = nn.Tanh()
        else:
            raise ValueError(f"Unknown final_activation: {final_activation}")

    def forward(self, x):
        """
        x: (B, 1, T) → y: (B, 1, T). pytorch_wavelets tự padding đối xứng.
        """
        # Encoder chung
        f = self.enc_conv(x)                 # (B, Cb, T)

        # Nhánh wavelet
        yl, yh_list = self.dwt(f)           # yl: (B, Cb, T/2^J); yh_list: J mức
        yl_p = self.proc_low(yl)
        yh_p = [self.proc_hi[k](yh_list[k]) for k in range(self.J)]
        f_wave = self.idwt((yl_p, yh_p))    # (B, Cb, T)

        # Nhánh CNN trực tiếp
        f_conv = self.conv_branch(f)        # (B, Cb, T)

        # Fusion 2 nhánh
        f_fused = f_wave + f_conv           # (B, Cb, T)
        f_fused = self.se_fusion(f_fused)   # (B, Cb, T)

        # Decoder + output
        y = self.decoder(f_fused)           # (B, Cb, T)
        y = self.final_conv(y)              # (B, 1, T)
        y = self.final_act(y)
        return y

    # Debug tiện ích
    def summary_once(self, T=1024, device="cpu"):
        self.eval()
        with torch.no_grad():
            x = torch.zeros(1, 1, T, device=device)
            print("Input:", tuple(x.shape))
            f = self.enc_conv(x); print("After encoder:", tuple(f.shape))

            yl, yh = self.dwt(f)
            print("yl:", tuple(yl.shape))
            for i, yhi in enumerate(yh):
                print(f"yh[{i}]:", tuple(yhi.shape))

            yl_p = self.proc_low(yl); print("proc yl:", tuple(yl_p.shape))
            yh_p = [self.proc_hi[i](yhi) for i, yhi in enumerate(yh)]
            for i, yhp in enumerate(yh_p):
                print(f"proc yh[{i}]:", tuple(yhp.shape))

            f_wave = self.idwt((yl_p, yh_p)); print("After IDWT (wave):", tuple(f_wave.shape))
            f_conv = self.conv_branch(f);     print("Conv branch:", tuple(f_conv.shape))

            f_fused = f_wave + f_conv
            f_fused = self.se_fusion(f_fused); print("Fused + SE:", tuple(f_fused.shape))

            y = self.decoder(f_fused); print("After decoder:", tuple(y.shape))
            y = self.final_conv(y);    print("After final_conv:", tuple(y.shape))
            y = self.final_act(y);     print("Output:", tuple(y.shape))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==== Test nhanh + in tham số ====
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HybridCWT_LIRA(
        input_channels=1,
        output_channels=1,
        base_channels=64,
        J=4,
        wave="db6",
        band_depth=1,
        decoder_depth=3,
        final_activation="tanh",   # phù hợp normalize [-1,1]
        # có thể override thêm:
        # conv_branch_depth=2,
        # use_se_fusion=True,
        # se_reduction=8,
    ).to(device)

    B, T = 1, 1024
    x = torch.randn(B, 1, T, device=device)
    with torch.no_grad():
        y = model(x)
    print("Input :", x.shape)
    print("Output:", y.shape)
    print(f"Trainable parameters: {count_parameters(model):,}")
