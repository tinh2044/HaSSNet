import torch
import torch.nn as nn
import torch.nn.functional as F
from sssm import SSSM
from loss import MainLoss


class Downsample(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size=3, stride=2, padding=1, bias=True
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class Upsample(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        out = self.conv(x)
        return out


class HaSSNet(nn.Module):
    """
    High-Resolution Image Restoration backbone.
    Encoder-Decoder with SSSMs (HS2M -> FFB -> RASK), skip connections, and residual output.

    Args:
        in_ch:   input channels (e.g., 3 for RGB)
        out_ch:  output channels (e.g., 3 for RGB restoration)
        base_ch: base width (e.g., 64). Stages scale as [1,2,4,8] * base_ch by default.
        depths:  list of number of SSSMs per stage (len=4). Example: [2,2,2,2]
        bottleneck_depth: number of SSSMs in bottleneck (latent) stage
        hs2m_cfg, ffb_cfg, rask_cfg: dict configs forwarded to each SSSM
        norm_in: whether to normalize input to [-1,1] range in forward
        final_activation: None | 'tanh' | 'sigmoid' for output activation
    """

    def __init__(
        self,
        in_ch=3,
        out_ch=3,
        base_ch=64,
        depths=(2, 2, 2, 2),
        bottleneck_depth=2,
        hs2m_cfg={},
        rask_cfg={},
        norm_in=False,
        loss_weights={},
        num_refine_blocks=1,
        **kwargs,
    ):
        super().__init__()
        assert len(depths) == 4, (
            "depths must have 4 integers (for 4 encoder/decoder stages)"
        )
        self.depths = depths
        self.norm_in = norm_in
        self.loss_weights = loss_weights
        self.num_refine_blocks = num_refine_blocks
        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        self.stem = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1, bias=True)

        enc_blocks = nn.ModuleList()
        downs = nn.ModuleList()
        ch = base_ch
        for si, d in enumerate(depths):
            stage = nn.Sequential(
                *[
                    SSSM(
                        C=ch,
                        hs2m_cfg=hs2m_cfg,
                        rask_cfg=rask_cfg,
                        use_ln=True,
                    )
                    for j in range(d)
                ]
            )
            enc_blocks.append(stage)
            if si < len(depths) - 1:
                downs.append(Downsample(ch, ch * 2))
                ch = ch * 2
        self.enc_blocks = enc_blocks
        self.downs = downs

        dec_blocks = nn.ModuleList()
        ups = nn.ModuleList()
        fuse_convs = nn.ModuleList()
        for si in reversed(range(len(depths) - 1)):
            up = Upsample(ch, ch // 2)
            ups.append(up)
            ch = ch // 2
            fuse_convs.append(nn.Conv2d(ch + chs[si], ch, kernel_size=1, bias=True))
            d = depths[si]
            dec_stage = nn.Sequential(
                *[
                    SSSM(
                        C=ch,
                        hs2m_cfg=hs2m_cfg,
                        rask_cfg=rask_cfg,
                        use_ln=True,
                    )
                    for j in range(d)
                ]
            )
            dec_blocks.append(dec_stage)

        self.refine_blocks = nn.Sequential(
            *[
                SSSM(C=ch, hs2m_cfg=hs2m_cfg, rask_cfg=rask_cfg, use_ln=True)
                for j in range(num_refine_blocks)
            ]
        )

        self.ups = ups
        self.fuse_convs = fuse_convs
        self.dec_blocks = dec_blocks

        self.head = nn.Conv2d(ch, out_ch, kernel_size=3, padding=1, bias=True)

        self.use_global_token = (
            loss_weights.get("use_global_token", True)
            if isinstance(loss_weights, dict)
            else True
        )
        global_token_dim = (
            loss_weights.get("global_token_dim", 64)
            if isinstance(loss_weights, dict)
            else 64
        )

        if self.use_global_token:
            self.global_token_encoder = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(ch, global_token_dim, kernel_size=1, bias=True),
                nn.GELU(),
                nn.Conv2d(global_token_dim, global_token_dim, kernel_size=1, bias=True),
            )
        else:
            self.global_token_encoder = None

        self.loss_fn = MainLoss(self.loss_weights)

    def forward(self, x, gt=None, g_token=None):
        """
        x: (B, in_ch, H, W)
        g_token: optional global token (B, token_dim) passed to SSSMs
        returns: restored image (B, out_ch, H, W)
        """

        inp = x
        feats = []
        x = self.stem(x)

        if (
            g_token is None
            and self.use_global_token
            and self.global_token_encoder is not None
        ):
            x_for_token = x.clone()
            for si, stage in enumerate(self.enc_blocks):
                for blk_idx, blk in enumerate(stage):
                    x_for_token = blk(x_for_token, g_token=None)  # Use None first
                if si == 0:
                    break
                if si < len(self.enc_blocks) - 1:
                    x_for_token = self.downs[si](x_for_token)

            g_token = self.global_token_encoder(x_for_token).squeeze(-1).squeeze(-1)
        elif g_token is None:
            g_token = None

        for si, stage in enumerate(self.enc_blocks):
            for blk_idx, blk in enumerate(stage):
                x = blk(x, g_token=g_token)

            feats.append(x)

            if si < len(self.enc_blocks) - 1:
                x = self.downs[si](x)

        for i, si in enumerate(reversed(range(len(self.depths) - 1))):
            x = self.ups[i](x)

            skip = feats[si]

            x = torch.cat([x, skip], dim=1)

            x = self.fuse_convs[i](x)

            for blk_idx, blk in enumerate(self.dec_blocks[i]):
                x = blk(x, g_token=g_token)

        out = self.refine_blocks(x)

        out = self.head(x)

        out = inp + out

        # Check for NaN before tanh
        if torch.isnan(out).any():
            valid_values = out[~torch.isnan(out)]
            if valid_values.numel() > 0:
                max_val = torch.max(torch.abs(valid_values))
            else:
                max_val = float("inf")
            print(f"WARNING: NaN detected before tanh, max abs value: {max_val}")

        out = torch.tanh(out)

        # Check for NaN after tanh
        if torch.isnan(out).any():
            print(
                f"WARNING: NaN detected after tanh, input max abs: {torch.max(torch.abs(inp))}"
            )
            # Replace NaN with zeros
            out = torch.where(torch.isnan(out), torch.zeros_like(out), out)

        loss = self.loss_fn(out, gt)

        return {
            "input": inp,
            "output": out,
            "loss": loss,
        }


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyze_parameter_distribution(model):
    """
    Phân tích phân bố parameters của HaSSNet theo từng module type.

    Returns:
        dict: Thông tin chi tiết về parameters của từng module type
    """
    total_params = 0
    module_params = {"HS2M": 0, "RASK": 0, "Other": 0}

    # Đếm parameters cho từng module
    for name, module in model.named_modules():
        param_count = sum(
            p.numel() for p in module.parameters(recurse=False) if p.requires_grad
        )

        if param_count > 0:  # Chỉ đếm modules có parameters
            if "hs2m" in name.lower():
                module_params["HS2M"] += param_count
            elif "rask" in name.lower():
                module_params["RASK"] += param_count
            else:
                module_params["Other"] += param_count

            total_params += param_count

    # Tính phần trăm
    percentages = {}
    for module_type, count in module_params.items():
        if total_params > 0:
            percentages[module_type] = (count / total_params) * 100
        else:
            percentages[module_type] = 0.0

    # Thống kê chi tiết
    stats = {
        "total_parameters": total_params,
        "module_counts": module_params,
        "percentages": percentages,
        "summary": {
            "HS2M_percentage": percentages["HS2M"],
            "RASK_percentage": percentages["RASK"],
            "Other_percentage": percentages["Other"],
        },
    }

    return stats


def print_parameter_analysis(model):
    """
    In ra phân tích parameters của HaSSNet một cách dễ đọc.
    """
    stats = analyze_parameter_distribution(model)

    print("=" * 60)
    print("PHÂN TÍCH PARAMETERS CỦA HASSNET")
    print("=" * 60)
    print(f"Tổng số parameters: {stats['total_parameters']:,}")
    print()

    print("PHÂN BỐ THEO MODULE TYPE:")
    print("-" * 40)
    for module_type, count in stats["module_counts"].items():
        percentage = stats["percentages"][module_type]
        print(f"{module_type:10}: {count:8,} parameters ({percentage:5.2f}%)")

    print()
    print("TỔNG KẾT:")
    print("-" * 40)
    print(f"HS2M chiếm: {stats['summary']['HS2M_percentage']:5.2f}%")
    print(f"RASK chiếm: {stats['summary']['RASK_percentage']:5.2f}%")
    print(f"Khác chiếm: {stats['summary']['Other_percentage']:5.2f}%")
    print("=" * 60)

    return stats


if __name__ == "__main__":
    print("=== DEBUGGING NaN ISSUE ===")

    B, C, H, W = 1, 3, 256, 256
    x = torch.randn(B, C, H, W)
    y = torch.randn(B, C, H, W)

    print(f"Input shape: {x.shape}")
    print(".4f")

    model = HaSSNet(
        in_ch=3,
        out_ch=3,
        base_ch=36,
        depths=(2, 2, 2, 3),
        num_refine_blocks=3,
    )

    print("Model created successfully")

    with torch.no_grad():
        print("Running forward pass...")
        y = model(x, gt=y, g_token=torch.randn(B, 64))

    print("Forward pass completed")
    print(f"Output keys: {list(y.keys())}")
    print(f"Output shape: {y['output'].shape}")

    print("Params (M):", count_params(model) / 1e6)

    print()

    print(y["loss"])

    print_parameter_analysis(model)
