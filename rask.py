import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class RASK(nn.Module):
    def __init__(
        self,
        C: int,
        K=4,
        region_size=16,
        hidden=64,
        use_sharpen=True,
        kernel_list=None,
    ):
        """
        Args:
            C: number of channels
            K: number of kernel choices
            kernel_list: list of tuples (kernel_size, dilation, separable_flag)
            region_size: size of regions for per_region prediction (region_size x region_size)
            hidden: hidden channels for region predictor
            use_sharpen: whether to apply anisotropic sharpening
        """
        super().__init__()
        self.C = C
        self.K = K
        self.region_size = region_size
        self.hidden = hidden
        self.use_sharpen = use_sharpen

        # Kernel bank: (kernel_size, dilation)
        if kernel_list is None:
            # 4 kernels: 3x3, 5x5, 3x3 dil2, 7x7
            kernel_list = [(3, 1), (5, 1), (3, 2), (7, 1)]
        assert len(kernel_list) == K, "kernel_list length must equal K"

        self.kernel_list = kernel_list

        # Build depthwise conv bank: groups=C
        self.dw_convs = nn.ModuleList()
        for ksize, dil in kernel_list:
            pad = dil * (ksize - 1) // 2
            conv = nn.Conv2d(
                C, C, kernel_size=ksize, padding=pad, dilation=dil, groups=C, bias=True
            )
            nn.init.kaiming_normal_(conv.weight, a=0.2)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
            self.dw_convs.append(conv)

        # Optimized Region-level predictor:
        # Reduce channels via depthwise conv for efficiency, then region pooling
        # then lightweight conv MLP on region grid
        C_reduced = max(8, C // 8)  # Reduced from C//4 to C//8

        self.reduce_conv = nn.Conv2d(
            C, C_reduced, kernel_size=1, bias=True
        )  # reduce to C'

        # Use depthwise conv for efficiency - ensure out_channels is divisible by groups
        hidden_adjusted = (
            hidden // C_reduced
        ) * C_reduced  # Make divisible by C_reduced
        if hidden_adjusted == 0:
            hidden_adjusted = C_reduced  # Minimum 1 channel per group
        self.pred_conv1 = nn.Conv2d(
            C_reduced,
            hidden_adjusted,
            kernel_size=3,
            padding=1,
            groups=C_reduced,
            bias=True,
        )
        self.pred_conv2 = nn.Conv2d(
            hidden_adjusted, K + 1, kernel_size=1, bias=True
        )  # outputs K logits + 1 beta per region
        nn.init.zeros_(self.pred_conv2.bias)  # stable start

        # Optimized refine conv with depthwise separable
        self.refine = nn.Sequential(
            # Depthwise separable conv
            nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C, bias=True),  # depthwise
            nn.Conv2d(C, C, kernel_size=1, bias=True),  # pointwise
            nn.GELU(),
            nn.Conv2d(C, C, kernel_size=1, bias=True),
        )

        # Optimized sharpening: shared sobel kernels for gradient computation
        # Use single conv for both gradients to reduce computation
        sobel_filters = torch.tensor(
            [
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],  # Gx
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],  # Gy
            ]
        ).view(2, 1, 3, 3)
        self.register_buffer("sobel_filters", sobel_filters)

        # Simplified second derivative kernels (shared across channels)
        # Use single conv with 3 output channels for dxx, dyy, dxy
        second_deriv_filters = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]],  # dxx
                [[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]],  # dyy
                [[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]],  # dxy
            ]
        ).view(3, 1, 3, 3)
        self.register_buffer("second_deriv_filters", second_deriv_filters)

    def forward(self, Z: torch.Tensor, scale_id: int = 0):
        """
        Args:
            Z: (B, C, H, W) tensor
            scale_id: integer encoding current scale (not used directly as embed here; could be concatenated)
        Returns:
            out: (B, C, H, W)
            debug dict (optional) with weights etc.
        """
        assert Z.ndim == 4 and Z.size(1) == self.C, "Z must be (B,C,H,W)"
        B, C, H, W = Z.shape  # Z: (B, C, H, W) - input feature from FFB/HS2M
        r = self.region_size
        assert H % r == 0 and W % r == 0, "H and W must be divisible by region_size"

        # Calculate region grid dimensions
        G_h, G_w = H // r, W // r  # G_h, G_w: number of regions per dimension

        # ----------------------
        # 1) Region-level features and predictor
        # ----------------------
        # Optimized reduce channels: Z (B, C, H, W) -> red (B, C', H, W) where C' = max(8, C//8)
        red = self.reduce_conv(Z)  # red: (B, C', H, W)

        # pool per region (non-overlap): red (B, C', H, W) -> red_region (B, C', G_h, G_w)
        red_region = F.avg_pool2d(
            red, kernel_size=r, stride=r
        )  # red_region: (B, C', G_h, G_w)

        # Optimized predictor with depthwise conv
        # red_region (B, C', G_h, G_w) -> x (B, hidden, G_h, G_w)
        x = self.pred_conv1(red_region)  # depthwise conv for efficiency
        x = F.gelu(x)  # activation

        # Conv1x1: x (B, hidden, G_h, G_w) -> logits_beta (B, K+1, G_h, G_w)
        logits_beta = self.pred_conv2(x)  # logits_beta: (B, K+1, G_h, G_w)

        # separate logits and beta
        # logits: (B, K, G_h, G_w) - K logits for kernel weights
        logits = logits_beta[:, : self.K, :, :]  # logits: (B, K, G_h, G_w)
        # beta_reg: (B, 1, G_h, G_w) - 1 logit for beta parameter (anisotropic sharpening)
        beta_reg = logits_beta[:, self.K :, :, :]  # beta_reg: (B, 1, G_h, G_w)

        # softmax across K for region weights: logits (B, K, G_h, G_w) -> weights_reg (B, K, G_h, G_w)
        weights_reg = torch.softmax(logits, dim=1)  # weights_reg: (B, K, G_h, G_w)

        # upsample weights to full resolution (bilinear or nearest)
        # weights_reg (B, K, G_h, G_w) -> weights_up (B, K, H, W)
        weights_up = F.interpolate(
            weights_reg, size=(H, W), mode="nearest"
        )  # weights_up: (B, K, H, W)

        # move K to last dim for easy broadcasting: (B, K, H, W) -> (B, H, W, K)
        weights_up_perm = weights_up.permute(
            0, 2, 3, 1
        ).contiguous()  # weights_up_perm: (B, H, W, K)

        # beta: apply activation to ensure positivity, upsample to pixel grid
        # beta_reg (B, 1, G_h, G_w) -> (B, 1, G_h, G_w) via sigmoid
        beta_reg = torch.sigmoid(beta_reg)  # beta_reg: (B, 1, G_h, G_w) approx (0,1)

        # upsample beta: beta_reg (B, 1, G_h, G_w) -> beta_up (B, 1, H, W)
        beta_up = F.interpolate(
            beta_reg, size=(H, W), mode="nearest"
        )  # beta_up: (B, 1, H, W)

        # ----------------------
        # 2) Kernel bank responses (depthwise)
        # ----------------------

        # compute each depthwise conv response: list of (B,C,H,W)
        outs = []
        for i, conv in enumerate(self.dw_convs):
            # Z (B, C, H, W) -> outs_k (B, C, H, W) via depthwise conv
            outs_k = conv(Z)  # outs_k: (B, C, H, W)
            outs.append(outs_k)

        # stack into (B, K, C, H, W) then move to (B, C, H, W, K) for weighted sum
        # stack K outputs: list of (B,C,H,W) -> stacked (B, K, C, H, W)
        stacked = torch.stack(outs, dim=1)  # stacked: (B, K, C, H, W)

        # permute to (B, C, H, W, K) for easy broadcasting with weights
        stacked = stacked.permute(
            0, 2, 3, 4, 1
        ).contiguous()  # stacked: (B, C, H, W, K)

        # weights_up_perm is (B, H, W, K) -> need shape (B, 1, H, W, K) to broadcast with stacked
        # weights_up_perm (B, H, W, K) -> w_bhwk (B, 1, H, W, K) for channel broadcasting
        w_bhwk = weights_up_perm.unsqueeze(1)  # w_bhwk: (B, 1, H, W, K)

        # weighted sum: sum_k w_k * stacked[...,k]
        # stacked (B, C, H, W, K) * w_bhwk (B, 1, H, W, K) -> sum over K -> fused (B, C, H, W)
        fused = (stacked * w_bhwk).sum(dim=-1)  # fused: (B, C, H, W)

        # ----------------------
        # 3) Optional anisotropic sharpening (optimized)
        # ----------------------
        if self.use_sharpen:
            # compute channel-mean for gradients: Z (B, C, H, W) -> Z_mean (B, 1, H, W)
            Z_mean = Z.mean(dim=1, keepdim=True)  # Z_mean: (B, 1, H, W)

            # Optimized: single conv for both Gx, Gy gradients
            gradients = F.conv2d(Z_mean, self.sobel_filters, padding=1)  # (B, 2, H, W)
            Gx, Gy = gradients[:, 0:1], gradients[:, 1:2]  # Split into Gx, Gy

            # compute normalized gradient components (u,v) = (Gx, Gy)/norm
            norm = torch.sqrt(Gx * Gx + Gy * Gy + 1e-6)  # (B, 1, H, W)
            u = Gx / (norm + 1e-6)  # cos theta
            v = Gy / (norm + 1e-6)  # sin theta

            # directional coefficients: a, b, c (B, 1, H, W)
            a = u * u
            b = 2.0 * u * v
            c = v * v

            # Optimized: single conv for all second derivatives
            second_derivs = F.conv2d(
                fused,
                self.second_deriv_filters.repeat(self.C, 1, 1, 1),
                groups=self.C,
                padding=1,
            )  # (B, 3*C, H, W)
            dxx, dyy, dxy = torch.chunk(second_derivs, 3, dim=1)  # Each (B, C, H, W)

            # Expand directional coefficients to match channels
            a_c = a.expand(-1, self.C, -1, -1)
            b_c = b.expand(-1, self.C, -1, -1)
            c_c = c.expand(-1, self.C, -1, -1)

            # directional sharpening: δ_θ = a*dxx + b*dxy + c*dyy
            delta_theta = a_c * dxx + b_c * dxy + c_c * dyy

            # Apply sharpening with beta scaling
            beta_c = beta_up.expand(-1, self.C, -1, -1)
            fused = fused + beta_c * delta_theta

        # final refine conv: fused (B, C, H, W) -> out (B, C, H, W)

        out = self.refine(fused)  # out: (B, C, H, W)

        # return out and some debug info
        debug = {
            "weights_grid": weights_reg,
            "weights_up": weights_up,
            "beta_reg": beta_reg,
            "beta_up": beta_up,
        }
        return out, debug


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cpu")  # set to "cuda" if GPU available
    B = 4
    C = 64
    H = 256
    W = 256  # small test sizes
    region_size = 8

    # Test current version
    print("=== RASK TEST ===")
    model_opt = RASK(
        C=C,
        K=4,
        region_size=region_size,
        hidden=64,
        use_sharpen=True,
    ).to(device)

    # Calculate optimized params
    opt_params = sum(p.numel() for p in model_opt.parameters() if p.requires_grad)
    print(f"Optimized RASK params: {opt_params:,}")

    # Test inference
    x = torch.randn(B, C, H, W, device=device)
    t0 = time.time()
    with torch.no_grad():
        y_opt, dbg_opt = model_opt(x, scale_id=0)
    t1 = time.time()

    print("Input shape:", x.shape)
    print("Output shape:", y_opt.shape)
    print("Weights grid shape (regions):", dbg_opt["weights_grid"].shape)  # (B,K,Gh,Gw)
    print("Beta grid shape:", dbg_opt["beta_reg"].shape)
    print("Elapsed (no grad): %.1f ms" % ((t1 - t0) * 1000))

    print("\n=== RASK CONFIGURATION ===")
    print("Current configuration:")
    print(f"   K=4, hidden=64, use_sharpen=True")
    print(f"   Kernel list: [(3,1), (5,1), (3,2), (7,1)]")

    print(f"\n📊 Parameter count: {opt_params:,}")

    # Create current config dict
    current_config = {
        "K": 4,
        "hidden": 64,
        "region_size": 16,
        "use_sharpen": True,
        "kernel_list": [(3, 1), (5, 1), (3, 2), (7, 1)],
    }
    print(f"\n🔧 Current rask_cfg dict:")
    print(f"   rask_cfg = {current_config}")

    print(f"\n💡 For optimization, consider:")
    print(f"   - Reducing K from 4 to 3 (remove 7x7 kernel)")
    print(f"   - Reducing hidden from 64 to 32")
    print(f"   - Using C//8 instead of C//4 for reduced channels")
