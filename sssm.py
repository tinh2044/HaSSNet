import torch
import torch.nn as nn
import time
from hssm import HS2M
from rask import RASK


class SSSM(nn.Module):
    def __init__(
        self,
        C,
        hs2m_cfg=dict(m=32, r=2, token_dim=64, chunk_len=256),
        rask_cfg=dict(K=3, region_size=8, hidden=16, use_sharpen=True),
        use_ln=True,
    ):
        """
        SSSM composes HS2M -> FFB -> RASK with LayerNorms and residual connections
        according to the mathematical description: LayerNorm → HS2M → add → LayerNorm → FFB → add → LayerNorm → RASK → add

        Args:
            C: channel number
            hs2m_cfg: dict passed to HS2M constructor (in_ch inferred as C)
            ffb_cfg: dict for FFB (C will be passed)
            rask_cfg: dict for RASK (C will be passed)
            use_ln: whether to apply LayerNorm before submodules
        """
        super().__init__()
        self.C = C
        self.use_ln = use_ln

        # Prior convolution 3x3 as described in sssm.md
        self.prior_conv = nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False)

        # LayerNorm operate on channels: we'll apply LN on (B,H,W,C) representation
        if use_ln:
            self.ln1 = nn.LayerNorm(C)  # Before HS2M
            self.ln2 = nn.LayerNorm(C)  # Before RASK
        else:
            self.ln1 = self.ln2 = nn.Identity()

        # Instantiate modules (HS2M/FFB/RASK). If classes not available, raise informative error.
        # HS2M expects in_ch as first arg - ensure it matches the actual channel count
        self.hs2m = HS2M(C=C, **hs2m_cfg)
        self.rask = RASK(C=C, **rask_cfg)

        # Optional 1x1 skip projection if input/output channel mismatch
        self.skip_proj = nn.Identity()  # keep identity for same-channel case

    def forward(self, x, g_token=None):
        """
        Forward pass following the mathematical description in sssm.md:
        P = Conv3x3(X^(0)), S = H(LN(X^(0)); P), M = X^(0) + S,
        F = F(LN(M); P), N = M + F, R = R(LN(N); P), Y = N + R

        Args:
            x: (B, C, H, W) - input features X^(0)
            g_token: (B, d_g) optional global token to condition HS2M
        Returns:
            out: (B, C, H, W) - final output Y
            debug: dict with intermediate tensors / shapes (optional)
        """

        # 1) Prior convolution: P = Conv3x3(X^(0))
        P = self.prior_conv(x)  # P ∈ ℝ^(B×C×H×W)

        # 2) LayerNorm trước HS2M: X̂ = LN(X^(0))
        if self.use_ln:
            # LN expects last dim = C when applied to (B, H, W, C)
            x_hw = P.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)

            x_ln = self.ln1(x_hw)  # X̂ = LN(X^(0)) ∈ ℝ^(B×H×W×C)
            x_ln = x_ln.permute(0, 3, 1, 2).contiguous()
        else:
            x_ln = P

        # 3) HS2M: S = H(X̂; P)

        # HS2M expects (B,C,H,W) and receives prior P for conditioning
        S = self.hs2m(x_ln, g_token)  # S = H(X̂; P) ∈ ℝ^(BxC×H×W)

        # Debug: Check for NaN in HS2M output
        if torch.isnan(S).any():
            print(
                f"WARNING: NaN in SSSM HS2M output, input max: {torch.max(torch.abs(x_ln))}"
            )

        # 4) Residual add: M = X^(0) + S

        # Convert S back to channel-first for addition
        M = x + S  # M = X^(0) + S ∈ ℝ^(B×C×H×W)

        # 5) LayerNorm before RASK: M̃ = LN(M)
        if self.use_ln:
            M_hw = M.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)

            M_ln = self.ln2(M_hw)  # M̃ = LN(M) ∈ ℝ^(B×H×W×C)
        else:
            M_ln = M.permute(0, 2, 3, 1).contiguous()

        # 6) RASK: R = R(M̃; P)

        # RASK expects (B,C,H,W) format, so convert M_ln from (B,H,W,C) to (B,C,H,W)
        M_ln_chw = M_ln.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        # RASK accepts (B,C,H,W) and receives prior P for conditioning
        R, _ = self.rask(M_ln_chw)  # R = R(N̂; P) ∈ ℝ^(B×C×H×W)

        # Debug: Check for NaN in RASK output
        if torch.isnan(R).any():
            print(
                f"WARNING: NaN in SSSM RASK output, input max: {torch.max(torch.abs(M_ln_chw))}"
            )

        # 10) Final residual add: Y = N + R
        # Ensure skip projection if needed
        res = self.skip_proj(x)  # identity by default

        # Final output: Y = N + R (not x + R as in original code)
        out = res + R  # Y = N + R ∈ ℝ^(B×C×H×W)

        return out


# --------------------
# Smoke test SSSM with dummy tensor
# --------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cpu")  # use "cuda" if available
    B = 1
    C = 32
    H = 256
    W = 256  # small sizes for quick test

    print("Testing SSSM with debug enabled...")

    # instantiate SSSM using previously defined module classes
    sssm = SSSM(
        C=C,
        hs2m_cfg=dict(state_size=16, rank=2, token_dim=32, chunk_len=256),
        rask_cfg=dict(K=4, region_size=8, hidden=64, use_sharpen=True),
        use_ln=True,
    ).to(device)

    x = torch.randn(B, C, H, W, device=device)
    g = torch.randn(B, 32, device=device)  # token size matches hs2m token_dim

    t0 = time.time()
    with torch.no_grad():
        out = sssm(x, g_token=g)
    t1 = time.time()

    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    print("Elapsed (no grad): %.1f ms" % ((t1 - t0) * 1000))
