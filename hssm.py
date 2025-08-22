import torch
import torch.nn as nn


class HS2M(nn.Module):
    def __init__(self, C, m=128, r=8, token_dim=64, chunk_len=512):
        super().__init__()
        self.C = C
        self.m = m
        self.r = r
        self.d_g = token_dim
        self.chunk_len = chunk_len
        self.in_proj = nn.Conv2d(C, m, kernel_size=1, bias=True)
        self.out_proj = nn.Conv2d(m, C, kernel_size=1, bias=True)
        self.U_base = nn.Parameter(torch.randn(m, r) * 0.02)
        self.V_base = nn.Parameter(torch.randn(m, r) * 0.02)
        self.a_base = nn.Parameter(torch.zeros(m))
        self.mod_uv = nn.Linear(self.d_g, 2 * r)
        self.mod_a = nn.Linear(self.d_g, m)
        self.mod_B = nn.Linear(self.d_g, m)
        self.mod_C = nn.Linear(self.d_g, C)
        self.h0 = nn.Parameter(torch.zeros(m))
        nn.init.normal_(self.U_base, std=0.02)
        nn.init.normal_(self.V_base, std=0.02)

    def forward(self, x, g=None):
        # x: (B,C,H,W), g: (B,d_g)
        B, C, H, W = x.shape
        if g is None:
            g = x.new_zeros((B, self.d_g))
        elif g.shape[0] == 1 and B > 1:
            g = g.expand(B, -1)
        x_proj = self.in_proj(x)  # (B,m,H,W)
        # modulations
        uv = self.mod_uv(g)  # (B,2r)
        gamma_U = uv[:, : self.r].unsqueeze(1)  # (B,1,r)
        gamma_V = uv[:, self.r :].unsqueeze(1)  # (B,1,r)
        gamma_a = self.mod_a(g)  # (B,m)
        gamma_B = self.mod_B(g)  # (B,m)
        gamma_C = self.mod_C(g)  # (B,C)
        gamma_C = torch.clamp(gamma_C, -0.5, 0.5)  # Prevent extreme output scaling
        # Clamp modulation parameters for numerical stability
        gamma_U_clamped = torch.clamp(gamma_U, -0.5, 0.5)
        gamma_V_clamped = torch.clamp(gamma_V, -0.5, 0.5)
        gamma_a_clamped = torch.clamp(gamma_a, -2.0, 2.0)

        Ug = self.U_base.unsqueeze(0) * (1.0 + gamma_U_clamped)  # (B,m,r)
        Vg = self.V_base.unsqueeze(0) * (1.0 + gamma_V_clamped)  # (B,m,r)
        a = self.a_base.unsqueeze(0) + gamma_a_clamped  # (B,m)
        # sequence
        L = H * W
        x_seq = x_proj.view(B, self.m, L).permute(0, 2, 1).contiguous()  # (B,L,m)

        # Add numerical stability
        gamma_B_clamped = torch.clamp(gamma_B, -1.0, 1.0)  # Prevent extreme values
        x_seq = x_seq * (1.0 + gamma_B_clamped.unsqueeze(1))

        # pad to chunk_len
        L_orig = x_seq.shape[1]
        pad = (self.chunk_len - (L_orig % self.chunk_len)) % self.chunk_len
        if pad > 0:
            x_seq = torch.cat([x_seq, x_seq.new_zeros((B, pad, self.m))], dim=1)
        L = x_seq.shape[1]
        M = L // self.chunk_len
        x_chunks = x_seq.view(B, M, self.chunk_len, self.m)  # (B,M,Lc,m)
        h = x_chunks.new_zeros((B, M, self.m)) + self.h0.view(1, 1, -1)
        Vg_T = Vg.transpose(-1, -2).contiguous()  # (B,r,m)
        outs = []
        for t in range(self.chunk_len):
            xt = x_chunks[:, :, t, :]  # (B,M,m)

            # State-space computation with numerical stability
            v = torch.einsum("brm,bkm->bkr", Vg_T, h)  # (B,M,r)
            Av = torch.einsum("bmr,bkr->bkm", Ug, v)  # (B,M,m)
            ah = h * a.unsqueeze(1)

            # Combine state update with gradient clipping
            h_new = Av + ah + xt

            # Clip hidden states to prevent explosion
            h_new = torch.clamp(h_new, -10.0, 10.0)

            h = h_new

            # Output computation
            Cbase = self.out_proj.weight.view(self.C, self.m)
            Cg = Cbase.unsqueeze(0) * (1.0 + gamma_C.unsqueeze(-1))  # (B,C,m)
            yc = torch.einsum("bcm,bkm->bkc", Cg, h)  # (B,M,C)

            # Clip output to prevent NaN
            yc = torch.clamp(yc, -10.0, 10.0)

            outs.append(yc)
        out_stack = torch.stack(outs, dim=2)  # (B,M,Lc,C)
        out_flat = out_stack.view(B, L, C)
        if pad > 0:
            out_flat = out_flat[:, :L_orig, :]
        out = out_flat.permute(0, 2, 1).contiguous().view(B, C, H, W)

        # Debug: Check for NaN in HS2M output
        if torch.isnan(out).any():
            print(f"WARNING: NaN in HS2M output, input max: {torch.max(torch.abs(x))}")
            print(
                f"  g_token max: {torch.max(torch.abs(g)) if g is not None else 'None'}"
            )
            print(f"  Ug max: {torch.max(torch.abs(Ug))}")
            print(f"  Vg max: {torch.max(torch.abs(Vg))}")

        return out


# test
def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
