from torch import nn as nn
import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
from einops import rearrange, repeat
import math

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

from mamba_ssm.modules.mamba_simple import Mamba



class atrous_SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            # v0:SS2D   v1: vanilla Mamba   v2:atrous vanilla Mamba     
            # v3: atrous SS2D   v4 efficient ss2d
            # v5: atrousv2 vanilla scan     v6: atrousv2 ss2d
            forward_type='v0',
            atrous_step=2,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.forward_type = forward_type
        self.K = self.atrous_step = 1
        if self.forward_type == 'v0' or self.forward_type == 'v0_seq' or self.forward_type == 'v3' or self.forward_type == 'v6':
            self.K = 4
        if self.forward_type == 'v2' or self.forward_type == 'v3':
            self.atrous_step = atrous_step
        if self.forward_type == 'v4': # efficient scan
            self.K = 4
            self.atrous_step = atrous_step
        if self.forward_type == 'v5' or self.forward_type == 'v6':
            self.atrous_step = atrous_step

        # self.forward_core = self.get_forward_core()

        self.forward_core = dict(
            v0=self.forward_corev0,     # ss2d
            v0_seq=self.forward_corev0_seq,     # ss2d sequence
            v1=self.forward_corev1,     # vanilla mamba
            v2=self.forward_corev2,     # atrous vanilla mamba
            v3=self.forward_corev3,     # atrous ss2d
            v4=self.forward_corev4,     # efficient scan
            v5=self.forward_corev5,     # atrous v2 vanilla mamba
            v6=self.forward_corev6,     # atrous v2 ss2d mamba
        ).get(forward_type, self.forward_corev0)

        self.selective_scan = selective_scan_fn

        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,  # 3          can not be even
            # kernel_size=3,
            padding=(d_conv - 1) // 2,
            # padding=1,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        if forward_type == 'v4':        # efficient scan  only 4
            self.x_proj = (nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
                           for _ in range(4))
        elif forward_type == 'v5' or forward_type == 'v6':
            self.x_proj = (nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
                           for _ in range(self.K * self.atrous_step * self.atrous_step))
        else:
            self.x_proj = (nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
                           for _ in range(self.K*self.atrous_step))

        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        if forward_type == 'v4':
            self.dt_projs = [
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs) for _ in range(4)]
        elif forward_type == 'v5' or forward_type == 'v6':
            self.dt_projs = [
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs) for _ in range(self.K*self.atrous_step*self.atrous_step)]
        else:
            self.dt_projs = [
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs) for _ in range(self.K*self.atrous_step)]

        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        if forward_type == 'v4':
            self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K, D, N)
            self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        elif forward_type =='v5' or forward_type == 'v6':
            self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K*self.atrous_step*self.atrous_step, merge=True)  # (K, D, N)
            self.Ds = self.D_init(self.d_inner, copies=self.K*self.atrous_step*self.atrous_step, merge=True)  # (K, D, N)
        else:
            self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K*self.atrous_step, merge=True)  # (K, D, N)
            self.Ds = self.D_init(self.d_inner, copies=self.K*self.atrous_step, merge=True)  # (K, D, N)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    """ VMamba SS2D implementation"""
    def forward_corev0(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = self.K

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        return y

    """ VMamba SS2D sequence implementation"""
    def forward_corev0_seq(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = self.K

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float()  # (b, k, d, l)
        dts = dts.contiguous().float()  # (b, k, d, l)
        Bs = Bs.float()  # (b, k, d_state, l)
        Cs = Cs.float()  # (b, k, d_state, l)

        As = -torch.exp(self.A_logs.float()).view(K, -1, self.d_state)  # (k, d, d_state)
        Ds = self.Ds.float().view(K, -1)  # (k, d)
        dt_projs_bias = self.dt_projs_bias.float().view(K, -1)  # (k, d)

        # assert len(xs.shape) == 4 and len(dts.shape) == 4 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 3 and len(Ds.shape) == 2 and len(dt_projs_bias.shape) == 2

        out_y = []
        for i in range(4):
            yi = self.selective_scan(
                xs[:, i], dts[:, i],
                As[i], Bs[:, i], Cs[:, i], Ds[i],
                delta_bias=dt_projs_bias[i],
                delta_softplus=True,
            ).view(B, -1, L)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, C)  # (B, H, W, C)
        return y.to(x.dtype)

    """ Mamba"""
    def forward_corev1(self, x: torch.Tensor):

        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = self.K

        xs = x.view(B, -1, L).unsqueeze(1)  # [B, 1, C, L]

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, -1, L)        # [B, C, L]
        assert out_y.dtype == torch.float

        return out_y

    def atrous_sample(self, x: torch.Tensor):
        B, K, C, L = x.shape
        atrous_L = math.ceil(L/self.atrous_step)
        xs = x.new_empty((B, self.atrous_step*self.K, C, atrous_L)) # [B, K*step, C, atrous_L]

        pad_length = (self.atrous_step - (L % self.atrous_step)) % self.atrous_step
        if pad_length != 0:
            padding = (0, pad_length)  # (left, right) padding
            x = F.pad(x, padding, mode='constant', value=0)

        for i in range(self.K):
            for j in range(self.atrous_step):
                xs[:, i*self.atrous_step+j, :, :] = x[:, i, :, j::self.atrous_step]

        return xs       # [B, K*step, C, atrous_L]

    def atrous_merge(self, y:torch.Tensor, L):
        B, K, C, aL = y.shape

        out_y = y.new_empty((B, self.K, C, aL*self.atrous_step))

        for i in range(self.K):
            for j in range(self.atrous_step):
                out_y[:, i, :, j::self.atrous_step] = y[:,i*self.atrous_step+j, :, :]

        pad_length = (self.atrous_step - (L % self.atrous_step)) % self.atrous_step
        if pad_length != 0:
            out_y = out_y[:, :, :, :-pad_length]

        return out_y        # [B, K*step, C, aL]

    """ atrousv1 Mamba, v1 is a different scan method from our paper. This is just a try"""
    def forward_corev2(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        B, C, H, W = x.shape
        L = H * W
        K = self.K * self.atrous_step
        aL = math.ceil(L / self.atrous_step)

        # atrous sample
        xs = x.view(B, -1, L).unsqueeze(1)  # [B, 1, C, L]
        xs = self.atrous_sample(xs)     # [B, step, C, aL]

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, aL), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, aL), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, aL)  # (b, k * d, al)
        dts = dts.contiguous().float().view(B, -1, aL)  # (b, k * d, al)
        Bs = Bs.float().view(B, K, -1, aL)  # (b, k, d_state, al)
        Cs = Cs.float().view(B, K, -1, aL)  # (b, k, d_state, al)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, aL)        # [B, k*d, aL]->[B, K, C, aL]
        assert out_y.dtype == torch.float

        out_y = self.atrous_merge(out_y, L).view(B, C, L)        # ->[B, 1, C, L]->[B,C,L]

        return out_y

    """atrousv1 Mamba + SS2D """
    def forward_corev3(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = self.K * self.atrous_step
        aL = math.ceil(L / self.atrous_step)

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, 4, d, l)

        xs = self.atrous_sample(xs)     # [B,4*step, D, aL]

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, aL), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, aL), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, aL)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, aL)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, aL)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, aL)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, aL)
        assert out_y.dtype == torch.float

        out_y = self.atrous_merge(out_y, L)       # ->[B, 4, C, L]

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        return y

    """ Efficient SS2D Mamba"""
    def forward_corev4(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        self.atrous_step = 2
        B, C, H, W = x.shape
        L = H * W
        K = self.K
        aL = math.ceil(L / (self.atrous_step*self.atrous_step))

        # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
        #                      dim=1).view(B, 2, -1, L)
        # x = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, 4, d, l)

        xs = self.efficient_scan(x)     # [B, 4, C, aL]

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, aL), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, aL), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, aL)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, aL)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, aL)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, aL)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, 4, -1, aL)        # [B, 4, C, aL]
        assert out_y.dtype == torch.float

        out_y = self.efficient_merge(out_y, H, W).view(B, C, L)       # ->[B, C, H, W]->[B, C, L]

        return out_y


    def efficient_scan(self, x:torch.Tensor):   # [B, C, H, W] -> [B, 4, C， H/w * W/w]
        step_size = 2       # efficient scan step size 2
        B, C, org_h, org_w = x.shape

        if org_w % step_size != 0:
            pad_w = step_size - org_w % step_size
            x = F.pad(x, (0, pad_w, 0, 0))
        W = x.shape[3]

        if org_h % step_size != 0:
            pad_h = step_size - org_h % step_size
            x = F.pad(x, (0, 0, 0, pad_h))
        H = x.shape[2]

        H = H // step_size
        W = W // step_size

        xs = x.new_empty((B, 4, C, H*W))

        xs[:, 0] = x[:, :, ::step_size, ::step_size].contiguous().view(B, C, -1)
        xs[:, 1] = x.transpose(dim0=2, dim1=3)[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1)
        xs[:, 2] = x[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1)
        xs[:, 3] = x.transpose(dim0=2, dim1=3)[:, :, 1::step_size, 1::step_size].contiguous().view(B, C, -1)

        return xs

    def efficient_merge(self, y: torch.Tensor, H, W, step_size=2):     # [B, 4, C, H/w * W/w] -> [B, C, H*W]
        B, K, C, aL = y.shape
        ori_h, ori_w = H, W
        H, W = math.ceil(ori_h / step_size), math.ceil(ori_w / step_size)

        new_h = H * step_size
        new_w = W * step_size

        y_out = y.new_empty((B, C, new_h, new_w))

        y_out[:, :, ::step_size, ::step_size] = y[:, 0].reshape(B, C, H, W)
        y_out[:, :, 1::step_size, ::step_size] = y[:, 1].reshape(B, C, W, H).transpose(dim0=2, dim1=3)
        y_out[:, :, ::step_size, 1::step_size] = y[:, 2].reshape(B, C, H, W)
        y_out[:, :, 1::step_size, 1::step_size] = y[:, 3].reshape(B, C, W, H).transpose(dim0=2, dim1=3)

        if ori_h != new_h or ori_w != new_w:
            y_out = y[:, :, :ori_h, :ori_w].contiguous()

        return y_out

    def atrousv2_scan(self, x: torch.Tensor):   # [B, C, H, W]
        step_size = self.atrous_step
        B, C, org_h, org_w = x.shape

        if org_w % step_size != 0:
            pad_w = step_size - org_w % step_size
            x = F.pad(x, (0, pad_w, 0, 0))
        W = x.shape[3]

        if org_h % step_size != 0:
            pad_h = step_size - org_h % step_size
            x = F.pad(x, (0, 0, 0, pad_h))
        H = x.shape[2]

        H = H // step_size
        W = W // step_size

        xs = x.new_empty((B, step_size*step_size, C, H*W))

        for i in range(step_size):
            for j in range(step_size):
                xs[:, i*step_size+j] = x[:, :, i::step_size, j::step_size].contiguous().view(B, C, -1)

        # xs[:, 0] = x[:, :, ::step_size, ::step_size].contiguous().view(B, C, -1)
        # xs[:, 1] = x[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1)
        # xs[:, 2] = x[:, :, 1::step_size, ::step_size].contiguous().view(B, C, -1)
        # xs[:, 3] = x[:, :, 1::step_size, 1::step_size].contiguous().view(B, C, -1)

        return xs, H, W, H*W

    def atrousv2_merge(self, y: torch.Tensor, ori_h, ori_w, step_size=2):
        # [B, step*step, C, H/step * W/step] -> [B, C, H*W]
        B, K, C, aL = y.shape       # K = step*step
        H, W = math.ceil(ori_h / step_size), math.ceil(ori_w / step_size)

        new_h = H * step_size
        new_w = W * step_size

        y_out = y.new_empty((B, C, new_h, new_w))

        for i in range(step_size):
            for j in range(step_size):
                y_out[:, :, i::step_size, j::step_size] = y[:, i*step_size+j].reshape(B, C, H, W)

        # y_out[:, :, ::step_size, ::step_size] = y[:, 0].reshape(B, C, H, W)
        # y_out[:, :, ::step_size, 1::step_size] = y[:, 1].reshape(B, C, W, H)
        # y_out[:, :, 1::step_size, ::step_size] = y[:, 2].reshape(B, C, H, W)
        # y_out[:, :, 1::step_size, 1::step_size] = y[:, 3].reshape(B, C, W, H)

        if ori_h != new_h or ori_w != new_w:
            y_out = y_out[:, :, :ori_h, :ori_w].contiguous()

        return y_out.view(B, C, -1)


    """ vanilla atrousv2 Mamba """
    def forward_corev5(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, orig_H, orig_W = x.shape
        orig_L = orig_H * orig_W
        K = self.K * self.atrous_step * self.atrous_step    # step*step*k (vanilla k=1)

        xs, H, W, aL = self.atrousv2_scan(x)     # [B,step*step, D, aL], aL = L/(step*step)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, aL), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, aL), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, aL)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, aL)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, aL)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, aL)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, aL).contiguous()
        assert out_y.dtype == torch.float

        out_y = self.atrousv2_merge(out_y, orig_H, orig_W, self.atrous_step) # [B,step*step,C,H/step,W/step]->[B,C,L]

        return out_y

    """ atrousv2 ss2d Mamba """
    def forward_corev6(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, orig_H, orig_W = x.shape
        orig_L = orig_H * orig_W
        K = self.K * self.atrous_step * self.atrous_step    # step * step * k
        N = self.atrous_step * self.atrous_step

        x_flip = x.view(B, -1, orig_L).flip(dims=[-1]).contiguous().view(B, -1, orig_H, orig_W)
        x_trans = x.transpose(dim0=2, dim1=3).contiguous()
        x_trans_flip = x.view(B, -1, orig_L).flip(dims=[-1]).contiguous().view(B, -1, orig_H, orig_W).contiguous().\
            transpose(dim0=2, dim1=3).contiguous()

        xs_normal, H, W, aL = self.atrousv2_scan(x)     # [B,step*step, D, aL], aL = L/(step*step)
        xs_flip, H, W, aL = self.atrousv2_scan(x_flip)
        xs_trans, H, W, aL = self.atrousv2_scan(x_trans)
        xs_trans_flip, H, W, aL = self.atrousv2_scan(x_trans_flip)

        # xs_normal, H, W, aL = self.atrousv2_scan(x)     # [B,step*step, D, aL], aL = L/(step*step)
        # xs_trans = xs_normal.view(B, -1, C, H, W).contiguous().transpose(dim0=3, dim1=4).contiguous().view(B, -1, C, aL).contiguous()
        # xs_flip = xs_normal.flip([-1]).contiguous()
        # xs_trans_flip = xs_trans.flip([-1]).contiguous()

        xs = torch.cat([xs_normal, xs_trans, xs_flip, xs_trans_flip], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, aL), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, aL), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, aL)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, aL)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, aL)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, aL)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, aL).contiguous()
        assert out_y.dtype == torch.float

        out_y_normal = self.atrousv2_merge(out_y[:, :N], orig_H, orig_W, self.atrous_step).view(B, -1, orig_H, orig_W).contiguous() # [B,step*step,C,H/step,W/step]->[B,C,L]
        out_y_trans = self.atrousv2_merge(out_y[:, N:2*N], orig_H, orig_W, self.atrous_step).view(B, -1, orig_H, orig_W).transpose(dim0=2, dim1=3).contiguous()
        out_y_flip = self.atrousv2_merge(out_y[:, 2*N:3*N], orig_H, orig_W, self.atrous_step).flip(dims=[-1]).contiguous().view(B, -1, orig_H, orig_W).contiguous()
        out_y_trans_flip = self.atrousv2_merge(out_y[:, 3*N:], orig_H, orig_W, self.atrous_step).\
            view(B, -1, orig_H, orig_W).contiguous().transpose(dim0=2, dim1=3).contiguous().view(B, -1, orig_L).flip(dims=[-1]).contiguous().view(B, -1, orig_H, orig_W)

        out_y = (out_y_normal + out_y_trans + out_y_flip + out_y_trans_flip).view(B, -1, orig_L)

        return out_y


    def forward(self, x: torch.Tensor):
        change = False
        if len(x.shape) == 3:   # In UltraLight-VMUNet，the input is [B, L, C]
            change = True
            B, L, C = x.shape
            H = W = int(math.sqrt(L))
            x = x.view(B, H, W, C)
        else:
            B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)   [B, H, W, D]

        x = x.permute(0, 3, 1, 2).contiguous()  # [B, D, H, W]
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y = self.forward_core(x)  # [B, D, L]
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # B, H, W, C
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)

        if change:
            out = out.view(B, L, C)

        return out


