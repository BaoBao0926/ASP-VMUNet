import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
import math
from mamba_ssm.modules.mamba_simple import Mamba
from models.Atrous.atrous_modules import atrous_SS2D
import torchvision




class CNNlayer(nn.Module):

    def __init__(self, hidden_dim, out_dim  ):
        super(CNNlayer, self).__init__()
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1, groups=hidden_dim)
        self.conv_branch = nn.Sequential(
            self.conv,
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_dim, 1)
    )
    def forward(self, x):
        return self.conv_branch(x)


class BiAttn(nn.Module):
    def __init__(self, in_channels, act_ratio=0.125, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = nn.LayerNorm(in_channels)
        self.global_reduce = nn.Linear(in_channels, reduce_channels)
        # self.local_reduce = nn.Linear(in_channels, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_channels)
        # self.spatial_select = nn.Linear(reduce_channels * 2, 1)
        self.gate_fn = gate_fn()

    def forward(self, x:torch.Tensor):   # [B, C, H, W]
        ori_x = x.permute(0, 2, 3, 1)   # [B, H, W, C]
        x = self.norm(x.permute(0, 2, 3, 1))
        x_global = x.mean([1, 2], keepdim=True)
        x_global = self.act_fn(self.global_reduce(x_global))
        # x_local = self.act_fn(self.local_reduce(x))

        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn)

        attn = c_attn
        out = ori_x * attn
        return out.permute(0, 3, 1, 2)


class SKAttention(nn.Module):
    # 通道数channel, 卷积核尺度kernels, 降维系数reduction, 分组数group, 降维后的通道数L
    def __init__(self, channel=512, reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)

        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(2):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x_conv, x_mamba):
        B, C, H, W = x_conv.size()

        conv_outs = [x_conv, x_mamba]
        feats = torch.stack(conv_outs, 0)  #  (B,C,H,W)-->(K,B,C,H,W)

        ## Fuse
        U = sum(conv_outs)  # (K,B,C,H,W)-->sum-->(B,C,H,W)
        #  average pooling (B,C,H,W)-->mean-->(B,C,H)-->mean-->(B,C) 
        S = U.mean(-1).mean(-1)
        #  (B,C)-->(B,d)
        Z = self.fc(S)

        weights = []
        for fc in self.fcs:
            weight = fc(Z)  #  (B,d)-->(B,C)
            weights.append(weight.view(B, C, 1, 1))  # (B,C)-->(B,C,1,1)
        scale_weight = torch.stack(weights, 0)  #  (K,B,C,1,1)
        scale_weight = self.softmax(scale_weight)  #  (K,B,C,1,1)

        ##  Select (K,B,C,1,1) * (K,B,C,H,W) = (K,B,C,H,W)-->sum-->(B,C,H,W)
        V = (scale_weight * feats).sum(0)
        return V




class PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2, if_ss2d=True, forward_type='v1',
                 atrous_step=2, if_mapping=False, if_CNN=False, if_SE=False, if_SK=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        if not if_ss2d:
            self.mamba = Mamba(d_model=input_dim // 4, d_state=d_state, d_conv=d_conv,  expand=expand,  )
        else:
            # d_conv in ss2d cannot be even(default is 3)
            d_conv = d_conv - 1 if d_conv % 2 == 0 else d_conv
            self.mamba = atrous_SS2D(d_model=input_dim//4, d_state=d_state, d_conv=d_conv, forward_type=forward_type,
                                     atrous_step=atrous_step)

        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

        self.if_mapping = if_mapping
        if self.if_mapping:
            self.mapping_proj = nn.Linear(input_dim, input_dim)
        self.if_CNN = if_CNN
        if self.if_CNN:
            self.CNNlayer = CNNlayer(hidden_dim=input_dim, out_dim=output_dim)
        self.if_SE = if_SE
        if if_SE:
            self.SE = BiAttn(in_channels=output_dim)
        self.if_SK = if_SK
        if self.if_SK:
            self.SK = SKAttention(channel=output_dim, reduction=2)      # PVM dim is small, reduction 2 is ok


    def forward(self, x):       # [B, C, H, W]
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        if self.if_CNN:
            out_CNN = self.CNNlayer(x)

        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        if self.if_mapping:
            x_norm = self.mapping_proj(x_norm)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out_mamba = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)

        if not self.if_CNN:
            return out_mamba
        else:
            if self.if_SE:
                out_mamba, out_CNN = self.SE(out_mamba), self.SE(out_CNN)

            out = out_mamba + out_CNN if not self.if_SK else self.SK(out_CNN, out_mamba)
            return out


class PVMLayer_shifted(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2, if_shifted_round=False, if_ss2d=True,
                 forward_type='v1', atrous_step=2, if_CNN=False, if_SE=False, if_SK=False,):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.if_shifted_round = if_shifted_round
        if not if_ss2d:
            if if_shifted_round:
                self.mamba = Mamba(d_model=input_dim // 4, d_state=d_state,  d_conv=d_conv,  expand=expand,  )

            else:
                self.mamba = Mamba(d_model=input_dim // 4, d_state=d_state,  d_conv=d_conv,  expand=expand,  )
                self.mamba_small = Mamba(d_model=input_dim // 8, d_state=d_state,  d_conv=d_conv,  expand=expand,  )
        else:
            d_conv = d_conv - 1 if d_conv % 2 == 0 else d_conv    # d_conv in ss2d cannot be even(default is 3)
            if if_shifted_round:
                self.mamba = atrous_SS2D(d_model=input_dim // 4, d_state=d_state, d_conv=d_conv,
                                         forward_type=forward_type, atrous_step=atrous_step)
            else:
                self.mamba = atrous_SS2D(d_model=input_dim // 4, d_state=d_state, d_conv=d_conv,
                                         forward_type=forward_type, atrous_step=atrous_step)
                self.mamba_small = atrous_SS2D(d_model=input_dim // 8, d_state=d_state, d_conv=d_conv,
                                         forward_type=forward_type, atrous_step=atrous_step)
        self.if_CNN = if_CNN
        if self.if_CNN:
            self.CNNlayer = CNNlayer(hidden_dim=input_dim, out_dim=output_dim )
        self.if_SE = if_SE
        if self.if_SE:
            self.SE = BiAttn(output_dim)
        self.if_SK = if_SK
        if self.if_SK:
            self.SK = SKAttention(channel=output_dim, reduction=2)      # PVM dim is small, reduction 2 is ok

        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        if self.if_CNN:
            out_CNN = self.CNNlayer(x)

        if self.if_shifted_round:  # yes shifted round
            x1, x2, x3, x4, x5, x6, x7, x8 = torch.chunk(x_norm, 8, dim=2)
            x1 = torch.cat([x8, x1], dim=2)
            x2 = torch.cat([x2, x3], dim=2)
            x3 = torch.cat([x4, x5], dim=2)
            x4 = torch.cat([x6, x7], dim=2)
            x_mamba1 = self.mamba(x1) + self.skip_scale * x1
            x_mamba2 = self.mamba(x2) + self.skip_scale * x2
            x_mamba3 = self.mamba(x3) + self.skip_scale * x3
            x_mamba4 = self.mamba(x4) + self.skip_scale * x4
            t8, t1 = torch.chunk(x_mamba1, 2, dim=2)
            x_mamba = torch.cat([t1, x_mamba2, x_mamba3, x_mamba4, t8], dim=2)
        else:  # no shifted around
            x1, x2, x3, x4, x5, x6, x7, x8 = torch.chunk(x_norm, 8, dim=2)
            x2 = torch.cat([x2, x3], dim=2)
            x3 = torch.cat([x4, x5], dim=2)
            x4 = torch.cat([x6, x7], dim=2)
            x_mamba1 = self.mamba_small(x1) + self.skip_scale * x1
            x_mamba2 = self.mamba(x2) + self.skip_scale * x2
            x_mamba3 = self.mamba(x3) + self.skip_scale * x3
            x_mamba4 = self.mamba(x4) + self.skip_scale * x4
            x_mamba5 = self.mamba_small(x8) + self.skip_scale * x8
            x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4, x_mamba5], dim=2)
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out_mamba = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)

        if not self.if_CNN:
            return out_mamba
        else:
            if self.if_SE:
                out_mamba, out_CNN = self.SE(out_mamba), self.SE(out_CNN)

            out = out_mamba + out_CNN if not self.if_SK else self.SK(out_CNN, out_mamba)
            return out



class Channel_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att1 = nn.Linear(c_list_sum, c_list[0]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[0], 1)
        self.att2 = nn.Linear(c_list_sum, c_list[1]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att3 = nn.Linear(c_list_sum, c_list[2]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.att4 = nn.Linear(c_list_sum, c_list[3]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[3], 1)
        self.att5 = nn.Linear(c_list_sum, c_list[4]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[4], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, t3, t4, t5):
        att = torch.cat((self.avgpool(t1),
                         self.avgpool(t2),
                         self.avgpool(t3),
                         self.avgpool(t4),
                         self.avgpool(t5)), dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        att4 = self.sigmoid(self.att4(att))
        att5 = self.sigmoid(self.att5(att))
        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(t3)
            att4 = att4.transpose(-1, -2).unsqueeze(-1).expand_as(t4)
            att5 = att5.transpose(-1, -2).unsqueeze(-1).expand_as(t5)
        else:
            att1 = att1.unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).expand_as(t3)
            att4 = att4.unsqueeze(-1).expand_as(t4)
            att5 = att5.unsqueeze(-1).expand_as(t5)

        return att1, att2, att3, att4, att5


class Spatial_Att_Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
                                           nn.Sigmoid())

    def forward(self, t1, t2, t3, t4, t5):
        t_list = [t1, t2, t3, t4, t5]
        att_list = []
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.shared_conv2d(att)
            att_list.append(att)
        return att_list[0], att_list[1], att_list[2], att_list[3], att_list[4]


class SC_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()

        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()

    def forward(self, t1, t2, t3, t4, t5):
        r1, r2, r3, r4, r5 = t1, t2, t3, t4, t5

        satt1, satt2, satt3, satt4, satt5 = self.satt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = satt1 * t1, satt2 * t2, satt3 * t3, satt4 * t4, satt5 * t5

        r1_, r2_, r3_, r4_, r5_ = t1, t2, t3, t4, t5
        t1, t2, t3, t4, t5 = t1 + r1, t2 + r2, t3 + r3, t4 + r4, t5 + r5

        catt1, catt2, catt3, catt4, catt5 = self.catt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = catt1 * t1, catt2 * t2, catt3 * t3, catt4 * t4, catt5 * t5

        return t1 + r1_, t2 + r2_, t3 + r3_, t4 + r4_, t5 + r5_


""" UltraLight VM UNet """
class UltraLight_VM_UNet(nn.Module):

    def __init__(self, num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64],
                 split_att='fc', bridge=True):
        super().__init__()

        self.bridge = bridge

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            PVMLayer(input_dim=c_list[2], output_dim=c_list[3])
        )
        self.encoder5 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[4])
        )
        self.encoder6 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[5])
        )

        if bridge:
            self.scab = SC_Att_Bridge(c_list, split_att)
            print('SC_Att_Bridge was used')

        self.decoder1 = nn.Sequential(
            PVMLayer(input_dim=c_list[5], output_dim=c_list[4])
        )
        self.decoder2 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[3])
        )
        self.decoder3 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[2])
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c1, H/4, W/4

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # b, c2, H/8, W/8

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # b, c3, H/16, W/16

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # b, c4, H/32, W/32

        if self.bridge: t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)

        out = F.gelu(self.encoder6(out))  # b, c5, H/32, W/32

        out5 = F.gelu(self.dbn1(self.decoder1(out)))  # b, c4, H/32, W/32
        out5 = torch.add(out5, t5)  # b, c4, H/32, W/32

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c3, H/16, W/16
        out4 = torch.add(out4, t4)  # b, c3, H/16, W/16

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c2, H/8, W/8
        out3 = torch.add(out3, t3)  # b, c2, H/8, W/8

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c1, H/4, W/4
        out2 = torch.add(out2, t2)  # b, c1, H/4, W/4

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c0, H/2, W/2
        out1 = torch.add(out1, t1)  # b, c0, H/2, W/2

        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)  # b, num_class, H, W

        return torch.sigmoid(out0)

    class UltraLight_VM_UNet(nn.Module):

        def __init__(self, num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64],
                     split_att='fc', bridge=True):
            super().__init__()

            self.bridge = bridge

            self.encoder1 = nn.Sequential(
                nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
            )
            self.encoder2 = nn.Sequential(
                nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
            )
            self.encoder3 = nn.Sequential(
                nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
            )
            self.encoder4 = nn.Sequential(
                PVMLayer(input_dim=c_list[2], output_dim=c_list[3])
            )
            self.encoder5 = nn.Sequential(
                PVMLayer(input_dim=c_list[3], output_dim=c_list[4])
            )
            self.encoder6 = nn.Sequential(
                PVMLayer(input_dim=c_list[4], output_dim=c_list[5])
            )

            if bridge:
                self.scab = SC_Att_Bridge(c_list, split_att)
                print('SC_Att_Bridge was used')

            self.decoder1 = nn.Sequential(
                PVMLayer(input_dim=c_list[5], output_dim=c_list[4])
            )
            self.decoder2 = nn.Sequential(
                PVMLayer(input_dim=c_list[4], output_dim=c_list[3])
            )
            self.decoder3 = nn.Sequential(
                PVMLayer(input_dim=c_list[3], output_dim=c_list[2])
            )
            self.decoder4 = nn.Sequential(
                nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
            )
            self.decoder5 = nn.Sequential(
                nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
            )
            self.ebn1 = nn.GroupNorm(4, c_list[0])
            self.ebn2 = nn.GroupNorm(4, c_list[1])
            self.ebn3 = nn.GroupNorm(4, c_list[2])
            self.ebn4 = nn.GroupNorm(4, c_list[3])
            self.ebn5 = nn.GroupNorm(4, c_list[4])
            self.dbn1 = nn.GroupNorm(4, c_list[4])
            self.dbn2 = nn.GroupNorm(4, c_list[3])
            self.dbn3 = nn.GroupNorm(4, c_list[2])
            self.dbn4 = nn.GroupNorm(4, c_list[1])
            self.dbn5 = nn.GroupNorm(4, c_list[0])

            self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

            self.apply(self._init_weights)

        def _init_weights(self, m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

        def forward(self, x):

            out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
            t1 = out  # b, c0, H/2, W/2

            out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
            t2 = out  # b, c1, H/4, W/4

            out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
            t3 = out  # b, c2, H/8, W/8

            out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
            t4 = out  # b, c3, H/16, W/16

            out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
            t5 = out  # b, c4, H/32, W/32

            if self.bridge: t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)

            out = F.gelu(self.encoder6(out))  # b, c5, H/32, W/32

            out5 = F.gelu(self.dbn1(self.decoder1(out)))  # b, c4, H/32, W/32
            out5 = torch.add(out5, t5)  # b, c4, H/32, W/32

            out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',
                                        align_corners=True))  # b, c3, H/16, W/16
            out4 = torch.add(out4, t4)  # b, c3, H/16, W/16

            out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',
                                        align_corners=True))  # b, c2, H/8, W/8
            out3 = torch.add(out3, t3)  # b, c2, H/8, W/8

            out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                        align_corners=True))  # b, c1, H/4, W/4
            out2 = torch.add(out2, t2)  # b, c1, H/4, W/4

            out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',
                                        align_corners=True))  # b, c0, H/2, W/2
            out1 = torch.add(out1, t1)  # b, c0, H/2, W/2

            out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',
                                 align_corners=True)  # b, num_class, H, W

            return torch.sigmoid(out0)

        def load_from(self):
            if self.load_ckpt_path is not None:
                model_dict = self.vmunet.state_dict()
                modelCheckpoint = torch.load(self.load_ckpt_path)
                pretrained_dict = modelCheckpoint['model']
                # 过滤操作
                new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
                model_dict.update(new_dict)
                # 打印出来，更新了多少的参数
                print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict),
                                                                                           len(pretrained_dict),
                                                                                           len(new_dict)))
                self.vmunet.load_state_dict(model_dict)

                not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
                print('Not loaded keys:', not_loaded_keys)
                print("encoder loaded finished!")

""" best model UL+Shift+CNN [2,2,2,2,6,2] [2,2,2,2,2,2]"""
class atrous_ULPSR_basev3_CNN(nn.Module):


    def __init__(self, num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], d_conv=3,
                 split_att='fc', bridge=True, if_shifted_round=False, if_ss2d=True, forward_type='v1',
                  if_CNN=False, if_SE=False, if_SK=False,
                  encoder_atrous_step=[[2, 2], [2, 2], [2, 2], [2, 2, 2, 2, 2, 2], [2, 2]],
                  decoder_atrous_step=[[2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]):
        super().__init__()

        self.bridge = bridge

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
            nn.Conv2d(c_list[0], c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            # nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
            PVMLayer(input_dim=c_list[0], output_dim=c_list[1], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[0][0], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK, ),
            PVMLayer_shifted(input_dim=c_list[1], output_dim=c_list[1], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[0][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,)
        )
        self.encoder3 = nn.Sequential(
            # nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
            PVMLayer(input_dim=c_list[1], output_dim=c_list[2], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[1][0], d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK,),
            PVMLayer_shifted(input_dim=c_list[2], output_dim=c_list[2], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[1][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,)
        )
        self.encoder4 = nn.Sequential(
            PVMLayer(input_dim=c_list[2], output_dim=c_list[3], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[2][0], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
            PVMLayer_shifted(input_dim=c_list[3], output_dim=c_list[3], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[2][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
        )
        self.encoder5 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[4], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][0], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
            PVMLayer_shifted(input_dim=c_list[4], output_dim=c_list[4], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
            PVMLayer(input_dim=c_list[4], output_dim=c_list[4], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][2], d_conv=d_conv, if_CNN=if_CNN, if_SK=if_SK,),
            PVMLayer_shifted(input_dim=c_list[4], output_dim=c_list[4], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][3], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
            PVMLayer(input_dim=c_list[4], output_dim=c_list[4], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][4], d_conv=d_conv, if_CNN=if_CNN,  if_SK=if_SK,),
            PVMLayer_shifted(input_dim=c_list[4], output_dim=c_list[4], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][5], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
        )
        self.encoder6 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[5], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[4][0], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
            PVMLayer_shifted(input_dim=c_list[5], output_dim=c_list[5], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[4][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,)
        )

        if bridge:
            self.scab = SC_Att_Bridge(c_list, split_att)
            print('SC_Att_Bridge was used')

        self.decoder1 = nn.Sequential(
            PVMLayer(input_dim=c_list[5], output_dim=c_list[4], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[0][0], d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK,),
            PVMLayer_shifted(input_dim=c_list[4], output_dim=c_list[4], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[0][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,)
        )
        self.decoder2 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[3], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[1][0], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
            PVMLayer_shifted(input_dim=c_list[3], output_dim=c_list[3], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[1][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,)
        )
        self.decoder3 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[2], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[2][0], d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK,),
            PVMLayer_shifted(input_dim=c_list[2], output_dim=c_list[2], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[2][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,)

        )
        self.decoder4 = nn.Sequential(
            PVMLayer(input_dim=c_list[2], output_dim=c_list[1], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[3][0], d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK,),
            PVMLayer_shifted(input_dim=c_list[1], output_dim=c_list[1], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[3][1], d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK,)
        )
        self.decoder5 = nn.Sequential(
            PVMLayer(input_dim=c_list[1], output_dim=c_list[0], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[4][0], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
            PVMLayer_shifted(input_dim=c_list[0], output_dim=c_list[0], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[4][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,)
        )
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c1, H/4, W/4

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # b, c2, H/8, W/8

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # b, c3, H/16, W/16

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # b, c4, H/32, W/32

        if self.bridge: t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)

        out = F.gelu(self.encoder6(out))  # b, c5, H/32, W/32

        out5 = F.gelu(self.dbn1(self.decoder1(out)))  # b, c4, H/32, W/32
        out5 = torch.add(out5, t5)  # b, c4, H/32, W/32

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c3, H/16, W/16
        out4 = torch.add(out4, t4)  # b, c3, H/16, W/16

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c2, H/8, W/8
        out3 = torch.add(out3, t3)  # b, c2, H/8, W/8

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c1, H/4, W/4
        out2 = torch.add(out2, t2)  # b, c1, H/4, W/4

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c0, H/2, W/2
        out1 = torch.add(out1, t1)  # b, c0, H/2, W/2

        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)  # b, num_class, H, W

        return torch.sigmoid(out0)

""" UL+CNN without shift"""
class atrous_ULP_basev3_CNN(nn.Module):


    def __init__(self, num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], d_conv=3,
                 split_att='fc', bridge=True, if_ss2d=True, forward_type='v1',
                  if_CNN=True, if_SE=True, if_SK=True,
                  encoder_atrous_step=[[2, 2], [2, 2], [2, 2], [2, 2, 2, 2, 2, 2], [2, 2]],
                  decoder_atrous_step=[[2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]):
        super().__init__()

        self.bridge = bridge

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
            nn.Conv2d(c_list[0], c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            # nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
            PVMLayer(input_dim=c_list[0], output_dim=c_list[1], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[0][0], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK, ),
            PVMLayer(input_dim=c_list[1], output_dim=c_list[1],  if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[0][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,)
        )
        self.encoder3 = nn.Sequential(
            # nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
            PVMLayer(input_dim=c_list[1], output_dim=c_list[2], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[1][0], d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK,),
            PVMLayer(input_dim=c_list[2], output_dim=c_list[2],  if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[1][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,)
        )
        self.encoder4 = nn.Sequential(
            PVMLayer(input_dim=c_list[2], output_dim=c_list[3], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[2][0], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
            PVMLayer(input_dim=c_list[3], output_dim=c_list[3],  if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[2][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
        )
        self.encoder5 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[4], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][0], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
            PVMLayer(input_dim=c_list[4], output_dim=c_list[4],  if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
            PVMLayer(input_dim=c_list[4], output_dim=c_list[4], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][2], d_conv=d_conv, if_CNN=if_CNN, if_SK=if_SK,),
            PVMLayer(input_dim=c_list[4], output_dim=c_list[4], if_ss2d=if_ss2d, forward_type=forward_type,atrous_step=encoder_atrous_step[3][1], d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK, ),
            PVMLayer(input_dim=c_list[4], output_dim=c_list[4], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][4], d_conv=d_conv, if_CNN=if_CNN,  if_SK=if_SK,),
            PVMLayer(input_dim=c_list[4], output_dim=c_list[4], if_ss2d=if_ss2d, forward_type=forward_type,atrous_step=encoder_atrous_step[3][1], d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK, ),
        )
        self.encoder6 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[5], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[4][0], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
            PVMLayer(input_dim=c_list[5], output_dim=c_list[5],  if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[4][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,)
        )

        if bridge:
            self.scab = SC_Att_Bridge(c_list, split_att)
            print('SC_Att_Bridge was used')

        self.decoder1 = nn.Sequential(
            PVMLayer(input_dim=c_list[5], output_dim=c_list[4], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[0][0], d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK,),
            PVMLayer(input_dim=c_list[4], output_dim=c_list[4], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[0][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,)
        )
        self.decoder2 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[3], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[1][0], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
            PVMLayer(input_dim=c_list[3], output_dim=c_list[3], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[1][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,)
        )
        self.decoder3 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[2], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[2][0], d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK,),
            PVMLayer(input_dim=c_list[2], output_dim=c_list[2],  if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[2][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,)

        )
        self.decoder4 = nn.Sequential(
            PVMLayer(input_dim=c_list[2], output_dim=c_list[1], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[3][0], d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK,),
            PVMLayer(input_dim=c_list[1], output_dim=c_list[1], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[3][1], d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK,)
        )
        self.decoder5 = nn.Sequential(
            PVMLayer(input_dim=c_list[1], output_dim=c_list[0], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[4][0], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
            PVMLayer(input_dim=c_list[0], output_dim=c_list[0], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[4][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,)
        )
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c1, H/4, W/4

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # b, c2, H/8, W/8

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # b, c3, H/16, W/16

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # b, c4, H/32, W/32

        if self.bridge: t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)

        out = F.gelu(self.encoder6(out))  # b, c5, H/32, W/32

        out5 = F.gelu(self.dbn1(self.decoder1(out)))  # b, c4, H/32, W/32
        out5 = torch.add(out5, t5)  # b, c4, H/32, W/32

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c3, H/16, W/16
        out4 = torch.add(out4, t4)  # b, c3, H/16, W/16

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c2, H/8, W/8
        out3 = torch.add(out3, t3)  # b, c2, H/8, W/8

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c1, H/4, W/4
        out2 = torch.add(out2, t2)  # b, c1, H/4, W/4

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c0, H/2, W/2
        out1 = torch.add(out1, t1)  # b, c0, H/2, W/2

        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)  # b, num_class, H, W

        return torch.sigmoid(out0)


""" UL+Shift+CNN现在的层数为 [2,2,2,2,2,2] [2,2,2,2,2,2]"""
class atrous_ULPSR_basev3_CNN_stru2(nn.Module):


    def __init__(self, num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], d_conv=3,
                 split_att='fc', bridge=True, if_shifted_round=False, if_ss2d=True, forward_type='v1',
                  if_CNN=False, if_SE=False, if_SK=False,
                  encoder_atrous_step=[[2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
                  decoder_atrous_step=[[2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]):
        super().__init__()

        self.bridge = bridge

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
            nn.Conv2d(c_list[0], c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            # nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
            PVMLayer(input_dim=c_list[0], output_dim=c_list[1], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[0][0], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK, ),
            PVMLayer_shifted(input_dim=c_list[1], output_dim=c_list[1], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[0][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,)
        )
        self.encoder3 = nn.Sequential(
            # nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
            PVMLayer(input_dim=c_list[1], output_dim=c_list[2], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[1][0], d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK,),
            PVMLayer_shifted(input_dim=c_list[2], output_dim=c_list[2], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[1][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,)
        )
        self.encoder4 = nn.Sequential(
            PVMLayer(input_dim=c_list[2], output_dim=c_list[3], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[2][0], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
            PVMLayer_shifted(input_dim=c_list[3], output_dim=c_list[3], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[2][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
        )
        self.encoder5 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[4], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][0], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
            PVMLayer_shifted(input_dim=c_list[4], output_dim=c_list[4], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
            )
        self.encoder6 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[5], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[4][0], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
            PVMLayer_shifted(input_dim=c_list[5], output_dim=c_list[5], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[4][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,)
        )

        if bridge:
            self.scab = SC_Att_Bridge(c_list, split_att)
            print('SC_Att_Bridge was used')

        self.decoder1 = nn.Sequential(
            PVMLayer(input_dim=c_list[5], output_dim=c_list[4], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[0][0], d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK,),
            PVMLayer_shifted(input_dim=c_list[4], output_dim=c_list[4], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[0][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,)
        )
        self.decoder2 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[3], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[1][0], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
            PVMLayer_shifted(input_dim=c_list[3], output_dim=c_list[3], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[1][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,)
        )
        self.decoder3 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[2], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[2][0], d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK,),
            PVMLayer_shifted(input_dim=c_list[2], output_dim=c_list[2], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[2][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,)

        )
        self.decoder4 = nn.Sequential(
            PVMLayer(input_dim=c_list[2], output_dim=c_list[1], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[3][0], d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK,),
            PVMLayer_shifted(input_dim=c_list[1], output_dim=c_list[1], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[3][1], d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK,)
        )
        self.decoder5 = nn.Sequential(
            PVMLayer(input_dim=c_list[1], output_dim=c_list[0], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[4][0], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
            PVMLayer_shifted(input_dim=c_list[0], output_dim=c_list[0], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[4][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,)
        )
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c1, H/4, W/4

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # b, c2, H/8, W/8

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # b, c3, H/16, W/16

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # b, c4, H/32, W/32

        if self.bridge: t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)

        out = F.gelu(self.encoder6(out))  # b, c5, H/32, W/32

        out5 = F.gelu(self.dbn1(self.decoder1(out)))  # b, c4, H/32, W/32
        out5 = torch.add(out5, t5)  # b, c4, H/32, W/32

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c3, H/16, W/16
        out4 = torch.add(out4, t4)  # b, c3, H/16, W/16

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c2, H/8, W/8
        out3 = torch.add(out3, t3)  # b, c2, H/8, W/8

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c1, H/4, W/4
        out2 = torch.add(out2, t2)  # b, c1, H/4, W/4

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c0, H/2, W/2
        out1 = torch.add(out1, t1)  # b, c0, H/2, W/2

        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)  # b, num_class, H, W

        return torch.sigmoid(out0)

""" UL+Shift+CNN现在的层数为 [2,2,2,2,18,2] [2,2,2,2,2,2]"""
class atrous_ULPSR_basev3_CNN_stru18(nn.Module):


    def __init__(self, num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], d_conv=3,
                 split_att='fc', bridge=True, if_shifted_round=False, if_ss2d=True, forward_type='v1',
                  if_CNN=False, if_SE=False, if_SK=False,
                  encoder_atrous_step=[[2, 2], [2, 2], [2, 2], [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2], [2, 2]],
                  decoder_atrous_step=[[2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]):
        super().__init__()

        self.bridge = bridge

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
            nn.Conv2d(c_list[0], c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            # nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
            PVMLayer(input_dim=c_list[0], output_dim=c_list[1], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[0][0], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK, ),
            PVMLayer_shifted(input_dim=c_list[1], output_dim=c_list[1], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[0][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,)
        )
        self.encoder3 = nn.Sequential(
            # nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
            PVMLayer(input_dim=c_list[1], output_dim=c_list[2], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[1][0], d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK,),
            PVMLayer_shifted(input_dim=c_list[2], output_dim=c_list[2], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[1][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,)
        )
        self.encoder4 = nn.Sequential(
            PVMLayer(input_dim=c_list[2], output_dim=c_list[3], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[2][0], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
            PVMLayer_shifted(input_dim=c_list[3], output_dim=c_list[3], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[2][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
        )
        self.encoder5 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[4], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][0], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
            PVMLayer_shifted(input_dim=c_list[4], output_dim=c_list[4], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
            PVMLayer(input_dim=c_list[4], output_dim=c_list[4], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][2], d_conv=d_conv, if_CNN=if_CNN, if_SK=if_SK, ),
            PVMLayer_shifted(input_dim=c_list[4], output_dim=c_list[4], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][3],d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK, ),
            PVMLayer(input_dim=c_list[4], output_dim=c_list[4], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][4], d_conv=d_conv, if_CNN=if_CNN, if_SK=if_SK, ),
            PVMLayer_shifted(input_dim=c_list[4], output_dim=c_list[4], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][5], d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK, ),
            PVMLayer(input_dim=c_list[4], output_dim=c_list[4], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][6], d_conv=d_conv, if_CNN=if_CNN, if_SK=if_SK, ),
            PVMLayer_shifted(input_dim=c_list[4], output_dim=c_list[4], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][7], d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK, ),
            PVMLayer(input_dim=c_list[4], output_dim=c_list[4], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][8], d_conv=d_conv, if_CNN=if_CNN, if_SK=if_SK, ),
            PVMLayer_shifted(input_dim=c_list[4], output_dim=c_list[4], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][9], d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK, ),
            PVMLayer(input_dim=c_list[4], output_dim=c_list[4], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][10], d_conv=d_conv, if_CNN=if_CNN, if_SK=if_SK, ),
            PVMLayer_shifted(input_dim=c_list[4], output_dim=c_list[4], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][11], d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK, ),
            PVMLayer(input_dim=c_list[4], output_dim=c_list[4], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][12], d_conv=d_conv, if_CNN=if_CNN, if_SK=if_SK, ),
            PVMLayer_shifted(input_dim=c_list[4], output_dim=c_list[4], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][13], d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK, ),
            PVMLayer(input_dim=c_list[4], output_dim=c_list[4], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][14], d_conv=d_conv, if_CNN=if_CNN, if_SK=if_SK, ),
            PVMLayer_shifted(input_dim=c_list[4], output_dim=c_list[4], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][15], d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK, ),
            PVMLayer(input_dim=c_list[4], output_dim=c_list[4], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][16], d_conv=d_conv, if_CNN=if_CNN, if_SK=if_SK, ),
            PVMLayer_shifted(input_dim=c_list[4], output_dim=c_list[4], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[3][17], d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK, ),
        )
        self.encoder6 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[5], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[4][0], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
            PVMLayer_shifted(input_dim=c_list[5], output_dim=c_list[5], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=encoder_atrous_step[4][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,)
        )

        if bridge:
            self.scab = SC_Att_Bridge(c_list, split_att)
            print('SC_Att_Bridge was used')

        self.decoder1 = nn.Sequential(
            PVMLayer(input_dim=c_list[5], output_dim=c_list[4], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[0][0], d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK,),
            PVMLayer_shifted(input_dim=c_list[4], output_dim=c_list[4], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[0][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,)
        )
        self.decoder2 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[3], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[1][0], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
            PVMLayer_shifted(input_dim=c_list[3], output_dim=c_list[3], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[1][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,)
        )
        self.decoder3 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[2], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[2][0], d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK,),
            PVMLayer_shifted(input_dim=c_list[2], output_dim=c_list[2], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[2][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,)

        )
        self.decoder4 = nn.Sequential(
            PVMLayer(input_dim=c_list[2], output_dim=c_list[1], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[3][0], d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK,),
            PVMLayer_shifted(input_dim=c_list[1], output_dim=c_list[1], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[3][1], d_conv=d_conv, if_CNN=if_CNN, if_SE=if_SE, if_SK=if_SK,)
        )
        self.decoder5 = nn.Sequential(
            PVMLayer(input_dim=c_list[1], output_dim=c_list[0], if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[4][0], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,),
            PVMLayer_shifted(input_dim=c_list[0], output_dim=c_list[0], if_shifted_round=if_shifted_round, if_ss2d=if_ss2d, forward_type=forward_type, atrous_step=decoder_atrous_step[4][1], d_conv=d_conv, if_CNN=if_CNN,  if_SE=if_SE, if_SK=if_SK,)
        )
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c1, H/4, W/4

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # b, c2, H/8, W/8

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # b, c3, H/16, W/16

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # b, c4, H/32, W/32

        if self.bridge: t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)

        out = F.gelu(self.encoder6(out))  # b, c5, H/32, W/32

        out5 = F.gelu(self.dbn1(self.decoder1(out)))  # b, c4, H/32, W/32
        out5 = torch.add(out5, t5)  # b, c4, H/32, W/32

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c3, H/16, W/16
        out4 = torch.add(out4, t4)  # b, c3, H/16, W/16

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c2, H/8, W/8
        out3 = torch.add(out3, t3)  # b, c2, H/8, W/8

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c1, H/4, W/4
        out2 = torch.add(out2, t2)  # b, c1, H/4, W/4

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c0, H/2, W/2
        out1 = torch.add(out1, t1)  # b, c0, H/2, W/2

        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)  # b, num_class, H, W

        return torch.sigmoid(out0)





