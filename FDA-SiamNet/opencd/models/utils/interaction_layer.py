# Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
from mmengine.model import BaseModule
import torch.nn.functional as F
from opencd.models.utils.builder import ITERACTION_LAYERS

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


@ITERACTION_LAYERS.register_module()
class SDAM(BaseModule):
    def __init__(self, in_channels):
        super(SDAM, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        # self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        cosSim = self.cos(x1, x2).unsqueeze(1)
        M = 1 - cosSim
        M1 = 1 - cosSim
        M1[(M1 > 0.5) & (M1 < 2)] = 1
        x1 = x1 * M
        x2 = x2 * M1
        x1_mean = torch.mean(x1, dim=1, keepdim=True)
        x2_mean = torch.mean(x2, dim=1, keepdim=True)

        x1_sim = x1_mean * M
        x2_sim = x2_mean * M1
        x1_sim_c = torch.cat([x1_mean, x1_sim], dim=1)
        x2_sim_c = torch.cat([x2_mean, x2_sim], dim=1)

        x1_enhance = x1 * (self.sigmoid(self.conv2d(x1_sim_c))) + x1
        x2_enhance = x2 * (self.sigmoid(self.conv2d(x2_sim_c))) + x2

        x1_enhance_max, _ = torch.max(x1_enhance, dim=1, keepdim=True)
        x2_enhance_max, _ = torch.max(x2_enhance, dim=1, keepdim=True)
        cosSim1 = self.cos(x1_enhance, x2_enhance).unsqueeze(1)
        m = 1 - cosSim1
        m1 = 1 - cosSim1
        m1[(m1 > 0.5) & (m1 < 2)] = 1

        x1_sim1 = x1_enhance_max * m
        x2_sim1 = x2_enhance_max * m1
        x1_sim1_c = torch.cat([x1_enhance_max, x1_sim1], dim=1)
        x2_sim1_c = torch.cat([x2_enhance_max, x2_sim1], dim=1)

        x1_en = x1 * (self.sigmoid(self.conv2d(x1_sim1_c)))
        x2_en = x2 * (self.sigmoid(self.conv2d(x2_sim1_c)))

        # x1_out = x1 + x1_enhance + x1_en
        # x2_out = x2 + x2_enhance + x2_en
        x1_out = x1 + x1_en
        x2_out = x2 + x2_en

        return x1_out, x2_out



@ITERACTION_LAYERS.register_module()
class CDAM(BaseModule):
    def __init__(self, channel):
        super(CDAM, self).__init__()
        self.dim = channel
        self.avg_pool_H = nn.AdaptiveAvgPool2d((1, None))
        self.avg_pool_W = nn.AdaptiveAvgPool2d((None, 1))
        self.max_pool_H = nn.AdaptiveMaxPool2d((1, None))
        self.max_pool_W = nn.AdaptiveMaxPool2d((None, 1))
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1_avg_h = self.avg_pool_H(x1)
        x2_avg_h = self.avg_pool_H(x2)
        cos_1 = self.cos(x1_avg_h, x2_avg_h).unsqueeze(1)
        Matrix_1 = 1 - cos_1

        x1_avg_w = self.avg_pool_W(x1)
        x2_avg_w = self.avg_pool_W(x2)
        cos_2 = self.cos(x1_avg_w, x2_avg_w).unsqueeze(1)
        Matrix_2 = 1 - cos_2

        x1_en1 = Matrix_1 * x1 * Matrix_2 + x1
        x2_en1 = x2

        x1_max_h = self.max_pool_H(x1_en1)
        x2_max_h = self.max_pool_H(x2_en1)
        cos_3 = self.cos(x1_max_h, x2_max_h).unsqueeze(1)
        Matrix_3 = 1 - cos_3

        x1_max_w = self.max_pool_W(x1_en1)
        x2_max_w = self.max_pool_W(x2_en1)
        cos_4 = self.cos(x1_max_w, x2_max_w).unsqueeze(1)
        Matrix_4 = 1 - cos_4

        x1_en2 = Matrix_3 * x1_en1 * Matrix_4 + x1_en1
        x2_en2 = x2_en1

        out1 = x1_en2
        out2 = x2_en2

        return out1, out2



@ITERACTION_LAYERS.register_module()
class CEFI(BaseModule):
    def __init__(self, channel, reduction=2):
        super().__init__()
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.Mish(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.depth_conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                    groups=channel)
        self.point_conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0,
                                    groups=1)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x1, x2):
        b, c, _, _ = x1.size()
        y1_avg = self.avg_pool(x1).view(b, c)
        y1_max = self.max_pool(x1).view(b, c)
        y1 = y1_avg + y1_max
        y1 = self.fc(y1).view(b, c, 1, 1)

        gsconv_x1 = self.depth_conv(x1)
        gsconv_out1 = self.point_conv(gsconv_x1)
        gsconv_out1 = gsconv_out1 + x1

        gsconv_x2 = self.depth_conv(x2)
        gsconv_out2 = self.point_conv(gsconv_x2)
        gsconv_out2 = gsconv_out2 + x2
        out11 = x1 * y1.expand_as(x1)
        out12 = gsconv_out2 * y1.expand_as(x1)
        out_last1 = out11 + out12 + gsconv_out1

        y2_avg = self.avg_pool(x2).view(b, c)
        y2_max = self.max_pool(x2).view(b, c)
        y2 = y2_avg + y2_max
        y2 = self.fc(y2).view(b, c, 1, 1)
        out21 = x2 * y2.expand_as(x2)
        out22 = gsconv_out1 * y2.expand_as(x2)
        out_last2 = out21 + out22 + gsconv_out2

        return out_last1, out_last2

@ITERACTION_LAYERS.register_module()
class SEFI(BaseModule):
    def __init__(self,in_channels):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        # self.act=SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # map尺寸不变，缩减通道
        avgout1 = torch.mean(x1, dim=1, keepdim=True)
        avgout2 = torch.mean(x2, dim=1, keepdim=True)
        maxout1, _ = torch.max(x1, dim=1, keepdim=True)
        maxout2, _ = torch.max(x2, dim=1, keepdim=True)
        out1 = torch.cat([avgout1, maxout2], dim=1)
        out2 = torch.cat([avgout2, maxout1], dim=1)
        out1 = self.sigmoid(self.conv2d(out1))
        out2 = self.sigmoid(self.conv2d(out2))
        out_1 = out1 * x1
        out_2 = out2 * x2

        return out_1, out_2
#########################################################################################################################
@ITERACTION_LAYERS.register_module()
class ChannelExchange(BaseModule):
    """
    channel exchange
    Args:
        p (float, optional): p of the features will be exchanged.
            Defaults to 1/2.
    """

    def __init__(self, p=1 / 2):
        super().__init__()
        assert p >= 0 and p <= 1
        self.p = int(1 / p)

    def forward(self, x1, x2):
        N, c, h, w = x1.shape

        exchange_map = torch.arange(c) % self.p == 0
        exchange_mask = exchange_map.unsqueeze(0).expand((N, -1))

        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]

        return out_x1, out_x2


@ITERACTION_LAYERS.register_module()
class SpatialExchange(BaseModule):
    """
    spatial exchange
    Args:
        p (float, optional): p of the features will be exchanged.
            Defaults to 1/2.
    """

    def __init__(self, p=1 / 2):
        super().__init__()
        assert p >= 0 and p <= 1
        self.p = int(1 / p)

    def forward(self, x1, x2):
        N, c, h, w = x1.shape
        exchange_mask = torch.arange(w) % self.p == 0

        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[..., ~exchange_mask] = x1[..., ~exchange_mask]
        out_x2[..., ~exchange_mask] = x2[..., ~exchange_mask]
        out_x1[..., exchange_mask] = x2[..., exchange_mask]
        out_x2[..., exchange_mask] = x1[..., exchange_mask]

        return out_x1, out_x2
#########################################################################################################################
@ITERACTION_LAYERS.register_module()
class Aggregation_distribution(BaseModule):
    # Aggregation_Distribution Layer (AD)
    def __init__(self, 
                 channels, 
                 num_paths=2, 
                 attn_channels=None, 
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.num_paths = num_paths # `2` is supported.
        attn_channels = attn_channels or channels // 16
        attn_channels = max(attn_channels, 8)
        
        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.bn = build_norm_layer(norm_cfg, attn_channels)[1]
        self.act = build_activation_layer(act_cfg)
        self.fc_select = nn.Conv2d(attn_channels, channels * num_paths, kernel_size=1, bias=False)

    def forward(self, x1, x2):
        x = torch.stack([x1, x2], dim=1)
        attn = x.sum(1).mean((2, 3), keepdim=True)
        attn = self.fc_reduce(attn)
        attn = self.bn(attn)
        attn = self.act(attn)
        attn = self.fc_select(attn)
        B, C, H, W = attn.shape
        attn1, attn2 = attn.reshape(B, self.num_paths, C // self.num_paths, H, W).transpose(0, 1)
        attn1 = torch.sigmoid(attn1)
        attn2 = torch.sigmoid(attn2)
        return x1 * attn1, x2 * attn2


@ITERACTION_LAYERS.register_module()
class TwoIdentity(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x1, x2):
        return x1, x2
