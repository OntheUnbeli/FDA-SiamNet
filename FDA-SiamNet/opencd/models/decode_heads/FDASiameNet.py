# Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn
from torch.nn import init
from mmcv.cnn import Conv2d, ConvModule, build_activation_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmengine.model import BaseModule, Sequential
from torch.nn import functional as F

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from opencd.registry import MODELS
from ..necks.feature_fusion import FeatureFusionNeck

class ECAAttention(nn.Module):
    def __init__(self, channel, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(channel, 1, kernel_size=1, padding=0, bias=False)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.gap(x)  # bs,c,1,1
        y = y.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        y = self.conv(y)  # bs,1,c
        y = self.sigmoid(y)  # bs,1,c
        y = y.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
        z = x * y.expand_as(x)
        z = self.conv1(z)
        return z


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        # self.relu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class EGFM(BaseModule):
    """Flow Dual-Alignment Fusion Module.
    Args:
        in_channels (int): Input channels of features.
        conv_cfg (dict | None): Config of conv layers.
            Default: None
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN')
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
    """

    def __init__(self,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='IN'),
                 act_cfg=dict(type='GELU')):
        super(EGFM, self).__init__()
        self.in_channels = in_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        # TODO
        conv_cfg = None
        norm_cfg = dict(type='IN')
        act_cfg = dict(type='GELU')

        self.dim = in_channels
        self.conv31 = BasicConv2d(self.dim, self.dim, kernel_size=(7, 1), stride=1, padding=(3, 0), groups=1)
        self.conv13 = BasicConv2d(self.dim, self.dim, kernel_size=(1, 7), stride=1, padding=(0, 3), groups=1)
        self.conv3 = BasicConv2d(self.dim, self.dim, 7, padding=3, dilation=1, groups=1)
        #
        # self.conv31 = BasicConv2d(self.dim, self.dim, kernel_size=(3, 1), stride=1, padding=(1, 0), groups=1)
        # self.conv13 = BasicConv2d(self.dim, self.dim, kernel_size=(1, 3), stride=1, padding=(0, 1), groups=1)
        # self.conv3 = BasicConv2d(self.dim, self.dim, 3, padding=1, groups=1)

        self.sig = nn.Sigmoid()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.conv11 = BasicConv2d(self.dim*2, self.dim//4, kernel_size=3, dilation=3, padding=3, groups=self.dim//4)
        self.conv21 = BasicConv2d(self.dim*2, self.dim//4, kernel_size=3, dilation=2, padding=2, groups=self.dim//4)
        self.conv51 = BasicConv2d(self.dim*2, self.dim//4, kernel_size=3, dilation=4, padding=4, groups=self.dim//4)
        self.conv1 = BasicConv2d(self.dim*2, self.dim//4, kernel_size=1, groups=self.dim//4)

        self.ECA = ECAAttention(channel=self.dim, kernel_size=5)

    def forward(self, x1, x2):
        """Forward function."""
        cosSim = self.cos(x1, x2).unsqueeze(1)
        M = 1 - cosSim
        M1 = 1 - cosSim
        M1[(M1 > 0.5) & (M1 < 2)] = 1
        x1 = x1 * M
        x2 = x2 * M1

        output = torch.cat([x1, x2], dim=1)
        output11 = self.conv11(output)
        output21 = self.conv21(output)
        output51 = self.conv51(output)
        output1 = self.conv1(output)
        output_l = torch.cat([output11, output21, output51, output1], dim=1)
        Att = self.ECA(output_l)


        high_31 = self.conv31(x1)
        high_13 = self.conv13(x1)
        high_3 = self.conv3(x1)
        x1_pro = high_3 + high_13 + high_31

        low_31 = self.conv31(x2)
        low_13 = self.conv13(x2)
        low_3 = self.conv3(x2)
        x2_pro = low_3 + low_13 + low_31

        mask1 = Att*x1_pro
        mask2 = Att*x2_pro

        x1_feat = x2_pro - mask1
        x2_feat = x1_pro - mask2

        y = torch.cat((x1_feat, x2_feat), dim=1)

        return y


class MixFFN(BaseModule):
    """An implementation of MixFFN of Segformer. \
        Here MixFFN is uesd as projection head of Changer.
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 init_cfg=None):
        super(MixFFN, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 3x3 depth wise conv to provide positional encode information
        pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, identity=None):
        out = self.layers(x)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


@MODELS.register_module()
class FDASiamNet(BaseDecodeHead):
    """The Head of Changer.
    This head is the implementation of
    `Changer <https://arxiv.org/abs/2209.08290>` _.
    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels // 2,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        self.neck_layer = EGFM(in_channels=self.channels//2)

        # projection head
        self.discriminator = MixFFN(
            embed_dims=self.channels,
            feedforward_channels=self.channels,
            ffn_drop=0.,
            dropout_layer=dict(type='DropPath', drop_prob=0.),
            act_cfg=dict(type='GELU'))

    def base_forward(self, inputs):
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        return out


    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        inputs1 = []
        inputs2 = []
        for input in inputs:
            f1, f2 = torch.chunk(input, 2, dim=1)
            inputs1.append(f1)
            inputs2.append(f2)

        out1 = self.base_forward(inputs1)
        out2 = self.base_forward(inputs2)
        out = self.neck_layer(out1, out2)
        # out = torch.cat([out1, out2], dim=1)

        out = self.discriminator(out)
        out = self.cls_seg(out)

        return out