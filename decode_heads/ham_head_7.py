from collections import OrderedDict

import ipdb
import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from timm.models.layers import DropPath
from torch.nn import Softmax

from mmseg.ops import resize

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from functools import  partial
from inplace_abn import InPlaceABN, InPlaceABNSync
nonlinearity = partial(F.relu, inplace = True)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, multiple):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=multiple, stride=multiple, padding=0, output_padding=0)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
class _MatrixDecomposition2DBase(nn.Module):
    def __init__(self, args=dict()):
        super().__init__()

        self.spatial = args.setdefault('SPATIAL', True)

        self.S = args.setdefault('MD_S', 1)
        self.D = args.setdefault('MD_D', 512)
        self.R = args.setdefault('MD_R', 64)

        self.train_steps = args.setdefault('TRAIN_STEPS', 6)
        self.eval_steps = args.setdefault('EVAL_STEPS', 7)

        self.inv_t = args.setdefault('INV_T', 100)
        self.eta = args.setdefault('ETA', 0.9)

        self.rand_init = args.setdefault('RAND_INIT', True)

        print('spatial', self.spatial)
        print('S', self.S)
        print('D', self.D)
        print('R', self.R)
        print('train_steps', self.train_steps)
        print('eval_steps', self.eval_steps)
        print('inv_t', self.inv_t)
        print('eta', self.eta)
        print('rand_init', self.rand_init)

    def _build_bases(self, B, S, D, R, cuda=False):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    # @torch.no_grad()
    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        if self.spatial:
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)

        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R, cuda=True)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, cuda=True)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        if self.spatial:
            x = x.view(B, C, H, W)
        else:
            x = x.transpose(1, 2).view(B, C, H, W)

        # (B * H, D, R) -> (B, H, N, D)
        bases = bases.view(B, self.S, D, self.R)

        return x


class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self, args=dict()):
        super().__init__(args)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R, cuda=False):
        if cuda:
            bases = torch.rand((B * S, D, R)).cuda()
        else:
            bases = torch.rand((B * S, D, R))

        bases = F.normalize(bases, dim=1)

        return bases

    # @torch.no_grad()
    def local_step(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # print('max_out:',max_out.shape)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # print('avg_out:',avg_out.shape)
        a=torch.cat([max_out, avg_out], dim=1)
        # print('a:',a.shape)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        # print('spatial:',spatial_out.shape)
        x = spatial_out * x
        # print('x:',x.shape)
        return x


class Dense_CBAM_ASPP(nn.Module):                       ##加入通道注意力机制
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(Dense_CBAM_ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in * 2, dim_out * 2, 3, 1, padding=4 * rate, dilation=4 * rate, bias=True),
            nn.BatchNorm2d(dim_out * 2, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in * 4, dim_out * 4, 3, 1, padding=8 * rate, dilation=8 * rate, bias=True),
            nn.BatchNorm2d(dim_out * 4, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in * 8, dim_out * 8, 3, 1, padding=16 * rate, dilation=16 * rate, bias=True),
            nn.BatchNorm2d(dim_out * 8, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 16, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # print('dim_in:',dim_in)
        # print('dim_out:',dim_out)
        self.cbam=CBAMLayer(channel=dim_out*16)

    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x) #  32 32 512
        x1 = torch.cat([x, conv1x1], dim=1) # 32 32 512 + 512
        conv3x3_1 = self.branch2(x1) # 32 32 1024
        x2 = torch.cat([x, conv1x1, conv3x3_1], dim=1) # 32 32 512+512+1024=2048
        conv3x3_2 = self.branch3(x2) # 32 32 2048
        x3 = torch.cat([x, conv1x1, conv3x3_1, conv3x3_2], dim=1) # 32 32 512+512+1024+2048
        conv3x3_3 = self.branch4(x3) # 32 32 4096
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        # print('feature:',feature_cat.shape)
        # 加入cbam注意力机制
        cbamaspp=self.cbam(feature_cat)
        result1=self.conv_cat(cbamaspp)
        return result1




class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU, norm_cfg=None):
        super().__init__()
        if norm_cfg:
            self.norm1 = build_norm_layer(norm_cfg, dim)[1]
            self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        else:
            self.norm1 = nn.BatchNorm2d(dim)
            self.norm2 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        # ++++++++++
        # self.downsample = nn.Sequential(nn.Conv2d(dim,dim,4,1,2),
        #                                 nn.SyncBatchNorm(dim),
        #                                 nn.Conv2d(dim,dim,1,1,0)
        #                                 )
    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))

        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x



class LSKmodule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.convl = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv0_s = nn.Conv2d(dim, dim // 2, 1)
        self.conv1_s = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv_m = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.convl(attn1)

        attn1 = self.conv0_s(attn1)
        attn2 = self.conv1_s(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv_m(attn)
        return x * attn

class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSKmodule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

@HEADS.register_module()
class LightHamHead7(BaseDecodeHead):
    """Is Attention Better Than Matrix Decomposition?
    This head is the implementation of `HamNet
    <https://arxiv.org/abs/2109.04553>`_.
    Args:
        ham_channels (int): input channels for Hamburger.
        ham_kwargs (int): kwagrs for Ham.

    TODO:
        Add other MD models (Ham).
    """

    def __init__(self,
                 ham_channels,
                 ham_kwargs=dict(),
                 **kwargs):
        super(LightHamHead7, self).__init__(
            input_transform='multiple_select', **kwargs)

# DDL-------------------
        self.firstconv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                                   bias=False)
        self.firstbn = nn.BatchNorm2d(64)
        self.firstrelu = nn.ReLU(inplace=True)

        # pooling
        self.pool21 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool22 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool23 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool24 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.pool41 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        self.pool42 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        self.pool43 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)

        self.pool81 = nn.MaxPool2d(kernel_size=8, stride=8, padding=0)
        self.pool82 = nn.MaxPool2d(kernel_size=8, stride=8, padding=0)

        self.pool161 = nn.MaxPool2d(kernel_size=16, stride=16, padding=0)

        self.conv64_641 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv64_642 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv64_643 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv64_644 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv64_645 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv64_646 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv64_647 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv128_1281 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv128_1282 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv320_3201 = nn.Conv2d(320, 320, kernel_size=3, stride=1, padding=1)

        self.bn641 = nn.BatchNorm2d(64)
        self.bn642 = nn.BatchNorm2d(64)
        self.bn643 = nn.BatchNorm2d(64)
        self.bn644 = nn.BatchNorm2d(64)
        self.bn645 = nn.BatchNorm2d(64)
        self.bn646 = nn.BatchNorm2d(64)
        self.bn647 = nn.BatchNorm2d(64)
        self.bn1281 = nn.BatchNorm2d(128)
        self.bn1282 = nn.BatchNorm2d(128)
        self.bn1283 = nn.BatchNorm2d(128)

        self.bn2561 = nn.BatchNorm2d(320)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.relu5 = nn.ReLU(inplace=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.relu8 = nn.ReLU(inplace=True)
        self.relu9 = nn.ReLU(inplace=True)
        self.relu10 = nn.ReLU(inplace=True)
        self.relu11 = nn.ReLU(inplace=True)

        self.decoder1088_256_2 = DecoderBlock(1088, 256,2)
        self.decoder1088_128_4 = DecoderBlock(1088,128,4)
        self.decoder1088_64_8 = DecoderBlock(1088, 64,8)
        self.decoder1088_64_16 = DecoderBlock(1088, 64, 16)

        self.decoder832_128_1 = DecoderBlock(832,128,2)
        self.decoder832_64_1 = DecoderBlock(832,64,4)
        self.decoder832_64_2 = DecoderBlock(832,64,8)

        self.decoder512_64_1 = DecoderBlock(512,64,2)
        self.decoder512_64_2 = DecoderBlock(512,64,4)

        self.decoder320_64_1 = DecoderBlock(320,64,2)


#最后的一系列调整通道 -------------------
        # self.score_dsn64 = nn.Conv2d(64,32,kernel_size=1,stride=1,padding=0)
        self.score_dsn1088_64 = nn.Conv2d(1088, 64, kernel_size=1, stride=1, padding=0)
        self.score_dsn832_64 = nn.Conv2d(832, 64, kernel_size=1, stride=1, padding=0)
        self.score_dsn512_64 = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0)
        self.score_dsn320_64 = nn.Conv2d(320, 64, kernel_size=1, stride=1, padding=0)

        self.unsample1_1 = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.unsample1_2 = nn.ConvTranspose2d(128,64,2,2,0)
        self.unsample1_3 = nn.ConvTranspose2d(320, 64, 2, 2, 0)
        self.unsample1_4 = nn.ConvTranspose2d(512, 64, 2, 2, 0)

        self.unsample64_64_1 = nn.ConvTranspose2d(64, 64, 1, 1, 0)
        self.unsample64_64_2 = nn.ConvTranspose2d(64,64,2,2,0)
        self.unsample64_64_4 = nn.ConvTranspose2d(64, 64, 4, 4, 0)
        self.unsample64_64_8 = nn.ConvTranspose2d(64, 64, 8, 8, 0)
        self.unsample64_64_16 = nn.ConvTranspose2d(64, 64, 16, 16, 0)

        self.finaldeconv1 = nn.Conv2d(64, 32, 1, padding=0)
        self.finalrelu1_1 = nonlinearity
        self.finalconv1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu1_2 = nonlinearity
        self.finalconv1_3 = nn.Conv2d(32, 1, 3, padding=1)  #num_classes代表分类数，即通道数

        self.aspp_512 = Dense_CBAM_ASPP(512,512)
        # self.lsk = Attention(64)
        self.lsk = Block(64)
    def forward(self, inputs):          #最小尺寸为20 并且加入了有CBAM注意力机制的ASPP通道注意力机制        加入lsk可变感受野注意力
        """Forward function."""
        # inputs = self._transform_inputs(inputs)
        # inputs = [resize(
        #     level,
        #     size=inputs[0].shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners
        # ) for level in inputs]
        # inputs = torch.cat(inputs, dim=1)
        # x = self.squeeze(inputs)
        # x = self.hamburger(x)
        # output = self.align(x)
        # output = self.cls_seg(output)


# -------------------------320 160 80 40
        x = inputs[0]       #torch.Size([1, 3, 320, 320])
        e1 = inputs[1]      #torch.Size([1, 64, 160, 160])
        e2 = inputs[2]      #torch.Size([1, 128, 80, 80])
        e3 = inputs[3]      #torch.Size([1, 320, 40, 40])
        e4 = inputs[4]      #torch.Size([1, 512, 20, 20])

        x0 = self.firstconv(x)  # 320 320 64
        x0 = self.firstbn(x0)
        x0 = self.firstrelu(x0)  # 320 320 64

        d5 = self.aspp_512(e4)
        # x0 = self.lsk(x0)


        x_1 = self.relu1(self.bn641(self.conv64_641(self.pool21(x0))))  # 160 160 64
        x_2 = self.relu2(self.bn642(self.conv64_642(self.pool41(x0))))  # 80 80 64
        x_3 = self.relu3(self.bn643(self.conv64_643(self.pool81(x0))))  # 40 40 64
        x_4 = self.relu4(self.bn644(self.conv64_644(self.pool161(x0)))) # 20 20 64

        e1_1 = self.relu5(self.bn645(self.conv64_645(self.pool22(e1))))  # 80 80 64
        e1_2 = self.relu6(self.bn646(self.conv64_646(self.pool42(e1))))  # 40 40 64
        e1_3 = self.relu7(self.bn647(self.conv64_647(self.pool82(e1))))  # 20 20 64


        e2_1 = self.relu8(self.bn1281(self.conv128_1281(self.pool23(e2))))  # 40 40 128
        e2_2 = self.relu9(self.bn1282(self.conv128_1282(self.pool43(e2))))  # 20 20 128

        e3_1 = self.relu10(self.bn2561(self.conv320_3201(self.pool24(e3))))  # 20 20 320

        d5_5 = torch.cat([d5, x_4, e1_3, e2_2, e3_1], dim=1)  # 20*20   512+64+64+128+320=1088
        d5_4 = self.decoder1088_256_2(d5_5)         #40 40 256
        d5_3 = self.decoder1088_128_4(d5_5)          #80 80 128
        d5_2 = self.decoder1088_64_8(d5_5)          #160 160 64
        d5_1 = self.decoder1088_64_16(d5_5)          #320 320 64

        d4_4 = torch.cat([e3, x_3, e1_2, e2_1,d5_4], dim=1)  # 40*40    320+64+64+128+256=832
        d4_3 = self.decoder832_128_1(d4_4)      #80 80 128
        d4_2 = self.decoder832_64_1(d4_4)       #160 160 64
        d4_1 = self.decoder832_64_2(d4_4)       #320 320 64

        d3_3 = torch.cat([e2, x_2,e1_1,d4_3,d5_3], dim=1)  # 80*80  128+64+64+128+128=448+64=512
        d3_2 = self.decoder512_64_1(d3_3)       #160 160 64
        d3_1 = self.decoder512_64_2(d3_3)       # 320 320 64

        d2_2 = torch.cat([e1,x_1,d5_2,d4_2,d3_2],dim=1)    #160*160     64+64+64+64+64=320
        d2_1 = self.decoder320_64_1(d2_2)   #320 320 64

        d1_1 = torch.cat([x0,d5_1,d4_1,d3_1,d2_1],dim=1)     #320*320   64+64+64+64+64=320

        # 调整通道
        d5_out = self.score_dsn1088_64(d5_5)        #20 20 64
        d4_out = self.score_dsn832_64(d4_4)        #40 40 64
        d3_out = self.score_dsn512_64(d3_3)        #80 80 64
        d2_out = self.score_dsn320_64(d2_2)        #160 160 64
        d1_out =  self.score_dsn320_64(d1_1)        #320 320 64


        # 上采样调整尺寸
        d5_out = self.unsample64_64_16(d5_out)        #torch.Size([4, 64, 320, 320])
        d4_out = self.unsample64_64_8(d4_out)       #torch.Size([4, 64, 320, 320])
        d3_out = self.unsample64_64_4(d3_out)       #torch.Size([4, 64, 320, 320])
        d2_out = self.unsample64_64_2(d2_out)       #torch.Size([4, 64, 320, 320])
        d1_out = self.unsample64_64_1(d1_out)       #torch.Size([4, 64, 320, 320])


        d1_out = self.finaldeconv1(d1_out)  #32*320*320
        d1_out = self.finalrelu1_1(d1_out)  #32*320*320
        d1_out = self.finalconv1_2(d1_out)  #32*320*320
        d1_out = self.finalrelu1_2(d1_out)  #32*320*320
        d1_out = self.finalconv1_3(d1_out)  #1*320*320

        d2_out = self.finaldeconv1(d2_out)  #32*320*320
        d2_out = self.finalrelu1_1(d2_out)  #32*320*320
        d2_out = self.finalconv1_2(d2_out)  #32*320*320
        d2_out = self.finalrelu1_2(d2_out)  #32*320*320
        d2_out = self.finalconv1_3(d2_out)  #1*320*320

        d3_out = self.finaldeconv1(d3_out)  #32*320*320
        d3_out = self.finalrelu1_1(d3_out)  #32*320*320
        d3_out = self.finalconv1_2(d3_out)  #32*320*320
        d3_out = self.finalrelu1_2(d3_out)  #32*320*320
        d3_out = self.finalconv1_3(d3_out)  #1*320*320

        d4_out = self.finaldeconv1(d4_out)  #32*320*320
        d4_out = self.finalrelu1_1(d4_out)  #32*320*320
        d4_out = self.finalconv1_2(d4_out)  #32*320*320
        d4_out = self.finalrelu1_2(d4_out)  #32*320*320
        d4_out = self.finalconv1_3(d4_out)  #1*320*320

        d5_out = self.finaldeconv1(d5_out)  # 32*320*320
        d5_out = self.finalrelu1_1(d5_out)  # 32*320*320
        d5_out = self.finalconv1_2(d5_out)  # 32*320*320
        d5_out = self.finalrelu1_2(d5_out)  # 32*320*320
        d5_out = self.finalconv1_3(d5_out)  # 1*320*320

        # # +++++++++++++=
        x0 = self.finaldeconv1(x0)  # 32*320*320
        x0 = self.finalrelu1_1(x0)  # 32*320*320
        x0 = self.finalconv1_2(x0)  # 32*320*320
        x0 = self.finalrelu1_2(x0)  # 32*320*320
        x0 = self.finalconv1_3(x0)  # 1*320*320

        fuse = (d1_out + d2_out + d3_out + d4_out + d5_out + x0) / 6.0
        #
        # fuse = (d1_out + d2_out + d3_out +d4_out+d5_out) / 5.0

        d1_out = torch.sigmoid(d1_out)
        d2_out = torch.sigmoid(d2_out)
        d3_out = torch.sigmoid(d3_out)
        d4_out = torch.sigmoid(d4_out)
        d5_out = torch.sigmoid(d5_out)
        fuse = torch.sigmoid(fuse)
    #+++++++++++
        x0 = torch.sigmoid(x0)

        # out = (d1_out + d2_out + d3_out + d4_out + d5_out + fuse ) / 6.0


        out = [d1_out,d2_out,d3_out,d4_out,d5_out,x0,fuse]


        return out
