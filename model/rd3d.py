import torch
import torch.nn as nn
import torch.nn.functional as F

from model.resnet3d import I3DResNet


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x


class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv3d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding,
                      dilation=dilation, stride=stride),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x


class DeconvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, bias=True):
        super(DeconvBlock, self).__init__()
        self.deconv_bn = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, padding=[0, 1, 1], output_padding=[0, 1, 1],
                               dilation=dilation, stride=stride),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.deconv_bn(x)
        return x


class Reduction3D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Reduction3D, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv3d(in_channel, out_channel, kernel_size=[1, 1, 1]),
            BasicConv3d(out_channel, out_channel, kernel_size=[3, 3, 3], padding=1),
            BasicConv3d(out_channel, out_channel, kernel_size=[3, 3, 3], padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ThreeDDecoder(nn.Module):
    def __init__(self, channel):
        super(ThreeDDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv3d(channel, channel, [3, 3, 3], padding=1)
        self.conv_upsample2 = BasicConv3d(channel, channel, [3, 3, 3], padding=1)
        self.conv_upsample3 = BasicConv3d(channel, channel, [3, 3, 3], padding=1)
        self.conv_upsample4 = BasicConv3d(channel, channel, [3, 3, 3], padding=1)

        self.conv_downsample2_1 = BasicConv3d(channel, channel, [1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])
        self.conv_downsample1_1 = BasicConv3d(channel, channel, [1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])
        self.conv_downsample1_2 = BasicConv3d(channel, channel, [1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])
        self.conv_downsample0_1 = BasicConv3d(channel, channel, [1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])
        self.conv_downsample0_2 = BasicConv3d(channel, channel, [1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])
        self.conv_downsample0_3 = BasicConv3d(channel, channel, [1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])

        self.conv_cat1 = nn.Sequential(
            BasicConv3d(channel, channel, [3, 3, 3], padding=1),
            BasicConv3d(channel, channel, [1, 1, 1])
        )
        self.conv_cat2 = nn.Sequential(
            BasicConv3d(channel, channel, [3, 3, 3], padding=1),
            BasicConv3d(channel, channel, [1, 1, 1])
        )
        self.conv_cat3 = nn.Sequential(
            BasicConv3d(channel, channel, [3, 3, 3], padding=1),
            BasicConv3d(channel, channel, [1, 1, 1])
        )
        self.conv_cat4 = nn.Sequential(
            BasicConv3d(channel, channel, [3, 3, 3], padding=1),
            BasicConv3d(channel, channel, [1, 1, 1])
        )
        self.downt1 = BasicConv3d(channel, channel, [10, 1, 1])
        self.downt2 = BasicConv3d(channel, channel, [7, 1, 1])
        self.downt3 = BasicConv3d(channel, channel, [5, 1, 1])
        self.downt4 = BasicConv3d(channel, channel, [3, 1, 1])

        self.out = nn.Sequential(
            BasicConv2d(channel, 2 * channel, [1, 1]),
            BasicConv2d(2 * channel, channel, [1, 1]),
            nn.Conv2d(channel, 1, [1, 1])
        )
        self.se3 = SELayer(channel * 10)
        self.se2 = SELayer(channel * 7)
        self.se1 = SELayer(channel * 5)
        self.se0 = SELayer(channel * 3)

    def forward(self, x0, x1, x2, x3, x4):
        x2_downconv1 = self.conv_downsample2_1(x2)  # 16
        x1_downconv1 = self.conv_downsample1_1(x1)  # 32
        x1_downconv2 = self.conv_downsample1_2(x1_downconv1)  # 16
        x0_downconv1 = self.conv_downsample0_1(x0)  # 64n
        x0_downconv2 = self.conv_downsample0_2(x0_downconv1)  # 32
        x0_downconv3 = self.conv_downsample0_3(x0_downconv2)  # 16

        # 3,4
        B, C4, T4, H4, W4 = x4.size()
        x4_flatten = x4.view(B, C4 * T4, H4, W4)
        x4_flatten_up = self.upsample(x4_flatten)
        x4_up = x4_flatten_up.view(B, C4, T4, 2 * H4, 2 * W4)
        x4_upconv = self.conv_upsample4(x4_up)
        residual3 = torch.cat((x4_upconv, x3, x2_downconv1, x1_downconv2, x0_downconv3), 2)
        x3_ = self.conv_cat4(residual3)
        x3_flatten = x3_.view(x3_.shape[0], x3_.shape[1] * x3_.shape[2], x3_.shape[3], x3_.shape[4])
        x3_flatten = self.se3(x3_flatten)
        x3_ = x3_flatten.view(x3_.shape[0], x3_.shape[1], x3_.shape[2], x3_.shape[3], x3_.shape[4])
        x3 = residual3 + x3_
        x3 = self.downt1(x3)

        # 2,3
        B, C3, T3, H3, W3 = x3.size()
        x3_flatten = x3.view(B, C3 * T3, H3, W3)
        x3_flatten_up = self.upsample(x3_flatten)
        x3_up = x3_flatten_up.view(B, C3, T3, 2 * H3, 2 * W3)
        x3_upconv = self.conv_upsample3(x3_up)
        residual2 = torch.cat((x3_upconv, x2, x1_downconv1, x0_downconv2), 2)
        x2_ = self.conv_cat3(residual2)
        x2_flatten = x2_.view(x2_.shape[0], x2_.shape[1] * x2_.shape[2], x2_.shape[3], x2_.shape[4])
        x2_flatten = self.se2(x2_flatten)
        x2_ = x2_flatten.view(x2_.shape[0], x2_.shape[1], x2_.shape[2], x2_.shape[3], x2_.shape[4])
        x2 = residual2 + x2_
        x2 = self.downt2(x2)

        # 1,2
        B, C2, T2, H2, W2 = x2.size()
        x2_flatten = x2.view(B, C2 * T2, H2, W2)
        x2_flatten_up = self.upsample(x2_flatten)
        x2_up = x2_flatten_up.view(B, C2, T2, 2 * H2, 2 * W2)
        x2_upconv = self.conv_upsample2(x2_up)
        residual1 = torch.cat((x2_upconv, x1, x0_downconv1), 2)
        x1_ = self.conv_cat2(residual1)  # 64
        x1_flatten = x1_.view(x1_.shape[0], x1_.shape[1] * x1_.shape[2], x1_.shape[3], x1_.shape[4])
        x1_flatten = self.se1(x1_flatten)
        x1_ = x1_flatten.view(x1_.shape[0], x1_.shape[1], x1_.shape[2], x1_.shape[3], x1_.shape[4])
        x1 = residual1 + x1_
        x1 = self.downt3(x1)

        # 0,1
        B, C1, T1, H1, W1 = x1.size()
        x1_flatten = x1.view(B, C1 * T1, H1, W1)
        x1_flatten_up = self.upsample(x1_flatten)
        x1_up = x1_flatten_up.view(B, C1, T1, 2 * H1, 2 * W1)
        x1_upconv = self.conv_upsample1(x1_up)
        residual0 = torch.cat((x0, x1_upconv), 2)
        x0_ = self.conv_cat1(residual0)
        x0_flatten = x0_.view(x0_.shape[0], x0_.shape[1] * x0_.shape[2], x0_.shape[3], x0_.shape[4])
        x0_flatten = self.se0(x0_flatten)
        x0_ = x0_flatten.view(x0_.shape[0], x0_.shape[1], x0_.shape[2], x0_.shape[3], x0_.shape[4])
        x0 = residual0 + x0_
        x0 = self.downt4(x0)
        x = x0.squeeze(2)

        out = self.out(x)

        return out


class DownBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DownBlock, self).__init__()
        self.down_conv1 = BasicConv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.down_conv2 = DeconvBlock(in_channels, out_channels, kernel_size, stride, padding)
        self.down_conv3 = BasicConv3d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        # ipdb.set_trace()
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class Unet3D(nn.Module):
    def __init__(self, channel):
        super(Unet3D, self).__init__()

        self.reductions0 = Reduction3D(64, channel)
        self.reductions1 = Reduction3D(256, channel)

        self.reductione0 = Reduction3D(64, channel)
        self.reductione1 = Reduction3D(256, channel)

        self.reductions2 = Reduction3D(512, channel)
        self.reductions3 = Reduction3D(1024, channel)
        self.reductions4 = Reduction3D(2048, channel)

        self.reductione2 = Reduction3D(512, channel)
        self.reductione3 = Reduction3D(1024, channel)
        self.reductione4 = Reduction3D(2048, channel)

        self.output_s = ThreeDDecoder(channel)

    def forward(self, x0, x1, x2, x3, x4):
        x_s0 = self.reductions0(x0)
        x_s1 = self.reductions1(x1)
        x_s2 = self.reductions2(x2)
        x_s3 = self.reductions3(x3)
        x_s4 = self.reductions4(x4)

        pred_s = self.output_s(x_s0, x_s1, x_s2, x_s3, x_s4)

        return pred_s


class RD3D(nn.Module):
    def __init__(self, channel, resnet):
        super(RD3D, self).__init__()

        self.resnet = I3DResNet(resnet)
        self.unet = Unet3D(channel)

    def forward(self, x):
        size = x.size()[3:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x0 = self.resnet.relu(x)
        x = self.resnet.maxpool(x0)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
        pred_s = self.unet(x0, x1, x2, x3, x4)
        pred_s = F.upsample(pred_s, size=size, mode='bilinear', align_corners=True)

        return pred_s
