import torch
from torch.nn import Parameter
import torch.nn as nn


def inflate_conv(conv2d,
                 time_dim=3,
                 time_padding=0,
                 time_stride=1,
                 time_dilation=1,
                 center=False):
    # To preserve activations, padding should be by continuity and not zero
    # or no padding in time dimension
    kernel_dim = (time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1])
    padding = (time_padding, conv2d.padding[0], conv2d.padding[1])
    stride = (time_stride, conv2d.stride[0], conv2d.stride[0])
    dilation = (time_dilation, conv2d.dilation[0], conv2d.dilation[1])
    conv3d = torch.nn.Conv3d(
        conv2d.in_channels,
        conv2d.out_channels,
        kernel_dim,
        padding=padding,
        dilation=dilation,
        stride=stride)
    # Repeat filter time_dim times along time dimension
    weight_2d = conv2d.weight.data
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim

    # Assign new params
    conv3d.weight = Parameter(weight_3d)
    conv3d.bias = conv2d.bias
    return conv3d

class inflate_conv_light(nn.Module):
    def __init__(self, conv2d, time_dim=3, time_padding=0, time_stride=1, time_dilation=1,center=False):
        super(inflate_conv_light,self).__init__()
        self.time_dim=time_dim
        kernel_dim = (time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1])
        padding = (time_padding, conv2d.padding[0], conv2d.padding[1])
        stride = (time_stride, conv2d.stride[0], conv2d.stride[0])
        dilation = (time_dilation, conv2d.dilation[0], conv2d.dilation[1])
        if time_dim == 1:
            self.conv3d = torch.nn.Conv3d(
                conv2d.in_channels,
                conv2d.out_channels,
                kernel_dim,
                padding=padding,
                dilation=dilation,
                stride=stride)
        # Repeat filter time_dim times along time dimension
            weight_2d = conv2d.weight.data
            weight_3d = weight_2d.unsqueeze(2)
            # Assign new params
            self.conv3d.weight = Parameter(weight_3d)
            self.conv3d.bias = conv2d.bias      
        else:
            self.conv3d_spatial = torch.nn.Conv3d(
                conv2d.in_channels,
                conv2d.out_channels,
                kernel_size=(1, kernel_dim[1], kernel_dim[2]),
                padding=(0, padding[1], padding[2]),
                dilation=(1, dilation[1], dilation[2]),
                stride=(1, stride[1], stride[2])
            )
            weight_2d = conv2d.weight.data
            self.conv3d_spatial.weight = Parameter(weight_2d.unsqueeze(2))
            self.conv3d_spatial.bias = conv2d.bias
            self.conv3d_time_1 = nn.Conv3d(conv2d.out_channels,conv2d.out_channels,[1,1,1],bias=False)
            self.conv3d_time_2 = nn.Conv3d(conv2d.out_channels,conv2d.out_channels,[1,1,1],bias=False)
            self.conv3d_time_3 = nn.Conv3d(conv2d.out_channels,conv2d.out_channels,[1,1,1],bias=False)
            torch.nn.init.constant_(self.conv3d_time_1.weight,0)
            torch.nn.init.constant_(self.conv3d_time_3.weight,0)
            torch.nn.init.eye_(self.conv3d_time_2.weight[:,:,0,0,0])

    def forward(self,x):
        #N,C,T,H,W
        if self.time_dim==1:
            return self.conv3d(x)
        else:
            x_spatial=self.conv3d_spatial(x)
            rgb = x_spatial[:,:,0:1,:,:]
            depth = x_spatial[:,:,1:2,:,:]
            f_rgb = self.conv3d_time_2(rgb)
            f_depth = self.conv3d_time_2(depth)
            f2_rgb = self.conv3d_time_1(rgb)
            f2_depth = self.conv3d_time_3(depth)
            x = torch.cat([f_rgb+f2_depth,f2_rgb+f_depth],dim=2)
            return x

def inflate_linear(linear2d, time_dim):
    """
    Args:
        time_dim: final time dimension of the features
    """
    linear3d = torch.nn.Linear(linear2d.in_features * time_dim,
                               linear2d.out_features)
    weight3d = linear2d.weight.data.repeat(1, time_dim)
    weight3d = weight3d / time_dim

    linear3d.weight = Parameter(weight3d)
    linear3d.bias = linear2d.bias
    return linear3d


def inflate_batch_norm(batch2d):
    # In pytorch 0.2.0 the 2d and 3d versions of batch norm
    # work identically except for the check that verifies the
    # input dimensions

    batch3d = torch.nn.BatchNorm3d(batch2d.num_features)
    # retrieve 3d _check_input_dim function
    batch2d._check_input_dim = batch3d._check_input_dim
    return batch2d


def inflate_pool(pool2d,
                 time_dim=1,
                 time_padding=0,
                 time_stride=None,
                 time_dilation=1):
    if isinstance(pool2d, torch.nn.AdaptiveAvgPool2d):
        pool3d = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
    else:
        kernel_dim = (time_dim, pool2d.kernel_size, pool2d.kernel_size)
        padding = (time_padding, pool2d.padding, pool2d.padding)
        if time_stride is None:
            time_stride = time_dim
        stride = (time_stride, pool2d.stride, pool2d.stride)
        if isinstance(pool2d, torch.nn.MaxPool2d):
            dilation = (time_dilation, pool2d.dilation, pool2d.dilation)
            pool3d = torch.nn.MaxPool3d(
                kernel_dim,
                padding=padding,
                dilation=dilation,
                stride=stride,
                ceil_mode=pool2d.ceil_mode)
        elif isinstance(pool2d, torch.nn.AvgPool2d):
            pool3d = torch.nn.AvgPool3d(kernel_dim, stride=stride)
        else:
            raise ValueError('{} is not among known pooling classes'.format(type(pool2d)))

    return pool3d
