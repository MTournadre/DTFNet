import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from ..nn_utils import default_init
from ..nn_utils import NormFn, DepthwiseSeparableConv2d

class SomeResidualBlock(nn.Module):
    def __init__(self,
            input_channels: int,
            output_channels: int,
            norm_fn: NormFn = NormFn.GROUP,
            stride: int = 1):
        """
        Inspired by RAFT (Teed & Deng 2020)
        https://github.com/princeton-vl/RAFT.git
        """
        super(SomeResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = norm_fn.get_module(output_channels)
        self.norm2 = norm_fn.get_module(output_channels)
        if not stride == 1 or input_channels != output_channels:
            self.norm3 = norm_fn.get_module(output_channels)

        if stride == 1 and input_channels == output_channels:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

class DebugEncoder(nn.Module):
    """
    Simple encoder for debugging purposes.
    Can be initialized from resnet50
    """
    def __init__( self,
            input_channels: int,
            output_channels: int,
            use_resnet: bool = False,
            norm_fn = NormFn.BATCH ):
        super(DebugEncoder,self).__init__()

        conv1_dim = 32
        if use_resnet:
            from torchvision.models import resnet50
            resnet = resnet50("IMAGENET1K_V1")
            resnet_modules = list(resnet.children())
            self.conv1 = nn.Sequential( *resnet_modules[:3] )
            for p in self.conv1.parameters():
                p.requires_grad_(False)
            conv1_dim = 64
        else:
            self.conv1 = nn.Sequential(
                    nn.Conv2d( input_channels, conv1_dim, kernel_size=7, padding=3, stride=2 ),
                    norm_fn.get_module(conv1_dim),
                    nn.ReLU(inplace=True) )
        self.conv2 = SomeResidualBlock( conv1_dim, 64, norm_fn=norm_fn, stride=2 )
        self.conv3 = SomeResidualBlock( 64, output_channels, norm_fn=norm_fn, stride=2 )
        self.scale = 8

    def forward( self, x ):
        B, T, C, H, W = x.shape

        x = 2./255.*x - 1.

        x = rearrange( x, "b t c h w -> (b t) c h w" )
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = rearrange( x, "(b t) c h w -> b t c h w", b=B, t=T )

        return x


class SmallConvEncoder(nn.Module):
    def __init__( self,
            input_channels: int = 3,
            output_channels: int = 128 ):
        super(SmallConvEncoder,self).__init__()

        self.conv0 = nn.Sequential(
                nn.Conv2d(
                        3, 32,
                        kernel_size=(7,7),
                        stride=(2,2),
                        padding=(3,3) ),
                nn.InstanceNorm2d(32),
                nn.LeakyReLU(inplace=True),
                )
        self.conv1 = nn.Sequential(
                SomeResidualBlock( 32, 48,
                    stride = 2,
                    norm_fn = NormFn.INSTANCE ),
                SomeResidualBlock( 48, 48,
                    norm_fn = NormFn.INSTANCE ) )
        self.conv2 = nn.Sequential(
                SomeResidualBlock( 48, output_channels,
                    stride = 2,
                    norm_fn = NormFn.INSTANCE ),
                SomeResidualBlock( output_channels, output_channels,
                    norm_fn = NormFn.INSTANCE ) )

        self.scale = 8

        default_init(self)

    def forward( self, x ):
        if x.dim() == 5:
            temporal = True
            B, T, C, H, W = x.shape
            x = rearrange( x, "b t c h w -> (b t) c h w" )
        else:
            temporal = False
            B, C, H, W = x.shape

        x = 2./255.*x - 1.

        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        if temporal:
            x = rearrange( x, "(b t) c h w -> b t c h w", b=B, t=T )

        return x


class ConvEncoder(nn.Module):
    def __init__( self,
            input_channels: int = 3,
            output_channels: int = 128 ):
        super(ConvEncoder,self).__init__()

        self.conv0 = nn.Sequential(
                nn.Conv2d(
                        3, 64,
                        kernel_size=(7,7),
                        stride=(2,2),
                        padding=(3,3) ),
                nn.InstanceNorm2d(64),
                #nn.LeakyReLU(inplace=True),
                nn.ReLU(inplace=True),
                )
        self.conv1 = nn.Sequential(
                SomeResidualBlock( 64, 64,
                    stride = 1,
                    norm_fn = NormFn.INSTANCE ),
                SomeResidualBlock( 64, 64,
                    norm_fn = NormFn.INSTANCE ) )
        self.conv2 = nn.Sequential(
                SomeResidualBlock( 64, 96,
                    stride = 2,
                    norm_fn = NormFn.INSTANCE ),
                SomeResidualBlock( 96, 96,
                    norm_fn = NormFn.INSTANCE ) )
        self.conv3 = nn.Sequential(
                SomeResidualBlock( 96, output_channels,
                    stride = 2,
                    norm_fn = NormFn.INSTANCE ),
                SomeResidualBlock( output_channels, output_channels,
                    norm_fn = NormFn.INSTANCE ) )

        self.scale = 8

        default_init(self)

    def forward( self, x ):
        if x.dim() == 5:
            temporal = True
            B, T, C, H, W = x.shape
            x = rearrange( x, "b t c h w -> (b t) c h w" )
        else:
            temporal = False
            B, C, H, W = x.shape

        x = 2./255.*x - 1.

        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if temporal:
            x = rearrange( x, "(b t) c h w -> b t c h w", b=B, t=T )

        return x
