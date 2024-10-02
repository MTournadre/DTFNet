from enum import Enum, auto

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, List
import math

from .img_utils import bilinear_sampling

class NormFn(Enum):
    NONE = auto()
    GROUP = auto()
    BATCH = auto()
    INSTANCE = auto()

    def get_module( self, num_channels ):
        if self == NormFn.GROUP:
            assert num_channels%8 == 0
            num_groups = num_channels // 8
            return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
        elif self == NormFn.BATCH:
            return nn.BatchNorm2d(num_channels)
        elif self == NormFn.INSTANCE:
            return nn.InstanceNorm2d(num_channels)
        elif self == NormFn.NONE:
            return nn.Sequential()
        else:
            raise ValueError( f"Invalid norm function: {self!r}" )

    def get_module_1d( self, num_channels ):
        if self == NormFn.GROUP:
            assert num_channels%8 == 0
            num_groups = num_channels // 8
            return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
        elif self == NormFn.BATCH:
            return nn.BatchNorm1d(num_channels)
        elif self == NormFn.INSTANCE:
            return nn.InstanceNorm1d(num_channels)
        elif self == NormFn.NONE:
            return nn.Sequential()
        else:
            raise ValueError( f"Invalid norm function: {self!r}" )


def default_init( module ):
    # From our experiments, the default init does not work well,
    # all gradients can vanish to zero.
    # Best strategies we found are:
    # - xavier_normal, with normal bias
    # - kaiming_uniform/normal, fan_in, with normal bias
    for m in module.modules():
        if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear) or isinstance(m,nn.Conv1d):
            #nn.init.kaiming_normal_( m.weight, mode='fan_in', nonlinearity='relu' )
            nn.init.xavier_normal_( m.weight )
            if hasattr(m,"bias") and m.bias is not None:
                std = m.bias.dim()/math.sqrt(sum(m.bias.shape))
                #nn.init.uniform_( m.bias, -std, std )
                nn.init.normal_( m.bias, std=std )
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d,
                nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm1d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if hasattr(m,"bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)

class Scale(nn.Module):
    def __init__( self, scale ):
        super(Scale,self).__init__()
        self.register_buffer("scale", torch.as_tensor(scale) )

    def forward( self, x ):
        return self.scale*x

class DepthwiseSeparableConv2d(nn.Module):
    """
    Return a separable 2D convolution, applying at first spatial-only filters,
    and then channels-only (=1x1) convolutions

    Spatial dimensions can also be separable, resulting in one filter along x,
    one filter along y, and one filter along channel dims
    """
    def __init__( self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            spatial_separation: bool = False ):
        super(DepthwiseSeparableConv2d, self).__init__()

        self.spatial_separation = spatial_separation
        if self.spatial_separation:
            self.depth_conv_x = nn.Conv2d( in_channels, in_channels,
                    kernel_size=(1,kernel_size),
                    padding=(0,kernel_size//2),
                    stride=(1,stride),
                    groups=in_channels )
            self.depth_conv_y = nn.Conv2d( in_channels, in_channels,
                    kernel_size=(kernel_size,1),
                    padding=(kernel_size//2,0),
                    stride=(stride,1),
                    groups=in_channels )
        else:
            self.depth_conv = nn.Conv2d( in_channels, in_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size//2,
                    stride=stride,
                    groups=in_channels )

        self.point_conv = nn.Conv2d( in_channels, out_channels,
                kernel_size=1 )

    def forward( self, x: torch.Tensor ):
        if self.spatial_separation:
            x = self.depth_conv_y(self.depth_conv_x(x))
        else:
            x = self.depth_conv(x)
        x = self.point_conv(x)

        return x

class FixedPositionEmbedding(nn.Module):
    def __init__( self, input_dim: int, nb_per_input: int,
            min_freq: Optional[List[float]] = None,
            max_freq: Optional[List[float]] = None ):
        super(FixedPositionEmbedding,self).__init__()

        self.input_dim = input_dim
        self.output_dim = 2*input_dim*nb_per_input
        if min_freq is None:
            min_freq = input_dim*[1e-2]
        else:
            assert len(min_freq) == input_dim
        if max_freq is None:
            max_freq = input_dim*[1e0]
        else:
            assert len(max_freq) == input_dim
        self.register_buffer( "freq", torch.stack([
            torch.exp( torch.linspace(math.log(a),math.log(b),nb_per_input) )
            for a, b in zip(min_freq,max_freq) ],
            dim=0 ) )

    def forward( self, x ):
        x = 2*math.pi*x.unsqueeze(-1)*self.freq
        x = torch.stack( (torch.sin(x), torch.cos(x)), dim=-1 )
        x = rearrange( x, "... x f s -> ... (x f s)" )
        return x

class LearnedPositionEmbedding(nn.Module):
    def __init__( self,
            input_dim: int,
            nb_per_input: int,
            scale: Optional[List[float]] = None ):
        super(LearnedPositionEmbedding,self).__init__()

        self.input_dim = input_dim
        self.output_dim = 2*input_dim*nb_per_input

        # The frequency is on a log scale
        if scale is None:
            scale = nb_per_input*[1.]
        self.freq = torch.nn.Parameter(
                torch.stack([
                    s*torch.randn((nb_per_input))
                    for s in scale],
                    dim=0) )

    def forward( self, x ):
        x = 2*math.pi*x.unsqueeze(-1)*torch.exp(self.freq)
        x = torch.stack( (torch.sin(x),torch.cos(x)), dim=-1 )
        x = rearrange( x, "... x f s -> ... (x f s)" )
        return x

class Residual(nn.Module):
    def __init__( self, module ):
        super(Residual,self).__init__()
        self.module = module
        #self.skip_weight = nn.Parameter( torch.tensor(1.) )
        self.skip_weight = 1.

    def forward( self, *args, **kwargs ):
        x = self.module(*args,**kwargs)
        if type(x) is tuple:
            return tuple( [self.skip_weight*args[0] + x[0]] + x[1:] )
        else:
            return self.skip_weight*args[0] + x

