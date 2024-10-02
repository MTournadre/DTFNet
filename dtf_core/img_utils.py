import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import math
import numpy as np

def bilinear_sampling( img: torch.Tensor, coords: torch.Tensor ):
    """
    Bilinear sampling with 'border' extrapolation, based on
    torch.nn.functional.grid_sample.
    Coordinates are expressed in pixels (not normalized)

    img:    [ B, C, H, W ]
    coords: [ B, N, 2 ] or [ B, H_out, W_out, 2 ]
    out:    [ B, C, N ] or [ B, C, H_out, W_out ]
    """
    assert img.dim() == 4
    if coords.dim() == 3:
        do_squeeze = True
        coords = coords.unsqueeze(1)
    else:
        assert coords.dim() == 4
        do_squeeze = False

    B, C, H, W = img.shape
    device = img.device

    # Convert to normalized coordinates
    scale = torch.tensor([2./W,2./H],device=device)
    coords = coords*scale[None,None,None,:] - 1.

    # Sample
    sampled = F.grid_sample(
            img, coords,
            mode='bilinear',
            padding_mode='border',
            align_corners=True )
    if do_squeeze:
        sampled = sampled.squeeze(2)
    return sampled

def upsampling(img, factor, mode='bilinear'):
    B, C, H, W = img.shape
    new_size = factor*H, factor*W
    return F.interpolate(img, size=new_size, mode=mode, align_corners=True)

def mask_upsampling( img, mask, border_padding: bool = True ):
    """
    Upsample the input image according to a linear upsampling mask.
    The mask uses linear combinations on patches of size [KH,KW],
    upsampling the image by a factor [FH,FW],
    giving the resulting image of size [ B, C, FH*H, FW*W ]

    img:  [ B, C, H, W ]
    mask: [ B, KH, KW, FH, FW, H, W ]
    """
    B, C, H, W = img.shape
    _, KH, KW, FH, FW, _, _ = mask.shape

    if border_padding:
        img = torch.nn.functional.pad( img,
                ( KW//2, KW//2, KH//2, KH//2 ),
                mode='replicate' )
        patches = F.unfold( img, kernel_size=(KH,KW), padding=(0,0) )
    else:
        patches = F.unfold( img, kernel_size=(KH,KW), padding=(KH//2,KW//2) )
    patches = rearrange( patches, "B (C KH KW) (H W) -> B C KH KW H W",
            C=C, KH=KH, KW=KW, H=H, W=W )

    upsampled = torch.einsum( "B C k K H W,  B k K f F H W -> B C f F H W",
            patches, mask )
    upsampled = rearrange( upsampled, "B C FH FW H W -> B C (H FH) (W FW)" )

    return upsampled

def coords_grid( B, H, W, dtype=torch.float32, device="cuda" ):
    """
    Generate a tensor whose shape will be [ B, 2, H, W ],
    containing the x/y position of each pixel
    """
    grid_y, grid_x = torch.meshgrid(
            torch.arange(0.5,H,dtype=dtype,device=device),
            torch.arange(0.5,W,dtype=dtype,device=device),
            indexing="ij" )
    coords = torch.stack((grid_x,grid_y),dim=0).unsqueeze(0)
    return coords.repeat(B,1,1,1)

def generate_test_wheel( size, amp=1.,
        dtype=torch.float32,
        device=torch.device("cuda") ):
    """
    Generate a squared image with a test optical flow in all directions
    """
    center = 0.5*size
    grid_y, grid_x = torch.meshgrid(
            torch.arange(0.5,size,dtype=dtype,device=device),
            torch.arange(0.5,size,dtype=dtype,device=device),
            indexing="ij" )
    dist_center_x = grid_x-center
    dist_center_y = grid_y-center
    dist_center = torch.sqrt(dist_center_x**2 + dist_center_y**2)
    flow = torch.where( dist_center <= 0.5*size,
            torch.stack((dist_center_x,dist_center_y)),
            torch.zeros((2,size,size),dtype=dtype,device=device) )
    flow *= amp*2./size
    return flow

def hsv_to_rgb( hsv ):
    """
    Convert a color array from HSL to RGB.
    Input should have the shape [...,3] and in the range [0,1] for all 3 channels
    """
    h, s, v = hsv[...,0], hsv[...,1], hsv[...,2]
    c = v*s
    h_ = h*6.
    x = c * ( 1 - torch.abs((h_%2) - 1) )

    cad0 = h_ < 1 #(h_ >= 0).logical_and(h_ < 1)
    cad1 = (h_ >= 1).logical_and(h_ < 2)
    cad2 = (h_ >= 2).logical_and(h_ < 3)
    cad3 = (h_ >= 3).logical_and(h_ < 4)
    cad4 = (h_ >= 4).logical_and(h_ < 5)
    cad5 = (h_ >= 5)#.logical_and(h_ < 6)

    rgb = torch.zeros_like(hsv)
    rgb[cad0,0] = c[cad0] ; rgb[cad0,1] = x[cad0]
    rgb[cad1,0] = x[cad1] ; rgb[cad1,1] = c[cad1]
    rgb[cad2,1] = c[cad2] ; rgb[cad2,2] = x[cad2]
    rgb[cad3,1] = x[cad3] ; rgb[cad3,2] = c[cad3]
    rgb[cad4,2] = c[cad4] ; rgb[cad4,0] = x[cad4]
    rgb[cad5,2] = x[cad5] ; rgb[cad5,0] = c[cad5]

    rgb += (v-c)[...,None]

    return rgb

def flow_img( flow, max_norm = None, wheel_size = None ):
    """
    Convert an optical flow into an RGB image.
    Give wheel_size a value to show the legend wheel
    """
    norm = flow.norm(p=2,dim=0)
    if max_norm is None:
        max_norm = norm.max()

    if wheel_size is not None and wheel_size > 0:
        flow = flow.clone()
        wheel = generate_test_wheel( wheel_size, amp=max_norm,
                dtype=flow.dtype, device=flow.device )
        flow[:,:wheel_size,:wheel_size] = wheel
        norm = flow.norm(p=2,dim=0)

    # Convert flow to HSL:
    #  - hue:   angle
    #  - sat:   norm
    #  - light: constant (0.5)
    hue = torch.atan2( flow[1], flow[0] )
    sat = (norm/max_norm).clamp(0.,1.)
    light = torch.tensor(0.5)

    # Convert HSL to RGB
    c = ( 1 - torch.abs(2*light-1) )*sat
    hue_ = hue / (math.pi/3)
    x = c * ( 1 - torch.abs((hue_%2)-1) )

    cad0 = (hue_ >= 0).logical_and(hue_ < 1)
    cad1 = (hue_ >= 1).logical_and(hue_ < 2)
    cad2 = (hue_ >= 2).logical_and(hue_ < 3)
    cad3 = (hue_ >= -3).logical_and(hue_ < -2)
    cad4 = (hue_ >= -2).logical_and(hue_ < -1)
    cad5 = (hue_ >= -1).logical_and(hue_ < 0)

    res = torch.zeros((3,flow.shape[1],flow.shape[2]),
            dtype=flow.dtype,
            device=flow.device)
    res[0][cad0] = c[cad0] ; res[1][cad0] = x[cad0]
    res[0][cad1] = x[cad1] ; res[1][cad1] = c[cad1]
    res[1][cad2] = c[cad2] ; res[2][cad2] = x[cad2]
    res[1][cad3] = x[cad3] ; res[2][cad3] = c[cad3]
    res[2][cad4] = c[cad4] ; res[0][cad4] = x[cad4]
    res[2][cad5] = x[cad5] ; res[0][cad5] = c[cad5]

    res += light - 0.5*c

    return res

def np_img( torch_img: torch.Tensor ):
    """
    Return the image into a np.array of shape [ H, W, C ] on CPU from the
    torch tensor of shape [ C, H, W ]
    """
    if torch_img.ndim == 2:
        torch_img = torch_img.unsqueeze(0)
    if torch_img.dtype is torch.float32:
        # Image can be in [0,1] or in [0,255].
        # This tests whether we are in the first case
        if torch_img.max() > 1.5:
            torch_img = torch_img/255
        torch_img = torch_img.to(torch.float64)

    return torch_img.moveaxis(0,2).detach().cpu().numpy()
