import torch
import torch.nn as nn
from einops import rearrange, repeat
from typing import Optional, List, Union
import math

from ..nn_utils import default_init, NormFn, FixedPositionEmbedding, Scale, Residual
from ..attention import PEStrategy, CentroidSummarization, MotionNetwork
from .. import img_utils
from .encoder import ConvEncoder, SmallConvEncoder

class ConstantTokens(nn.Module):
    def __init__( self,
            nb_tokens: int,
            token_dim: int,
            num_heads: int = 2,
            ):
        super(ConstantTokens,self).__init__()

        self.tokens = nn.Parameter(
                torch.randn((nb_tokens,token_dim)) )

    def forward( self, tokens, qk_softmax, qk_softmin ):
        B, Z, C = tokens.shape
        return repeat( self.tokens, "z c -> b z c", b=B )

class SimpleTemporalFFN(nn.Module):
    def __init__( self,
            nb_channels: int,
            inner_dim: int,
            groups: int = 1,
            ):
        super(SimpleTemporalFFN,self).__init__()
        self.nb_channels = nb_channels
        self.inner_dim = inner_dim

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(nb_channels,inner_dim,
                kernel_size=3,
                padding=1,
                groups=groups )
        self.norm1 = nn.BatchNorm2d(inner_dim)
        self.conv_t = nn.Conv1d(inner_dim,inner_dim,
                kernel_size=3,
                padding=1,
                groups=groups )
        self.norm_t = nn.BatchNorm1d(inner_dim)
        self.conv2 = nn.Conv2d(inner_dim,nb_channels,
                kernel_size=3,
                padding=1,
                groups=groups )
        self.norm2 = nn.BatchNorm2d(nb_channels)

    def forward( self, x ):
        B, T, C, H, W = x.shape
        y = rearrange( x, "b t c h w -> (b t) c h w" )
        y = self.relu(self.norm1(self.conv1(y)))
        y = rearrange( y, "(b t) c h w -> (b h w) c t",
                b=B, t=T )
        y = self.relu(self.norm_t(self.conv_t(y)))
        y = rearrange( y, "(b h w) c t -> (b t) c h w",
                b=B, h=H, w=W )
        y = self.norm2(self.conv2(y))
        y = rearrange( y, "(b t) c h w -> b t c h w",
                b=B, t=T )

        return x+y

class SimpleMotionHead(nn.Module):
    def __init__( self ):
        super( SimpleMotionHead, self ).__init__()

    def forward( self, x ):
        return x[...,:2]

class SimpleVisHead(nn.Module):
    def __init__( self ):
        super( SimpleVisHead, self ).__init__()

    def forward( self, x ):
        return x[...,[2]]

class MaskHead(nn.Module):
    def __init__( self,
            input_dim: int = 256,
            scale: int = 8,
            ):
        super(MaskHead,self).__init__()
        self.ksize = 3
        self.scale = scale
        mask_output_dim = self.ksize**2 * scale**2
        mask_inner_dim = min( mask_output_dim//2, input_dim*2 )
        self.layers = nn.Sequential(
                nn.Conv2d( input_dim, mask_inner_dim,
                    kernel_size=3, padding=1 ),
                nn.ReLU(inplace=True),
                nn.Conv2d( mask_inner_dim, mask_output_dim,
                    kernel_size=1, padding=0 ),

                Scale(0.25),
                )
        default_init( self )

    def forward( self, x ):
        return self.layers(x)

class DtfNet(nn.Module):
    def __init__( self,
            input_channels: int = 3,
            img_dim: int = 256,
            xy_emb_dim: int = 16,
            t_emb_dim: int = 16,
            token_dim: int = 96,
            nb_tokens: int = 128,
            motion_dim: int = 128,
            mlp_inner_dim: int = 256,
            nb_layers: int = 8,
            mixer_depth: int = 2,
            mask_upsampling: bool = True,
            qk_dim: int = 96,
            vfeats_dim: int = 96,
            sim_levels: int = 3,
            sim_radius: int = 3,
            proj_kernel_size: int = 1,
            pos_emb_strat: PEStrategy = PEStrategy.NORM_CONCAT,
            qk_bias: bool = False,
            softmax_temp: float = 1.,
            num_heads: int = 1,
            shared_motion_net: bool = False,
            small: bool = False,
            ):
        super(DtfNet,self).__init__()
        #TODO A better encoder (e.g. Twins-SVT, TSM-ResNet, ...)
        # or pretrained (e.g. Masked AutoEncoder, ...) could improve results
        if small:
            self.encoder = SmallConvEncoder( input_channels, img_dim )
        else:
            self.encoder = ConvEncoder( input_channels, img_dim )
        self.motion_dim = motion_dim
        self.nb_layers = nb_layers

        self.pos_emb_x = FixedPositionEmbedding(
                1, xy_emb_dim,
                min_freq=[2e-3], max_freq=[0.5] )
        self.pos_emb_y = FixedPositionEmbedding(
                1, xy_emb_dim,
                min_freq=[2e-3], max_freq=[0.5] )
        self.pos_emb_t = FixedPositionEmbedding(
                1, t_emb_dim,
                min_freq=[2e-3], max_freq=[0.5] )
        self.pos_emb_dim = 4*xy_emb_dim + 2*t_emb_dim

        self.tokens = nn.Parameter(
                torch.randn((nb_tokens,token_dim)) )

        self.motion_head = nn.Sequential(
                nn.Linear( motion_dim, motion_dim*2 ),
                nn.ReLU(inplace=True),
                nn.Linear( motion_dim*2, 2, bias=False ),
                )
        default_init(self.motion_head)
        self.vis_head = nn.Sequential(
                nn.Linear( motion_dim, motion_dim*2 ),
                nn.ReLU(inplace=True),
                nn.Linear( motion_dim*2, 1, bias=False ),
                )
        default_init(self.vis_head)

        self.shared_motion_net = shared_motion_net
        self.motion_nets = nn.ModuleList([
            MotionNetwork(
                img_dim, motion_dim,
                sim_levels, sim_radius,
                num_heads,
                inner_dim = mlp_inner_dim,
                depth = mixer_depth,
                expansion_factor = 2,
                groups = 1 )
            for _ in range(1 if shared_motion_net else nb_layers) ])

        self.process = nn.ModuleList([ CentroidSummarization(
                img_dim=img_dim,
                pos_emb_dim=self.pos_emb_dim,
                token_dim=token_dim,
                motion_dim=motion_dim,
                mlp_inner_dim=mlp_inner_dim,
                qk_dim=qk_dim,
                vfeats_dim=vfeats_dim,
                sim_levels=sim_levels,
                sim_radius=sim_radius,
                motion_head=self.motion_head,
                motion_net=self.motion_nets[0 if shared_motion_net else l],
                proj_kernel_size=proj_kernel_size,
                pos_emb_strat=pos_emb_strat,
                qk_bias=qk_bias,
                softmax_temp=softmax_temp,
                num_heads=num_heads )
            for l in range(nb_layers) ])
        expansion_factor = 0.5
        def ffn(dim,inner_dim):
            return Residual( nn.Sequential(
                nn.Conv2d( dim, inner_dim,
                    kernel_size=3, padding=1 ),
                nn.BatchNorm2d(inner_dim),
                nn.ReLU(),
                nn.Conv2d( inner_dim, dim,
                    kernel_size=3, padding=1 ),
                nn.BatchNorm2d(dim),
                ))
        self.ffns = nn.ModuleList([
            SimpleTemporalFFN(img_dim,int(expansion_factor*img_dim),groups=1)
            for _ in range(nb_layers) ])
        self.motion_ffns = nn.ModuleList([
            SimpleTemporalFFN(motion_dim,int(expansion_factor*motion_dim),groups=1)
            for _ in range(nb_layers) ])

        self.token_updates = nn.ModuleList([ ConstantTokens(
            nb_tokens = nb_tokens,
            token_dim = token_dim,
            )
            for _ in range(nb_layers) ])

        self.scale = self.encoder.scale
        if mask_upsampling and self.scale > 1:
            self.mask_ksize = 3
            self.mask_head = MaskHead( img_dim, self.encoder.scale )
        else:
            self.mask_head = None

    def forward( self,
            seq: torch.Tensor,
            ref_idx: Union[torch.Tensor,int] = 0,
            upsample: bool = True,
            return_img_feats: bool = False ):
        B, T, C, H, W = seq.shape
        if hasattr(ref_idx, '__len__'):
            B_ = len(ref_idx)
        else:
            B_ = 1
        if B==1 and B_ > 1:
            broadcast = True
            B = B_
        else:
            broadcast = False
        dtype=seq.dtype
        device=seq.device
        tokens = repeat( self.tokens, "z c -> b z c", b=B )
        time = torch.arange(T,dtype=dtype,device=device) # [ 2-N, ..., 0, 1 ]
        b_range = torch.arange(B,dtype=torch.long,device=device)

        # Position embedding
        dH, dW = (H+self.scale-1)//self.scale, (W+self.scale-1)//self.scale
        x = torch.linspace( 0.5, dW-0.5, dW, dtype=dtype, device=device )
        y = torch.linspace( 0.5, dH-0.5, dH, dtype=dtype, device=device )

        if False and self.training:
            # Virtually add an offset, for better generalization to longer/larger sesquences
            x_emb = self.pos_emb_x(repeat(x,"w -> b w 1",b=B)-dW/2+torch.randint(-100,100,(B,1,1),device=device))
            y_emb = self.pos_emb_y(repeat(y,"h -> b h 1",b=B)-dH/2+torch.randint(-100,100,(B,1,1),device=device))
            t_emb = self.pos_emb_t(repeat(time,"t -> b t 1",b=B)+torch.randint(-200,200,(B,1,1),device=device))
        else:
            x_emb = repeat( self.pos_emb_x((x-dW/2).unsqueeze(1)), "w c -> b w c", b=B )
            y_emb = repeat( self.pos_emb_y((y-dH/2).unsqueeze(1)), "h c -> b h c", b=B )
            t_emb = repeat( self.pos_emb_t(time.unsqueeze(1)), "t c -> b t c", b=B )

        x_emb = repeat( x_emb, "b w c -> b t c h w", b=B, t=T, h=dH, w=dW )
        y_emb = repeat( y_emb, "b h c -> b t c h w", b=B, t=T, h=dH, w=dW )
        t_emb = repeat( t_emb, "b t c -> b t c h w", b=B, t=T, h=dH, w=dW )
        pos_emb = torch.cat((x_emb,y_emb,t_emb),dim=2)

        x = repeat( x, "w -> b t h w", b=B, t=T, h=dH )
        y = repeat( y, "h -> b t h w", b=B, t=T, w=dW )
        pos = torch.stack((x,y),dim=2)

        motion = torch.zeros((B,T,2,dH,dW),dtype=dtype,device=device)
        visibility = torch.zeros((B,T,1,dH,dW),dtype=dtype,device=device)
        motion_feats = torch.zeros((B,T,self.motion_dim,dH,dW),dtype=dtype,device=device)

        # Process sequence
        seq_feats = self.encoder(seq)
        if broadcast:
            seq_feats = repeat( seq_feats, "1 t c h w -> b t c h w", b=B )
        if return_img_feats:
            all_seq_feats = [seq_feats]

        motion_predictions = []
        vis_predictions = []
        for l in range(self.nb_layers):
            vis = torch.sigmoid(visibility)
            d_seq, d_motion, qk_softmax, qk_softmin = self.process[l](
                    tokens,
                    seq_feats,
                    motion_feats,
                    motion, vis,
                    pos, pos_emb,
                    ref_idx = ref_idx )

            # Update tokens
            tokens = self.token_updates[l]( tokens, qk_softmax, qk_softmin )

            # Update image/sequence features
            seq_feats = seq_feats+d_seq
            seq_feats = self.ffns[l](seq_feats)

            # Update motion features
            motion_feats = motion_feats+d_motion
            motion_feats = self.motion_ffns[l](motion_feats)

            if return_img_feats:
                all_seq_feats.append(seq_feats)

            # Flow
            motion_feats_ = rearrange( motion_feats, "b t c h w -> b t h w c" )
            motion = self.motion_head(motion_feats_)
            motion = rearrange( motion, "b t h w c -> b t c h w" )
            visibility = self.vis_head(motion_feats_)
            visibility = rearrange( visibility, "b t h w c -> b t c h w" )

            # Upsampling
            if upsample:
                if self.mask_head is not None:
                    up_mask = 0.25 * self.mask_head(seq_feats[b_range,ref_idx])
                    up_mask = rearrange( up_mask, "b (kh kw fh fw) h w -> b (kh kw) fh fw h w",
                            kh=self.mask_ksize, kw=self.mask_ksize,
                            fh=self.scale, fw=self.scale )
                    up_mask = torch.softmax( up_mask, dim=1 )
                    up_mask = rearrange( up_mask, "b (kh kw) fh fw h w -> b kh kw fh fw h w",
                            kh=self.mask_ksize, kw=self.mask_ksize )

                    motion_ = rearrange( motion, "b t c h w -> b (t c) h w" )
                    up_motion = self.scale*img_utils.mask_upsampling(motion_,up_mask)
                    up_motion = rearrange( up_motion, "b (t c) h w -> b t c h w", t=T, c=2 )
                    visibility_ = rearrange( visibility, "b t c h w -> b (t c) h w" )
                    up_vis = img_utils.mask_upsampling(visibility_,up_mask)
                    up_vis = up_vis.unsqueeze(2)
                elif self.scale != 1:
                    motion_ = rearrange( motion, "b t c h w -> b (t c) h w" )
                    up_motion = self.scale*img_utils.upsampling(motion_,self.scale)
                    up_motion = rearrange( up_motion, "b (t c) h w -> b t c h w", t=T, c=2 )
                    visibility_ = rearrange( visibility, "b t c h w -> b (t c) h w" )
                    up_vis = img_utils.upsampling(visibility_,self.scale)
                    up_vis = up_vis.unsqueeze(2)
                else:
                    up_motion = motion
                    up_vis = visibility
            else:
                up_motion = motion
                up_vis = visibility
            up_vis = torch.sigmoid(up_vis)
            motion_predictions.append(up_motion)
            vis_predictions.append(up_vis)
        if type(motion_predictions) is list:
            motion_predictions = torch.stack(motion_predictions,dim=1)
            vis_predictions = torch.stack(vis_predictions,dim=1)

        grid = torch.stack( torch.meshgrid(
            (torch.arange(W,dtype=dtype,device=device),
             torch.arange(H,dtype=dtype,device=device)),
            indexing='xy' ), dim=0 )
        trajs_predictions = motion_predictions + grid[None,None,None,:,:,:]

        if return_img_feats:
            if type(all_seq_feats) is list:
                all_seq_feats = torch.stack(all_seq_feats,dim=1)
            return trajs_predictions, vis_predictions, all_seq_feats
        else:
            return trajs_predictions, vis_predictions

class SmallDtfNet(DtfNet):
    def __init__( self ):
        super(SmallDtfNet,self).__init__(
            input_channels = 3,
            img_dim = 96,
            xy_emb_dim = 16,
            t_emb_dim = 12,
            token_dim = 48,
            nb_tokens = 64,
            motion_dim = 64,
            mlp_inner_dim = 192,
            nb_layers = 2,
            mixer_depth = 6,
            mask_upsampling = True,
            qk_dim = 48,
            vfeats_dim = 64,
            sim_levels = 3,
            sim_radius = 1,
            proj_kernel_size = 1,
            pos_emb_strat = PEStrategy.NORM_CONCAT,
            qk_bias = False,
            softmax_temp = 3.,
            num_heads = 1,
            shared_motion_net = False,
            small = True )

class BigDtfNet(DtfNet):
    def __init__( self ):
        super(MySuperNetwork5,self).__init__(
            input_channels = 3,
            img_dim = 192,
            xy_emb_dim = 24,
            t_emb_dim = 16,
            token_dim = 128,
            nb_tokens = 128,
            motion_dim = 128,
            mlp_inner_dim = 192,
            nb_layers = 16,
            mixer_depth = 2,
            mask_upsampling = True,
            qk_dim = 96,
            vfeats_dim = 128,
            sim_levels = 4,
            sim_radius = 3, #TODO 2
            proj_kernel_size = 1,
            pos_emb_strat = PEStrategy.NORM_CONCAT,
            qk_bias = False,
            softmax_temp = 1.,
            num_heads = 4,
            small = False )
