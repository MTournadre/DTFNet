import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

import math
from typing import Optional, List, Union
from dataclasses import dataclass
from enum import Enum, auto
import functools

from .nn_utils import NormFn, Residual
from .nn_utils import default_init
from .img_utils import bilinear_sampling

class PEStrategy(Enum):
    CONCAT_NORM = auto()
    NORM_CONCAT = auto()
    ADD_NORM    = auto()
    NORM_ADD    = auto()

def visu_attn( attn, coords=None, heads=[0,1], batch_idx=0, time_idx=-1 ):
    if isinstance(attn,tuple) or isinstance(attn,list):
        B, N, HQ, WQ, _ = attn[0][0].shape
        TK, HK, WK = attn[1][0].tolist()
    else:
        B, N, HQ, WQ, TK, HK, WK = attn.shape
    if coords is None:
        ys = [ HQ//4, HQ//2, (3*HQ)//4 ]
        xs = [ WQ//4, WQ//2, (3*WQ)//4 ]
        coords = [ (y,x) for y in ys for x in xs ]

    if isinstance(attn,tuple) or isinstance(attn,list):
        q_coords = [ x*WQ+y for (x,y) in coords ]
        Q = len(q_coords)
        coords_tensor = torch.as_tensor(coords)
        coords_y = coords_tensor[:,0].tolist()
        coords_x = coords_tensor[:,1].tolist()
        attn0 = [
            a[[batch_idx]][:,heads,:,:,:][:,:,coords_y,coords_x,:].unsqueeze(2)
            for a in attn[0] ]
        attn1 = attn[1]
        attn2 = attn[2]
        attn3 = [
            i[[batch_idx]][:,heads,:,:,:][:,:,coords_y,coords_x,:].unsqueeze(2)
            for i in attn[3] ]
        attn = reconstruct_pyramid_attn_2d( attn0, attn1, attn2, attn3 )
        visu_coords = coords
        coords = [ (0,i) for i in range(Q) ]
        heads = list(range(len(heads)))
        attn = attn[0]
        attn = attn[:,:,:,time_idx,:,:]
    else:
        if attn.dim() > 5:
            attn = attn[batch_idx]
        if attn.dim() > 5:
            attn = attn[:,:,:,time_idx]
        visu_coords = coords

    imgs = []
    for ic, (i,j) in enumerate(coords):
        for n in heads:
            img = torch.empty((3,HK,WK))
            vi, vj = visu_coords[ic]
            a = attn[n,i,j,:,:]
            scale = min( 1e5, 1./a.max() )
            img[:] = a.unsqueeze(0)*scale
            x = img[0,vi,vj]
            img[0,vi,vj] = 1.-x
            img[2,vi,vj] = 0.

            imgs.append(img)

    return torch.stack(imgs,dim=0)

class MotionNetwork(nn.Module):
    def __init__( self,
            img_dim: int,
            motion_dim: int,
            corr_levels: int,
            corr_radius: int,
            num_heads: int,
            inner_dim: int,
            expansion_factor: int = 2,
            depth: int = 1,
            padding_mode: str = 'zeros',
            groups: int = 1,
            ):
        super(MotionNetwork,self).__init__()

        self.img_dim = img_dim
        self.motion_dim = motion_dim
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.inner_dim = inner_dim
        self.num_heads = num_heads

        self.corr_patch_size = (2*self.corr_radius+1)**2
        self.corr_feat_size = self.corr_patch_size**2

        # Same projection for all patch correlation, for all heads
        self.corr_proj_dim = 4*self.corr_patch_size
        self.corr_proj = nn.Sequential(
                nn.Conv1d(
                    self.corr_levels*self.corr_patch_size*self.corr_patch_size,
                    self.corr_levels*self.corr_proj_dim,
                    kernel_size=3, padding=1,
                    groups=self.corr_levels*self.corr_patch_size ),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    self.corr_levels*self.corr_proj_dim,
                    self.corr_proj_dim,
                    kernel_size=3, padding=1,
                    groups=1 ),
                )

        # Same projection for all heads
        input_dim = self.num_heads*(self.img_dim+self.motion_dim+self.corr_proj_dim)
        self.proj = nn.Sequential(
                nn.Conv1d( input_dim, inner_dim*2,
                          kernel_size=3,
                          padding=1,
                          groups=groups ),
                nn.ReLU(inplace=True),
                nn.Conv1d( inner_dim*2, inner_dim,
                          kernel_size=3,
                          padding=1,
                          groups=groups ),
                nn.ReLU(inplace=True),
                )
        self.process = nn.Sequential(
            *[ Residual(
                    nn.Sequential(
                        nn.GroupNorm(1,inner_dim),
                        nn.Conv1d(inner_dim,expansion_factor*inner_dim,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            padding_mode=padding_mode,
                            groups=groups,
                            ),
                        nn.ReLU(inplace=True),
                        nn.GroupNorm(1,expansion_factor*inner_dim),
                        nn.Conv1d(expansion_factor*inner_dim,inner_dim,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            padding_mode=padding_mode,
                            groups=groups,
                            ),
                        #nn.ReLU(inplace=True),
                    ),
                )
            for _ in range(depth) ] )
        self.final_layer = nn.Sequential(
                #nn.InstanceNorm1d(inner_dim),
                nn.Conv1d(inner_dim,expansion_factor*inner_dim,kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(expansion_factor*inner_dim,img_dim+motion_dim,kernel_size=1)
                )

        default_init(self)

    def forward( self, feats, corr, motion_feats, motion, vis ):
        B, T, Z, N, _, _, _, = corr.shape

        # Project correlations features
        corr = rearrange( corr, "b t z n l p P -> (b z n) (l p P) t" )
        corr = self.corr_proj(corr)

        # Merge all features
        corr = rearrange( corr, "(b z n) c t -> b z n c t",
                         b=B, t=T, z=Z, n=N, c=self.corr_proj_dim )
        feats = rearrange( feats, "b t z n c -> b z n c t" )
        motion_feats = rearrange( motion_feats, "b t z n c -> b z n c t" )
        x = torch.cat((feats,corr,motion_feats),dim=3)
        x = rearrange( x, "b z n c t -> (b z) (n c) t" )

        # Process altogether
        x = self.proj(x)
        x = self.process(x)
        x = self.final_layer(x)

        # Extract image and motion updates
        x = rearrange( x, "(b z) (n c) t -> b t z n c",
                b=B, z=Z, n=N )
        feats, motion = x.split((
            self.img_dim//self.num_heads,
            self.motion_dim//self.num_heads),dim=-1)
        return feats, motion

class CentroidSummarization(nn.Module):
    def __init__( self,
            img_dim: int,
            pos_emb_dim: int,
            token_dim: int,
            qk_dim: int,
            vfeats_dim: int,
            motion_dim: int,
            mlp_inner_dim: int,
            sim_levels: int,
            sim_radius: int,
            motion_head: nn.Module,
            motion_net: nn.Module,
            proj_kernel_size: int = 1,
            mixer_depth: int = 4,
            pos_emb_strat: PEStrategy = PEStrategy.NORM_CONCAT,
            qk_bias: bool = False,
            softmax_temp: float = 1.,
            num_heads: int = 4,
            ):
        super(CentroidSummarization,self).__init__()

        self.num_heads = num_heads
        self.dot_scale = (qk_dim / num_heads) ** -0.5
        #TODO Actually there is no "V" in centroid summarization
        # We keep this parameter for legacy reasons...
        self.sim_scale = (vfeats_dim / num_heads) ** -0.5
        self.pos_emb_dim = pos_emb_dim
        self.pos_emb_strat = pos_emb_strat
        self.softmax_temp = softmax_temp
        self.sim_levels = sim_levels
        self.sim_radius = sim_radius

        self.motion_head = motion_head

        if pos_emb_strat == PEStrategy.CONCAT_NORM:
            self.compress_norm_kv = nn.GroupNorm(1,img_dim+self.pos_emb_dim)
        else:
            self.compress_norm_kv = nn.GroupNorm(1,img_dim)

        if pos_emb_strat in [PEStrategy.CONCAT_NORM,PEStrategy.NORM_CONCAT]:
            img_pos_dim = img_dim + self.pos_emb_dim
        else:
            img_pos_dim = img_dim

        # Note: if you plan to learn absolute tokens,
        # then just learn their projection instead of token+proj...
        #self.proj_q = nn.Linear(token_dim,qk_dim,bias=qk_bias)
        self.proj_q = nn.Identity()
        self.proj_k = nn.Conv2d(img_pos_dim,qk_dim,
                kernel_size=proj_kernel_size,
                bias=qk_bias)

        corr_feat_dim = sim_levels * (2*sim_radius+1)**4
        self.motion_net = motion_net

        self.motion_proj_out = nn.Conv2d(
                motion_dim, motion_dim,
                kernel_size=1,
                bias=qk_bias )
        self.feat_proj_out = nn.Conv2d(
                img_dim, img_dim,
                kernel_size=1,
                bias=qk_bias )

    def forward( self,
            tokens: torch.Tensor,
            seq: torch.Tensor,
            seq_motion_feats: torch.Tensor,
            seq_motion: torch.Tensor,
            seq_vis: torch.Tensor,
            seq_pos: torch.Tensor,
            seq_pos_emb: torch.Tensor,
            ref_idx: Union[torch.Tensor,int] ):
        B, T, C, H, W = seq.shape
        Z = tokens.shape[1]
        device = seq.device
        seq_pyr = self.construct_pyramid(seq)
        b_range = torch.arange(B,dtype=torch.long,device=device)
        N=self.num_heads
        compress_kv = rearrange( seq, "b t c h w -> (b t) c h w" )
        # = pool motion features over all frames
        compress_pos_emb = rearrange( seq_pos_emb, "b t c h w -> (b t) c h w" )

        # Compressor cross-attention
        if self.pos_emb_strat in [PEStrategy.NORM_ADD,PEStrategy.NORM_CONCAT]:
            # Apply normalization BEFORE using positional enbeddings
            compress_kv = self.compress_norm_kv(compress_kv)

        if compress_pos_emb is not None:
            if self.pos_emb_strat in [PEStrategy.NORM_ADD,PEStrategy.ADD_NORM]:
                # Add positional Embedding
                compress_kv = compress_kv+compress_pos_emb
            elif self.pos_emb_strat in [PEStrategy.CONCAT_NORM,PEStrategy.NORM_CONCAT]:
                # Concat positional Embedding
                compress_kv = torch.cat((compress_kv,compress_pos_emb),dim=1)

        if self.pos_emb_strat in [PEStrategy.ADD_NORM,PEStrategy.CONCAT_NORM]:
            # Apply normalization AFTER using positional enbeddings
            compress_kv = self.compress_norm_kv(compress_kv)

        q = self.proj_q(tokens)
        k = self.proj_k(compress_kv)

        q = rearrange( q, "b z (n c) -> b z n c", n=N )
        k = rearrange(
                k, "(b t) (n c) h w -> b t n c h w", b=B, t=T, n=N )

        qk_dot = torch.einsum( "b z n c, b t n c h w -> b t n z h w",
                q, k ) * self.dot_scale
        #if self.training and self.dropout_rate > 0.:
        #    qk_dot.where(
        #        torch.rand_like(qk_dot) < self.dropout_rate,
        #        float("-inf")*torch.ones_like(qk_dot) )

        qk_dot = rearrange( qk_dot, "b t n z h w -> b t n z (h w)" )
        attn = torch.softmax( qk_dot*self.softmax_temp, dim=4 )
        reciprocal_attn = torch.softmax( qk_dot*self.softmax_temp, dim=3 )
        attn = rearrange( attn, "b t n z (h w) -> b t n z h w",
                h=H, w=W )
        attn_ref = attn[b_range,ref_idx]
        reciprocal_attn = rearrange( reciprocal_attn, "b t n z (h w) -> b t n z h w",
                h=H, w=W )

        if False:
            # Compute centroid position by applying the attention
            # to the pos/motion map (= softargmax)
            # = cluster motion
            centroid_ref_pos = torch.einsum( "b n z h w, b c h w -> b z n c",
                    attn_ref, seq_pos[b_range,ref_idx] )
        else:
            # Compute real argmax for position ('hardargmax')
            # This avoids landing outside of a non-convex attention cluster
            # = centroid motion
            # This is more valid. From our experiments, it converges faster.
            attn_ = rearrange( attn, "b t n z h w -> b t n z (h w)" )
            centroid_idx = attn_[b_range,ref_idx].argmax( dim=-1 )

            img_ref_pos = rearrange( seq_pos[b_range,ref_idx], "b c h w -> b c (h w)" )
            centroid_ref_pos = rearrange(
                    img_ref_pos[ b_range[:,None,None], :, centroid_idx ],
                    "b n z c -> b z n c" )

            seq_motion_feats_ = rearrange( seq_motion_feats, "b t c h w -> b t c (h w)" )

            seq_motion_ = rearrange( seq_motion, "b t c h w -> b t c (h w)" )
            centroid_motion = rearrange(
                    seq_motion_[ b_range[:,None,None], :, :, centroid_idx ],
                    "b n z t c -> b t z n c" )

            seq_vis_ = rearrange( seq_vis, "b t c h w -> b t c (h w)" )
            latent_vis = rearrange(
                    seq_vis_[ b_range[:,None,None], :, :, centroid_idx ],
                    "b n z t c -> b t z n c" )

        # Use a soft-argmax for motion features
        cluster_motion_feats = torch.einsum( "b n z h w, b t c h w -> b t z n c",
                attn_ref, seq_motion_feats )

        centroid_all_pos = centroid_ref_pos.unsqueeze(1) + centroid_motion
        centroid_feats, centroid_corr_feats = self.extract_correlation_features(
                seq_pyr, centroid_ref_pos, centroid_all_pos, ref_idx )
        # Use attention weights instead
        centroid_feats = torch.einsum( "b t n z h w, b t c h w -> b t z n c",
                attn, seq )

        d_seq_feats, d_motion_feats = self.motion_net(
                centroid_feats, centroid_corr_feats, cluster_motion_feats,
                centroid_motion, latent_vis )

        # Apply reciprocal projection
        res_d_motion = torch.einsum( "b n z h w, b t z n c -> b t n c h w",
                reciprocal_attn[b_range,ref_idx], d_motion_feats )
        res_d_seq_feats = torch.einsum( "b t n z h w, b t z n c -> b t n c h w",
                reciprocal_attn, d_seq_feats )

        # Output projection
        res_d_motion = rearrange( res_d_motion, "b t n c h w -> (b t) (n c) h w" )
        res_d_motion = self.motion_proj_out(res_d_motion)
        res_d_motion = rearrange( res_d_motion, "(b t) c h w -> b t c h w", b=B, t=T )

        res_d_seq_feats = rearrange( res_d_seq_feats, "b t n c h w -> (b t) (n c) h w" )
        res_d_seq_feats = self.feat_proj_out(res_d_seq_feats)
        res_d_seq_feats = rearrange( res_d_seq_feats, "(b t) c h w -> b t c h w", b=B, t=T )

        # Compute per-cluster softmax/softmin for TokenProcessor
        compress_softmin_attn_ref = torch.softmax(
                -qk_dot[b_range,ref_idx]*self.softmax_temp, dim=-1 )
        compress_softmin_attn_ref = rearrange( compress_softmin_attn_ref,
                "b n z (h w) -> b n z h w",
                h=H, w=W )
        k_softmax = torch.einsum( "b n z h w, b n c h w -> b z n c",
                attn_ref, k[b_range,ref_idx] )
        k_softmax = rearrange( k_softmax, "b z n c -> b z (n c)" )
        k_softmin = torch.einsum( "b n z h w, b n c h w -> b z n c",
                compress_softmin_attn_ref, k[b_range,ref_idx] )
        k_softmin = rearrange( k_softmin, "b z n c -> b z (n c)" )

        return res_d_seq_feats, res_d_motion, k_softmax, k_softmin

    def construct_pyramid( self, seq ):
        B, T, C, H, W = seq.shape
        seq = rearrange( seq, "b t c h w -> (b t) c h w" )
        seq_pyr = [seq]
        for l in range(self.sim_levels-1):
            seq = nn.functional.avg_pool2d(seq,kernel_size=2)
            seq_pyr.append(seq)

        seq_pyr = [ rearrange( s, "(b t) c h w -> b t c h w", b=B, t=T )
                for s in seq_pyr ]
        return seq_pyr

    def extract_correlation_features( self,
            seq_pyr: List[torch.Tensor],
            ref_pos: torch.Tensor,
            pos: torch.Tensor,
            ref_idx: Union[torch.Tensor,int]
            ):
        # The difference between ref_pos and pos[ref_idx] is that
        # the latter might not be exactly at centroid position,
        # because we added the predicted motion to it

        # seq: L x [B, T, C, H, W]
        # pos: B, T, Z, N, 2 (in XY order)
        L = self.sim_levels
        R = self.sim_radius
        N = self.num_heads

        ref_pos = ref_pos.detach()
        pos = pos.detach()

        B, T, C, H, W = seq_pyr[0].shape
        _, _, Z, N, _ = pos.shape
        device = seq_pyr[0].device
        dtype = seq_pyr[0].dtype

        wh = torch.tensor([W,H],dtype=dtype,device=device)
        norm_pos = pos*(2./wh.broadcast_to(pos.shape)) - 1.
        norm_ref_pos = ref_pos*(2./wh.broadcast_to(ref_pos.shape)) - 1.

        b_range = torch.arange(B,dtype=torch.long,device=device)
        l_range = torch.arange(L,dtype=dtype,device=device)
        r_range = torch.arange(-R,R+1,dtype=dtype,device=device)
        scales = 2.**l_range
        dpos = torch.stack( torch.meshgrid((r_range,r_range),indexing='xy') )
        dpos = rearrange( dpos, "c w h -> (w h) c" )
        norm_dpos = dpos*(2./wh.unsqueeze(0))
        dot_factor = 1./math.sqrt(C)

        all_norm_dpos = scales[:,None,None] * norm_dpos[None,:,:]
        all_norm_pos = norm_pos[None,:,:,:,:,None,:] + all_norm_dpos[:,None,None,None,None,:,:]
        all_norm_ref_pos = norm_ref_pos[None,:,:,:,None,:] + all_norm_dpos[:,None,None,None,:,:]

        all_corr_feats = []
        for l in range(L):
            l_seq = rearrange( seq_pyr[l], "b t c h w -> (b t) c h w" )
            l_pos = rearrange( all_norm_pos[l], "b t z n r c -> (b t) (z n) r c" )
            feats = nn.functional.grid_sample(
                    l_seq, l_pos,
                    mode='bilinear',
                    padding_mode='zeros', #TODO border or zeros ?
                    align_corners=False )
            feats = rearrange( feats, "(b t) c (z n) r -> b t c z n r",
                    b=B, t=T, z=Z, n=N )

            l_ref_pos = rearrange( all_norm_ref_pos[l], "b z n r c -> b (z n) r c" )
            ref_feats = nn.functional.grid_sample(
                    seq_pyr[l][b_range,ref_idx],
                    l_ref_pos,
                    mode='bilinear',
                    padding_mode='zeros', #TODO border or zeros ?
                    align_corners=False )
            ref_feats = rearrange( ref_feats, "b c (z n) r -> b c z n r",
                    z=Z, n=N )

            # Extract raw features at center, at finest level, for MotionNetwork
            if l == 0:
                res_feats = feats[:,:,:,:,:,((2*R+1)**2)//2]
                res_feats = rearrange( res_feats, "b t c z n -> b t z n c" )

            corr_feats = torch.einsum( "b c z n r, b t c z n R -> b t z n r R",
                    ref_feats, feats ) * dot_factor
            all_corr_feats.append(corr_feats)

        all_corr_feats = torch.stack(all_corr_feats, dim=-3)
        return res_feats, all_corr_feats

