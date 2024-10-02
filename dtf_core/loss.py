import torch
import math
from enum import Enum, auto

from typing import Optional
from einops import rearrange, repeat

class Reduction(Enum):
    NONE = auto()
    SUM = auto()
    MEAN = auto()
    L2 = auto()

class WingLoss(torch.nn.Module):
    """
    This loss comes from the 'Feng et al. 2018' paper.
    It is designed to have good gradient properties (robust + accurate)
    for large and small errors.
    It is formulated as follow:

        wing(x) = w*ln(1+|x|/eps)    if |x| < w
                  |x|-C              else

    reduction can be 'none', 'mean', or 'sum'

    You can apply l2 norm on some dimensions BEFORE applying the wing loss
    """
    def __init__( self, w=1., eps=2., reduction: Reduction = Reduction.MEAN, l2_dim=None ):
        super( WingLoss, self ).__init__()
        self.w = w
        self.eps = eps
        self.C = self.w-self.w*math.log(1.+self.w/self.eps)
        self.reduction = reduction
        self.l2_dim = l2_dim

    def forward( self, x, y=None ):
        if y is not None:
            x = y-x
        if self.l2_dim is not None:
            x = x.norm(p=2,dim=self.l2_dim)
        abs_x = x.abs()
        loss = torch.where( abs_x < self.w,
            self.w*torch.log(1.+abs_x/self.eps),
            abs_x-self.C )
        if self.reduction == Reduction.MEAN:
            return loss.mean()
        elif self.reduction == Reduction.SUM:
            return loss.sum()
        elif self.reduction == Reduction.NONE:
            return loss
        elif self.reduction == Reduction.L2:
            return (loss**2.).sum()**0.5
        else:
            raise ValueError("Unknown reduction: '%s'" % self.reduction)

class HuberLoss(torch.nn.Module):
    """
    Robust loss, mixing L2 when small and L1 when large

    huber(x) = 0.5 * x**2      if |x] < d
               d*(|x|-0.5*d)   else
    """
    def __init__( self, d: float = 1., reduction: Reduction = Reduction.MEAN, l2_dim=None ):
        super( HuberLoss, self ).__init__()
        self.d = d
        self.reduction = reduction
        self.l2_dim = l2_dim

    def forward( self, x, y=None ):
        if y is not None:
            x = y-x
        if self.l2_dim is not None:
            x = x.norm(p=2,dim=self.l2_dim)
        abs_x = x.abs()
        loss = torch.where( abs_x < self.d,
                0.5*x**2,
                self.d * (abs_x - 0.5*self.d) )
        if self.reduction == Reduction.MEAN:
            return loss.mean()
        elif self.reduction == Reduction.SUM:
            return loss.sum()
        elif self.reduction == Reduction.NONE:
            return loss
        elif self.reduction == Reduction.L2:
            return (loss**2.).sum()**0.5
        else:
            raise ValueError("Unknown reduction: '%s'" % self.reduction)

class DTFLossType(Enum):
    EPE = 'epe'
    AAE = 'aae'
    WING = 'wing'
    HUBER = 'huber'
    L2 = 'l2'
    MSE = 'mse'
    MAX_MAG = 'max_mag'
    F1 = 'f1px'
    F3 = 'f3px'
    F5 = 'f5px'
    VIS = 'vis'
    CORR = 'corr'
    LAPLACIAN = 'lap'

def traj_losses(
        ref_idx: torch.Tensor,
        trajs_preds: torch.Tensor,
        vis_preds: torch.Tensor,
        feats: Optional[torch.Tensor],
        trajs_gt: torch.Tensor,
        vis_gt: torch.Tensor,
        gamma: float = 0.8,
        max_motion: float = 300.,
        w: float = 2.,
        eps: float = 0.5,
        ):
    """
    Loss function defined over a sequence of trajectories and visibilities
    ref_idx:      [ B, 1 ]
    trajs_preds:  [ B, P, T, 2, H, W ]
    vis_preds:    [ B, P, T, 1, H, W ]
    feats:        [ B, P, T, C, H, W ]

    trajs_gt:     [ B,    T, 2, H, W ]
    vis_gt:       [ B,    T, 1, H, W ]
    """

    device = trajs_preds.device
    dtype = trajs_preds.dtype
    B, P, T, _, H, W = trajs_preds.shape
    #dtype_eps = torch.finfo(dtype).eps
    dtype_eps = 1e-3 if dtype is torch.float16 else 1e-5

    pred_weights = gamma**torch.arange(P-1,-1,-1,dtype=dtype,device=device)
    pred_weights = pred_weights/pred_weights.sum()

    # Flow loss (+ first frame, whose motion should be 0)
    # exlude invalid pixels and extremely large diplacements
    b_range = torch.arange(B,dtype=torch.long,device=device)
    trajs_gt = trajs_gt.unsqueeze(1)
    vis_gt = vis_gt.unsqueeze(1)
    ref_pos_preds = trajs_preds[b_range,:,ref_idx]
    ref_pos_gt = trajs_gt[b_range,:,ref_idx] # B,P,2,H,W
    motion_gt = trajs_gt - ref_pos_gt.unsqueeze(2)
    motion_preds = trajs_preds - ref_pos_preds.unsqueeze(2)
    mag_preds = torch.norm( motion_preds, dim=3, p=2 )
    mag_gt = torch.norm( motion_gt, dim=3, p=2 )
    traj_diff = trajs_gt - trajs_preds
    diff_mag2 = torch.einsum( "b p t c h w, b p t c h w -> b p t h w", traj_diff, traj_diff )
    diff_mag2 = diff_mag2.clamp( max=max_motion**2 )
    diff_mag = torch.sqrt(diff_mag2+dtype_eps)
    mask = torch.where( mag_gt < max_motion,
            torch.ones_like(mag_gt),
            torch.zeros_like(mag_gt) )
    #mask /= mask.sum()
    all_weights = pred_weights[None,:,None,None,None] * mask # B, P, T, H, W
    all_weights /= all_weights.sum()

    epe = (all_weights*diff_mag).sum()
    mse = (all_weights*diff_mag2).sum()
    l2 = (all_weights*diff_mag2).sum((3,4)).sqrt().sum()
    wing = (all_weights*WingLoss(w=w,eps=eps,reduction=Reduction.NONE)(diff_mag)).sum()
    huber = (all_weights*HuberLoss(d=w,reduction=Reduction.NONE)(diff_mag)).sum()

    # AAE loss
    dot_product = torch.einsum( "b p t c h w, b p t c h w -> b p t h w", motion_gt, motion_preds )
    #weights_ae_preds = 1.-torch.exp(-2.*mag_preds) # ignore very low magnitudes
    #weights_ae_gt = 1.-torch.exp(-2.*mag_gt)
    #mask_ae = weights_ae_preds*weights_ae_gt
    #mask_ae = mask_ae / mask_ae.sum(dim=(0,3,4),keepdim=True)
    #mask_ae = mask_ae / mask_ae.sum()
    mask_ae = (mag_gt>0.3).logical_and(mag_preds>0.3).to(dtype)
    all_weights_ae = pred_weights[None,:,None,None,None] * mask_ae # B, P, T, H, W
    all_weights_ae /= all_weights_ae.sum()
    ae = dot_product / (mag_gt*mag_preds + dtype_eps)
    ae = torch.acos( ae.clamp(-1.,1.) )
    aae = (all_weights_ae*ae).sum()
    max_mag = mag_preds.max()

    # Visibility
    vis_cross = -vis_gt*torch.log(vis_preds+dtype_eps) \
            - (1-vis_gt)*torch.log(1-vis_preds+dtype_eps)
    vis_cross = vis_cross.squeeze(3)
    vis_loss = (all_weights*vis_cross).sum()
    #TODO test balanced_ce_loss from pips ?

    # Laplacian
    gt_lap = 0.25 * (4*trajs_gt[:,:,:,:,1:-1,1:-1]
            - trajs_gt[:,:,:,:,2:,1:-1] - trajs_gt[:,:,:,:,:-2,1:-1]
            - trajs_gt[:,:,:,:,1:-1,2:] - trajs_gt[:,:,:,:,2:,1:-1])
    pred_lap = 0.25 * (4*trajs_preds[:,:,:,:,1:-1,1:-1]
            - trajs_preds[:,:,:,:,2:,1:-1] - trajs_preds[:,:,:,:,:-2,1:-1]
            - trajs_preds[:,:,:,:,1:-1,2:] - trajs_preds[:,:,:,:,2:,1:-1])
    lap_diff = torch.sqrt( (gt_lap-pred_lap).norm(p=2,dim=3) + dtype_eps )
    lap_loss = (all_weights[:,:,:,1:-1,1:-1]*lap_diff).sum()


    # Score loss
    if feats is not None:
        C = feats.shape[3]
        feats_ = rearrange( feats[:,:-1], "b p t c h w -> (b p t) c h w" )
        hw8_scale = torch.tensor([ 2/(W/8), 2/(H/8) ], dtype=dtype, device=device )
        feat_sampling_coords = hw8_scale[None,None,None,:] * \
                rearrange( trajs_preds, " b p t c h w -> (b p t) h w c" ) - 1.
        trajs_feats = torch.nn.functional.grid_sample(
                feats_, feat_sampling_coords,
                padding_mode='border',
                align_corners=False)
        trajs_feats = rearrange( trajs_feats, "(b p t) c h w -> b p t c h w",
                b=B, t=T, p=P )
        traj_corr = torch.einsum( "b p c h w, b p t c H W -> b p t h w H W",
                trajs_feats[b_range,:,ref_idx], feats ) / math.sqrt(C)
        traj_corr.clamp_(-20.,20.)
        self_corr = torch.einsum( "b p c h w, b p t c h w -> b p t h w",
                trajs_feats[b_range,:,ref_idx], trajs_feats ) / math.sqrt(C)
        self_corr.clamp_(-20.,20.)
        corr_loss = -torch.log( torch.exp(self_corr) / torch.exp(traj_corr).mean((-2,-1)) + dtype_eps )
        corr_loss = (vis_gt.squeeze(3) * corr_loss * all_weights).sum()

    last_mask = mask[:,-1] / mask[:,-1].sum()
    losses = {
        DTFLossType.EPE: epe,
        DTFLossType.WING: wing,
        DTFLossType.HUBER: huber,
        DTFLossType.L2: l2,
        DTFLossType.MSE: mse,
        DTFLossType.AAE: aae*180./math.pi,
        DTFLossType.MAX_MAG: max_mag,

        # Only on last prediction
        DTFLossType.F1: 1 - ( last_mask*(diff_mag[:,-1] < 1).float() ).sum(),
        DTFLossType.F3: 1 - ( last_mask*(diff_mag[:,-1] < 3).float() ).sum(),
        DTFLossType.F5: 1 - ( last_mask*(diff_mag[:,-1] < 5).float() ).sum(),

        DTFLossType.VIS: vis_loss,
        DTFLossType.LAPLACIAN: lap_loss,
    }
    if feats is not None:
        losses[DTFLossType.CORR] = corr_loss

    return losses
