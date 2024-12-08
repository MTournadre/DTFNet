import os
import numpy as np
import einops
import math
import time
import glob
import errno
import cv2

import torch
import torch.nn.functional as F
import torchvision

os.environ["NO_GCE_CHECK"] = "true"
import tensorflow_datasets
tensorflow_datasets.core.utils.gcs_utils._is_gcs_disabled = True

import tapnet.evaluation_datasets as evd
from dtf_core.networks import dtfnet
import dtf_core.longterm_flow_datasets as ltfds
import dtf_core.traj_datasets as trajds

from enum import Enum
from typing import Optional, Tuple

class Archi(Enum):
    DTFNet = 0
    PIPS = 1
    PIPS2 = 2
    TAPNET = 3
    TAPIR = 4
    RAFT = 5
    FLOWFORMER = 6
    FLOWFORMERPP = 7
    MFT = 8
    COTRACKER = 9

class Dataset(Enum):
    DAVIS = 0
    KUBRIC = 1
    RGB_STACKING = 2
    KINETICS = 3
    SINTEL = 4
    FLYINGTHINGS = 5
    KUBRIC_DTF = 6

class OFCropStrategy(Enum):
    All = 0
    Causal = 1
    Centered = 2
    Future = 3

torch.set_grad_enabled(False)

def hsv_to_rgb( hsv ):
    """
    Convert a color array from HSL to RGB.
    Input should have the shape [...,3] and in the range [0,1] for all 3 channels
    """
    h, s, v = hsv[...,0], hsv[...,1], hsv[...,2]
    if hsv.shape[-1] == 4: # Alpha channel
        a = hsv[...,3]
    c = v*s
    h_ = h*6.
    x = c * ( 1 - np.abs((h_%2) - 1) )

    cad0 = np.logical_and(h_ >= 0,h_ < 1)
    cad1 = np.logical_and(h_ >= 1,h_ < 2)
    cad2 = np.logical_and(h_ >= 2,h_ < 3)
    cad3 = np.logical_and(h_ >= 3,h_ < 4)
    cad4 = np.logical_and(h_ >= 4,h_ < 5)
    cad5 = np.logical_and(h_ >= 5,h_ < 6)

    rgb = np.zeros_like(hsv)
    rgb[cad0,0] = c[cad0] ; rgb[cad0,1] = x[cad0]
    rgb[cad1,0] = x[cad1] ; rgb[cad1,1] = c[cad1]
    rgb[cad2,1] = c[cad2] ; rgb[cad2,2] = x[cad2]
    rgb[cad3,1] = x[cad3] ; rgb[cad3,2] = c[cad3]
    rgb[cad4,2] = c[cad4] ; rgb[cad4,0] = x[cad4]
    rgb[cad5,2] = x[cad5] ; rgb[cad5,0] = c[cad5]

    rgb += (v-c)[...,None]
    if hsv.shape[-1] == 4:
        rgb[...,3] = a

    return rgb

def line_alpha( img, pt0, pt1, color ):
    H, W, _ = img.shape
    if len(color) == 3:
        return cv2.line( img, pt0, pt1,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA )
    else:
        min_x, max_x = min(pt0[0],pt1[0]), max(pt0[0],pt1[0])
        min_y, max_y = min(pt0[1],pt1[1]), max(pt0[1],pt1[1])
        min_x, max_x = max(0,min(W-1,min_x)), max(0,min(W-1,max_x))
        min_y, max_y = max(0,min(H-1,min_y)), max(0,min(H-1,max_y))
        rect = img[min_y:max_y+1,min_x:max_x+1,:].copy()

        alpha = color[-1]
        img = cv2.line( img, pt0, pt1,
            color=color[:-1],
            thickness=1,
            lineType=cv2.LINE_AA )
        img[min_y:max_y+1,min_x:max_x+1,:] = \
                alpha * img[min_y:max_y+1,min_x:max_x+1,:] + \
                (1-alpha) * rect

        return img

def batch_list(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def mkdir_p(path,exist_ok=True):
    """
    Recursively creates a directory, as the 'mkdir -p' command
    Will not raise an exception if the directory exists, and exist_ok is True
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and exist_ok:
            pass
        else:
            raise

def inclusive_range(a,b):
    if a < b:
        return range(a,b+1)
    else:
        return range(a,b-1,-1)


def load_model(
        model_path: os.PathLike,
        model_archi: Archi = Archi.DTFNet,
        device: str = "cuda",
        iters: int = 12 ):
    """
    Generic function to load a model, given its architecture
    """
    if model_archi == Archi.DTFNet:
        net = dtfnet.DtfNet()
        state_dict = torch.load(model_path)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # Fix 'model.' prefix in state_dict
        fixed_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                key = k[len("module."):]
            else:
                key = k
            fixed_state_dict[key] = v
        net.load_state_dict(fixed_state_dict)
        net = net.to(device)
        net.requires_grad_(False)
        net.eval()
        return net
    elif model_archi == Archi.TAPIR:
        from tapnet.torch import tapir_model

        model = tapir_model.TAPIR(pyramid_level=1)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        net.requires_grad_(False)
        model.eval()

        return model
    elif model_archi == Archi.RAFT:
        import dtf_core.networks.raft_wrapper as raft
        net = torch.nn.DataParallel(raft.RAFTWrapper(small=False))
        state_dict = torch.load(model_path)
        net.load_state_dict(state_dict)
        net = net.module
        net = net.to(device)
        net.requires_grad_(False)
        net.eval()
        return net
    elif model_archi == Archi.PIPS:
        from nets.pips import Pips

        net = Pips(stride=4)
        state_dict = torch.load(model_path)
        net.load_state_dict(state_dict["model_state_dict"])
        net = net.to(device)
        net.requires_grad_(False)
        net.eval()
        return net
    elif model_archi == Archi.PIPS2:
        from nets.pips2 import Pips

        net = Pips(stride=8)
        state_dict = torch.load(model_path)
        net.load_state_dict(state_dict["model_state_dict"])
        net = net.to(device)
        net.requires_grad_(False)
        net.eval()
        return net
    elif model_archi == Archi.MFT:
        from MFT.config import load_config

        config = load_config(model_path)
        tracker = config.tracker_class(config)
        if device == "cuda":
            os.environ['CUDA_VISIBLE_DEVICES'] = device
        # According to MFT's demo code, it seems like there is nothing else to do
        # (eval, ...) ?
        return tracker
    elif model_archi == Archi.FLOWFORMER:
        from configs.submission import get_cfg
        from core.FlowFormer import build_flowformer
        cfg = get_cfg()
        cfg.model = model_path
        cfg.decoder_depth = iters
        model = torch.nn.DataParallel(build_flowformer(cfg))
        model.load_state_dict(torch.load(cfg.model))

        model = model.module
        model = model.to(device)
        model.eval()

        return model
    elif model_archi == Archi.COTRACKER:
        from cotracker.predictor import CoTrackerPredictor

        model = CoTrackerPredictor( checkpoint=model_path )
        model.to(device)
        model.eval()

        return model
    else:
        raise NotImplementedError()

def query_dtf( queries, dtf, vis, t_ref ):
    """
    Query a DTF (and visbility) built at t_ref, at several positions,
    to reconstruct the trajectory of the queried points
    """
    B, T, C, H, W = dtf.shape
    assert B==1
    q_idx = np.where(queries[0,:,0].astype(np.int64)==t_ref)[0]
    # It seems like the subpix convention for trajectories is to sample
    # integer coordinates in the middle of pixels
    # But in our DTF definition, the center of pixels are the .5 coordinate,
    # (we align with the corners), which is why we add (and further remove)
    # this 0.5
    t_queries = torch.from_numpy(queries[0][q_idx])+0.5
    t_queries = (t_queries[:,[2,1]]) / torch.tensor([W,H],dtype=torch.float).unsqueeze(0)*2 - 1.
    t_queries = t_queries.unsqueeze(0).to( dtf.dtype ).to( dtf.device )

    dtf = einops.rearrange( dtf, "b t c h w -> b (t c) h w" )
    traj = torch.nn.functional.grid_sample(
        dtf,
        t_queries.unsqueeze(0),
        padding_mode="border",
        align_corners=False,
        )
    traj = einops.rearrange( traj, "1 (t c) 1 n -> n t c", t=T, c=C ) - 0.5

    if vis is not None:
        vis = einops.rearrange( vis, "b t c h w -> b (t c) h w" )
        vis = torch.nn.functional.grid_sample(
            vis,
            t_queries.unsqueeze(0),
            padding_mode="zeros",
            align_corners=False,
            )
        vis = vis[0,:,0,:].permute(1,0)

    return traj, vis, q_idx

def evaluate_model_traj(
        model,
        model_archi,
        dataset,
        ds_name=None,
        query_mode="first",
        max_seq_len=None,
        device="cuda",
        batch_size=1,
        iters: int = 12,
        chain_len: Optional[int] = None,
        nb_max_samples: Optional[int] = None,
        resize_infer: Optional[Tuple[int,int]] = None,
        ):
    scores = {}
    nb_samples = 0
    nb_trajs = 0
    nb_total_evals = 0
    for i, sample in enumerate(dataset):
        if nb_max_samples is not None and i >= nb_max_samples:
            break
        print( f"  Sample: {nb_samples}", end='\r' )
        sample_ = sample[ds_name]
        video = sample_["video"]
        queries = sample_["query_points"]
        gt_traj = sample_["target_points"]
        gt_occ = sample_["occluded"]
        assert video.shape[0] == 1

        if max_seq_len is not None:
            video = video[:,:max_seq_len]
            valid_queries = np.where( queries[0,:,0] < max_seq_len )[0]
            queries = np.ascontiguousarray(queries[:,valid_queries])
            gt_traj = np.ascontiguousarray(gt_traj[:,valid_queries,:max_seq_len])
            gt_occ = np.ascontiguousarray(gt_occ[:,valid_queries,:max_seq_len])

        torch_video = torch.from_numpy(video).to(device).permute(0,1,4,2,3)
        # The TAP-Net dataloader returns float arrays in [-1,1]
        torch_video = (torch_video+1.)*(255./2.)
        B, T, C, H, W = torch_video.shape
        dtype = torch_video.dtype

        if model_archi in [Archi.DTFNet,Archi.RAFT,Archi.FLOWFORMER]:
            # Model works on [0,255] images

            # DTF are evaluated at one single time ref
            # When several times are queried, we need to run it several times
            trefs = sorted(list(set( queries[:,:,0].ravel().astype(np.int64) )))
            pred_traj = np.empty_like(gt_traj)
            pred_occ = np.empty_like(gt_occ)
            for tref in trefs:
                dtf, vis = compute_dtf( model, model_archi,
                        torch_video.squeeze(0),
                        tref,
                        batch_size=batch_size,
                        chain_len=chain_len,
                        iters=iters,
                        resize_infer=resize_infer,
                        )

                # Select concerned trajectories
                t_pred_traj, t_pred_vis, q_idx = query_dtf(
                    queries,
                    dtf.unsqueeze(0),
                    vis.unsqueeze(0) if vis is not None else None,
                    tref,
                    )
                pred_traj[0,q_idx] = t_pred_traj.detach().contiguous().cpu().numpy()

                if vis is not None:
                    pred_occ[0,q_idx] = (t_pred_vis<0.5).detach().contiguous().cpu().numpy()

            if model_archi in [Archi.RAFT,Archi.FLOWFORMER]:
                # Does not predic occlusion
                pred_occ = gt_occ
        elif model_archi == Archi.PIPS:
            # Model seem to work on [0,255] images,
            # but apparently video needs to be padded ??
            B, T, C, H, W = torch_video.shape
            pad = 0
            if pad > 0:
                pad_video = einops.rearrange( torch_video, "b t c h w -> (b t) c h w" )
                pad_video = F.pad( pad_video, (pad,pad,pad,pad), mode='constant', value=0. )
                pad_video = einops.rearrange( pad_video, "(b t) c h w -> b t c h w",
                                        b=B, t=T )
                B, T, C, H, W = pad_video.shape
            else:
                pad_video = torch_video

            trefs = sorted(list(set( queries[:,:,0].ravel().astype(np.int64) )))
            pred_traj = np.empty_like(gt_traj)
            pred_occ = np.empty_like(gt_occ)
            for tref in trefs:
                # Build query particles:
                q_idx = np.where(queries[0,:,0].astype(np.int64)==tref)[0]
                for b_qidx in batch_list(q_idx,batch_size):
                    t_queries = queries[0,b_qidx]
                    qpos = t_queries[:,[2,1]]
                    N = qpos.shape[0]
                    trajs = torch.zeros((T,N,2),dtype=dtype,device=device)
                    occs = torch.zeros((T,N),dtype=bool,device=device)
                    current_pos = torch.from_numpy(qpos).to(dtype).to(device)
                    trajs[tref] = current_pos

                    # Build trajectory forward
                    for t0 in range(tref,T-1,model.S-1):
                        local_seq = pad_video[:,t0:t0+model.S]
                        T_ = local_seq.shape[1]
                        if T_ < model.S:
                            # PIPs only works on sequences of model.S frames
                            # so we pad (border) the sequence
                            local_seq = torch.cat((
                                local_seq, local_seq[:,[-1]].repeat(1,model.S-T_,1,1,1)),
                                dim=1 )

                        # Run the model
                        local_traj, _, local_vis, _ = model(
                                (current_pos+pad).unsqueeze(0),
                                local_seq,
                                iters=iters )
                        local_traj = local_traj[-1].squeeze(0)[:T_] - pad
                        local_vis = local_vis.squeeze(0)[:T_]

                        trajs[t0+1:t0+model.S] = local_traj[1:]
                        occs[t0+1:t0+model.S] = (local_vis<0)[1:]

                        current_pos = local_traj[-1]

                    # Trajectory backward
                    if tref > 0:
                        current_pos = torch.from_numpy(qpos).to(dtype).to(device)
                        for t0 in range(tref,0,-(model.S-1)):
                            backward_t = list(range(t0,max(-1,t0-model.S),-1))
                            local_seq = pad_video[:,backward_t]
                            T_ = local_seq.shape[1]
                            if T_ < model.S:
                                # PIPs only works on sequences of model.S frames
                                # so we pad (border) the sequence
                                local_seq = torch.cat((
                                    local_seq, local_seq[:,[-1]].repeat(1,model.S-T_,1,1,1)),
                                    dim=1 )

                            # Run the model
                            local_traj, _, local_vis, _ = model(
                                    (current_pos+pad).unsqueeze(0),
                                    local_seq,
                                    iters=iters )
                            local_traj = local_traj[-1].squeeze(0)[:T_] - pad
                            local_vis = local_vis.squeeze(0)[:T_]

                            trajs[backward_t[1:]] = local_traj[1:]
                            occs[backward_t[1:]] = (local_vis<0)[1:]

                            current_pos = local_traj[-1]

                    # Store in final result
                    pred_traj[0,b_qidx,:,:] = trajs.permute(1,0,2).detach().contiguous().cpu().numpy()
                    pred_occ[0,b_qidx,:] = occs.permute(1,0).detach().contiguous().cpu().numpy()
        elif model_archi in [Archi.PIPS2]:
            # Model seem to work on [0,255] images,
            B, T, C, H, W = torch_video.shape

            trefs = sorted(list(set( queries[:,:,0].ravel().astype(np.int64) )))
            pred_traj = np.empty_like(gt_traj)
            pred_occ = gt_occ # is not predicted by PIPs++
            if chain_len is None:
                chain_len_ = T
            else:
                chain_len_ = chain_len
            for tref in trefs:
                # Build query particles:
                q_idx = np.where(queries[0,:,0].astype(np.int64)==tref)[0]
                for b_qidx in batch_list(q_idx,batch_size):
                    t_queries = queries[0,b_qidx]
                    qpos = t_queries[:,[2,1]]
                    N = qpos.shape[0]
                    trajs = torch.zeros((T,N,2),dtype=dtype,device=device)
                    current_pos = torch.from_numpy(qpos).to(dtype).to(device)
                    trajs[tref] = current_pos

                    # Run the model forward
                    for t0 in range(tref,T-1,chain_len_-1):
                        if model_archi in [Archi.PIPS2]:
                            init_trajs = einops.repeat(current_pos,"n c -> 1 t n c",
                                                       t=min(chain_len_,T-t0))
                            local_traj, _, _, _ = model(
                                    init_trajs, # B T N 2
                                    torch_video[:,t0:t0+chain_len_],
                                    iters=iters,
                                    feat_init=None)
                            local_traj = local_traj[-1].squeeze(0)

                        trajs[t0+1:t0+chain_len_] = local_traj[1:]
                        current_pos = local_traj[-1]

                    # Run the model backward
                    if tref > 0:
                        current_pos = torch.from_numpy(qpos).to(dtype).to(device)
                        for t0 in range(tref,0,-(chain_len_-1)):
                            backward_t = list(range(t0,max(-1,t0-chain_len_),-1))
                            backward_seq = torch_video[:,backward_t]

                            if model_archi in [Archi.PIPS2]:
                                init_trajs = einops.repeat(current_pos,"n c -> 1 t n c",t=min(t0+1,chain_len_))
                                local_traj, _, _, _ = model(
                                        init_trajs, # B T N 2
                                        backward_seq,
                                        iters=iters,
                                        feat_init=None)
                                local_traj = local_traj[-1].squeeze(0)
                            else:
                                import ipdb; ipdb.set_trace()

                            trajs[backward_t[1:]] = local_traj[1:]
                            current_pos = local_traj[-1]

                    # Store in final result
                    pred_traj[0,b_qidx,:,:] = trajs.permute(1,0,2).detach().contiguous().cpu().numpy()
        elif model_archi in [Archi.TAPIR]:
            # Can process backward + forward at the same time
            vid = torch_video*(2./255.)-1.
            vid = einops.rearrange( vid, "b t c h w -> b t h w c" )
            q = torch.from_numpy(queries).detach().to(device)
            outputs = model( vid, q )
            pred_traj, pred_occ, pred_dist = \
                outputs['tracks'], outputs['occlusion'], outputs['expected_dist']
            pred_vis = (1 - F.sigmoid(pred_occ)) * (1 - F.sigmoid(pred_dist)) > 0.5
            pred_occ = torch.logical_not(pred_vis)

            pred_traj = pred_traj.detach().contiguous().cpu().numpy()
            pred_occ = pred_occ.detach().contiguous().cpu().numpy()
        elif model_archi == Archi.MFT:
            from MFT.point_tracking import convert_to_point_tracking

            # Model seems to work on [0,255] images,
            _, T, H, W, C = video.shape
            video_ = (video+1.)*(255./2.)

            trefs = sorted(list(set( queries[0,:,0].ravel().astype(np.int64) )))
            pred_traj = np.empty_like(gt_traj)
            pred_occ = np.empty_like(gt_occ)
            for tref in trefs:
                q_idx = np.where(queries[0,:,0].astype(np.int64)==tref)[0]
                xys = np.stack((queries[0,q_idx,2],queries[0,q_idx,1]), axis=1)
                xys = torch.from_numpy(xys).to(torch.float32)

                skip_forward = False
                if tref >= T-1:
                    skip_forward = True
                else:
                    # Forward prediction
                    skip_forward = False
                    for t in range(tref,T):
                        if t == tref:
                            meta = model.init(video_[0,t])
                        else:
                            meta = model.track(video_[0,t])
                        coords, occlusions = convert_to_point_tracking(meta.result, xys)
                        pred_traj[0,q_idx,t] = coords
                        pred_occ[0,q_idx,t] = (occlusions >= 0.5)

                if tref >= 1:
                    # Backward prediction
                    for t in range(tref,-1,-1):
                        if t == tref:
                            meta = model.init(video[0,t])
                        else:
                            meta = model.track(video[0,t])

                        if t != tref or skip_forward:
                            coords, occlusions = convert_to_point_tracking(meta.result, xys)
                            pred_traj[0,q_idx,t] = coords
                            pred_occ[0,q_idx,t] = (occlusions >= 0.5)
        elif model_archi == Archi.COTRACKER:
            # Works on [0,255] float images

            q = torch.from_numpy(queries)[:,:,[0,2,1]].detach().float().contiguous().to(device)
            pred_traj, pred_vis = model(
                    torch_video,
                    queries=q,
                    backward_tracking=True )
            pred_traj = pred_traj.permute(0,2,1,3)
            pred_occ = torch.logical_not(pred_vis.permute(0,2,1))

            pred_traj = pred_traj.contiguous().detach().cpu().numpy()
            pred_occ = pred_occ.contiguous().detach().cpu().numpy()
        else:
            raise NotImplementedError()

        # Visualization
        if False:
            import matplotlib.pyplot as plt
            import cv2

            def traj_img( img, queries, trajs, occ=None ):
                # Darken image for better traj visu
                img = img*(0.7/255.)
                N, T, _ = trajs.shape

                for query in queries:
                    cv2.drawMarker( img, (int(query[2]),int(query[1])), (1.,1.,1.),
                        markerType = cv2.MARKER_CROSS,
                        markerSize = 3 )

                for n, (query, traj) in enumerate(zip(queries,trajs)):
                    t0 = query[0]
                    for t in range(T-1):
                        if occ is not None:
                            if occ[n,t] or occ[n,t+1]:
                                continue
                        dt = abs(t-t0)
                        color = hsv_to_rgb( np.array(
                            [ n/N, 0.4+0.6*dt/T, 0.9 ] ))
                        cv2.line( img, (int(traj[t,0]),int(traj[t,1])), (int(traj[t+1,0]),int(traj[t+1,1])),
                                 color=color, thickness=1, lineType=cv2.LINE_AA )

                return img

            # First image of the sequence
            plt.subplot(121)
            np_img = torch_video[0,-1].permute(1,2,0).detach().contiguous().cpu().numpy()
            pred_img = traj_img(np_img,queries[0],pred_traj[0],pred_occ[0])
            plt.imshow( pred_img )
            plt.title( "Prediction" )
            plt.subplot(122)
            gt_img = traj_img(np_img,queries[0],gt_traj[0],gt_occ[0])
            plt.imshow( gt_img )
            plt.title( "Ground truth" )

            plt.show()

        score = evd.compute_tapvid_metrics(
                queries,
                gt_occ, gt_traj,
                pred_occ, pred_traj,
                query_mode=query_mode,
                )
        for k, v in score.items():
            if k in scores.keys():
                scores[k] += v.item() #* gt_traj.shape[1]
            else:
                if model_archi in [Archi.RAFT,Archi.FLOWFORMER,Archi.PIPS2]:
                    # Occlusion not predicted
                    if "jaccard" in k or "occlusion" in k:
                        continue
                scores[k] = v.item() #* gt_traj.shape[1]

        # For final normalization
        nb_samples += 1
        nb_trajs += gt_traj.shape[1]
        nb_total_evals += gt_traj.shape[1]*gt_traj.shape[2]
    print( "" )

    for k in scores:
        scores[k] /= nb_samples
        #scores[k] /= nb_trajs
        #scores[k] /= nb_total_evals

    return scores

def crop_flow_sequence(
        imgs, flows, occs,
        crop_strategy: OFCropStrategy = OFCropStrategy.All,
        crop_window_size: int = 6,
        ):
    T, C, H, W = imgs.shape
    device = imgs.device
    dtype = imgs.dtype

    if crop_strategy == OFCropStrategy.All:
        all_seq = einops.repeat(imgs,"t c h w -> n t c h w",n=T-1)
        all_flows = flows
        all_occs = occs
        all_idx = torch.arange(T-1,dtype=torch.long,device=device)
        return all_seq, all_flows, all_occs, all_idx
    elif crop_strategy == OFCropStrategy.Causal:
        all_seq = []
        all_flows = []
        all_occs = []
        all_idx = []
        for t_ref in range(crop_window_size-2,T-1):
            all_seq.append(imgs[t_ref-crop_window_size+2:t_ref+2])
            all_flows.append( flows[t_ref] )
            if occs is not None:
                all_occs.append( occs[t_ref] )
            all_idx.append(crop_window_size-2)

        all_seq = torch.stack(all_seq,dim=0)
        all_flows = torch.stack(all_flows,dim=0)
        all_occs = torch.stack(all_occs,dim=0) if occs is not None else None
        all_idx = torch.tensor(all_idx,dtype=torch.long,device=device)

        return all_seq, all_flows, all_occs, all_idx
    elif crop_strategy == OFCropStrategy.Centered:
        all_seq = []
        all_flows = []
        all_occs = []
        all_idx = []
        for t_ref in range(T-1):
            t0 = max(0,t_ref-(crop_window_size-1)//2)
            t1 = min(T,t0+crop_window_size)
            t0 = max(0,t1-crop_window_size)
            all_seq.append( imgs[t0:t1] )
            all_flows.append( flows[t_ref] )
            if occs is not None:
                all_occs.append( occs[t_ref] )
            all_idx.append(t_ref-t0)
        all_seq = torch.stack(all_seq,dim=0)
        all_flows = torch.stack(all_flows,dim=0)
        all_occs = torch.stack(all_occs,dim=0) if occs is not None else None
        all_idx = torch.tensor(all_idx,dtype=torch.long,device=device)

        return all_seq, all_flows, all_occs, all_idx
    elif crop_strategy == OFCropStrategy.Future:
        all_seq = []
        all_flows = []
        all_occs = []
        all_idx = []
        for t_ref in range(T-crop_window_size):
            all_seq.append(imgs[t_ref:t_ref+crop_window_size])
            all_flows.append( flows[t_ref] )
            if occs is not None:
                all_occs.append( occs[t_ref] )
            all_idx.append(0)

        all_seq = torch.stack(all_seq,dim=0)
        all_flows = torch.stack(all_flows,dim=0)
        all_occs = torch.stack(all_occs,dim=0) if occs is not None else None
        all_idx = torch.tensor(all_idx,dtype=torch.long,device=device)

        return all_seq, all_flows, all_occs, all_idx
    else:
        raise NotImplementedError()

def evaluate_model_optical_flow(
        model,
        model_archi: Archi,
        dataset: ltfds.LTFlowDataset,
        crop_strategy: OFCropStrategy = OFCropStrategy.All,
        crop_window_size: int = 6, # When using centered
        max_seq_len: Optional[int] = None,
        device: str = "cuda",
        batch_size: int = 1,
        downscale_max_dim: Optional[int] = None,
        iters: int = 12,
        dtf_grid_size: int = 1,
        nb_max_samples: Optional[int] = None,
        resize_infer: Optional[Tuple[int,int]] = None,
        ):
    nb_seq = 0
    nb_samples = 0
    nb_occ_pixels = 0
    nb_aae_pixels = 0
    nb_occ_aae_pixels = 0
    all_scores = {
            "AEPE"     : 0.,
            "AAE"      : 0.,
            "F-1px"    : 0.,
            "F-3px"    : 0.,
            "F-5px"    : 0.,
            "Occ-AEPE" : 0.,
            "Occ-AAE"  : 0.,
            "Occ-F-1px": 0.,
            "Occ-F-3px": 0.,
            "Occ-F-5px": 0.,
            }
    for i, (seq, flows, occs) in enumerate(dataset):
        if nb_max_samples is not None and i >= nb_max_samples:
            break
        print( f"  Sequence: {nb_seq}", end='\r' )
        T, C, H, W = seq.shape
        dtype = seq.dtype
        if model_archi == Archi.MFT:
            # MFT API is numpy (CPU) arrays only
            device="cpu"

        # Downscale the image and flows if too big
        if downscale_max_dim is not None:
            scale = downscale_max_dim / max(H,W)
            if scale < 1.:
                new_H = 8*int(math.floor(H*scale/8))
                new_W = 8*int(math.floor(W*scale/8))
                resize = torchvision.transforms.Resize((new_H,new_W),antialias=False)
                seq = resize(seq)
                flows = resize(flows)
                if occs is not None:
                    occs = resize(occs.unsqueeze(1)).squeeze(1)
                    #occs = (occs > 0.5)
                scale_xy = torch.tensor([new_W/W,new_H/H],dtype=dtype,device=seq.device)
                flows *= scale_xy[None,:,None,None]

                H, W = new_H, new_W
        else:
            scale = 1.

        all_seq, all_flows, all_occs, all_idx = crop_flow_sequence(seq,flows,occs,
            crop_strategy = crop_strategy,
            crop_window_size = crop_window_size )

        pred_flows = torch.zeros((len(all_seq),2,H,W),dtype=dtype)

        if model_archi in [Archi.DTFNet,Archi.RAFT,Archi.FLOWFORMER]:
            if resize_infer is not None:
                rH, rW = resize_infer
                oH, oW = H, W # original size
                resize_training = torchvision.transforms.Resize((rH,rW))
                resize_original = torchvision.transforms.Resize((H,W))
                gH, gW = rH, rW
            else:
                gH, gW = H, W

            grid = torch.stack( torch.meshgrid((
                    torch.arange(gW,dtype=dtype,device=device)+0.5,
                    torch.arange(gH,dtype=dtype,device=device)+0.5 ),
                    indexing='xy'),
                    dim=0 )

            # Batch over frames
            for b in range(0,all_seq.shape[0],batch_size):
                b_seq = all_seq[b:b+batch_size].to(device)
                b_flows = all_flows[b:b+batch_size].to(device)
                #b_occs = all_occs[b:b+batch_size].to(device) if occs is not None else None
                b_idx = all_idx[b:b+batch_size].to(device)
                B = b_seq.shape[0]
                b_range = torch.arange(B,device=device)

                if resize_infer is not None:
                    b_seq = einops.rearrange( b_seq, "b t c h w -> (b t) c h w" )
                    b_seq = resize_training(b_seq)
                    b_seq = einops.rearrange( b_seq, "(b t) c h w -> b t c h w", b=B )

                if model_archi == Archi.DTFNet:
                    dtf, vis = model(b_seq,b_idx)
                    # Forward optical-flow is DTF from t_ref to t_ref+1
                    res = dtf[b_range,-1,b_idx+1]-grid.unsqueeze(0)
                elif model_archi in [Archi.RAFT]:
                    b_range = torch.arange(b_seq.shape[0],device=device)
                    res = model( b_seq[b_range,b_idx], b_seq[b_range,b_idx+1],
                               iters=iters )[-1]
                elif model_archi in [Archi.FLOWFORMER]:
                    b_range = torch.arange(b_seq.shape[0],device=device)
                    res = model( b_seq[b_range,b_idx], b_seq[b_range,b_idx+1] )[0]

                if resize_infer is not None:
                    res = resize_original(res)*torch.tensor(
                            [oW/rW,oH/rH],device=device)[None,:,None,None]

                pred_flows[b:b+batch_size] = res
        elif model_archi in [Archi.PIPS,Archi.PIPS2,Archi.TAPIR,Archi.COTRACKER]:
            # Batch over number of queries
            if model_archi == Archi.PIPS:
                assert crop_strategy == OFCropStrategy.Future and crop_window_size == model.S, \
                    f"PIPs only works with 'future' cropping strategy, with a window size of {model.S}"
            elif model_archi == Archi.PIPS2:
                assert crop_strategy == OFCropStrategy.Future, \
                    f"PIPs++ only works with 'future' cropping strategy"

            grid = torch.stack( torch.meshgrid((
                    torch.arange(W,dtype=dtype,device=device)+0.5,
                    torch.arange(H,dtype=dtype,device=device)+0.5 ),
                    indexing='xy'),
                    dim=0 )

            for i, (seq, idx) in enumerate(zip(all_seq,all_idx)):
                dtf, vis = compute_dtf(
                        model, model_archi,
                        seq.to(device), idx.to(device),
                        batch_size=batch_size,
                        iters=iters,
                        chain_len=None,
                        dtf_grid_size=dtf_grid_size,
                        resize_infer=resize_infer )
                pred_flows[i] = (dtf[idx+1]-grid).detach().contiguous().cpu()
        elif model_archi == Archi.MFT:
            from MFT.point_tracking import convert_to_point_tracking
            grid = torch.stack( torch.meshgrid((
                    torch.arange(W,dtype=dtype,device=device)+0.5,
                    torch.arange(H,dtype=dtype,device=device)+0.5 ),
                    indexing='xy'),
                    dim=0 )
            xys = einops.rearrange( grid, "c h w -> (h w) c" )
            for i, (seq, idx) in enumerate(zip(all_seq,all_idx)):
                model.init(seq[idx].permute(1,2,0).detach().contiguous().cpu().numpy())
                meta = model.track(seq[idx+1].permute(1,2,0).detach().contiguous().cpu().numpy())
                coords, occlusions = convert_to_point_tracking(meta.result, xys)
                coords = einops.rearrange( torch.from_numpy(coords), "(h w) c -> c h w", h=H, w=W )
                pred_flows[i] = coords-grid
        else:
            raise NotImplementedError()

        # Compute losses
        diff = (pred_flows-all_flows).norm(p=2,dim=1)
        all_scores["AEPE"] += diff.sum()/(H*W)
        all_scores["F-1px"] += (diff < 1.).sum()/(H*W)
        all_scores["F-3px"] += (diff < 3.).sum()/(H*W)
        all_scores["F-5px"] += (diff < 5.).sum()/(H*W)

        dot = torch.einsum("t c h w, t c h w -> t h w",pred_flows,all_flows)
        pred_norm = pred_flows.norm(p=2,dim=1)
        gt_norm = all_flows.norm(p=2,dim=1)
        # Don't consider flows whose norm is too small to compute relevant angle
        # (this also leads to dividing by 0)
        valid_aae = torch.logical_and(pred_norm > 0.3, gt_norm > 0.3)
        if valid_aae.any():
            aae_cos = dot / torch.sqrt(pred_norm*gt_norm).clamp(min=1e-4)
            aae = torch.acos( aae_cos[valid_aae].clamp(-1.,1.) ) * 180./math.pi
            all_scores["AAE"] += aae.sum()
            nb_aae_pixels += valid_aae.sum()

            if occs is not None:
                occluded = (all_occs>0.5)
                valid_aae_occ = torch.logical_and(occluded,valid_aae)
                nb_occ_pixels += occluded.sum()
                occ_diff = diff[occluded]
                all_scores["Occ-AEPE"] += occ_diff.sum()
                if valid_aae_occ.any():
                    occ_aae = torch.acos( aae_cos[valid_aae_occ].clamp(-1.,1.) ) * 180./math.pi
                    all_scores["Occ-AAE"] += occ_aae.sum()
                    nb_occ_aae_pixels += valid_aae_occ.sum()
                all_scores["Occ-F-1px"] += (occ_diff < 1.).sum()
                all_scores["Occ-F-3px"] += (occ_diff < 3.).sum()
                all_scores["Occ-F-5px"] += (occ_diff < 5.).sum()

        # Visualization
        if False:
            import matplotlib.pyplot as plt
            from dtf_core import img_utils

            img0 = all_seq[0,all_idx[0]].detach().permute(1,2,0).contiguous().cpu().numpy()
            img1 = all_seq[0,all_idx[0]+1].detach().permute(1,2,0).contiguous().cpu().numpy()
            pred_flow_img = img_utils.flow_img(pred_flows[0],max_norm=40.*scale
                    ).detach().permute(1,2,0).contiguous().cpu().numpy()
            gt_flow_img = img_utils.flow_img(all_flows[0],max_norm=40.,wheel_size=24,
                    ).detach().permute(1,2,0).contiguous().cpu().numpy()
            plt.subplot(221)
            plt.imshow(img0/255.)
            plt.title("Image 0")
            plt.subplot(222)
            plt.imshow(img1/255.)
            plt.title("Image 1")
            plt.subplot(223)
            plt.imshow(pred_flow_img)
            plt.title("Prediction")
            plt.subplot(224)
            plt.imshow(gt_flow_img)
            plt.title("Ground-Truth")

            plt.show()

        nb_samples += len(all_seq)

        nb_seq += 1

    print("")
    for k in ["AEPE","F-1px","F-3px","F-5px"]:
        all_scores[k] /= nb_samples
    for k in ["Occ-AEPE","Occ-F-1px","Occ-F-3px","Occ-F-5px"]:
        all_scores[k] /= nb_occ_pixels
    all_scores["AAE"] /= nb_aae_pixels
    all_scores["Occ-AAE"] /= nb_occ_aae_pixels

    for k in ["F-1px","F-3px","F-5px",
              "Occ-F-1px","Occ-F-3px","Occ-F-5px"]:
        all_scores[k] = 100 * (1-all_scores[k])

    return all_scores

def _compute_dtf_contiguous(
        model,
        model_archi,
        seq: torch.Tensor,
        ref_idx: int = 0,
        batch_size: int = 1,
        iters: int = 12,
        dtf_grid_size: int = 1,
        resize_infer: Optional[Tuple[int,int]] = None,
        ):
    dtype = seq.dtype
    device = seq.device
    T, C, H, W = seq.shape

    if resize_infer is not None:
        rH, rW = resize_infer
        oH, oW = H, W # original size
        resize_training = torchvision.transforms.Resize((rH,rW))
        resize_original = torchvision.transforms.Resize((H,W))
        seq = resize_training(seq)
        H, W = rH, rW

    if model_archi == Archi.DTFNet:
        # Model works on [0,255] images

        # DTF are evaluated at one single time ref
        # When several times are queried, we need to run it several times
        dtf, vis = model( seq.unsqueeze(0), ref_idx )
        dtf = dtf.squeeze(0)[-1]
        vis = vis.squeeze(0)[-1]
    elif model_archi == Archi.PIPS:
        # Model seems to work on [0,255] images,
        # but apparently video needs to be padded ??
        pad = 0
        padded_seq = F.pad( seq, (pad,pad,pad,pad), mode='constant', value=0. )

        grid = torch.stack( torch.meshgrid((
                torch.arange(dtf_grid_size//2,W,dtf_grid_size,dtype=dtype,device=device),
                torch.arange(dtf_grid_size//2,H,dtf_grid_size,dtype=dtype,device=device) ),
                indexing='xy'),
                dim=2 )
        dH, dW, _ = grid.shape
        xys = einops.rearrange( grid, "h w c -> (h w) c" )
        dtf = torch.zeros((T,2,dH,dW),device=device,dtype=dtype)
        vis = torch.zeros((T,1,dH,dW),device=device,dtype=dtype)
        for xy in batch_list(xys,batch_size):
            xy_ = (xy+pad).unsqueeze(0)
            pred_trajs, pred_trajs2, pred_vis, _ = model(xy_,padded_seq.unsqueeze(0),iters=iters)
            pred_trajs = pred_trajs[-1].squeeze(0) - pad
            pred_vis = pred_vis.squeeze(0)

            xy_long = (xy/dtf_grid_size).to(torch.long)
            dtf[:,:,xy_long[:,1],xy_long[:,0]] = pred_trajs.permute(0,2,1)
            vis[:,:,xy_long[:,1],xy_long[:,0]] = torch.sigmoid(pred_vis.unsqueeze(1))
        dtf += 0.5
        if dtf_grid_size > 1:
            resize = torchvision.transforms.Resize([dH*dtf_grid_size,dW*dtf_grid_size])
            dtf = resize(dtf)[:,:,:H,:W]
            vis = resize(vis)[:,:,:H,:W]
    elif model_archi == Archi.PIPS2:
        grid = torch.stack( torch.meshgrid((
                torch.arange(dtf_grid_size//2,W,dtf_grid_size,dtype=dtype,device=device),
                torch.arange(dtf_grid_size//2,H,dtf_grid_size,dtype=dtype,device=device) ),
                indexing='xy'),
                dim=2 )
        dH, dW, _ = grid.shape
        xys = einops.rearrange( grid, "h w c -> (h w) c" )
        dtf = torch.zeros((T,2,dH,dW),device=device,dtype=dtype)
        for xy in batch_list(xys,batch_size):
            xy_ = einops.repeat( xy, "n c -> 1 t n c", t=T )
            pred_trajs, _, _, _ = model(
                    xy_,
                    seq.unsqueeze(0),
                    iters=iters )
            pred_trajs = pred_trajs[-1].squeeze(0)

            xy_long = (xy/dtf_grid_size).to(torch.long)
            dtf[:,:,xy_long[:,1],xy_long[:,0]] = pred_trajs.permute(0,2,1)
        dtf += 0.5
        if dtf_grid_size > 1:
            resize = torchvision.transforms.Resize([dH*dtf_grid_size,dW*dtf_grid_size])
            dtf = resize(dtf)[:,:,:H,:W]
        vis = None
    elif model_archi in [Archi.TAPIR]:
        # Resize the original video to 256x256, training size of TAPIR
        grid = torch.stack( torch.meshgrid((
                torch.arange(dtf_grid_size//2,W,dtf_grid_size,dtype=dtype,device=device),
                torch.arange(dtf_grid_size//2,H,dtf_grid_size,dtype=dtype,device=device) ),
                indexing='xy'),
                dim=2 )
        dH, dW, _ = grid.shape
        xys = einops.rearrange( grid, "h w c -> (h w) c" )
        dtf = torch.zeros((T,2,dH,dW),device=device,dtype=dtype)
        vis = torch.zeros((T,1,dH,dW),device=device,dtype=dtype)
        for xy in batch_list(xys,batch_size):
            vid = einops.rearrange( seq, "t c h w -> 1 t h w c" )
            q = torch.stack((ref_idx*torch.ones_like(xy[:,0]),xy[:,1],xy[:,0]),dim=-1)
            q = q.unsqueeze(0)
            outputs = model( vid, q )
            pred_traj, pred_occ, pred_dist = \
                outputs['tracks'][0], outputs['occlusion'][0], outputs['expected_dist'][0]
            pred_vis = (1 - F.sigmoid(pred_occ)) * (1 - F.sigmoid(pred_dist))

            xy_long = (xy/dtf_grid_size).to(torch.long)
            dtf[:,:,xy_long[:,1],xy_long[:,0]] = pred_traj.permute(1,2,0)
            vis[:,:,xy_long[:,1],xy_long[:,0]] = einops.rearrange( pred_vis, "n t -> t 1 n" )
        dtf += 0.5
        if dtf_grid_size > 1:
            resize = torchvision.transforms.Resize([dH*dtf_grid_size,dW*dtf_grid_size])
            dtf = resize(dtf)[:,:,:H,:W]
            vis = resize(vis)[:,:,:H,:W]
    elif model_archi in [Archi.RAFT,Archi.FLOWFORMER]:
        grid = torch.stack( torch.meshgrid((
                torch.arange(W,dtype=dtype,device=device)+0.5,
                torch.arange(H,dtype=dtype,device=device)+0.5 ),
                indexing='xy'),
                dim=0 )

        #  Construct the DTF for one reference time
        # (don't compute ref->ref flow)
        dtf = torch.zeros((T,2,H,W),dtype=dtype,device=device)
        ts = list(range(T))
        ts.remove(ref_idx)
        for t in batch_list(ts,batch_size):
            T_ = len(t)
            img0 = einops.repeat( seq[ref_idx], "c h w -> t c h w", t=T_ )
            img1 = seq[t]
            if model_archi in [Archi.RAFT]:
                all_pred_flows = model(img0,img1,iters=iters)[-1]
            elif model_archi in [Archi.FLOWFORMER]:
                all_pred_flows = model(img0,img1)[0]
            else:
                raise NotImplementedError()
            dtf[t] = all_pred_flows

        dtf += grid[None,:,:,:]
        vis = None
    elif model_archi == Archi.MFT:
        from MFT.point_tracking import convert_to_point_tracking
        #grid = np.stack( np.meshgrid(
        #        np.arange(dtf_grid_size//2,W,dtf_grid_size,dtype=np.float32)+0.5,
        #        np.arange(dtf_grid_size//2,H,dtf_grid_size,dtype=np.float32)+0.5,
        #        indexing='xy'),
        #        axis=2 )
        grid = torch.stack( torch.meshgrid((
                torch.arange(dtf_grid_size//2,W,dtf_grid_size,dtype=dtype,device=device)+0.5,
                torch.arange(dtf_grid_size//2,H,dtf_grid_size,dtype=dtype,device=device)+0.5 ),
                indexing='xy'),
                dim=2 )
        dH, dW, _ = grid.shape
        xys = einops.rearrange( grid, "h w c -> (h w) c" )
        xy_long = (xys/dtf_grid_size).to(torch.long).cpu().numpy()
        dtf = np.zeros((T,dH,dW,2),np.float32)
        vis = np.zeros((T,dH,dW),np.float32)

        np_seq = seq.permute(0,2,3,1).contiguous().detach().cpu().numpy()

        for t in range(T):
            if t == 0:
                meta = model.init(np_seq[t])
            else:
                meta = model.track(np_seq[t])
            coords, occlusions = convert_to_point_tracking(meta.result, xys)

            dtf[t,xy_long[:,1],xy_long[:,0],:] = coords
            vis[t,xy_long[:,1],xy_long[:,0]] = 1.-occlusions

        dtf = torch.from_numpy(dtf).to(device).permute(0,3,1,2)
        vis = torch.from_numpy(vis).to(device).unsqueeze(1)
        if dtf_grid_size > 1:
            resize = torchvision.transforms.Resize([dH*dtf_grid_size,dW*dtf_grid_size])
            dtf = resize(dtf)[:,:,:H,:W]
            vis = resize(vis)[:,:,:H,:W]
    elif model_archi == Archi.COTRACKER:
        # Model runs on [0,255] float videos
        if dtf_grid_size == 1:
            pred_traj, pred_vis = model(
                    seq.unsqueeze(0),
                    grid_query_frame=ref_idx,
                    backward_tracking=True )
            xy_long = pred_traj[0,ref_idx,:,:].to(torch.long)
            dtf = torch.zeros((T,2,H,W),dtype=torch.float32,device=device)
            vis = torch.zeros((T,1,H,W),dtype=torch.float32,device=device)
            dtf[:,:,xy_long[:,1],xy_long[:,0]] = einops.rearrange( pred_traj, "1 t n c -> t c n" )
            vis[:,:,xy_long[:,1],xy_long[:,0]] = einops.rearrange( pred_vis, "1 t n -> t 1 n" ).to(torch.float32)
        else:
            grid = torch.stack( torch.meshgrid((
                    torch.arange(dtf_grid_size//2,W,dtf_grid_size,dtype=dtype,device=device),
                    torch.arange(dtf_grid_size//2,H,dtf_grid_size,dtype=dtype,device=device) ),
                    indexing='xy'),
                    dim=2 )

            dH, dW, _ = grid.shape
            xys = einops.rearrange( grid, "h w c -> (h w) c" )
            queries = torch.concatenate((ref_idx*torch.ones_like(xys[:,[0]]),xys),dim=-1).unsqueeze(0)

            dtf, vis = model(
                    seq.unsqueeze(0),
                    queries=queries,
                    backward_tracking=True )

            dtf = einops.rearrange( dtf, "1 t (h w) c -> t c h w", h=dH, w=dW )
            vis = einops.rearrange( vis, "1 t (h w) -> t 1 h w", h=dH, w=dW )

        dtf += 0.5

        if dtf_grid_size > 1:
            resize = torchvision.transforms.Resize([H,W])
            dtf = resize(dtf)
            vis = resize(vis)
    else:
        raise NotImplementedError()

    if resize_infer is not None:
        dtf = resize_original(dtf)*torch.tensor([oW/rW,oH/rH],device=device)[None,:,None,None]
        vis = resize_original(vis)

    # Select and reconstruct concerned trajectories
    return dtf, vis

def compute_dtf(
        model,
        model_archi,
        seq: torch.Tensor,
        ref_idx: int = 0,
        batch_size: int = 1,
        iters: int = 12,
        chain_len: Optional[int] = None,
        dtf_grid_size: int = 1,
        resize_infer: Optional[Tuple[int,int]] = None,
        ):
    dtype = seq.dtype
    device = seq.device
    T, C, H, W = seq.shape

    if chain_len is not None and model_archi in [Archi.DTFNet] and chain_len >= T:
        # Avoid using chains of tref->0 and tref->T-1
        # for models supporting any tref, with a small enough sequence
        chain_len = None

    if chain_len is None:
        if model_archi in [Archi.DTFNet,Archi.TAPIR,Archi.COTRACKER]:
            # DTF are evaluated at one single time ref
            # When several times are queried, we need to run it several times
            dtf, vis = _compute_dtf_contiguous(
                    model,
                    model_archi,
                    seq,
                    ref_idx,
                    batch_size=batch_size,
                    iters=iters,
                    dtf_grid_size=dtf_grid_size,
                    resize_infer=resize_infer )
        else:
            # These methods need the reference frame to be the first
            # We use the chaining code for this run, with maximal chain_len
            chain_len = T

    if chain_len is not None:
        # Here is the chain logic

        final_dtf = []
        final_vis = []
        warp=None
        skip_forward = (ref_idx >= T-1)

        grid = torch.stack( torch.meshgrid((
                torch.arange(W,dtype=dtype,device=device)+0.5,
                torch.arange(H,dtype=dtype,device=device)+0.5 ),
                indexing='xy'),
                dim=0 )
        for t0 in list(range(ref_idx,len(seq)-1))[::chain_len-1]:
            seq_ = seq[t0:t0+chain_len]
            T_ = seq_.shape[0]

            # Pad sequence if the model needs it
            if model_archi in [Archi.PIPS]:
                if T_ < chain_len:
                    pad = einops.repeat(seq_[-1], "c h w -> t c h w", t=chain_len-T_)
                    seq_ = torch.cat((seq_,pad),dim=0)

            dtf_, vis_ = _compute_dtf_contiguous(
                    model,
                    model_archi,
                    seq_,
                    0,
                    batch_size=batch_size,
                    iters=iters,
                    dtf_grid_size=dtf_grid_size,
                    resize_infer=resize_infer )
            dtf_ = dtf_[:T_]
            if vis_ is not None:
                vis_ = vis_[:T_]
            if warp is not None:
                # Warp previous latest frame
                warp_ = warp.permute(1,2,0)
                wh_scale = torch.tensor([2./W,2./H],dtype=dtype,device=device)
                warp_ = warp_*wh_scale.broadcast_to(warp_.shape) - 1.
                warp_ = warp_.unsqueeze(0)
                dtf_ = einops.rearrange( dtf_-grid[None], "t c h w -> 1 (t c) h w" )
                dtf_ = F.grid_sample( dtf_, warp_,
                                      padding_mode='border',
                                      align_corners=False )
                dtf_ = einops.rearrange( dtf_, "1 (t c) h w -> t c h w",
                                         t=T_, c=2 )
                dtf_ = dtf_ + warp
                if vis_ is not None:
                    vis_ = einops.rearrange( vis_, "t c h w -> 1 (t c) h w" )
                    vis_ = F.grid_sample( vis_, warp_,
                                          padding_mode='zeros',
                                          align_corners=False )
                    vis_ = einops.rearrange( vis_, "1 t h w -> t 1 h w" )
            if len(final_dtf) > 0:
                final_dtf.append(dtf_[1:])
                if vis_ is not None:
                    final_vis.append(vis_[1:])
            else:
                final_dtf.append(dtf_)
                if vis_ is not None:
                    final_vis.append(vis_)
            warp = dtf_[-1]
        if ref_idx > 0:
            warp = None
            for t0 in inclusive_range(ref_idx,1)[::chain_len-1]:
                seq_ = seq[range(t0,max(-1,t0-chain_len),-1)]
                #seq_ = seq[t0:max(0,t0-chain_len):-1]
                T_ = seq_.shape[0]

                # Pad sequence if the model needs it
                if model_archi in [Archi.PIPS]:
                    if T_ < chain_len:
                        pad = einops.repeat(seq_[-1], "c h w -> t c h w", t=chain_len-T_)
                        seq_ = torch.cat((seq_,pad),dim=0)
                dtf_, vis_ = _compute_dtf_contiguous(
                        model,
                        model_archi,
                        seq_,
                        0,
                        batch_size=batch_size,
                        iters=iters,
                        dtf_grid_size=dtf_grid_size,
                        resize_infer=resize_infer )
                dtf_ = dtf_[:T_]
                if vis_ is not None:
                    vis_ = vis_[:T_]
                if warp is not None:
                    # Warp previous latest frame
                    warp_ = warp.permute(1,2,0)
                    wh_scale = torch.tensor([2./W,2./H],dtype=dtype,device=device)
                    warp_ = warp_*wh_scale.broadcast_to(warp_.shape) - 1.
                    warp_ = warp_.unsqueeze(0)
                    dtf_ = einops.rearrange( dtf_-grid[None], "t c h w -> 1 (t c) h w" )
                    dtf_ = F.grid_sample( dtf_, warp_,
                                          padding_mode='border',
                                          align_corners=False )
                    dtf_ = einops.rearrange( dtf_, "1 (t c) h w -> t c h w",
                                             t=T_, c=2 )
                    dtf_ = dtf_ + warp
                    if vis_ is not None:
                        vis_ = einops.rearrange( vis_, "t c h w -> 1 (t c) h w" )
                        vis_ = F.grid_sample( vis_, warp_,
                                              padding_mode='zeros',
                                              align_corners=False )
                        vis_ = einops.rearrange( vis_, "1 t h w -> t 1 h w" )
                if skip_forward:
                    # First frame has not been handled by forward pass
                    final_dtf.insert(0,dtf_[range(T_-1,-1,-1)])
                    if vis_ is not None:
                        final_vis.insert(0,vis_[range(T_-1,-1,-1)])
                    skip_forward = False
                else:
                    final_dtf.insert(0,dtf_[range(T_-1,0,-1)])
                    if vis_ is not None:
                        final_vis.insert(0,vis_[range(T_-1,0,-1)])
                warp = dtf_[-1]
        dtf = torch.cat(final_dtf,dim=0)
        if len(final_vis) > 0:
            vis = torch.cat(final_vis,dim=0)
        else:
            vis = None

    # Select and reconstrut concerned trajectories
    return dtf, vis

def evaluate_model_dtf(
        model,
        model_archi: Archi,
        dataset: trajds.TrajDataset,
        device: str = "cuda",
        batch_size: int = 1,
        iters: int = 12,
        chain_len: Optional[int] = None,
        dtf_grid_size: int = 1,
        nb_max_samples: Optional[int] = 250,
        resize_infer: Optional[Tuple[int,int]] = None,
        ):
    nb_samples = 0
    nb_frames = 0
    nb_occ_pixels = 0
    nb_vis_pixels = 0
    nb_aae_pixels = 0
    nb_occ_aae_pixels = 0
    all_scores = {
            "AEPE"      : 0.,
            "AAE"       : 0.,
            "F-1px"     : 0.,
            "F-3px"     : 0.,
            "F-5px"     : 0.,
            "Occ-AEPE"  : 0.,
            "Occ-AAE"   : 0.,
            "Occ-F-1px" : 0.,
            "Occ-F-3px" : 0.,
            "Occ-F-5px" : 0.,
            "Occ-Delta" : 0.,
            "Delta"     : 0.,
            "Vis-Delta" : 0.,
            "OF-Lap"    : 0.,
            "T2-Der"    : 0.,
            }
    all_vis_scores = {
            "OA"        : 0.,
            "AJ"        : 0.,
            "Vis-AJ"    : 0.,
            "Occ-AJ"    : 0.,
            }

    for i, (imgs, gt_trajs, gt_vis, valid, ref_idx) in enumerate(dataset):
        print( f"  Sample: {i}", end='\r' )
        if nb_max_samples is not None and i >= nb_max_samples:
            break
        imgs = imgs.to(device)
        gt_trajs = gt_trajs.to(device)
        gt_vis = gt_vis.to(device)
        dtf, vis = compute_dtf(
                model, model_archi,
                imgs,
                ref_idx.item(),
                batch_size=batch_size,
                chain_len=chain_len,
                iters=iters,
                dtf_grid_size=dtf_grid_size,
                resize_infer=resize_infer )

        # EPE losses
        S, _, H, W = imgs.shape
        grid = torch.stack( torch.meshgrid(
                torch.arange(W,device=device,dtype=torch.float)+0.5,
                torch.arange(H,device=device,dtype=torch.float)+0.5,
                indexing='xy' ),
                dim=0 )
        diff = (dtf-gt_trajs).norm(p=2,dim=1)
        all_scores["AEPE"] += diff.sum() / (H*W)
        all_scores["F-1px"] += (diff < 1.).sum() / (H*W)
        all_scores["F-3px"] += (diff < 3.).sum() / (H*W)
        all_scores["F-5px"] += (diff < 5.).sum() / (H*W)

        correct_2px  = (diff < 2.)
        correct_4px  = (diff < 4.)
        correct_8px  = (diff < 8.)
        correct_16px = (diff < 16.)
        correct_all = 0.25 * (
                correct_2px.to(torch.float) + correct_4px.to(torch.float)
                + correct_8px.to(torch.float) + correct_16px.to(torch.float) )
        all_scores["Delta"] += correct_all.sum() / (H*W)

        # AAE losses
        flow = dtf-grid[None]
        gt_flow = gt_trajs-grid[None]
        dot = torch.einsum("t c h w, t c h w -> t h w",flow,gt_flow)
        pred_norm = flow.norm(p=2,dim=1)
        gt_norm = gt_flow.norm(p=2,dim=1)
        valid_aae = torch.logical_and(pred_norm > 0.3, gt_norm > 0.3)
        if valid_aae.any():
            aae_cos = dot / torch.sqrt(pred_norm*gt_norm).clamp(min=1e-4)
            aae = torch.acos( aae_cos[valid_aae].clamp(-1.,1.) ) * 180./math.pi
            all_scores["AAE"] += aae.sum()
            nb_aae_pixels += valid_aae.sum()

        # Occlusion loss
        gt_vis = gt_vis.squeeze(1)
        if vis is not None:
            vis = vis.squeeze(1)
            pred_vis = (vis>0.5)
            correct_vis = (pred_vis==gt_vis)
            all_vis_scores["OA"] += correct_vis.sum() / (H*W)

        # Occluded version of all losses
        occluded = (gt_vis < 0.5)
        visible = torch.logical_not(occluded)
        all_scores["Occ-AEPE"]   += diff[occluded].sum()
        all_scores["Occ-F-1px"]  += (diff[occluded] < 1.).sum()
        all_scores["Occ-F-3px"]  += (diff[occluded] < 3.).sum()
        all_scores["Occ-F-5px"]  += (diff[occluded] < 5.).sum()

        occ_valid_aae = torch.logical_and(valid_aae,occluded)
        if occ_valid_aae.any():
            occ_aae = torch.acos( aae_cos[occ_valid_aae].clamp(-1.,1.) ) * 180./math.pi
            all_scores["Occ-AAE"] += occ_aae.sum()
            nb_occ_aae_pixels += occ_valid_aae.sum()

        # Average Jaccard
        if vis is not None:
            aj_2px = torch.logical_and( correct_2px, correct_vis ).to(torch.float)
            aj_4px = torch.logical_and( correct_4px, correct_vis ).to(torch.float)
            aj_8px = torch.logical_and( correct_8px, correct_vis ).to(torch.float)
            aj_16px = torch.logical_and( correct_16px, correct_vis ).to(torch.float)
            aj_all = 0.25 * (aj_2px + aj_4px + aj_8px + aj_16px)
            all_vis_scores["AJ"] += aj_all.sum() / (H*W)
            all_vis_scores["Occ-AJ"] += aj_all[occluded].sum()
            all_vis_scores["Vis-AJ"] += aj_all[visible].sum()

        all_scores["Vis-Delta"] += correct_all[visible].sum()
        all_scores["Occ-Delta"] += correct_all[occluded].sum()

        all_scores["OF-Lap"] += ( 0.25 * ( 4*flow[:,:,1:-1,1:-1]
                    - flow[:,:,2:,1:-1:] - flow[:,:,:-2,1:-1]
                    - flow[:,:,1:-1:,2:] - flow[:,:,1:-1,:-2]
        ) ).norm(dim=1).mean()
        all_scores["T2-Der"] += 0.5*(flow[2:] - 2*flow[1:-1] + flow[:-2]).abs().mean()

        # Update total number for further normalization
        nb_frames += S
        nb_occ_pixels += occluded.sum()
        nb_vis_pixels += visible.sum()
        nb_samples += 1

    for k in ["AEPE","F-1px","F-3px","F-5px","Delta"]:
        all_scores[k] /= nb_frames
    for k in ["Occ-AEPE","Occ-F-1px","Occ-F-3px","Occ-F-5px","Occ-Delta"]:
        all_scores[k] /= nb_occ_pixels
    for k in ["Vis-Delta"]:
        all_scores[k] /= nb_vis_pixels
    for k in ["OF-Lap","T2-Der"]:
        all_scores[k] /= nb_samples
    all_scores["AAE"] /= nb_aae_pixels
    all_scores["Occ-AAE"] /= nb_occ_aae_pixels

    for k in ["F-1px","F-3px","F-5px",
              "Occ-F-1px","Occ-F-3px","Occ-F-5px"]:
        all_scores[k] = 100 * (1-all_scores[k])
    for k in ["Delta","Vis-Delta","Occ-Delta"]:
        all_scores[k] *= 100
    if vis is not None:
        for k in ["AJ","OA"]:
            all_vis_scores[k] /= nb_frames
        for k in ["Occ-AJ"]:
            all_vis_scores[k] /= nb_occ_pixels
        for k in ["Vis-AJ"]:
            all_vis_scores[k] /= nb_vis_pixels

        for k in ["AJ","Vis-AJ","Occ-AJ","OA"]:
            all_scores[k] = all_vis_scores[k] * 100

    return all_scores

def test_sequence(
        model,
        model_archi: Archi,
        test_seq: os.PathLike,
        test_ref_idx: int,
        test_output: os.PathLike,
        test_visu_grid_size: int = 16,
        test_start: int = 0,
        test_end: int = 150,
        device: str = "cuda",
        benchmark_iter: int = 1,
        chain_len: Optional[int] = None,
        downscale_max_dim: Optional[int] = None,
        batch_size: int = 1,
        iters: int = 12,
        dtf_grid_size: int = 1,
        resize_infer: Optional[Tuple[int,int]] = None,
        ):
    from dtf_core import img_utils

    ext = ["jpg","png","webp","bmp"]
    imgs = []
    for e in ext:
        imgs += glob.glob(os.path.join(test_seq,f"*.{e}"))
    imgs = sorted(imgs)
    seq = np.stack([cv2.imread(img,cv2.IMREAD_COLOR)[:,:,::-1].astype(np.float32)
                    for img in imgs[test_start:test_end] ], axis=0 )
    test_ref_idx -= test_start
    T, H, W, C = seq.shape
    seq = seq[:,:H-H%8,:W-W%8,:]
    T, H, W, C = seq.shape

    # Resize sequence if asked for
    if downscale_max_dim is not None:
        scale = downscale_max_dim / max(H,W)
        if scale < 1.:
            new_H = 8*int(math.floor(H*scale/8))
            new_W = 8*int(math.floor(W*scale/8))
            seq = np.stack( [cv2.resize(img,(new_W,new_H)) for img in seq],
                             axis=0 )
            H, W = new_H, new_W

    torch_seq = torch.from_numpy(seq).permute(0,3,1,2).contiguous().to(device)

    # Computation of the DTF
    t0 = time.time()
    for _ in range(benchmark_iter):
        dtf, vis = compute_dtf( model, model_archi,
                torch_seq, test_ref_idx,
                chain_len=chain_len,
                batch_size=batch_size,
                iters=iters,
                dtf_grid_size=dtf_grid_size,
                resize_infer=resize_infer )
    t1 = time.time()
    processing_time = (t1-t0)/benchmark_iter
    nb_pix = H*W if resize_infer is None else resize_infer[0]*resize_infer[1]
    print( f"Processed {T} frames of {H}x{W} in {processing_time:.5f}s." )
    print( f"  = {processing_time/T*1000.:.5f}ms per frame" )
    print( f"  = {processing_time/nb_pix*1000.:.5f}ms per pixel" )
    print( f"  = {processing_time/T/nb_pix*1e6:.3f}µs per pixel per frame" )

    dtf = dtf.detach().contiguous().cpu()
    if vis is not None:
        vis = vis.detach().contiguous().cpu()
    grid_traj = torch.stack( torch.meshgrid(
        torch.arange(test_visu_grid_size//2,W,test_visu_grid_size,
                     dtype=torch.long),
        torch.arange(test_visu_grid_size//2,H,test_visu_grid_size,
                     dtype=torch.long),
        indexing='xy' ),
        axis=0 ).reshape(2,-1)
    grid = torch.stack( torch.meshgrid(
        torch.arange(W,dtype=torch.float) + 0.5,
        torch.arange(H,dtype=torch.float) + 0.5,
        indexing='xy' ),
        dim=0 )

    # Give metrics on DTF
    T, _, H, W = dtf.shape
    d_dtf = dtf - grid[None]
    img_lap = ( 0.25 * ( 4*d_dtf[:,:,1:-1,1:-1]
                - d_dtf[:,:,2:,1:-1:] - d_dtf[:,:,:-2,1:-1]
                - d_dtf[:,:,1:-1:,2:] - d_dtf[:,:,1:-1,:-2]
    ) ).norm(dim=1).mean()
    time_2der = 0.5*(d_dtf[2:] - 2*d_dtf[1:-1] + d_dtf[:-2]).abs().mean()
    print( f"OF laplacian: {img_lap:.4f}" )
    print( f"T 2nd der   : {time_2der:.4f}" )

    # Store result in output dir
    mkdir_p(test_output)

    # Draw reference image with reference points
    ref_img = (1./255.)*seq[test_ref_idx]
    N = grid_traj.shape[1]
    for n in range(N):
        color = hsv_to_rgb( np.array(
            [ n/N, 1., 0.9 ] ))
        cv2.drawMarker( ref_img,
            (grid_traj[0,n].item(),grid_traj[1,n].item()),
            color = color,
            markerType = cv2.MARKER_CROSS,
            markerSize = 3 )
    ref_img = (ref_img*255).astype(np.uint8)
    cv2.imwrite( os.path.join(test_output,"ref.png"),
                 ref_img[:,:,::-1] )

    # For optical-flow
    max_flow_norm = max(H,W)/4
    #max_flow_norm = 40.
    wheel_size = 40
    wheel = img_utils.generate_test_wheel(
            wheel_size, amp=max_flow_norm, device="cpu" )
    wheel = img_utils.flow_img(wheel,max_flow_norm)
    wheel = wheel.permute(1,2,0).detach().contiguous().cpu().numpy()
    wheel = (wheel*255).astype(np.uint8)
    cv2.imwrite( os.path.join(test_output,"wheel.png"), wheel[:,:,::-1] )

    # Visu: trajectories and optical-flow
    for t in range(T):
        trajs = dtf[:,:,grid_traj[1],grid_traj[0]].permute(2,0,1).detach().contiguous().cpu().numpy()
        if vis is not None:
            cur_vis = vis[:,0,grid_traj[1],grid_traj[0]].permute(1,0).detach().contiguous().cpu().numpy()

        # Flow image
        flow = dtf[t] - grid
        flow_img = img_utils.flow_img(flow,max_norm=max_flow_norm,
            ).detach().permute(1,2,0).contiguous().cpu().numpy()

        # Vis image (if any)
        if vis is not None:
            vis_img = (0.7/255.)*seq[test_ref_idx]
            vis_img = 0.3 + vis[t,0,:,:,None] * vis_img
            vis_img = vis_img.detach().contiguous().cpu().numpy()

        # Trajectory image
        traj_img = 0.3+(0.7/255.)*seq[t] # Darken a little bit the image
        t_range = inclusive_range(test_ref_idx,t)
        for t0, t1 in zip(t_range[:-1],t_range[1:]):
            dt = abs(t1-t)
            for n in range(N):
                color_v = 0.9
                #if vis is not None:
                #    if not cur_vis[n,t0] or not cur_vis[n,t1]:
                #        color_v = 0.3

                color = hsv_to_rgb( np.array(
                    [ n/N, 1., color_v, math.exp(-0.3*dt) ] ))
                line_alpha( traj_img,
                         (int(trajs[n,t0,0]),int(trajs[n,t0,1])),
                         (int(trajs[n,t1,0]),int(trajs[n,t1,1])),
                         color=color )
        for n in range(N):
            color = hsv_to_rgb( np.array(
                [ n/N, 1., 0.9, 1. ] ))
            cv2.drawMarker( traj_img,
                (int(trajs[n,t,0]),int(trajs[n,t,1])),
                color = color,
                markerType = cv2.MARKER_CROSS,
                markerSize = 3 )

        traj_img = (traj_img*255).astype(np.uint8)
        flow_img = (flow_img*255).astype(np.uint8)
        if vis is not None:
            vis_img = (vis_img*255).astype(np.uint8)

        # Compose images
        #final_img = np.zeros((2*H,2*W if vis is None else 3*W,3),dtype=np.uint8)
        final_img = np.zeros((2*H,2*W,3),dtype=np.uint8)
        #final_img[:H,W//2:W//2+W,:] = ref_img
        final_img[:H,:W,:] = ref_img
        final_img[H:,:W,:] = traj_img
        final_img[H:,W:2*W,:] = flow_img
        if vis is not None:
            final_img[:H,W:,:] = vis_img
        #final_img[H-wheel_size//2:H-wheel_size//2+wheel_size,
        #          W-wheel_size//2:W-wheel_size//2+wheel_size, :] = wheel
        final_img[H-wheel_size:H,
                  W:W+wheel_size, :] = wheel

        # Save images
        cv2.imwrite( os.path.join(test_output,"traj_%04d.png" % t), traj_img[:,:,::-1] )
        cv2.imwrite( os.path.join(test_output,"flow_%04d.png" % t), flow_img[:,:,::-1] )
        if vis is not None:
            cv2.imwrite( os.path.join(test_output,"vis_%04d.png" % t), vis_img[:,:,::-1] )
        cv2.imwrite( os.path.join(test_output,"final_%04d.png" % t), final_img[:,:,::-1] )

def main(
        model_path: os.PathLike,
        model_archi: Archi = Archi.DTFNet,
        davis_pkl_path: Optional[os.PathLike] = None,
        kubric: bool = False,
        rgb_stacking_pkl_path: Optional[os.PathLike] = None,
        kinetics_dir_path: Optional[os.PathLike] = None,
        sintel_dir_path: Optional[os.PathLike] = None,
        flying_things_dir_path: Optional[os.PathLike] = None,
        kubric_dtf_dir_path: Optional[os.PathLike] = None,
        device: str = "cuda",
        batch_size: int = 1,
        query_mode: str = "first",
        iters: int = 12,
        max_seq_len: Optional[int] = 100,
        downscale_max_dim: Optional[int] = None,
        chain_len: Optional[int] = None,
        crop_strategy: OFCropStrategy = OFCropStrategy.All,
        crop_window_size: int = 6,
        dtf_grid_size: int = 1,
        resize_infer: Optional[Tuple[int,int]] = None,
        output_table: Optional[os.PathLike] = None,
        test_seq: Optional[os.PathLike] = None,
        test_ref_idx: int = 0,
        test_output: Optional[os.PathLike] = None,
        test_visu_grid_size: int = 16,
        test_start: int = 0,
        test_end: int = 150,
        benchmark_iter: int = 1,
        ):
    print( "Loading model..." )
    model = load_model( model_path, model_archi, device, iters=iters )

    print( "Evaluating model" )

    all_scores = {}
    def print_scores():
        print("")
        for ds, score in all_scores.items():
            print( ds.name )
            for loss_type, value in score.items():
                print( f"  {loss_type:>15}: {value:2.4f}" )
            print( "" )

    if davis_pkl_path is not None:
        print( "Davis..." )
        dataset = evd.create_davis_dataset(
                davis_pkl_path,
                query_mode=query_mode,
                full_resolution=False )
        all_scores[Dataset.DAVIS] = evaluate_model_traj(
                model, model_archi,
                dataset, "davis",
                query_mode=query_mode,
                max_seq_len=max_seq_len,
                device=device,
                batch_size=batch_size,
                iters=iters,
                chain_len=chain_len,
                resize_infer=resize_infer,
                )
        print_scores()
    if kubric:
        print( "Kubric..." )
        dataset = evd.create_kubric_eval_dataset( mode=query_mode )
        all_scores[Dataset.KUBRIC] = evaluate_model_traj(
                model, model_archi,
                dataset, "kubric",
                query_mode=query_mode,
                max_seq_len=max_seq_len,
                device=device,
                batch_size=batch_size,
                iters=iters,
                chain_len=chain_len,
                resize_infer=resize_infer,
                )
        print_scores()
    if rgb_stacking_pkl_path is not None:
        print( "RGB Stacking..." )
        dataset = evd.create_rgb_stacking_dataset(
                rgb_stacking_pkl_path,
                query_mode=query_mode )
        all_scores[Dataset.RGB_STACKING] = evaluate_model_traj(
                model, model_archi,
                dataset, "robotics",
                query_mode=query_mode,
                max_seq_len=max_seq_len,
                device=device,
                batch_size=batch_size,
                iters=iters,
                chain_len=chain_len,
                resize_infer=resize_infer,
                )
        print_scores()
    if kinetics_dir_path is not None:
        print( "Kinetics..." )
        dataset = evd.create_kinetics_dataset(
                kinetics_dir_path,
                query_mode=query_mode )
        all_scores[Dataset.RGB_STACKING] = evaluate_model_traj(
                model, model_archi,
                dataset, "kinetics",
                query_mode=query_mode,
                max_seq_len=max_seq_len,
                device=device,
                batch_size=batch_size,
                iters=iters,
                chain_len=chain_len,
                resize_infer=resize_infer,
                )
        print_scores()

    if sintel_dir_path is not None:
        print( "Sintel..." )
        dataset = ltfds.MpiSintel(sintel_dir_path,stype=ltfds.SintelType.Clean)
        all_scores[Dataset.SINTEL] = evaluate_model_optical_flow(
                model, model_archi,
                dataset,
                max_seq_len=max_seq_len,
                device=device,
                batch_size=batch_size,
                crop_strategy=crop_strategy,
                crop_window_size=crop_window_size,
                downscale_max_dim=downscale_max_dim,
                iters=iters,
                dtf_grid_size=dtf_grid_size,
                resize_infer=resize_infer,
                )
        print_scores()

    if kubric_dtf_dir_path is not None:
        dataset = trajds.MoviEDataset(None,
            kubric_dtf_dir_path,
            seq_len = max_seq_len,
            split = trajds.Split.VALIDATION,
            )
        print( "Kubric DTF..." )
        all_scores[Dataset.KUBRIC_DTF] = evaluate_model_dtf(
                model, model_archi,
                dataset,
                device=device,
                batch_size=batch_size,
                iters=iters,
                chain_len=chain_len,
                dtf_grid_size=dtf_grid_size,
                resize_infer=resize_infer,
                )
        print_scores()

    if output_table is not None:
        with open(output_table,"w") as f:
            for ds, score in all_scores.items():
                f.write( evd.latex_table(score) )
                f.write( "\n" )

    if test_seq is not None:
        assert test_output is not None
        print( "Testing model" )
        test_sequence(
                model, model_archi, test_seq,
                test_ref_idx = test_ref_idx,
                test_output = test_output,
                test_visu_grid_size=test_visu_grid_size,
                test_start=test_start,
                test_end=test_end,
                benchmark_iter=benchmark_iter,
                device=device,
                chain_len=chain_len,
                downscale_max_dim=downscale_max_dim,
                batch_size=batch_size,
                iters=iters,
                dtf_grid_size=dtf_grid_size,
                resize_infer=resize_infer,
                )

if __name__ == "__main__":
    import argparse
    from dtf_core.parsing import EnumAction, PathType

    argparser = argparse.ArgumentParser( prog="evaluate",
            description="Evaluate models on TAP-Vid datasets",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter )

    argparser.add_argument( "--model-path",
            help="Path to the model",
            type=PathType(exists=True, type='file'),
            required=True )
    argparser.add_argument( "--model-archi",
            help="Architecture ot the model",
            type=Archi, action=EnumAction,
            default=Archi.DTFNet )

    argparser.add_argument( "--davis-pkl-path",
            help="Path to the DAVIS .pkl file",
            type=PathType(exists=True, type='file'),
            default=None )
    argparser.add_argument( "--kubric",
            help="Evaluate on the Kubric dataset",
            action=argparse.BooleanOptionalAction,
            type=bool,
            default=False )
    argparser.add_argument( "--rgb-stacking-pkl-path",
            help="Path to the Robotics RGB Stacking .pkl file",
            type=PathType(exists=True, type='file'),
            default=None )
    argparser.add_argument( "--kinetics-dir-path",
            help="Path to the Kinetics folder containing .pkl files",
            type=PathType(exists=True, type='dir'),
            default=None )

    argparser.add_argument( "--sintel-dir-path",
            help="Path to the MPI-Sintel dataset root for optical flow",
            type=PathType(exists=True, type='dir'),
            default=None )
    argparser.add_argument( "--flying-things-dir-path",
            help="Path to the FlyingThings3D dataset root for optical flow",
            type=PathType(exists=True, type='dir'),
            default=None )
    argparser.add_argument( "--kubric-dtf-dir-path",
            help="Path to the Kubric++ dataset root for DTF",
            type=PathType(exists=True, type='dir'),
            default=None )

    argparser.add_argument( "--device",
            help="PyTorch device to use for running model",
            type=str,
            default="cuda" )
    argparser.add_argument( "--batch-size",
            help="Batch size when processing queries with different times on the same sequence",
            type=int,
            default=1 )
    argparser.add_argument( "--query-mode",
            help="Query mode for loading TAP-Vid data",
            type=str,
            default="first" )
    argparser.add_argument( "--iters",
            help="Number of iterations for prediction, if the model supports it",
            type=int,
            default=12 )
    argparser.add_argument( "--max-seq-len",
            help="Max sequence length, for memory usage.",
            type=int,
            nargs='?',
            default=None )
    argparser.add_argument( "--downscale-max-dim",
            help="Maximum dimension for images",
            type=int,
            nargs='?',
            default=None )

    argparser.add_argument( "--chain-len",
            help="Chain optical-flow or trajectories for the given size, instead "
                 "of processing the whole sequence at once.",
            type=int,
            nargs='?',
            default=None )
    argparser.add_argument( "--crop-strategy",
            help="Cropping strategy for long-term optical-flow",
            type=OFCropStrategy, action=EnumAction,
            default=OFCropStrategy.All )
    argparser.add_argument( "--crop-window-size",
            help="Cropping window size when using Centered / Causal strategies",
            type=int,
            default=6 )
    argparser.add_argument( "--dtf-grid-size",
            help="Sparse methods can be very slow. "
                 "Use this to predict trajectory only on a sparse grid and interpolate it. "
                 "Leave 1 for dense prediction.",
            type=int,
            default=1 )
    argparser.add_argument( "--resize-infer",
            help="Downsize input sequence for inference, like for TAPIR. "
                 "Resize the result back to original resolution",
            type=int,
            nargs=2,
            default=None )

    argparser.add_argument( "--output-table",
            help="Output latex file containing table of results",
            type=PathType( exists=None, type='file' ),
            default=None )

    argparser.add_argument( "--test-seq",
            help="Path to a folder containing images of a test sequence",
            type=PathType( exists=True, type='dir' ),
            default=None )
    argparser.add_argument( "--test-ref-idx",
            help="Reference frame index for the test sequence",
            type=int,
            default=0 )
    argparser.add_argument( "--test-output",
            help="Output folder containing visuals of the test sequence",
            type=PathType( exists=None, type='dir' ),
            default=None )
    argparser.add_argument( "--test-visu-grid-size",
            help="Grid size for displaying trajectories on the test sequence",
            type=int,
            default=16 )
    argparser.add_argument( "--test-start",
            help="Start index for test sequence",
            type=int,
            default=0 )
    argparser.add_argument( "--test-end",
            help="End index for test sequence",
            type=int,
            default=150 )
    argparser.add_argument( "--benchmark-iter",
            help="Number of inference to benchmark the network on the test sequence",
            type=int,
            default=1 )

    args = argparser.parse_args()

    net = main(**vars(args))
