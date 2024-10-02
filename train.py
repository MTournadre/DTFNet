import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

import torchvision
from einops import rearrange, repeat

import os
import random
import numpy as np
from enum import Enum
from typing import Optional, Union, List
from packaging import version

from dtf_core import traj_datasets as datasets
from dtf_core.loss import traj_losses, DTFLossType
from dtf_core.networks.dtfnet import SmallDtfNet, DtfNet, BigDtfNet
from dtf_core import img_utils
from dtf_core.schedule import LinearAnnealing, ConstantLR, CosineAnnealing, OneCycleLR

# TODO Would be cleaner to refactor this training code with lightning...

def compose_trajs( imgs, trajs_gt, trajs_preds, vis_gt, vis_preds, ref_idx, grid_size=20 ):
    """
    Compose images of inputs, trajectories and optical-flows
    for visualizations during training
    """
    T, _, H, W = imgs.shape
    dtype = imgs.dtype
    device = imgs.device

    # Only show a grid of keypoints
    hw_scale = torch.tensor( [ 2/W, 2/H ], device=device, dtype=dtype )
    grid_coords = torch.stack( torch.meshgrid((
            torch.arange(grid_size/2+0.5,W,grid_size,device=device,dtype=dtype),
            torch.arange(grid_size/2+0.5,H,grid_size,device=device,dtype=dtype)),
            indexing='xy' ),
            dim=-1 )
    grid_coords = grid_coords*hw_scale - 1.
    grid_coords = repeat( grid_coords, "h w c -> t h w c", t=T )

    all_data = torch.cat((trajs_gt,trajs_preds,vis_gt,vis_preds),dim=1)

    grid_data = torch.nn.functional.grid_sample(
            all_data, grid_coords,
            padding_mode='border',
            align_corners=False )
    grid_data = rearrange( grid_data, "t c h w -> (h w) t c" )
    grid_trajs_gt, grid_trajs_preds, grid_vis_gt, grid_vis_preds = \
            grid_data.split((2,2,1,1),dim=-1)

    grid_vis_gt = grid_vis_gt.squeeze(-1) > 0.5
    grid_vis_preds = grid_vis_preds.squeeze(-1) > 0.5

    def inclusive_range(a,b):
        if a <= b:
            return list(range(a,b+1))
        else:
            return list(range(a,b-1,-1))

    # Draw trajectories
    traj_imgs = imgs.to(torch.uint8)
    for t in range(T):
        nb = abs(ref_idx-t)+1
        connectivity = list(zip(range(nb-1),range(1,nb)))

        traj_imgs[t] = torchvision.utils.draw_keypoints(
            traj_imgs[t],
            grid_trajs_gt[:,inclusive_range(ref_idx,t)],
            connectivity=connectivity,
            colors=(0,0,0),
            radius=0,
            width=1 )
        traj_imgs[t] = torchvision.utils.draw_keypoints(
            traj_imgs[t],
            grid_trajs_gt[:,t][
                torch.where(grid_vis_gt[:,t]) ].unsqueeze(1),
            connectivity=None,
            colors=(0,255,0),
            radius=2,
            width=0 )
        traj_imgs[t] = torchvision.utils.draw_keypoints(
            traj_imgs[t],
            grid_trajs_gt[:,t][
                torch.where(grid_vis_gt[:,t].logical_not()) ].unsqueeze(1),
            connectivity=None,
            colors=(150,100,0),
            radius=2,
            width=0 )

        traj_imgs[t] = torchvision.utils.draw_keypoints(
            traj_imgs[t],
            grid_trajs_preds[:,inclusive_range(ref_idx,t)],
            connectivity=connectivity,
            colors=(0,0,200),
            radius=0,
            width=1 )
        traj_imgs[t] = torchvision.utils.draw_keypoints(
            traj_imgs[t],
            grid_trajs_preds[:,t][
                torch.where(grid_vis_preds[:,t]) ].unsqueeze(1),
            connectivity=None,
            colors=(0,0,255),
            radius=2,
            width=0 )
        traj_imgs[t] = torchvision.utils.draw_keypoints(
            traj_imgs[t],
            grid_trajs_preds[:,t][
                torch.where(grid_vis_preds[:,t].logical_not()) ].unsqueeze(1),
            connectivity=None,
            colors=(150,0,100),
            radius=2,
            width=0 )

    # Draw optical flow
    coords = torch.stack( torch.meshgrid((
            torch.arange(0.5,W,device=device,dtype=dtype),
            torch.arange(0.5,H,device=device,dtype=dtype)),
            indexing='xy' ),
            dim=0 )
    flow_imgs = torch.empty_like(imgs)
    wheel_size = min( max( 2*(min(H,W)//10), 8 ), 64 )
    max_norm = min( max( min(H/2,W/2), 20. ), 100. )
    flows = trajs_preds-coords.unsqueeze(0)
    for t in range(T):
        flow_imgs[t] = img_utils.flow_img(flows[t],max_norm=max_norm,wheel_size=wheel_size)

    # Draw error maps
    err_imgs = torch.empty_like(imgs)
    errors = (trajs_gt-trajs_preds).norm(p=2,dim=1,keepdim=True)
    err_imgs[:] = (errors/max_norm).clip(0.,1.)

    # Compose images
    flow_imgs = (flow_imgs*255).to(torch.uint8)
    err_imgs = (err_imgs*255).to(torch.uint8)
    res_imgs = torch.cat((traj_imgs,flow_imgs,err_imgs),dim=3)

    return res_imgs

class NetworkType(Enum):
    SMALL_DTFNET = SmallDtfNet
    DTFNET = DtfNet
    BIG_DTFNET = BigDtfNet

def train_trajs(
        network: Optional[Union[str,nn.Module]],
        stage: datasets.Dataset,
        val_datasets: List[datasets.Dataset],
        *,

        network_type: NetworkType,
        model_name: str = "dtfnet",
        output_folder: str = "models",
        log_dir: str = "runs",
        image_size: (int,int) = (256,256),
        downscale: int = 1,

        datasets_root: str = "datasets",
        validation_datasets: list[datasets.Dataset] = [datasets.Dataset.MOVI_E],
        traj_length: int = 8,

        traj_weight: float = 1.,
        vis_weight:  float = 1.,
        corr_weight: float = 0.,
        lap_weight: float = 1.,

        freeze_encoder: int = 0,
        freeze_mask:    int = 0,

        devices: list[str] = ["cuda"],
        mixed_precision: bool = True,
        loss_type: DTFLossType = DTFLossType.EPE,
        grad_accumulate: int = 1,
        num_steps: int = 200000,
        warmup_steps: Optional[int],
        constant_steps: Optional[int],
        init_step: int = 0,
        batch_size: int = 4,
        num_workers: int = 4,
        learning_rate: float = 2e-4,
        wdecay: float = 0.01,
        epsilon: float = 1e-8,
        gamma: float = 0.8,
        clip_gradient: Optional[float] = None,
        seed: int = 3141593,
        reproducible: bool = False,
        recover_from: Optional[str] = None,
        optimizer_path: Optional[os.PathLike] = None,
        ):

    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    if reproducible:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        num_workers = 0

    # Load or init the network if necessary
    if network is None:
        network = network_type.value()
    elif type(network) is str:
        network_path = network
        print( f"Reading network from '{network_path}'..." )
        network = network_type.value()
        network.load_state_dict(torch.load(network_path))

    upsample = True #TODO ?

    print( f"Training {network_type.name}" )
    nb_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print( f"Number of parameters: {nb_params}" )

    # Prepare the model on devices for training
    devices = [ torch.device(d) for d in devices ]
    device = devices[0]
    network = network.to( device )
    model = nn.DataParallel( network, device_ids=devices )
    model.train()

    writer = SummaryWriter(log_dir=os.path.join(log_dir,f"{model_name}"))
    scaler = GradScaler(init_scale=2.**12,enabled=mixed_precision)

    if False:
        torch.autograd.set_detect_anomaly(True)

    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr = learning_rate,
            weight_decay = wdecay,
            eps = epsilon,
            betas=(0.9,0.999),
            amsgrad=False )
    if warmup_steps is None:
        warmup_steps = num_steps//20
    if constant_steps is None:
        constant_steps = num_steps//8
    if optimizer_path is not None:
        print( f"Recovering optimizer: '{optimizer_path}'" )
        opt_state_dict = torch.load(optimizer_path)
        optimizer.load_state_dict(opt_state_dict)
    scheduler = LinearAnnealing(
            optimizer,
            learning_rate,
            warmup_steps=warmup_steps,
            constant_steps=constant_steps,
            total_steps=num_steps,
            min_lr=1e-7 )

    # Load the dataset according to the current stage
    training_dataloader = datasets.training_dataloader(
            stage = stage,
            image_size = image_size,
            datasets_root = datasets_root,
            seq_len = traj_length,
            batch_size = batch_size,
            num_workers = num_workers,
            downscale = downscale,
            verbose = True )

    val_dataloaders = datasets.validation_dataloaders(
            datasets = val_datasets,
            #image_size = image_size,
            datasets_root = datasets_root,
            seq_len = traj_length,
            batch_size = batch_size,
            num_workers = num_workers,
            downscale = downscale )

    if init_step != 0:
        print( f"Initializing at init step {init_step}" )
        scheduler.steps = init_step

    # Manage encoder and mask gradient freezing
    if freeze_encoder > 0:
        for p in network.encoder.parameters():
            p.requires_grad_(False)
    if freeze_mask > 0:
        for p in network.mask_head.parameters():
            p.requires_grad_(False)

    epoch = 0
    val_freq = 10000
    print_freq = 500
    total_steps = init_step
    batch_steps = 0
    val_steps = 0
    print_steps = 0
    print_loss = {
            DTFLossType.EPE: 0.,
            DTFLossType.AAE: 0.,
            #LossType.WING: 0.,
            #LossType.L2: 0.,
            DTFLossType.F1: 0.,
            DTFLossType.F3: 0.,
            DTFLossType.F5: 0.,
            DTFLossType.VIS: 0.,
            DTFLossType.LAPLACIAN: 0.,
            }
    if corr_weight != 0.:
        print_loss[DTFLossType.CORR] = 0.
    if lap_weight != 0.:
        print_loss[DTFLossType.LAPLACIAN] = 0.
    print_total_loss = 0.
    print_max_flow = 0.
    while total_steps < num_steps:
        for batch in training_dataloader:
            if freeze_encoder > 0 and total_steps >= freeze_encoder:
                print( "  Unfreezed encoder weights" )
                for p in network.encoder.parameters():
                    p.requires_grad_(True)
                freeze_encoder = 0
            if freeze_mask > 0 and total_steps >= freeze_mask:
                print( "  Unfreezed mask upsampler weights" )
                for p in network.mask_head.parameters():
                    p.requires_grad_(True)
                freeze_mask = 0

            imgs, trajs_gt, vis_gt, valid, ref_idx = [ x.to(device) for x in batch ]
            if recover_from is not None:
                print( f"WARNING: loading debug checkpoint from '{recover_from}'" )
                checkpoint = torch.load(recover_from)
                network.load_state_dict(checkpoint['network'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                torch.set_rng_state( checkpoint['torch_rng_state'] )
                random.setstate( checkpoint['random_state'] )
                np.random.set_state( checkpoint['np_random_state'] )

                imgs = checkpoint['imgs']
                trajs_gt = checkpoint['trajs_gt']
                vis_gt = checkpoint['vis_gt']
                ref_idx = checkpoint['ref_idx']

            B = imgs.shape[0]
            if batch_steps % grad_accumulate == 0:
                optimizer.zero_grad()

            # Run prediction
            printing = print_steps+B >= print_freq or total_steps+B >= num_steps
            with autocast(enabled=mixed_precision):
                res = model( imgs,
                        upsample = upsample,
                        ref_idx = ref_idx,
                        return_img_feats = corr_weight != 0.,
                        )
                if corr_weight != 0.:
                    trajs_preds, vis_preds, feats = res
                else:
                    trajs_preds, vis_preds = res
            trajs_gt = trajs_gt.type( trajs_preds.dtype )
            vis_gt = vis_gt.type( trajs_preds.dtype )

            losses = traj_losses(
                    ref_idx,
                    trajs_preds, vis_preds,
                    feats if corr_weight != 0 else None,
                    trajs_gt, vis_gt,
                    gamma=gamma,
                    w=2.,
                    eps=0.5,
                    max_motion=100. if mixed_precision else 400.,
                    )

            loss = (1/grad_accumulate) * (
                    traj_weight * losses[loss_type] +
                    vis_weight * losses[DTFLossType.VIS] +
                    lap_weight * losses[DTFLossType.LAPLACIAN]
                    )
            if corr_weight != 0.:
                loss += (1/grad_accumulate) * (
                        corr_weight * losses[DTFLossType.CORR] )

            # Backprop
            try:
                scaler.scale(loss).backward()
            except RuntimeError as e:
                import sys
                debug_path = "/tmp/debug_checkpoint.pt"
                print( f"Error while running backward: {e}", file=sys.stderr )

                checkpoint = {}
                checkpoint['network'] = network.state_dict()
                checkpoint['optimizer'] = optimizer.state_dict()
                checkpoint['torch_rng_state'] = torch.get_rng_state()
                checkpoint['random_state'] = random.getstate()
                checkpoint['np_random_state'] = np.random.get_state()

                checkpoint['imgs'] = imgs
                checkpoint['trajs_gt'] = trajs_gt
                checkpoint['vis_gt'] = vis_gt
                checkpoint['ref_idx'] = ref_idx

                torch.save( checkpoint, debug_path )
                print( f"Saved debug checkpoint to: {debug_path}", file=sys.stderr )
                raise e

            if clip_gradient is not None:
                nn.utils.clip_grad_norm_( model.parameters(), clip_gradient )
            if (batch_steps+1) % grad_accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step_lr(B*grad_accumulate)

            for l in print_loss.keys():
                print_loss[l] += B*losses[l].item()

            total_steps += B
            val_steps += B
            print_steps += B
            batch_steps += 1

            # Log infos
            writer.add_scalar( "lr", scheduler.get_last_lr(), total_steps )
            for l in print_loss.keys():
                writer.add_scalar( f"train_{l.value}", losses[l], total_steps )

            max_motion = losses[DTFLossType.MAX_MAG]
            print_total_loss += B*loss.item()
            print_max_flow = max( max_motion, print_max_flow )
            writer.add_scalar( "max_motion", max_motion, total_steps )

            # Print & show results in writer
            if printing:
                print( f"[{epoch:3}, {total_steps:6}]: " )
                for l in print_loss.keys():
                    print( f"  {l.value}={print_loss[l]/print_steps:.4f}" )
                print( f"  total_loss={print_total_loss/print_steps:.4f}" )
                # We monitor the max norm of optical-flows to ensure
                # the network is not fading to zero
                print( f"  max norm={print_max_flow:.4f}" )
                print("")
                print_steps = 0
                for l in print_loss.keys():
                    print_loss[l] = 0.
                print_total_loss = 0.
                print_max_flow = 0.

                traj_seq = compose_trajs(
                        imgs[0],
                        trajs_gt[0], trajs_preds[0,-1],
                        vis_gt[0], vis_preds[0,-1],
                        ref_idx[0] )
                writer.add_video(
                    "trajs",traj_seq.unsqueeze(0),total_steps, fps=4 )

                #TODO attention clusters ?

            # Validation
            if val_steps >= val_freq or total_steps >= num_steps:
                print(  "======== Validation ========" )
                model.eval()
                with torch.no_grad():
                    for dataloader, ds in val_dataloaders:
                        print( f" {ds.name}" )

                        val_loss = {
                                DTFLossType.EPE: 0.,
                                DTFLossType.AAE: 0.,
                                #LossType.WING: 0.,
                                #LossType.L2: 0.,
                                DTFLossType.F1: 0.,
                                DTFLossType.F3: 0.,
                                DTFLossType.F5: 0.,
                                DTFLossType.VIS: 0.,
                                DTFLossType.LAPLACIAN: 0.,
                                }
                        if corr_weight != 0.:
                            val_loss[DTFLossType.CORR] = 0.
                        val_total_loss = 0.
                        val_len = 0
                        for batch in dataloader:
                            imgs, trajs_gt, vis_gt, valid, ref_idx = [ x.to(device) for x in batch ]
                            with autocast(enabled=mixed_precision):
                                res = model( imgs,
                                        upsample = upsample,
                                        ref_idx = ref_idx,
                                        return_img_feats = corr_weight != 0.,
                                        )
                                if corr_weight != 0.:
                                    trajs_preds, vis_preds, feats = res
                                else:
                                    trajs_preds, vis_preds = res
                                # Only take last prediction for validation losses
                                trajs_preds = trajs_preds[:,[-1]]
                                vis_preds = vis_preds[:,[-1]]
                                trajs_gt = trajs_gt.type( trajs_preds.dtype )
                                vis_gt = vis_gt.type( trajs_preds.dtype )

                            losses = traj_losses(
                                    ref_idx,
                                    trajs_preds, vis_preds,
                                    feats if corr_weight != 0 else None,
                                    trajs_gt, vis_gt,
                                    gamma=gamma,
                                    w=2.,
                                    eps=0.5,
                                    max_motion=100. if mixed_precision else 400.,
                                    )
                            loss = (1/grad_accumulate) * (
                                    traj_weight * losses[loss_type] +
                                    vis_weight * losses[DTFLossType.VIS] +
                                    lap_weight * losses[DTFLossType.LAPLACIAN]
                                    )
                            if corr_weight != 0:
                                loss += (1/grad_accumulate) * (
                                        corr_weight * losses[DTFLossType.CORR] )

                            for l in val_loss.keys():
                                val_loss[l] += B*losses[l]
                            val_total_loss += B*loss.item()
                            val_len += B
                    for l in val_loss.keys():
                        writer.add_scalar(
                            f"val_{ds.name}_{l.value}",
                            val_loss[l]/val_len,
                            total_steps )
                        print( f"  {l.value}={val_loss[l]/val_len:.4f}" )
                    writer.add_scalar(
                        f"val_total",
                        val_total_loss/val_len,
                        total_steps )
                    print( f"  total_loss={val_total_loss/val_len:.4f}" )

                val_steps = 0

                # Save the model after each validation
                if output_folder is not None:
                    if not os.path.exists(output_folder):
                        os.mkdir(output_folder)
                    model_file = os.path.join(output_folder,f"{model_name}_tmp.pt")
                    torch.save( network.state_dict(), model_file )
                    print( f"Saved model to '{model_file}' for step {total_steps}" )

                    opt_file = os.path.join(output_folder,f"{model_name}_optimizer.pt")
                    torch.save( optimizer.state_dict(), opt_file )
                    print( f"Saved optimizer to '{opt_file}' for step {total_steps}" )

                print(  "===========================" )
                print( "" )
                model.train()

            if total_steps >= num_steps:
                break
        epoch += 1

    # Save the final model
    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        model_file = os.path.join(output_folder,f"{model_name}_{stage.name}.pt")
        torch.save( network.state_dict(), model_file )
        print( "" )
        print( f"Saved final model to '{model_file}'" )

    return network

if __name__ == "__main__":
    import argparse
    from dtf_core.parsing import EnumAction, PathType
    from enum import Enum

    argparser = argparse.ArgumentParser( prog="train",
            description="Train an optical flow network over one given stage",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter )

    argparser.add_argument( "--network",
            help="Load an initial network",
            type=PathType(exists=True,type='file'), nargs='?', default=None )

    argparser.add_argument( "--network_type", type=NetworkType, action=EnumAction,
            default=NetworkType.DTFNET,
            help="Type of the network to load/init" )
    argparser.add_argument( "--model_name", type=str, default="dtfnet",
            help="Name of the network" )
    argparser.add_argument( "--output_folder", default="models",
            help="Output folder where to save the networks",
            type=PathType(exists=None,type='dir') )
    argparser.add_argument( "--log_dir",
            help="Log folder", default="runs",
            type=PathType(exists=None,type='dir') )
    argparser.add_argument( "--image_size", type=int, nargs=2, default=[256,256],
            help="Image size (H,W) used for training" )
    argparser.add_argument( "--downscale", type=int, default=1,
            help="Downscale images by integer factor after data perturbation, for lighter training" )

    argparser.add_argument( "--datasets_root", type=PathType(exists=True,type='dir'),
            default="datasets",
            help="Root directory for all training/validation datasets" )
    argparser.add_argument( "--stage", type=datasets.Dataset, action=EnumAction,
            default=datasets.Dataset.MOVI_E,
            help="Training stage to indicate on which dataset(s) to train" )
    argparser.add_argument( "--val_datasets", type=datasets.Dataset, action=EnumAction,
            nargs="+",
            default=[datasets.Dataset.MOVI_E],
            help="Training stage to indicate on which dataset(s) to train" )

    argparser.add_argument( "--traj_length", type=int, default=8,
            help="Length of trajectories" )

    argparser.add_argument( "--traj_weight", type=float, default=1.,
            help="Weight of flow contribution in total loss" )
    argparser.add_argument( "--vis_weight", type=float, default=1.,
            help="Weight of visibility contribution in total loss" )
    argparser.add_argument( "--corr_weight", type=float, default=0.,
            help="Weight of correlation contribution in total loss" )
    argparser.add_argument( "--lap_weight", type=float, default=0.,
            help="Weight of correlation contribution in total loss" )

    argparser.add_argument( "--freeze_encoder", type=int, default=0,
            help="Freeze the encoder for a certain number of steps. Leave 0 to never freeze." )
    argparser.add_argument( "--freeze_mask", type=int, default=0,
            help="Freeze the mask upsampler for a certain number of steps. Leave 0 to never freeze." )

    argparser.add_argument( "--devices", type=str, nargs='+', default=["cuda"],
            help="Devices used for training" )
    argparser.add_argument( "--mixed_precision", type=bool, default=True,
            action=argparse.BooleanOptionalAction,
            help="Use mixed precision to speed-up the training" )
    argparser.add_argument( "--loss_type", type=DTFLossType, action=EnumAction,
            default=DTFLossType.EPE,
            help="Loss type used for training objective" )
    argparser.add_argument( "--grad_accumulate", type=int, default=1,
            help="Use gradient accumulation to virtually extend batch size without using more GPU memory" )
    argparser.add_argument( "--num_steps", type=int, default=200000,
            help="Number of samples on which to train for this stage" )
    argparser.add_argument( "--warmup_steps", type=int, nargs='?', default=None,
            help="Number of warmup steps. None for default" )
    argparser.add_argument( "--constant_steps", type=int, nargs='?', default=None,
            help="Number of constant steps after warmup. None for default" )
    argparser.add_argument( "--init_step", type=int, default=0,
            help="Initial step for resuming training" )
    argparser.add_argument( "--batch_size", type=int, default=4,
            help="Batch size" )
    argparser.add_argument( "--num_workers", type=int, default=4,
            help="Number of workers for data loading. 0 for loading on main thread" )
    argparser.add_argument( "--learning_rate", type=float, default=2e-4,
            help="Initial learning rate" )
    argparser.add_argument( "--wdecay", type=float, default=0.01,
            help="Weight decay for AdamW optimizer" )
    argparser.add_argument( "--epsilon", type=float, default=1e-8,
            help="epsilon parameter for OneCycleLR scheduler" )
    argparser.add_argument( "--gamma", type=float, default="0.8",
            help="Decay factor used for successive predictions of the GRU" )
    argparser.add_argument( "--clip_gradient", type=float, nargs='?', default=None,
            help="Gradient clippingt before running optimizer step. 0 for no clipping" )
    argparser.add_argument( "--seed", type=int, nargs='?', default=3141593,
            help="Manual seed for deterministic behaviour. Set None for no manual seed" )
    argparser.add_argument( "--reproducible", type=bool, default=False,
            action=argparse.BooleanOptionalAction,
            help="Try to use reproducible functions, possibly at the cost of performance..." )
    argparser.add_argument( "--recover_from",
            help="Load an accurate checkpoint for debugging",
            type=PathType(exists=True,type='file'), nargs='?', default=None )
    argparser.add_argument( "--optimizer_path",
            help="Load a previous optimizer parameters",
            type=PathType(exists=True,type='file'), nargs='?', default=None )

    args = argparser.parse_args()

    net = train_trajs(**vars(args))
