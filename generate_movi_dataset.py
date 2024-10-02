import os
import errno
import numpy as np
import random
from enum import Enum
from typing import Optional, Union, List
import shutil

import tensorflow_datasets as tfds
from pyquaternion import Quaternion

import torch
import cv2

from PIL import Image
import subprocess
import json

def transf_mat_from_quat( quat, pos, device="cuda" ):
    q = Quaternion(quat.cpu().numpy())
    mat = torch.from_numpy(q.transformation_matrix.astype(np.float32)).to(device)
    mat[:3,3] = pos

    return mat

def inv_transf_mat( mat ):
    r = mat[:3,:3]
    t = mat[:3,3]

    inv = mat.clone()
    inv[:3,:3] = r.T
    inv[:3,3] = -r.T @ t

    return inv

def apply_range( array, data_range, float_type=np.float32, device="cuda" ):
    dtype_max = np.iinfo(array.dtype).max
    numpy_res = array.astype(float_type) * ((data_range[1]-data_range[0])/dtype_max) \
            + data_range[0]
    return torch.from_numpy(numpy_res).to(device)

def homo_coords( nd_coords ):
    shape = list(nd_coords.shape)
    shape[-1] = 1
    return torch.cat(
        (nd_coords,torch.ones(shape,dtype=nd_coords.dtype,device=nd_coords.device)),
        dim=-1 )

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


def generate_dataset_from_kubric_data(
        output_folder: os.PathLike,
        dataset_name: str = "movi_e",
        subset: str = "train",
        data_dir: str = "gs://kubric-public/tfds",
        check_depth_threshold: float = 0.03,
        check_coords_threshold: float = 0.03,
        ref_idx: Optional[int] = None,
        device: str = "cuda",
        ):
    """
    Generate DTF ground-truth on downloaded Kubric data.
    """

    torch.set_grad_enabled(False)
    ds = tfds.load(dataset_name, data_dir=data_dir, decoders=None)[subset]
    nb_samples = len(ds)

    ref_idx_initial = ref_idx
    for idx_sample, sample in enumerate(ds):
        # Parse metadata
        metadata = sample['metadata']
        seq_name = metadata['video_name'].numpy().decode()
        print( f"Processing '{seq_name}' ({idx_sample}/{nb_samples})" )
        seq_len = metadata['num_frames'].numpy()
        depth_range = metadata['depth_range'].numpy()
        flow_range = metadata['forward_flow_range'].numpy()
        height = metadata['height'].numpy()
        width = metadata['width'].numpy()
        num_obj = metadata['num_instances'].numpy()

        # Images
        seq = torch.from_numpy(sample['video'].numpy()).to(device)
        flow = sample['forward_flow'].numpy()
        flow = apply_range(flow,flow_range)
        depth = sample['depth'].numpy()
        depth = apply_range(depth,depth_range)

        # Object index and coordinates for each pixel of each frame
        # 0 is background, others are off by one with object parameters
        obj_idx = torch.from_numpy(sample['segmentations'].numpy()).to(device)
        obj_coords = apply_range( sample['object_coordinates'].numpy(), [0., 1.] )

        # Parse camera parameters
        camera = sample['camera']
        focal = camera['focal_length'].numpy()
        sensor_width = camera['sensor_width'].numpy()
        cam_pos = torch.from_numpy(camera['positions'].numpy())
        cam_quat = torch.from_numpy(camera['quaternions'].numpy())
        # We have the camera pose, but we want the transformation to apply
        # to world coordinates
        cam_transf_mat = torch.stack( [
            inv_transf_mat( transf_mat_from_quat(quat,pos,device=device) )
            for quat, pos in zip(cam_quat,cam_pos) ], dim=0 )

        # Parse object parameters for each object, for each frame
        # All shapes are (Nobj, Nframes, ...)
        instances = sample['instances']
        bboxes_3d = torch.from_numpy(instances['bboxes_3d'].numpy()).to(device)
        #obj_scale = torch.from_numpy(instances['scale'].numpy())
        obj_quat = torch.from_numpy(instances['quaternions'].numpy())
        obj_pos = torch.from_numpy(instances['positions'].numpy())

        obj_mat = torch.stack([ torch.stack([
            transf_mat_from_quat(quat,pos,device=device)
            for quat, pos in zip(oquat,opos) ], dim=0 )
            for oquat, opos in zip(obj_quat,obj_pos) ], dim=0 )

        # Select reference frame, and reconstruct trajectories from reference
        # object coordinates. The STRONG assumption here is that objects
        # are rigid, so we can still apply these local object coordinates to
        # other frames
        if ref_idx_initial is None:
            ref_idx = random.randrange(0,seq_len)
        else:
            ref_idx = ref_idx_initial

        # This part is valid only for instances (not background)
        # pp: per pixel
        # pi: per instance
        pi_ref_bboxes_3d = homo_coords(bboxes_3d[:,ref_idx])
        pi_ref_obj_mat_inv = torch.stack(
                [ inv_transf_mat(mat) for mat in obj_mat[:,ref_idx] ],
            axis=0 )
        pi_ref_bboxes_3d_local = torch.einsum( "i y x, i n x -> i n y",
                pi_ref_obj_mat_inv, pi_ref_bboxes_3d )
        pi_ref_bboxes_3d_local_orig = pi_ref_bboxes_3d_local[:,0,:]
        pi_ref_bboxes_3d_local_xyz = pi_ref_bboxes_3d_local[:,[4,2,1],:] \
                - pi_ref_bboxes_3d_local_orig[:,None,:]

        pp_ref_obj_idx = obj_idx[ref_idx].to(torch.int64).squeeze(-1) - 1
        pp_ref_obj_coords = obj_coords[ref_idx]

        pp_ref_bboxes_3d_local_orig = pi_ref_bboxes_3d_local_orig[pp_ref_obj_idx]
        pp_ref_bboxes_3d_local_xyz = pi_ref_bboxes_3d_local_xyz[pp_ref_obj_idx]
        pp_ref_local_coords = pp_ref_bboxes_3d_local_orig + torch.einsum(
                "h w x y, h w x -> h w y",
                pp_ref_bboxes_3d_local_xyz,
                pp_ref_obj_coords )

        pp_mat = obj_mat[pp_ref_obj_idx]
        pp_world_coords = torch.einsum( "h w n y x, h w x -> n h w y",
                pp_mat, pp_ref_local_coords )
        pp_cam_coords_obj = torch.einsum( "n y x, n h w x -> n h w y",
                cam_transf_mat, pp_world_coords )

        # Some checks...
        if True:
            obj0_idx = torch.where( obj_idx[ref_idx,:,:,0] == 1 )
            obj0_depth = depth[ref_idx,:,:,0][obj0_idx]
            obj0_cam_coords = pp_cam_coords_obj[ref_idx][obj0_idx][:,:3]
            obj0_cam_depth = obj0_cam_coords.norm(p=2,dim=-1)

        # Reconstruct the 3D of reference frame
        # This part is valid only for static objects (= background)
        pix_grid = torch.stack(
                torch.meshgrid(
                    -torch.arange(-width/2+0.5, width/2+0.5, device=device),
                    torch.arange(-height/2+0.5, height/2+0.5, device=device),
                    indexing='xy'),
                dim=-1 )
        pix_coords = pix_grid * sensor_width/width
        pp_z = -depth[ref_idx,:,:,0] * focal \
                / torch.sqrt(focal**2 + (pix_coords**2).sum(dim=-1))
        pp_xy = pp_z.unsqueeze(-1)/focal * pix_coords
        pp_coords_3d = torch.cat((pp_xy,pp_z.unsqueeze(-1)),dim=-1)

        ref_to_frame_transf_mat = torch.einsum( "t z y, y x -> t z x",
                cam_transf_mat,
                inv_transf_mat(cam_transf_mat[ref_idx]) )
        pp_cam_coords_from_ref = torch.einsum( "t y x, h w x -> t h w y",
                ref_to_frame_transf_mat,
                homo_coords(pp_coords_3d) )

        # Merge both background and objects 3D data
        pp_cam_coords = torch.where( obj_idx[[ref_idx],:,:,:] == 0,
                pp_cam_coords_from_ref,
                pp_cam_coords_obj )
        pp_depth = pp_cam_coords[:,:,:,:3].norm(p=2,dim=-1)

        pp_traj_2d = (width/sensor_width * focal) * (
                pp_cam_coords[:,:,:,:2] / pp_cam_coords[:,:,:,[2]] )

        if True:
            reproj_err = (pp_traj_2d[ref_idx] - pix_grid).norm(p=2,dim=-1)
            depth_err = (pp_depth[ref_idx] - depth[ref_idx,:,:,0]).norm(p=2,dim=-1)
            print( f"  Max reprojection error: {reproj_err.max()}" )
            print( f"  Max depth error: {depth_err.max()}" )

        # Fix ground-truth for reference frame...
        pp_depth[ref_idx,:,:] = depth[ref_idx,:,:,0]
        pp_traj_2d[ref_idx,:,:,:] = pix_grid

        #pp_traj_2d = pp_traj_2d / torch.tensor([-width/2,height/2],device=device,dtype=torch.float32)
        pp_traj_2d[:,:,:,0] *= -1
        sampling_coords = pp_traj_2d*torch.tensor([2/width,2/height],dtype=torch.float32,device=device)
        pp_traj_2d += torch.tensor([width/2,height/2],dtype=torch.float32,device=device)

        inbounds = (pp_traj_2d[:,:,:,0] > 0).logical_and(
                pp_traj_2d[:,:,:,0] < width).logical_and(
                pp_traj_2d[:,:,:,1] > 0).logical_and(
                pp_traj_2d[:,:,:,1] < height)

        # Use 'nearest' sampling strategy to test occlusion
        traj_obj_idx = torch.nn.functional.grid_sample(
                obj_idx.permute(0,3,1,2).to(torch.float32), # can't use grid_sample on integer tensors...
                sampling_coords,
                mode='nearest',
                padding_mode='border',
                align_corners=False ).to(torch.int8).squeeze(1)
        same_idx = (traj_obj_idx == obj_idx[[ref_idx]].squeeze(-1))

        # Checking the object index is not enough, because it does not
        # handle self-occclusions. So we also check that the depth
        # is approximately the one expected
        traj_depth = torch.nn.functional.grid_sample(
                depth.permute(0,3,1,2),
                sampling_coords,
                mode='bilinear',
                padding_mode='border',
                align_corners=False ).squeeze(1)

        #same_depth = ( (traj_depth-pp_depth).abs() < check_depth_threshold )
        same_depth = ( (traj_depth-pp_depth).abs()/traj_depth < check_depth_threshold )

        # Checking object local coordinates
        traj_obj_coords = torch.nn.functional.grid_sample(
                obj_coords.permute(0,3,1,2),
                sampling_coords,
                mode='nearest',
                padding_mode='border',
                align_corners=False ).permute(0,2,3,1)
        same_obj_coords = ( (traj_obj_coords - obj_coords[[ref_idx]]).norm(p=2,dim=-1)
                < check_coords_threshold )

        visible = inbounds.logical_and(same_idx).logical_and(same_depth.logical_or(same_obj_coords))

        invalid = torch.isnan(pp_traj_2d).any() \
                or (pp_traj_2d < -5*width).any() \
                or (pp_traj_2d > 6*width).any()
        if False and invalid:
            folder = "/tmp"
            frame_name = "frame_%08d.png"
            imgs = seq.contiguous().cpu().numpy()
            ref_points = np.stack(
                    np.meshgrid(np.arange(7,width,15),np.arange(10,height,20)),
                    axis=-1 )
            ref_points = ref_points.reshape(-1,2)

            def inclusive_range(a,b):
                if a < b:
                    return range(a,b+1)
                else:
                    return range(a,b-1,-1)

            for t in range(len(imgs)):
                img = seq[t].clone().contiguous().cpu().numpy()
                for point in ref_points:
                    prev_pos = None
                    for i in inclusive_range(ref_idx,t):
                        ipoint = point.astype(np.int64)
                        ipos = pp_traj_2d[i,ipoint[1],ipoint[0]].to(torch.int64).tolist()
                        if i == ref_idx:
                            color = (0,0,0)
                        elif visible[i,ipoint[1],ipoint[0]]:
                            color = (0,200,0)
                        else:
                            color = (0,0,200)

                        if prev_pos is not None:
                            cv2.line(img, prev_pos, ipos, color=(200,0,0), thickness=1)

                        prev_pos = ipos

                    cv2.drawMarker( img, ipos,
                            color=color,
                            markerType=cv2.MARKER_CROSS,
                            markerSize=5,
                            thickness=1 )

                cv2.imwrite( os.path.join(folder,frame_name%t), img )

            import ipdb; ipdb.set_trace()

        # Save the result
        if invalid:
            print( "Invalid sample!" )
        else:
            print( "  Saving..." )
            imgs = seq.contiguous().cpu().numpy()
            mkdir_p( os.path.join(output_folder,seq_name) )
            for i, img in enumerate(imgs):
                cv2.imwrite( os.path.join(output_folder,seq_name,"%04d.png"%i), img )
            np.savez( os.path.join(output_folder,seq_name,"traj_data.npz"),
                    ref_idx=ref_idx,
                    traj=pp_traj_2d.contiguous().cpu().numpy(),
                    vis=visible.contiguous().cpu().numpy() )

def generate_dataset_from_kubric_source(
        output_folder: os.PathLike,
        kubric_source_path: os.PathLike,
        dataset_name: str = "movi_f",
        split: str = "train",
        check_depth_threshold: float = 0.03,
        check_coords_threshold: float = 0.03,
        check_reproj_threshold: float = 1.5,
        ref_idx: Optional[int] = None,
        device: str = "cuda",
        start_sample: int = 0,
        end_sample: int = 10000,
        resolution: str = "512x512",
        ):
    """
    Generate data from scratch using Kubric original repository:
    https://github.com/google-research/kubric.git

    The repository should be already cloned, in a path indicated in 'kubric_source_path'.

    The advantage of using this method is that, when failing,
    it can roll a new random sample until it works.
    """

    torch.set_grad_enabled(False)
    nb_samples = end_sample-start_sample

    uid = os.getuid()
    gid = os.getgid()

    ref_idx_initial = ref_idx
    for idx_sample in range(start_sample,end_sample):
        valid_sample = False
        try_idx = 0
        while not valid_sample:
            try_idx += 1
            if try_idx >= 100:
                raise RuntimeError( f"Too many attempts for sample {idx_sample}" )

            # Fix seed for deterministic data generation
            seed = 100*idx_sample + try_idx
            random.seed(seed)

            tmp_name = f"{dataset_name}_{split}_{idx_sample:05d}"
            output_dir = os.path.abspath( f"output_{tmp_name}")
            mkdir_p(output_dir)

            # Generate data
            print( "Generating Kubric sample..." )
            if True:
                res = subprocess.run([
                    "docker", "run", "--rm", "--interactive",
                        "--user", f"{uid}:{gid}",
                        "--volume", f"{kubric_source_path}:/kubric",
                        "--volume", f"{output_dir}:/output",
                        "kubricdockerhub/kubruntu",
                        "/usr/bin/python3", "challenges/movi/movi_def_worker.py",
                            "--camera=linear_movement",
                            "--resolution", resolution,
                            "--objects_split", split,
                            "--backgrounds_split", split,
                            "--min_num_static_objects", "1",
                            "--max_num_static_objects", "10",
                            "--min_num_dynamic_objects", "1",
                            "--max_num_dynamic_objects", "8",
                            "--max_motion_blur", "2.0" if dataset_name == "movi_f" else "0.0",
                            "--seed", f"{seed}",
                            "--job-dir", "/output",
                            #"--frame_end", "24",
                            ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    )
                if res.returncode != 0:
                    print( f"Error {res.returncode} while processing sample {idx_sample}" )
                    shutil.rmtree(output_dir)
                    continue

            # Parse metadata
            with open(os.path.join(output_dir,"metadata.json"),"r") as f:
                config = json.load(f)
            metadata = config["metadata"]
            #with open(os.path.join(output_dir,"data_ranges.json"),"r") as f:
            #    ranges = json.load(f)
            seq_name = f"{idx_sample:05}"
            print( f"Processing '{seq_name}' trajectories..." )
            seq_len = metadata['num_frames']
            height, width = metadata['resolution']
            num_obj = metadata['num_instances']

            # Read depth and segmentation
            depth = np.zeros((seq_len,height,width),dtype=np.float32)
            obj_idx = np.zeros((seq_len,height,width),dtype=np.uint8)
            obj_coords = np.zeros((seq_len,height,width,3),dtype=np.uint8)
            for i in range(seq_len):
                depth[i] = np.array( Image.open(
                    os.path.join(output_dir,f"depth_{i:05d}.tiff") ) )
                obj_idx[i] = np.array( Image.open(
                    os.path.join(output_dir,f"segmentation_{i:05d}.png") ) )
                obj_coords[i] = np.array( Image.open(
                    os.path.join(output_dir,f"object_coordinates_{i:05d}.png") ) )

            # Object index and coordinates for each pixel of each frame
            # 0 is background, others are off by one with object parameters
            depth = torch.from_numpy(depth).to(device)
            obj_idx = torch.from_numpy(obj_idx).to(device)
            obj_coords = apply_range( obj_coords, [0., 1.] )

            # Parse camera parameters
            camera = config['camera']
            focal = camera['focal_length']
            sensor_width = camera['sensor_width']
            cam_pos = torch.tensor(camera['positions'],device=device)
            cam_quat = torch.tensor(camera['quaternions'],device=device)
            # We have the camera pose, but we want the transformation to apply
            # to world coordinates
            cam_transf_mat = torch.stack( [
                inv_transf_mat( transf_mat_from_quat(quat,pos,device=device) )
                for quat, pos in zip(cam_quat,cam_pos) ], dim=0 )

            # Parse object parameters for each object, for each frame
            # All shapes are (Nobj, Nframes, ...)
            instances = config['instances']
            bboxes_3d = torch.tensor([inst['bboxes_3d'] for inst in instances],device=device)
            #obj_scale = torch.tensor([inst['scale'] for inst in instances],device=device)
            obj_quat = torch.tensor([inst['quaternions'] for inst in instances],device=device)
            obj_pos = torch.tensor([inst['positions'] for inst in instances],device=device)

            obj_mat = torch.stack([ torch.stack([
                transf_mat_from_quat(quat,pos,device=device)
                for quat, pos in zip(oquat,opos) ], dim=0 )
                for oquat, opos in zip(obj_quat,obj_pos) ], dim=0 )

            # Select reference frame, and reconstruct trajectories from reference
            # object coordinates. The STRONG assumption here is that objects
            # are rigid, so we can still apply these local object coordinates to
            # other frames
            if ref_idx_initial is None:
                ref_idx = random.randrange(0,seq_len)
            else:
                ref_idx = ref_idx_initial

            # This part is valid only for instances (not background)
            # pp: per pixel
            # pi: per instance
            pi_ref_bboxes_3d = homo_coords(bboxes_3d[:,ref_idx])
            pi_ref_obj_mat_inv = torch.stack(
                    [ inv_transf_mat(mat) for mat in obj_mat[:,ref_idx] ],
                axis=0 )
            pi_ref_bboxes_3d_local = torch.einsum( "i y x, i n x -> i n y",
                    pi_ref_obj_mat_inv, pi_ref_bboxes_3d )
            pi_ref_bboxes_3d_local_orig = pi_ref_bboxes_3d_local[:,0,:]
            pi_ref_bboxes_3d_local_xyz = pi_ref_bboxes_3d_local[:,[4,2,1],:] \
                    - pi_ref_bboxes_3d_local_orig[:,None,:]

            pp_ref_obj_idx = obj_idx[ref_idx].to(torch.int64).squeeze(-1) - 1
            pp_ref_obj_coords = obj_coords[ref_idx]

            pp_ref_bboxes_3d_local_orig = pi_ref_bboxes_3d_local_orig[pp_ref_obj_idx]
            pp_ref_bboxes_3d_local_xyz = pi_ref_bboxes_3d_local_xyz[pp_ref_obj_idx]
            pp_ref_local_coords = pp_ref_bboxes_3d_local_orig + torch.einsum(
                    "h w x y, h w x -> h w y",
                    pp_ref_bboxes_3d_local_xyz,
                    pp_ref_obj_coords )

            pp_mat = obj_mat[pp_ref_obj_idx]
            pp_world_coords = torch.einsum( "h w n y x, h w x -> n h w y",
                    pp_mat, pp_ref_local_coords )
            pp_cam_coords_obj = torch.einsum( "n y x, n h w x -> n h w y",
                    cam_transf_mat, pp_world_coords )

            # Some checks...
            if False:
                obj0_idx = torch.where( obj_idx[ref_idx,:,:] == 1 )
                obj0_depth = depth[ref_idx,:,:][obj0_idx]
                obj0_cam_coords = pp_cam_coords_obj[ref_idx][obj0_idx][:,:3]
                obj0_cam_depth = obj0_cam_coords.norm(p=2,dim=-1)

            # Reconstruct the 3D of reference frame
            # This part is valid only for static objects (= background)
            pix_grid = torch.stack(
                    torch.meshgrid(
                        -torch.arange(-width/2+0.5, width/2+0.5, device=device),
                        torch.arange(-height/2+0.5, height/2+0.5, device=device),
                        indexing='xy'),
                    dim=-1 )
            pix_coords = pix_grid * sensor_width/width
            pp_z = -depth[ref_idx,:,:] * focal \
                    / torch.sqrt(focal**2 + (pix_coords**2).sum(dim=-1))
            pp_xy = pp_z.unsqueeze(-1)/focal * pix_coords
            pp_coords_3d = torch.cat((pp_xy,pp_z.unsqueeze(-1)),dim=-1)

            ref_to_frame_transf_mat = torch.einsum( "t z y, y x -> t z x",
                    cam_transf_mat,
                    inv_transf_mat(cam_transf_mat[ref_idx]) )
            pp_cam_coords_from_ref = torch.einsum( "t y x, h w x -> t h w y",
                    ref_to_frame_transf_mat,
                    homo_coords(pp_coords_3d) )

            # Merge both background and objects 3D data
            pp_cam_coords = torch.where( obj_idx[[ref_idx],:,:,None] == 0,
                    pp_cam_coords_from_ref,
                    pp_cam_coords_obj )
            pp_depth = pp_cam_coords[:,:,:,:3].norm(p=2,dim=-1)

            pp_traj_2d = (width/sensor_width * focal) * (
                    pp_cam_coords[:,:,:,:2] / pp_cam_coords[:,:,:,[2]] )

            if True:
                reproj_err = (pp_traj_2d[ref_idx] - pix_grid).norm(p=2,dim=-1)
                depth_err = (pp_depth[ref_idx] - depth[ref_idx,:,:]).norm(p=2,dim=-1)
                print( f"  Max reprojection error: {reproj_err.max()}" )
                print( f"  Max depth error: {depth_err.max()}" )

                if reproj_err.max() > check_reproj_threshold:
                    print( "Invalid reprojection..." )
                    valid_sample = False
                    shutil.rmtree(output_dir)
                    continue

            # Fix ground-truth for reference frame...
            pp_depth[ref_idx,:,:] = depth[ref_idx,:,:]
            pp_traj_2d[ref_idx,:,:,:] = pix_grid

            #pp_traj_2d = pp_traj_2d / torch.tensor([-width/2,height/2],device=device,dtype=torch.float32)
            pp_traj_2d[:,:,:,0] *= -1
            sampling_coords = pp_traj_2d*torch.tensor([2/width,2/height],dtype=torch.float32,device=device)
            pp_traj_2d += torch.tensor([width/2,height/2],dtype=torch.float32,device=device)

            inbounds = (pp_traj_2d[:,:,:,0] > 0).logical_and(
                    pp_traj_2d[:,:,:,0] < width).logical_and(
                    pp_traj_2d[:,:,:,1] > 0).logical_and(
                    pp_traj_2d[:,:,:,1] < height)

            # Use 'nearest' sampling strategy to test occlusion
            traj_obj_idx = torch.nn.functional.grid_sample(
                    obj_idx.unsqueeze(1).to(torch.float32), # can't use grid_sample on integer tensors...
                    sampling_coords,
                    mode='nearest',
                    padding_mode='border',
                    align_corners=False ).to(torch.int8).squeeze(1)
            same_idx = (traj_obj_idx == obj_idx[[ref_idx]].squeeze(-1))

            # Checking the object index is not enough, because it does not
            # handle self-occclusions. So we also check that the depth
            # is approximately the one expected
            traj_depth = torch.nn.functional.grid_sample(
                    depth.unsqueeze(1),
                    sampling_coords,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=False ).squeeze(1)

            #same_depth = ( (traj_depth-pp_depth).abs() < check_depth_threshold )
            same_depth = ( (traj_depth-pp_depth).abs()/traj_depth < check_depth_threshold )

            # Checking object local coordinates
            traj_obj_coords = torch.nn.functional.grid_sample(
                    obj_coords.permute(0,3,1,2),
                    sampling_coords,
                    mode='nearest',
                    padding_mode='border',
                    align_corners=False ).permute(0,2,3,1)
            same_obj_coords = ( (traj_obj_coords - obj_coords[[ref_idx]]).norm(p=2,dim=-1)
                    < check_coords_threshold )

            visible = inbounds.logical_and(same_idx).logical_and(same_depth.logical_or(same_obj_coords))

            invalid = pp_traj_2d.isnan().any() \
                    or (pp_traj_2d < -5*width).any() \
                    or (pp_traj_2d > 6*width).any()
            if False and invalid:
                folder = "/tmp"
                frame_name = "frame_%08d.png"
                imgs = seq.contiguous().cpu().numpy()
                ref_points = np.stack(
                        np.meshgrid(np.arange(7,width,15),np.arange(10,height,20)),
                        axis=-1 )
                ref_points = ref_points.reshape(-1,2)

                def inclusive_range(a,b):
                    if a < b:
                        return range(a,b+1)
                    else:
                        return range(a,b-1,-1)

                for t in range(len(imgs)):
                    img = seq[t].clone().contiguous().cpu().numpy()
                    for point in ref_points:
                        prev_pos = None
                        for i in inclusive_range(ref_idx,t):
                            ipoint = point.astype(np.int64)
                            ipos = pp_traj_2d[i,ipoint[1],ipoint[0]].to(torch.int64).tolist()
                            if i == ref_idx:
                                color = (0,0,0)
                            elif visible[i,ipoint[1],ipoint[0]]:
                                color = (0,200,0)
                            else:
                                color = (0,0,200)

                            if prev_pos is not None:
                                cv2.line(img, prev_pos, ipos, color=(200,0,0), thickness=1)

                            prev_pos = ipos

                        cv2.drawMarker( img, ipos,
                                color=color,
                                markerType=cv2.MARKER_CROSS,
                                markerSize=5,
                                thickness=1 )

                    cv2.imwrite( os.path.join(folder,frame_name%t), img )

                import ipdb; ipdb.set_trace()

            if invalid:
                valid_sample = False
                print( "Unstable results. Need to recompute sample" )
                shutil.rmtree(output_dir)
                continue
            else:
                valid_sample = True

            # Save the result
            if valid_sample:
                print( "  Saving sample..." )
                mkdir_p( os.path.join(output_folder,seq_name) )
                for i in range(seq_len):
                    img = np.array(Image.open(
                            os.path.join(output_dir,f"rgba_{i:05d}.png") ))
                    img = Image.fromarray(img[:,:,:3])
                    img.save( os.path.join(output_folder,seq_name,"%04d.png"%i) )
                np.savez( os.path.join(output_folder,seq_name,"traj_data.npz"),
                        ref_idx=ref_idx,
                        traj=pp_traj_2d.contiguous().cpu().numpy(),
                        vis=visible.contiguous().cpu().numpy() )

                shutil.rmtree(output_dir)

if __name__ == "__main__":
    import argparse
    from dtf_core.parsing import EnumAction, PathType
    from enum import Enum

    parser = argparse.ArgumentParser( prog="generate_movi_dataset",
            description="Train an optical flow network over one given stage",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    subparsers = parser.add_subparsers()

    source_parser = subparsers.add_parser("from_source")
    source_parser.set_defaults(_action=generate_dataset_from_kubric_source)
    source_parser.add_argument( "--output_folder", type=str, required=True )
    source_parser.add_argument( "--dataset_name", type=str, default="movi_e" )
    source_parser.add_argument( "--split", type=str, default="train" )
    source_parser.add_argument( "--check_depth_threshold", type=float, default=0.03 )
    source_parser.add_argument( "--check_coords_threshold", type=float, default=0.03 )
    source_parser.add_argument( "--check_reproj_threshold", type=float, default=1.5 )
    source_parser.add_argument( "--ref_idx", type=int, nargs='?', default=None,
                   help = "Index of reference frame. Leave None for random index" )
    source_parser.add_argument( "--device", type=str, default="cuda" )
    source_parser.add_argument( "--resolution", type=str, default="512x512" )
    source_parser.add_argument( "--start_sample", type=int, default=0 )
    source_parser.add_argument( "--end_sample", type=int, default=10000 )
    source_parser.add_argument( "--kubric_source_path", type=str,
            help="Path to the kubric repository",
            required=True )

    data_parser = subparsers.add_parser("from_data")
    data_parser.set_defaults(_action=generate_dataset_from_kubric_data)
    data_parser.add_argument( "--output_folder", type=str, required=True )
    data_parser.add_argument( "--dataset_name", type=str, default="movi_e" )
    data_parser.add_argument( "--subset", type=str, default="train" )
    data_parser.add_argument( "--data_dir", type=str, default="gs://kubric-public/tfds")
    data_parser.add_argument( "--check_depth_threshold", type=float, default=0.03 )
    data_parser.add_argument( "--check_coords_threshold", type=float, default=0.03 )
    data_parser.add_argument( "--ref_idx", type=int, nargs='?', default=None,
                 help = "Index of reference frame. Leave None for random index" )
    data_parser.add_argument( "--device", type=str, default="cuda" )

    args = parser.parse_args()
    action = args._action
    del args._action
    action( **vars(args) )
