# Inspired from https://github.com/princeton-vl/RAFT.git

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

from torchvision.transforms import ColorJitter
from PIL import Image

import os
import math
import random
from glob import glob
import os.path as osp
from enum import Enum
from typing import Optional
import gc

from .raft_utils import frame_utils

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class Dataset(Enum):
    MOVI_A = 0
    MOVI_B = 1
    MOVI_C = 2
    MOVI_D = 3
    MOVI_E = 4
    MOVI_E_BIG = 5
    MOVI_F = 6
    POINT_ODYSSEY = 7

class Split(Enum):
    TRAINING = 0
    VALIDATION = 1

class TrajAugmentor:
    def __init__( self,
                  crop_size=(256,256),
                  min_scale=-0.2,
                  max_scale=0.5,
                  do_flip=True):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.stretch_min = 0.2
        self.stretch_max = 0.3
        self.max_mvt = 0. # max norm of linear movement, per frame

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1
        self.t_flip_prob = 0.3

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, imgs):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            imgs = np.stack(
                    [ np.array(self.photo_aug(Image.fromarray(img)), dtype=np.uint8)
                        for img in imgs ],
                    axis=0)

        # symmetric
        else:
            shape = imgs.shape
            image_stack = np.concatenate(imgs, axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            imgs = image_stack.reshape(shape)

        return imgs

    def eraser_transform(self, imgs, trajs, vis, ref_idx, bounds=[50, 100]):
        """
        Occlusion augmentation
        Randomly add static rectangles over one or several frames.
        The reference frame cannot be occluded
        """

        S, H, W = imgs.shape[:3]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(imgs.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 5)):
                x0 = np.random.randint(0, W)
                y0 = np.random.randint(0, H)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                t0 = np.random.randint(0,S-1)
                if t0 >= ref_idx:
                    t0 = t0+1
                    t1 = np.random.randint(t0+1,S+1)
                else:
                    t1 = np.random.randint(t0+1,ref_idx+1)
                imgs[t0:t1, y0:y0+dy, x0:x0+dx, :] = mean_color
                if trajs.ndim == 4:
                    occ_idx = np.where(
                        np.logical_and(np.logical_and(np.logical_and(
                            trajs[t0:t1,:,:,0] > x0,
                            trajs[t0:t1,:,:,0] < x0+dx),
                            trajs[t0:t1,:,:,1] > y0),
                            trajs[t0:t1,:,:,1] < y0+dy))
                    vis[t0:t1][occ_idx] = 0.
                else:
                    occ_idx = np.where(
                        np.logical_and(np.logical_and(np.logical_and(
                            trajs[t0:t1,:,0] > x0,
                            trajs[t0:t1,:,0] < x0+dx),
                            trajs[t0:t1,:,1] > y0),
                            trajs[t0:t1,:,1] < y0+dy))
                    vis[t0:t1][occ_idx] = 0.

        return imgs, vis

    def spatial_transform(self, imgs, trajs, vis, ref_idx):
        # randomly sample scale
        S, H, W = imgs.shape[:3]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(H),
            (self.crop_size[1] + 8) / float(W))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.stretch_min, 2*self.stretch_max)
            scale_y *= 2 ** np.random.uniform(-self.stretch_min, 2*self.stretch_max)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            imgs = np.stack(
                    [cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                        for img in imgs],
                    axis=0 )

            if trajs.ndim == 4:
                trajs = np.stack(
                        [cv2.resize(traj, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                            for traj in trajs],
                        axis=0 )
                trajs *= np.array([scale_x, scale_y])[None,None,None]
                #vis = np.moveaxis( cv2.resize(
                #        np.moveaxis(vis,0,-1), None,
                #        fx=scale_x, fy=scale_y,
                #        interpolation=cv2.INTER_LINEAR),
                #        -1, 0 )
                vis = np.stack(
                        [cv2.resize(v, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                            for v in vis],
                        axis=0 )
            else:
                trajs *= np.array([scale_x, scale_y])[None,None]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                imgs = imgs[:, :, ::-1, :]
                if trajs.ndim == 4:
                    trajs = trajs[:, :, ::-1, :]
                    trajs[:,:,:,0] = trajs.shape[2] - trajs[:,:,:,0].copy()
                    vis = vis[:, :, ::-1]
                else:
                    trajs[:,:,0] = imgs.shape[2] - trajs[:,:,0].copy()

            if np.random.rand() < self.v_flip_prob: # v-flip
                imgs = imgs[:, ::-1, :, :]
                if trajs.ndim == 4:
                    trajs = trajs[:, ::-1, :, :]
                    trajs[:,:,:,1] = trajs.shape[1] - trajs[:,:,:,1].copy()
                    vis = vis[:, ::-1, :]
                else:
                    trajs[:,:,1] = imgs.shape[1] - trajs[:,:,1].copy()

            if np.random.rand() < self.t_flip_prob: # t-flip
                imgs = imgs[::-1, :, :, :]
                ref_idx = S-1-ref_idx
                if trajs.ndim == 4:
                    trajs = trajs[::-1, :, :, :]
                    vis = vis[::-1, :, :]
                else:
                    trajs = trajs[::-1]
                    vis = vis[::-1]

        y0 = 0 if self.crop_size[0] >= imgs.shape[1] else np.random.randint(
                0, imgs.shape[1] - self.crop_size[0])
        x0 = 0 if self.crop_size[1] >= imgs.shape[2] else np.random.randint(
                0, imgs.shape[2] - self.crop_size[1])

        # Apply random linear camera movement
        cropped_imgs = np.zeros((S,self.crop_size[0],self.crop_size[1],3),dtype=np.uint8)
        if trajs.ndim == 4:
            cropped_trajs = np.zeros((S,self.crop_size[0],self.crop_size[1],2),dtype=np.float32)
            cropped_vis = np.zeros((S,self.crop_size[0],self.crop_size[1]),dtype=np.float32)
        else:
            remaining_trajs = np.where(
                    np.logical_and(np.logical_and(np.logical_and(
                        trajs[ref_idx,:,0] > x0,
                        trajs[ref_idx,:,0] < x0+self.crop_size[1]),
                        trajs[ref_idx,:,1] > y0),
                        trajs[ref_idx,:,1] < y0+self.crop_size[0]))
            cropped_trajs = trajs[:,remaining_trajs,:]
            cropped_vis = vis[:,remaining_trajs,:]
        mvt_norm = random.uniform(0.,self.max_mvt)
        mvt_angle = random.uniform(0.,2*math.pi)
        mvt = mvt_norm * np.array([math.cos(mvt_angle),math.sin(mvt_angle)])

        for t in range(S):
            # Crop with linear movement
            # Add padding of necessary (considered as occlusion)
            d_xy = np.round( (t-ref_idx.item())*mvt ).astype(int)
            crop_corner0 = d_xy + [x0,y0]
            crop_corner1 = crop_corner0 + [self.crop_size[1],self.crop_size[0]]
            sample_corner0 = np.clip(crop_corner0,0,imgs.shape[2:0:-1])
            pad_corner0 = sample_corner0 - crop_corner0
            sample_corner1 = np.clip(crop_corner1,0,imgs.shape[2:0:-1])
            #pad_corner1 = crop_corner1 - sample_corner1
            sample_size = sample_corner1 - sample_corner0

            cropped_imgs[t,
                         pad_corner0[1]:pad_corner0[1]+sample_size[1],
                         pad_corner0[0]:pad_corner0[0]+sample_size[0]] = \
                    imgs[t,
                         sample_corner0[1]:sample_corner1[1],
                         sample_corner0[0]:sample_corner1[0]]

            if trajs.ndim == 4:
                cropped_trajs[t] = trajs[t,y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]] \
                        - crop_corner0[np.newaxis,np.newaxis,:]
                cropped_vis[t] = vis[t,y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            else:
                cropped_trajs[t] -= crop_corner0[np.newaxis]
        imgs = cropped_imgs
        trajs = cropped_trajs
        vis = cropped_vis

        # Some points can now be outside of the resulting image
        if trajs.ndim == 4:
            out_idx = np.where( np.logical_or( np.logical_or( np.logical_or(
                trajs[:,:,:,0] < 0.,
                trajs[:,:,:,0] > self.crop_size[1]),
                trajs[:,:,:,1] < 0.),
                trajs[:,:,:,1] > self.crop_size[0]) )
            vis[out_idx] = 0.
        else:
            out_idx = np.where( np.logical_or( np.logical_or( np.logical_or(
                trajs[:,:,0] < 0.,
                trajs[:,:,0] > self.crop_size[1]),
                trajs[:,:,1] < 0.),
                trajs[:,:,1] > self.crop_size[0]) )
            vis[out_idx] = 0.

        if False:
            # Fix coordinates of reference pixels to match
            # perfectly the center of pixels
            grid = np.stack( np.meshgrid(
                    np.arange(self.crop_size[0],dtype=np.float32),
                    np.arange(self.crop_size[1],dtype=np.float32) ),
                    axis=-1 ) + 0.5
            correction = grid - trajs[ref_idx]
            trajs += correction[None]

            vis[ref_idx] = 1.

        gc.collect(0)

        return imgs, trajs, vis, ref_idx

    def __call__(self, imgs, trajs, vis, ref_idx):
        imgs = self.color_transform(imgs)
        imgs, vis = self.eraser_transform(imgs, trajs, vis, ref_idx)
        imgs, trajs, vis, ref_idx = self.spatial_transform(
                imgs, trajs, vis, ref_idx)

        if self.sparse:
            if self.nb_sample_trajs is not None:
                # Randomly sample a fixed number of trajectories for training
                if len(trajs) > self.nb_sample_trajs:
                    sampling = list(range(len(trajs)))
                    random.shuffle(sampling)
                    sampling = sampling[:self.nb_sample_trajs]
                    trajs = trajs[:,sampling]
                    vis = vis[:,sampling]
                    valid = np.ones((self.nb_sample_trajs),dtype=np.float32)
                else:
                    valid = np.concatenate((
                        np.ones(len(trajs),dtype=np.float32),
                        np.zeros(self.nb_sample_trajs-len(trajs),dtype=np.float32) ))
        else:
            valid = None

        imgs = np.ascontiguousarray(imgs)
        trajs = np.ascontiguousarray(trajs)
        vis = np.ascontiguousarray(vis)

        gc.collect(0)

        return imgs, trajs, vis, None, ref_idx

class DenseTrajAugmentor(TrajAugmentor):
    def __init__( self, **kwargs ):
        super(DenseTrajAugmentor,self).__init__(**kwargs)
        self.sparse = False

class SparseTrajAugmentor(TrajAugmentor):
    def __init__( self, **kwargs ):
        super(SparseTrajAugmentor,self).__init__(**kwargs)
        self.sparse = True


class TrajDataset(data.Dataset):
    def __init__(self, aug_params=None, seq_len=12, downscale=1):
        self.augmentor = None
        if aug_params is not None:
            self.augmentor = DenseTrajAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.sample_list = []
        self.seq_len = seq_len
        self.downscale = downscale

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError

        #index = index % len(self.sample_list)
        sample_path = self.sample_list[index].decode('utf-8')
        if self.is_test:
            # Load all trajectory data
            traj_data = np.load(osp.join(sample_path,"traj_data.npz"))
            trajs = torch.from_numpy(traj_data['traj'])
            vis = torch.from_numpy(traj_data['vis']).float()
            ref_idx = torch.from_numpy(traj_data['ref_idx'])

            # Fix sequence length
            # In this special test case, we load from the highest valid index
            S = trajs.shape[0]
            if self.seq_len is None:
                seq_len = S
            else:
                assert S >= self.seq_len
                seq_len = self.seq_len
            idx0 = min(ref_idx.item(),S-seq_len)
            trajs = trajs[idx0:idx0+seq_len]
            vis = vis[idx0:idx0+seq_len]
            ref_idx = ref_idx-idx0

            # Only load needed images
            img_list = []
            for ext in ['jpg','png','ppm','webp']:
                img_list += sorted(glob(osp.join(
                    sample_path,f"*.{ext}")))
            img_list = img_list[idx0:idx0+seq_len]
            imgs = np.stack([
                    np.array(frame_utils.read_gen(img)).astype(np.uint8)[...,:3]
                    for img in img_list ],
                axis=0)
            imgs = torch.from_numpy(imgs).permute(0,3,1,2).float()

            trajs = trajs.permute(0,3,1,2) # SHWC -> SCHW
            vis = vis.unsqueeze(1) # SHW -> SCHW

            return imgs, trajs, vis, ref_idx

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            seed = worker_info.id if worker_info is not None else 0
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            self.init_seed = True

        # Load all trajectory data
        traj_data = np.load(osp.join(sample_path,"traj_data.npz"))
        trajs = traj_data['traj']
        vis = traj_data['vis'].astype(np.float32)
        ref_idx = torch.from_numpy(traj_data['ref_idx'])

        # Fix sequence length
        # Randomly pick a starting index
        # (the reference frame must always be in the resulting sub-sequence)
        S = trajs.shape[0]
        if self.seq_len is None:
            seq_len = S
        else:
            assert S >= self.seq_len, \
                f"The sample '{sample_path}' is shorter than the expected length"
            seq_len = self.seq_len
        min_idx0 = max(0,ref_idx-seq_len+1)
        max_idx0 = min(ref_idx.item(),S-seq_len)
        idx0 = np.random.randint(min_idx0,max_idx0+1)
        trajs = trajs[idx0:idx0+seq_len].copy()
        vis = vis[idx0:idx0+seq_len].copy()
        ref_idx = ref_idx-idx0

        del traj_data.f
        traj_data.close()
        del traj_data

        # Only load needed images
        img_list = []
        for ext in ['jpg','png','ppm','webp']:
            img_list += sorted(glob(osp.join(
                sample_path,f"*.{ext}")))
        img_list = img_list[idx0:idx0+seq_len]
        imgs = np.stack([
                np.array(frame_utils.read_gen(img)).astype(np.uint8)[...,:3]
                for img in img_list ],
            axis=0)

        if self.augmentor is not None:
            # Fix augmentor size if higher than image size...
            S, H, W, _ = imgs.shape
            if self.augmentor.crop_size[0] > H:
                self.augmentor.crop_size = H, self.augmentor.crop_size[1]
            if self.augmentor.crop_size[1] > W:
                self.augmentor.crop_size = self.augmentor.crop_size[0], W

            imgs, trajs, vis, valid, ref_idx = self.augmentor(imgs,trajs,vis,ref_idx)
        else:
            valid = None

        imgs = torch.from_numpy(imgs).permute(0,3,1,2).float() # SHWC -> SCHW
        trajs = torch.from_numpy(trajs).permute(0,3,1,2) # SHWC -> SCHW
        vis = torch.from_numpy(vis).unsqueeze(1) # SHW -> SCHW

        if self.downscale != 1:
            imgs = torch.nn.functional.avg_pool2d(imgs,self.downscale)
            trajs = torch.nn.functional.avg_pool2d(trajs,self.downscale)/self.downscale
            vis = torch.nn.functional.avg_pool2d(vis,self.downscale)

        #assert trajs.isfinite().all() and imgs.isfinite().all() and vis.isfinite().all(), \
        #        f"Invalid values found in sample '{sample_path}'"

        gc.collect(0)

        if valid is None:
            valid = np.zeros((),dtype=np.float32)
        return imgs, trajs, vis, valid, ref_idx


    def __rmul__(self, v):
        self.sample_list = np.concatenate(v * [self.sample_list])
        return self

    def __len__(self):
        return len(self.sample_list)

class MoviDatasetBase(TrajDataset):
    def __init__( self, aug_params, dataset_root, seq_len, split, val_ratio=None, downscale=1 ):
        super(MoviDatasetBase,self).__init__(aug_params=aug_params,seq_len=seq_len,downscale=downscale)

        if split == Split.TRAINING:
            train_samples = sorted(glob(
                osp.join(dataset_root,"train","*","traj_data.npz")))
            train_samples = [ osp.dirname(s) for s in train_samples ]
            if val_ratio is not None:
                nb_val = int(val_ratio*len(train_samples))
                self.sample_list = train_samples[nb_val:]
            else:
                self.sample_list = train_samples

        elif split == Split.VALIDATION:
            if val_ratio is None:
                val_samples = sorted(glob(
                    osp.join(dataset_root,"test","*","traj_data.npz")))
                val_samples = [ osp.dirname(s) for s in val_samples ]
                self.sample_list = val_samples
            else:
                # All samples (train+validation) are in the same 'train' folder
                # We split it according to val_ratio
                all_samples = sorted(glob(
                    osp.join(dataset_root,"train","*","traj_data.npz")))
                all_samples = [ osp.dirname(s) for s in all_samples ]
                nb_val = int(val_ratio*len(all_samples))
                self.sample_list = all_samples[:nb_val]

        # This fixes the "memory leak" caused by Python multiprocessing
        self.sample_list = np.char.encode( self.sample_list, encoding='utf-8' )

class MoviADataset(MoviDatasetBase):
    def __init__( self, aug_params, dataset_root, seq_len, split=Split.TRAINING, downscale=1 ):
        super(MoviADataset,self).__init__(
                aug_params = aug_params,
                dataset_root = osp.join(dataset_root,"movi_a"),
                seq_len = seq_len,
                val_ratio = 0.10,
                split = split,
                downscale = downscale,
                )

class MoviBDataset(MoviDatasetBase):
    def __init__( self, aug_params, dataset_root, seq_len, split=Split.TRAINING, downscale=1 ):
        super(MoviBDataset,self).__init__(
                aug_params = aug_params,
                dataset_root = osp.join(dataset_root,"movi_b"),
                seq_len = seq_len,
                val_ratio = 0.12,
                split = split,
                downscale = downscale,
                )

class MoviCDataset(MoviDatasetBase):
    def __init__( self, aug_params, dataset_root, seq_len, split=Split.TRAINING, downscale=1 ):
        super(MoviCDataset,self).__init__(
                aug_params = aug_params,
                dataset_root = osp.join(dataset_root,"movi_c"),
                seq_len = seq_len,
                val_ratio = None,
                split = split,
                downscale = downscale,
                )

class MoviDDataset(MoviDatasetBase):
    def __init__( self, aug_params, dataset_root, seq_len, split=Split.TRAINING, downscale=1 ):
        super(MoviDDataset,self).__init__(
                aug_params = aug_params,
                dataset_root = osp.join(dataset_root,"movi_d"),
                seq_len = seq_len,
                val_ratio = None,
                split = split,
                downscale = downscale,
                )

class MoviEDataset(MoviDatasetBase):
    def __init__( self, aug_params, dataset_root, seq_len, split=Split.TRAINING, downscale=1 ):
        super(MoviEDataset,self).__init__(
                aug_params = aug_params,
                dataset_root = osp.join(dataset_root,"movi_e"),
                seq_len = seq_len,
                val_ratio = None,
                split = split,
                downscale = downscale,
                )

class MoviEBigDataset(MoviDatasetBase):
    def __init__( self, aug_params, dataset_root, seq_len, split=Split.TRAINING, downscale=1 ):
        super(MoviEBigDataset,self).__init__(
                aug_params = aug_params,
                dataset_root = osp.join(dataset_root,"movi_e_big"),
                seq_len = seq_len,
                val_ratio = None,
                split = split,
                downscale = downscale,
                )


class MoviFDataset(MoviDatasetBase):
    def __init__( self, aug_params, dataset_root, seq_len, split=Split.TRAINING, downscale=1 ):
        super(MoviFDataset,self).__init__(
                aug_params = aug_params,
                dataset_root = osp.join(dataset_root,"movi_f"),
                seq_len = seq_len,
                val_ratio = 0.08,
                split = split,
                downscale = downscale,
                )

def training_dataloader(
        stage: Dataset,
        datasets_root: Optional[os.PathLike] = 'datasets',
        image_size: (int,int) = (256, 256),
        seq_len: int = 8,
        batch_size: int = 4,
        num_workers: int = 4,
        downscale: int = 1,
        verbose: bool = True,
        ):
    aug_params = { 'crop_size': image_size }
    if stage == Dataset.MOVI_A:
        dataset = MoviADataset(
                aug_params = aug_params,
                dataset_root = datasets_root,
                seq_len = seq_len,
                split = Split.TRAINING,
                downscale = downscale )
    elif stage == Dataset.MOVI_B:
        dataset = MoviBDataset(
                aug_params = aug_params,
                dataset_root = datasets_root,
                seq_len = seq_len,
                split = Split.TRAINING,
                downscale = downscale )
    elif stage == Dataset.MOVI_C:
        dataset = MoviCDataset(
                aug_params = aug_params,
                dataset_root = datasets_root,
                seq_len = seq_len,
                split = Split.TRAINING,
                downscale = downscale )
    elif stage == Dataset.MOVI_D:
        dataset = MoviDDataset(
                aug_params = aug_params,
                dataset_root = datasets_root,
                seq_len = seq_len,
                split = Split.TRAINING,
                downscale = downscale )
    elif stage == Dataset.MOVI_E:
        dataset = MoviEDataset(
                aug_params = aug_params,
                dataset_root = datasets_root,
                seq_len = seq_len,
                split = Split.TRAINING,
                downscale = downscale )
    elif stage == Dataset.MOVI_E_BIG:
        dataset = MoviEBigDataset(
                aug_params = aug_params,
                dataset_root = datasets_root,
                seq_len = seq_len,
                split = Split.TRAINING,
                downscale = downscale )
    elif stage == Dataset.MOVI_F:
        dataset = MoviFDataset(
                aug_params = aug_params,
                dataset_root = datasets_root,
                seq_len = seq_len,
                split = Split.TRAINING,
                downscale = downscale )

    dataloader = data.DataLoader(dataset,
            batch_size=batch_size,
            pin_memory=False,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            persistent_workers=False)

    if verbose:
        print('Training with %d image sequences' % len(dataset))
    return dataloader

class PointOdyssey(data.Dataset):
    """
    That's a WIP. This dataset is not supported yet.
    """
    def __init__( self,
                 dataset_root: os.PathLike,
                 split: Split = Split.VALIDATION,
                 nb_sample_trajs: Optional[int] = None,
                 aug_params: Optional[SparseTrajAugmentor] = None,
                 seq_len: int = 12,
                 downscale = 1,
                 timestep = 1 ):
        self.augmentor = None
        if aug_params is not None:
            self.augmentor = SparseTrajAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.seq_len = seq_len
        self.downscale = downscale
        self.nb_sample_trajs = nb_sample_trajs

        self.all_seq_length = 2600 #TODO check me, or detect it
        self.all_sequences = sorted( glob( os.path.join( dataset_root,
                "val" if split == Split.VALIDATION else "train",
                "annot.npz" )))

    def __len__( self ):
        return self.all_seq_length*len(self.all_sequences)

    def __getitem__( self, index ):
        if index >= len(self):
            raise IndexError

        seq_idx = index//self.all_seq_length
        frame_idx = index%self.all_seq_length
        seq_dir = os.path.dirname(self.all_sequences[seq_idx])

        #TODO random timestep ?

        if self.augmentor is None:
            begin_idx = max(0, min(self.all_seq_length-self.seq_len, frame_idx-self.seq_len//2))
        else:
            # When perturbing data, choose random time cropping
            begin_idx = random.randint(
                max(0,min(self.all_seq_length-self.seq_len,frame_idx-self.seq_len+1)),
                max(0,min(self.all_seq_length-self.seq_len,frame_idx)) )
        end_idx = begin_idx + self.seq_len
        ref_idx = frame_idx - begin_idx

        # Read trajectory data
        annots = np.load(self.all_sequences[seq_idx])
        trajs = annots["trajs_2d"][begin_idx:end_idx]
        vis = annots["visibs"][begin_idx:end_idx]
        valids = annots["valids"][begin_idx:end_idx]

        del annots.f
        annots.close()
        del annots

        # Select valid trajectories only
        valid_trajs = valids.all(axis=0)
        trajs = trajs[:,valid_trajs]
        vis = vis[:,valid_trajs]

        # Select trajectories where reference point is visible
        vis_trajs = vis[ref_idx]
        trajs = trajs[vis_trajs,:]
        vis = vis[vis_trajs,:]

        # Read images
        img_list = [ os.path.join(seq_dir,rgbs,f"rgb_{idx:05d}.jpg")
                     for idx in range(begin_idx,end_idx) ]
        imgs = np.stack([
                np.array(frame_utils.read_gen(img)).astype(np.uint8)[...,:3]
                for img in img_list ],
            axis=0)

        if self.augmentor is not None:
            # Fix augmentor size if higher than image size...
            S, H, W, _ = imgs.shape
            if self.augmentor.crop_size[0] > H:
                self.augmentor.crop_size = H, self.augmentor.crop_size[1]
            if self.augmentor.crop_size[1] > W:
                self.augmentor.crop_size = self.augmentor.crop_size[0], W

            imgs, trajs, vis, ref_idx = self.augmentor(imgs,trajs,vis,ref_idx)

        imgs = torch.from_numpy(imgs).permute(0,3,1,2).float() # SHWC -> SCHW
        trajs = torch.from_numpy(trajs).permute(0,2,1) # SNC -> SCN
        vis = torch.from_numpy(vis).unsqueeze(1) # SN -> SCN

        if self.downscale != 1:
            imgs = torch.nn.functional.avg_pool2d(imgs,self.downscale)
            trajs /= self.downscale

        #assert trajs.isfinite().all() and imgs.isfinite().all() and vis.isfinite().all(), \
        #        f"Invalid values found in sample '{sample_path}'"

        gc.collect(0)

        return imgs, trajs, vis, ref_idx

def validation_dataloaders(
        datasets: list[Dataset],
        datasets_root: Optional[os.PathLike] = 'datasets',
        #image_size: (int,int) = (256, 256),
        seq_len: int = 8,
        batch_size: int = 4,
        num_workers: int = 4,
        downscale: int = 1,
        ):

    #aug_params = { 'crop_size': image_size }
    aug_params = None
    all_datasets = []
    for ds in datasets:
        if ds == Dataset.MOVI_A:
            all_datasets.append((
                MoviADataset(
                    aug_params = aug_params,
                    dataset_root = datasets_root,
                    seq_len = seq_len,
                    split = Split.VALIDATION,
                    downscale = downscale ),
                ds ))
        elif ds == Dataset.MOVI_B:
            all_datasets.append((
                MoviBDataset(
                    aug_params = aug_params,
                    dataset_root = datasets_root,
                    seq_len = seq_len,
                    split = Split.VALIDATION,
                    downscale = downscale ),
                ds ))
        elif ds == Dataset.MOVI_C:
            all_datasets.append((
                MoviCDataset(
                    aug_params = aug_params,
                    dataset_root = datasets_root,
                    seq_len = seq_len,
                    split = Split.VALIDATION,
                    downscale = downscale ),
                ds ))
        elif ds == Dataset.MOVI_D:
            all_datasets.append((
                MoviDDataset(
                    aug_params = aug_params,
                    dataset_root = datasets_root,
                    seq_len = seq_len,
                    split = Split.VALIDATION,
                    downscale = downscale ),
                ds ))
        elif ds == Dataset.MOVI_E:
            all_datasets.append((
                MoviEDataset(
                    aug_params = aug_params,
                    dataset_root = datasets_root,
                    seq_len = seq_len,
                    split = Split.VALIDATION,
                    downscale = downscale ),
                ds ))
        elif ds == Dataset.MOVI_E_BIG:
            all_datasets.append((
                MoviEBigDataset(
                    aug_params = aug_params,
                    dataset_root = datasets_root,
                    seq_len = seq_len,
                    split = Split.VALIDATION,
                    downscale = downscale ),
                ds ))
        elif ds == Dataset.MOVI_F:
            all_datasets.append((
                MoviFDataset(
                    aug_params = aug_params,
                    dataset_root = datasets_root,
                    seq_len = seq_len,
                    split = Split.VALIDATION,
                    downscale = downscale ),
                ds ))

    return [ ( data.DataLoader(
                   dataset, batch_size=batch_size,
                   pin_memory=False,
                   shuffle=False,
                   num_workers=num_workers,
                   drop_last=False),
               name )
            for (dataset,name) in all_datasets ]
