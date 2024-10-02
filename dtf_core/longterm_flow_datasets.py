# Inspired from https://github.com/princeton-vl/RAFT.git

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

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
    Sintel       = 0
    FlyingThings = 1

class SintelType(Enum):
    Clean = 0
    Final = 1

class LTFlowDataset(data.Dataset):
    def __init__( self, max_seq_len=None ):
        # Lists of lists
        self.seq_list = []
        self.flow_list = []
        self.occ_list = None

    def __getitem__( self, idx ):
        if idx >= len(self):
            raise IndexError

        img_folder = self.seq_list[idx]
        flow_folder = self.flow_list[idx]
        if self.occ_list is not None:
            occ_folder = self.occ_list[idx]

        img_list = []
        for ext in ['jpg','png','ppm','webp']:
            img_list += sorted(glob(osp.join(
                img_folder,f"*.{ext}")))
        flow_list = []
        for ext in ['flo','pfm']:
            flow_list += sorted(glob(osp.join(
                flow_folder,f"*.{ext}")))
        if self.occ_list is not None:
            occ_list = []
            for ext in ['png','npy']:
                occ_list += sorted(glob(osp.join(
                    occ_folder,f"*.{ext}")))
        if len(img_list) == len(flow_list):
            flow_list = flow_list[:-1]
        else:
            assert len(img_list) == len(flow_list)+1

        imgs = np.stack([
                np.array(frame_utils.read_gen(img)).astype(np.float32)[...,:3]
                for img in img_list ],
            axis=0)
        flows = np.stack([
                np.array(frame_utils.read_gen(flo)).astype(np.float32)
                for flo in flow_list ],
            axis=0)
        if self.occ_list is not None:
            occs = []
            for occ in occ_list:
                if os.path.splitext(occ) == ".npy":
                    raise NotImplementedError()
                else:
                    occ = cv2.imread(occ,cv2.IMREAD_GRAYSCALE)
                occs.append(occ)
            occs = np.stack(occs, axis=0)

        imgs = torch.from_numpy(imgs).permute(0,3,1,2)
        flows = torch.from_numpy(flows).permute(0,3,1,2)
        if self.occ_list is not None:
            occs = torch.from_numpy(occs)
        else:
            occs = None

        T, C, H, W = imgs.shape
        if H%8 != 0 or W%8 != 0:
            imgs = imgs[:,:,:H-H%8,:W-W%8]
            flows = flows[:,:,:H-H%8,:W-W%8]
            if occs is not None:
                occs = occs[:,:H-H%8,:W-W%8]

        if occs.dtype == torch.uint8:
            occs = occs.to(torch.float32)/255.

        return imgs, flows, occs

    def __len__( self ):
        return len(self.seq_list)


class MpiSintel(LTFlowDataset):
    def __init__( self, root_path, stype ):
        super(MpiSintel,self).__init__()
        self.seq_list = sorted(
            glob(os.path.join(root_path,
                "clean" if stype==SintelType.Clean else "final",
                "*")) )
        self.flow_list = sorted(
            glob(os.path.join(root_path,"flow","*")) )
        self.occ_list = sorted(
            glob(os.path.join(root_path,"occlusions","*")) )

class FlyingThings(LTFlowDataset):
    def __init__( self, root_path ):
        super(FlyingThings,self).__init__()
        self.seq_list = sorted(
            glob(os.path.join(root_path,
                "frames_cleanpass","TEST","*","*","*")))
        self.flow_list = sorted(
            glob(os.path.join(root_path,
                "optical_flow","TEST","*","*","into_future","*")))
        self.occ_list = None
