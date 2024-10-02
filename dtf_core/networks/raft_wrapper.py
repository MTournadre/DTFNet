import torch
import torch.nn as nn
import sys
import os

from dataclasses import dataclass
import argparse

from RAFT.core import raft

_raft_path = raft.__file__
sys.path.append( os.path.dirname(_raft_path) )

@dataclass
class RAFTArgs:
    dropout: bool = False
    corr_levels: int = 4
    corr_radius: int = 4
    alternate_corr: bool = False
    small: bool = False
    mixed_precision: bool = False

class RAFTWrapper(raft.RAFT):
    def __init__( self, small: bool = False ):
        #TODO adapt other attributes according to 'small'
        args = RAFTArgs(small=small)
        args = argparse.Namespace( **vars(args) )
        super(RAFTWrapper,self).__init__(args)
