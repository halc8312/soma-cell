# coding: utf-8
"""Pythonista companion for SOMA-CELL 0.6.8-GPU A3.

The iPhone executes the frozen full-detail 0.6.6 CPU reference Scene.  It does
not execute the A3 Torch/CUDA kernels and is not an A3 performance benchmark.
"""
from __future__ import division

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for rel in ('.', '../0_6_6', '../0_6_5', '../0_6_4', '../0_6_3',
            '../0_6_2', '../0_6_1', '../0_6', '../0_6_p2', '../0_6_p1',
            '../0_6_p0', '../baseline'):
    path = os.path.abspath(os.path.join(HERE, rel))
    if path not in sys.path:
        sys.path.insert(0, path)

import SOMA_CELL_0_6_6_pythonista as reference

BUILD = 'SOMA-CELL 0.6.8-GPU A3 COMPANION'
REFERENCE_BUILD = reference.BUILD
GPU_EXECUTION = False
FULL_GPU_WORLD_STEP = False


if __name__ == '__main__':
    print(BUILD)
    print('Interactive Scene uses frozen {} CPU particle reference.'.format(
        REFERENCE_BUILD))
    print('A3 Torch/CUDA kernels run on the PC; no GPU claim is made on iPhone.')
    reference._run_scene()
