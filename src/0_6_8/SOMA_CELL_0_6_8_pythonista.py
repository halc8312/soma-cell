# coding: utf-8
"""Pythonista companion for SOMA-CELL 0.6.8-GPU.

Pythonista/iOS does not provide the PyTorch CUDA runtime.  This companion opens
the frozen full-detail SOMA-CELL 0.6.6 particle reference world for observation
and small interventions.  It deliberately does not pretend to execute GPU
kernels.  PC/GPU experiments are launched with SOMA_CELL_0_6_8_gpu.py.
"""
from __future__ import division
import os,sys
HERE=os.path.dirname(os.path.abspath(__file__))
for rel in ('.','../0_6_6','../0_6_5','../0_6_4','../0_6_3','../0_6_2','../0_6_1','../0_6','../0_6_p2','../0_6_p1','../0_6_p0','../baseline'):
    path=os.path.abspath(os.path.join(HERE,rel))
    if path not in sys.path:sys.path.insert(0,path)
import SOMA_CELL_0_6_6_pythonista as reference
BUILD='SOMA-CELL 0.6.8-GPU COMPANION'
REFERENCE_BUILD=reference.BUILD
GPU_EXECUTION=False

if __name__=='__main__':
    print(BUILD)
    print('Interactive mode uses frozen {} CPU particle reference.'.format(REFERENCE_BUILD))
    print('CUDA execution belongs on the PC runner; no GPU claim is made on iPhone.')
    reference._run_scene()
