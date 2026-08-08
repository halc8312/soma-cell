#!/usr/bin/env python3
# coding: utf-8
"""Benchmark and capacity estimator for SOMA-CELL 0.6.8-GPU."""
from __future__ import division
import argparse,json,os,time
import SOMA_CELL_0_6_8_gpu as gpu


def estimate_schema_bytes(worlds,particles,cells,genome_symbols,precision='float32'):
    fbytes=8 if str(precision).lower() in ('float64','fp64','double') else 4
    ibytes=8; bbytes=1; u8=1
    B=int(worlds);P=int(particles);C=int(cells);G=int(genome_symbols)
    # Mirrors TensorWorldBatch fields. Opaque CPU objects are deliberately not
    # included because they live in host RAM, not VRAM.
    total=0
    total += B*P*2*fbytes + B*P*ibytes + B*P*fbytes + B*P*bbytes
    total += B*4*2*fbytes
    total += B*C*bbytes*2 + B*C*ibytes*4
    total += B*C*2*fbytes*3 + B*C*fbytes
    total += B*C*gpu.MEMBRANE_SEGMENTS*fbytes*2
    total += B*C*gpu.MEMBRANE_SEGMENTS*gpu.CHANNEL_COUNT*fbytes
    total += B*C*gpu.POOL_COUNT*fbytes
    total += B*C*gpu.MEMBRANE_SEGMENTS*2*fbytes
    total += B*C*gpu.MEMBRANE_SEGMENTS*fbytes
    total += B*C*2*G*u8 + B*C*2*ibytes + B*C*ibytes
    total += B*fbytes*3
    return int(total)


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='auto')
    p.add_argument('--precision',default='float32')
    p.add_argument('--worlds',type=int,default=32)
    p.add_argument('--particles',type=int,default=1024)
    p.add_argument('--cells',type=int,default=16)
    p.add_argument('--genome-symbols',type=int,default=1024)
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--output',default='soma_cell_0_6_8_gpu_benchmark.json')
    a=p.parse_args(argv)
    report=gpu.benchmark_kernels(a.worlds,a.particles,a.cells,a.steps,a.device,a.precision)
    report['schema_capacity_bytes']=estimate_schema_bytes(a.worlds,a.particles,a.cells,a.genome_symbols,a.precision)
    report['schema_capacity_gib']=report['schema_capacity_bytes']/(1024.0**3)
    report['note']='Kernel benchmark only; full 0.6.6 world-step is not yet GPU-ported.'
    with open(a.output,'w',encoding='utf-8') as h:json.dump(report,h,indent=2,sort_keys=True);h.write('\n')
    print(json.dumps(report,indent=2,sort_keys=True))
    return report

if __name__=='__main__':main()
