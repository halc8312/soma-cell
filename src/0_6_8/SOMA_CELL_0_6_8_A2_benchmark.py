# coding: utf-8
from __future__ import division
import argparse,copy,json,time
import numpy as np
import torch
import SOMA_CELL_0_6_8_gpu_a2 as a2


def make_world(seed,cells):
    cfg=a2.s66.shared_ecology_config(environment=a2.s66.ENV_STABLE,mutation=True,hgt=True,eco66_washout=False,eco66_chemostat=False)
    return a2.s66.Formal066World(seed=seed,initial_cells=cells,config=cfg)

def max_diff(a,b):
    values=[]
    if len(a.field.pos):values.append(np.max(np.abs(a.field.pos-b.field.pos)))
    if len(a.field.amount):values.append(np.max(np.abs(a.field.amount-b.field.amount)))
    for x,y in zip(a.cells,b.cells):
        values.extend([np.max(np.abs(x.pos-y.pos)),np.max(np.abs(x.vel-y.vel)),abs(x.radius-y.radius),np.max(np.abs(x.pools-y.pools)),np.max(np.abs(x.membrane-y.membrane)),np.max(np.abs(x.surface_flux-y.surface_flux))])
    return float(max(values or [0.0]))

def run(steps=20,cells=3,seed=101,device='cpu',precision='float64'):
    cpu=make_world(seed,cells);hybrid_world=a2.s66.Formal066World.from_state(copy.deepcopy(cpu.state_dict()))
    hybrid=a2.Hybrid066WorldA2(hybrid_world,{'device':device,'precision':precision})
    dt=.1
    t=time.perf_counter()
    for _ in range(steps):cpu.step(dt)
    cpu_seconds=time.perf_counter()-t
    if device.startswith('cuda'):torch.cuda.synchronize()
    t=time.perf_counter()
    for _ in range(steps):hybrid.step(dt)
    if device.startswith('cuda'):torch.cuda.synchronize()
    hybrid_seconds=time.perf_counter()-t
    return {
        'build':a2.BUILD,'schema':a2.SCHEMA_VERSION,'device':device,'precision':precision,
        'torch_version':torch.__version__,'cuda_available':bool(torch.cuda.is_available()),
        'steps':steps,'cells':cells,'particles_final':len(cpu.field.amount),
        'cpu_seconds':cpu_seconds,'hybrid_seconds':hybrid_seconds,
        'hybrid_over_cpu_ratio':hybrid_seconds/max(cpu_seconds,1e-12),
        'max_lockstep_abs_diff':max_diff(cpu,hybrid.world),
        'rng_equal':cpu.rng.bit_generator.state==hybrid.world.rng.bit_generator.state,
        'material_residual_cpu':cpu.matter_ledger_residual(),
        'material_residual_hybrid':hybrid.world.matter_ledger_residual(),
        'backend_stats':hybrid.backend.stats(),
        'full_gpu_world_step':False,
        'note':'Correctness-first per-cell sequential surface scan; performance is not the A2 acceptance claim.'
    }

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=20);p.add_argument('--cells',type=int,default=3);p.add_argument('--seed',type=int,default=101);p.add_argument('--device',default='cpu');p.add_argument('--precision',default='float64');p.add_argument('--output',default='soma_cell_0_6_8_gpu_a2_benchmark_cpu.json');a=p.parse_args()
    result=run(a.steps,a.cells,a.seed,a.device,a.precision)
    with open(a.output,'w',encoding='utf-8') as f:json.dump(result,f,indent=2,sort_keys=True)
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
