# coding: utf-8
"""Validation for SOMA-CELL 0.6.8-GPU A2."""
from __future__ import division

import copy
import csv
import json
import os
import tempfile
import time

import numpy as np
import torch

import SOMA_CELL_0_6_8_gpu as a1
import SOMA_CELL_0_6_8_gpu_a2 as a2

RESULT_TXT='SOMA_CELL_0_6_8_GPU_A2_VALIDATION_RESULTS.txt'
RESULT_CSV='soma_cell_0_6_8_gpu_a2_validation.csv'


def close(a,b,atol=2e-12,rtol=2e-12,label='value'):
    aa=np.asarray(a);bb=np.asarray(b)
    if not np.allclose(aa,bb,atol=atol,rtol=rtol):
        raise AssertionError('{} max_abs={}'.format(label,float(np.max(np.abs(aa-bb)))))


def make_world(seed=101,cells=2,environment=None):
    environment=environment or a2.s66.ENV_STABLE
    cfg=a2.s66.shared_ecology_config(environment=environment,mutation=True,hgt=True,eco66_washout=False,eco66_chemostat=False)
    return a2.s66.Formal066World(seed=seed,initial_cells=cells,config=cfg)


def state_arrays(world):
    out={'field_pos':np.asarray(world.field.pos),'field_amount':np.asarray(world.field.amount),'field_kind':np.asarray(world.field.kind),'age':world.age,'dissipated':world.dissipated_energy,'divisions':world.divisions,'deaths':world.deaths}
    for i,c in enumerate(world.cells):
        out.update({f'{i}_pos':np.asarray(c.pos),f'{i}_vel':np.asarray(c.vel),f'{i}_radius':c.radius,f'{i}_pools':np.asarray(c.pools),f'{i}_membrane':np.asarray(c.membrane),f'{i}_oxidation':np.asarray(c.membrane_oxidation),f'{i}_transporters':np.asarray(c.transporters),f'{i}_contact':np.asarray(c.contact_trace),f'{i}_damage':np.asarray(c.damage_trace),f'{i}_flux':np.asarray(c.surface_flux)})
    return out


def assert_worlds(cpu,gpu,atol=3e-11):
    a=state_arrays(cpu);b=state_arrays(gpu)
    assert a.keys()==b.keys()
    for k in a:close(a[k],b[k],atol=atol,rtol=atol,label=k)
    assert cpu.rng.bit_generator.state==gpu.rng.bit_generator.state


def test_build_schema_and_honest_scope():
    assert a2.BUILD=='SOMA-CELL 0.6.8-GPU A2'
    assert a2.SCHEMA_VERSION=='0.6.8-GPU-A2.0'
    assert a2.PORT_STATUS['division']=='cpu-authoritative'
    assert a2.environment_report()['full_gpu_world_step'] is False
    return 'A2 identity frozen; full GPU world explicitly false'


def test_spatial_hash_matches_dense():
    rng=np.random.default_rng(1);pos=rng.random((400,2))
    index=a2.ToroidalSpatialHash(pos,bins=24)
    for _ in range(30):
        centre=rng.random(2);radius=float(rng.uniform(.01,.15))
        got=index.query(centre,radius)
        want=np.where(np.linalg.norm(a2.torus_delta_numpy(pos,centre),axis=1)<radius)[0]
        assert np.array_equal(got,want)
    return 'toroidal CPU grid returns exact sorted dense candidates'


def test_spatial_hash_wrap_boundary():
    pos=np.asarray([[.99,.5],[.01,.5],[.5,.5]])
    got=a2.ToroidalSpatialHash(pos,bins=16).query([0,.5],.03)
    assert np.array_equal(got,[0,1])
    return 'spatial grid respects torus boundary'


def test_dense_candidate_mask_torch():
    p=torch.tensor([[[.99,.5],[.01,.5],[.5,.5]]],dtype=torch.float64)
    pm=torch.ones((1,3),dtype=torch.bool);c=torch.tensor([[[0.,.5]]],dtype=torch.float64);r=torch.tensor([[.02]],dtype=torch.float64)
    got=a2.dense_candidate_mask_torch(p,pm,c,r,band=.01).numpy()
    assert np.array_equal(got,np.asarray([[[True,True,False]]]))
    return 'batched torch candidate mask exact'


def test_grid_keys_stable():
    p=torch.tensor([[[.2,.2],[.2,.2],[.9,.9]]],dtype=torch.float64);m=torch.ones((1,3),dtype=torch.bool)
    key,order=a2.spatial_grid_keys_torch(p,m,bins=8)
    assert order[0,0].item()==0 and order[0,1].item()==1
    assert torch.all(key[:,1:]>=key[:,:-1])
    return 'stable grid keys preserve original-index tie order'


def test_closure_osmolyte_tension_match_cell():
    w=make_world(2,1);c=w.cells[0]
    close(a2._closure_numpy_exact(c.membrane,c.membrane_oxidation,c.radius),c.closure_array(),label='closure')
    close(a2._osmolyte_numpy_exact(c.pools,c.genome_mass()),c.osmolyte(),label='osmolyte')
    close(a2._tension_numpy_exact(c.pools,c.membrane,c.genome_mass()),c.tension(),label='tension')
    return 'damage-aware closure and polymer-aware osmosis exact'


def _surface_args(world):
    c=world.cells[0];field=world.field
    idx=a2.ToroidalSpatialHash(field.pos,32).query(c.pos,c.radius+a2.INTERACTION_BAND)
    return c,field,idx,(field.pos,field.kind,field.amount,c.pos,c.radius,c.membrane,c.membrane_oxidation,c.transporters,c.pools,c.contact_trace,c.alt_contact_trace,c.damage_trace,c.surface_flux,.1)


def test_surface_numpy_torch_lockstep():
    w=make_world(3,1);c,f,idx,args=_surface_args(w)
    n=a2.surface_exchange_numpy(*args,transport=True,candidate_indices=idx)
    targs=[torch.as_tensor(x,dtype=torch.int64 if i==1 else torch.float64) if isinstance(x,(np.ndarray,np.generic,float,int)) else x for i,x in enumerate(args[:-1])]
    t=a2.surface_exchange_torch(*targs,args[-1],transport=True,candidate_indices=idx)
    for k in ('particle_pos','particle_amount','pools','contact_trace','alt_contact_trace','damage_trace','surface_flux','last_uptake','last_uptake_alt','atp_spent'):
        tv=t[k].detach().cpu().numpy();close(n[k],tv,atol=3e-14,rtol=3e-14,label='surface '+k)
    return 'sequential surface exchange NumPy/Torch fp64 lockstep'


def test_surface_original_method_lockstep():
    cpu=make_world(4,1);gpu=a2.s66.Formal066World.from_state(copy.deepcopy(cpu.state_dict()))
    cpu.cells[0].surface_exchange(cpu.field,.1,cpu.config)
    backend=a2.TorchKernelBackendA2({'device':'cpu','precision':'float64'})
    backend.surface_exchange_inplace(gpu.field,gpu.cells[0],.1,gpu.config)
    assert_worlds(cpu,gpu,atol=3e-13)
    return 'hybrid surface exchange matches inherited 0.4 post-processing'


def test_surface_transport_off_lockstep():
    cpu=make_world(5,1);cpu.config.transport=False;gpu=a2.s66.Formal066World.from_state(copy.deepcopy(cpu.state_dict()))
    cpu.cells[0].surface_exchange(cpu.field,.1,cpu.config)
    a2.TorchKernelBackendA2({'device':'cpu'}).surface_exchange_inplace(gpu.field,gpu.cells[0],.1,gpu.config)
    assert_worlds(cpu,gpu,atol=3e-13)
    return 'passive gap exchange lockstep when transport disabled'


def test_closed_membrane_particle_push():
    w=make_world(6,1);c=w.cells[0]
    c.membrane[:]=max(c.required_segment_mass()*2.0,.03);c.membrane_oxidation[:]=0
    w.field.pos[0]=(c.pos+np.asarray([0.010,0.0]))%1.0;before=w.field.pos[0].copy()
    a2.TorchKernelBackendA2({'device':'cpu'}).surface_exchange_inplace(w.field,c,.1,w.config)
    assert np.linalg.norm(a2.torus_delta_numpy(w.field.pos[0],c.pos))>np.linalg.norm(a2.torus_delta_numpy(before,c.pos))
    return 'closed membrane geometrically excludes interior particle'


def test_waste_export_numpy_torch():
    w=make_world(7,1);c=w.cells[0];c.pools[a2.s5.POOL_WASTE]=.25;c.pools[a2.s5.POOL_ATP]=.3
    args=(c.pools,c.membrane,c.membrane_oxidation,c.transporters,c.radius,c.pos,c.surface_flux,.1)
    n=a2.waste_export_numpy(*args,enabled=True)
    t=a2.waste_export_torch(*[torch.as_tensor(x,dtype=torch.float64) if isinstance(x,(np.ndarray,float,int)) else x for x in args],enabled=True)
    for k in ('pools','surface_flux','amount','powered','position'):close(n[k],t[k].detach().cpu().numpy(),atol=2e-14,rtol=2e-14,label='export '+k)
    assert n['segment']==t['segment']
    return 'waste export NumPy/Torch lockstep'


def test_waste_export_original_lockstep():
    cpu=make_world(8,1);c=cpu.cells[0];c.pools[a2.s5.POOL_WASTE]=.25;c.pools[a2.s5.POOL_ATP]=.3
    gpu=a2.s66.Formal066World.from_state(copy.deepcopy(cpu.state_dict()))
    cpu.cells[0].export_waste(cpu.field,.1,cpu.config)
    a2.TorchKernelBackendA2({'device':'cpu'}).waste_export_inplace(gpu.field,gpu.cells[0],.1,gpu.config)
    assert_worlds(cpu,gpu,atol=4e-13)
    return 'hybrid ATP-paid waste export matches CPU'


def test_leak_plan_numpy_torch():
    w=make_world(9,1);c=w.cells[0];c.membrane[3]*=.01;c.membrane_oxidation[3]=1.5
    n=a2.leak_plan_numpy(c.pools,c.membrane,c.membrane_oxidation,c.radius,.1,c.genome_mass())
    t=a2.leak_plan_torch(*[torch.as_tensor(x,dtype=torch.float64) if isinstance(x,(np.ndarray,float,int)) else x for x in (c.pools,c.membrane,c.membrane_oxidation,c.radius,.1,c.genome_mass())])
    for k in ('fraction','gap_strength','base_amounts','extra_amounts','atp_loss'):close(n[k],t[k].detach().cpu().numpy(),atol=2e-14,rtol=2e-14,label='leak '+k)
    assert n['segment']==t['segment']
    return 'damage-aware leakage plan lockstep'


def test_leak_original_lockstep_and_rng():
    cpu=make_world(10,1);c=cpu.cells[0];c.membrane[0]=0;c.membrane_oxidation[0]=2.0;c.pools[a2.s5.POOL_FUEL]+=.2;c.pools[a2.s5.POOL_ALT]+=.1
    gpu=a2.s66.Formal066World.from_state(copy.deepcopy(cpu.state_dict()))
    cpu.cells[0].leak(cpu,.1)
    a2.TorchKernelBackendA2({'device':'cpu'}).leak_inplace(gpu,gpu.cells[0],.1)
    assert_worlds(cpu,gpu,atol=4e-13)
    return 'material emissions and RNG stream match inherited leakage'


def test_radius_numpy_torch_and_original():
    w=make_world(11,1);c=w.cells[0]
    n=a2.radius_relax_numpy(c.pools,c.membrane,c.radius,.1,c.genome_mass())
    t=a2.radius_relax_torch(torch.tensor(c.pools,dtype=torch.float64),torch.tensor(c.membrane,dtype=torch.float64),torch.tensor(c.radius,dtype=torch.float64),.1,torch.tensor(c.genome_mass(),dtype=torch.float64)).item()
    clone=copy.deepcopy(c);clone.update_radius(.1)
    close(n,t,atol=1e-14,rtol=1e-14,label='radius torch');close(n,clone.radius,atol=1e-14,rtol=1e-14,label='radius CPU')
    return 'polymer-aware osmotic radius exact'


def test_motion_numpy_torch_and_original():
    w=make_world(12,1);c=w.cells[0];c.surface_flux=np.asarray([.01,-.02]);c.vel=np.asarray([.003,.001]);brown=np.asarray([.0002,-.0001])
    n=a2.motion_numpy(c.pos,c.vel,c.surface_flux,c.radius,brown,.1)
    t=a2.motion_torch(torch.tensor(c.pos),torch.tensor(c.vel),torch.tensor(c.surface_flux),torch.tensor(c.radius),torch.tensor(brown),.1)
    for k in n:close(n[k],t[k].numpy(),atol=2e-14,rtol=2e-14,label='motion '+k)
    class FixedRNG:
        def normal(self,*args):return brown.copy()
    clone=copy.deepcopy(c);clone.update_motion(FixedRNG(),.1)
    for k,got in [('pos',clone.pos),('vel',clone.vel),('surface_flux',clone.surface_flux)]:close(n[k],got,atol=2e-14,rtol=2e-14,label='motion cpu '+k)
    return 'surface-flux/Brownian motion lockstep'


def test_hybrid_one_step_lockstep():
    cpu=make_world(13,2);gpu=a2.s66.Formal066World.from_state(copy.deepcopy(cpu.state_dict()));h=a2.Hybrid066WorldA2(gpu,{'device':'cpu','precision':'float64'})
    cpu.step(.1);h.step(.1);assert_worlds(cpu,h.world)
    return 'one full 0.6.6 event step lockstep'


def test_hybrid_ten_step_lockstep():
    cpu=make_world(14,2);gpu=a2.s66.Formal066World.from_state(copy.deepcopy(cpu.state_dict()));h=a2.Hybrid066WorldA2(gpu,{'device':'cpu','precision':'float64'})
    for _ in range(10):cpu.step(.1);h.step(.1)
    assert_worlds(cpu,h.world,atol=5e-11)
    return 'ten-step event and RNG lockstep'


def test_hybrid_reversal_lockstep():
    cpu=make_world(15,2,environment=a2.s66.ENV_PERIODIC);gpu=a2.s66.Formal066World.from_state(copy.deepcopy(cpu.state_dict()));h=a2.Hybrid066WorldA2(gpu,{'device':'cpu','precision':'float64'})
    for _ in range(20):cpu.step(.1);h.step(.1)
    assert_worlds(cpu,h.world,atol=8e-11)
    return 'periodic/reversal hooks remain lockstep'


def test_hybrid_clone_deterministic():
    h=a2.Hybrid066WorldA2.new(seed=16,initial_cells=2,gpu_config={'device':'cpu'})
    for _ in range(4):h.step(.1)
    c=h.clone()
    for _ in range(8):h.step(.1);c.step(.1)
    assert_worlds(h.world,c.world)
    return 'A2 hybrid clone deterministic'


def test_hybrid_save_restore_deterministic():
    h=a2.Hybrid066WorldA2.new(seed=17,initial_cells=2,gpu_config={'device':'cpu'})
    for _ in range(3):h.step(.1)
    with tempfile.TemporaryDirectory() as td:
        p=os.path.join(td,'a2.pkl');h.save(p);r=a2.Hybrid066WorldA2.load(p)
        for _ in range(7):h.step(.1);r.step(.1)
        assert_worlds(h.world,r.world)
    return 'A2 save/restore deterministic'


def test_new_cells_receive_hooks():
    w=make_world(18,1);h=a2.Hybrid066WorldA2(w,{'device':'cpu'})
    parent=w.cells[0];parent.division_progress=1.0
    if len(parent.genomes)<2:parent.genomes.append(parent.genomes[0].copy())
    daughters=parent.split(w)
    assert daughters and len(daughters)==2
    w.cells=daughters;a2._attach_backend_to_world_a2(w,h.backend)
    assert all(hasattr(c,'_soma068a2_original_surface_exchange') for c in daughters)
    return 'post-division daughters receive A2 hooks before next step'


def test_material_ledger_remains_small():
    h=a2.Hybrid066WorldA2.new(seed=19,initial_cells=2,gpu_config={'device':'cpu'})
    for _ in range(50):h.step(.1)
    assert abs(h.world.matter_ledger_residual())<3e-5
    return 'hybrid material ledger remains within detailed-world tolerance'


def test_summary_disclaims_full_gpu():
    h=a2.Hybrid066WorldA2.new(seed=20,initial_cells=1,gpu_config={'device':'cpu'})
    s=h.summary();assert s['gpu_full_world_step'] is False;assert s['gpu_port_status']['genome_replication']=='cpu-authoritative'
    return 'summary fails closed on unported life mechanisms'


def test_backend_stats_count_all_a2_kernels():
    h=a2.Hybrid066WorldA2.new(seed=21,initial_cells=1,gpu_config={'device':'cpu'});h.step(.1);s=h.backend.stats()
    for k in ('surface_calls','waste_export_calls','leak_calls','radius_calls','motion_calls','spatial_queries'):assert s[k]>=1
    return 'A2 kernel calls explicitly counted'


def test_no_external_fitness_or_reward():
    text=open(a2.__file__,encoding='utf-8').read().lower()
    assert 'fitness_value' not in text and 'correct_direction' not in text
    return 'migration introduces no external fitness/reward channel'


def test_cuda_request_fails_closed_if_unavailable():
    if torch.cuda.is_available():return 'CUDA available; fail-closed branch not applicable'
    try:a2.TorchKernelBackendA2({'device':'cuda'})
    except RuntimeError:return 'CUDA request rejected by CPU-only build'
    raise AssertionError('CUDA silently accepted')


def test_fp32_discrepancy_is_measured():
    w=make_world(22,1);c,f,idx,args=_surface_args(w)
    ref=a2.surface_exchange_numpy(*args,transport=True,candidate_indices=idx)
    targs=[]
    for i,x in enumerate(args[:-1]):targs.append(torch.as_tensor(x,dtype=torch.int64 if i==1 else torch.float32))
    out=a2.surface_exchange_torch(*targs,args[-1],transport=True,candidate_indices=idx)
    diff=float(np.max(np.abs(ref['pools']-out['pools'].numpy())))
    assert np.isfinite(diff) and diff>=0
    return 'fp32 surface discrepancy measured ({:.3g})'.format(diff)


def test_parent_a1_environment_still_partial():
    assert a1.PORT_STATUS['surface_exchange']=='planned'
    assert a1.PORT_STATUS['division']=='cpu-authoritative'
    return 'A1 remains frozen and separate from A2'


def test_tensor_schema_still_lossless():
    w=make_world(23,2);batch=a1.FullFidelity066Adapter(a1.GPU068Config(device='cpu')).pack(w);r=a1.FullFidelity066Adapter(a1.GPU068Config(device='cpu')).restore_cpu_worlds(batch)[0]
    assert a1.semantic_state_hash(w.state_dict())==a1.semantic_state_hash(r.state_dict())
    return 'A1 lossless tensor mirror preserved'


def test_a2_port_status_complete_and_explicit():
    for k in (a2.KERNEL_SURFACE_EXCHANGE,a2.KERNEL_WASTE_EXPORT,a2.KERNEL_LEAK_PLAN,a2.KERNEL_RADIUS,a2.KERNEL_MOTION,a2.KERNEL_SPATIAL_INDEX):assert 'integrated' in a2.PORT_STATUS[k]
    for k in ('genome_replication','translation','division','death_corpse_edna_hgt','neural_causal_system'):assert a2.PORT_STATUS[k]=='cpu-authoritative'
    return 'A2 coverage table is explicit and fail-closed'


def test_all_arrays_finite_after_cycle():
    h=a2.Hybrid066WorldA2.new(seed=24,initial_cells=3,gpu_config={'device':'cpu'})
    for _ in range(20):h.step(.1)
    for value in state_arrays(h.world).values():assert np.all(np.isfinite(np.asarray(value)))
    return 'detailed hybrid state finite after full kernel cycle'

TESTS=[v for k,v in sorted(globals().items()) if k.startswith('test_') and callable(v)]

def main():
    start=time.time();rows=[]
    for fn in TESTS:
        try:detail=fn();rows.append((fn.__name__,'PASS',detail));print('PASS',fn.__name__,detail)
        except Exception as exc:rows.append((fn.__name__,'FAIL',repr(exc)));print('FAIL',fn.__name__,repr(exc))
    passed=sum(r[1]=='PASS' for r in rows)
    with open(RESULT_CSV,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['test','status','detail']);w.writerows(rows)
    with open(RESULT_TXT,'w',encoding='utf-8') as f:
        f.write('SOMA-CELL 0.6.8-GPU A2 VALIDATION RESULTS\n');f.write('='*52+'\n');f.write('{} / {} PASS\n'.format(passed,len(rows)));f.write('elapsed {:.3f}s\n\n'.format(time.time()-start))
        for r in rows:f.write('{}: {} - {}\n'.format(r[1],r[0],r[2]))
    if passed!=len(rows):raise SystemExit(1)
    print('{} / {} PASS'.format(passed,len(rows)))

if __name__=='__main__':main()
