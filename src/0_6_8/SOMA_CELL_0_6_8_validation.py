# coding: utf-8
"""Validation for SOMA-CELL 0.6.8-GPU migration foundation."""
from __future__ import division

import csv
import json
import math
import os
import tempfile
import time

import numpy as np
import torch

import SOMA_CELL_0_6_8_gpu as gpu

RESULT_TXT = 'SOMA_CELL_0_6_8_GPU_VALIDATION_RESULTS.txt'
RESULT_CSV = 'soma_cell_0_6_8_gpu_validation.csv'


def assert_close(a, b, atol=1e-11, rtol=1e-11, label='value'):
    aa = np.asarray(a); bb = np.asarray(b)
    if not np.allclose(aa, bb, atol=atol, rtol=rtol):
        diff = float(np.max(np.abs(aa - bb)))
        raise AssertionError('{} mismatch max_abs={}'.format(label, diff))


def make_world(seed=101, cells=3):
    cfg = gpu.s66.shared_ecology_config(
        environment=gpu.s66.ENV_STABLE,
        mutation=True,
        hgt=True,
        eco66_washout=False,
        eco66_chemostat=False,
    )
    return gpu.s66.Formal066World(seed=seed, initial_cells=cells, config=cfg)


def test_build_and_schema():
    assert gpu.BUILD == 'SOMA-CELL 0.6.8-GPU'
    assert gpu.SCHEMA_VERSION == '0.6.8-GPU-A1.0'
    return 'build/schema frozen'


def test_environment_report_is_honest():
    r = gpu.environment_report()
    assert r['torch_installed'] is True
    assert gpu.PORT_STATUS['division'] == 'cpu-authoritative'
    assert gpu.PORT_STATUS[gpu.KERNEL_METABOLISM_CORE].startswith('validated-standalone')
    return 'report exposes partial migration and CPU-authoritative subsystems'


def test_counter_uniform_matches_numpy():
    counters_np = np.arange(4096, dtype=np.uint64).reshape(64, 64)
    a = gpu.counter_uniform_numpy(1234, counters_np)
    counters_t = torch.arange(4096, dtype=torch.int64).reshape(64, 64)
    b = gpu.counter_uniform_torch(1234, counters_t, dtype=torch.float64).numpy()
    assert_close(a, b, atol=0.0, rtol=0.0, label='counter uniform')
    return 'device-neutral xorshift counter stream exact'


def test_counter_normal_is_deterministic():
    c = torch.arange(2048, dtype=torch.int64)
    a = gpu.counter_normal_torch(88, c, dtype=torch.float64)
    b = gpu.counter_normal_torch(88, c, dtype=torch.float64)
    assert torch.equal(a, b)
    assert torch.isfinite(a).all()
    return 'counter normal repeatable and finite'


def test_circular_smooth_lockstep():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(3, 5, gpu.MEMBRANE_SEGMENTS))
    a = gpu.circular_smooth_numpy(x, 0.24)
    b = gpu.circular_smooth_torch(torch.tensor(x, dtype=torch.float64), 0.24).numpy()
    assert_close(a, b, atol=1e-14, rtol=1e-14, label='smooth')
    return 'circular smoothing lockstep'


def synthetic_state(seed=5, B=3, P=91, C=4):
    rng = np.random.default_rng(seed)
    pos = rng.random((B, P, 2)); kind = rng.integers(0, 4, (B, P), dtype=np.int64)
    amount = rng.uniform(0.001, 0.04, (B, P)); pmask = rng.random((B, P)) > 0.12
    centres = np.broadcast_to(gpu.s4.SensorimotorParticleField.DEFAULT_CENTRES, (B, 4, 2)).copy()
    cpos = rng.random((B, C, 2)); radius = rng.uniform(0.025, 0.08, (B, C)); cmask = rng.random((B, C)) > 0.10
    noise = rng.normal(size=(B, P, 2))
    return pos, kind, amount, pmask, centres, cpos, radius, cmask, noise


def test_diffusion_lockstep_fp64():
    pos, kind, amount, pmask, centres, cpos, radius, cmask, noise = synthetic_state()
    a = gpu.diffuse_particles_numpy(pos, kind, pmask, centres, 0.1, noise)
    b = gpu.diffuse_particles_torch(
        torch.tensor(pos, dtype=torch.float64), torch.tensor(kind), torch.tensor(pmask),
        torch.tensor(centres, dtype=torch.float64), 0.1, torch.tensor(noise, dtype=torch.float64),
    ).numpy()
    assert_close(a, b, atol=2e-15, rtol=2e-15, label='diffusion')
    return 'particle diffusion/drift lockstep fp64'


def test_diffusion_mask_and_torus():
    pos, kind, amount, pmask, centres, cpos, radius, cmask, noise = synthetic_state(seed=8)
    out = gpu.diffuse_particles_torch(
        torch.tensor(pos, dtype=torch.float64), torch.tensor(kind), torch.tensor(pmask),
        torch.tensor(centres, dtype=torch.float64), 0.7, torch.tensor(noise * 20.0, dtype=torch.float64),
    ).numpy()
    assert np.all((out >= 0.0) & (out < 1.0))
    assert_close(out[~pmask], pos[~pmask], atol=0.0, rtol=0.0, label='masked particles')
    return 'torus bounds and inactive-particle immobility'


def test_ligand_profile_lockstep_fp64():
    pos, kind, amount, pmask, centres, cpos, radius, cmask, noise = synthetic_state(seed=9, B=2, P=77, C=5)
    a = gpu._ligand_profiles_numpy(pos, kind, amount, pmask, cpos, radius, cmask)
    b = gpu.ligand_profiles_torch(
        torch.tensor(pos, dtype=torch.float64), torch.tensor(kind), torch.tensor(amount, dtype=torch.float64),
        torch.tensor(pmask), torch.tensor(cpos, dtype=torch.float64), torch.tensor(radius, dtype=torch.float64), torch.tensor(cmask),
    ).numpy()
    assert_close(a, b, atol=2e-13, rtol=2e-13, label='ligand profiles')
    return 'membrane-local particle profiles lockstep fp64'


def test_ligand_profiles_respect_masks():
    pos, kind, amount, pmask, centres, cpos, radius, cmask, noise = synthetic_state(seed=10, B=1, P=30, C=3)
    cmask[:] = False
    b = gpu.ligand_profiles_torch(
        torch.tensor(pos, dtype=torch.float64), torch.tensor(kind), torch.tensor(amount, dtype=torch.float64),
        torch.tensor(pmask), torch.tensor(cpos, dtype=torch.float64), torch.tensor(radius, dtype=torch.float64), torch.tensor(cmask),
    )
    assert float(b.abs().max()) == 0.0
    return 'masked cells produce zero receptor occupancy'


def test_adapter_pack_shapes_and_finiteness():
    worlds = [make_world(101), make_world(202)]
    adapter = gpu.FullFidelity066Adapter(gpu.GPU068Config(device='cpu', precision='float64'))
    batch = adapter.pack(worlds)
    assert batch.batch_size == 2
    assert batch.particle_pos.shape[-1] == 2
    assert batch.membrane.shape[-1] == gpu.MEMBRANE_SEGMENTS
    assert batch.transporters.shape[-1] == gpu.CHANNEL_COUNT
    assert batch.finite()
    return 'lossless padded tensor schema packs two detailed worlds'


def test_adapter_preserves_full_opaque_state_hash():
    world = make_world(303)
    adapter = gpu.FullFidelity066Adapter(gpu.GPU068Config(device='cpu', precision='float64'))
    batch = adapter.pack(world)
    assert batch.source_hashes[0] == gpu.canonical_pickle_hash(world.state_dict())
    restored = adapter.restore_cpu_worlds(batch)[0]
    assert gpu.semantic_state_hash(restored.state_dict()) == gpu.semantic_state_hash(batch.opaque_states[0])
    return 'full CPU state round-trips through opaque canonical state'


def test_adapter_genome_bytes_are_exact():
    world = make_world(404)
    adapter = gpu.FullFidelity066Adapter(gpu.GPU068Config(device='cpu', precision='float64'))
    batch = adapter.pack(world)
    for ci, cell in enumerate(world.cells):
        for gi, genome in enumerate(cell.genomes[:2]):
            n = int(batch.genome_length[0, ci, gi].item())
            got = batch.genome_data[0, ci, gi, :n].cpu().numpy()
            assert np.array_equal(got, np.asarray(genome, dtype=np.uint8))
    return 'variable-length material genomes mirrored byte-for-byte'


def test_adapter_capacity_fails_closed():
    world = make_world(505)
    try:
        gpu.FullFidelity066Adapter(gpu.GPU068Config(device='cpu', max_particles=8)).pack(world)
    except ValueError:
        return 'particle capacity overflow rejected'
    raise AssertionError('capacity overflow was silently truncated')


def test_tensor_batch_clone_independence():
    batch = gpu.FullFidelity066Adapter(gpu.GPU068Config(device='cpu')).pack(make_world(606))
    clone = batch.clone()
    clone.particle_amount += 1.0
    assert not torch.equal(batch.particle_amount, clone.particle_amount)
    return 'tensor clone owns independent storage'


def test_metabolism_core_numpy_torch_lockstep():
    world = make_world(707)
    adapter = gpu.FullFidelity066Adapter(gpu.GPU068Config(device='cpu', precision='float64'))
    batch = adapter.pack(world)
    args_np = [
        batch.pools.cpu().numpy(), batch.membrane.cpu().numpy(), batch.transporters.cpu().numpy(),
        batch.contact_trace.cpu().numpy(), batch.damage_trace.cpu().numpy(), batch.radius.cpu().numpy(),
        batch.cell_alive.cpu().numpy(), 0.05,
    ]
    n = gpu.metabolism_core_numpy(*args_np)
    t = gpu.metabolism_core_torch(
        batch.pools, batch.membrane, batch.transporters, batch.contact_trace,
        batch.damage_trace, batch.radius, batch.cell_alive, 0.05,
    )
    for key in ('pools', 'membrane', 'transporters', 'contact_trace', 'damage_trace', 'dissipated_delta', 'catalysis'):
        assert_close(n[key], t[key].cpu().numpy(), atol=3e-13, rtol=3e-13, label='metabolism '+key)
    return 'standalone core metabolism lockstep against independent NumPy transcription'


def test_metabolism_core_finite_and_nonnegative():
    world = make_world(808)
    batch = gpu.FullFidelity066Adapter(gpu.GPU068Config(device='cpu')).pack(world)
    out = gpu.metabolism_core_torch(batch.pools, batch.membrane, batch.transporters, batch.contact_trace, batch.damage_trace, batch.radius, batch.cell_alive, 0.1)
    assert torch.isfinite(out['pools']).all()
    assert float(out['pools'].min()) >= -1e-14
    assert float(out['membrane'].min()) >= -1e-14
    assert float(out['transporters'].min()) >= -1e-14
    return 'standalone metabolism remains finite and nonnegative'


def _world_arrays(world):
    return {
        'field_pos': np.asarray(world.field.pos), 'field_amount': np.asarray(world.field.amount),
        'field_kind': np.asarray(world.field.kind), 'age': float(world.age),
        'divisions': int(world.divisions), 'deaths': int(world.deaths),
        'cell_pos': np.asarray([c.pos for c in world.cells]),
        'cell_pools': np.asarray([c.pools for c in world.cells]),
        'cell_membrane': np.asarray([c.membrane for c in world.cells]),
    }


def test_hybrid_one_step_calls_offloaded_kernels():
    hybrid = gpu.Hybrid066World.new(
        seed=909, initial_cells=3,
        world_config=gpu.s66.shared_ecology_config(eco66_chemostat=False, eco66_washout=False),
        gpu_config=gpu.GPU068Config(device='cpu', precision='float64'),
    )
    hybrid.step(0.1)
    stats = hybrid.backend.stats()
    assert stats['diffusion_calls'] == 1
    assert stats['profile_calls'] >= 1
    assert hybrid.world.finite()
    return 'hybrid world offloads diffusion/profile while full CPU world stays finite'


def test_hybrid_one_step_cpu_lockstep():
    base = make_world(1001)
    state = base.state_dict()
    cpu = gpu.s66.Formal066World.from_state(copy_state(state))
    hybrid = gpu.Hybrid066World(gpu.s66.Formal066World.from_state(copy_state(state)), gpu.GPU068Config(device='cpu', precision='float64'))
    cpu.step(0.1); hybrid.step(0.1)
    a = _world_arrays(cpu); b = _world_arrays(hybrid.world)
    assert a['divisions'] == b['divisions'] and a['deaths'] == b['deaths']
    for key in ('field_pos','field_amount','cell_pos','cell_pools','cell_membrane'):
        assert_close(a[key], b[key], atol=2e-11, rtol=2e-11, label='hybrid '+key)
    return 'one full 0.6.6 step lockstep with Torch CPU offload'


def copy_state(state):
    return pickle_roundtrip(state)


def pickle_roundtrip(value):
    import pickle
    return pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def test_hybrid_short_event_lockstep():
    base = make_world(1002)
    state = base.state_dict()
    cpu = gpu.s66.Formal066World.from_state(copy_state(state))
    hybrid = gpu.Hybrid066World(gpu.s66.Formal066World.from_state(copy_state(state)), gpu.GPU068Config(device='cpu', precision='float64'))
    for _ in range(8):
        cpu.step(0.1); hybrid.step(0.1)
    assert cpu.divisions == hybrid.world.divisions
    assert cpu.deaths == hybrid.world.deaths
    assert len(cpu.cells) == len(hybrid.world.cells)
    assert_close(cpu.field.pos, hybrid.world.field.pos, atol=3e-9, rtol=3e-9, label='short field position')
    assert_close([c.pools for c in cpu.cells], [c.pools for c in hybrid.world.cells], atol=3e-9, rtol=3e-9, label='short pools')
    return 'eight-step event-level lockstep within fp64 tolerance'


def test_hybrid_clone_is_deterministic():
    hybrid = gpu.Hybrid066World.new(
        seed=1101, initial_cells=3,
        world_config=gpu.s66.shared_ecology_config(eco66_chemostat=False, eco66_washout=False),
        gpu_config=gpu.GPU068Config(device='cpu', precision='float64'),
    )
    for _ in range(3): hybrid.step(0.1)
    twin = hybrid.clone()
    for _ in range(5): hybrid.step(0.1); twin.step(0.1)
    assert_close(hybrid.world.field.pos, twin.world.field.pos, atol=0.0, rtol=0.0, label='hybrid clone field')
    assert gpu.semantic_state_hash(hybrid.world.state_dict()) == gpu.semantic_state_hash(twin.world.state_dict())
    return 'hybrid clone preserves CPU RNG and tensor backend configuration exactly'


def test_hybrid_save_restore_is_deterministic():
    hybrid = gpu.Hybrid066World.new(
        seed=1201, initial_cells=3,
        world_config=gpu.s66.shared_ecology_config(eco66_chemostat=False, eco66_washout=False),
        gpu_config=gpu.GPU068Config(device='cpu', precision='float64'),
    )
    for _ in range(3): hybrid.step(0.1)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'hybrid.pkl')
        hybrid.save(path); restored = gpu.Hybrid066World.load(path)
        for _ in range(4): hybrid.step(0.1); restored.step(0.1)
    assert gpu.semantic_state_hash(hybrid.world.state_dict()) == gpu.semantic_state_hash(restored.world.state_dict())
    return 'hybrid save/restore deterministic'


def test_hybrid_summary_disclaims_full_port():
    h = gpu.Hybrid066World.new(seed=1301, gpu_config=gpu.GPU068Config(device='cpu'))
    s = h.summary()
    assert s['gpu_full_world_step'] is False
    assert s['gpu_port_status']['division'] == 'cpu-authoritative'
    return 'summary cannot mislabel hybrid as complete GPU world'


def test_batch_worlds_are_independent():
    d = gpu.synthetic_batch(worlds=4, particles=80, cells=3, seed=99, device='cpu', precision='float64')
    noise = gpu.counter_normal_torch(20, torch.arange(4*80*2,dtype=torch.int64).reshape(4,80,2), dtype=torch.float64)
    out = gpu.diffuse_particles_torch(d['pos'], d['kind'], d['pmask'], d['centres'], 0.1, noise)
    assert not torch.equal(out[0], out[1])
    original = out[1].clone(); out[0] += 0.1
    assert torch.equal(out[1], original)
    return 'batched worlds have independent tensor slices'


def test_benchmark_runs_and_reports_capacity():
    r = gpu.benchmark_kernels(worlds=3, particles=96, cells=4, steps=3, device='cpu', precision='float64')
    assert r['world_steps_per_second'] > 0.0
    assert math.isfinite(r['kernel_checksum'])
    assert r['working_tensor_bytes'] > 0
    assert r['full_world_step_ported'] is False
    return 'batched benchmark finite with explicit partial-port flag'


def test_cuda_request_fails_closed_when_unavailable():
    if torch.cuda.is_available():
        return 'CUDA available; fail-closed branch not applicable'
    try:
        gpu.resolve_device('cuda')
    except RuntimeError:
        return 'CUDA request rejected by CPU-only build'
    raise AssertionError('CUDA silently fell back to CPU')


def test_fp32_is_explicitly_less_strict_than_fp64():
    pos, kind, amount, pmask, centres, cpos, radius, cmask, noise = synthetic_state(seed=13, B=2, P=100, C=3)
    ref = gpu.diffuse_particles_numpy(pos, kind, pmask, centres, 0.1, noise)
    out = gpu.diffuse_particles_torch(torch.tensor(pos,dtype=torch.float32),torch.tensor(kind),torch.tensor(pmask),torch.tensor(centres,dtype=torch.float32),0.1,torch.tensor(noise,dtype=torch.float32)).numpy()
    error = float(np.max(np.abs(ref-out)))
    assert error < 2e-6 and error > 0.0
    return 'fp32 tolerance measured rather than claimed bit-exact'


def test_adapter_memory_estimator_is_positive():
    batch = gpu.FullFidelity066Adapter(gpu.GPU068Config(device='cpu')).pack(make_world(1401))
    assert batch.estimated_bytes() > 1000
    assert batch.summary()['active_particles'] == len(batch.opaque_states[0]['field']['amount'])
    return 'tensor memory/capacity accounting available'


def test_kernel_coverage_table_is_complete_enough_to_fail_closed():
    for required in ('surface_exchange','genome_replication','translation','division','death_corpse_edna_hgt','neural_causal_system'):
        assert required in gpu.PORT_STATUS
        assert gpu.PORT_STATUS[required] != 'hybrid-integrated'
    return 'unported life mechanisms remain explicitly CPU-authoritative'


def test_no_external_fitness_added():
    source = open(gpu.__file__, 'r', encoding='utf-8').read()
    assert 'external_fitness_selection=True' not in source
    h = gpu.Hybrid066World.new(seed=1501, gpu_config=gpu.GPU068Config(device='cpu'))
    assert h.world.summary().get('eco66_external_fitness_events', 0) == 0
    return 'GPU migration adds no fitness/reward channel'


def test_material_arrays_are_not_silently_truncated():
    world = make_world(1601)
    adapter = gpu.FullFidelity066Adapter(gpu.GPU068Config(device='cpu', max_genome_symbols=10))
    try:
        adapter.pack(world)
    except ValueError:
        return 'variable genome overflow rejected'
    raise AssertionError('genome was silently truncated')


def test_state_schema_includes_dead_and_living_cell_slots():
    world = make_world(1701)
    world.cells[0].alive = False
    batch = gpu.FullFidelity066Adapter(gpu.GPU068Config(device='cpu')).pack(world)
    assert bool(batch.cell_mask[0,0]) is True
    assert bool(batch.cell_alive[0,0]) is False
    return 'dead cell slots are represented rather than discarded'


def test_all_tensor_fields_are_finite_after_kernel_cycle():
    worlds=[make_world(1801),make_world(1802)]
    batch=gpu.FullFidelity066Adapter(gpu.GPU068Config(device='cpu',precision='float64')).pack(worlds)
    counters=torch.arange(batch.batch_size*batch.particle_capacity*2,dtype=torch.int64).reshape(batch.batch_size,batch.particle_capacity,2)
    noise=gpu.counter_normal_torch(44,counters,dtype=torch.float64)
    batch.particle_pos=gpu.diffuse_particles_torch(batch.particle_pos,batch.particle_kind,batch.particle_mask,batch.patch_centres,0.1,noise)
    profiles=gpu.ligand_profiles_torch(batch.particle_pos,batch.particle_kind,batch.particle_amount,batch.particle_mask,batch.cell_pos,batch.radius,batch.cell_mask)
    assert batch.finite() and torch.isfinite(profiles).all()
    return 'packed detailed worlds remain finite through tensor kernel cycle'


TESTS = [
    test_build_and_schema,
    test_environment_report_is_honest,
    test_counter_uniform_matches_numpy,
    test_counter_normal_is_deterministic,
    test_circular_smooth_lockstep,
    test_diffusion_lockstep_fp64,
    test_diffusion_mask_and_torus,
    test_ligand_profile_lockstep_fp64,
    test_ligand_profiles_respect_masks,
    test_adapter_pack_shapes_and_finiteness,
    test_adapter_preserves_full_opaque_state_hash,
    test_adapter_genome_bytes_are_exact,
    test_adapter_capacity_fails_closed,
    test_tensor_batch_clone_independence,
    test_metabolism_core_numpy_torch_lockstep,
    test_metabolism_core_finite_and_nonnegative,
    test_hybrid_one_step_calls_offloaded_kernels,
    test_hybrid_one_step_cpu_lockstep,
    test_hybrid_short_event_lockstep,
    test_hybrid_clone_is_deterministic,
    test_hybrid_save_restore_is_deterministic,
    test_hybrid_summary_disclaims_full_port,
    test_batch_worlds_are_independent,
    test_benchmark_runs_and_reports_capacity,
    test_cuda_request_fails_closed_when_unavailable,
    test_fp32_is_explicitly_less_strict_than_fp64,
    test_adapter_memory_estimator_is_positive,
    test_kernel_coverage_table_is_complete_enough_to_fail_closed,
    test_no_external_fitness_added,
    test_material_arrays_are_not_silently_truncated,
    test_state_schema_includes_dead_and_living_cell_slots,
    test_all_tensor_fields_are_finite_after_kernel_cycle,
]


def run_all(write=True):
    rows=[]; start=time.time()
    for fn in TESTS:
        t0=time.time()
        try:
            detail=fn(); status='PASS'; error=''
        except Exception as exc:
            detail=''; status='FAIL'; error='{}: {}'.format(type(exc).__name__, exc)
        rows.append({'test':fn.__name__,'status':status,'detail':detail,'error':error,'seconds':time.time()-t0})
        print('{}: {}{}'.format(status,fn.__name__,(' - '+detail) if detail else (' - '+error if error else '')))
    passed=sum(r['status']=='PASS' for r in rows)
    total=len(rows)
    if write:
        here=os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here,RESULT_CSV),'w',newline='',encoding='utf-8') as h:
            w=csv.DictWriter(h,fieldnames=('test','status','detail','error','seconds'));w.writeheader();w.writerows(rows)
        lines=[gpu.BUILD+' VALIDATION RESULTS','='*48,'{} / {} PASS'.format(passed,total),'elapsed {:.3f}s'.format(time.time()-start),'']
        lines += ['{}: {}{}'.format(r['status'],r['test'],(' - '+r['detail']) if r['detail'] else (' - '+r['error'] if r['error'] else '')) for r in rows]
        with open(os.path.join(here,RESULT_TXT),'w',encoding='utf-8') as h:h.write('\n'.join(lines)+'\n')
    if passed != total:
        raise AssertionError('{} / {} tests passed'.format(passed,total))
    return rows


if __name__=='__main__':
    run_all(write=True)
