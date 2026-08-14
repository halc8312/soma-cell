#!/usr/bin/env python3
# coding: utf-8
"""Correctness-gated benchmark for SOMA-CELL 0.6.8-GPU A3.

This runner does not call a timing result a speedup unless the corresponding
device has first passed full-state fp64 lockstep.  fp32 is measured only after
that fp64 gate and always carries its discrepancy report.
"""
from __future__ import division

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import SOMA_CELL_0_6_8_A3_validation as validation

FP32_CANDIDATE_LEDGER_ATOL = float(64 * np.finfo(np.float32).eps)
FP32_CANDIDATE_LEDGER_REASON = (
    '64*float32 epsilon bounds ordered-reduction versus scalar-rounding drift; '
    'strict_material_ledger remains enabled and this is not an equivalence tolerance.'
)
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
SOURCE_PATHS = (
    'src/0_6_8/SOMA_CELL_0_6_8_gpu_a3.py',
    'src/0_6_8/SOMA_CELL_0_6_8_gpu_a3_scheduler.py',
    'src/0_6_8/SOMA_CELL_0_6_8_A3_validation.py',
    'src/0_6_8/SOMA_CELL_0_6_8_A3_benchmark.py',
    'src/0_6_8/SOMA_CELL_0_6_8_gpu_a2.py',
    'src/0_6_6/SOMA_CELL_0_6_6_pythonista.py',
)
FORMAL_WORLD_COUNTS = (1, 8, 32, 128)
FORMAL_CELL_SWEEP = (1, 3, 8)
FORMAL_PARTICLE_SWEEP = (216, 288, 360)
FORMAL_CAPACITY_SWEEP = (64, 128, 256)
FORMAL_TIMING_STEPS = 1


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def source_sha256_map():
    result = {}
    for relative in SOURCE_PATHS:
        path = os.path.join(PROJECT_ROOT, *relative.split('/'))
        if not os.path.isfile(path):
            raise FileNotFoundError('benchmark source hash input missing: {}'.format(path))
        result[relative] = sha256(path)
    return result


def git_value(*args):
    try:
        return subprocess.check_output(['git'] + list(args), cwd=os.path.dirname(os.path.dirname(HERE)), text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return 'unavailable'


def synchronise(torch, device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(torch.device(device))


def environment_report(torch):
    report = {
        'captured_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version,
        'platform': platform.platform(),
        'numpy': np.__version__,
        'torch': torch.__version__,
        'torch_cuda_version': torch.version.cuda,
        'cuda_available': bool(torch.cuda.is_available()),
        'cuda_device_count': int(torch.cuda.device_count()),
        'devices': [],
    }
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        report['devices'].append({
            'index': index,
            'name': props.name,
            'total_memory_bytes': int(props.total_memory),
            'compute_capability': [int(props.major), int(props.minor)],
            'multi_processor_count': int(props.multi_processor_count),
        })
    try:
        report['nvidia_smi'] = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
            text=True, timeout=20, stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        report['nvidia_smi'] = 'unavailable: {!r}'.format(exc)
    return report


def summary_samples(samples, work_items):
    if not samples:
        return {'samples_seconds': [], 'error': 'no samples'}
    median = statistics.median(samples)
    return {
        'samples_seconds': [float(value) for value in samples],
        'minimum_seconds': float(min(samples)),
        'median_seconds': float(median),
        'maximum_seconds': float(max(samples)),
        'mean_seconds': float(statistics.mean(samples)),
        'pstdev_seconds': float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0,
        'work_items': int(work_items),
        'work_items_per_second_at_median': float(work_items / max(median, 1e-15)),
    }


def _set_particle_target(world, target):
    """Reach an explicit benchmark target without deleting matter or clipping."""
    if target is None:
        return len(world.field.amount)
    target = int(target)
    current = len(world.field.amount)
    if target < current:
        raise ValueError(
            'particle target {} is below canonical initial {}; shrinking is forbidden'.format(
                target, current,
            )
        )
    if target == current:
        return current
    field_module = sys.modules.get(type(world.field).__module__)
    limit = int(getattr(field_module, 'PARTICLE_LIMIT', 360))
    if target > limit:
        raise ValueError('particle target {} exceeds frozen limit {}'.format(target, limit))
    count = target - current
    positions = world.rng.random((count, 2))
    amounts = np.full((count,), 0.002, dtype=np.float64)
    kind = int(world.field.kind[0]) if current else 0
    world.field.add_many(kind, positions, amounts, count_as_injection=True)
    actual = len(world.field.amount)
    if actual != target:
        raise AssertionError('particle target {} produced {}'.format(target, actual))
    return actual


def _state_cache_key(worlds, cells, seed, particles):
    return (int(worlds), int(cells), int(seed), None if particles is None else int(particles))


def _base_states(worlds, cells, seed, particles=None, cache=None):
    key = _state_cache_key(worlds, cells, seed, particles)
    if cache is not None and key in cache:
        cached_states, cached_particles = cache[key]
        return validation.pickle_clone(cached_states), list(cached_particles)
    states = []
    actual_particles = []
    for index in range(worlds):
        world = validation.make_world(seed=seed + index, cells=cells)
        actual_particles.append(_set_particle_target(world, particles))
        states.append(world.state_dict())
    if cache is not None:
        # Store an isolated semantic snapshot.  Every read returns another
        # deep pickle clone so runner construction can never poison the cache.
        cache[key] = (validation.pickle_clone(states), tuple(actual_particles))
    return states, actual_particles


def _cpu_runners(module, states):
    return [module.s66.Formal066World.from_state(validation.pickle_clone(state)) for state in states]


def _hybrid_runners(states, device, precision, config_overrides=None):
    return [
        validation._hybrid_from_state(
            state, device=device, precision=precision,
            config_overrides=config_overrides,
        )
        for state in states
    ]


def _advance(runners, steps, dt):
    for _ in range(steps):
        for runner in runners:
            runner.step(dt)


def timed_reference(torch, module, states, steps, dt, warmup, repeats):
    for _ in range(warmup):
        _advance(_cpu_runners(module, states), steps, dt)
    samples = []
    particle_final = []
    for _ in range(repeats):
        runners = _cpu_runners(module, states)
        started = time.perf_counter()
        _advance(runners, steps, dt)
        samples.append(time.perf_counter() - started)
        particle_final.append(sum(len(world.field.amount) for world in runners))
    report = summary_samples(samples, len(states) * steps)
    report.update({
        'runner': 'frozen-0.6.6-cpu-reference',
        'device': 'cpu', 'precision': 'float64', 'warmup_repeats': warmup,
        'timed_repeats': repeats, 'steps': steps,
        'particles_final_per_repeat': particle_final,
    })
    return report


def timed_hybrid(torch, states, steps, dt, warmup, repeats, device, precision,
                 config_overrides=None):
    synchronise(torch, device)
    for _ in range(warmup):
        runners = _hybrid_runners(states, device, precision, config_overrides)
        _advance(runners, steps, dt)
        synchronise(torch, device)
    samples = []
    peak_allocated = []
    peak_reserved = []
    backend_stats = []
    particle_final = []
    for _ in range(repeats):
        if str(device).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(torch.device(device))
        runners = _hybrid_runners(states, device, precision, config_overrides)
        synchronise(torch, device)
        started = time.perf_counter()
        _advance(runners, steps, dt)
        synchronise(torch, device)
        samples.append(time.perf_counter() - started)
        if str(device).startswith('cuda'):
            peak_allocated.append(int(torch.cuda.max_memory_allocated(torch.device(device))))
            peak_reserved.append(int(torch.cuda.max_memory_reserved(torch.device(device))))
        backend_stats.append([runner.backend.stats() for runner in runners])
        particle_final.append(sum(len(runner.world.field.amount) for runner in runners))
    report = summary_samples(samples, len(states) * steps)
    report.update({
        'runner': 'A3-hybrid-independent-worlds-sequential-host-loop',
        'device': str(device), 'precision': precision,
        'warmup_repeats': warmup, 'timed_repeats': repeats, 'steps': steps,
        'peak_vram_allocated_bytes_per_repeat': peak_allocated,
        'peak_vram_reserved_bytes_per_repeat': peak_reserved,
        'peak_vram_allocated_bytes_max': max(peak_allocated or [0]),
        'peak_vram_reserved_bytes_max': max(peak_reserved or [0]),
        'particles_final_per_repeat': particle_final,
        'backend_stats_per_repeat': backend_stats,
        'configured_capacities': copy.deepcopy(config_overrides or {}),
        'full_gpu_world_step': False,
        'execution_note': 'Worlds are independent but this A3 hybrid runner still uses a sequential host loop.',
    })
    return report


def _diff_report(cpu, hybrid):
    differences = validation.recursive_numeric_differences(cpu.state_dict(), hybrid.world.state_dict())
    ordered = sorted(differences, key=lambda item: item[1], reverse=True)
    finite = [value for _, value in differences if math.isfinite(value)]
    return {
        'numeric_field_count': len(differences),
        'max_abs_diff': float(max(finite or [0.0])),
        'structural_or_discrete_mismatch': any(not math.isfinite(value) for _, value in differences),
        'top_field_differences': [{'path': path, 'max_abs_diff': value} for path, value in ordered[:30]],
    }


def correctness_probe(torch, module, device, precision, steps, seed, cells=2,
                      dt=0.1, particles=None, config_overrides=None):
    """Run a full-state probe; fp32 is discrepancy-only, never equivalence."""
    is_fp64 = precision == 'float64'
    effective_overrides = _precision_overrides(precision, config_overrides)
    report = {
        'device': str(device), 'precision': precision, 'steps': int(steps),
        'seed': int(seed), 'cells_requested': int(cells),
        'particles_requested': None if particles is None else int(particles),
        'configured_capacities': _capacity_report(
            device, precision, effective_overrides,
        ),
        'candidate_only': not is_fp64,
        'state_tolerance': validation.WORLD_FP64_ATOL if is_fp64 else None,
        'material_ledger_tolerance': validation.LEDGER_ATOL if is_fp64 else None,
        'execution_succeeded': False,
    }
    if not is_fp64:
        report.update({
            'candidate_internal_ledger_tolerance': FP32_CANDIDATE_LEDGER_ATOL,
            'candidate_internal_ledger_tolerance_reason': FP32_CANDIDATE_LEDGER_REASON,
            'strict_material_ledger': True,
        })
    try:
        base = validation.make_world(seed=seed, cells=cells)
        actual_particles = _set_particle_target(base, particles)
        state = validation.pickle_clone(base.state_dict())
        cpu = module.s66.Formal066World.from_state(validation.pickle_clone(state))
        hybrid = validation._hybrid_from_state(
            state, device=device, precision=precision,
            config_overrides=effective_overrides,
        )
        for _ in range(steps):
            cpu.step(dt)
            hybrid.step(dt)
        synchronise(torch, device)
        report.update(_diff_report(cpu, hybrid))
        report.update({
            'execution_succeeded': True,
            'cells_initial': int(cells),
            'particles_initial': int(actual_particles),
            'cells_final_cpu': len(cpu.cells),
            'cells_final_hybrid': len(hybrid.world.cells),
            'particles_final_cpu': len(cpu.field.amount),
            'particles_final_hybrid': len(hybrid.world.field.amount),
            'rng_equal': cpu.rng.bit_generator.state == hybrid.world.rng.bit_generator.state,
            'material_residual_cpu': float(cpu.matter_ledger_residual()),
            'material_residual_hybrid': float(hybrid.world.matter_ledger_residual()),
        })
        ledger_difference = abs(
            report['material_residual_cpu'] - report['material_residual_hybrid']
        )
        report['material_residual_abs_diff'] = float(ledger_difference)
        report['nonfinite_numeric_difference'] = any(
            not math.isfinite(item['max_abs_diff'])
            for item in report.get('top_field_differences', [])
        )
        try:
            validation.assert_recursive_close(
                cpu.state_dict(), hybrid.world.state_dict(),
                atol=(validation.WORLD_FP64_ATOL if is_fp64 else 0.0),
                rtol=0.0, path='correctness_probe',
            )
            recursive_ok = True
            recursive_error = ''
        except Exception as exc:
            recursive_ok = False
            recursive_error = '{}: {}'.format(type(exc).__name__, exc)
        report['recursive_state_within_fixed_fp64_tolerance'] = (
            recursive_ok if is_fp64 else None
        )
        report['recursive_error'] = recursive_error
        report['fp64_lockstep_pass'] = (
            bool(
                recursive_ok and report['rng_equal']
                and ledger_difference <= validation.LEDGER_ATOL
            ) if is_fp64 else None
        )
        report['fp32_discrepancy_recorded'] = (not is_fp64)
        if not is_fp64:
            report['candidate_bound_exceeded'] = bool(
                report['structural_or_discrete_mismatch']
                or report['nonfinite_numeric_difference']
                or not report['rng_equal']
                or ledger_difference > FP32_CANDIDATE_LEDGER_ATOL
            )
    except Exception as exc:
        report['execution_error'] = '{}: {}'.format(type(exc).__name__, exc)
        report['fp64_lockstep_pass'] = False if is_fp64 else None
    return report


def kernel_precision_probe(torch, device):
    """Gate fp32 with pure-kernel CUDA/CPU fp64 parity first."""
    result = {'device': str(device), 'fp64_gate_pass': False, 'fp32_run': False}
    seed = 9901
    try:
        numpy64, torch64 = validation._kernel_pair(
            'stressed', device=device, precision='float64',
            atol=validation.PURE_FP64_ATOL,
            seed=seed,
        )
        result['fp64_gate_pass'] = True
        result['fp64_tolerance'] = validation.PURE_FP64_ATOL
    except Exception as exc:
        result['fp64_error'] = '{}: {}'.format(type(exc).__name__, exc)
        return result
    try:
        cell, cpu_config = validation.scenario_cell('stressed', seed=seed)
        packed = validation._normalise_kernel_config(
            validation.pack_cell(cell), cpu_config,
        )
        state32 = validation._to_torch_state(
            copy.deepcopy(packed), device=device, precision='float32',
        )
        torch_fn, = validation.require_api('metabolism_damage_torch')
        output32 = torch_fn(state32, 0.1, config=cpu_config)
        synchronise(torch, device)
        differences = validation.recursive_numeric_differences(numpy64, output32)
        ordered = sorted(differences, key=lambda item: item[1], reverse=True)
        result.update({
            'fp32_run': True,
            'fp32_candidate_only': True,
            'fp32_numeric_field_count': len(differences),
            'fp32_structural_or_discrete_mismatch': any(
                not math.isfinite(v) for _, v in differences
            ),
            'fp32_max_abs_diff': max(
                [v for _, v in differences if math.isfinite(v)] or [0.0]
            ),
            'fp32_top_field_differences': [
                {'path': path, 'max_abs_diff': value}
                for path, value in ordered[:30]
            ],
        })
    except Exception as exc:
        result['fp32_error'] = '{}: {}'.format(type(exc).__name__, exc)
    return result


def _capacity_overrides(spec):
    return {
        'max_protein_species': int(spec['protein_capacity']),
        'batch_worlds': int(spec['worlds']),
    }


def _precision_overrides(precision, overrides):
    result = dict(overrides or {})
    result['strict_material_ledger'] = True
    if precision == 'float32':
        result['ledger_atol'] = FP32_CANDIDATE_LEDGER_ATOL
    return result


def _capacity_report(device, precision, overrides):
    effective = _precision_overrides(precision, overrides)
    config = validation._a3_config(
        device=device, precision=precision, **effective
    )
    result = {
        name: int(getattr(config, name))
        for name in (
            'batch_worlds', 'max_cells', 'max_particles',
            'max_protein_species', 'max_genome_copies',
            'max_genome_symbols',
        )
    }
    result.update({
        'strict_material_ledger': bool(config.strict_material_ledger),
        'ledger_atol': float(config.ledger_atol),
    })
    if precision == 'float32':
        result['fp32_candidate_ledger_tolerance_reason'] = FP32_CANDIDATE_LEDGER_REASON
    return result


def _sweep_specs(args, particle_targets):
    specs = []
    for count in args.worlds:
        specs.append({
            'axis': 'worlds', 'axis_value': int(count),
            'worlds': int(count), 'cells': int(args.cells),
            'particles': args.particles,
            'protein_capacity': int(args.base_protein_capacity),
        })
    for cells in args.cell_sweep:
        specs.append({
            'axis': 'cells', 'axis_value': int(cells),
            'worlds': 1, 'cells': int(cells),
            'particles': args.particles,
            'protein_capacity': int(args.base_protein_capacity),
        })
    for particles in particle_targets:
        specs.append({
            'axis': 'particles', 'axis_value': int(particles),
            'worlds': 1, 'cells': int(args.cells),
            'particles': int(particles),
            'protein_capacity': int(args.base_protein_capacity),
        })
    for capacity in args.capacity_sweep:
        specs.append({
            'axis': 'protein_capacity', 'axis_value': int(capacity),
            'worlds': 1, 'cells': int(args.cells),
            'particles': args.particles,
            'protein_capacity': int(capacity),
        })
    return specs


def _annotate_measurement(measurement, spec, actual_particles):
    measurement.update({
        'status': measurement.get('status', 'PASS'),
        'sweep_axis': spec['axis'], 'sweep_axis_value': spec['axis_value'],
        'worlds': spec['worlds'], 'cells_initial_per_world': spec['cells'],
        'particles_requested_per_world': spec['particles'],
        'particles_actual_initial_per_world': [int(v) for v in actual_particles],
        'protein_capacity_requested': spec['protein_capacity'],
    })
    return measurement


def _timing_policy(args, spec):
    if spec['axis'] == 'worlds':
        return int(args.world_warmup), int(args.world_repeats)
    return int(args.other_axis_warmup), int(args.other_axis_repeats)


def _finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _correctness_matrix_errors(report, requested_devices, requested_steps,
                               fp32_requested):
    """Independently audit every requested correctness probe.

    The booleans stored by an individual probe are useful diagnostics, but the
    benchmark completion gate is derived again from the primitive results so
    an empty or partial probe list can never be interpreted as success.
    """
    errors = []
    if (
        not requested_devices
        or len(requested_devices) != len(set(requested_devices))
    ):
        errors.append('requested correctness devices are empty or duplicate')
    if not requested_steps or len(requested_steps) != len(set(requested_steps)):
        errors.append('requested correctness steps are empty or duplicate')
    correctness = report.get('correctness')
    precision = report.get('precision_discrepancy')
    if not isinstance(correctness, dict):
        return errors + ['correctness payload is not a mapping']
    if set(correctness) != set(requested_devices):
        errors.append('correctness device set does not match requested devices')
    if not isinstance(precision, dict):
        errors.append('precision_discrepancy payload is not a mapping')
        precision = {}
    elif set(precision) != set(requested_devices):
        errors.append('precision discrepancy device set does not match requests')

    for device in requested_devices:
        device_report = correctness.get(device)
        if not isinstance(device_report, dict):
            errors.append('{} correctness report is missing'.format(device))
            continue
        if device_report.get('status') != 'PASS':
            errors.append('{} correctness status is not PASS'.format(device))
        probes = device_report.get('probes')
        if not isinstance(probes, list):
            errors.append('{} correctness probes are missing'.format(device))
            continue
        observed_steps = [probe.get('steps') for probe in probes if isinstance(probe, dict)]
        if observed_steps != list(requested_steps) or len(probes) != len(requested_steps):
            errors.append('{} correctness steps are incomplete or reordered'.format(device))
        for probe in probes:
            if not isinstance(probe, dict):
                errors.append('{} correctness probe is not a mapping'.format(device))
                continue
            step = probe.get('steps')
            prefix = '{} correctness step {}'.format(device, step)
            if probe.get('device') != str(device) or probe.get('precision') != 'float64':
                errors.append('{} identity mismatch'.format(prefix))
            if probe.get('execution_succeeded') is not True:
                errors.append('{} did not execute'.format(prefix))
            if probe.get('fp64_lockstep_pass') is not True:
                errors.append('{} fp64 lockstep gate failed'.format(prefix))
            if probe.get('recursive_state_within_fixed_fp64_tolerance') is not True:
                errors.append('{} recursive state gate failed'.format(prefix))
            if probe.get('rng_equal') is not True:
                errors.append('{} RNG gate failed'.format(prefix))
            if probe.get('structural_or_discrete_mismatch') is not False:
                errors.append('{} structural/discrete gate failed'.format(prefix))
            if probe.get('nonfinite_numeric_difference') is not False:
                errors.append('{} contains a non-finite difference'.format(prefix))
            state_difference = probe.get('max_abs_diff')
            if (
                not _finite_number(state_difference)
                or float(state_difference) > validation.WORLD_FP64_ATOL
            ):
                errors.append('{} exceeds the fixed state tolerance'.format(prefix))
            ledger_difference = probe.get('material_residual_abs_diff')
            if (
                not _finite_number(ledger_difference)
                or float(ledger_difference) > validation.LEDGER_ATOL
            ):
                errors.append('{} exceeds the fixed ledger tolerance'.format(prefix))
            for name in ('material_residual_cpu', 'material_residual_hybrid'):
                if not _finite_number(probe.get(name)):
                    errors.append('{} lacks finite {}'.format(prefix, name))

        discrepancy = precision.get(device)
        if not isinstance(discrepancy, dict):
            errors.append('{} precision discrepancy report is missing'.format(device))
            continue
        if discrepancy.get('fp64_gate_pass') is not True:
            errors.append('{} pure fp64 precision gate failed'.format(device))
        if not fp32_requested:
            continue
        if discrepancy.get('fp32_run') is not True:
            errors.append('{} fp32 discrepancy probe did not run'.format(device))
        if discrepancy.get('fp32_structural_or_discrete_mismatch') is not False:
            errors.append('{} fp32 pure probe has a structural mismatch'.format(device))
        fp32_probes = discrepancy.get('fp32_world_probes')
        if not isinstance(fp32_probes, list):
            errors.append('{} fp32 world probes are missing'.format(device))
            continue
        fp32_steps = [probe.get('steps') for probe in fp32_probes if isinstance(probe, dict)]
        if fp32_steps != list(requested_steps) or len(fp32_probes) != len(requested_steps):
            errors.append('{} fp32 world probe steps are incomplete'.format(device))
        for probe in fp32_probes:
            if not isinstance(probe, dict):
                errors.append('{} fp32 world probe is not a mapping'.format(device))
                continue
            step = probe.get('steps')
            prefix = '{} fp32 candidate step {}'.format(device, step)
            if probe.get('execution_succeeded') is not True:
                errors.append('{} did not execute'.format(prefix))
            if probe.get('candidate_bound_exceeded') is not False:
                errors.append('{} exceeded its candidate bound'.format(prefix))
            if probe.get('rng_equal') is not True:
                errors.append('{} RNG differs'.format(prefix))
            if probe.get('structural_or_discrete_mismatch') is not False:
                errors.append('{} has a structural/discrete mismatch'.format(prefix))
            if probe.get('nonfinite_numeric_difference') is not False:
                errors.append('{} contains a non-finite difference'.format(prefix))
            ledger_difference = probe.get('material_residual_abs_diff')
            if (
                not _finite_number(ledger_difference)
                or float(ledger_difference) > FP32_CANDIDATE_LEDGER_ATOL
            ):
                errors.append('{} exceeds candidate ledger bound'.format(prefix))
    return errors


def _timing_row_errors(item, warmup, repeats, worlds, steps, cuda):
    errors = []
    if item.get('warmup_repeats') != warmup:
        errors.append('warmup_repeats mismatch')
    if item.get('timed_repeats') != repeats:
        errors.append('timed_repeats mismatch')
    samples = item.get('samples_seconds')
    if (
        not isinstance(samples, list)
        or len(samples) != repeats
        or any(not _finite_number(value) or float(value) <= 0.0 for value in samples)
    ):
        errors.append('samples_seconds missing/non-finite/non-positive')
    for name in (
        'minimum_seconds', 'median_seconds', 'maximum_seconds',
        'mean_seconds', 'pstdev_seconds', 'work_items_per_second_at_median',
    ):
        if not _finite_number(item.get(name)) or float(item[name]) < 0.0:
            errors.append('{} missing/non-finite/negative'.format(name))
    expected_work = int(worlds) * int(steps)
    if item.get('work_items') != expected_work or expected_work <= 0:
        errors.append('work_items mismatch')
    final_particles = item.get('particles_final_per_repeat')
    if (
        not isinstance(final_particles, list)
        or len(final_particles) != repeats
        or any(not isinstance(value, int) or value < 0 for value in final_particles)
    ):
        errors.append('particles_final_per_repeat malformed')
    if cuda:
        cuda_maxima = {}
        for sequence_name, maximum_name in (
            ('peak_vram_allocated_bytes_per_repeat', 'peak_vram_allocated_bytes_max'),
            ('peak_vram_reserved_bytes_per_repeat', 'peak_vram_reserved_bytes_max'),
        ):
            sequence = item.get(sequence_name)
            maximum = item.get(maximum_name)
            if (
                not isinstance(sequence, list)
                or len(sequence) != repeats
                or any(not isinstance(value, int) or value < 0 for value in sequence)
                or not isinstance(maximum, int)
                or maximum < 0
                or maximum != max(sequence or [0])
            ):
                errors.append('{} / {} malformed'.format(sequence_name, maximum_name))
            else:
                cuda_maxima[maximum_name] = maximum
        if cuda_maxima.get('peak_vram_allocated_bytes_max', 0) <= 0:
            errors.append('CUDA peak allocated VRAM is not positive')
        if (
            cuda_maxima.get('peak_vram_reserved_bytes_max', -1)
            < cuda_maxima.get('peak_vram_allocated_bytes_max', 0)
        ):
            errors.append('CUDA peak reserved VRAM is below allocated VRAM')
    return errors


def _measurement_matrix_errors(report, specs, requested_devices,
                               requested_precisions, args):
    """Require one reference plus every device/precision row per sweep spec."""
    errors = []
    if not specs:
        return ['requested sweep is empty']
    identities = [(spec.get('axis'), spec.get('axis_value')) for spec in specs]
    if len(identities) != len(set(identities)):
        errors.append('requested sweep contains duplicate axis/value identities')
    if (
        not requested_devices or len(requested_devices) != len(set(requested_devices))
        or not requested_precisions
        or len(requested_precisions) != len(set(requested_precisions))
    ):
        errors.append('requested device/precision matrix is empty or duplicate')

    expected = {}
    for spec in specs:
        axis_value = (spec['axis'], spec['axis_value'])
        reference_key = axis_value + ('frozen-0.6.6-cpu-reference', 'cpu', 'float64')
        expected[reference_key] = (spec, 'PASS')
        for device in requested_devices:
            for precision in requested_precisions:
                key = axis_value + (
                    'A3-hybrid-independent-worlds-sequential-host-loop',
                    str(device), precision,
                )
                expected[key] = (
                    spec,
                    'MEASURED_CANDIDATE' if precision == 'float32' else 'PASS',
                )

    measurements = report.get('measurements')
    if not isinstance(measurements, list):
        return errors + ['measurements payload is not a list']
    observed = {}
    for index, item in enumerate(measurements):
        if not isinstance(item, dict):
            errors.append('measurement {} is not a mapping'.format(index))
            continue
        key = (
            item.get('sweep_axis'), item.get('sweep_axis_value'),
            item.get('runner'), item.get('device'), item.get('precision'),
        )
        if key in observed:
            errors.append('duplicate measurement identity {!r}'.format(key))
            continue
        observed[key] = item
        if key not in expected:
            errors.append('unexpected measurement identity {!r}'.format(key))
            continue
        spec, expected_status = expected[key]
        if item.get('status') != expected_status or item.get('error') not in (None, ''):
            errors.append('measurement {!r} has invalid status/error'.format(key))
        precision = item.get('precision')
        is_hybrid = item.get('runner') == 'A3-hybrid-independent-worlds-sequential-host-loop'
        if item.get('prerequisite_fp64_correctness_gate') != 'PASS':
            errors.append('measurement {!r} lacks prerequisite fp64 gate'.format(key))
        if is_hybrid:
            expected_precision_status = (
                'fp32-candidate-only-after-fp64-pass'
                if precision == 'float32'
                else 'fp64-reference-correctness-gated'
            )
            if item.get('precision_status') != expected_precision_status:
                errors.append('measurement {!r} precision label mismatch'.format(key))
        warmup, repeats = _timing_policy(args, spec)
        row_errors = _timing_row_errors(
            item, warmup, repeats, spec['worlds'], args.steps,
            is_hybrid and str(item.get('device')).startswith('cuda'),
        )
        errors.extend(
            'measurement {!r}: {}'.format(key, message)
            for message in row_errors
        )
        actual_particles = item.get('particles_actual_initial_per_world')
        if (
            not isinstance(actual_particles, list)
            or len(actual_particles) != int(spec['worlds'])
            or any(not isinstance(value, int) or value <= 0 for value in actual_particles)
        ):
            errors.append('measurement {!r} initial particle counts malformed'.format(key))

    missing = sorted(set(expected) - set(observed), key=repr)
    if missing:
        errors.append('missing measurement identities: {!r}'.format(missing))
    if len(measurements) != len(expected):
        errors.append(
            'measurement row count {} != expected {}'.format(
                len(measurements), len(expected),
            )
        )
    return errors


def run_benchmark(args):
    torch = validation._torch()
    module = validation.a3_module()
    validation.require_api('Hybrid066WorldA3', 'TorchKernelBackendA3', 'metabolism_damage_torch')
    requested_devices = list(args.device or ['cpu'])
    if args.auto_cuda and torch.cuda.is_available() and 'cuda' not in requested_devices:
        requested_devices.append('cuda')
    requested_precisions = (
        ['float64', 'float32'] if args.precision == 'both' or args.include_fp32
        else [args.precision]
    )
    probe_world = validation.make_world(seed=args.seed, cells=args.cells)
    canonical_particles = len(probe_world.field.amount)
    particle_targets = list(args.particle_sweep or [
        canonical_particles,
        min(360, canonical_particles + max(1, (360 - canonical_particles) // 2)),
        360,
    ])
    particle_targets = list(dict.fromkeys(int(value) for value in particle_targets))
    specs = _sweep_specs(args, particle_targets)
    formal_release_profile = bool(
        requested_devices == ['cpu', 'cuda']
        and requested_precisions == ['float64', 'float32']
        and list(args.worlds) == list(FORMAL_WORLD_COUNTS)
        and list(args.cell_sweep) == list(FORMAL_CELL_SWEEP)
        and particle_targets == list(FORMAL_PARTICLE_SWEEP)
        and list(args.capacity_sweep) == list(FORMAL_CAPACITY_SWEEP)
        and int(args.cells) == 3
        and args.particles is None
        and int(args.base_protein_capacity) == 256
        and int(args.steps) == FORMAL_TIMING_STEPS
        and int(args.world_warmup) == 1
        and int(args.world_repeats) == 2
        and int(args.other_axis_warmup) == 0
        and int(args.other_axis_repeats) == 1
        and list(args.correctness_steps) == [1, 10, 20]
    )
    report = {
        'schema_version': '2.0',
        'build': module.BUILD,
        'schema': module.SCHEMA_VERSION,
        'source': os.path.basename(module.__file__),
        'source_sha256': sha256(module.__file__),
        'source_sha256_map': source_sha256_map(),
        'git_head': git_value('rev-parse', 'HEAD'),
        'git_status': git_value('status', '--porcelain'),
        'captured_utc': datetime.now(timezone.utc).isoformat(),
        'environment': environment_report(torch),
        'parameters': {
            'devices': requested_devices, 'precisions': requested_precisions,
            'world_counts': args.worlds,
            'cells': args.cells, 'steps': args.steps, 'dt': args.dt,
            'fixed_particles': args.particles,
            'timing_policy_by_axis': {
                'worlds': {
                    'warmup': int(args.world_warmup),
                    'repeats': int(args.world_repeats),
                },
                'cells_particles_capacity': {
                    'warmup': int(args.other_axis_warmup),
                    'repeats': int(args.other_axis_repeats),
                },
            },
            'correctness_steps': args.correctness_steps,
            'cell_sweep': args.cell_sweep,
            'particle_sweep': particle_targets,
            'capacity_sweep': args.capacity_sweep,
            'base_protein_capacity': args.base_protein_capacity,
            'canonical_initial_particles': canonical_particles,
            'include_fp32': 'float32' in requested_precisions,
        },
        'fixed_tolerances': {
            'pure_fp64_max_abs': validation.PURE_FP64_ATOL,
            'world_fp64_max_abs': validation.WORLD_FP64_ATOL,
            'material_ledger_residual_abs_diff': validation.LEDGER_ATOL,
            'discrete_rng_keys_orders': 'exact',
        },
        'full_gpu_world_step': False,
        'formal_release_profile_requested': formal_release_profile,
        'correctness': {}, 'precision_discrepancy': {}, 'measurements': [],
        'errors': [],
        'claim_limit': 'Engineering benchmark only; no life, full-GPU, or speedup claim is implied.',
    }
    timing_precisions = {}
    for device in requested_devices:
        if str(device).startswith('cuda') and not torch.cuda.is_available():
            report['errors'].append('CUDA requested but torch.cuda.is_available() is false')
            report['correctness'][device] = {'status': 'NOT_RUN', 'reason': 'CUDA unavailable'}
            continue
        probes = [
            correctness_probe(
                torch, module, device, 'float64', steps,
                args.seed + steps, args.cells, args.dt,
            )
            for steps in args.correctness_steps
        ]
        fp64_pass = all(probe.get('fp64_lockstep_pass') is True for probe in probes)
        report['correctness'][device] = {
            'status': 'PASS' if fp64_pass else 'FAIL',
            'precision': 'float64', 'probes': probes,
        }
        precision_probe = kernel_precision_probe(torch, device)
        report['precision_discrepancy'][device] = precision_probe
        if not fp64_pass or not precision_probe['fp64_gate_pass']:
            report['errors'].append('{} fp64 correctness gate failed; timing blocked'.format(device))
            continue
        precisions = ['float64'] if 'float64' in requested_precisions else []
        if 'float32' in requested_precisions:
            if not precision_probe.get('fp32_run'):
                report['errors'].append('{} fp32 requested but discrepancy probe did not run'.format(device))
            else:
                # Full-world fp32 is a discrepancy measurement, not an
                # equivalence assertion.  Record all structural/RNG/ledger
                # consequences before allowing throughput timing.
                fp32_probe_overrides = _precision_overrides('float32', {})
                precision_probe['fp32_candidate_internal_ledger_tolerance'] = (
                    FP32_CANDIDATE_LEDGER_ATOL
                )
                precision_probe['fp32_candidate_internal_ledger_tolerance_reason'] = (
                    FP32_CANDIDATE_LEDGER_REASON
                )
                precision_probe['fp32_world_probes'] = [
                    correctness_probe(
                        torch, module, device, 'float32', steps,
                        args.seed + 5000 + steps, args.cells, args.dt,
                        config_overrides=fp32_probe_overrides,
                    )
                    for steps in args.correctness_steps
                ]
                fp32_execution = all(
                    probe.get('execution_succeeded') is True
                    and probe.get('candidate_bound_exceeded') is False
                    for probe in precision_probe['fp32_world_probes']
                )
                if fp32_execution:
                    precisions.append('float32')
                else:
                    report['errors'].append(
                        '{} fp32 candidate full-world execution failed; fp32 timing blocked'.format(
                            device,
                        )
                    )
        timing_precisions[device] = precisions

    preliminary_all_fp64_gates = bool(requested_devices) and all(
        report['correctness'].get(device, {}).get('status') == 'PASS'
        and report['precision_discrepancy'].get(device, {}).get('fp64_gate_pass') is True
        for device in requested_devices
    )
    if preliminary_all_fp64_gates:
        state_cache = {}
        reference_cache = {}
        for spec_index, spec in enumerate(specs):
            warmup, repeats = _timing_policy(args, spec)
            state_seed = (
                args.seed + 10000 + spec['worlds'] * 100
                + spec['cells'] * 10 + int(spec['particles'] or 0)
            )
            try:
                states, actual_particles = _base_states(
                    spec['worlds'], spec['cells'],
                    # A spec is seeded from its concrete shape, not its axis
                    # ordinal.  Duplicate shape/specs can therefore reuse the
                    # cache without changing the benchmark state.
                    state_seed,
                    particles=spec['particles'],
                    cache=state_cache,
                )
            except Exception as exc:
                report['measurements'].append(_annotate_measurement({
                    'runner': 'state-construction', 'status': 'FAIL',
                    'error': '{}: {}'.format(type(exc).__name__, exc),
                }, spec, []))
                report['errors'].append(
                    '{} sweep {} state construction failed: {}'.format(
                        spec['axis'], spec['axis_value'], exc,
                    )
                )
                continue
            try:
                reference_key = (
                    _state_cache_key(
                        spec['worlds'], spec['cells'], state_seed,
                        spec['particles'],
                    ),
                    int(args.steps), float(args.dt), warmup, repeats,
                )
                if reference_key in reference_cache:
                    reference = copy.deepcopy(reference_cache[reference_key])
                    reference['measurement_reused_from_safe_cache'] = True
                else:
                    reference = timed_reference(
                        torch, module, states, args.steps, args.dt,
                        warmup, repeats,
                    )
                    reference['measurement_reused_from_safe_cache'] = False
                    reference_cache[reference_key] = copy.deepcopy(reference)
                _annotate_measurement(reference, spec, actual_particles)
                reference['prerequisite_fp64_correctness_gate'] = 'PASS'
                report['measurements'].append(reference)
            except Exception as exc:
                report['measurements'].append(_annotate_measurement({
                    'runner': 'frozen-0.6.6-cpu-reference', 'status': 'FAIL',
                    'error': '{}: {}'.format(type(exc).__name__, exc),
                }, spec, actual_particles))
                report['errors'].append(
                    '{} sweep {} CPU reference timing failed: {}'.format(
                        spec['axis'], spec['axis_value'], exc,
                    )
                )
            overrides = _capacity_overrides(spec)
            for device in requested_devices:
                for precision in timing_precisions.get(device, []):
                    try:
                        measurement = timed_hybrid(
                            torch, states, args.steps, args.dt,
                            warmup, repeats, device, precision,
                            config_overrides=_precision_overrides(
                                precision, overrides,
                            ),
                        )
                        _annotate_measurement(measurement, spec, actual_particles)
                        measurement['configured_capacities'] = _capacity_report(
                            device, precision, overrides,
                        )
                        measurement['prerequisite_fp64_correctness_gate'] = 'PASS'
                        measurement['precision_status'] = (
                            'fp64-reference-correctness-gated'
                            if precision == 'float64'
                            else 'fp32-candidate-only-after-fp64-pass'
                        )
                        if precision == 'float32':
                            measurement['status'] = 'MEASURED_CANDIDATE'
                            measurement['fp32_discrepancy_report'] = copy.deepcopy(
                                report['precision_discrepancy'][device]
                            )
                        report['measurements'].append(measurement)
                    except Exception as exc:
                        report['measurements'].append(_annotate_measurement({
                            'runner': 'A3-hybrid-independent-worlds-sequential-host-loop',
                            'device': device, 'precision': precision,
                            'status': 'FAIL',
                            'error': '{}: {}'.format(type(exc).__name__, exc),
                            'configured_capacities': _capacity_report(
                                device, precision, overrides,
                            ),
                            'prerequisite_fp64_correctness_gate': 'PASS',
                            'precision_status': (
                                'fp64-reference-correctness-gated'
                                if precision == 'float64'
                                else 'fp32-candidate-only-after-fp64-pass'
                            ),
                        }, spec, actual_particles))
                        report['errors'].append(
                            '{} {} {}={} timing failed: {}'.format(
                                device, precision, spec['axis'],
                                spec['axis_value'], exc,
                            )
                        )
    fp64_correctness_errors = _correctness_matrix_errors(
        report, requested_devices, args.correctness_steps, False,
    )
    correctness_errors = _correctness_matrix_errors(
        report, requested_devices, args.correctness_steps,
        'float32' in requested_precisions,
    )
    report['correctness_matrix_validation'] = {
        'status': 'PASS' if not correctness_errors else 'FAIL',
        'expected_devices': list(requested_devices),
        'expected_steps': list(args.correctness_steps),
        'fp64_errors': fp64_correctness_errors,
        'errors': correctness_errors,
    }
    for message in correctness_errors:
        report['errors'].append('correctness matrix: {}'.format(message))
    report['all_requested_fp64_gates_pass'] = not fp64_correctness_errors
    report['fp64_gate_passed_before_fp32'] = bool(
        report['all_requested_fp64_gates_pass']
        and 'float32' in requested_precisions
    )
    measurement_errors = _measurement_matrix_errors(
        report, specs, requested_devices, requested_precisions, args,
    )
    expected_measurements = len(specs) * (
        1 + len(requested_devices) * len(requested_precisions)
    )
    report['measurement_matrix_validation'] = {
        'status': 'PASS' if not measurement_errors else 'FAIL',
        'expected_sweep_specs': len(specs),
        'expected_rows': expected_measurements,
        'observed_rows': len(report.get('measurements', [])),
        'errors': measurement_errors,
    }
    for message in measurement_errors:
        report['errors'].append('measurement matrix: {}'.format(message))
    report['all_requested_measurements_completed'] = not measurement_errors
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', action='append', choices=('cpu', 'cuda'),
                        help='repeat to benchmark multiple devices; default cpu')
    parser.add_argument('--auto-cuda', action='store_true',
                        help='also use CUDA when available')
    parser.add_argument('--worlds', type=int, nargs='+', default=list(FORMAL_WORLD_COUNTS))
    parser.add_argument('--cells', type=int, default=3)
    parser.add_argument('--particles', type=int,
                        help='fixed particle target for world/cell/capacity axes')
    parser.add_argument('--cell-sweep', type=int, nargs='+', default=list(FORMAL_CELL_SWEEP))
    parser.add_argument('--particle-sweep', type=int, nargs='+',
                        default=list(FORMAL_PARTICLE_SWEEP))
    parser.add_argument('--capacity-sweep', type=int, nargs='+',
                        default=list(FORMAL_CAPACITY_SWEEP),
                        help='packed max_protein_species values')
    parser.add_argument('--base-protein-capacity', type=int, default=256)
    parser.add_argument('--steps', type=int, default=FORMAL_TIMING_STEPS)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--world-warmup', type=int, default=1,
                        help='warmup repeats for the 1/8/32/128 worlds axis')
    parser.add_argument('--world-repeats', type=int, default=2,
                        help='timed repeats for the worlds axis')
    parser.add_argument('--other-axis-warmup', type=int, default=0,
                        help='warmup repeats for cells/particles/capacity axes')
    parser.add_argument('--other-axis-repeats', type=int, default=1,
                        help='timed repeats for cells/particles/capacity axes')
    parser.add_argument('--correctness-steps', type=int, nargs='+', default=[1, 10, 20])
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--include-fp32', action='store_true')
    parser.add_argument('--precision', choices=('float64', 'float32', 'both'),
                        default='float64')
    parser.add_argument('--output', default='soma_cell_0_6_8_gpu_a3_benchmark.json')
    return parser


def _write_report_bundle(report, output):
    output = os.path.abspath(output)
    directory = os.path.dirname(output)
    if directory:
        os.makedirs(directory, exist_ok=True)
    stem, extension = os.path.splitext(output)
    if extension.lower() != '.json':
        stem = output
        output = output + '.json'
    csv_path = stem + '.csv'
    fields = (
        'status', 'runner', 'device', 'precision', 'precision_status',
        'sweep_axis', 'sweep_axis_value', 'worlds',
        'cells_initial_per_world', 'particles_requested_per_world',
        'particles_actual_initial_per_world', 'protein_capacity_requested',
        'warmup_repeats', 'timed_repeats', 'steps',
        'minimum_seconds', 'median_seconds', 'maximum_seconds',
        'mean_seconds', 'pstdev_seconds', 'work_items',
        'work_items_per_second_at_median', 'peak_vram_allocated_bytes_max',
        'peak_vram_reserved_bytes_max', 'error',
    )
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for measurement in report.get('measurements', []):
            row = {}
            for name in fields:
                value = measurement.get(name, '')
                if isinstance(value, (list, tuple, dict)):
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                row[name] = value
            writer.writerow(row)
    txt_path = stem + '.txt'
    lines = [
        'SOMA-CELL 0.6.8-GPU A3 BENCHMARK', '=' * 52,
        'captured_utc: {}'.format(report.get('captured_utc')),
        'fp64 gates: {}'.format(report.get('all_requested_fp64_gates_pass')),
        'measurements complete: {}'.format(
            report.get('all_requested_measurements_completed')
        ),
        'measurements: {}'.format(len(report.get('measurements', []))),
        'errors: {}'.format(len(report.get('errors', []))),
        'full_gpu_world_step: false',
        'fp32: candidate only after fp64 PASS', '',
    ]
    lines.extend('ERROR: {}'.format(item) for item in report.get('errors', []))
    with open(txt_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    fp32_path = os.path.join(
        os.path.dirname(output),
        'SOMA_CELL_0_6_8_GPU_A3_FP32_DISCREPANCY.json',
    )
    fp32_payload = {
        'build': report.get('build'), 'schema': report.get('schema'),
        'captured_utc': report.get('captured_utc'),
        'fp64_gate_passed_before_fp32': report.get(
            'fp64_gate_passed_before_fp32', False
        ),
        'candidate_only': True,
        'candidate_internal_ledger_tolerance': FP32_CANDIDATE_LEDGER_ATOL,
        'candidate_internal_ledger_tolerance_reason': FP32_CANDIDATE_LEDGER_REASON,
        'source_sha256_map': copy.deepcopy(report.get('source_sha256_map', {})),
        'parent_benchmark': {
            'path': os.path.basename(output),
            'sha256': None,
        },
        'precision_discrepancy': copy.deepcopy(
            report.get('precision_discrepancy', {})
        ),
        'fp32_measurements': [
            copy.deepcopy(item) for item in report.get('measurements', [])
            if item.get('precision') == 'float32'
        ],
    }
    output_files = {
        'json': output, 'csv': csv_path, 'txt': txt_path,
        'fp32_discrepancy_json': fp32_path,
    }
    # The benchmark JSON cannot contain its own digest.  Freeze it first with
    # all sibling paths, then bind the fp32 artifact to that exact byte stream.
    report['output_files'] = copy.deepcopy(output_files)
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')
    fp32_payload['parent_benchmark']['sha256'] = sha256(output)
    with open(fp32_path, 'w', encoding='utf-8') as handle:
        json.dump(fp32_payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')
    return output_files


def main(argv=None):
    args = build_parser().parse_args(argv)
    if any(value <= 0 for value in args.worlds):
        raise SystemExit('--worlds values must be positive')
    if any(value <= 0 for value in args.cell_sweep):
        raise SystemExit('--cell-sweep values must be positive')
    if any(value <= 0 for value in args.capacity_sweep):
        raise SystemExit('--capacity-sweep values must be positive')
    if args.particle_sweep and any(value <= 0 for value in args.particle_sweep):
        raise SystemExit('--particle-sweep values must be positive')
    if args.steps <= 0:
        raise SystemExit('steps must be positive')
    if (
        args.world_warmup < 0 or args.world_repeats <= 0
        or args.other_axis_warmup < 0 or args.other_axis_repeats <= 0
    ):
        raise SystemExit('axis repeats must be positive and axis warmups nonnegative')
    report = run_benchmark(args)
    _write_report_bundle(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if (
        report['all_requested_fp64_gates_pass']
        and report['all_requested_measurements_completed']
        and not report['errors']
    ) else 1


if __name__ == '__main__':
    raise SystemExit(main())
