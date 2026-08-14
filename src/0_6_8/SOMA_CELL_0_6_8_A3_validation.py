# coding: utf-8
"""Strict validation for SOMA-CELL 0.6.8-GPU A3.

The A3 module is imported lazily so this validator can be syntax-checked while
the implementation is being assembled.  Missing A3 APIs are failures, never
skips.  Only a genuinely unavailable CUDA device is reported as NOT_RUN.
"""
from __future__ import division

import copy
import csv
import argparse
import dataclasses
import hashlib
import importlib
import inspect
import json
import math
import os
import pickle
import sys
import tempfile
import time
from collections.abc import Mapping

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

RESULT_TXT = 'SOMA_CELL_0_6_8_GPU_A3_VALIDATION_RESULTS.txt'
RESULT_CSV = 'soma_cell_0_6_8_gpu_a3_validation.csv'
RESULT_JSON = 'SOMA_CELL_0_6_8_GPU_A3_VALIDATION_RESULTS.json'

PURE_FP64_ATOL = 2e-12
WORLD_FP64_ATOL = 5e-12
LEDGER_ATOL = 5e-10

SOURCE_SHA256_PATHS = (
    'src/0_6_8/SOMA_CELL_0_6_8_gpu_a3.py',
    'src/0_6_8/SOMA_CELL_0_6_8_gpu_a3_scheduler.py',
    'src/0_6_8/SOMA_CELL_0_6_8_A3_validation.py',
    'src/0_6_8/SOMA_CELL_0_6_8_gpu_a2.py',
    'src/0_6_6/SOMA_CELL_0_6_6_pythonista.py',
)

CANONICAL_CELL_EVENTS = (
    'surface_exchange', 'gene_refresh', 'generic_reactions',
    'precursor_synthesis', 'maintenance', 'translation_cpu',
    'replication_cpu', 'surface_assembly', 'protein_damage', 'aggregation',
    'reactive_byproduct', 'membrane_oxidation', 'genome_lesion_gain',
    'genome_hydrolysis_cpu_rng', 'transporter_smoothing',
    'membrane_smoothing', 'ordinary_decay', 'waste_export', 'leak', 'radius',
    'motion', 'division_update_cpu', 'base_viability', 'cell_age',
    'repair_antioxidant', 'repair_chaperone', 'repair_protease',
    'repair_genome', 'repair_membrane', 'damage_viability',
    'sensorimotor_learning_cpu', 'mobile_export_cpu',
    'formal066_supplemental', 'segregation_plan', 'actual_split_cpu',
    'death_release_cpu',
)


class ValidationNotRun(RuntimeError):
    """A hardware-specific check could not be run on this machine."""


class RecursiveMismatch(AssertionError):
    pass


_A3 = None
_MEASURED_MAXIMA = {}


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def source_sha256_map():
    """Bind evidence to repo-relative authoritative source identities."""
    root = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
    result = {}
    for relative in SOURCE_SHA256_PATHS:
        path = os.path.join(root, *relative.split('/'))
        if not os.path.isfile(path):
            # The deterministic release is flat, but the evidence keys remain
            # repo-relative so the builder can compare the same identity set.
            path = os.path.join(HERE, os.path.basename(relative))
        if not os.path.isfile(path):
            raise AssertionError('source hash input is missing: {}'.format(relative))
        result[relative] = _sha256_file(path)
    return result


def a3_module():
    global _A3
    if _A3 is None:
        try:
            _A3 = importlib.import_module('SOMA_CELL_0_6_8_gpu_a3')
        except Exception as exc:
            raise AssertionError('cannot import A3 implementation: {!r}'.format(exc))
    return _A3


def require_api(*names):
    module = a3_module()
    scheduler_module = None
    values = []
    missing = []
    for name in names:
        if hasattr(module, name):
            values.append(getattr(module, name))
            continue
        try:
            if scheduler_module is None:
                scheduler_module = importlib.import_module('SOMA_CELL_0_6_8_gpu_a3_scheduler')
            values.append(getattr(scheduler_module, name))
        except (ImportError, AttributeError):
            missing.append(name)
    if missing:
        raise AssertionError('A3 required API missing: {}'.format(', '.join(missing)))
    return values


def _torch():
    try:
        import torch
    except Exception as exc:
        raise AssertionError('PyTorch unavailable: {!r}'.format(exc))
    return torch


def require_cuda_available(torch_module=None):
    """Fail closed for an explicit CUDA request; never select CPU instead."""
    torch_module = _torch() if torch_module is None else torch_module
    if not bool(torch_module.cuda.is_available()):
        raise ValidationNotRun('CUDA explicitly required but unavailable; CPU fallback forbidden')
    return 'cuda'


def _path(parent, child):
    if isinstance(child, int):
        return '{}[{}]'.format(parent, child)
    return '{}.{}'.format(parent, child) if parent else str(child)


def _normalise_leaf(value):
    torch = sys.modules.get('torch')
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.generic):
        return value.item()
    return value


def assert_recursive_close(expected, actual, atol=3e-11, rtol=3e-11, path='root'):
    """Compare the complete nested state, including types, shapes and keys."""
    expected = _normalise_leaf(expected)
    actual = _normalise_leaf(actual)
    if dataclasses.is_dataclass(expected):
        if not dataclasses.is_dataclass(actual):
            raise RecursiveMismatch('{} dataclass/type mismatch'.format(path))
        expected = {f.name: getattr(expected, f.name) for f in dataclasses.fields(expected)}
        actual = {f.name: getattr(actual, f.name) for f in dataclasses.fields(actual)}
    if dataclasses.is_dataclass(actual):
        actual = {f.name: getattr(actual, f.name) for f in dataclasses.fields(actual)}
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise RecursiveMismatch('{} mapping/type mismatch'.format(path))
        if set(expected) != set(actual):
            missing = sorted(set(expected) - set(actual), key=repr)
            extra = sorted(set(actual) - set(expected), key=repr)
            raise RecursiveMismatch('{} keys mismatch missing={} extra={}'.format(path, missing, extra))
        for key in expected:
            assert_recursive_close(expected[key], actual[key], atol, rtol, _path(path, repr(key)))
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(expected) != len(actual):
            raise RecursiveMismatch('{} sequence mismatch'.format(path))
        for index, (left, right) in enumerate(zip(expected, actual)):
            assert_recursive_close(left, right, atol, rtol, _path(path, index))
        return
    if isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
        left = np.asarray(expected)
        right = np.asarray(actual)
        if left.shape != right.shape:
            raise RecursiveMismatch('{} shape {} != {}'.format(path, left.shape, right.shape))
        if left.dtype.kind in 'biuOSU' or right.dtype.kind in 'biuOSU':
            if left.dtype.kind != right.dtype.kind or not np.array_equal(left, right):
                raise RecursiveMismatch('{} discrete array mismatch'.format(path))
        else:
            if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
                if not np.array_equal(left, right, equal_nan=True):
                    raise RecursiveMismatch('{} non-finite array mismatch'.format(path))
            elif not np.allclose(left, right, atol=atol, rtol=rtol):
                absolute = np.abs(left.astype(np.float64) - right.astype(np.float64))
                index = np.unravel_index(int(np.argmax(absolute)), absolute.shape)
                diff = float(absolute[index])
                raise RecursiveMismatch(
                    '{} index={} expected={!r} actual={!r} max_abs={}'.format(
                        path, index, left[index].item(), right[index].item(), diff,
                    )
                )
        return
    if isinstance(expected, (bool, int, str, bytes, type(None))):
        if type(expected) is not type(actual) or expected != actual:
            raise RecursiveMismatch('{} exact value/type mismatch {!r} != {!r}'.format(path, expected, actual))
        return
    if isinstance(expected, (float, np.floating)) or isinstance(actual, (float, np.floating)):
        if not math.isclose(float(expected), float(actual), abs_tol=atol, rel_tol=rtol):
            raise RecursiveMismatch('{} float mismatch {!r} != {!r}'.format(path, expected, actual))
        return
    if type(expected) is not type(actual) or expected != actual:
        raise RecursiveMismatch('{} value/type mismatch {!r} != {!r}'.format(path, expected, actual))


def recursive_numeric_differences(expected, actual, path='root', output=None):
    """Return every comparable floating-field error; structural errors are inf."""
    output = [] if output is None else output
    expected = _normalise_leaf(expected)
    actual = _normalise_leaf(actual)
    if dataclasses.is_dataclass(expected):
        expected = {f.name: getattr(expected, f.name) for f in dataclasses.fields(expected)}
    if dataclasses.is_dataclass(actual):
        actual = {f.name: getattr(actual, f.name) for f in dataclasses.fields(actual)}
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        if set(expected) != set(actual):
            output.append((path, float('inf')))
            return output
        for key in expected:
            recursive_numeric_differences(expected[key], actual[key], _path(path, repr(key)), output)
        return output
    if isinstance(expected, (list, tuple)) and isinstance(actual, type(expected)):
        if len(expected) != len(actual):
            output.append((path, float('inf')))
        else:
            for i, pair in enumerate(zip(expected, actual)):
                recursive_numeric_differences(pair[0], pair[1], _path(path, i), output)
        return output
    if isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
        left, right = np.asarray(expected), np.asarray(actual)
        if left.shape != right.shape:
            output.append((path, float('inf')))
        elif left.dtype.kind in 'fc' and right.dtype.kind in 'fc':
            output.append((path, float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))) if left.size else 0.0))
        elif not np.array_equal(left, right):
            output.append((path, float('inf')))
        return output
    if isinstance(expected, (float, np.floating)) and isinstance(actual, (float, np.floating)):
        output.append((path, abs(float(expected) - float(actual))))
    elif isinstance(expected, (bool, int, str, bytes, type(None))) and expected != actual:
        output.append((path, float('inf')))
    return output


def _record_numeric_difference(label, expected, actual):
    differences = recursive_numeric_differences(expected, actual)
    structural = any(not math.isfinite(value) for _, value in differences)
    finite = [value for _, value in differences if math.isfinite(value)]
    maximum = float(max(finite or [0.0]))
    _MEASURED_MAXIMA[str(label)] = {
        'max_abs_diff': maximum,
        'structural_or_discrete_mismatch': structural,
        'numeric_field_count': len(differences),
    }
    return maximum, structural


def pickle_clone(value):
    return pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def make_world(seed=301, cells=2, environment=None):
    module = a3_module()
    s66 = getattr(module, 's66', None)
    if s66 is None:
        s66 = importlib.import_module('SOMA_CELL_0_6_8_gpu_a2').s66
    environment = environment or s66.ENV_STABLE
    config = s66.shared_ecology_config(
        environment=environment, mutation=True, hgt=True,
        eco66_washout=False, eco66_chemostat=False,
    )
    return s66.Formal066World(seed=seed, initial_cells=cells, config=config)


def _config_for_world(world, **changes):
    config = copy.deepcopy(world.config)
    for key, value in changes.items():
        if not hasattr(config, key):
            raise AssertionError('CPU reference config has no {}'.format(key))
        setattr(config, key, value)
    return config


def scenario_cell(name, seed=401):
    module = a3_module()
    world = make_world(seed=seed, cells=1)
    cell = world.cells[0]
    config = copy.deepcopy(world.config)
    p_atp = module.s5.POOL_ATP
    p_waste = module.s5.POOL_WASTE
    p_fuel = module.s5.POOL_FUEL
    p_damaged = getattr(module, 'POOL_DAMAGED_PROTEIN', 10)
    p_aggregate = getattr(module, 'POOL_AGGREGATE', 11)
    p_reactive = getattr(module, 'POOL_REACTIVE', 12)
    if name == 'clean':
        cell.membrane_oxidation[:] = 0.0
        cell.damage_trace[:] = 0.0
        cell.pools[p_reactive] = 0.0
        cell.pools[p_aggregate] = 0.0
        cell.damaged_proteins = {}
    elif name == 'stressed':
        cell.current_stress = 1.8
        cell.damage_trace[:] = np.linspace(0.2, 1.4, len(cell.damage_trace))
        cell.pools[p_waste] += 0.34
        cell.pools[p_reactive] += 0.18
    elif name == 'oxidized':
        cell.membrane_oxidation[:] = np.linspace(0.0, 2.4, len(cell.membrane_oxidation))
        cell.pools[p_reactive] += 0.12
        cell.pools[module.s5.POOL_MEM_PRECURSOR] += 0.2
    elif name == 'aggregate_heavy':
        fingerprint = next(iter(cell.proteins))
        moved = min(float(cell.proteins[fingerprint]) * 0.45, 0.03)
        cell.proteins[fingerprint] -= moved
        cell.damaged_proteins[fingerprint] = cell.damaged_proteins.get(fingerprint, 0.0) + moved
        cell.pools[p_aggregate] += 0.22
        cell.pools[p_reactive] += 0.08
    elif name == 'atp_poor':
        cell.pools[p_atp] = 0.001
        cell.pools[p_fuel] += 0.1
        cell.pools[p_reactive] += 0.1
    elif name == 'repair_disabled':
        config = _config_for_world(
            world, protein_repair=False, genome_repair=False,
            membrane_repair=False, damage_segregation=False,
        )
        cell.pools[p_reactive] += 0.16
        cell.membrane_oxidation[:] = 0.8
    else:
        raise ValueError('unknown scenario {}'.format(name))
    if hasattr(cell, '_sync_protein_pool'):
        cell._sync_protein_pool()
    if hasattr(cell, '_sync_damage_pool'):
        cell._sync_damage_pool()
    return cell, config


def _a3_config(**overrides):
    config_cls, = require_api('GPU068A3Config')
    signature = inspect.signature(config_cls)
    values = {k: v for k, v in overrides.items() if k in signature.parameters}
    return config_cls(**values)


def pack_cell(cell, config=None):
    adapter_cls, = require_api('FullFidelityA3Adapter')
    adapter = adapter_cls(config or _a3_config(device='cpu', precision='float64'))
    return adapter.pack_cell(cell)


def _to_torch_state(state, device='cpu', precision='float64'):
    torch = _torch()
    if not hasattr(state, 'to_torch'):
        raise AssertionError('A3PackedState.to_torch is required')
    signature = inspect.signature(state.to_torch)
    kwargs = {}
    if 'device' in signature.parameters:
        kwargs['device'] = device
    if 'precision' in signature.parameters:
        kwargs['precision'] = precision
    elif 'dtype' in signature.parameters:
        kwargs['dtype'] = torch.float64 if precision == 'float64' else torch.float32
    converted = state.to_torch(**kwargs)
    if converted is None:
        raise AssertionError('A3PackedState.to_torch returned None')
    return converted


def _normalise_kernel_config(packed, model_config):
    """Pure kernels accept packed flags; adapters may also expose a helper."""
    module = a3_module()
    helper = getattr(module, 'apply_model_config_to_a3_state', None)
    if helper is not None:
        return helper(packed, model_config)
    # The packer normally copied the model flags.  Override the known frozen
    # toggles explicitly for scenario-specific configs.
    out = packed.clone() if hasattr(packed, 'clone') else copy.deepcopy(packed)
    for name in (
        'gene_expression', 'generic_reactions', 'membrane_synthesis',
        'protein_repair', 'genome_repair', 'membrane_repair',
        'endogenous_damage', 'damage_segregation',
        'forced_symmetric_damage', 'quiescence', 'damage_rate_scale',
        'repair_cost_scale',
    ):
        if hasattr(out, name) and hasattr(model_config, name):
            setattr(out, name, copy.deepcopy(getattr(model_config, name)))
    if hasattr(out, 'transport_enabled') and hasattr(model_config, 'transport'):
        out.transport_enabled = bool(model_config.transport)
    return out


def _call_pure(fn, state, dt, config):
    return fn(state, dt, config=config)


def _kernel_pair(scenario, device='cpu', precision='float64',
                 atol=PURE_FP64_ATOL, seed=401):
    numpy_fn, torch_fn = require_api('metabolism_damage_numpy', 'metabolism_damage_torch')
    cell, cpu_config = scenario_cell(scenario, seed=seed)
    packed = _normalise_kernel_config(pack_cell(cell), cpu_config)
    numpy_input = copy.deepcopy(packed)
    torch_input = _to_torch_state(copy.deepcopy(packed), device=device, precision=precision)
    numpy_before = copy.deepcopy(numpy_input)
    torch_before = copy.deepcopy(torch_input)
    numpy_out = _call_pure(numpy_fn, numpy_input, 0.1, cpu_config)
    torch_out = _call_pure(torch_fn, torch_input, 0.1, cpu_config)
    if numpy_out is None or torch_out is None:
        raise AssertionError('pure kernel returned None')
    assert_recursive_close(numpy_before, numpy_input, atol=0.0, rtol=0.0, path='numpy_input')
    assert_recursive_close(torch_before, torch_input, atol=0.0, rtol=0.0, path='torch_input')
    torch_numpy = _to_numpy_tree(torch_out)
    _record_numeric_difference(
        'pure.{}.{}.{}'.format(scenario, device, precision), numpy_out, torch_numpy,
    )
    assert_recursive_close(
        numpy_out, torch_numpy, atol=atol, rtol=0.0, path='kernel_output',
    )
    return numpy_out, torch_out


def _assert_finite_nonnegative_state(value):
    value = _normalise_leaf(value)
    if dataclasses.is_dataclass(value):
        value = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite_nonnegative_state(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_finite_nonnegative_state(child)
    elif isinstance(value, np.ndarray) and value.dtype.kind in 'fc':
        if not np.all(np.isfinite(value)):
            raise AssertionError('non-finite kernel output')


def _to_numpy_tree(value):
    """Convert Torch leaves without weakening structural comparison."""
    value = _normalise_leaf(value)
    if dataclasses.is_dataclass(value):
        clone = copy.deepcopy(value)
        for item in dataclasses.fields(clone):
            setattr(clone, item.name, _to_numpy_tree(getattr(clone, item.name)))
        return clone
    if isinstance(value, Mapping):
        return {key: _to_numpy_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_to_numpy_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_to_numpy_tree(child) for child in value)
    return value


def _extract_packed_state(value):
    if dataclasses.is_dataclass(value) and hasattr(value, 'pools'):
        return value
    if isinstance(value, Mapping) and 'state' in value:
        return _extract_packed_state(value['state'])
    return None


def _assert_physical_state_valid(value):
    packed = _extract_packed_state(value)
    if packed is None:
        raise AssertionError('kernel output does not expose an A3 packed state')
    for name in (
        'pools', 'membrane', 'membrane_oxidation', 'transporters',
        'active_mass', 'damaged_mass', 'aggregate_mass', 'genome_lesions',
    ):
        array = np.asarray(_normalise_leaf(getattr(packed, name)), dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise AssertionError('{} contains non-finite values'.format(name))
        if array.size and float(np.min(array)) < -2e-10:
            raise AssertionError('{} contains negative material/state'.format(name))
    validate_fn, = require_api('validate_a3_state')
    validate_fn(packed)


def _hybrid_from_state(state, device='cpu', precision='float64',
                       config_overrides=None):
    hybrid_cls, = require_api('Hybrid066WorldA3')
    module = a3_module()
    raw = module.s66.Formal066World.from_state(pickle_clone(state))
    config_values = dict(config_overrides or {})
    config_values.update({'device': device, 'precision': precision})
    config = _a3_config(**config_values)
    signature = inspect.signature(hybrid_cls)
    if 'backend' in signature.parameters:
        backend_cls, = require_api('TorchKernelBackendA3')
        return hybrid_cls(raw, backend_cls(config))
    return hybrid_cls(raw, config)


def _assert_world_pair(cpu, hybrid, state_atol=WORLD_FP64_ATOL,
                       ledger_atol=LEDGER_ATOL, label='world'):
    expected = cpu.state_dict()
    actual = hybrid.world.state_dict()
    _record_numeric_difference(label, expected, actual)
    assert_recursive_close(
        expected, actual, atol=state_atol, rtol=0.0, path='world',
    )
    if cpu.rng.bit_generator.state != hybrid.world.rng.bit_generator.state:
        raise AssertionError('world RNG state differs')
    left = float(cpu.matter_ledger_residual())
    right = float(hybrid.world.matter_ledger_residual())
    difference = abs(left - right)
    _MEASURED_MAXIMA[str(label) + '.material_ledger'] = {
        'max_abs_diff': difference,
        'cpu_residual': left,
        'hybrid_residual': right,
        'tolerance': float(ledger_atol),
    }
    if difference > float(ledger_atol):
        raise AssertionError(
            'material ledger differs: cpu={!r} hybrid={!r} abs_diff={!r}'.format(
                left, right, difference,
            )
        )


def _run_world_lockstep(steps, seed=501, mutate=None, environment=None, precision='float64', device='cpu'):
    base = make_world(seed=seed, cells=2, environment=environment)
    if mutate:
        mutate(base)
    state = pickle_clone(base.state_dict())
    cpu = a3_module().s66.Formal066World.from_state(pickle_clone(state))
    hybrid = _hybrid_from_state(state, device=device, precision=precision)
    receipts = []
    for _ in range(steps):
        cpu.step(0.1)
        receipts.append(hybrid.step(0.1))
    _assert_world_pair(
        cpu, hybrid, state_atol=WORLD_FP64_ATOL, ledger_atol=LEDGER_ATOL,
        label='world.steps{}.seed{}.{}.{}'.format(steps, seed, device, precision),
    )
    return cpu, hybrid, receipts


def _event_order():
    module = a3_module()
    for name in ('A3_CELL_EVENT_ORDER', 'CELL_EVENT_ORDER', 'A3_STAGE_ORDER'):
        value = getattr(module, name, None)
        if value is not None:
            if isinstance(value, Mapping):
                value = value.get('cell') or value.get('cell_events')
            if value is not None:
                return tuple(value)
    try:
        scheduler_module = importlib.import_module('SOMA_CELL_0_6_8_gpu_a3_scheduler')
        return tuple(scheduler_module.CELL_EVENT_ORDER)
    except Exception:
        pass
    raise AssertionError('A3 canonical event-order constant is missing')


def _scheduler_fixture():
    scheduler_cls, = require_api('A3EventScheduler')
    world = make_world(seed=601, cells=1)
    scheduler = scheduler_cls()
    scheduler.begin_step(world, world.cells)
    return scheduler, world, world.cells[0]


def _expected_exception_names(kind):
    module = a3_module()
    names = {
        'duplicate': ('A3DuplicateEventError', 'DuplicateStageError'),
        'order': ('A3EventOrderError', 'StageOrderError'),
    }[kind]
    values = []
    scheduler_module = importlib.import_module('SOMA_CELL_0_6_8_gpu_a3_scheduler')
    for name in names:
        owner = module if hasattr(module, name) else scheduler_module
        if hasattr(owner, name):
            values.append(getattr(owner, name))
    values = tuple(values)
    if not values:
        raise AssertionError('{} scheduler exception API missing'.format(kind))
    return values


def test_api_build_schema_scope():
    module = a3_module()
    require_api(
        'GPU068A3Config', 'A3PackedState', 'FullFidelityA3Adapter',
        'metabolism_damage_numpy', 'metabolism_damage_torch',
        'segregation_plan_numpy', 'segregation_plan_torch',
        'A3EventScheduler', 'TorchKernelBackendA3', 'Hybrid066WorldA3',
    )
    assert module.BUILD == 'SOMA-CELL 0.6.8-GPU A3'
    assert module.SCHEMA_VERSION == '0.6.8-GPU-A3.0'
    assert getattr(module, 'FULL_GPU_WORLD_STEP', False) is False
    report = module.environment_report()
    assert report['full_gpu_world_step'] is False
    return 'A3 public API present; full GPU claim remains false'


def test_recursive_comparator_covers_nested_discrete_and_float_state():
    left = {'x': np.asarray([1.0, 2.0]), 'nested': [{'flag': True, 'id': 8}], 'raw': b'ab'}
    assert_recursive_close(left, copy.deepcopy(left), atol=0.0, rtol=0.0)
    bad = copy.deepcopy(left); bad['nested'][0]['id'] = 9
    try:
        assert_recursive_close(left, bad)
    except RecursiveMismatch:
        return 'recursive comparator catches nested discrete mismatch'
    raise AssertionError('recursive comparator missed nested mismatch')


def test_packed_state_roundtrip_and_invariants():
    adapter_cls, validate_fn = require_api('FullFidelityA3Adapter', 'validate_a3_state')
    cell, _ = scenario_cell('aggregate_heavy')
    before = pickle_clone(cell.state_dict())
    adapter = adapter_cls(_a3_config(device='cpu', precision='float64'))
    packed = adapter.pack_cell(cell)
    validate_fn(packed)
    target = copy.deepcopy(cell)
    adapter.unpack_cell(copy.deepcopy(packed), target)
    assert_recursive_close(
        before, target.state_dict(), atol=PURE_FP64_ATOL, rtol=0.0,
        path='packed_roundtrip',
    )
    return 'packed active/damaged/aggregate/lesion state round-trips losslessly'


def _expect_schema_rejected(state, label, config=None):
    module = a3_module()
    try:
        module.validate_a3_state(state, config=config)
    except (module.A3SchemaError, module.A3CapacityError):
        return
    raise AssertionError('{} malformed state was accepted'.format(label))


def _mutated_rejection(base, label, mutate, config=None):
    state = base.clone()
    mutate(state)
    _expect_schema_rejected(state, label, config=config)


def test_packed_schema_rejects_mask_order_and_mass_corruption():
    module = a3_module()
    cell, _ = scenario_cell('aggregate_heavy', seed=406)
    base = pack_cell(cell)
    if base.active_count < 2 or base.species_count >= base.capacity:
        raise AssertionError('schema corruption fixture lacks required active/tail slots')

    _mutated_rejection(
        base, 'species mask prefix',
        lambda state: state.species_mask.__setitem__(0, False),
    )
    _mutated_rejection(
        base, 'unused fingerprint sentinel',
        lambda state: state.species_fingerprint.__setitem__(state.species_count, 19),
    )
    _mutated_rejection(
        base, 'unused active order sentinel',
        lambda state: state.active_order.__setitem__(state.active_count, 0),
    )
    _mutated_rejection(
        base, 'duplicate active order',
        lambda state: state.active_order.__setitem__(1, state.active_order[0]),
    )

    def hide_last_active(state):
        state.active_count -= 1
        state.active_order[state.active_count] = -1
    _mutated_rejection(base, 'positive mass omitted from order', hide_last_active)

    def zero_used_active(state):
        slot = int(state.active_order[0])
        state.active_mass[slot] = module.ACTIVE_MASS_THRESHOLD
    _mutated_rejection(base, 'active mass at unpack threshold', zero_used_active)

    def hide_mass_outside_union(state):
        state.active_mass[state.species_count] = 0.001
    _mutated_rejection(base, 'mass outside species_count', hide_mass_outside_union)

    def append_orphan_species(state):
        slot = int(state.species_count)
        state.species_count += 1
        state.species_mask[slot] = True
        state.species_fingerprint[slot] = 700000000000000001
    _mutated_rejection(base, 'orphan species union slot', append_orphan_species)
    return '8 mask/order/tail/mass coverage corruptions fail closed'


def test_packed_schema_rejects_backend_dtype_and_gene_corruption():
    module = a3_module()
    cell, _ = scenario_cell('clean', seed=407)
    base = pack_cell(cell)

    _mutated_rejection(
        base, 'mixed NumPy float dtype',
        lambda state: setattr(state, 'active_mass', state.active_mass.astype(np.float32)),
    )
    _mutated_rejection(
        base, 'unsupported NumPy float16',
        lambda state: setattr(state, 'pools', state.pools.astype(np.float16)),
    )
    _mutated_rejection(
        base, 'list mixed into NumPy arrays',
        lambda state: setattr(state, 'active_mass', state.active_mass.tolist()),
    )
    torch_state = base.to_torch(device='cpu', dtype=_torch().float64)
    _mutated_rejection(
        torch_state, 'NumPy array mixed into Torch state',
        lambda state: setattr(
            state, 'active_mass', state.active_mass.detach().cpu().numpy(),
        ),
    )
    _mutated_rejection(
        torch_state, 'Torch scalar dtype mismatch',
        lambda state: setattr(
            state, 'radius', state.radius.to(dtype=_torch().float32),
        ),
    )
    _mutated_rejection(
        base, 'non-gene tail metadata',
        lambda state: state.gene_promoter.__setitem__(state.species_count, 0.2),
    )
    _mutated_rejection(
        base, 'invalid used gene role',
        lambda state: state.gene_role.__setitem__(int(state.gene_order[0]), -1),
    )
    return '7 backend/device-dtype/gene metadata corruptions fail closed'


def test_pack_rejects_subthreshold_and_invalid_raw_composition():
    module = a3_module()
    adapter = module.FullFidelityA3Adapter(_a3_config())

    active, _ = scenario_cell('clean', seed=408)
    active_key = next(iter(active.proteins))
    active.proteins[active_key] = module.ACTIVE_MASS_THRESHOLD
    active.pools[module.POOL_CATALYST] = sum(active.proteins.values())
    try:
        adapter.pack_cell(active)
    except module.A3SchemaError:
        pass
    else:
        raise AssertionError('active dictionary threshold entry was accepted')

    damaged, _ = scenario_cell('clean', seed=409)
    damaged_key = next(iter(damaged.proteins))
    damaged.damaged_proteins[damaged_key] = module.DAMAGED_MASS_THRESHOLD
    damaged.pools[module.POOL_DAMAGED_PROTEIN] = sum(damaged.damaged_proteins.values())
    try:
        adapter.pack_cell(damaged)
    except module.A3SchemaError:
        pass
    else:
        raise AssertionError('damaged dictionary threshold entry was accepted')

    for label, amount in (('zero', 0.0), ('negative', -0.001)):
        aggregate, _ = scenario_cell('clean', seed=410)
        aggregate_key = next(iter(aggregate.proteins))
        aggregate._soma068a3_aggregate_composition = {aggregate_key: amount}
        aggregate._soma068a3_aggregate_unresolved = 0.0
        aggregate.pools[module.POOL_AGGREGATE] = 0.0
        try:
            adapter.pack_cell(aggregate)
        except module.A3SchemaError:
            continue
        raise AssertionError('{} raw aggregate entry was accepted'.format(label))
    return 'active/damaged thresholds and zero/negative aggregate entries fail closed'


def test_packed_schema_rejects_genome_tail_total_and_replication_corruption():
    cell, _ = scenario_cell('clean', seed=411)
    config = _a3_config(max_genome_copies=4)
    adapter_cls, = require_api('FullFidelityA3Adapter')
    base = adapter_cls(config).pack_cell(cell)
    if base.genome_count >= base.genome_capacity:
        raise AssertionError('genome corruption fixture lacks a tail slot')
    tail = int(base.genome_count)

    _mutated_rejection(
        base, 'genome mask prefix',
        lambda state: state.genome_mask.__setitem__(0, False), config=config,
    )
    _mutated_rejection(
        base, 'genome length tail',
        lambda state: state.genome_lengths.__setitem__(tail, 1), config=config,
    )
    _mutated_rejection(
        base, 'genome lesion tail',
        lambda state: state.genome_lesions.__setitem__(
            int(state.genome_lesion_count), 0.2,
        ), config=config,
    )
    _mutated_rejection(
        base, 'genome hazard tail',
        lambda state: state.genome_hydrolysis_hazard.__setitem__(tail, 0.2),
        config=config,
    )
    _mutated_rejection(
        base, 'total genome symbols',
        lambda state: setattr(
            state, 'total_genome_symbols', state.total_genome_symbols + 1,
        ), config=config,
    )
    _mutated_rejection(
        base, 'non-integer genome count',
        lambda state: setattr(state, 'genome_count', 1.5), config=config,
    )
    _mutated_rejection(
        base, 'inactive replication copy',
        lambda state: setattr(state, 'replication_copy_length', 1), config=config,
    )

    def inconsistent_active_replication(state):
        state.replication_active = True
        state.replication_template_length = 10
        state.replication_copy_length = 11
        state.replication_template_lesion = 0.0
    _mutated_rejection(
        base, 'active replication lengths', inconsistent_active_replication,
        config=config,
    )
    return '8 genome mask/tail/total/count/replication corruptions fail closed'


def test_source_sha256_map_covers_authoritative_artifacts():
    mapping = source_sha256_map()
    if tuple(mapping) != SOURCE_SHA256_PATHS:
        raise AssertionError('source_sha256_map identity/order mismatch')
    for relative, digest in mapping.items():
        if (not isinstance(digest, str) or len(digest) != 64 or
                any(character not in '0123456789abcdef' for character in digest)):
            raise AssertionError('invalid SHA256 for {}'.format(relative))
    return 'source_sha256_map binds core/scheduler/validator/A2/frozen-0.6.6'


def test_capacity_exact_and_plus_one_atomic_failure():
    module = a3_module()
    cell, _ = scenario_cell('aggregate_heavy')
    union = set(cell.proteins) | set(cell.damaged_proteins) | set(cell.gene_specs)
    typed = getattr(cell, 'aggregate_typed_composition', {})
    union |= set(typed)
    exact = max(1, len(union))
    config = _a3_config(
        device='cpu', precision='float64', max_protein_species=exact,
        max_genome_copies=max(3, len(cell.genomes)),
        max_genome_symbols=max(
            int(getattr(module.g2, 'MAX_GENOME_LENGTH', 0)),
            max(len(g) for g in cell.genomes),
        ),
        strict_capacity=True,
    )
    adapter = module.FullFidelityA3Adapter(config)
    adapter.pack_cell(cell)
    overflow = copy.deepcopy(cell)
    key = max(union or {0}) + 1000003
    overflow.proteins[key] = 0.001
    if hasattr(overflow, '_sync_protein_pool'):
        overflow._sync_protein_pool()
    before = pickle_clone(overflow.state_dict())
    expected_error = getattr(module, 'A3CapacityError', ValueError)
    try:
        adapter.pack_cell(overflow)
    except expected_error:
        assert_recursive_close(before, overflow.state_dict(), atol=0.0, rtol=0.0, path='overflow_atomic')
        return 'exact protein capacity passes; capacity+1 fails atomically'
    raise AssertionError('protein capacity+1 was accepted or truncated')


def test_genome_copy_capacity_exact_and_plus_one_atomic_failure():
    module = a3_module()
    cell, _ = scenario_cell('clean', seed=403)
    while len(cell.genomes) < 3:
        cell.genomes.append(cell.genomes[0].copy())
    while len(cell.genome_lesions) < 3:
        cell.genome_lesions.append(0.0)
    config = _a3_config(
        device='cpu', precision='float64', max_genome_copies=3,
        max_genome_symbols=max(
            int(module.g2.MAX_GENOME_LENGTH),
            max(len(genome) for genome in cell.genomes),
        ),
        strict_capacity=True,
    )
    adapter = module.FullFidelityA3Adapter(config)
    exact_before = pickle_clone(cell.state_dict())
    packed = adapter.pack_cell(cell)
    assert packed.genome_count == 3
    assert_recursive_close(
        exact_before, cell.state_dict(), 0.0, 0.0, 'genome_copy_exact_atomic',
    )
    overflow = copy.deepcopy(cell)
    overflow.genomes.append(overflow.genomes[0].copy())
    overflow.genome_lesions.append(0.0)
    before = pickle_clone(overflow.state_dict())
    try:
        adapter.pack_cell(overflow)
    except module.A3CapacityError:
        assert_recursive_close(
            before, overflow.state_dict(), 0.0, 0.0,
            'genome_copy_overflow_atomic',
        )
        return 'exact three genomes pass; genome copy capacity+1 fails atomically'
    raise AssertionError('genome copy capacity+1 was accepted or truncated')


def test_genome_symbol_capacity_exact_and_plus_one_atomic_failure():
    module = a3_module()
    cell, _ = scenario_cell('clean', seed=404)
    limit = int(module.g2.MAX_GENOME_LENGTH)
    source = np.asarray(cell.genomes[0], dtype=np.uint8)
    cell.genomes[0] = np.resize(source, limit).astype(np.uint8, copy=False)
    config = _a3_config(
        device='cpu', precision='float64', max_genome_copies=3,
        max_genome_symbols=limit, strict_capacity=True,
    )
    adapter = module.FullFidelityA3Adapter(config)
    exact_before = pickle_clone(cell.state_dict())
    packed = adapter.pack_cell(cell)
    assert int(packed.genome_lengths[0]) == limit
    assert_recursive_close(
        exact_before, cell.state_dict(), 0.0, 0.0, 'genome_symbol_exact_atomic',
    )
    overflow = copy.deepcopy(cell)
    overflow.genomes[0] = np.concatenate([
        overflow.genomes[0], np.asarray([0], dtype=np.uint8),
    ])
    before = pickle_clone(overflow.state_dict())
    try:
        adapter.pack_cell(overflow)
    except module.A3CapacityError:
        assert_recursive_close(
            before, overflow.state_dict(), 0.0, 0.0,
            'genome_symbol_overflow_atomic',
        )
        return 'exact genome symbol capacity passes; length+1 fails atomically'
    raise AssertionError('genome symbol capacity+1 was accepted or truncated')


def test_transient_genome_lesion_count_preserved_until_damage():
    """Two genomes/one lesion is a real pre-damage frozen CPU state."""
    module = a3_module()
    cell, cpu_config = scenario_cell('clean', seed=405)
    if len(cell.genomes) < 2:
        cell.genomes.append(cell.genomes[0].copy())
    cell.genome_lesions = [0.37]
    packed = pack_cell(cell)
    if not hasattr(packed, 'genome_lesion_count'):
        raise AssertionError('packed schema lacks explicit genome_lesion_count')
    assert packed.genome_count == 2 and packed.genome_lesion_count == 1
    before_mean = float(module._mean_genome_lesion(packed))
    if not math.isclose(before_mean, 0.37, abs_tol=0.0, rel_tol=0.0):
        raise AssertionError('pre-damage lesion mean was implicitly zero-extended')
    target = copy.deepcopy(cell)
    module.FullFidelityA3Adapter(_a3_config()).unpack_cell(packed.clone(), target)
    assert target.genome_lesions == [0.37]
    damaged = module.damage_numpy(packed, 0.1, config=cpu_config)
    if damaged.genome_lesion_count != damaged.genome_count:
        raise AssertionError('damage stage did not canonically extend lesions')
    values = np.asarray(damaged.genome_lesions, dtype=np.float64)
    if values[1] <= 0.0:
        raise AssertionError('new lesion slot did not receive canonical damage gain')
    return 'transient lesion count preserved; damage stage alone zero-extends then gains'


def _kernel_test(name):
    out, _ = _kernel_pair(name)
    _assert_finite_nonnegative_state(out)
    _assert_physical_state_valid(out)
    return '{} NumPy/Torch fp64 parity and pure-input contract'.format(name)


def test_kernel_clean_fp64(): return _kernel_test('clean')
def test_kernel_stressed_fp64(): return _kernel_test('stressed')
def test_kernel_oxidized_fp64(): return _kernel_test('oxidized')
def test_kernel_aggregate_heavy_fp64(): return _kernel_test('aggregate_heavy')
def test_kernel_atp_poor_fp64(): return _kernel_test('atp_poor')
def test_kernel_repair_disabled_fp64(): return _kernel_test('repair_disabled')


def test_segregation_plan_numpy_torch_fp64():
    numpy_fn, torch_fn = require_api('segregation_plan_numpy', 'segregation_plan_torch')
    cell, cpu_config = scenario_cell('aggregate_heavy', seed=402)
    packed = pack_cell(cell)
    n_input = copy.deepcopy(packed); t_input = _to_torch_state(copy.deepcopy(packed))
    n_before = copy.deepcopy(n_input); t_before = copy.deepcopy(t_input)
    n_out = numpy_fn(n_input, config=cpu_config)
    t_out = torch_fn(t_input, config=cpu_config)
    assert_recursive_close(n_before, n_input, 0.0, 0.0, 'seg_numpy_input')
    assert_recursive_close(t_before, t_input, 0.0, 0.0, 'seg_torch_input')
    torch_numpy = _to_numpy_tree(t_out)
    _record_numeric_difference('pure.segregation.cpu.float64', n_out, torch_numpy)
    assert_recursive_close(
        n_out, torch_numpy, PURE_FP64_ATOL, 0.0, 'segregation',
    )
    return 'damage segregation plan fp64 parity and non-mutation'


def test_all_public_stage_pairs_fp64():
    module = a3_module()
    state = module.make_synthetic_a3_state()
    paired = (
        'generic_metabolism', 'precursor_synthesis', 'maintenance',
        'translation', 'surface_assembly', 'damage', 'repair',
        'housekeeping', 'supplemental_066',
    )
    for name in paired:
        numpy_fn, torch_fn = require_api(name + '_numpy', name + '_torch')
        numpy_input = state.clone()
        torch_input = state.to_torch(device='cpu', dtype=_torch().float64)
        n_before = numpy_input.clone(); t_before = torch_input.clone()
        n_out = numpy_fn(numpy_input, 0.1)
        t_out = torch_fn(torch_input, 0.1)
        assert_recursive_close(n_before, numpy_input, 0.0, 0.0, name + '_numpy_input')
        assert_recursive_close(t_before, torch_input, 0.0, 0.0, name + '_torch_input')
        _record_numeric_difference('pure.stage.{}.cpu.float64'.format(name), n_out, t_out)
        assert_recursive_close(n_out, t_out, PURE_FP64_ATOL, 0.0, name)
    return 'all public A3 stage pairs preserve inputs and agree in fp64'


def test_scheduler_duplicate_rejected_before_second_claim():
    scheduler, world, cell = _scheduler_fixture()
    event = _event_order()[0]
    scheduler.claim(cell, event, status='skipped', metadata={'test': True})
    try:
        scheduler.claim(cell, event, status='skipped')
    except _expected_exception_names('duplicate'):
        return 'duplicate scheduler claim rejected'
    raise AssertionError('duplicate scheduler claim accepted')


def test_scheduler_out_of_order_rejected():
    scheduler, world, cell = _scheduler_fixture()
    order = _event_order()
    if tuple(order[:len(CANONICAL_CELL_EVENTS)]) != CANONICAL_CELL_EVENTS:
        raise AssertionError('exported scheduler order differs from frozen A3 contract')
    try:
        scheduler.claim(cell, order[5], status='skipped')
    except _expected_exception_names('order'):
        return 'out-of-order scheduler claim rejected'
    raise AssertionError('out-of-order scheduler claim accepted')


def test_scheduler_omitted_receipts_exactly_once():
    scheduler, world, cell = _scheduler_fixture()
    order = _event_order()
    for event in order:
        scheduler.claim(cell, event, status='skipped', metadata={'reason': 'validation-omitted'})
    scheduler.finish_cell(cell)
    scheduler_module = importlib.import_module('SOMA_CELL_0_6_8_gpu_a3_scheduler')
    for event in scheduler_module.WORLD_EVENT_ORDER:
        scheduler.claim_world(event, status='skipped', metadata={'reason': 'validation-omitted'})
    receipt = scheduler.finish_step(world)
    events = receipt['cells'][0]['events']
    names = [entry['event'] for entry in events]
    assert names == list(order)
    assert len(names) == len(set(names))
    assert all(entry['status'] == 'skipped' for entry in events)
    return 'operations deliberately omitted by the fixture are receipted once as skipped'


def test_scheduler_claim_failure_precedes_generic_phase_mutation():
    """A duplicate injected at the generic claim must leave its plan uncommitted."""
    scheduler_module = importlib.import_module('SOMA_CELL_0_6_8_gpu_a3_scheduler')

    class DuplicateAtGeneric(scheduler_module.A3EventScheduler):
        def __init__(self, world):
            super(DuplicateAtGeneric, self).__init__()
            self.world_under_test = world
            self.injected = False
            self.cell_at_failed_claim = None
            self.world_at_failed_claim = None
            self.ledger_at_failed_claim = None

        def claim(self, cell, event, status='executed', metadata=None):
            if event == 'generic_reactions' and not self.injected:
                self.injected = True
                # The first call establishes a real scheduler claim.  The
                # second call raises the scheduler's own duplicate exception;
                # neither call has permission to commit the computed plan.
                super(DuplicateAtGeneric, self).claim(
                    cell, event, status=status, metadata=metadata,
                )
                self.cell_at_failed_claim = pickle_clone(cell.state_dict())
                self.world_at_failed_claim = pickle_clone(self.world_under_test.state_dict())
                self.ledger_at_failed_claim = self.world_under_test.matter_ledger_residual()
                return super(DuplicateAtGeneric, self).claim(
                    cell, event, status=status, metadata=metadata,
                )
            return super(DuplicateAtGeneric, self).claim(
                cell, event, status=status, metadata=metadata,
            )

    module = a3_module()
    world = make_world(seed=607, cells=1)
    cell = world.cells[0]
    backend_cls, = require_api('TorchKernelBackendA3')
    backend = backend_cls(_a3_config(device='cpu', precision='float64'))
    scheduler = DuplicateAtGeneric(world)
    scheduler.begin_step(world, [cell])
    for event in scheduler_module.WORLD_EVENT_ORDER[:5]:
        scheduler.claim_world(event, status='skipped', metadata={'validation': True})
    scheduler.claim(cell, 'surface_exchange', status='skipped', metadata={'validation': True})
    scheduler.reserve_metabolism_dispatch(world, cell, 0.1)
    caught = None
    try:
        backend.metabolism_inplace(world, cell, 0.1, scheduler)
    except Exception as exc:
        caught = exc
    if not scheduler.injected:
        raise AssertionError('generic duplicate injection was not reached')
    if caught is None:
        raise AssertionError('injected duplicate did not fail metabolism')
    duplicate_type = scheduler_module.A3DuplicateEventError
    if not isinstance(caught, duplicate_type):
        raise AssertionError('unexpected injected-claim exception {!r}'.format(caught))
    assert_recursive_close(
        scheduler.cell_at_failed_claim, cell.state_dict(), 0.0, 0.0,
        'claim_failure_cell',
    )
    assert_recursive_close(
        scheduler.world_at_failed_claim, world.state_dict(), 0.0, 0.0,
        'claim_failure_world',
    )
    if scheduler.ledger_at_failed_claim != world.matter_ledger_residual():
        raise AssertionError('world material ledger changed after failed claim')
    scheduler.abort_step(caught)
    return 'real duplicate claim fails before generic phase commits cell/world material'


def test_world_one_step_lockstep():
    _run_world_lockstep(1, seed=701)
    return 'frozen CPU vs A3 hybrid one-step full-state/RNG/ledger lockstep'


def test_world_ten_step_lockstep():
    _run_world_lockstep(10, seed=702)
    return 'frozen CPU vs A3 hybrid ten-step full-state/RNG/ledger lockstep'


def test_world_stressed_repair_heavy_lockstep():
    def mutate(world):
        module = a3_module()
        for cell in world.cells:
            cell.current_stress = 1.5
            cell.damage_trace[:] = 0.9
            cell.membrane_oxidation[:] = np.linspace(0.3, 1.8, len(cell.membrane_oxidation))
            cell.pools[module.POOL_REACTIVE] += 0.16
            cell.pools[module.POOL_ATP] += 0.4
            if hasattr(cell, '_sync_damage_pool'): cell._sync_damage_pool()
    _run_world_lockstep(3, seed=703, mutate=mutate)
    return 'stressed repair-heavy world lockstep'


def test_world_atp_poor_maintenance_shortfall_lockstep():
    def mutate(world):
        module = a3_module()
        world.field.amount[:] = 0.0
        for cell in world.cells:
            cell.pools[module.POOL_ATP] = 0.0
            cell.pools[module.POOL_FUEL] = 0.0
            cell.pools[module.POOL_ALT] = 0.0
            cell.pools[module.POOL_INTERMEDIATE] = 0.0
    _, hybrid, _ = _run_world_lockstep(1, seed=708, mutate=mutate)
    shortfalls = [float(getattr(cell, 'maintenance_shortfall', 0.0))
                  for cell in hybrid.world.cells]
    if not shortfalls or max(shortfalls) <= 0.0:
        raise AssertionError('ATP-poor integrated path did not retain maintenance_shortfall')
    return 'ATP-poor integrated housekeeping observes retained maintenance shortfall'


def test_world_transport_disabled_surface_invocation_lockstep():
    def mutate(world):
        world.config.transport = False
        if len(world.field.amount) == 0:
            raise AssertionError('transport-off fixture requires canonical particles')
        cell = world.cells[0]
        world.field.pos[0] = np.asarray(cell.pos, dtype=float)
        world.field.kind[0] = a3_module().s5.PARTICLE_FUEL
        world.field.amount[0] = max(float(world.field.amount[0]), 0.08)
        cell.membrane[0] *= 0.02
    _, hybrid, receipts = _run_world_lockstep(1, seed=709, mutate=mutate)
    receipt = receipts[-1]
    if not isinstance(receipt, Mapping):
        raise AssertionError('transport-off integrated step returned no receipt')
    for cell_record in receipt.get('cells', ()): 
        matches = [entry for entry in cell_record.get('events', ())
                   if entry.get('event') == 'surface_exchange']
        if len(matches) != 1:
            raise AssertionError('transport-off surface receipt is not exact-once')
        entry = matches[0]
        if entry.get('status') != 'executed':
            raise AssertionError('transport-off canonical surface invocation was marked skipped')
        metadata = entry.get('metadata', {})
        if metadata.get('enabled') is not False or metadata.get('particle_count', 0) <= 0:
            raise AssertionError('transport-off surface metadata is not truthful')
    if bool(hybrid.world.config.transport):
        raise AssertionError('transport-off config was not preserved')
    return 'transport=False still invokes passive A2 surface operation and locksteps'


def test_world_predivision_lockstep():
    def mutate(world):
        cell = world.cells[0]
        if len(cell.genomes) < 2:
            cell.genomes.append(cell.genomes[0].copy())
            cell.genome_lesions.append(float(cell.genome_lesions[0] if cell.genome_lesions else 0.0))
        cell.septum_mass = min(0.119, max(float(getattr(cell, 'septum_mass', 0.0)), 0.115))
        cell.division_progress = cell.septum_mass / 0.12
        cell.pools[a3_module().POOL_ATP] = max(cell.pools[a3_module().POOL_ATP], 0.6)
        cell.pools[a3_module().POOL_MEM_PRECURSOR] = max(
            cell.pools[a3_module().POOL_MEM_PRECURSOR], 0.3,
        )
    _run_world_lockstep(1, seed=704, mutate=mutate)
    return 'pre-division detailed state lockstep'


def test_world_periodic_reversal_lockstep():
    env = a3_module().s66.ENV_PERIODIC
    _run_world_lockstep(10, seed=705, environment=env)
    return 'periodic/reversal environment lockstep'


def test_world_receipt_has_no_duplicate_a2_or_a3_events():
    cpu, hybrid, receipts = _run_world_lockstep(1, seed=706)
    receipt = receipts[-1] or getattr(hybrid.scheduler, 'last_receipt', None)
    if not isinstance(receipt, Mapping):
        raise AssertionError('Hybrid066WorldA3.step must return an event receipt')
    cells = receipt.get('cells', [])
    if not cells:
        raise AssertionError('integrated receipt contains no cell records')
    for cell in cells:
        names = [entry['event'] for entry in cell.get('events', [])]
        if len(names) != len(set(names)):
            raise AssertionError('duplicate event in integrated receipt')
        for required in ('surface_exchange', 'waste_export', 'leak', 'radius', 'motion'):
            if names.count(required) != 1:
                raise AssertionError('{} count is {}'.format(required, names.count(required)))
    return 'integrated receipt proves A2/A3 stages exactly once'


def test_clone_deterministic_full_state():
    hybrid = _hybrid_from_state(make_world(seed=801, cells=2).state_dict())
    for _ in range(3): hybrid.step(0.1)
    twin = hybrid.clone()
    for _ in range(5): hybrid.step(0.1); twin.step(0.1)
    _assert_world_pair(
        hybrid.world, twin, state_atol=0.0, ledger_atol=0.0,
        label='clone.bit_exact',
    )
    return 'A3 clone preserves complete state and RNG'


def test_save_restore_deterministic_full_state():
    hybrid = _hybrid_from_state(make_world(seed=802, cells=2).state_dict())
    for _ in range(3): hybrid.step(0.1)
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, 'a3.pkl')
        hybrid.save(path)
        restored = type(hybrid).load(path)
        for _ in range(5): hybrid.step(0.1); restored.step(0.1)
    _assert_world_pair(
        hybrid.world, restored, state_atol=0.0, ledger_atol=0.0,
        label='save_restore.bit_exact',
    )
    return 'A3 atomic save/restore preserves complete continuation'


def test_cuda_fp64_gate_then_fp32_discrepancy():
    torch = _torch()
    require_cuda_available(torch)
    seed = 901
    n64, t64 = _kernel_pair(
        'stressed', device='cuda', precision='float64',
        atol=PURE_FP64_ATOL, seed=seed,
    )
    torch.cuda.synchronize()
    cell, cpu_config = scenario_cell('stressed', seed=seed)
    packed = _normalise_kernel_config(pack_cell(cell), cpu_config)
    fp32 = _to_torch_state(copy.deepcopy(packed), device='cuda', precision='float32')
    torch_fn, = require_api('metabolism_damage_torch')
    out32 = torch_fn(fp32, 0.1, config=cpu_config)
    differences = recursive_numeric_differences(n64, out32)
    finite = [value for _, value in differences if math.isfinite(value)]
    if not differences or not finite or any(not math.isfinite(value) for _, value in differences):
        raise AssertionError('fp32 discrepancy is missing, structural, or non-finite')
    maximum = max(finite or [0.0])
    _MEASURED_MAXIMA['pure.stressed.cuda.float32_candidate_vs_numpy_fp64'] = {
        'max_abs_diff': float(maximum),
        'structural_or_discrete_mismatch': False,
        'candidate_only': True,
    }
    return 'CUDA fp64 parity passed before fp32; fp32 max discrepancy {:.6g}'.format(maximum)


def test_explicit_cuda_request_fails_closed_without_fallback():
    class UnavailableCUDA(object):
        @staticmethod
        def is_available(): return False

    class TorchStub(object):
        cuda = UnavailableCUDA()

    try:
        require_cuda_available(TorchStub())
    except ValidationNotRun as exc:
        if 'fallback forbidden' not in str(exc):
            raise AssertionError('CUDA fail-closed reason is ambiguous')
        return 'explicit unavailable CUDA request fails without CPU fallback'
    raise AssertionError('explicit unavailable CUDA request silently fell back')


TESTS = [
    test_api_build_schema_scope,
    test_source_sha256_map_covers_authoritative_artifacts,
    test_recursive_comparator_covers_nested_discrete_and_float_state,
    test_packed_state_roundtrip_and_invariants,
    test_packed_schema_rejects_mask_order_and_mass_corruption,
    test_packed_schema_rejects_backend_dtype_and_gene_corruption,
    test_pack_rejects_subthreshold_and_invalid_raw_composition,
    test_packed_schema_rejects_genome_tail_total_and_replication_corruption,
    test_capacity_exact_and_plus_one_atomic_failure,
    test_genome_copy_capacity_exact_and_plus_one_atomic_failure,
    test_genome_symbol_capacity_exact_and_plus_one_atomic_failure,
    test_transient_genome_lesion_count_preserved_until_damage,
    test_kernel_clean_fp64,
    test_kernel_stressed_fp64,
    test_kernel_oxidized_fp64,
    test_kernel_aggregate_heavy_fp64,
    test_kernel_atp_poor_fp64,
    test_kernel_repair_disabled_fp64,
    test_segregation_plan_numpy_torch_fp64,
    test_all_public_stage_pairs_fp64,
    test_scheduler_duplicate_rejected_before_second_claim,
    test_scheduler_out_of_order_rejected,
    test_scheduler_omitted_receipts_exactly_once,
    test_scheduler_claim_failure_precedes_generic_phase_mutation,
    test_world_one_step_lockstep,
    test_world_ten_step_lockstep,
    test_world_stressed_repair_heavy_lockstep,
    test_world_atp_poor_maintenance_shortfall_lockstep,
    test_world_transport_disabled_surface_invocation_lockstep,
    test_world_predivision_lockstep,
    test_world_periodic_reversal_lockstep,
    test_world_receipt_has_no_duplicate_a2_or_a3_events,
    test_clone_deterministic_full_state,
    test_save_restore_deterministic_full_state,
    test_cuda_fp64_gate_then_fp32_discrepancy,
    test_explicit_cuda_request_fails_closed_without_fallback,
]


def run_all(write=True, output_dir=None):
    _MEASURED_MAXIMA.clear()
    rows = []
    started = time.time()
    for fn in TESTS:
        then = time.perf_counter()
        try:
            detail = fn(); status = 'PASS'; error = ''
        except ValidationNotRun as exc:
            detail = str(exc); status = 'NOT_RUN'; error = ''
        except Exception as exc:
            detail = ''; status = 'FAIL'; error = '{}: {}'.format(type(exc).__name__, exc)
        row = {'test': fn.__name__, 'status': status, 'detail': detail, 'error': error,
               'seconds': time.perf_counter() - then}
        rows.append(row)
        print('{}: {} - {}'.format(status, fn.__name__, detail or error))
    passed = sum(row['status'] == 'PASS' for row in rows)
    failed = sum(row['status'] == 'FAIL' for row in rows)
    not_run = sum(row['status'] == 'NOT_RUN' for row in rows)
    if write:
        output_dir = os.path.abspath(output_dir or HERE)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, RESULT_CSV), 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=('test', 'status', 'detail', 'error', 'seconds'))
            writer.writeheader(); writer.writerows(rows)
        lines = [
            'SOMA-CELL 0.6.8-GPU A3 VALIDATION RESULTS', '=' * 52,
            '{} PASS / {} FAIL / {} NOT_RUN / {} TOTAL'.format(passed, failed, not_run, len(rows)),
            'elapsed {:.3f}s'.format(time.time() - started), '',
        ]
        lines.extend('{}: {} - {}'.format(r['status'], r['test'], r['detail'] or r['error']) for r in rows)
        with open(os.path.join(output_dir, RESULT_TXT), 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(lines) + '\n')
        payload = {
            'build': getattr(a3_module(), 'BUILD', None),
            'schema': getattr(a3_module(), 'SCHEMA_VERSION', None),
            'source_sha256_map': source_sha256_map(),
            'passed': passed, 'failed': failed, 'not_run': not_run,
            'total': len(rows), 'elapsed_seconds': time.time() - started,
            'fixed_tolerances': {
                'pure_fp64_max_abs': PURE_FP64_ATOL,
                'world_fp64_max_abs': WORLD_FP64_ATOL,
                'material_ledger_residual_abs_diff': LEDGER_ATOL,
                'discrete_rng_keys_orders': 'exact',
            },
            'measured_max_abs_differences': copy.deepcopy(_MEASURED_MAXIMA),
            'rows': rows,
        }
        with open(os.path.join(output_dir, RESULT_JSON), 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write('\n')
    if failed:
        raise AssertionError('{} validation failures'.format(failed))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default=HERE)
    parser.add_argument('--no-write', action='store_true')
    parser.add_argument('--require-cuda', action='store_true',
                        help='fail nonzero rather than NOT_RUN when CUDA is unavailable')
    args = parser.parse_args(argv)
    if args.require_cuda:
        try:
            require_cuda_available()
        except ValidationNotRun as exc:
            print('FAIL: explicit CUDA requirement - {}'.format(exc))
            return 1
    try:
        run_all(write=not args.no_write, output_dir=args.output_dir)
    except AssertionError:
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
