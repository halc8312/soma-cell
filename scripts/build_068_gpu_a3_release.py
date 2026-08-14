#!/usr/bin/env python3
"""Build and verify the deterministic SOMA-CELL 0.6.8-GPU A3 release."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src/0_6_8'
OUT = ROOT / 'releases/SOMA_CELL_0_6_8_GPU_A3_CHECKPOINT_20260814.zip'
NAME = 'SOMA_CELL_0_6_8_GPU_A3_RELEASE_20260814'
ZIP_TIMESTAMP = (2026, 8, 14, 0, 0, 0)

VALIDATION_SOURCE_PATHS = (
    'src/0_6_8/SOMA_CELL_0_6_8_gpu_a3.py',
    'src/0_6_8/SOMA_CELL_0_6_8_gpu_a3_scheduler.py',
    'src/0_6_8/SOMA_CELL_0_6_8_A3_validation.py',
    'src/0_6_8/SOMA_CELL_0_6_8_gpu_a2.py',
    'src/0_6_6/SOMA_CELL_0_6_6_pythonista.py',
)
BENCHMARK_SOURCE_PATHS = VALIDATION_SOURCE_PATHS[:3] + (
    'src/0_6_8/SOMA_CELL_0_6_8_A3_benchmark.py',
) + VALIDATION_SOURCE_PATHS[3:]
FORMAL_DEVICES = ('cpu', 'cuda')
FORMAL_PRECISIONS = ('float64', 'float32')
FORMAL_WORLD_COUNTS = (1, 8, 32, 128)
FORMAL_CELL_SWEEP = (1, 3, 8)
FORMAL_PARTICLE_SWEEP = (216, 288, 360)
FORMAL_CAPACITY_SWEEP = (64, 128, 256)
FORMAL_CORRECTNESS_STEPS = (1, 10, 20)
WORLD_FP64_ATOL = 5e-12
LEDGER_ATOL = 5e-10
FP32_CANDIDATE_LEDGER_ATOL = 7.62939453125e-06


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def current_source_sha256_map(relative_paths):
    result = {}
    for relative in relative_paths:
        path = ROOT.joinpath(*str(relative).split('/'))
        if not path.is_file():
            raise RuntimeError('source hash input is missing: {}'.format(path))
        result[str(relative)] = sha(path)
    return result


def verify_source_sha256_map(payload, expected_paths, label):
    observed = payload.get('source_sha256_map')
    if not isinstance(observed, dict):
        raise RuntimeError('{} lacks source_sha256_map'.format(label))
    expected = current_source_sha256_map(expected_paths)
    if set(observed) != set(expected):
        raise RuntimeError(
            '{} source hash paths mismatch: expected={} observed={}'.format(
                label, sorted(expected), sorted(observed),
            )
        )
    mismatches = [
        relative for relative in expected
        if observed.get(relative) != expected[relative]
    ]
    if mismatches:
        raise RuntimeError(
            '{} source hashes do not match current files: {}'.format(
                label, ', '.join(mismatches),
            )
        )
    return expected


def expected_validation_test_ids():
    path = SRC / 'SOMA_CELL_0_6_8_A3_validation.py'
    name = '_soma068a3_release_validation_contract'
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError('cannot load final A3 validation contract')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tests = getattr(module, 'TESTS', None)
    if not isinstance(tests, list) or not tests:
        raise RuntimeError('final A3 validation TESTS is missing/empty')
    identifiers = [getattr(test, '__name__', None) for test in tests]
    if any(not value for value in identifiers) or len(identifiers) != len(set(identifiers)):
        raise RuntimeError('final A3 validation TESTS names are invalid/duplicate')
    return identifiers


def verify_validation_rows(validation):
    expected = expected_validation_test_ids()
    rows = validation.get('rows')
    if not isinstance(rows, list):
        raise RuntimeError('A3 validation evidence lacks rows')
    identifiers = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError('A3 validation row {} is not a mapping'.format(index))
        identifier = row.get('test')
        identifiers.append(identifier)
        if row.get('status') != 'PASS':
            raise RuntimeError(
                'A3 validation row {} is not PASS: {!r}'.format(identifier, row.get('status'))
            )
        if row.get('error') not in ('', None):
            raise RuntimeError('A3 validation PASS row contains an error: {}'.format(identifier))
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError('A3 validation evidence contains duplicate test IDs')
    if identifiers != expected:
        raise RuntimeError(
            'A3 validation rows differ from final TESTS: expected={} observed={}'.format(
                expected, identifiers,
            )
        )
    if (
        validation.get('total') != len(expected)
        or validation.get('passed') != len(expected)
        or validation.get('failed') != 0
        or validation.get('not_run') != 0
    ):
        raise RuntimeError('A3 validation summary is not all-PASS/zero-NOT_RUN')
    return expected


def _finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value == value
        and value not in (float('inf'), float('-inf'))
    )


def _verify_timing_row(item, spec, warmup, repeats):
    identity = (
        item.get('sweep_axis'), item.get('sweep_axis_value'),
        item.get('runner'), item.get('device'), item.get('precision'),
    )
    if item.get('warmup_repeats') != warmup or item.get('timed_repeats') != repeats:
        raise RuntimeError('A3 benchmark timing policy mismatch for {!r}'.format(identity))
    samples = item.get('samples_seconds')
    if (
        not isinstance(samples, list)
        or len(samples) != repeats
        or any(not _finite_number(value) or value <= 0 for value in samples)
    ):
        raise RuntimeError('A3 benchmark samples are incomplete/non-finite for {!r}'.format(identity))
    for name in (
        'minimum_seconds', 'median_seconds', 'maximum_seconds',
        'mean_seconds', 'pstdev_seconds', 'work_items_per_second_at_median',
    ):
        if not _finite_number(item.get(name)) or item[name] < 0:
            raise RuntimeError('A3 benchmark {} is invalid for {!r}'.format(name, identity))
    expected_work = spec['worlds'] * spec['steps']
    if item.get('work_items') != expected_work or expected_work <= 0:
        raise RuntimeError('A3 benchmark work_items mismatch for {!r}'.format(identity))
    final_particles = item.get('particles_final_per_repeat')
    if (
        not isinstance(final_particles, list)
        or len(final_particles) != repeats
        or any(not isinstance(value, int) or value < 0 for value in final_particles)
    ):
        raise RuntimeError('A3 benchmark final particle counts malformed for {!r}'.format(identity))
    actual_particles = item.get('particles_actual_initial_per_world')
    if (
        not isinstance(actual_particles, list)
        or len(actual_particles) != spec['worlds']
        or any(value != spec['particles'] for value in actual_particles)
    ):
        raise RuntimeError('A3 benchmark initial particle counts mismatch for {!r}'.format(identity))
    for name, expected in (
        ('worlds', spec['worlds']),
        ('cells_initial_per_world', spec['cells']),
        ('particles_requested_per_world', spec['particles_requested']),
        ('protein_capacity_requested', spec['capacity']),
        ('steps', spec['steps']),
    ):
        if item.get(name) != expected:
            raise RuntimeError('A3 benchmark {} mismatch for {!r}'.format(name, identity))

    is_hybrid = item.get('runner') == 'A3-hybrid-independent-worlds-sequential-host-loop'
    if is_hybrid:
        capacities = item.get('configured_capacities')
        if not isinstance(capacities, dict):
            raise RuntimeError('A3 benchmark capacities missing for {!r}'.format(identity))
        if capacities.get('max_protein_species') != spec['capacity']:
            raise RuntimeError('A3 benchmark protein capacity mismatch for {!r}'.format(identity))
        if capacities.get('strict_material_ledger') is not True:
            raise RuntimeError('A3 benchmark strict material ledger disabled for {!r}'.format(identity))
        expected_ledger = (
            FP32_CANDIDATE_LEDGER_ATOL
            if item.get('precision') == 'float32'
            else 2e-10
        )
        if capacities.get('ledger_atol') != expected_ledger:
            raise RuntimeError('A3 benchmark internal ledger tolerance mismatch for {!r}'.format(identity))
    if is_hybrid and item.get('device') == 'cuda':
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
                raise RuntimeError(
                    'A3 CUDA benchmark VRAM fields malformed for {!r}'.format(identity)
                )
            cuda_maxima[maximum_name] = maximum
        if cuda_maxima['peak_vram_allocated_bytes_max'] <= 0:
            raise RuntimeError('A3 CUDA benchmark allocated no measured VRAM for {!r}'.format(identity))
        if (
            cuda_maxima['peak_vram_reserved_bytes_max']
            < cuda_maxima['peak_vram_allocated_bytes_max']
        ):
            raise RuntimeError('A3 CUDA reserved VRAM is below allocated for {!r}'.format(identity))


def verify_benchmark_correctness(benchmark):
    correctness = benchmark.get('correctness')
    discrepancy = benchmark.get('precision_discrepancy')
    if not isinstance(correctness, dict) or set(correctness) != set(FORMAL_DEVICES):
        raise RuntimeError('A3 benchmark correctness device set is incomplete')
    if not isinstance(discrepancy, dict) or set(discrepancy) != set(FORMAL_DEVICES):
        raise RuntimeError('A3 benchmark precision discrepancy device set is incomplete')
    for device in FORMAL_DEVICES:
        device_report = correctness[device]
        if not isinstance(device_report, dict) or device_report.get('status') != 'PASS':
            raise RuntimeError('{} fp64 correctness report is not PASS'.format(device))
        probes = device_report.get('probes')
        if not isinstance(probes, list) or len(probes) != len(FORMAL_CORRECTNESS_STEPS):
            raise RuntimeError('{} fp64 correctness probes are incomplete'.format(device))
        if [probe.get('steps') for probe in probes if isinstance(probe, dict)] != list(FORMAL_CORRECTNESS_STEPS):
            raise RuntimeError('{} fp64 correctness steps mismatch'.format(device))
        for probe in probes:
            step = probe.get('steps')
            prefix = '{} fp64 correctness step {}'.format(device, step)
            if probe.get('device') != device or probe.get('precision') != 'float64':
                raise RuntimeError('{} identity mismatch'.format(prefix))
            for name in (
                'execution_succeeded', 'fp64_lockstep_pass',
                'recursive_state_within_fixed_fp64_tolerance', 'rng_equal',
            ):
                if probe.get(name) is not True:
                    raise RuntimeError('{} failed {}'.format(prefix, name))
            for name in ('structural_or_discrete_mismatch', 'nonfinite_numeric_difference'):
                if probe.get(name) is not False:
                    raise RuntimeError('{} failed {}'.format(prefix, name))
            if (
                not _finite_number(probe.get('max_abs_diff'))
                or probe['max_abs_diff'] > WORLD_FP64_ATOL
            ):
                raise RuntimeError('{} exceeds state tolerance'.format(prefix))
            if (
                not _finite_number(probe.get('material_residual_abs_diff'))
                or probe['material_residual_abs_diff'] > LEDGER_ATOL
            ):
                raise RuntimeError('{} exceeds ledger tolerance'.format(prefix))
            for name in ('material_residual_cpu', 'material_residual_hybrid'):
                if not _finite_number(probe.get(name)):
                    raise RuntimeError('{} lacks finite {}'.format(prefix, name))

        precision = discrepancy[device]
        if (
            not isinstance(precision, dict)
            or precision.get('fp64_gate_pass') is not True
            or precision.get('fp32_run') is not True
            or precision.get('fp32_structural_or_discrete_mismatch') is not False
            or precision.get('fp32_candidate_internal_ledger_tolerance')
            != FP32_CANDIDATE_LEDGER_ATOL
        ):
            raise RuntimeError('{} precision discrepancy gate is incomplete'.format(device))
        fp32_probes = precision.get('fp32_world_probes')
        if not isinstance(fp32_probes, list) or len(fp32_probes) != len(FORMAL_CORRECTNESS_STEPS):
            raise RuntimeError('{} fp32 world probes are incomplete'.format(device))
        if [probe.get('steps') for probe in fp32_probes if isinstance(probe, dict)] != list(FORMAL_CORRECTNESS_STEPS):
            raise RuntimeError('{} fp32 world probe steps mismatch'.format(device))
        for probe in fp32_probes:
            step = probe.get('steps')
            prefix = '{} fp32 candidate step {}'.format(device, step)
            if probe.get('device') != device or probe.get('precision') != 'float32':
                raise RuntimeError('{} identity mismatch'.format(prefix))
            if probe.get('execution_succeeded') is not True:
                raise RuntimeError('{} did not execute'.format(prefix))
            if probe.get('candidate_bound_exceeded') is not False:
                raise RuntimeError('{} exceeded candidate bound'.format(prefix))
            if probe.get('rng_equal') is not True:
                raise RuntimeError('{} RNG differs'.format(prefix))
            for name in ('structural_or_discrete_mismatch', 'nonfinite_numeric_difference'):
                if probe.get(name) is not False:
                    raise RuntimeError('{} failed {}'.format(prefix, name))
            if (
                probe.get('candidate_internal_ledger_tolerance')
                != FP32_CANDIDATE_LEDGER_ATOL
                or not _finite_number(probe.get('material_residual_abs_diff'))
                or probe['material_residual_abs_diff'] > FP32_CANDIDATE_LEDGER_ATOL
            ):
                raise RuntimeError('{} candidate ledger gate failed'.format(prefix))
    audit = benchmark.get('correctness_matrix_validation')
    if (
        not isinstance(audit, dict)
        or audit.get('status') != 'PASS'
        or audit.get('errors') != []
        or audit.get('fp64_errors') != []
        or audit.get('expected_devices') != list(FORMAL_DEVICES)
        or audit.get('expected_steps') != list(FORMAL_CORRECTNESS_STEPS)
    ):
        raise RuntimeError('A3 benchmark self-audited correctness matrix is not clean')


def verify_benchmark_matrix(benchmark):
    parameters = benchmark.get('parameters')
    measurements = benchmark.get('measurements')
    if not isinstance(parameters, dict) or not isinstance(measurements, list):
        raise RuntimeError('A3 benchmark parameters/measurements are malformed')
    expected_parameters = {
        'devices': list(FORMAL_DEVICES),
        'precisions': list(FORMAL_PRECISIONS),
        'world_counts': list(FORMAL_WORLD_COUNTS),
        'cell_sweep': list(FORMAL_CELL_SWEEP),
        'particle_sweep': list(FORMAL_PARTICLE_SWEEP),
        'capacity_sweep': list(FORMAL_CAPACITY_SWEEP),
        'correctness_steps': list(FORMAL_CORRECTNESS_STEPS),
        'cells': 3,
        'steps': 1,
        'base_protein_capacity': 256,
        'canonical_initial_particles': 216,
        'fixed_particles': None,
        'include_fp32': True,
    }
    for name, expected in expected_parameters.items():
        if parameters.get(name) != expected:
            raise RuntimeError(
                'A3 benchmark formal parameter {} mismatch: expected={!r} observed={!r}'.format(
                    name, expected, parameters.get(name),
                )
            )
    if benchmark.get('formal_release_profile_requested') is not True:
        raise RuntimeError('A3 benchmark was not run with the formal release profile')

    steps = parameters['steps']
    canonical_particles = parameters['canonical_initial_particles']
    specs = []
    for value in FORMAL_WORLD_COUNTS:
        specs.append({
            'axis': 'worlds', 'value': value, 'worlds': value, 'cells': 3,
            'particles': canonical_particles, 'particles_requested': None,
            'capacity': 256, 'steps': steps,
        })
    for value in FORMAL_CELL_SWEEP:
        specs.append({
            'axis': 'cells', 'value': value, 'worlds': 1, 'cells': value,
            'particles': canonical_particles, 'particles_requested': None,
            'capacity': 256, 'steps': steps,
        })
    for value in FORMAL_PARTICLE_SWEEP:
        specs.append({
            'axis': 'particles', 'value': value, 'worlds': 1, 'cells': 3,
            'particles': value, 'particles_requested': value,
            'capacity': 256, 'steps': steps,
        })
    for value in FORMAL_CAPACITY_SWEEP:
        specs.append({
            'axis': 'protein_capacity', 'value': value, 'worlds': 1, 'cells': 3,
            'particles': canonical_particles, 'particles_requested': None,
            'capacity': value, 'steps': steps,
        })
    if len(specs) != 13:
        raise AssertionError('internal formal sweep specification is not 13 axes')

    expected = {}
    for spec in specs:
        axis_value = (spec['axis'], spec['value'])
        expected[axis_value + ('frozen-0.6.6-cpu-reference', 'cpu', 'float64')] = (
            spec, 'PASS', 1 if spec['axis'] == 'worlds' else 0,
            2 if spec['axis'] == 'worlds' else 1,
        )
        for device in FORMAL_DEVICES:
            for precision in FORMAL_PRECISIONS:
                expected[axis_value + (
                    'A3-hybrid-independent-worlds-sequential-host-loop',
                    device, precision,
                )] = (
                    spec,
                    'MEASURED_CANDIDATE' if precision == 'float32' else 'PASS',
                    1 if spec['axis'] == 'worlds' else 0,
                    2 if spec['axis'] == 'worlds' else 1,
                )
    if len(expected) != 65:
        raise AssertionError('internal formal measurement matrix is not 65 rows')

    observed = {}
    for index, item in enumerate(measurements):
        if not isinstance(item, dict):
            raise RuntimeError('A3 benchmark measurement {} is not a mapping'.format(index))
        key = (
            item.get('sweep_axis'), item.get('sweep_axis_value'),
            item.get('runner'), item.get('device'), item.get('precision'),
        )
        if key in observed:
            raise RuntimeError('A3 benchmark duplicate measurement identity {!r}'.format(key))
        if key not in expected:
            raise RuntimeError('A3 benchmark unexpected measurement identity {!r}'.format(key))
        observed[key] = item
        spec, status, warmup, repeats = expected[key]
        if item.get('status') != status or item.get('error') not in (None, ''):
            raise RuntimeError('A3 benchmark measurement status/error invalid for {!r}'.format(key))
        if item.get('prerequisite_fp64_correctness_gate') != 'PASS':
            raise RuntimeError('A3 benchmark measurement lacks fp64 gate for {!r}'.format(key))
        if item.get('runner') == 'A3-hybrid-independent-worlds-sequential-host-loop':
            expected_label = (
                'fp32-candidate-only-after-fp64-pass'
                if item.get('precision') == 'float32'
                else 'fp64-reference-correctness-gated'
            )
            if item.get('precision_status') != expected_label:
                raise RuntimeError('A3 benchmark precision label mismatch for {!r}'.format(key))
        _verify_timing_row(item, spec, warmup, repeats)

    missing = set(expected) - set(observed)
    if missing or len(measurements) != 65:
        raise RuntimeError(
            'A3 benchmark matrix incomplete: rows={} missing={!r}'.format(
                len(measurements), sorted(missing, key=repr),
            )
        )
    audit = benchmark.get('measurement_matrix_validation')
    if (
        not isinstance(audit, dict)
        or audit.get('status') != 'PASS'
        or audit.get('errors') != []
        or audit.get('expected_sweep_specs') != 13
        or audit.get('expected_rows') != 65
        or audit.get('observed_rows') != 65
    ):
        raise RuntimeError('A3 benchmark self-audited measurement matrix is not clean')
    return specs


def cp(src, dst):
    if not src.is_file():
        raise FileNotFoundError('required release input is missing: {}'.format(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def require_final_text(path, required_markers=()):
    if not path.is_file():
        raise RuntimeError('required final evidence is missing: {}'.format(path))
    text = path.read_text(encoding='utf-8')
    for token in ('PENDING', 'NOT_RUN', 'FAIL-CLOSED'):
        if token in text:
            raise RuntimeError('{} still contains unfinished marker {!r}'.format(path, token))
    for marker in required_markers:
        if marker not in text:
            raise RuntimeError('{} lacks required marker {!r}'.format(path, marker))
    return text


def load_required_json(path):
    if not path.is_file():
        raise RuntimeError('required final evidence is missing: {}'.format(path))
    return json.loads(path.read_text(encoding='utf-8'))


def verify_benchmark_siblings(benchmark):
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
    csv_path = ROOT / 'results/soma_cell_0_6_8_gpu_a3_benchmark.csv'
    if not csv_path.is_file():
        raise RuntimeError('A3 benchmark CSV sibling is missing')
    with csv_path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        csv_rows = list(reader)
        if tuple(reader.fieldnames or ()) != fields:
            raise RuntimeError('A3 benchmark CSV columns differ from the frozen writer')
    measurements = benchmark.get('measurements', [])
    if len(csv_rows) != len(measurements):
        raise RuntimeError('A3 benchmark CSV/JSON row counts differ')
    for index, (observed, measurement) in enumerate(zip(csv_rows, measurements)):
        expected = {}
        for name in fields:
            value = measurement.get(name, '')
            if isinstance(value, (list, tuple, dict)):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True)
            elif value is None:
                value = ''
            else:
                value = str(value)
            expected[name] = value
        if observed != expected:
            differing = [name for name in fields if observed.get(name) != expected.get(name)]
            raise RuntimeError(
                'A3 benchmark CSV row {} differs from JSON fields {}'.format(
                    index, ', '.join(differing),
                )
            )

    txt_path = ROOT / 'results/soma_cell_0_6_8_gpu_a3_benchmark.txt'
    if not txt_path.is_file():
        raise RuntimeError('A3 benchmark TXT sibling is missing')
    lines = [
        'SOMA-CELL 0.6.8-GPU A3 BENCHMARK', '=' * 52,
        'captured_utc: {}'.format(benchmark.get('captured_utc')),
        'fp64 gates: {}'.format(benchmark.get('all_requested_fp64_gates_pass')),
        'measurements complete: {}'.format(
            benchmark.get('all_requested_measurements_completed')
        ),
        'measurements: {}'.format(len(measurements)),
        'errors: {}'.format(len(benchmark.get('errors', []))),
        'full_gpu_world_step: false',
        'fp32: candidate only after fp64 PASS', '',
    ]
    lines.extend('ERROR: {}'.format(item) for item in benchmark.get('errors', []))
    expected_txt = '\n'.join(lines) + '\n'
    if txt_path.read_text(encoding='utf-8') != expected_txt:
        raise RuntimeError('A3 benchmark TXT sibling differs from the JSON summary')


def validate_frozen_evidence():
    validation = load_required_json(SRC / 'SOMA_CELL_0_6_8_GPU_A3_VALIDATION_RESULTS.json')
    verify_validation_rows(validation)
    validation_sources = verify_source_sha256_map(
        validation, VALIDATION_SOURCE_PATHS, 'A3 validation evidence',
    )

    benchmark_path = ROOT / 'results/soma_cell_0_6_8_gpu_a3_benchmark.json'
    benchmark = load_required_json(benchmark_path)
    if benchmark.get('build') != 'SOMA-CELL 0.6.8-GPU A3':
        raise RuntimeError('A3 benchmark build marker mismatch')
    if (
        not benchmark.get('all_requested_fp64_gates_pass')
        or not benchmark.get('all_requested_measurements_completed')
        or benchmark.get('errors')
    ):
        raise RuntimeError('A3 benchmark correctness gate is not clean')
    if benchmark.get('full_gpu_world_step') is not False:
        raise RuntimeError('A3 benchmark must keep full_gpu_world_step=false')
    if benchmark.get('fp64_gate_passed_before_fp32') is not True:
        raise RuntimeError('A3 benchmark lacks fp64_gate_passed_before_fp32=true')
    benchmark_sources = verify_source_sha256_map(
        benchmark, BENCHMARK_SOURCE_PATHS, 'A3 benchmark evidence',
    )
    for relative in VALIDATION_SOURCE_PATHS:
        if benchmark_sources[relative] != validation_sources[relative]:
            raise RuntimeError(
                'validation/benchmark source binding differs for {}'.format(relative)
            )
    verify_benchmark_correctness(benchmark)
    verify_benchmark_matrix(benchmark)
    verify_benchmark_siblings(benchmark)
    parameters = benchmark.get('parameters', {})
    if parameters.get('correctness_steps') != [1, 10, 20]:
        raise RuntimeError('A3 benchmark did not retain correctness steps 1/10/20')
    policy = parameters.get('timing_policy_by_axis', {})
    if policy.get('worlds') != {'warmup': 1, 'repeats': 2}:
        raise RuntimeError('A3 benchmark worlds timing policy must be warmup=1/repeats=2')
    if policy.get('cells_particles_capacity') != {'warmup': 0, 'repeats': 1}:
        raise RuntimeError('A3 benchmark secondary-axis timing policy mismatch')
    if parameters.get('world_counts') != [1, 8, 32, 128]:
        raise RuntimeError('A3 benchmark lacks required 1/8/32/128 worlds sweep')

    discrepancy_path = ROOT / 'results/SOMA_CELL_0_6_8_GPU_A3_FP32_DISCREPANCY.json'
    discrepancy = load_required_json(discrepancy_path)
    if discrepancy.get('build') != 'SOMA-CELL 0.6.8-GPU A3':
        raise RuntimeError('A3 fp32 discrepancy build marker mismatch')
    if discrepancy.get('fp64_gate_passed_before_fp32') is not True:
        raise RuntimeError('fp32 evidence lacks the required preceding fp64 gate')
    if discrepancy.get('candidate_only') is not True:
        raise RuntimeError('fp32 evidence is not labelled candidate-only')
    if discrepancy.get('candidate_internal_ledger_tolerance') != FP32_CANDIDATE_LEDGER_ATOL:
        raise RuntimeError('fp32 evidence candidate ledger tolerance mismatch')
    discrepancy_sources = discrepancy.get('source_sha256_map')
    if discrepancy_sources != benchmark_sources:
        raise RuntimeError('fp32 evidence source map is not bound to benchmark sources')
    parent = discrepancy.get('parent_benchmark')
    if not isinstance(parent, dict):
        raise RuntimeError('fp32 evidence lacks parent benchmark linkage')
    if parent.get('path') != benchmark_path.name:
        raise RuntimeError('fp32 evidence parent benchmark path mismatch')
    if parent.get('sha256') != sha(benchmark_path):
        raise RuntimeError('fp32 evidence parent benchmark hash mismatch')
    if discrepancy.get('precision_discrepancy') != benchmark.get('precision_discrepancy'):
        raise RuntimeError('fp32 discrepancy payload differs from parent benchmark')
    expected_fp32_measurements = [
        item for item in benchmark.get('measurements', [])
        if item.get('precision') == 'float32'
    ]
    if discrepancy.get('fp32_measurements') != expected_fp32_measurements:
        raise RuntimeError('fp32 measurement linkage differs from parent benchmark')

    require_final_text(
        ROOT / 'results/SOMA_CELL_0_6_8_GPU_A3_REGRESSION_RESULTS.txt',
        ('36 / 36 PASS', '279 / 279 PASS', '379 / 379 PASS', '21 / 22'),
    )
    require_final_text(
        ROOT / 'results/SOMA_CELL_0_6_8_GPU_A3_EXPERIMENT_REPORT.txt',
        (
            '36 / 36 PASS', '379 / 379 PASS', 'full_gpu_world_step: false',
            'There is no speedup.', sha(benchmark_path),
        ),
    )


def cuda_available():
    command = [
        sys.executable,
        '-c',
        'import torch; print(1 if torch.cuda.is_available() else 0)',
    ]
    try:
        return subprocess.check_output(command, text=True, timeout=30).strip() == '1'
    except (OSError, subprocess.SubprocessError):
        return False


def write_deterministic_zip(stage):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(OUT.suffix + '.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(stage.iterdir(), key=lambda value: value.name):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(
                str(Path(NAME) / path.name).replace(os.sep, '/'),
                date_time=ZIP_TIMESTAMP,
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    os.replace(temporary, OUT)


def main():
    validate_frozen_evidence()
    with tempfile.TemporaryDirectory(prefix='soma068gpu_a3_') as directory:
        stage = Path(directory) / NAME
        stage.mkdir()

        for number in range(1, 6):
            filename = 'SOMA_CELL_0_{}_pythonista.py'.format(number)
            cp(ROOT / 'src/baseline' / filename, stage / filename)

        dependencies = [
            ('0_6_p0', 'SOMA_CELL_0_6_P0_pythonista.py'),
            ('0_6_p1', 'SOMA_CELL_0_6_P1_pythonista.py'),
            ('0_6_p2', 'SOMA_CELL_0_6_P2_pythonista.py'),
            ('0_6', 'SOMA_CELL_0_6_pythonista.py'),
            ('0_6_1', 'SOMA_CELL_0_6_1_pythonista.py'),
            ('0_6_2', 'SOMA_CELL_0_6_2_pythonista.py'),
            ('0_6_3', 'SOMA_CELL_0_6_3_pythonista.py'),
            ('0_6_4', 'SOMA_CELL_0_6_4_pythonista.py'),
            ('0_6_5', 'SOMA_CELL_0_6_5_pythonista.py'),
            ('0_6_6', 'SOMA_CELL_0_6_6_pythonista.py'),
            ('0_6_7', 'SOMA_CELL_0_6_7_pythonista.py'),
        ]
        for source_dir, filename in dependencies:
            cp(ROOT / 'src' / source_dir / filename, stage / filename)

        own_files = [
            'SOMA_CELL_0_6_8_gpu.py',
            'SOMA_CELL_0_6_8_validation.py',
            'SOMA_CELL_0_6_8_gpu_benchmark.py',
            'SOMA_CELL_0_6_8_install_check.py',
            'SOMA_CELL_0_6_8_pythonista.py',
            'SOMA_CELL_0_6_8_GPU_START_HERE.txt',
            'SOMA_CELL_0_6_8_GPU_README_JA.md',
            'SOMA_CELL_0_6_8_GPU_ENVIRONMENT_REPORT.json',
            'soma_cell_0_6_8_gpu_benchmark_cpu.json',
            'soma_cell_0_6_8_gpu_validation.csv',
            'SOMA_CELL_0_6_8_gpu_a2.py',
            'SOMA_CELL_0_6_8_A2_validation.py',
            'SOMA_CELL_0_6_8_A2_benchmark.py',
            'SOMA_CELL_0_6_8_GPU_A2_pythonista.py',
            'SOMA_CELL_0_6_8_GPU_A2_START_HERE.txt',
            'SOMA_CELL_0_6_8_GPU_A2_README_JA.md',
            'SOMA_CELL_0_6_8_GPU_A2_ENVIRONMENT_REPORT.json',
            'soma_cell_0_6_8_gpu_a2_benchmark_cpu.json',
            'soma_cell_0_6_8_gpu_a2_validation.csv',
            'SOMA_CELL_0_6_8_gpu_a3.py',
            'SOMA_CELL_0_6_8_gpu_a3_scheduler.py',
            'SOMA_CELL_0_6_8_A3_validation.py',
            'SOMA_CELL_0_6_8_A3_benchmark.py',
            'SOMA_CELL_0_6_8_GPU_A3_pythonista.py',
            'SOMA_CELL_0_6_8_GPU_A3_START_HERE.txt',
            'SOMA_CELL_0_6_8_GPU_A3_README_JA.md',
            'SOMA_CELL_0_6_8_GPU_A3_ENVIRONMENT_REPORT.json',
            'SOMA_CELL_0_6_8_GPU_A3_VALIDATION_RESULTS.json',
            'SOMA_CELL_0_6_8_GPU_A3_VALIDATION_RESULTS.txt',
            'soma_cell_0_6_8_gpu_a3_validation.csv',
        ]
        for filename in own_files:
            cp(SRC / filename, stage / filename)

        documentation = [
            'SOMA_CELL_0_6_8_GPU_CONTRACT.md',
            'SOMA_CELL_0_6_8_GPU_SCHEMA.json',
            'SOMA_CELL_0_6_8_GPU_A2_CONTRACT.md',
            'SOMA_CELL_0_6_8_GPU_A2_SCHEMA.json',
            'SOMA_CELL_0_6_8_GPU_A3_CONTRACT.md',
            'SOMA_CELL_0_6_8_GPU_A3_SCHEMA.json',
            'SOMA_CELL_0_6_8_GPU_A3_EVENT_ORDER_JA.md',
        ]
        for filename in documentation:
            cp(ROOT / 'docs' / filename, stage / filename)
        cp(
            ROOT / 'planning/SOMA_CELL_0_6_8_GPU_A3_PREREGISTRATION_JA.md',
            stage / 'SOMA_CELL_0_6_8_GPU_A3_PREREGISTRATION_JA.md',
        )

        result_files = [
            'SOMA_CELL_0_6_8_GPU_VALIDATION_RESULTS.txt',
            'SOMA_CELL_0_6_8_GPU_REGRESSION_RESULTS.txt',
            'SOMA_CELL_0_6_8_GPU_EXPERIMENT_REPORT.txt',
            'SOMA_CELL_0_6_8_GPU_A2_VALIDATION_RESULTS.txt',
            'SOMA_CELL_0_6_8_GPU_A2_REGRESSION_RESULTS.txt',
            'SOMA_CELL_0_6_8_GPU_A2_EXPERIMENT_REPORT.txt',
            'SOMA_CELL_0_6_8_GPU_A3_VALIDATION_RESULTS.txt',
            'SOMA_CELL_0_6_8_GPU_A3_REGRESSION_RESULTS.txt',
            'SOMA_CELL_0_6_8_GPU_A3_EXPERIMENT_REPORT.txt',
            'SOMA_CELL_0_6_8_GPU_A3_A2_BASELINE_20260814.json',
            'soma_cell_0_6_8_gpu_a3_benchmark.json',
            'soma_cell_0_6_8_gpu_a3_benchmark.csv',
            'soma_cell_0_6_8_gpu_a3_benchmark.txt',
            'SOMA_CELL_0_6_8_GPU_A3_FP32_DISCREPANCY.json',
        ]
        for filename in result_files:
            cp(ROOT / 'results' / filename, stage / filename)

        note = (
            'Install a CUDA-enabled PyTorch build appropriate for the NVIDIA driver.\n'
            'Run SOMA_CELL_0_6_8_install_check.py, then A3 validation with --require-cuda.\n'
            'CUDA is never silently replaced with CPU when explicitly requested.\n'
            'A3 remains a hybrid checkpoint: full_gpu_world_step is false.\n'
        )
        (stage / 'CUDA_INSTALL_NOTE.txt').write_bytes(note.encode('utf-8'))

        for path in stage.glob('*.py'):
            py_compile.compile(str(path), doraise=True)
        shutil.rmtree(stage / '__pycache__', ignore_errors=True)

        env = dict(os.environ)
        env['PYTHONPATH'] = str(stage)
        subprocess.check_call(
            [
                sys.executable, '-c',
                "import SOMA_CELL_0_6_8_gpu_a3 as g; "
                "assert g.BUILD == 'SOMA-CELL 0.6.8-GPU A3'; "
                "assert g.SCHEMA_VERSION == '0.6.8-GPU-A3.0'; "
                "assert g.FULL_GPU_WORLD_STEP is False; "
                "assert g.PORT_STATUS['division'].startswith('cpu-authoritative')",
            ],
            cwd=stage,
            env=env,
        )
        validation_command = [
            sys.executable,
            'SOMA_CELL_0_6_8_A3_validation.py',
            '--no-write',
        ]
        if cuda_available():
            validation_command.append('--require-cuda')
        subprocess.check_call(validation_command, cwd=stage, env=env)

        files = sorted(
            path for path in stage.iterdir()
            if path.is_file() and path.name != 'SHA256SUMS.txt'
        )
        sums = ''.join('{}  ./{}\n'.format(sha(path), path.name) for path in files)
        (stage / 'SHA256SUMS.txt').write_bytes(sums.encode('utf-8'))
        write_deterministic_zip(stage)

    with zipfile.ZipFile(OUT) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('release ZIP CRC verification failed')
    print(OUT)
    print(sha(OUT))


if __name__ == '__main__':
    main()
