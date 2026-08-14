# coding: utf-8
"""Focused validation for SOMA-CELL 0.6.8-GPU A4.1 through A4.3 slices."""
from __future__ import division

import argparse
import copy
import csv
import hashlib
import inspect
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import SOMA_CELL_0_6_8_gpu_a4 as a4
import SOMA_CELL_0_6_8_A3_validation as v3

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

RESULT_JSON = 'SOMA_CELL_0_6_8_GPU_A4_3_VALIDATION_RESULTS.json'
RESULT_CSV = 'soma_cell_0_6_8_gpu_a4_3_validation.csv'
RESULT_TXT = 'SOMA_CELL_0_6_8_GPU_A4_3_VALIDATION_RESULTS.txt'

SOURCE_PATHS = (
    'src/0_6_8/SOMA_CELL_0_6_8_gpu_a4.py',
    'src/0_6_8/SOMA_CELL_0_6_8_A4_validation.py',
    'src/0_6_8/SOMA_CELL_0_6_8_gpu_a3.py',
    'src/0_6_8/SOMA_CELL_0_6_8_gpu_a3_scheduler.py',
    'src/0_6_8/SOMA_CELL_0_6_8_A3_validation.py',
    'src/0_6_6/SOMA_CELL_0_6_6_pythonista.py',
    'src/0_6_5/SOMA_CELL_0_6_5_pythonista.py',
    'src/baseline/SOMA_CELL_0_5_pythonista.py',
    'src/baseline/SOMA_CELL_0_4_pythonista.py',
    'src/baseline/SOMA_CELL_0_3_pythonista.py',
    'src/baseline/SOMA_CELL_0_2_pythonista.py',
    'src/0_6_p0/SOMA_CELL_0_6_P0_pythonista.py',
    'docs/SOMA_CELL_0_6_8_GPU_A4_CONTRACT.md',
    'docs/SOMA_CELL_0_6_8_GPU_A4_SCHEMA.json',
    'planning/SOMA_CELL_0_6_8_GPU_A4_PREREGISTRATION_JA.md',
)

_REQUIRE_CUDA = False


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def source_sha256_map():
    result = {}
    for relative in SOURCE_PATHS:
        path = os.path.join(ROOT, *relative.split('/'))
        if not os.path.isfile(path):
            raise AssertionError('source hash input missing: %s' % relative)
        result[relative] = _sha256(path)
    return result


def _fixture_cells(seed=7401):
    world = v3.make_world(seed=seed, cells=4)
    cells = world.cells
    gene_a = a4.g2.make_gene(
        a4.g2.ROLE_ENERGY, parameter=1, promoter=7, efficiency=6,
    )
    gene_b = a4.g2.make_gene(
        a4.g2.ROLE_GENERIC,
        parameter=a4.g2.REACTION_INTERMEDIATE_TO_WASTE,
        promoter=4, efficiency=5,
    )
    damaged_prefix = np.asarray((4, 3, 2, 1, 7), dtype=np.uint8)
    sequence_a = np.concatenate((gene_a, damaged_prefix, gene_b)).astype(np.uint8)
    sequence_b = np.concatenate((gene_b, gene_a, gene_a)).astype(np.uint8)

    cells[0].genomes = []
    cells[0].genome_lesions = []
    cells[0].replication_template = None
    cells[0].replication_copy = []
    cells[0].replication_template_lesion = 0.0
    cells[0].replication_fractional = 0.0

    cells[1].genomes = [sequence_a.copy()]
    # This shorter-than-genome-count vector is legal until the damage phase.
    cells[1].genome_lesions = []
    # An active replication record must retain even an empty copy sequence.
    cells[1].replication_template = sequence_a.copy()
    cells[1].replication_copy = []
    cells[1].replication_template_lesion = 0.125
    cells[1].replication_fractional = 0.25

    cells[2].genomes = [sequence_b.copy(), sequence_a.copy()]
    cells[2].genome_lesions = [0.25]
    cells[2].replication_template = sequence_b.copy()
    cells[2].replication_copy = [int(value) for value in sequence_b[:11]]
    cells[2].replication_template_lesion = 0.375
    cells[2].replication_fractional = 0.625

    # Frozen damage state can transiently contain three complete polymers;
    # this explicitly guards against the old A1 `[:2]` representation loss.
    cells[3].genomes = [
        sequence_a.copy(), sequence_b.copy(), sequence_a.copy(),
    ]
    cells[3].genome_lesions = [0.1, 0.2, 0.3]
    cells[3].replication_template = None
    cells[3].replication_copy = []
    cells[3].replication_template_lesion = 0.0
    cells[3].replication_fractional = 0.0

    for cell in cells:
        cell._refresh_gene_cache()
    return world, cells


def _structural_state(cell):
    return {
        'cell_id': int(cell.cell_id),
        'genomes': [np.asarray(item, dtype=np.uint8).copy()
                    for item in cell.genomes],
        'genome_lesions': list(float(value) for value in cell.genome_lesions),
        'replication_template': (
            None if cell.replication_template is None
            else np.asarray(cell.replication_template, dtype=np.uint8).copy()
        ),
        'replication_copy': list(int(value) for value in cell.replication_copy),
        'replication_template_lesion': float(cell.replication_template_lesion),
        'replication_fractional': float(cell.replication_fractional),
        'gene_order': list(cell.gene_specs.keys()),
        'gene_specs': copy.deepcopy(cell.gene_specs),
    }


def _assert_structural_equal(expected, actual, label='cell'):
    v3.assert_recursive_close(
        _structural_state(expected), _structural_state(actual),
        atol=0.0, rtol=0.0, path=label,
    )


def _target_cells(seed=7501):
    world = v3.make_world(seed=seed, cells=4)
    return world, world.cells


def _gene_decode_fixture(seed=7601):
    """One-cell fixture fixing malformed-marker and cache-order semantics."""
    world = v3.make_world(seed=seed, cells=1)
    cell = world.cells[0]
    non_generic = a4.g2.make_gene(
        a4.g2.ROLE_TRANSLATOR,
        parameter=5, regulator=2, promoter=7, efficiency=0,
        fidelity=6, localisation=7, spare=(1, 2, 3, 4),
    )
    generic = a4.g2.make_gene(
        a4.g2.ROLE_GENERIC,
        parameter=a4.g2.REACTION_WASTE_TO_INTERMEDIATE,
        regulator=7, promoter=0, efficiency=7,
        fidelity=0, localisation=6, spare=(4, 5, 6, 7),
    )
    grammar = a4.s65.make_grammar_gene(
        a4.s65.GRAMMAR_READINESS,
        value=6, promoter=5, efficiency=4, fidelity=6,
    )
    bad_start = np.asarray([7, 6] + [1] * 12 + [0, 0], dtype=np.uint8)
    bad_stop = np.asarray([7, 7] + [1] * 12 + [0, 1], dtype=np.uint8)
    # Both outer @0 and nested @2 are syntactically valid.  Frozen parsing
    # accepts outer then advances 16, suppressing the nested candidate.
    overlapping_valid = np.concatenate((
        grammar, np.asarray([0, 0], dtype=np.uint8),
    ))
    # Outer @0 has the wrong stop; one-symbol recovery finds generic @2.
    invalid_outer_valid_inner = np.concatenate((
        np.asarray([7, 7], dtype=np.uint8), generic,
    ))
    genome0 = np.concatenate((
        bad_start, np.asarray([3, 4, 5], dtype=np.uint8),
        non_generic, generic,
    )).astype(np.uint8)
    genome1 = np.concatenate((
        generic, np.asarray([1, 2, 3], dtype=np.uint8),
        overlapping_valid, bad_stop,
    )).astype(np.uint8)
    genome2 = np.concatenate((
        np.asarray([1, 2], dtype=np.uint8), non_generic,
        invalid_outer_valid_inner, bad_stop, non_generic[:-1],
    )).astype(np.uint8)
    template_only = a4.g2.make_gene(
        a4.g2.ROLE_MEMBRANE, parameter=0, promoter=6, efficiency=5,
    )
    template = np.concatenate((template_only, non_generic)).astype(np.uint8)

    cell.genomes = [genome0, genome1, genome2]
    cell.genome_lesions = [0.1, 0.2, 0.3]
    cell.replication_template = template
    cell.replication_copy = [int(value) for value in template[:7]]
    cell.replication_template_lesion = 0.25
    cell.replication_fractional = 0.5
    cell._refresh_gene_cache()
    config = a4.GPU068A4Config(
        max_cells=1, max_sequences=5, max_symbols=210,
        max_sequence_symbols=67,
    )
    return world, cell, config


def _assert_gene_specs_equal(expected, actual, label):
    if list(expected.keys()) != list(actual.keys()):
        raise AssertionError('%s insertion order differs' % label)
    v3.assert_recursive_close(
        expected, actual, atol=0.0, rtol=0.0, path=label,
    )


def _assert_raises(kind, fn):
    try:
        fn()
    except kind:
        return
    raise AssertionError('expected %s' % kind.__name__)


def _paid_translation_fixture(seed=7701):
    """Four Formal066 cells fixing payment, early-gate, and MRO signals."""
    world = v3.make_world(seed=seed, cells=4)
    cells = world.cells
    for ci, cell in enumerate(cells):
        cell.last_receptor_activity = np.linspace(
            0.03 + 0.01 * ci, 0.42 + 0.01 * ci,
            a4.s4.LIGAND_COUNT, dtype=np.float64,
        )
        cell.last_control_vectors = np.asarray([
            (0.01 * (index + 1), -0.006 * (ci + index + 1))
            for index in range(a4.s4.CONTROL_COUNT)
        ], dtype=np.float64)
        cell.last_control_scalars = np.linspace(
            -0.08, 0.11 + 0.01 * ci, a4.s4.CONTROL_COUNT,
            dtype=np.float64,
        )
        cell.last_edna_signal = 0.07 + 0.02 * ci
        cell.last_corpse_signal = 0.05 + 0.01 * ci
        cell.last_necrotoxin_signal = 0.04 + 0.015 * ci
        cell.behavioural_quiescence = 0.18 + 0.03 * ci
        cell.last_translation = 0.41 + 0.01 * ci
        cell.last_quiescence = 0.61 + 0.01 * ci
        cell._refresh_gene_cache()
        cell._sync_protein_pool()
        cell._sync_damage_pool()

    # Formal066 inherits P0ProtoCell.osmolyte(), which includes every neural
    # attachment.  Keep one real attachment in the shared fixture so both the
    # NumPy oracle and resident Torch path exercise that dynamic-MRO input.
    p0 = sys.modules['SOMA_CELL_0_6_P0_pythonista']
    attachment = p0.NeuralAttachmentState(
        'a4-neural-osmolyte',
        stores=np.full((p0.BUDGET_COUNT,), 10.0, dtype=np.float64),
        tissue_material=np.full(
            (p0.TISSUE_MATERIAL_COUNT,), 10.0, dtype=np.float64,
        ),
    )
    cells[0].neural_attachments['a4-neural-osmolyte'] = attachment

    # Reaching the weight gate with exactly the ATP reserve must update
    # quiescence but produce no protein.
    cells[1].pools[a4.a3.POOL_ATP] = 0.042
    # One early gene consumes this tiny mineral budget; later genes see the
    # already depleted material in frozen cache order.
    cells[2].pools[a4.a3.POOL_MINERAL] = 0.36e-6
    # Remove every active translator while retaining its genes.  This returns
    # before quiescence and leaves that observable at its sentinel value.
    translator_keys = [
        fingerprint for fingerprint, spec in cells[3].gene_specs.items()
        if int(spec['role']) == int(a4.a3.ROLE_TRANSLATOR)
    ]
    for fingerprint in translator_keys:
        cells[3].proteins.pop(fingerprint, None)
    cells[3]._sync_protein_pool()
    cells[3]._sync_damage_pool()

    config = a4.GPU068A4Config(
        max_cells=4, max_sequences=16, max_symbols=8192,
        max_sequence_symbols=a4.MAX_FROZEN_GENOME_SYMBOLS,
        max_proteins_per_cell=64,
    )
    return world, cells, config, 0.1


def _assert_translation_matches_cells(state, cells, label, atol=2e-12):
    if int(state.cell_count) != len(cells):
        raise AssertionError('%s cell count differs' % label)
    for ci, cell in enumerate(cells):
        v3.assert_recursive_close(
            cell.pools, state.pools[ci], atol=atol, rtol=0.0,
            path='%s.cell[%d].pools' % (label, ci),
        )
        v3.assert_recursive_close(
            float(cell.last_translation), float(state.last_translation[ci]),
            atol=atol, rtol=0.0,
            path='%s.cell[%d].last_translation' % (label, ci),
        )
        v3.assert_recursive_close(
            float(cell.last_quiescence), float(state.last_quiescence[ci]),
            atol=atol, rtol=0.0,
            path='%s.cell[%d].last_quiescence' % (label, ci),
        )
        active_count = int(state.active_count[ci])
        damaged_count = int(state.damaged_count[ci])
        active_order = [int(value) for value in
                        state.active_fingerprints[ci, :active_count]]
        damaged_order = [int(value) for value in
                         state.damaged_fingerprints[ci, :damaged_count]]
        if active_order != list(cell.proteins.keys()):
            raise AssertionError('%s cell[%d] active insertion order differs' %
                                 (label, ci))
        if damaged_order != list(cell.damaged_proteins.keys()):
            raise AssertionError('%s cell[%d] damaged insertion order differs' %
                                 (label, ci))
        v3.assert_recursive_close(
            np.asarray(list(cell.proteins.values()), dtype=np.float64),
            state.active_mass[ci, :active_count],
            atol=atol, rtol=0.0,
            path='%s.cell[%d].active' % (label, ci),
        )
        v3.assert_recursive_close(
            np.asarray(list(cell.damaged_proteins.values()), dtype=np.float64),
            state.damaged_mass[ci, :damaged_count],
            atol=atol, rtol=0.0,
            path='%s.cell[%d].damaged' % (label, ci),
        )


def test_api_scope():
    required = (
        'GPU068A4Config', 'A4RaggedGenomeBatch',
        'FullFidelityA4GenomeAdapter', 'validate_a4_ragged',
        'pack_a4_cells', 'unpack_a4_cells', 'A4GeneCacheBatch',
        'validate_a4_gene_cache', 'decode_a4_gene_cache_numpy',
        'decode_a4_gene_cache_torch', 'decode_a4_gene_cache',
        'A4TranslationStateBatch', 'FullFidelityA4TranslationAdapter',
        'pack_a4_translation_state', 'bind_a4_translation',
        'paid_translation_plan_numpy', 'paid_translation_plan_torch',
        'paid_translation_plan', 'validate_a4_translation_state',
    )
    missing = [name for name in required if not hasattr(a4, name)]
    if missing:
        raise AssertionError('missing A4 API: %s' % missing)
    if a4.FULL_GPU_WORLD_STEP is not False:
        raise AssertionError('A4.1 must not claim full GPU world-step')
    if hasattr(a4.A4GeneCacheBatch, 'from_state_dict'):
        raise AssertionError('derived cache must not expose a deserialize authority')
    if a4.PORT_STATUS.get('gene_cache_decode') != (
            'a4.2-batched-resident-derived-cache'):
        raise AssertionError('A4.2 gene-cache status is missing')
    if a4.PORT_STATUS.get('translation') != (
            'a4.3-full-formal066-paid-plan-not-integrated-cpu-authoritative'):
        raise AssertionError('A4.3 translation plan status is missing')
    if a4.PORT_STATUS.get('genome_replication') != 'cpu-authoritative-next-a4-slice':
        raise AssertionError('replication authority changed in representation slice')
    return '%s / %s + %s / full_gpu=false' % (
        a4.BUILD, a4.SCHEMA_VERSION,
        a4.GENE_CACHE_SCHEMA_VERSION + ' + ' + a4.TRANSLATION_SCHEMA_VERSION,
    )


def test_source_hash_inputs_present():
    values = source_sha256_map()
    if set(values) != set(SOURCE_PATHS):
        raise AssertionError('source hash identity set differs')
    if any(len(value) != 64 for value in values.values()):
        raise AssertionError('invalid SHA-256')
    return '%d authoritative development inputs hashed' % len(values)


def test_ragged_batch_roundtrip_lossless():
    source_world, cells = _fixture_cells()
    source_before = v3.pickle_clone(source_world.state_dict())
    adapter = a4.FullFidelityA4GenomeAdapter()
    batch = adapter.pack_cells(cells)
    a4.validate_a4_ragged(batch)
    v3.assert_recursive_close(
        source_before, source_world.state_dict(), atol=0.0, rtol=0.0,
        path='pack_source',
    )
    if list(batch.genome_counts[:4]) != [0, 1, 2, 3]:
        raise AssertionError('complete genome counts/order changed')
    if list(batch.replication_active[:4]) != [False, True, True, False]:
        raise AssertionError('replication flags changed')
    if list(batch.lesion_offsets[:5]) != [0, 0, 0, 1, 4]:
        raise AssertionError('transient lesion vector was padded')
    expected_material = np.asarray((0, len(cells[1].genomes[0]),
                                    sum(len(g) for g in cells[2].genomes) +
                                    len(cells[2].replication_copy),
                                    sum(len(g) for g in cells[3].genomes)),
                                   dtype=np.int64)
    if not np.array_equal(batch.material_symbol_counts_host()[:4],
                          expected_material):
        raise AssertionError('template was counted as material or copy omitted')

    _, targets = _target_cells()
    adapter.unpack_cells(batch, targets)
    for index, (expected, actual) in enumerate(zip(cells, targets)):
        _assert_structural_equal(expected, actual, 'cell[%d]' % index)
    return '%d cells / %d sequences / %d symbols exact' % (
        batch.cell_count, batch.sequence_count, batch.symbol_count,
    )


def test_numpy_torch_fp64_roundtrip_and_deep_clone():
    _, cells = _fixture_cells(seed=7402)
    batch = a4.pack_a4_cells(cells)
    clone = batch.clone()
    original = int(batch.symbols[0])
    clone.symbols[0] = (original + 1) % a4.ALPHABET_SIZE
    if int(batch.symbols[0]) != original:
        raise AssertionError('clone aliases source symbol storage')
    restored = a4.A4RaggedGenomeBatch.from_state_dict(batch.state_dict())
    v3.assert_recursive_close(
        batch, restored, atol=0.0, rtol=0.0, path='state_dict',
    )
    if torch is None:
        raise AssertionError('PyTorch is required for A4.1')
    _assert_raises(ValueError, lambda: batch.to_torch(device='auto'))
    _assert_raises(ValueError, lambda: batch.to_torch(device='meta'))
    tensor = batch.to_torch(device='cpu')
    back = tensor.to_numpy()
    v3.assert_recursive_close(
        batch, back, atol=0.0, rtol=0.0, path='numpy_torch_numpy',
    )
    return 'deep clone + NumPy/Torch fp64 exact'


def test_ragged_schema_rejects_corruption_atomically():
    _, cells = _fixture_cells(seed=7403)
    adapter = a4.FullFidelityA4GenomeAdapter()
    batch = adapter.pack_cells(cells)
    corruptions = []

    def add(name, mutate):
        item = batch.clone()
        mutate(item)
        corruptions.append((name, item))

    add('offset_start', lambda x: x.sequence_offsets.__setitem__(0, 1))
    add('offset_nonmonotonic',
        lambda x: x.sequence_offsets.__setitem__(1, -1))
    add('alphabet', lambda x: x.symbols.__setitem__(0, a4.ALPHABET_SIZE))
    add('symbol_tail',
        lambda x: x.symbols.__setitem__(x.symbol_count, 1))
    add('unused_offset_tail',
        lambda x: x.sequence_offsets.__setitem__(x.sequence_count + 1, 0))
    add('cell_sequence_relation',
        lambda x: x.genome_counts.__setitem__(0, 1))
    add('inactive_replication_fractional',
        lambda x: x.replication_fractional.__setitem__(0, 0.25))
    add('dtype', lambda x: setattr(
        x, 'sequence_offsets', x.sequence_offsets.astype(np.int32),
    ))
    add('count_scalar_type', lambda x: setattr(x, 'cell_count', 3.0))

    _, targets = _target_cells(seed=7503)
    target_before = [_structural_state(cell) for cell in targets]
    for name, corrupt in corruptions:
        _assert_raises(a4.A4SchemaError,
                       lambda corrupt=corrupt: adapter.unpack_cells(corrupt, targets))
        for index, cell in enumerate(targets):
            v3.assert_recursive_close(
                target_before[index], _structural_state(cell),
                atol=0.0, rtol=0.0,
                path='atomic.%s.cell[%d]' % (name, index),
            )
    return '%d corruptions rejected before target mutation' % len(corruptions)


def test_capacity_exact_and_plus_one_atomic():
    world, cells = _fixture_cells(seed=7404)
    probe = a4.pack_a4_cells(cells)
    lengths = np.diff(probe.sequence_offsets[:probe.sequence_count + 1])
    exact_values = {
        'max_cells': probe.cell_count,
        'max_sequences': probe.sequence_count,
        'max_symbols': probe.symbol_count,
        'max_sequence_symbols': int(np.max(lengths)),
    }
    exact = a4.GPU068A4Config(**exact_values)
    packed = a4.FullFidelityA4GenomeAdapter(exact).pack_cells(cells)
    if (packed.cell_capacity, packed.sequence_capacity, packed.symbol_capacity) != (
            probe.cell_count, probe.sequence_count, probe.symbol_count):
        raise AssertionError('exact capacity did not remain exact')

    before = v3.pickle_clone(world.state_dict())
    _assert_raises(
        ValueError,
        lambda: a4.GPU068A4Config(max_cells=float(exact_values['max_cells'])),
    )
    _assert_raises(
        ValueError,
        lambda: a4.GPU068A4Config.from_state({'max_cell': exact_values['max_cells']}),
    )
    for key in exact_values:
        if exact_values[key] <= 1:
            raise AssertionError('fixture too small for +1 capacity assay')
        values = dict(exact_values)
        values[key] -= 1
        config = a4.GPU068A4Config(**values)
        _assert_raises(
            a4.A4CapacityError,
            lambda config=config: a4.FullFidelityA4GenomeAdapter(config).pack_cells(cells),
        )
        v3.assert_recursive_close(
            before, world.state_dict(), atol=0.0, rtol=0.0,
            path='capacity_source.%s' % key,
        )
    return '4 exact axes PASS / each capacity+1 rejected atomically'


def test_cell_identity_order_and_duplicate_rejected():
    _, cells = _fixture_cells(seed=7405)
    adapter = a4.FullFidelityA4GenomeAdapter()
    batch = adapter.pack_cells(cells)
    duplicate = copy.deepcopy(cells)
    duplicate[1].cell_id = duplicate[0].cell_id
    _assert_raises(a4.A4SchemaError, lambda: adapter.pack_cells(duplicate))
    missing_genomes = copy.deepcopy(cells)
    del missing_genomes[0].genomes
    _assert_raises(a4.A4SchemaError,
                   lambda: adapter.pack_cells(missing_genomes))
    nonnumeric_lesion = copy.deepcopy(cells)
    nonnumeric_lesion[2].genome_lesions = ['0.25']
    _assert_raises(a4.A4SchemaError,
                   lambda: adapter.pack_cells(nonnumeric_lesion))
    complex_lesion = copy.deepcopy(cells)
    complex_lesion[2].genome_lesions = [0.25 + 0.5j]
    _assert_raises(a4.A4SchemaError,
                   lambda: adapter.pack_cells(complex_lesion))

    _, targets = _target_cells(seed=7505)
    target_before = [_structural_state(cell) for cell in targets]
    reordered = [targets[1], targets[0], targets[2], targets[3]]
    _assert_raises(a4.A4SchemaError,
                   lambda: adapter.unpack_cells(batch, reordered))
    fractional_id = copy.deepcopy(targets)
    fractional_id[0].cell_id = float(batch.cell_ids[0]) + 0.5
    _assert_raises(a4.A4SchemaError,
                   lambda: adapter.unpack_cells(batch, fractional_id))
    for index, cell in enumerate(targets):
        v3.assert_recursive_close(
            target_before[index], _structural_state(cell),
            atol=0.0, rtol=0.0, path='identity.cell[%d]' % index,
        )
    return 'duplicate pack and reordered commit rejected'


def test_explicit_device_residency_checksum_smoke():
    if torch is None:
        raise AssertionError('PyTorch unavailable')
    if _REQUIRE_CUDA and not torch.cuda.is_available():
        raise AssertionError('CUDA required but unavailable; CPU fallback forbidden')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, base_cells = _fixture_cells(seed=7406)
    cells = []
    for index in range(32):
        cell = copy.deepcopy(base_cells[2])
        cell.cell_id = index
        cells.append(cell)
    probe = a4.FullFidelityA4GenomeAdapter(a4.GPU068A4Config(
        max_cells=32, max_sequences=32 * 4,
        max_symbols=32 * 2048,
    )).pack_cells(cells)
    expected = int(np.sum(probe.symbols[:probe.symbol_count], dtype=np.int64))
    resident = probe.to_torch(device=device)
    requested_type = torch.device(device).type
    if any(getattr(resident, name).device.type != requested_type
           for name in a4._ARRAY_FIELDS):
        raise AssertionError('not every array is on the requested device')
    pointers_before = resident.data_ptrs()
    warm = resident.resident_symbol_checksum()
    if device == 'cuda':
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = warm
    for _ in range(10):
        result = resident.resident_symbol_checksum()
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    pointers_after = resident.data_ptrs()
    if pointers_before != pointers_after:
        raise AssertionError('resident input storage was reallocated')
    # One explicit final readback phase after the resident loop: verify both
    # the reduction output and every packed array.
    resident_checksum = int(result.detach().item())
    back = resident.to_numpy()
    v3.assert_recursive_close(
        probe, back, atol=0.0, rtol=0.0, path='resident_cuda_readback',
    )
    if resident_checksum != expected:
        raise AssertionError('resident checksum differs from host source')
    return '%s canonical dtypes (float fields fp64), 10 resident reads, stable pointers, %.6fs (no speed claim)' % (
        device, elapsed,
    )


def test_a4_gene_decode_frozen_oracle():
    world, cell, config = _gene_decode_fixture()
    before = v3.pickle_clone(world.state_dict())
    packed = a4.FullFidelityA4GenomeAdapter(config).pack_cells([cell])
    cache = a4.decode_a4_gene_cache_numpy(packed)
    if int(cache.entry_count) != 3:
        raise AssertionError('unique gene count differs from frozen cache')
    expected_order = [40116138652, 55715248503, 68679218097]
    if list(cache.fingerprints[:3]) != expected_order:
        raise AssertionError('first-occurrence fingerprint order differs')
    if list(cache.copy_numbers[:3]) != [2, 3, 1]:
        raise AssertionError('duplicate copy counts differ')
    if list(cache.starts[:3]) != [19, 35, 19]:
        raise AssertionError('first record starts differ')
    materialized = cache.materialize_gene_specs_host()[0]
    _assert_gene_specs_equal(cell.gene_specs, materialized, 'gene_specs')
    if 9105062057 in materialized:
        raise AssertionError('replication template was decoded as a genome')
    if int(materialized[40116138652]['copy_number']) != 2:
        raise AssertionError('template duplicate changed complete-genome count')
    grammar = materialized[68679218097]
    if not a4.s65.is_grammar_spec(grammar):
        raise AssertionError('0.6.5 grammar payload was not preserved')
    if tuple(grammar['payload'][8:10]) != (1, 6):
        raise AssertionError('grammar module/value payload was truncated')
    for fingerprint, spec in materialized.items():
        if spec['role'] == a4.g2.ROLE_GENERIC:
            if 'reaction' not in spec or 'reaction_name' not in spec:
                raise AssertionError('generic reaction fields are missing')
        elif 'reaction' in spec or 'reaction_name' in spec:
            raise AssertionError('non-generic cache fabricated reaction fields')
    v3.assert_recursive_close(
        before, world.state_dict(), atol=0.0, rtol=0.0,
        path='gene_decode_source',
    )
    return '6 accepted -> 3 unique; overlap/recovery/order/template exclusion exact'


def test_a4_gene_decode_numpy_torch_devices():
    if torch is None:
        raise AssertionError('PyTorch is required for A4.2')
    if _REQUIRE_CUDA and not torch.cuda.is_available():
        raise AssertionError('CUDA required but unavailable; CPU fallback forbidden')
    _, cell, config = _gene_decode_fixture(seed=7602)
    packed = a4.FullFidelityA4GenomeAdapter(config).pack_cells([cell])
    expected = a4.decode_a4_gene_cache_numpy(packed)
    expected_specs = expected.materialize_gene_specs_host()

    source = '\n'.join(inspect.getsource(item) for item in (
        a4.decode_a4_gene_cache_torch,
        a4._validate_resident_ragged_metadata,
        a4._gene_decode_dimensions,
        a4._validate_gene_backend_and_dtypes,
    ))
    forbidden = ('.item(', '.cpu(', '.numpy(', '.tolist(',
                 'nonzero(', 'masked_select(', 'unique(')
    found = [token for token in forbidden if token in source]
    if found:
        raise AssertionError('resident decoder contains host/dynamic op: %s' % found)

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    details = []
    for device in devices:
        resident = packed.to_torch(device=device)
        pointers_before = resident.data_ptrs()
        cache = a4.decode_a4_gene_cache_torch(resident)
        if any(getattr(cache, name).device.type != device
               for name in a4._GENE_ARRAY_FIELDS):
            raise AssertionError('%s cache escaped requested device' % device)
        if pointers_before != resident.data_ptrs():
            raise AssertionError('%s decode reallocated source arena' % device)
        if device == 'cuda':
            torch.cuda.synchronize()
        back = cache.to_numpy()
        v3.assert_recursive_close(
            expected.state_dict(), back.state_dict(),
            atol=0.0, rtol=0.0, path='gene_cache.%s' % device,
        )
        actual_specs = back.materialize_gene_specs_host()
        for ci, (left, right) in enumerate(zip(expected_specs, actual_specs)):
            _assert_gene_specs_equal(left, right, '%s.cell[%d]' % (device, ci))
        v3.assert_recursive_close(
            packed.state_dict(), resident.to_numpy().state_dict(),
            atol=0.0, rtol=0.0, path='resident_source.%s' % device,
        )
        details.append(device)
    return 'NumPy/Torch %s fixed-shape exact; no decoder D2H/dynamic op' % '/'.join(details)


def test_a4_gene_decode_capacity_exact_and_plus_one():
    _assert_raises(
        ValueError,
        lambda: a4.GPU068A4Config(
            max_cells=a4.MAX_CELL_CAPACITY_FOR_FINGERPRINT_KEY + 1,
        ),
    )
    world = v3.make_world(seed=7603, cells=1)
    cell = world.cells[0]
    genes = [a4.g2.make_gene(
        role, parameter=role, regulator=role,
        promoter=role, efficiency=7 - role,
    ) for role in range(6)]
    cell.genomes = [np.concatenate(genes).astype(np.uint8)]
    cell.genome_lesions = [0.0]
    cell.replication_template = None
    cell.replication_copy = []
    cell.replication_template_lesion = 0.0
    cell.replication_fractional = 0.0
    cell._refresh_gene_cache()
    config = a4.GPU068A4Config(
        max_cells=1, max_sequences=1, max_symbols=6 * a4.GENE_SPAN,
        max_sequence_symbols=6 * a4.GENE_SPAN,
    )
    before = v3.pickle_clone(world.state_dict())
    packed = a4.FullFidelityA4GenomeAdapter(config).pack_cells([cell])
    decoded = a4.decode_a4_gene_cache_numpy(packed)
    if decoded.entry_capacity != 6 or int(decoded.entry_count) != 6:
        raise AssertionError('exact derived entry capacity did not pass')
    resident = packed.to_torch(device='cuda' if torch.cuda.is_available() else 'cpu')
    resident_cache = a4.decode_a4_gene_cache_torch(resident).to_numpy()
    if resident_cache.entry_capacity != 6 or int(resident_cache.entry_count) != 6:
        raise AssertionError('Torch exact derived entry capacity did not pass')

    overflow = copy.deepcopy(cell)
    overflow.genomes[0] = np.concatenate((
        overflow.genomes[0],
        a4.g2.make_gene(a4.g2.ROLE_GENERIC, parameter=7, spare=(7, 7, 7, 7)),
    )).astype(np.uint8)
    overflow._refresh_gene_cache()
    _assert_raises(
        a4.A4CapacityError,
        lambda: a4.FullFidelityA4GenomeAdapter(config).pack_cells([overflow]),
    )
    v3.assert_recursive_close(
        before, world.state_dict(), atol=0.0, rtol=0.0,
        path='gene_capacity_source',
    )

    empty = copy.deepcopy(cell)
    empty.genomes = []
    empty.genome_lesions = []
    empty._refresh_gene_cache()
    empty_config = a4.GPU068A4Config(
        max_cells=1, max_sequences=1, max_symbols=1,
        max_sequence_symbols=15,
    )
    empty_batch = a4.FullFidelityA4GenomeAdapter(empty_config).pack_cells([empty])
    empty_cache = a4.decode_a4_gene_cache_numpy(empty_batch)
    if empty_cache.entry_capacity != 0 or int(empty_cache.entry_count) != 0:
        raise AssertionError('zero-gene capacity is not represented exactly')
    return 'key bound + derived capacity 6 exact / seventh fails before pack / zero exact'


def test_a4_gene_cache_schema_and_decode_nonmutation():
    world, cell, config = _gene_decode_fixture(seed=7604)
    world_before = v3.pickle_clone(world.state_dict())
    packed = a4.FullFidelityA4GenomeAdapter(config).pack_cells([cell])
    packed_before = packed.state_dict()
    cache = a4.decode_a4_gene_cache_numpy(packed)
    v3.assert_recursive_close(
        packed_before, packed.state_dict(), atol=0.0, rtol=0.0,
        path='numpy_decode_input',
    )
    v3.assert_recursive_close(
        world_before, world.state_dict(), atol=0.0, rtol=0.0,
        path='numpy_decode_cells',
    )

    corruptions = []
    def add(name, mutate):
        item = cache.clone()
        mutate(item)
        corruptions.append((name, item))
    add('mask', lambda x: x.entry_mask.__setitem__(0, False))
    add('offset', lambda x: x.cell_entry_offsets.__setitem__(1, 0))
    add('fingerprint', lambda x: x.fingerprints.__setitem__(0, 0))
    add('payload', lambda x: x.payloads.__setitem__((0, 0), a4.ALPHABET_SIZE))
    add('copy_number', lambda x: x.copy_numbers.__setitem__(0, 0))
    add('unused_tail', lambda x: x.starts.__setitem__(int(x.entry_count), 0))
    add('frozen_length_limit', lambda x: setattr(
        x, 'max_sequence_symbols', a4.MAX_FROZEN_GENOME_SYMBOLS + 1,
    ))
    for name, item in corruptions:
        _assert_raises(
            a4.A4SchemaError,
            lambda item=item: a4.validate_a4_gene_cache(item),
        )
    resident = packed.to_torch(device='cpu')
    for value in (0, a4.MAX_FROZEN_GENOME_SYMBOLS + 1):
        malformed = resident.clone()
        malformed.max_sequence_symbols = value
        _assert_raises(
            a4.A4SchemaError,
            lambda malformed=malformed: a4.decode_a4_gene_cache_torch(malformed),
        )
    return '%d cache + 2 resident metadata corruptions rejected; source/cells unchanged' % len(corruptions)


def test_a4_paid_translation_cpu_oracle_ledger_and_order():
    world, cells, config, dt = _paid_translation_fixture()
    world_before = v3.pickle_clone(world.state_dict())
    gene_orders = [list(cell.gene_specs.keys()) for cell in cells]
    ragged = a4.FullFidelityA4GenomeAdapter(config).pack_cells(cells)
    state = a4.pack_a4_translation_state(cells, ragged, world.config, config)
    v3.assert_recursive_close(
        float(state.neural_attachment_osmolyte[0]), 29.3,
        atol=4e-15, rtol=0.0, path='neural_attachment_osmolyte',
    )
    state_before = state.state_dict()
    plan = a4.paid_translation_plan_numpy(
        a4.bind_a4_translation(ragged, state), dt,
    )
    cpu_cells = copy.deepcopy(cells)
    for cell in cpu_cells:
        cell.translate(dt, world.config)
    _assert_translation_matches_cells(plan, cpu_cells, 'cpu_oracle')

    for ci, (before, after) in enumerate(zip(cells, cpu_cells)):
        if list(after.gene_specs.keys()) != gene_orders[ci]:
            raise AssertionError('translation changed gene-cache order')
        total = float(after.last_translation) * dt
        for pool, coefficient in (
            (a4.a3.POOL_FUEL, 0.64),
            (a4.a3.POOL_MINERAL, 0.36),
            (a4.a3.POOL_ATP, 0.52),
        ):
            v3.assert_recursive_close(
                float(before.pools[pool] - after.pools[pool]),
                coefficient * total, atol=3e-12, rtol=0.0,
                path='ledger.cell[%d].pool[%d]' % (ci, pool),
            )
        v3.assert_recursive_close(
            float(after.pools[a4.a3.POOL_CATALYST]),
            float(sum(after.proteins.values())), atol=2e-12, rtol=0.0,
            path='ledger.cell[%d].active' % ci,
        )
        v3.assert_recursive_close(
            float(after.pools[a4.a3.POOL_DAMAGED_PROTEIN]),
            float(sum(after.damaged_proteins.values())),
            atol=2e-12, rtol=0.0,
            path='ledger.cell[%d].damaged' % ci,
        )
    if cpu_cells[1].last_translation != 0.0:
        raise AssertionError('ATP reserve produced unpaid protein')
    if cpu_cells[1].last_quiescence == cells[1].last_quiescence:
        raise AssertionError('ATP reserve was mistaken for an early return')
    if not (cpu_cells[2].last_translation > 0.0
            and cpu_cells[2].pools[a4.a3.POOL_MINERAL] <= 2e-16):
        raise AssertionError('gene-order mineral exhaustion was not exercised')
    if (cpu_cells[3].last_translation != 0.0
            or cpu_cells[3].last_quiescence != cells[3].last_quiescence):
        raise AssertionError('no-translator early-return observables differ')
    v3.assert_recursive_close(
        state_before, state.state_dict(), atol=0.0, rtol=0.0,
        path='translation_numpy_input',
    )
    v3.assert_recursive_close(
        world_before, world.state_dict(), atol=0.0, rtol=0.0,
        path='translation_numpy_world',
    )
    return '4 Formal066 cells exact; ATP reserve/order/ledgers/dict order fixed'


def test_a4_paid_translation_numpy_torch_devices_nonmutation():
    if torch is None:
        raise AssertionError('PyTorch is required for A4.3')
    if _REQUIRE_CUDA and not torch.cuda.is_available():
        raise AssertionError('CUDA required but unavailable; CPU fallback forbidden')
    world, cells, config, dt = _paid_translation_fixture(seed=7702)
    ragged = a4.FullFidelityA4GenomeAdapter(config).pack_cells(cells)
    state = a4.pack_a4_translation_state(cells, ragged, world.config, config)
    expected = a4.paid_translation_plan_numpy(
        a4.bind_a4_translation(ragged, state), dt,
    )
    source = '\n'.join(inspect.getsource(item) for item in (
        a4.paid_translation_plan_torch,
        a4._torch_ordered_row_sum,
        a4._validate_translation_resident_metadata,
        a4._validate_translation_backend_and_dtypes,
    ))
    forbidden = ('.item(', '.cpu(', '.numpy(', '.tolist(',
                 'nonzero(', 'masked_select(', 'unique(')
    found = [token for token in forbidden if token in source]
    if found:
        raise AssertionError('resident translation contains host/dynamic op: %s' % found)

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    for device in devices:
        resident_ragged = ragged.to_torch(device=device)
        resident_state = state.to_torch(device=device)
        ragged_ptrs = resident_ragged.data_ptrs()
        state_ptrs = resident_state.data_ptrs()
        binding = a4.bind_a4_translation(resident_ragged, resident_state)
        cache_ptrs = binding.cache.data_ptrs()
        plan = a4.paid_translation_plan_torch(binding, dt)
        if any(getattr(plan, name).device.type != device
               for name in a4._TRANSLATION_ARRAY_FIELDS):
            raise AssertionError('%s translation output escaped device' % device)
        if device == 'cuda':
            torch.cuda.synchronize()
        back = plan.to_numpy()
        v3.assert_recursive_close(
            expected.state_dict(), back.state_dict(),
            atol=2e-12, rtol=0.0, path='translation.%s' % device,
        )
        if ragged_ptrs != resident_ragged.data_ptrs():
            raise AssertionError('%s translation reallocated ragged input' % device)
        if state_ptrs != resident_state.data_ptrs():
            raise AssertionError('%s translation reallocated physiology input' % device)
        if cache_ptrs != binding.cache.data_ptrs():
            raise AssertionError('%s translation reallocated decoded cache' % device)
        v3.assert_recursive_close(
            ragged.state_dict(), resident_ragged.to_numpy().state_dict(),
            atol=0.0, rtol=0.0, path='translation_ragged_source.%s' % device,
        )
        v3.assert_recursive_close(
            state.state_dict(), resident_state.to_numpy().state_dict(),
            atol=0.0, rtol=0.0, path='translation_state_source.%s' % device,
        )

    # External translator raises capacity but must not alter the endogenous
    # translator-protein need used while weights are built.
    external_config = copy.deepcopy(world.config)
    external_config.external_translator = True
    external_state = a4.pack_a4_translation_state(
        cells, ragged, external_config, config,
    )
    external_expected = a4.paid_translation_plan_numpy(
        a4.bind_a4_translation(ragged, external_state), dt,
    )
    external_cpu = copy.deepcopy(cells)
    for cell in external_cpu:
        cell.translate(dt, external_config)
    _assert_translation_matches_cells(
        external_expected, external_cpu, 'external_translator_cpu',
    )
    external_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    external_plan = a4.paid_translation_plan_torch(
        a4.bind_a4_translation(
            ragged.to_torch(external_device),
            external_state.to_torch(external_device),
        ), dt,
    ).to_numpy()
    v3.assert_recursive_close(
        external_expected.state_dict(), external_plan.state_dict(),
        atol=2e-12, rtol=0.0, path='external_translator.%s' % external_device,
    )

    # Resource exhaustion at a tiny positive budget is a sensitive fp64
    # boundary: the resident plan must preserve the literal per-gene payment
    # recurrence without producing a schema-invalid negative pool.
    tiny_cells = copy.deepcopy(cells)
    tiny_cells[0].pools[a4.a3.POOL_FUEL] = 0.64e-12
    tiny_cells[0].pools[a4.a3.POOL_MINERAL] = 0.36e-12
    tiny_cells[0].pools[a4.a3.POOL_ATP] = 0.042 + 0.52e-12
    tiny_ragged = a4.FullFidelityA4GenomeAdapter(config).pack_cells(tiny_cells)
    tiny_state = a4.pack_a4_translation_state(
        tiny_cells, tiny_ragged, external_config, config,
    )
    tiny_dt = 1e-8
    tiny_expected = a4.paid_translation_plan_numpy(
        a4.bind_a4_translation(tiny_ragged, tiny_state), tiny_dt,
    )
    tiny_cpu = copy.deepcopy(tiny_cells)
    for cell in tiny_cpu:
        cell.translate(tiny_dt, external_config)
    _assert_translation_matches_cells(
        tiny_expected, tiny_cpu, 'tiny_resource_cpu', atol=2e-12,
    )
    tiny_plan = a4.paid_translation_plan_torch(
        a4.bind_a4_translation(
            tiny_ragged.to_torch(external_device),
            tiny_state.to_torch(external_device),
        ), tiny_dt,
    ).to_numpy()
    v3.assert_recursive_close(
        tiny_expected.state_dict(), tiny_plan.state_dict(),
        atol=2e-12, rtol=0.0, path='tiny_resource.%s' % external_device,
    )

    # This exact fuel boundary makes the frozen Python recurrence retain a
    # signed fp64 residual.  Preserve it as measured evidence for the narrow
    # three-paid-pool ledger tolerance; do not hide it with a general clip.
    residual_cells = copy.deepcopy(cells)
    residual_cells[0].pools[a4.a3.POOL_FUEL] = 9.936293309191388e-08
    residual_ragged = a4.FullFidelityA4GenomeAdapter(config).pack_cells(
        residual_cells,
    )
    residual_state = a4.pack_a4_translation_state(
        residual_cells, residual_ragged, world.config, config,
    )
    residual_dt = 0.001
    residual_expected = a4.paid_translation_plan_numpy(
        a4.bind_a4_translation(residual_ragged, residual_state), residual_dt,
    )
    residual_cpu = copy.deepcopy(residual_cells)
    for cell in residual_cpu:
        cell.translate(residual_dt, world.config)
    _assert_translation_matches_cells(
        residual_expected, residual_cpu, 'signed_payment_residual',
        atol=2e-12,
    )
    observed_residual = float(
        residual_cpu[0].pools[a4.a3.POOL_FUEL]
    )
    if not (-a4.TRANSLATION_LEDGER_ATOL <= observed_residual < 0.0):
        raise AssertionError(
            'fixture did not exercise a signed fp64 payment residual'
        )
    residual_plan = a4.paid_translation_plan_torch(
        a4.bind_a4_translation(
            residual_ragged.to_torch(external_device),
            residual_state.to_torch(external_device),
        ), residual_dt,
    ).to_numpy()
    v3.assert_recursive_close(
        residual_expected.state_dict(), residual_plan.state_dict(),
        atol=2e-12, rtol=0.0,
        path='signed_payment_residual.%s' % external_device,
    )
    return 'NumPy/Torch %s fp64 + external translator parity; fixed output and stable inputs' % '/'.join(devices)


def test_a4_paid_translation_full_mro_signals_and_behavioural_quiescence():
    world, cells, config, dt = _paid_translation_fixture(seed=7703)
    localisations = {
        int(spec['localisation'])
        for spec in cells[0].gene_specs.values()
        if int(spec['role']) == int(a4.a3.ROLE_REGULATOR)
    }
    expected_localisations = {
        int(a4.s4.LOC_REPAIR), int(a4.s4.LOC_SENSOR),
        int(a4.s4.LOC_EFFECTOR), int(a4.s5.LOC_ECOLOGY),
    }
    if localisations != expected_localisations:
        raise AssertionError('fixture does not cover the full regulator MRO')

    model_config = copy.deepcopy(world.config)
    model_config.quiescence = False
    model_config.quiescence_effector = True
    cells[0].behavioural_quiescence = 0.47
    ragged = a4.FullFidelityA4GenomeAdapter(config).pack_cells(cells)
    state = a4.pack_a4_translation_state(
        cells, ragged, model_config, config,
    )
    plan = a4.paid_translation_plan_numpy(
        a4.bind_a4_translation(ragged, state), dt,
    )
    cpu_cells = copy.deepcopy(cells)
    for cell in cpu_cells:
        cell.translate(dt, model_config)
    _assert_translation_matches_cells(plan, cpu_cells, 'full_mro')
    v3.assert_recursive_close(
        float(plan.last_quiescence[0]), 0.47,
        atol=2e-12, rtol=0.0, path='behavioural_quiescence',
    )

    zero_signal_cells = copy.deepcopy(cells)
    for cell in zero_signal_cells:
        cell.last_receptor_activity[:] = 0.0
        cell.last_control_vectors[:] = 0.0
        cell.last_control_scalars[:] = 0.0
        cell.last_edna_signal = 0.0
        cell.last_corpse_signal = 0.0
        cell.last_necrotoxin_signal = 0.0
        cell.behavioural_quiescence = 0.0
    zero_ragged = a4.FullFidelityA4GenomeAdapter(config).pack_cells(zero_signal_cells)
    zero_state = a4.pack_a4_translation_state(
        zero_signal_cells, zero_ragged, model_config, config,
    )
    zero_plan = a4.paid_translation_plan_numpy(
        a4.bind_a4_translation(zero_ragged, zero_state), dt,
    )
    if np.allclose(plan.active_mass, zero_plan.active_mass, atol=1e-15, rtol=0.0):
        raise AssertionError('sensor/effector/ecology snapshots did not affect weights')
    return 'repair/sensor/effector/ecology MRO + behavioural quiescence exact'


def test_a4_paid_translation_early_gate_capacity_and_schema():
    world, cells, config, dt = _paid_translation_fixture(seed=7704)
    disabled_config = copy.deepcopy(world.config)
    disabled_config.gene_expression = False
    ragged = a4.FullFidelityA4GenomeAdapter(config).pack_cells(cells)
    state = a4.pack_a4_translation_state(
        cells, ragged, disabled_config, config,
    )
    plan = a4.paid_translation_plan_numpy(
        a4.bind_a4_translation(ragged, state), dt,
    )
    cpu_cells = copy.deepcopy(cells)
    for cell in cpu_cells:
        cell.translate(dt, disabled_config)
    _assert_translation_matches_cells(plan, cpu_cells, 'gene_expression_disabled')
    if np.any(plan.last_translation[:plan.cell_count] != 0.0):
        raise AssertionError('disabled translation did not reset last_translation')
    if not np.array_equal(plan.last_quiescence, state.last_quiescence):
        raise AssertionError('disabled translation changed last_quiescence')

    cache = a4.decode_a4_gene_cache_numpy(ragged)
    specs = cache.materialize_gene_specs_host()
    required = 0
    for cell, cell_specs in zip(cells, specs):
        gene_keys = set(cell_specs.keys())
        required = max(
            required,
            len(set(cell.proteins.keys()).union(gene_keys)),
            len(set(cell.damaged_proteins.keys()).union(gene_keys)),
        )
    exact_config = a4.GPU068A4Config(
        max_cells=config.max_cells, max_sequences=config.max_sequences,
        max_symbols=config.max_symbols,
        max_sequence_symbols=config.max_sequence_symbols,
        max_proteins_per_cell=required,
    )
    exact = a4.pack_a4_translation_state(
        cells, ragged, world.config, exact_config,
    )
    if exact.protein_capacity != required:
        raise AssertionError('exact protein capacity did not remain exact')
    smaller = a4.GPU068A4Config(
        max_cells=config.max_cells, max_sequences=config.max_sequences,
        max_symbols=config.max_symbols,
        max_sequence_symbols=config.max_sequence_symbols,
        max_proteins_per_cell=required - 1,
    )
    _assert_raises(
        a4.A4CapacityError,
        lambda: a4.pack_a4_translation_state(cells, ragged, world.config, smaller),
    )

    corruptions = []
    def add(name, mutate):
        item = state.clone()
        mutate(item)
        corruptions.append((name, item))
    add('negative_pool', lambda x: x.pools.__setitem__((0, 0), -1.0))
    add('active_count', lambda x: x.active_count.__setitem__(0, x.protein_capacity + 1))
    add('float32', lambda x: setattr(x, 'pools', x.pools.astype(np.float32)))
    add('fingerprint_tail', lambda x: x.active_fingerprints.__setitem__(
        (0, int(x.active_count[0])), 0,
    ))
    for _, item in corruptions:
        _assert_raises(a4.A4SchemaError,
                       lambda item=item: a4.validate_a4_translation_state(item))
    signed_roundoff = state.clone()
    signed_roundoff.pools[0, a4.a3.POOL_FUEL] = (
        -0.5 * a4.TRANSLATION_LEDGER_ATOL
    )
    a4.validate_a4_translation_state(signed_roundoff)
    beyond_roundoff = state.clone()
    beyond_roundoff.pools[0, a4.a3.POOL_FUEL] = (
        -2.0 * a4.TRANSLATION_LEDGER_ATOL
    )
    _assert_raises(
        a4.A4SchemaError,
        lambda: a4.validate_a4_translation_state(beyond_roundoff),
    )
    nonpayment_negative = state.clone()
    nonpayment_negative.pools[0, a4.a3.POOL_WASTE] = (
        -0.5 * a4.TRANSLATION_LEDGER_ATOL
    )
    _assert_raises(
        a4.A4SchemaError,
        lambda: a4.validate_a4_translation_state(nonpayment_negative),
    )
    mismatched = state.clone()
    mismatched.cell_ids[:2] = mismatched.cell_ids[1::-1]
    _assert_raises(
        a4.A4SchemaError,
        lambda: a4.bind_a4_translation(ragged, mismatched),
    )
    foreign = ragged.clone()
    foreign.symbols[0] = (int(foreign.symbols[0]) + 1) % a4.ALPHABET_SIZE
    _assert_raises(
        a4.A4SchemaError,
        lambda: a4.bind_a4_translation(foreign, state),
    )
    resident_foreign = foreign.to_torch('cuda' if torch.cuda.is_available() else 'cpu')
    resident_state = state.to_torch(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    _assert_raises(
        a4.A4SchemaError,
        lambda: a4.bind_a4_translation(resident_foreign, resident_state),
    )
    resident_ragged = ragged.to_torch(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    trusted_ragged_clone = resident_ragged.clone()
    trusted_state_clone = resident_state.clone()
    a4.bind_a4_translation(trusted_ragged_clone, trusted_state_clone)
    trusted_state_clone.pools[0, a4.a3.POOL_FUEL] += 1e-9
    _assert_raises(
        a4.A4SchemaError,
        lambda: a4.bind_a4_translation(
            trusted_ragged_clone, trusted_state_clone,
        ),
    )
    _assert_raises(
        a4.A4SchemaError,
        lambda: trusted_state_clone.clone(),
    )
    trusted_ragged_clone.symbols[0] = torch.remainder(
        trusted_ragged_clone.symbols[0].to(torch.int64) + 1,
        a4.ALPHABET_SIZE,
    ).to(torch.uint8)
    _assert_raises(
        a4.A4SchemaError,
        lambda: trusted_ragged_clone.clone(),
    )
    cache = a4.decode_a4_gene_cache_numpy(ragged)
    _assert_raises(
        a4.A4SchemaError,
        lambda: a4.A4TranslationBinding(
            ragged=ragged, cache=cache, state=state,
        ),
    )
    forged = object.__new__(a4.A4TranslationBinding)
    object.__setattr__(forged, 'ragged', ragged)
    object.__setattr__(forged, 'cache', cache)
    object.__setattr__(forged, 'state', state)
    object.__setattr__(forged, '_token', object())
    _assert_raises(
        a4.A4SchemaError,
        lambda: a4.paid_translation_plan_numpy(forged, dt),
    )
    changed_host_cache = a4.bind_a4_translation(ragged, state)
    changed_host_cache.cache.copy_numbers[0] += 1
    _assert_raises(
        a4.A4SchemaError,
        lambda: a4.paid_translation_plan_numpy(changed_host_cache, dt),
    )
    attest_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    changed_resident_cache = a4.bind_a4_translation(
        ragged.to_torch(attest_device), state.to_torch(attest_device),
    )
    changed_resident_cache.cache.copy_numbers[0] += 1
    _assert_raises(
        a4.A4SchemaError,
        lambda: a4.paid_translation_plan_torch(
            changed_resident_cache, dt,
        ),
    )
    if a4.FULL_GPU_WORLD_STEP is not False:
        raise AssertionError('A4.3 must not claim a full GPU world-step')
    if a4.PORT_STATUS['genome_replication'] != 'cpu-authoritative-next-a4-slice':
        raise AssertionError('translation slice changed replication authority')
    return 'early gate + exact P=%d/P+1 atomic + %d schema corruptions rejected' % (
        required, len(corruptions),
    )


def test_a3_focused_regression():
    # Invoke named tests only; never call A3 run_all or overwrite A3 evidence.
    for fn in (
        v3.test_packed_state_roundtrip_and_invariants,
        v3.test_transient_genome_lesion_count_preserved_until_damage,
        v3.test_world_one_step_lockstep,
        v3.test_clone_deterministic_full_state,
        v3.test_save_restore_deterministic_full_state,
    ):
        fn()
    return '5 promoted-A3 focused regressions PASS'


TESTS = (
    test_api_scope,
    test_source_hash_inputs_present,
    test_ragged_batch_roundtrip_lossless,
    test_numpy_torch_fp64_roundtrip_and_deep_clone,
    test_ragged_schema_rejects_corruption_atomically,
    test_capacity_exact_and_plus_one_atomic,
    test_cell_identity_order_and_duplicate_rejected,
    test_explicit_device_residency_checksum_smoke,
    test_a4_gene_decode_frozen_oracle,
    test_a4_gene_decode_numpy_torch_devices,
    test_a4_gene_decode_capacity_exact_and_plus_one,
    test_a4_gene_cache_schema_and_decode_nonmutation,
    test_a4_paid_translation_cpu_oracle_ledger_and_order,
    test_a4_paid_translation_numpy_torch_devices_nonmutation,
    test_a4_paid_translation_full_mro_signals_and_behavioural_quiescence,
    test_a4_paid_translation_early_gate_capacity_and_schema,
    test_a3_focused_regression,
)


def run_all(write=False, output_dir=None):
    rows = []
    started = time.time()
    for fn in TESTS:
        then = time.perf_counter()
        try:
            detail = fn()
            status = 'PASS'
            error = ''
        except Exception as exc:
            detail = ''
            status = 'FAIL'
            error = '%s: %s' % (type(exc).__name__, exc)
        row = {
            'test': fn.__name__, 'status': status,
            'detail': detail, 'error': error,
            'seconds': time.perf_counter() - then,
        }
        rows.append(row)
        print('%s: %s - %s' % (status, fn.__name__, detail or error))
    passed = sum(row['status'] == 'PASS' for row in rows)
    failed = sum(row['status'] == 'FAIL' for row in rows)
    elapsed = time.time() - started
    payload = {
        'build': a4.BUILD,
        'schema': {
            'ragged_genome': a4.SCHEMA_VERSION,
            'gene_cache': a4.GENE_CACHE_SCHEMA_VERSION,
            'translation_state': a4.TRANSLATION_SCHEMA_VERSION,
        },
        'development_slice': 'A4.3-resident-paid-translation-plan',
        'promoted_baseline_unchanged': 'SOMA-CELL 0.6.8-GPU A3',
        'full_gpu_world_step': False,
        'require_cuda': bool(_REQUIRE_CUDA),
        'source_sha256_map': source_sha256_map(),
        'passed': passed, 'failed': failed, 'total': len(rows),
        'elapsed_seconds': elapsed, 'rows': rows,
    }
    if write:
        output_dir = os.path.abspath(output_dir or HERE)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, RESULT_JSON), 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write('\n')
        with open(os.path.join(output_dir, RESULT_CSV), 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(
                handle, fieldnames=('test', 'status', 'detail', 'error', 'seconds'),
            )
            writer.writeheader()
            writer.writerows(rows)
        lines = [
            'SOMA-CELL 0.6.8-GPU A4.3 VALIDATION',
            '%d PASS / %d FAIL / %d TOTAL' % (passed, failed, len(rows)),
            'elapsed %.6fs' % elapsed,
            '',
        ]
        lines.extend('%s: %s - %s' % (
            row['status'], row['test'], row['detail'] or row['error'],
        ) for row in rows)
        with open(os.path.join(output_dir, RESULT_TXT), 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(lines) + '\n')
    return payload


def main(argv=None):
    global _REQUIRE_CUDA
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--output-dir')
    args = parser.parse_args(argv)
    _REQUIRE_CUDA = bool(args.require_cuda)
    payload = run_all(write=args.write, output_dir=args.output_dir)
    return 0 if payload['failed'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
