# coding: utf-8
"""Focused validation for SOMA-CELL 0.6.8-GPU A4.1 through A4.8c2 slices."""
from __future__ import division

import argparse
import copy
import csv
import hashlib
import inspect
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import SOMA_CELL_0_6_8_gpu_a4 as a4
import SOMA_CELL_0_6_8_gpu_a4_replication as a44
import SOMA_CELL_0_6_8_gpu_a4_hydrolysis as a47
import SOMA_CELL_0_6_8_gpu_a4_integration as a48
import SOMA_CELL_0_6_8_gpu_a4_translation_integration as a48b
import SOMA_CELL_0_6_8_gpu_a4_replication_integration as a48c
import SOMA_CELL_0_6_8_gpu_a4_replication_completion_integration as a48c2
import SOMA_CELL_0_6_8_gpu_a3_scheduler as a3s
import SOMA_CELL_0_6_8_A3_validation as v3

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

RESULT_JSON = 'SOMA_CELL_0_6_8_GPU_A4_8C2_VALIDATION_RESULTS.json'
RESULT_CSV = 'soma_cell_0_6_8_gpu_a4_8c2_validation.csv'
RESULT_TXT = 'SOMA_CELL_0_6_8_GPU_A4_8C2_VALIDATION_RESULTS.txt'

SOURCE_PATHS = (
    'src/0_6_8/SOMA_CELL_0_6_8_gpu_a4.py',
    'src/0_6_8/SOMA_CELL_0_6_8_gpu_a4_replication.py',
    'src/0_6_8/SOMA_CELL_0_6_8_gpu_a4_hydrolysis.py',
    'src/0_6_8/SOMA_CELL_0_6_8_gpu_a4_integration.py',
    'src/0_6_8/SOMA_CELL_0_6_8_gpu_a4_translation_integration.py',
    'src/0_6_8/SOMA_CELL_0_6_8_gpu_a4_replication_integration.py',
    'src/0_6_8/SOMA_CELL_0_6_8_gpu_a4_replication_completion_integration.py',
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

PROMOTED_A3_SHA256 = {
    'src/0_6_8/SOMA_CELL_0_6_8_gpu_a3.py': (
        '6ca66378702dabd5a388553ca92279376deae5ba8c39bea9bdbe3488aa41b4f6'
    ),
    'src/0_6_8/SOMA_CELL_0_6_8_gpu_a3_scheduler.py': (
        '5e207f68fd0d9aa674ab054dc43d62247e84c1f3c61fc6c7b455960ff84413aa'
    ),
}

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
    if isinstance(kind, tuple):
        label = '/'.join(item.__name__ for item in kind)
    else:
        label = kind.__name__
    raise AssertionError('expected %s' % label)


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


def _paid_replication_fixture(seed=7801):
    """Six active Formal066 templates spanning the A4.4a paid gates."""
    world = v3.make_world(seed=seed, cells=6)
    cells = world.cells
    for name, value in (
        ('genome_replication', True),
        ('mutation', False),
        ('proofreading', False),
        ('external_replicase', False),
        ('quiescence', False),
        ('quiescence_effector', False),
    ):
        setattr(world.config, name, value)

    near_one = np.nextafter(np.float64(1.0), np.float64(0.0))
    for ci, cell in enumerate(cells):
        genome = np.asarray(cell.genomes[0], dtype=np.uint8).copy()
        copy_length = 8 + ci
        if len(genome) <= copy_length + 2:
            raise AssertionError('replication fixture is too close to completion')
        cell.genomes = [genome.copy()]
        cell.genome_lesions = [0.025 * (ci + 1)]
        cell.replication_template = genome.copy()
        cell.replication_copy = [int(value) for value in genome[:copy_length]]
        cell.replication_template_lesion = 0.05 * (ci + 1)
        cell.replication_fractional = 0.125 if ci == 1 else near_one
        cell.last_replication_symbols = 71 + ci
        cell.last_effective_error_rate = 0.5 + 0.01 * ci
        cell.pools[a4.a3.POOL_NUCLEOTIDE] = 0.75
        cell.pools[a4.a3.POOL_ATP] = 0.75
        cell._refresh_gene_cache()
        cell._sync_protein_pool()
        cell._sync_damage_pool()
        if cell.role_activity(a4.g2.ROLE_REPLICASE) <= 1e-6:
            raise AssertionError('replication fixture lacks endogenous replicase')

    atp_gate = float(a4.g2.REPLICATION_ATP_PER_SYMBOL) + 0.022
    monomer_gate = float(a4.g2.MONOMER_MASS)
    cells[2].pools[a4.a3.POOL_ATP] = atp_gate
    cells[3].pools[a4.a3.POOL_ATP] = np.nextafter(atp_gate, 0.0)
    cells[4].pools[a4.a3.POOL_NUCLEOTIDE] = monomer_gate
    cells[5].pools[a4.a3.POOL_NUCLEOTIDE] = np.nextafter(
        monomer_gate, 0.0,
    )
    config = a4.GPU068A4Config(
        max_cells=6, max_sequences=18, max_symbols=16384,
        max_sequence_symbols=a4.MAX_FROZEN_GENOME_SYMBOLS,
        max_proteins_per_cell=64,
    )
    return world, cells, config, 1e-9


def _replication_model_config(
        base, proofreading=False, external_replicase=False,
        quiescence=False, quiescence_effector=False):
    config = copy.deepcopy(base)
    config.genome_replication = True
    config.mutation = False
    config.proofreading = bool(proofreading)
    config.external_replicase = bool(external_replicase)
    config.quiescence = bool(quiescence)
    config.quiescence_effector = bool(quiescence_effector)
    return config


def _cpu_raw_repair_activity(cell, kind):
    """Literal 0.4 grouping: (mass * efficiency) * promoter."""
    total = 0.0
    for fingerprint, amount in cell.proteins.items():
        spec = cell.gene_specs.get(fingerprint)
        if (
                spec is not None
                and int(spec['role']) == int(a4.a3.ROLE_REGULATOR)
                and int(spec['localisation']) == int(a4.s4.LOC_REPAIR)
                and int(spec['parameter']) % int(a4.a3.REPAIR_COUNT)
                == int(kind)):
            total += (
                float(amount) * float(spec['efficiency'])
            ) * float(spec['promoter'])
    inhibition = 1.0 / (1.0 + 2.6 * cell.aggregate_concentration())
    return float(total / 0.014 * inhibition)


def _paid_replication_b_fixture(seed=7901):
    """A4.4b proof, quiescence, error, and resource-boundary rows."""
    world, cells, config, dt = _paid_replication_fixture(seed=seed)
    world.config = _replication_model_config(
        world.config, proofreading=True, external_replicase=True,
        quiescence=True, quiescence_effector=True,
    )
    cumulative_before = (
        0.375, -0.0, float(2 ** 40), 17.0, 0.125, 2048.5,
    )
    for ci, cell in enumerate(cells):
        cell.pools[a4.a3.POOL_NUCLEOTIDE] = 0.75
        cell.pools[a4.a3.POOL_ATP] = 0.75
        cell.pools[a4.a3.POOL_REACTIVE] = 0.015 * (ci + 1)
        cell.replication_template_lesion = 0.035 + 0.09 * ci
        cell.behavioural_quiescence = 0.0
        cell.cumulative_proofreading_atp = cumulative_before[ci]

    # Complete-genome damage drives inherited quiescence, while the distinct
    # template lesion above drives replication error telemetry.
    cells[0].genome_lesions = [0.55]
    cells[0].membrane_oxidation[:] = 1.2
    cells[0].current_stress = 0.45
    cells[0].pools[a4.a3.POOL_AGGREGATE] = 0.18
    cells[0].pools[a4.a3.POOL_REACTIVE] = 0.16
    cells[0].behavioural_quiescence = 0.05
    cells[1].behavioural_quiescence = 0.70

    proof = _cpu_raw_repair_activity(
        cells[2], a4.a3.REPAIR_PROOFREADING,
    )
    if proof <= 0.0:
        raise AssertionError('A4.4b fixture lacks proofreading activity')
    proof_fraction = proof / (0.75 + proof)
    atp_per_symbol = (
        float(a4.g2.REPLICATION_ATP_PER_SYMBOL)
        + 0.00075 * proof_fraction
    )
    atp_gate = atp_per_symbol + 0.022
    # Keep the ordinary parity batch outside A4.4b's fp64 comparison guard.
    # Exact/nextafter proof-ATP boundaries are tested separately as explicit
    # CPU-authoritative scope exclusions.
    cells[2].pools[a4.a3.POOL_ATP] = atp_gate + 1e-10
    cells[3].pools[a4.a3.POOL_ATP] = atp_gate - 1e-10
    monomer_gate = float(a4.g2.MONOMER_MASS)
    cells[4].pools[a4.a3.POOL_NUCLEOTIDE] = monomer_gate
    cells[5].pools[a4.a3.POOL_NUCLEOTIDE] = np.nextafter(
        monomer_gate, 0.0,
    )
    return world, cells, config, dt


def _paid_completion_fixture(seed=8301):
    """Two active Formal066 rows completing a previously copied polymer."""
    world, source_cells, _, _ = _paid_replication_b_fixture(seed=seed)
    cells = [copy.deepcopy(source_cells[0]), copy.deepcopy(source_cells[1])]
    near_one = np.nextafter(np.float64(1.0), np.float64(0.0))
    for ci, cell in enumerate(cells):
        template = np.asarray(cell.replication_template, dtype=np.uint8).copy()
        partial = template[:-1].copy()
        # Preserve two substitutions made by an earlier replication call.
        # Completion must append to this paid partial copy, not clone template.
        for position in (5 + ci, len(partial) // 2 + ci):
            position %= len(partial)
            partial[position] = (int(partial[position]) + ci + 1) % a4.ALPHABET_SIZE
        cell.replication_copy = [int(value) for value in partial]
        cell.replication_fractional = near_one
        cell.replication_cycles = 13 + ci
        cell.mutation_events['substitution'] = 4 + ci
        cell.pools[a4.a3.POOL_NUCLEOTIDE] = 0.75
        cell.pools[a4.a3.POOL_ATP] = 0.75
    config = a4.GPU068A4Config(
        max_cells=2, max_sequences=8, max_symbols=8192,
        max_sequence_symbols=a4.MAX_FROZEN_GENOME_SYMBOLS,
        max_proteins_per_cell=64,
    )
    return world, cells, config, 1e-9


def _a46b1_completion_mutation_fixture(seed=9911):
    """Three completion rows exposing the complete frozen structural chain."""
    world, source_cells, _, _ = _paid_replication_b_fixture(seed=seed)
    # Exercise NumPy PCG64's cached uint32 path before byte-vector insertion
    # and padding draws.  The tape must preserve the complete state, not only
    # the 128-bit generator state.
    world.rng.integers(0, 7)
    if int(world.rng.bit_generator.state['has_uint32']) != 1:
        raise AssertionError('A4.6b1 fixture did not prime PCG64 uint32 cache')

    config = copy.deepcopy(world.config)
    config.mutation = True
    config.mutation_rate = 0.5
    config.structural_rate = 10.0
    config.variable_length = True
    config.gene_duplication = True
    world.config = config

    root = np.asarray(
        source_cells[0].replication_template, dtype=np.uint8,
    ).copy()
    if (len(root) != 576
            or int(a4.g2.MAX_GENOME_LENGTH) != 640
            or int(a4.g2.MIN_GENOME_LENGTH) != 32):
        raise AssertionError('A4.6b1 frozen genome limits drifted')
    sequences = (
        root.copy(),
        np.concatenate((root, root[:64])).astype(np.uint8),
        root[:8].copy(),
    )
    bases = (source_cells[0], source_cells[1], source_cells[0])
    nucleotide = (
        2.25 * float(a4.g2.MONOMER_MASS), 0.10, 0.10,
    )
    near_one = np.nextafter(np.float64(1.0), np.float64(0.0))
    cells = []
    for ci, (base, sequence) in enumerate(zip(bases, sequences)):
        cell = copy.deepcopy(base)
        cell.cell_id = int(source_cells[ci].cell_id)
        cell.genomes = [sequence.copy()]
        cell.genome_lesions = [0.0]
        cell.replication_template = sequence.copy()
        cell.replication_copy = [int(value) for value in sequence[:-1]]
        cell.replication_template_lesion = 0.0
        cell.replication_fractional = near_one
        cell.pools[a4.a3.POOL_REACTIVE] = 0.0
        cell.pools[a4.a3.POOL_ATP] = 0.75
        cell.pools[a4.a3.POOL_NUCLEOTIDE] = nucleotide[ci]
        cell._refresh_gene_cache()
        cell._sync_protein_pool()
        cells.append(cell)
    capacity = a4.GPU068A4Config(
        max_cells=4, max_sequences=12, max_symbols=16384,
        max_sequence_symbols=a4.MAX_FROZEN_GENOME_SYMBOLS,
        max_proteins_per_cell=64,
    )
    return world, cells, capacity, 1e-9


def _a46b1_replay_tape_row(tape, before, ci):
    """Independent fixed-order host application used only by validation."""
    alphabet = int(a4.g2.ALPHABET_SIZE)
    candidate = np.asarray(before.replication_copy, dtype=np.uint8).copy()
    template = np.asarray(before.replication_template, dtype=np.uint8)
    append_count = int(tape.append_count[ci])
    for rank in range(append_count):
        symbol = int(template[len(candidate)])
        if bool(tape.replacement_mask[ci, rank]):
            raw = int(tape.replacement_raw[ci, rank])
            symbol = (raw + (1 if raw >= symbol else 0)) % alphabet
        candidate = np.concatenate((
            candidate, np.asarray([symbol], dtype=np.uint8),
        ))
    if len(candidate) != int(tape.pre_structural_lengths[ci]):
        raise AssertionError('A4.6b1 replay pre-structural length differs')

    count = int(tape.insertion_count[ci])
    if count:
        position = int(tape.insertion_position[ci])
        inserted = np.asarray(
            tape.insertion_symbols[ci, :count], dtype=np.uint8,
        )
        candidate = np.concatenate((
            candidate[:position], inserted, candidate[position:],
        ))

    count = int(tape.deletion_count[ci])
    if count:
        position = int(tape.deletion_position[ci])
        candidate = np.concatenate((
            candidate[:position], candidate[position + count:],
        ))

    ordinal = int(tape.duplication_gene_ordinal[ci])
    if ordinal >= 0:
        genes = a4.g2.parse_genes(candidate)
        if ordinal >= len(genes):
            raise AssertionError('A4.6b1 replay duplication ordinal differs')
        start = int(genes[ordinal]['start'])
        if start != int(tape.duplication_source_start[ci]):
            raise AssertionError('A4.6b1 replay duplication start differs')
        fragment = candidate[start:start + int(a4.g2.GENE_SPAN)].copy()
        position = int(tape.duplication_position[ci])
        candidate = np.concatenate((
            candidate[:position], fragment, candidate[position:],
        ))

    left = int(tape.inversion_left[ci])
    right = int(tape.inversion_right[ci])
    if left >= 0:
        candidate[left:right] = candidate[left:right][::-1]

    count = int(tape.transposition_count[ci])
    if count:
        start = int(tape.transposition_start[ci])
        fragment = candidate[start:start + count].copy()
        remainder = np.concatenate((
            candidate[:start], candidate[start + count:],
        ))
        position = int(tape.transposition_position[ci])
        candidate = np.concatenate((
            remainder[:position], fragment, remainder[position:],
        ))

    count = int(tape.padding_count[ci])
    if count:
        candidate = np.concatenate((
            candidate,
            np.asarray(tape.padding_symbols[ci, :count], dtype=np.uint8),
        ))
    if len(candidate) > int(a4.g2.MAX_GENOME_LENGTH):
        candidate = candidate[:int(a4.g2.MAX_GENOME_LENGTH)]
    final_length = int(tape.post_structural_lengths[ci])
    if final_length > len(candidate):
        raise AssertionError('A4.6b1 replay final length exceeds candidate')
    candidate = candidate[:final_length]
    if (len(candidate) - int(tape.pre_structural_lengths[ci])
            != int(tape.material_delta_symbols[ci])):
        raise AssertionError('A4.6b1 replay material delta differs')
    return candidate.astype(np.uint8, copy=False)


def _a47_gene_sequence(role, start, length=48):
    """One isolated frozen gene whose selected deletion breaks decoding."""
    sequence = np.ones((int(length),), dtype=np.uint8)
    gene = a4.g2.make_gene(
        int(role), parameter=int(role), promoter=5, efficiency=4,
    )
    start = int(start)
    if start < 0 or start + len(gene) > len(sequence):
        raise AssertionError('A4.7a fixture gene does not fit its polymer')
    sequence[start:start + len(gene)] = gene
    return sequence


def _a47_hydrolysis_fixture(boundaries=False):
    """One Formal066 post-gain event with a primed PCG64 uint32 cache."""
    world = v3.make_world(seed=9971 if boundaries else 9972, cells=1)
    cell = world.cells[0]
    world.config.endogenous_damage = False
    minimum = int(a4.g2.MIN_GENOME_LENGTH)
    if boundaries:
        cell.genomes = [
            _a47_gene_sequence(a4.g2.ROLE_ENERGY, 0),
            _a47_gene_sequence(
                a4.g2.ROLE_MEMBRANE, 0, length=minimum,
            ),
            _a47_gene_sequence(a4.g2.ROLE_TRANSPORTER, 16),
        ]
        cell.genome_lesions = [0.75, 1.0, 1.0]
        dt = 0.0
    else:
        cell.genomes = [
            _a47_gene_sequence(a4.g2.ROLE_ENERGY, 0),
            _a47_gene_sequence(a4.g2.ROLE_MEMBRANE, 0),
            _a47_gene_sequence(a4.g2.ROLE_TRANSPORTER, 16),
        ]
        lesion = 0.5 / 0.00065
        if (0.00065 * lesion != 0.5
                or len(cell.genomes[0]) != 48):
            raise AssertionError('A4.7a probability fixture drifted')
        cell.genome_lesions = [lesion, lesion, lesion]
        dt = 1.0
    cell.replication_template = None
    cell.replication_copy = []
    cell.replication_template_lesion = 0.0
    cell.replication_fractional = 0.0
    cell.membrane_oxidation[:] = 0.0
    cell.pools[a4.a3.POOL_WASTE] = 0.004
    cell._refresh_gene_cache()
    cell._sync_protein_pool()

    # This one uint8 call leaves a cached uint32.  The hit/miss/hit fixture
    # then consumes that cache in the literal random()/integers() interleave.
    world.rng = np.random.default_rng(2)
    priming_value = int(world.rng.integers(
        0, 8, size=1, dtype=np.uint8,
    )[0])
    state = world.rng.bit_generator.state
    if (priming_value != 6 or int(state['has_uint32']) != 1
            or int(state['uinteger']) != 1123615560):
        raise AssertionError('A4.7a primed PCG64 fixture drifted')

    symbol_count = sum(len(genome) for genome in cell.genomes)
    capacity = a4.GPU068A4Config(
        max_cells=1, max_sequences=5, max_symbols=symbol_count,
        max_sequence_symbols=48, max_proteins_per_cell=64,
    )
    return world, cell, capacity, dt


def _a47_manual_apply_tape(cell, tape):
    """Apply only recorded deletion effects on a disposable CPU oracle copy."""
    out = copy.deepcopy(cell)
    for genome_index in range(len(out.genomes)):
        if not bool(tape.hit_mask[genome_index]):
            continue
        sequence = np.asarray(out.genomes[genome_index], dtype=np.uint8)
        position = int(tape.deletion_positions[genome_index])
        if position < 0 or position >= len(sequence):
            raise AssertionError('A4.7a tape deletion position is invalid')
        out.genomes[genome_index] = np.concatenate((
            sequence[:position], sequence[position + 1:],
        ))
        out.pools[a4.a3.POOL_WASTE] += float(a4.g2.MONOMER_MASS)
        out.genome_lesions[genome_index] *= 0.80
        out.genome_damage_events += 1
        out._refresh_gene_cache()
    return out


def _a47b_hydrolysis_fixture(zero_hit=False):
    """A4.7b one-cell source with non-candidate active intermediates."""
    world, cell, _, dt = _a47_hydrolysis_fixture(
        boundaries=bool(zero_hit),
    )
    template = _a47_gene_sequence(a4.g2.ROLE_REPLICASE, 0)
    copy_symbols = template[:7].copy()
    cell.replication_template = template.copy()
    cell.replication_copy = [int(value) for value in copy_symbols]
    cell.replication_template_lesion = 0.125
    cell.replication_fractional = 0.375
    cell._refresh_gene_cache()
    cell._sync_protein_pool()

    sequences = list(cell.genomes) + [template, copy_symbols]
    symbol_count = sum(len(sequence) for sequence in sequences)
    capacity = a4.GPU068A4Config(
        max_cells=1, max_sequences=len(sequences),
        max_symbols=symbol_count,
        max_sequence_symbols=max(len(sequence) for sequence in sequences),
        max_proteins_per_cell=64,
    )
    if (not zero_hit
            and (len(sequences) != 5 or symbol_count != 199
                 or sum(len(genome) for genome in cell.genomes)
                 + len(copy_symbols) != 151)):
        raise AssertionError('A4.7b exact-capacity fixture drifted')
    return world, cell, capacity, dt


def _a47b_formal066_oracle(world, cell, dt):
    """Return direct frozen-CPU hydrolysis state and descriptor expectations."""
    before = copy.deepcopy(cell)
    cpu_world = copy.deepcopy(world)
    actual = copy.deepcopy(cell)
    cpu_world.cells = [actual]
    actual._decay_information_and_proteins(cpu_world, dt)

    sequences = [
        np.asarray(genome, dtype=np.uint8).copy()
        for genome in actual.genomes
    ]
    if actual.replication_template is not None:
        sequences.extend((
            np.asarray(actual.replication_template, dtype=np.uint8).copy(),
            np.asarray(actual.replication_copy, dtype=np.uint8).copy(),
        ))
    width = max([len(sequence) for sequence in sequences] or [0])
    fixed = np.zeros((len(sequences), width), dtype=np.uint8)
    lengths = np.asarray(
        [len(sequence) for sequence in sequences], dtype=np.int64,
    )
    for index, sequence in enumerate(sequences):
        fixed[index, :len(sequence)] = sequence
    offsets = np.zeros((len(sequences) + 1,), dtype=np.int64)
    if len(sequences):
        offsets[1:] = np.cumsum(lengths, dtype=np.int64)

    lesion_after = np.asarray(actual.genome_lesions, dtype=np.float64)
    event_delta = (
        int(actual.genome_damage_events)
        - int(before.genome_damage_events)
    )
    source_sequences = list(before.genomes)
    if before.replication_template is not None:
        source_sequences.extend((
            before.replication_template, before.replication_copy,
        ))
    source_symbol_count = sum(len(sequence) for sequence in source_sequences)
    material_after = (
        sum(len(genome) for genome in actual.genomes)
        + len(actual.replication_copy)
    )
    return cpu_world, actual, {
        'final_symbols': fixed,
        'final_lengths': lengths,
        'final_offsets': offsets,
        'final_symbol_count': int(offsets[-1]),
        'pools_after': np.asarray(actual.pools, dtype=np.float64).copy(),
        'genome_lesions_after': lesion_after.copy(),
        'genome_lesion_mean_after': (
            float(np.mean(lesion_after)) if len(lesion_after) else 1.0
        ),
        'genome_damage_event_delta': event_delta,
        'topology_sequence_delta': 0,
        'topology_symbol_delta': int(offsets[-1]) - source_symbol_count,
        'genome_material_symbols_after': material_after,
        'cache_dirty': bool(event_delta),
        'gene_specs_after': copy.deepcopy(actual.gene_specs),
        'rng_after_state': copy.deepcopy(cpu_world.rng.bit_generator.state),
    }


def _a48_binding(world, cell, config):
    ragged = a4.FullFidelityA4GenomeAdapter(config).pack_cells([cell])
    state = a4.pack_a4_translation_state(
        [cell], ragged, world.config, config,
    )
    return a4.bind_a4_translation(ragged, state)


def _a48_begin_direct(scheduler, world, cell, dt):
    """Enter exactly the post-lesion-gain A3 event boundary."""
    scheduler.begin_step(world, [cell])
    metabolism_rank = a3s.WORLD_EVENT_ORDER.index('cell_metabolism_loop')
    for event in a3s.WORLD_EVENT_ORDER[:metabolism_rank]:
        scheduler.claim_world(
            event, status='skipped', metadata={'validation': 'A4.8a'},
        )
    scheduler.claim(
        cell, 'surface_exchange', status='skipped',
        metadata={'validation': 'A4.8a'},
    )
    scheduler.reserve_metabolism_dispatch(world, cell, dt)
    if not scheduler.begin_a3_metabolism(world, cell, dt):
        raise AssertionError('A4.8a direct fixture cell is not alive')
    stop = a3s.CELL_EVENT_ORDER.index('genome_hydrolysis_cpu_rng')
    for event in a3s.CELL_EVENT_ORDER[1:stop]:
        scheduler.claim(
            cell, event, status='skipped',
            metadata={'validation': 'A4.8a'},
        )
    return scheduler


def _a48_hydrolysis_entry(scheduler, cell):
    _, record = scheduler._record(cell)
    entries = [entry for entry in record['events']
               if entry['event'] == 'genome_hydrolysis_cpu_rng']
    if len(entries) != 1:
        raise AssertionError(
            'A4.8a hydrolysis event count is %d' % len(entries)
        )
    return copy.deepcopy(entries[0])


def _a48_abort_direct(scheduler, cell, error=None):
    cell._defer_damage_viability = False
    if scheduler.active:
        return scheduler.abort_step(
            RuntimeError('A4.8a direct validation close')
            if error is None else error
        )
    return None


def _a48_assert_committed_oracle(world, cell, expected_world,
                                 expected_cell, label):
    _assert_structural_equal(expected_cell, cell, label + '.structure')
    v3.assert_recursive_close(
        expected_cell.pools, cell.pools, atol=0.0, rtol=0.0,
        path=label + '.pools',
    )
    if (int(expected_cell.genome_damage_events)
            != int(cell.genome_damage_events)):
        raise AssertionError(label + ' damage-event counter differs')
    if expected_world.rng.bit_generator.state != world.rng.bit_generator.state:
        raise AssertionError(label + ' full PCG64 state differs')


def _a48_assert_plan_equal(expected, actual, label):
    left = expected.state_dict()
    right = actual.state_dict()
    lesion_mean = 'genome_lesion_mean_after'
    left_mean = np.asarray(left.pop(lesion_mean), dtype=np.float64)
    right_mean = np.asarray(right.pop(lesion_mean), dtype=np.float64)
    v3.assert_recursive_close(
        left, right, atol=0.0, rtol=0.0, path=label,
    )
    left_bits = left_mean.view(np.uint64).reshape(-1)
    right_bits = right_mean.view(np.uint64).reshape(-1)
    if (left_mean.shape != right_mean.shape
            or not all(abs(int(a) - int(b)) <= 1
                       for a, b in zip(left_bits, right_bits))):
        raise AssertionError(label + ' lesion mean differs by more than 1 ULP')


def _a48_hybrid_from_state(state, a4_config=None, device='cpu'):
    world = a4.a3.s66.Formal066World.from_state(v3.pickle_clone(state))
    backend = a4.a3.TorchKernelBackendA3(
        v3._a3_config(device=device, precision='float64'),
    )
    return a48.Hybrid066WorldA4Hydrolysis(
        world, backend=backend,
        a4_config=a4_config or a4.GPU068A4Config(),
        a4_device=device,
    )


def _a48b_begin_translation_direct(scheduler, world, cell, dt):
    """Enter exactly the post-maintenance translation event boundary."""
    scheduler.begin_step(world, [cell])
    metabolism_rank = a3s.WORLD_EVENT_ORDER.index('cell_metabolism_loop')
    for event in a3s.WORLD_EVENT_ORDER[:metabolism_rank]:
        scheduler.claim_world(
            event, status='skipped', metadata={'validation': 'A4.8b'},
        )
    scheduler.claim(
        cell, 'surface_exchange', status='skipped',
        metadata={'validation': 'A4.8b'},
    )
    scheduler.reserve_metabolism_dispatch(world, cell, dt)
    if not scheduler.begin_a3_metabolism(world, cell, dt):
        raise AssertionError('A4.8b direct fixture cell is not alive')
    stop = a3s.CELL_EVENT_ORDER.index('translation_cpu')
    for event in a3s.CELL_EVENT_ORDER[1:stop]:
        scheduler.claim(
            cell, event, status='skipped',
            metadata={'validation': 'A4.8b'},
        )
    return scheduler


def _a48b_translation_entry(scheduler, cell):
    _, record = scheduler._record(cell)
    entries = [entry for entry in record['events']
               if entry['event'] == 'translation_cpu']
    if len(entries) != 1:
        raise AssertionError(
            'A4.8b translation event count is %d' % len(entries)
        )
    return copy.deepcopy(entries[0])


def _a48b_assert_cell_matches(expected, actual, label, atol=2e-12):
    if list(expected.proteins.keys()) != list(actual.proteins.keys()):
        raise AssertionError(label + ' active protein order differs')
    if (list(expected.damaged_proteins.keys())
            != list(actual.damaged_proteins.keys())):
        raise AssertionError(label + ' damaged protein order differs')
    v3.assert_recursive_close(
        expected.state_dict(), actual.state_dict(),
        atol=atol, rtol=0.0, path=label,
    )


def _a48b_hybrid_from_state(state, a4_config=None, device='cpu'):
    world = a4.a3.s66.Formal066World.from_state(v3.pickle_clone(state))
    backend = a4.a3.TorchKernelBackendA3(
        v3._a3_config(device=device, precision='float64'),
    )
    return a48b.Hybrid066WorldA4Translation(
        world, backend=backend,
        a4_config=a4_config or a4.GPU068A4Config(),
        a4_device=device,
    )


def _a48b_replication_boundary_cases(seed=9971):
    """Post-translation CPU-replication branch controls with real RNG work."""
    cases = []
    world, cells, _, dt = _paid_replication_fixture(seed=seed)
    world.config.mutation = True
    world.config.mutation_rate = 1.0
    for label, index, copied, rng_changed in (
            ('atp_exact', 2, 1, True),
            ('atp_nextbelow', 3, 0, False),
            ('nucleotide_exact', 4, 1, True),
            ('nucleotide_nextbelow', 5, 0, False)):
        cases.append((
            label, v3.pickle_clone(world.state_dict()),
            int(cells[index].cell_id), dt, copied, rng_changed,
        ))

    gate_world, gate_cells, _, gate_dt = _paid_replication_fixture(
        seed=seed + 1,
    )
    gate_world.config.gene_expression = False
    gate_world.config.mutation = True
    gate_world.config.mutation_rate = 1.0
    gate = gate_cells[0]
    gate.genome_lesions = [0.0 for _ in gate.genomes]
    gate.damaged_proteins = {}
    gate.pools[a4.a3.POOL_DAMAGED_PROTEIN] = 0.0
    gate.pools[a4.a3.POOL_AGGREGATE] = 0.0
    replicases = [
        (fingerprint, spec)
        for fingerprint, spec in gate.gene_specs.items()
        if int(spec['role']) == int(a4.a3.ROLE_REPLICASE)
    ]
    if len(replicases) != 1:
        raise AssertionError('replicase boundary fixture is not singular')
    fingerprint, spec = replicases[0]
    gate_mass = 1e-6 * 0.040 / float(spec['efficiency'])
    for label, mass, copied, rng_changed in (
            ('replicase_below', np.nextafter(gate_mass, 0.0), 0, False),
            ('replicase_above', gate_mass, 1, True)):
        candidate_world = a4.a3.s66.Formal066World.from_state(
            v3.pickle_clone(gate_world.state_dict()),
        )
        candidate = candidate_world.cells[0]
        candidate.proteins = {fingerprint: float(mass)}
        candidate._sync_protein_pool()
        activity = candidate.role_activity(a4.a3.ROLE_REPLICASE)
        if ((label.endswith('below') and not activity <= 1e-6)
                or (label.endswith('above') and not activity > 1e-6)):
            raise AssertionError(label + ' did not straddle replicase gate')
        cases.append((
            label, v3.pickle_clone(candidate_world.state_dict()),
            int(candidate.cell_id), gate_dt, copied, rng_changed,
        ))
    return cases


def _assert_completion_cpu_parity(plan, before_cells, cpu_cells, label,
                                  atol=2e-12):
    if int(plan.cell_count) != len(before_cells):
        raise AssertionError('%s cell count differs' % label)
    for ci, (before, actual) in enumerate(zip(before_cells, cpu_cells)):
        if not bool(plan.completion_events[ci]):
            raise AssertionError('%s cell[%d] did not complete' % (label, ci))
        length = int(plan.completed_lengths[ci])
        completed = np.asarray(
            plan.completed_symbols[ci, :length], dtype=np.uint8,
        )
        append_count = int(plan.append_count[ci])
        appended = np.asarray(
            plan.append_symbols[ci, :append_count], dtype=np.uint8,
        )
        partial = np.asarray(before.replication_copy, dtype=np.uint8)
        if (len(actual.genomes) != len(before.genomes) + 1
                or not np.array_equal(completed, actual.genomes[-1])
                or not np.array_equal(completed[:len(partial)], partial)
                or not np.array_equal(completed[len(partial):], appended)):
            raise AssertionError(
                '%s cell[%d] completed payload differs' % (label, ci)
            )
        if (actual.replication_template is not None
                or actual.replication_copy
                or float(actual.replication_template_lesion) != 0.0
                or float(actual.replication_fractional) != 0.0):
            raise AssertionError(
                '%s cell[%d] frozen completion reset differs' % (label, ci)
            )
        if (int(plan.replication_cycle_deltas[ci]) != 1
                or int(actual.replication_cycles)
                != int(before.replication_cycles) + 1
                or int(plan.topology_sequence_deltas[ci]) != -1
                or int(plan.topology_symbol_deltas[ci])
                != append_count - length):
            raise AssertionError(
                '%s cell[%d] completion delta differs' % (label, ci)
            )
        if before.mutation_events != actual.mutation_events:
            raise AssertionError(
                '%s cell[%d] mutation counters changed' % (label, ci)
            )
        v3.assert_recursive_close(
            np.asarray(plan.pools_after[ci]), actual.pools,
            atol=atol, rtol=0.0, path='%s.cell[%d].pools' % (label, ci),
        )
        v3.assert_recursive_close(
            float(plan.new_genome_lesions[ci]),
            float(actual.genome_lesions[-1]),
            atol=atol, rtol=0.0,
            path='%s.cell[%d].new_lesion' % (label, ci),
        )
        v3.assert_recursive_close(
            float(plan.last_effective_error_rate[ci]),
            float(actual.last_effective_error_rate),
            atol=atol, rtol=0.0,
            path='%s.cell[%d].effective_error' % (label, ci),
        )
        v3.assert_recursive_close(
            float(plan.cumulative_proofreading_atp_after[ci]),
            float(actual.cumulative_proofreading_atp),
            atol=atol, rtol=0.0,
            path='%s.cell[%d].proofreading_atp' % (label, ci),
        )
        if (int(plan.last_replication_symbols[ci])
                != int(actual.last_replication_symbols)):
            raise AssertionError(
                '%s cell[%d] last copied count differs' % (label, ci)
            )


def _external_only_replication_cell(cell):
    out = copy.deepcopy(cell)
    keys = [
        fingerprint for fingerprint, spec in out.gene_specs.items()
        if int(spec['role']) == int(a4.a3.ROLE_REPLICASE)
    ]
    for fingerprint in keys:
        out.proteins.pop(fingerprint, None)
    out._sync_protein_pool()
    if out.role_activity(a4.a3.ROLE_REPLICASE) > 1e-6:
        raise AssertionError('external-only fixture retains replicase')
    return out


def _paid_replication_binding(cells, model_config, config):
    ragged = a4.FullFidelityA4GenomeAdapter(config).pack_cells(cells)
    state = a4.pack_a4_translation_state(
        cells, ragged, model_config, config,
    )
    return ragged, state, a4.bind_a4_translation(ragged, state)


def _assert_paid_replication_cpu_parity(plan, before_cells, cpu_cells, label):
    if int(plan.cell_count) != len(before_cells):
        raise AssertionError('%s cell count differs' % label)
    for ci, (before, actual) in enumerate(zip(before_cells, cpu_cells)):
        expected = copy.deepcopy(before)
        if bool(plan.template_start_events[ci]):
            selected = int(plan.selected_template_indices[ci])
            if (before.replication_template is not None or selected != 0
                    or len(before.genomes) != 1):
                raise AssertionError(
                    '%s cell[%d] template-start metadata differs' % (
                        label, ci,
                    )
                )
            expected.replication_template = np.asarray(
                before.genomes[selected], dtype=np.uint8,
            ).copy()
            expected.replication_template_lesion = (
                float(before.genome_lesions[selected])
                if selected < len(before.genome_lesions) else 0.0
            )
            expected.replication_copy = []
            expected.replication_fractional = 0.0
            if (int(plan.template_storage_symbols[ci])
                    != len(expected.replication_template)):
                raise AssertionError(
                    '%s cell[%d] template storage differs' % (label, ci)
                )
        count = int(plan.append_count[ci])
        appended = [int(value) for value in plan.append_symbols[ci, :count]]
        expected.replication_copy.extend(appended)
        expected.pools[:] = np.asarray(plan.pools_after[ci], dtype=np.float64)
        expected.replication_fractional = float(
            plan.replication_fractional_after[ci]
        )
        expected.last_replication_symbols = int(
            plan.last_replication_symbols[ci]
        )
        expected.last_effective_error_rate = float(
            plan.last_effective_error_rate[ci]
        )
        expected.cumulative_proofreading_atp = float(
            plan.cumulative_proofreading_atp_after[ci]
        )
        expected.mutation_events['substitution'] += int(
            plan.substitution_events[ci]
        )
        if actual.replication_copy[len(before.replication_copy):] != appended:
            raise AssertionError('%s cell[%d] copied suffix differs' % (label, ci))
        v3.assert_recursive_close(
            expected.state_dict(), actual.state_dict(),
            atol=2e-12, rtol=0.0,
            path='%s.cell[%d]' % (label, ci),
        )


def _assert_replication_failure_atomic(kind, binding, fn, label):
    ragged_before = binding.ragged.state_dict()
    state_before = binding.state.state_dict()
    cache_before = binding.cache.state_dict()
    _assert_raises(kind, fn)
    v3.assert_recursive_close(
        ragged_before, binding.ragged.state_dict(), atol=0.0, rtol=0.0,
        path='%s.ragged' % label,
    )
    v3.assert_recursive_close(
        state_before, binding.state.state_dict(), atol=0.0, rtol=0.0,
        path='%s.state' % label,
    )
    v3.assert_recursive_close(
        cache_before, binding.cache.state_dict(), atol=0.0, rtol=0.0,
        path='%s.cache' % label,
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
    replication_required = (
        'A4ReplicationError', 'A4ReplicationScopeError',
        'A4PaidElongationPlan', 'validate_a4_paid_elongation_plan',
        'paid_replication_elongation_numpy',
        'paid_replication_elongation_torch',
        'paid_replication_elongation_plan',
        'paid_replication_completion_plan',
        'A4SubstitutionRngTape', 'validate_a4_substitution_rng_tape',
        'prepare_substitution_rng_tape',
        'paid_replication_substitution_numpy',
        'paid_replication_substitution_torch',
        'paid_replication_substitution_plan',
        'A4CompletionMutationRngTape',
        'prepare_completion_mutation_rng_tape',
        'validate_a4_completion_mutation_rng_tape',
        'A4CompletionMutationPlan',
        'validate_a4_completion_mutation_plan',
        'paid_replication_completion_mutation_numpy',
        'paid_replication_completion_mutation_torch',
        'paid_replication_completion_mutation_plan',
    )
    missing = [name for name in replication_required if not hasattr(a44, name)]
    if missing:
        raise AssertionError('missing A4.6b2 API: %s' % missing)
    hydrolysis_required = (
        'A4HydrolysisError', 'A4HydrolysisScopeError',
        'A4HydrolysisRngTape', 'prepare_genome_hydrolysis_rng_tape',
        'validate_a4_hydrolysis_rng_tape',
        'A4HydrolysisDeletionPlan',
        'validate_a4_hydrolysis_deletion_plan',
        'genome_hydrolysis_deletion_numpy',
        'genome_hydrolysis_deletion_torch',
        'genome_hydrolysis_deletion_plan',
    )
    missing = [name for name in hydrolysis_required if not hasattr(a47, name)]
    if missing:
        raise AssertionError('missing A4.7b API: %s' % missing)
    integration_required = (
        'A4HydrolysisCommitError', 'A4HydrolysisEventScheduler',
        'Hybrid066WorldA4Hydrolysis',
    )
    missing = [name for name in integration_required
               if not hasattr(a48, name)]
    if missing:
        raise AssertionError('missing A4.8a API: %s' % missing)
    translation_integration_required = (
        'A4TranslationCommitError', 'A4TranslationEventScheduler',
        'Hybrid066WorldA4Translation',
    )
    missing = [name for name in translation_integration_required
               if not hasattr(a48b, name)]
    if missing:
        raise AssertionError('missing A4.8b API: %s' % missing)
    replication_integration_required = (
        'A4ReplicationCommitError', 'A4ReplicationEventScheduler',
        'Hybrid066WorldA4Replication',
    )
    missing = [name for name in replication_integration_required
               if not hasattr(a48c, name)]
    if missing:
        raise AssertionError('missing A4.8c1 API: %s' % missing)
    completion_integration_required = (
        'A4ReplicationCompletionCommitError',
        'A4ReplicationCompletionEventScheduler',
        'Hybrid066WorldA4ReplicationCompletion',
    )
    missing = [name for name in completion_integration_required
               if not hasattr(a48c2, name)]
    if missing:
        raise AssertionError('missing A4.8c2 API: %s' % missing)
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
    if a44.BUILD != 'SOMA-CELL 0.6.8-GPU A4.6b2':
        raise AssertionError('A4.6b2 build identity differs')
    if a44.SCHEMA_VERSION != (
            '0.6.8-GPU-A4.6a-replication-completion-plan'):
        raise AssertionError('A4.6a schema identity differs')
    if a44.RNG_TAPE_SCHEMA_VERSION != (
            '0.6.8-GPU-A4.5b-template-start-rng-tape'):
        raise AssertionError('A4.5b RNG tape schema identity differs')
    if a44.COMPLETION_MUTATION_RNG_TAPE_SCHEMA_VERSION != (
            '0.6.8-GPU-A4.6b1-completion-mutation-rng-tape'):
        raise AssertionError('A4.6b1 RNG tape schema identity differs')
    if a44.COMPLETION_MUTATION_PLAN_SCHEMA_VERSION != (
            '0.6.8-GPU-A4.6b2-completion-mutation-plan'):
        raise AssertionError('A4.6b2 completion plan schema differs')
    if tuple(a44.STRUCTURAL_EVENT_NAMES) != (
            'insertion', 'deletion', 'duplication',
            'inversion', 'transposition'):
        raise AssertionError('A4.6b1 structural event order differs')
    tape_fields = set(a44.A4CompletionMutationRngTape.__dataclass_fields__)
    forbidden_payload = {
        'completed_symbols', 'final_symbols', 'mutated_symbols',
        'completed_genome', 'final_genome',
    }
    if (tape_fields & forbidden_payload
            or len(a44._COMPLETION_TAPE_ARRAY_FIELDS) != 32
            or len(set(a44._COMPLETION_TAPE_ARRAY_FIELDS)) != 32):
        raise AssertionError(
            'A4.6b1 tape exposed a final genome or duplicate field'
        )
    plan_fields = tuple(a44._COMPLETION_PLAN_ARRAY_FIELDS)
    if (len(plan_fields) != len(set(plan_fields))
            or 'final_symbols' not in plan_fields
            or 'genome_material_symbols_after' not in plan_fields):
        raise AssertionError('A4.6b2 completion plan field set differs')
    if a44.FULL_GPU_WORLD_STEP is not False:
        raise AssertionError('A4.6b2 must not claim full GPU world-step')
    if (a47.BUILD != 'SOMA-CELL 0.6.8-GPU A4.7b'
            or a47.SCHEMA_VERSION
            != '0.6.8-GPU-A4.7a-genome-hydrolysis-rng-tape'
            or a47.DELETION_PLAN_SCHEMA_VERSION
            != '0.6.8-GPU-A4.7b-genome-hydrolysis-deletion-plan'
            or a47.FULL_GPU_WORLD_STEP is not False):
        raise AssertionError('A4.7b identity/authority differs')
    hydrolysis_fields = set(
        a47.A4HydrolysisRngTape.__dataclass_fields__
    )
    if (set(a47._TAPE_ARRAY_FIELDS) != {
            'genome_slot_mask', 'draw_mask', 'uniform_draws',
            'hit_mask', 'deletion_positions'}
            or hydrolysis_fields & {
                'final_symbols', 'completed_symbols', 'pools_after',
                'genome_lesions_after'}):
        raise AssertionError('A4.7a tape exposed application state')
    if tuple(a47._DELETION_PLAN_ARRAY_FIELDS) != (
            'scope_valid', 'scope_error_code', 'final_symbols',
            'final_lengths', 'genome_lesions_after', 'pools_after',
            'symbol_count_after', 'topology_symbol_delta',
            'genome_damage_event_delta', 'gene_cache_dirty',
            'gene_cache_refresh_count', 'genome_material_symbols_after',
            'genome_lesion_mean_after'):
        raise AssertionError('A4.7b deletion-plan field order differs')
    if (a47.DELETION_SCOPE_OK != 0
            or a47.DELETION_SCOPE_TAPE_MISMATCH != 1):
        raise AssertionError('A4.7b deletion scope codes differ')
    if (a44.SCOPE_FP64_DISCRETE_BOUNDARY != 6
            or a44.FP64_DISCRETE_GUARD_EPS != 4096.0):
        raise AssertionError('A4.5b fp64 discrete guard contract differs')
    if a44.SCOPE_RNG_TAPE_MISMATCH != 7:
        raise AssertionError('A4.5b RNG tape scope code differs')
    if a44.SCOPE_NONCOMPLETION != 8:
        raise AssertionError('A4.6a noncompletion scope code differs')
    expected_replication = (
        'a4.6b2-pre-existing-active-all-row-completion-combined-pcg64-'
        'substitution-structural-material-fixed-resident-plan-'
        'not-arena-committed-not-integrated-cpu-authoritative'
    )
    expected_material = (
        'a4.6b2-binding-aware-attested-tape-applied-to-pure-resident-'
        'descriptor-not-live-rng-authority-cpu-authoritative'
    )
    if (a44.PORT_STATUS.get('genome_replication') != expected_replication
            or a44.PORT_STATUS.get('material_mutation')
            != expected_material):
        raise AssertionError('A4.6b2 authority status differs')
    expected_hydrolysis = (
        'a4.7b-single-cell-row-padded-deletion-ledger-plan-'
        'not-arena-cache-live-rng-committed-not-integrated-'
        'cpu-authoritative'
    )
    if a47.PORT_STATUS.get('genome_symbol_hydrolysis') != expected_hydrolysis:
        raise AssertionError('A4.7b hydrolysis authority status differs')
    public_names = set(a48.__all__)
    if (a48.BUILD != 'SOMA-CELL 0.6.8-GPU A4.8a'
            or a48.SCHEMA_VERSION
            != '0.6.8-GPU-A4.8a-hydrolysis-atomic-commit'
            or a48.FULL_GPU_WORLD_STEP is not False
            or '_A4HydrolysisCommitCandidate' in public_names):
        raise AssertionError('A4.8a identity/public scope differs')
    signature = inspect.signature(
        a48.A4HydrolysisEventScheduler.cpu_genome_hydrolysis,
    )
    if tuple(signature.parameters) != ('self', 'world', 'cell', 'hazards', 'dt'):
        raise AssertionError('A4.8a bridge accepts external plan authority')
    if not issubclass(
            a48.A4HydrolysisEventScheduler, a3s.A3EventScheduler):
        raise AssertionError('A4.8a scheduler is not an A3 scheduler subtype')
    translation_public_names = set(a48b.__all__)
    if (a48b.BUILD != 'SOMA-CELL 0.6.8-GPU A4.8b'
            or a48b.SCHEMA_VERSION
            != '0.6.8-GPU-A4.8b-translation-atomic-commit'
            or a48b.FULL_GPU_WORLD_STEP is not False
            or '_A4TranslationCommitCandidate' in translation_public_names):
        raise AssertionError('A4.8b identity/public scope differs')
    translation_signature = inspect.signature(
        a48b.A4TranslationEventScheduler.cpu_translation,
    )
    if tuple(translation_signature.parameters) != (
            'self', 'world', 'cell', 'dt', 'config'):
        raise AssertionError('A4.8b bridge accepts external plan authority')
    if (not issubclass(
            a48b.A4TranslationEventScheduler,
            a48.A4HydrolysisEventScheduler)
            or not issubclass(
                a48b.Hybrid066WorldA4Translation,
                a48.Hybrid066WorldA4Hydrolysis)):
        raise AssertionError('A4.8b wrapper did not preserve A4.8a authority')
    replication_public_names = set(a48c.__all__)
    if (a48c.BUILD != 'SOMA-CELL 0.6.8-GPU A4.8c1'
            or a48c.SCHEMA_VERSION
            != ('0.6.8-GPU-A4.8c1-active-noncompletion-'
                'replication-atomic-commit')
            or a48c.FULL_GPU_WORLD_STEP is not False
            or '_A4ReplicationCommitCandidate' in replication_public_names):
        raise AssertionError('A4.8c1 identity/public scope differs')
    replication_signature = inspect.signature(
        a48c.A4ReplicationEventScheduler.cpu_replication,
    )
    if tuple(replication_signature.parameters) != (
            'self', 'world', 'cell', 'dt', 'config'):
        raise AssertionError('A4.8c1 bridge accepts external plan authority')
    if (not issubclass(
            a48c.A4ReplicationEventScheduler,
            a48b.A4TranslationEventScheduler)
            or not issubclass(
                a48c.Hybrid066WorldA4Replication,
                a48b.Hybrid066WorldA4Translation)):
        raise AssertionError('A4.8c1 wrapper did not preserve A4.8b authority')
    completion_public_names = set(a48c2.__all__)
    if (a48c2.BUILD != 'SOMA-CELL 0.6.8-GPU A4.8c2'
            or a48c2.SCHEMA_VERSION
            != ('0.6.8-GPU-A4.8c2-mutation-free-active-completion-'
                'atomic-commit')
            or a48c2.FULL_GPU_WORLD_STEP is not False
            or '_A4ReplicationCompletionCommitCandidate'
            in completion_public_names):
        raise AssertionError('A4.8c2 identity/public scope differs')
    completion_signature = inspect.signature(
        a48c2.A4ReplicationCompletionEventScheduler.cpu_replication,
    )
    if tuple(completion_signature.parameters) != (
            'self', 'world', 'cell', 'dt', 'config'):
        raise AssertionError('A4.8c2 bridge accepts external plan authority')
    if (not issubclass(
            a48c2.A4ReplicationCompletionEventScheduler,
            a48c.A4ReplicationEventScheduler)
            or not issubclass(
                a48c2.Hybrid066WorldA4ReplicationCompletion,
                a48c.Hybrid066WorldA4Replication)):
        raise AssertionError('A4.8c2 wrapper did not preserve A4.8c1 authority')
    return ('%s / %s + %s + %s + %s + %s + %s + %s + %s + %s / '
            'full_gpu=false') % (
        a48c2.BUILD, a4.SCHEMA_VERSION,
        a4.GENE_CACHE_SCHEMA_VERSION + ' + ' + a4.TRANSLATION_SCHEMA_VERSION,
        a44.SCHEMA_VERSION, a47.SCHEMA_VERSION,
        a47.DELETION_PLAN_SCHEMA_VERSION, a48.SCHEMA_VERSION,
        a48b.SCHEMA_VERSION, a48c.SCHEMA_VERSION, a48c2.SCHEMA_VERSION,
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

    # Scalar metadata is part of the binding trust identity even though it is
    # not tensor storage.  Cover every post-bind component on both backends.
    for component in ('ragged', 'state', 'cache'):
        scalar_binding = a4.bind_a4_translation(
            ragged.clone(), state.clone(),
        )
        if component == 'ragged':
            scalar_binding.ragged.symbol_count -= 1
        elif component == 'state':
            scalar_binding.state.quiescence = (
                not scalar_binding.state.quiescence
            )
        else:
            scalar_binding.cache.cell_count -= 1
        _assert_raises(
            a4.A4SchemaError,
            lambda scalar_binding=scalar_binding: a4.paid_translation_plan_numpy(
                scalar_binding, dt,
            ),
        )
    for component in ('ragged', 'state', 'cache'):
        scalar_binding = a4.bind_a4_translation(
            ragged.to_torch(attest_device),
            state.to_torch(attest_device),
        )
        if component == 'ragged':
            scalar_binding.ragged.symbol_count -= 1
        elif component == 'state':
            scalar_binding.state.quiescence = (
                not scalar_binding.state.quiescence
            )
        else:
            scalar_binding.cache.cell_count -= 1
        _assert_raises(
            a4.A4SchemaError,
            lambda scalar_binding=scalar_binding: a4.paid_translation_plan_torch(
                scalar_binding, dt,
            ),
        )

    # Resident uploads remember their scalar metadata before binding.
    prebind_ragged = ragged.to_torch(attest_device)
    prebind_state = state.to_torch(attest_device)
    prebind_ragged.symbol_count -= 1
    _assert_raises(
        a4.A4SchemaError,
        lambda: a4.bind_a4_translation(prebind_ragged, prebind_state),
    )
    prebind_ragged = ragged.to_torch(attest_device)
    prebind_state = state.to_torch(attest_device)
    prebind_state.quiescence = not prebind_state.quiescence
    _assert_raises(
        a4.A4SchemaError,
        lambda: a4.bind_a4_translation(prebind_ragged, prebind_state),
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


def test_a44_paid_replication_formal066_cpu_oracle_and_nonmutation():
    world, cells, config, dt = _paid_replication_fixture()
    if any(type(cell).__name__ != 'Formal066ProtoCell' for cell in cells):
        raise AssertionError('A4.4 CPU oracle is not Formal066')
    world_before = v3.pickle_clone(world.state_dict())
    rng_before = v3.pickle_clone(world.rng.bit_generator.state)
    ragged, state, binding = _paid_replication_binding(
        cells, world.config, config,
    )
    ragged_before = ragged.state_dict()
    state_before = state.state_dict()
    cache_before = binding.cache.state_dict()
    plan = a44.paid_replication_elongation_numpy(
        binding, dt, world.config,
    )
    v3.assert_recursive_close(
        world_before, world.state_dict(), atol=0.0, rtol=0.0,
        path='a44.numpy_world',
    )
    v3.assert_recursive_close(
        rng_before, world.rng.bit_generator.state, atol=0.0, rtol=0.0,
        path='a44.numpy_rng',
    )
    v3.assert_recursive_close(
        ragged_before, ragged.state_dict(), atol=0.0, rtol=0.0,
        path='a44.numpy_ragged',
    )
    v3.assert_recursive_close(
        state_before, state.state_dict(), atol=0.0, rtol=0.0,
        path='a44.numpy_state',
    )
    v3.assert_recursive_close(
        cache_before, binding.cache.state_dict(), atol=0.0, rtol=0.0,
        path='a44.numpy_cache',
    )

    cpu_cells = copy.deepcopy(cells)
    cpu_before = copy.deepcopy(cpu_cells)
    for cell in cpu_cells:
        cell._replicate_genome(world, dt, world.config)
    _assert_paid_replication_cpu_parity(
        plan, cpu_before, cpu_cells, 'a44.formal066',
    )
    v3.assert_recursive_close(
        world_before, world.state_dict(), atol=0.0, rtol=0.0,
        path='a44.cpu_oracle_world',
    )
    v3.assert_recursive_close(
        rng_before, world.rng.bit_generator.state, atol=0.0, rtol=0.0,
        path='a44.cpu_oracle_rng',
    )
    return '6 direct Formal066 rows exact; RNG/world/ragged/state/cache unchanged'


def test_a44_requested_zero_and_exact_resource_boundaries():
    world, cells, config, dt = _paid_replication_fixture(seed=7802)
    _, _, binding = _paid_replication_binding(cells, world.config, config)
    plan = a44.paid_replication_elongation_numpy(
        binding, dt, world.config,
    )
    requested = [int(value) for value in plan.requested_symbols[:6]]
    appended = [int(value) for value in plan.append_count[:6]]
    if requested != [1, 0, 1, 1, 1, 1]:
        raise AssertionError('requested-symbol boundary rows differ: %s' % requested)
    if appended != [1, 0, 1, 0, 1, 0]:
        raise AssertionError('paid resource gates differ: %s' % appended)

    cpu_cells = copy.deepcopy(cells)
    cpu_before = copy.deepcopy(cpu_cells)
    for cell in cpu_cells:
        cell._replicate_genome(world, dt, world.config)
    _assert_paid_replication_cpu_parity(
        plan, cpu_before, cpu_cells, 'a44.boundary_cpu',
    )

    if not np.array_equal(plan.pools_after[1], cells[1].pools):
        raise AssertionError('requested=0 changed a paid pool')
    if not (0.125 < float(plan.replication_fractional_after[1]) < 1.0):
        raise AssertionError('requested=0 fractional continuation was lost')
    atp_index = int(a4.a3.POOL_ATP)
    nucleotide_index = int(a4.a3.POOL_NUCLEOTIDE)
    v3.assert_recursive_close(
        float(plan.pools_after[2, atp_index]), 0.022,
        atol=4e-18, rtol=0.0, path='a44.atp_exact_after',
    )
    if float(plan.pools_after[3, atp_index]) != float(cells[3].pools[atp_index]):
        raise AssertionError('ATP nextafter exhaustion was not atomic')
    v3.assert_recursive_close(
        float(plan.pools_after[4, nucleotide_index]), 0.0,
        atol=0.0, rtol=0.0, path='a44.monomer_exact_after',
    )
    if (float(plan.pools_after[5, nucleotide_index])
            != float(cells[5].pools[nucleotide_index])):
        raise AssertionError('monomer nextafter exhaustion was not atomic')
    return 'requested=0 + ATP/monomer exact pass and nextafter exhaustion exact'


def test_a44_paid_replication_numpy_torch_devices_and_scope_readback():
    if torch is None:
        raise AssertionError('PyTorch is required for A4.4b')
    if _REQUIRE_CUDA and not torch.cuda.is_available():
        raise AssertionError('CUDA required but unavailable; CPU fallback forbidden')
    world, cells, config, dt = _paid_replication_fixture(seed=7803)
    ragged, state, binding = _paid_replication_binding(
        cells, world.config, config,
    )
    expected = a44.paid_replication_elongation_numpy(
        binding, dt, world.config,
    )
    source = '\n'.join(inspect.getsource(item) for item in (
        a44.paid_replication_elongation_torch,
        a44._torch_replicase,
        a44._validate_plan_metadata,
    ))
    forbidden = ('.item(', '.cpu(', '.numpy(', '.tolist(',
                 'nonzero(', 'masked_select(', 'unique(')
    found = [token for token in forbidden if token in source]
    if found:
        raise AssertionError('resident replication contains host/dynamic op: %s' % found)

    inactive = copy.deepcopy(cells[0])
    inactive.replication_template = None
    inactive.replication_copy = []
    inactive.replication_template_lesion = 0.0
    inactive.replication_fractional = 0.0
    inactive_ragged, inactive_state, _ = _paid_replication_binding(
        [inactive], world.config, config,
    )

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    for device in devices:
        resident_ragged = ragged.to_torch(device=device)
        resident_state = state.to_torch(device=device)
        ragged_ptrs = resident_ragged.data_ptrs()
        state_ptrs = resident_state.data_ptrs()
        resident_binding = a4.bind_a4_translation(
            resident_ragged, resident_state,
        )
        cache_ptrs = resident_binding.cache.data_ptrs()
        plan = a44.paid_replication_elongation_torch(
            resident_binding, dt, world.config,
        )
        plan_ptrs = plan.data_ptrs()
        if any(getattr(plan, name).device.type != device
               for name in a44._PLAN_ARRAY_FIELDS):
            raise AssertionError('%s replication output escaped device' % device)
        if device == 'cuda':
            torch.cuda.synchronize()
        back = plan.to_numpy()
        v3.assert_recursive_close(
            expected.state_dict(), back.state_dict(),
            atol=2e-12, rtol=0.0, path='a44.plan.%s' % device,
        )
        if ragged_ptrs != resident_ragged.data_ptrs():
            raise AssertionError('%s replication reallocated ragged input' % device)
        if state_ptrs != resident_state.data_ptrs():
            raise AssertionError('%s replication reallocated state input' % device)
        if cache_ptrs != resident_binding.cache.data_ptrs():
            raise AssertionError('%s replication reallocated cache input' % device)
        if plan_ptrs != plan.data_ptrs():
            raise AssertionError('%s plan readback changed resident pointers' % device)
        v3.assert_recursive_close(
            ragged.state_dict(), resident_ragged.to_numpy().state_dict(),
            atol=0.0, rtol=0.0, path='a44.ragged_source.%s' % device,
        )
        v3.assert_recursive_close(
            state.state_dict(), resident_state.to_numpy().state_dict(),
            atol=0.0, rtol=0.0, path='a44.state_source.%s' % device,
        )

        invalid_binding = a4.bind_a4_translation(
            inactive_ragged.to_torch(device=device),
            inactive_state.to_torch(device=device),
        )
        invalid_plan = a44.paid_replication_elongation_torch(
            invalid_binding, dt, world.config,
        )
        _assert_raises(a44.A4ReplicationScopeError, invalid_plan.to_numpy)
    return 'NumPy/Torch %s fp64 exact; fixed resident ops/pointers and scope readback rejection' % '/'.join(devices)


def test_a44_capacity_scope_fail_closed_and_a3_authority():
    world, cells, roomy_config, dt = _paid_replication_fixture(seed=7804)
    single = [cells[0]]
    probe = a4.FullFidelityA4GenomeAdapter(roomy_config).pack_cells(single)
    common = {
        'max_cells': 1,
        'max_sequences': int(probe.sequence_count),
        'max_sequence_symbols': int(roomy_config.max_sequence_symbols),
        'max_proteins_per_cell': int(roomy_config.max_proteins_per_cell),
    }
    exact_config = a4.GPU068A4Config(
        max_symbols=int(probe.symbol_count) + 1, **common
    )
    exact_ragged, _, exact_binding = _paid_replication_binding(
        single, world.config, exact_config,
    )
    exact_plan = a44.paid_replication_elongation_numpy(
        exact_binding, dt, world.config,
    )
    if (int(exact_ragged.symbol_capacity) != int(exact_ragged.symbol_count) + 1
            or int(exact_plan.append_count[0]) != 1):
        raise AssertionError('one-future-symbol exact capacity did not pass')

    forged_plan = exact_plan.clone()
    forged_plan.pools_after[0, a4.a3.POOL_NUCLEOTIDE] = -1e-12
    _assert_raises(
        a4.A4SchemaError,
        lambda: a44.validate_a4_paid_elongation_plan(forged_plan),
    )

    host_ragged = exact_ragged.clone()
    host_state = exact_binding.state.clone()
    host_ragged_binding = a4.bind_a4_translation(host_ragged, host_state)
    host_ragged.symbols[0] = (
        int(host_ragged.symbols[0]) + 1
    ) % a4.ALPHABET_SIZE
    _assert_replication_failure_atomic(
        a4.A4SchemaError, host_ragged_binding,
        lambda: a44.paid_replication_elongation_numpy(
            host_ragged_binding, dt, world.config,
        ),
        'a44.host_post_bind_ragged',
    )
    host_ragged = exact_ragged.clone()
    host_state = exact_binding.state.clone()
    host_state_binding = a4.bind_a4_translation(host_ragged, host_state)
    host_state.pools[0, a4.a3.POOL_FUEL] += 1e-9
    _assert_replication_failure_atomic(
        a4.A4SchemaError, host_state_binding,
        lambda: a44.paid_replication_elongation_numpy(
            host_state_binding, dt, world.config,
        ),
        'a44.host_post_bind_state',
    )

    trust_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resident_ragged_binding = a4.bind_a4_translation(
        exact_ragged.to_torch(device=trust_device),
        exact_binding.state.to_torch(device=trust_device),
    )
    resident_ragged_binding.ragged.symbols[0] = torch.remainder(
        resident_ragged_binding.ragged.symbols[0].to(torch.int64) + 1,
        a4.ALPHABET_SIZE,
    ).to(torch.uint8)
    _assert_raises(
        a4.A4SchemaError,
        lambda: a44.paid_replication_elongation_torch(
            resident_ragged_binding, dt, world.config,
        ),
    )
    resident_state_binding = a4.bind_a4_translation(
        exact_ragged.to_torch(device=trust_device),
        exact_binding.state.to_torch(device=trust_device),
    )
    resident_state_binding.state.pools[0, a4.a3.POOL_FUEL] += 1e-9
    _assert_raises(
        a4.A4SchemaError,
        lambda: a44.paid_replication_elongation_torch(
            resident_state_binding, dt, world.config,
        ),
    )

    full_config = a4.GPU068A4Config(
        max_symbols=int(probe.symbol_count), **common
    )
    full_ragged, full_state, full_binding = _paid_replication_binding(
        single, world.config, full_config,
    )
    world_before = v3.pickle_clone(world.state_dict())
    rng_before = v3.pickle_clone(world.rng.bit_generator.state)
    _assert_replication_failure_atomic(
        a4.A4CapacityError, full_binding,
        lambda: a44.paid_replication_elongation_numpy(
            full_binding, dt, world.config,
        ),
        'a44.capacity_plus_one',
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resident_full = a4.bind_a4_translation(
        full_ragged.to_torch(device=device),
        full_state.to_torch(device=device),
    )
    resident_capacity_plan = a44.paid_replication_elongation_torch(
        resident_full, dt, world.config,
    )
    _assert_raises(a4.A4CapacityError, resident_capacity_plan.to_numpy)

    flag_cases = (
        ('genome_replication', False),
        ('mutation', True),
        ('quiescence', True),
        ('quiescence_effector', True),
    )
    for name, value in flag_cases:
        unsupported = copy.deepcopy(world.config)
        setattr(unsupported, name, value)
        _assert_replication_failure_atomic(
            a44.A4ReplicationScopeError, exact_binding,
            lambda unsupported=unsupported: a44.paid_replication_elongation_numpy(
                exact_binding, dt, unsupported,
            ),
            'a44.unsupported.%s' % name,
        )

    inactive = copy.deepcopy(cells[0])
    inactive.replication_template = None
    inactive.replication_copy = []
    inactive.replication_template_lesion = 0.0
    inactive.replication_fractional = 0.0
    _, _, inactive_binding = _paid_replication_binding(
        [inactive], world.config, roomy_config,
    )
    _assert_replication_failure_atomic(
        a44.A4ReplicationScopeError, inactive_binding,
        lambda: a44.paid_replication_elongation_numpy(
            inactive_binding, dt, world.config,
        ),
        'a44.inactive',
    )

    completing = copy.deepcopy(cells[0])
    completing.replication_copy = [
        int(value) for value in completing.replication_template[:-1]
    ]
    completing.replication_fractional = np.nextafter(1.0, 0.0)
    _, _, completing_binding = _paid_replication_binding(
        [completing], world.config, roomy_config,
    )
    _assert_replication_failure_atomic(
        a44.A4ReplicationScopeError, completing_binding,
        lambda: a44.paid_replication_elongation_numpy(
            completing_binding, dt, world.config,
        ),
        'a44.completion',
    )

    expected_status = (
        'a4.6b2-pre-existing-active-all-row-completion-combined-pcg64-'
        'substitution-structural-material-fixed-resident-plan-'
        'not-arena-committed-not-integrated-cpu-authoritative'
    )
    if a44.PORT_STATUS.get('genome_replication') != expected_status:
        raise AssertionError('A4.6b2 CPU authority status differs')
    if (a44.FULL_GPU_WORLD_STEP is not False
            or a44.PORT_STATUS.get('full_gpu_world_step') is not False):
        raise AssertionError('A4.5b claimed full GPU authority')
    if a4.PORT_STATUS.get('genome_replication') != 'cpu-authoritative-next-a4-slice':
        raise AssertionError('A4 core authority was changed')
    if getattr(v3.a3_module(), 'FULL_GPU_WORLD_STEP', None) is not False:
        raise AssertionError('A3 full-world authority changed')
    event_order = v3._event_order()
    if event_order.count('replication_cpu') != 1:
        raise AssertionError('A3 CPU replication scheduler authority changed')
    v3.assert_recursive_close(
        world_before, world.state_dict(), atol=0.0, rtol=0.0,
        path='a44.fail_closed_world',
    )
    v3.assert_recursive_close(
        rng_before, world.rng.bit_generator.state, atol=0.0, rtol=0.0,
        path='a44.fail_closed_rng',
    )
    return 'future-symbol exact/+1 atomic; trust/plan rejects; 4 flags + inactive/completion; A3 CPU authority'


def test_a44b_feature_matrix_formal066_cpu_oracle():
    world, cells, config, dt = _paid_replication_b_fixture()
    if any(type(cell).__name__ != 'Formal066ProtoCell' for cell in cells):
        raise AssertionError('A4.4b CPU oracle is not Formal066')

    # Fix decoded promoter/efficiency arithmetic and the CPU multiplication
    # grouping used by repair-localised proofreading/quiescence proteins.
    repair_specs = [
        spec for spec in cells[0].gene_specs.values()
        if (int(spec['role']) == int(a4.a3.ROLE_REGULATOR)
            and int(spec['localisation']) == int(a4.s4.LOC_REPAIR))
    ]
    if not repair_specs:
        raise AssertionError('A4.4b fixture lacks repair-localised regulators')
    for spec in repair_specs:
        payload = spec['payload']
        expected_promoter = 0.18 + 1.22 * (payload[3] / 7.0)
        expected_efficiency = 0.52 + 0.96 * (payload[4] / 7.0)
        if (float(spec['promoter']) != expected_promoter
                or float(spec['efficiency']) != expected_efficiency):
            raise AssertionError('gene decode changed base + scale*(byte/7.0)')
    for kind in (a4.a3.REPAIR_PROOFREADING, a4.a3.REPAIR_QUIESCENCE):
        v3.assert_recursive_close(
            _cpu_raw_repair_activity(cells[0], kind),
            cells[0].raw_repair_activity(kind),
            atol=0.0, rtol=0.0, path='a44b.raw_repair.%d' % int(kind),
        )

    plans = {}
    for label, model_config in (
        ('proof_off', _replication_model_config(world.config)),
        ('proof_on', _replication_model_config(
            world.config, proofreading=True,
        )),
    ):
        _, _, binding = _paid_replication_binding(
            cells[:2], model_config, config,
        )
        plan = a44.paid_replication_elongation_numpy(
            binding, dt, model_config,
        )
        cpu_cells = copy.deepcopy(cells[:2])
        cpu_before = copy.deepcopy(cpu_cells)
        for cell in cpu_cells:
            cell._replicate_genome(world, dt, model_config)
        _assert_paid_replication_cpu_parity(
            plan, cpu_before, cpu_cells, 'a44b.%s' % label,
        )
        plans[label] = plan
    if not np.array_equal(
            plans['proof_off'].cumulative_proofreading_atp_after[:2],
            np.asarray([
                cells[0].cumulative_proofreading_atp,
                cells[1].cumulative_proofreading_atp,
            ], dtype=np.float64)):
        raise AssertionError('proofreading-off changed cumulative ATP')
    if not (
            float(plans['proof_on'].cumulative_proofreading_atp_after[0])
            > float(cells[0].cumulative_proofreading_atp)
            and float(plans['proof_on'].last_effective_error_rate[0])
            < float(plans['proof_off'].last_effective_error_rate[0])
            and float(plans['proof_on'].replication_fractional_after[1])
            < float(plans['proof_off'].replication_fractional_after[1])):
        raise AssertionError('proofreading speed/error/payment branches not exercised')

    all_on = _replication_model_config(
        world.config, proofreading=True, external_replicase=True,
        quiescence=True, quiescence_effector=True,
    )
    _, _, binding = _paid_replication_binding(cells[:2], all_on, config)
    combined = a44.paid_replication_elongation_numpy(binding, dt, all_on)
    cpu_cells = copy.deepcopy(cells[:2])
    cpu_before = copy.deepcopy(cpu_cells)
    for cell in cpu_cells:
        cell._replicate_genome(world, dt, all_on)
    _assert_paid_replication_cpu_parity(
        combined, cpu_before, cpu_cells, 'a44b.combined',
    )
    inherited = cells[0].quiescence_level(all_on)
    behavioural = cells[1].quiescence_level(all_on)
    if not (inherited > cells[0].behavioural_quiescence
            and behavioural == cells[1].behavioural_quiescence == 0.70):
        raise AssertionError('inherited/behavioural quiescence dominance missing')

    external = _external_only_replication_cell(cells[0])
    external_config = _replication_model_config(
        world.config, external_replicase=True,
    )
    _, _, external_binding = _paid_replication_binding(
        [external], external_config, config,
    )
    external_plan = a44.paid_replication_elongation_numpy(
        external_binding, dt, external_config,
    )
    external_cpu = [copy.deepcopy(external)]
    external_before = copy.deepcopy(external_cpu)
    external_cpu[0]._replicate_genome(world, dt, external_config)
    _assert_paid_replication_cpu_parity(
        external_plan, external_before, external_cpu, 'a44b.external_only',
    )
    no_external = _replication_model_config(world.config)
    _assert_raises(
        a44.A4ReplicationScopeError,
        lambda: a44.paid_replication_elongation_numpy(
            external_binding, dt, no_external,
        ),
    )
    return 'proof on/off + inherited/behavioural q + external-only direct Formal066 exact'


def test_a44b_proof_ledger_resource_error_and_nonmutation():
    world, cells, config, dt = _paid_replication_b_fixture(seed=7902)
    world_before = v3.pickle_clone(world.state_dict())
    rng_before = v3.pickle_clone(world.rng.bit_generator.state)
    ragged, state, binding = _paid_replication_binding(
        cells, world.config, config,
    )
    ragged_before = ragged.state_dict()
    state_before = state.state_dict()
    cache_before = binding.cache.state_dict()
    plan = a44.paid_replication_elongation_numpy(
        binding, dt, world.config,
    )
    requested = [int(value) for value in plan.requested_symbols[:6]]
    appended = [int(value) for value in plan.append_count[:6]]
    if requested != [1, 0, 1, 1, 1, 1]:
        raise AssertionError('A4.4b requested rows differ: %s' % requested)
    if appended != [1, 0, 1, 0, 1, 0]:
        raise AssertionError('A4.4b resource gates differ: %s' % appended)

    cpu_cells = copy.deepcopy(cells)
    cpu_before = copy.deepcopy(cpu_cells)
    for cell in cpu_cells:
        cell._replicate_genome(world, dt, world.config)
    _assert_paid_replication_cpu_parity(
        plan, cpu_before, cpu_cells, 'a44b.ledger_cpu',
    )
    for ci, (before, after) in enumerate(zip(cells, cpu_cells)):
        v3.assert_recursive_close(
            float(plan.cumulative_proofreading_atp_after[ci]),
            float(after.cumulative_proofreading_atp),
            atol=0.0, rtol=0.0,
            path='a44b.cumulative_after[%d]' % ci,
        )
        if appended[ci] == 0 and (
                float(after.cumulative_proofreading_atp)
                != float(before.cumulative_proofreading_atp)):
            raise AssertionError('unaccepted row paid proofreading ATP')

    atp = int(a4.a3.POOL_ATP)
    nucleotide = int(a4.a3.POOL_NUCLEOTIDE)
    proof = _cpu_raw_repair_activity(
        cells[2], a4.a3.REPAIR_PROOFREADING,
    )
    proof_fraction = proof / (0.75 + proof)
    expected_atp_after = (
        float(cells[2].pools[atp])
        - (float(a4.g2.REPLICATION_ATP_PER_SYMBOL)
           + 0.00075 * proof_fraction)
    )
    v3.assert_recursive_close(
        float(plan.pools_after[2, atp]), expected_atp_after,
        atol=4e-18, rtol=0.0, path='a44b.proof_atp_above_guard',
    )
    if float(plan.pools_after[3, atp]) != float(cells[3].pools[atp]):
        raise AssertionError('proof ATP nextafter gate was not atomic')
    if float(plan.pools_after[4, nucleotide]) != 0.0:
        raise AssertionError('monomer exact gate did not pay to zero')
    if (float(plan.pools_after[5, nucleotide])
            != float(cells[5].pools[nucleotide])):
        raise AssertionError('monomer nextafter gate was not atomic')
    if (not np.array_equal(plan.pools_after[1], cells[1].pools)
            or float(plan.cumulative_proofreading_atp_after[1])
            != float(cells[1].cumulative_proofreading_atp)):
        raise AssertionError('requested=0 changed a paid ledger')
    if (not np.signbit(plan.cumulative_proofreading_atp_after[1])
            or not np.signbit(cpu_cells[1].cumulative_proofreading_atp)):
        raise AssertionError('requested=0 lost frozen negative-zero telemetry')

    for ci in (0, 1):
        proof = _cpu_raw_repair_activity(
            cells[ci], a4.a3.REPAIR_PROOFREADING,
        )
        proof_fraction = proof / (0.75 + proof)
        volume = max(
            0.20, (cells[ci].radius / float(a4.s5.BASE_RADIUS)) ** 2,
        )
        reactive = cells[ci].pools[a4.a3.POOL_REACTIVE] / volume
        raw_error = max(
            0.0, float(world.config.mutation_rate)
            + 0.0012 * cells[ci].replication_template_lesion
            + 0.0010 * reactive,
        )
        expected_error = raw_error * (1.0 - 0.82 * proof_fraction)
        v3.assert_recursive_close(
            float(plan.last_effective_error_rate[ci]), expected_error,
            atol=2e-15, rtol=0.0, path='a44b.error[%d]' % ci,
        )

    for label, expected, actual in (
        ('world', world_before, world.state_dict()),
        ('rng', rng_before, world.rng.bit_generator.state),
        ('ragged', ragged_before, ragged.state_dict()),
        ('state', state_before, state.state_dict()),
        ('cache', cache_before, binding.cache.state_dict()),
    ):
        v3.assert_recursive_close(
            expected, actual, atol=0.0, rtol=0.0,
            path='a44b.nonmutation.%s' % label,
        )
    return ('proof after-state + guard-separated ATP ledgers + exact monomer '
            'boundaries + lesion/reactive error; RNG/world/input exact')


def test_a44b_numpy_torch_deterministic_devices_and_pointers():
    if torch is None:
        raise AssertionError('PyTorch is required for A4.4b')
    if _REQUIRE_CUDA and not torch.cuda.is_available():
        raise AssertionError('CUDA required but unavailable; CPU fallback forbidden')
    world, cells, config, dt = _paid_replication_b_fixture(seed=7903)
    ragged, state, binding = _paid_replication_binding(
        cells, world.config, config,
    )
    expected = a44.paid_replication_elongation_numpy(
        binding, dt, world.config,
    )
    expected_translation = a4.paid_translation_plan_numpy(binding, dt)
    boundary_dt = float.fromhex('0x1.c46204851f076p-4')
    boundary_ragged, boundary_state, boundary_binding = (
        _paid_replication_binding([cells[0]], world.config, config)
    )
    _assert_raises(
        a44.A4ReplicationScopeError,
        lambda: a44.paid_replication_elongation_numpy(
            boundary_binding, boundary_dt, world.config,
        ),
    )
    control_dt = boundary_dt + 1e-10
    control_expected = a44.paid_replication_elongation_numpy(
        boundary_binding, control_dt, world.config,
    )
    if int(control_expected.requested_symbols[0]) != 5:
        raise AssertionError('A4.4b nonambiguous boundary control drifted')
    division_cell = copy.deepcopy(cells[0])
    division_cell.radius = float.fromhex('0x1.5eaf9b4659ac2p-5')
    division_dt = float.fromhex('0x1.c96cc2eb79450p-4')
    division_ragged, division_state, division_binding = (
        _paid_replication_binding([division_cell], world.config, config)
    )
    _assert_raises(
        a44.A4ReplicationScopeError,
        lambda: a44.paid_replication_elongation_numpy(
            division_binding, division_dt, world.config,
        ),
    )
    if (not a44._numpy_fp64_comparison_boundary(1e-6, 1e-6)
            or a44._numpy_fp64_comparison_boundary(
                1e-6 + 1e-10, 1e-6,
            )
            or not a44._numpy_fp64_integer_boundary(
                float('nan'), float('nan'),
            )):
        raise AssertionError('NumPy derived comparison guard differs')
    proof = _cpu_raw_repair_activity(
        cells[2], a4.a3.REPAIR_PROOFREADING,
    )
    proof_fraction = proof / (0.75 + proof)
    proof_atp_gate = (
        float(a4.g2.REPLICATION_ATP_PER_SYMBOL)
        + 0.00075 * proof_fraction + 0.022
    )
    payment_boundaries = []
    for atp_value in (
            proof_atp_gate, np.nextafter(proof_atp_gate, 0.0)):
        payment_cell = copy.deepcopy(cells[2])
        payment_cell.pools[a4.a3.POOL_ATP] = atp_value
        payment_ragged, payment_state, payment_binding = (
            _paid_replication_binding(
                [payment_cell], world.config, config,
            )
        )
        _assert_raises(
            a44.A4ReplicationScopeError,
            lambda payment_binding=payment_binding:
                a44.paid_replication_elongation_numpy(
                    payment_binding, dt, world.config,
                ),
        )
        payment_boundaries.append((payment_ragged, payment_state))

    # Construct an actual plan row on the derived replicase gate while
    # preserving the active-protein pool ledger.  This pins code 6 at the
    # public plan boundary rather than testing only the comparison helper.
    replicase_cell = copy.deepcopy(cells[0])
    replicase_ragged, replicase_state, initial_replicase_binding = (
        _paid_replication_binding(
            [replicase_cell], world.config, config,
        )
    )
    replicase_specs = (
        initial_replicase_binding.cache.materialize_gene_specs_host()[0]
    )
    replicase_positions = []
    nonreplicase_positions = []
    for position in range(int(replicase_state.active_count[0])):
        fingerprint = int(
            replicase_state.active_fingerprints[0, position]
        )
        spec = replicase_specs.get(fingerprint)
        if (spec is not None
                and int(spec['role']) == int(a4.a3.ROLE_REPLICASE)):
            replicase_positions.append(position)
        else:
            nonreplicase_positions.append(position)
    if len(replicase_positions) != 1 or not nonreplicase_positions:
        raise AssertionError('replicase boundary fixture shape drifted')
    rep_position = replicase_positions[0]
    donor_position = nonreplicase_positions[0]
    rep_fingerprint = int(
        replicase_state.active_fingerprints[0, rep_position]
    )
    efficiency = float(replicase_specs[rep_fingerprint]['efficiency'])
    pools = replicase_state.pools[0]
    volume = max(
        0.20,
        (float(replicase_state.radius[0])
         / float(a4.s5.BASE_RADIUS)) ** 2,
    )
    aggregate_concentration = float(
        pools[a4.a3.POOL_AGGREGATE] / volume
    )
    active = max(0.0, float(pools[a4.a3.POOL_CATALYST]))
    damaged = float(pools[a4.a3.POOL_DAMAGED_PROTEIN])
    aggregate_pool = float(pools[a4.a3.POOL_AGGREGATE])
    functional = active / max(
        1e-9, active + damaged + aggregate_pool,
    )
    toxicity = 1.0 / (1.0 + 3.6 * aggregate_concentration)
    proteostasis = float(np.clip(functional * toxicity, 0.02, 1.0))
    genome_factor = 1.0 / (
        1.0 + 0.85 * float(replicase_state.genome_lesion_mean[0])
    )
    target_mass = (
        (1e-6 * 0.040 / (proteostasis * genome_factor)) / efficiency
    )
    old_mass = float(replicase_state.active_mass[0, rep_position])
    transfer = old_mass - target_mass
    replicase_state.active_mass[0, rep_position] = target_mass
    replicase_state.active_mass[0, donor_position] += transfer
    a4.validate_a4_translation_state(replicase_state)
    replicase_boundary_binding = a4.bind_a4_translation(
        replicase_ragged, replicase_state,
    )
    replicase_boundary_config = _replication_model_config(
        world.config, proofreading=True, external_replicase=False,
        quiescence=True, quiescence_effector=True,
    )
    observed_replicase = a44._numpy_replicase(
        replicase_boundary_binding, 0, replicase_specs,
    )
    if not a44._numpy_fp64_comparison_boundary(
            observed_replicase, 1e-6):
        raise AssertionError('actual replicase row missed comparison guard')
    _assert_raises(
        a44.A4ReplicationScopeError,
        lambda: a44.paid_replication_elongation_numpy(
            replicase_boundary_binding, dt, replicase_boundary_config,
        ),
    )

    overflow_ragged, overflow_state, initial_overflow_binding = (
        _paid_replication_binding(
            [copy.deepcopy(cells[0])], world.config, config,
        )
    )
    overflow_specs = (
        initial_overflow_binding.cache.materialize_gene_specs_host()[0]
    )
    proof_positions = []
    for position in range(int(overflow_state.active_count[0])):
        fingerprint = int(overflow_state.active_fingerprints[0, position])
        spec = overflow_specs.get(fingerprint)
        if (spec is not None
                and int(spec['role']) == int(a4.a3.ROLE_REGULATOR)
                and int(spec['localisation']) == int(a4.s4.LOC_REPAIR)
                and int(spec['parameter']) % int(a4.a3.REPAIR_COUNT)
                == int(a4.a3.REPAIR_PROOFREADING)):
            proof_positions.append(position)
    if not proof_positions:
        raise AssertionError('overflow fixture lacks proof regulator')
    overflow_state.active_mass[0, proof_positions[0]] = 1e308
    ordered_active_total = 0.0
    for position in range(int(overflow_state.active_count[0])):
        ordered_active_total += float(overflow_state.active_mass[0, position])
    overflow_state.pools[0, a4.a3.POOL_CATALYST] = ordered_active_total
    a4.validate_a4_translation_state(overflow_state)
    overflow_binding = a4.bind_a4_translation(
        overflow_ragged, overflow_state,
    )
    _assert_raises(
        a44.A4ReplicationScopeError,
        lambda: a44.paid_replication_elongation_numpy(
            overflow_binding, dt, world.config,
        ),
    )

    # Pin the late proofreading-payment boundary that motivated the 4096-eps
    # comparison band.  The ambiguous comparison occurs after 405 accepted
    # symbols: narrower 8/512/2048-eps bands allowed the CUDA row to continue
    # while NumPy/Torch CPU failed closed.  Keep this as an actual public plan
    # row, not only a helper-level comparison.
    late_payment_cell = copy.deepcopy(cells[0])
    late_proof_fingerprint = 66246424745
    late_proof_found = False
    for fingerprint, spec in late_payment_cell.gene_specs.items():
        if (int(spec['role']) == int(a4.a3.ROLE_REGULATOR)
                and int(spec['localisation']) == int(a4.s4.LOC_REPAIR)
                and int(spec['parameter']) % int(a4.a3.REPAIR_COUNT)
                == int(a4.a3.REPAIR_PROOFREADING)):
            late_payment_cell.proteins[fingerprint] = 0.0
            if int(fingerprint) == late_proof_fingerprint:
                late_proof_found = True
    if not late_proof_found:
        raise AssertionError('late payment proof fingerprint drifted')
    late_payment_cell.proteins[late_proof_fingerprint] = float.fromhex(
        '0x1.a027936981d40p-3'
    )
    late_payment_cell._sync_protein_pool()
    late_payment_cell.radius = float.fromhex('0x1.3c3336c953ab2p-6')
    late_payment_cell.pools[a4.a3.POOL_AGGREGATE] = float.fromhex(
        '0x1.c6be302cd832cp-3'
    )
    late_payment_cell.pools[a4.a3.POOL_ATP] = float.fromhex(
        '0x1.851a9a4931017p-1'
    )
    late_payment_cell.pools[a4.a3.POOL_NUCLEOTIDE] = 0.75
    late_payment_cell.replication_fractional = 0.2718281828459045
    if (len(late_payment_cell.replication_template) != 576
            or len(late_payment_cell.replication_copy) != 8):
        raise AssertionError('late payment template/copy shape drifted')
    late_ragged, late_state, late_binding = _paid_replication_binding(
        [late_payment_cell], world.config, config,
    )
    late_dt = 16.0
    late_specs = late_binding.cache.materialize_gene_specs_host()[0]
    late_proof = _cpu_raw_repair_activity(
        late_payment_cell, a4.a3.REPAIR_PROOFREADING,
    )
    late_fraction = late_proof / (0.75 + late_proof)
    late_gate = (
        float(a4.g2.REPLICATION_ATP_PER_SYMBOL)
        + 0.00075 * late_fraction + 0.022
    )
    if (late_gate.hex() != '0x1.863a7e0fc1aeap-6'
            or late_specs[late_proof_fingerprint]['copy_number'] < 1):
        raise AssertionError('late payment arithmetic fixture drifted')
    _assert_raises(
        a44.A4ReplicationScopeError,
        lambda: a44.paid_replication_elongation_numpy(
            late_binding, late_dt, world.config,
        ),
    )
    world_before = v3.pickle_clone(world.state_dict())
    source = '\n'.join(inspect.getsource(item) for item in (
        a44.paid_replication_elongation_torch,
        a44._torch_replicase,
        a44._torch_numpy_pairwise_sum_rows,
        a44._torch_fp64_integer_boundary,
        a44._torch_fp64_comparison_boundary,
        a4._torch_ordered_row_sum,
    ))
    forbidden = ('.item(', '.cpu(', '.numpy(', '.tolist(',
                 'nonzero(', 'masked_select(', 'unique(', 'cumsum(')
    found = [token for token in forbidden if token in source]
    if found:
        raise AssertionError('resident A4.4b contains host/dynamic op: %s' % found)

    deterministic_before = torch.are_deterministic_algorithms_enabled()
    warn_before = torch.is_deterministic_algorithms_warn_only_enabled()
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    try:
        torch.use_deterministic_algorithms(True)
        for device in devices:
            resident_ragged = ragged.to_torch(device=device)
            resident_state = state.to_torch(device=device)
            ragged_ptrs = resident_ragged.data_ptrs()
            state_ptrs = resident_state.data_ptrs()
            # Direct bind exercises deterministic resident gene decode.
            resident_binding = a4.bind_a4_translation(
                resident_ragged, resident_state,
            )
            cache_ptrs = resident_binding.cache.data_ptrs()
            translation = a4.paid_translation_plan_torch(
                resident_binding, dt,
            )
            plan = a44.paid_replication_elongation_torch(
                resident_binding, dt, world.config,
            )
            boundary_resident = a4.bind_a4_translation(
                boundary_ragged.to_torch(device=device),
                boundary_state.to_torch(device=device),
            )
            boundary_plan = a44.paid_replication_elongation_torch(
                boundary_resident, boundary_dt, world.config,
            )
            control_plan = a44.paid_replication_elongation_torch(
                boundary_resident, control_dt, world.config,
            ).to_numpy()
            division_resident = a4.bind_a4_translation(
                division_ragged.to_torch(device=device),
                division_state.to_torch(device=device),
            )
            division_plan = a44.paid_replication_elongation_torch(
                division_resident, division_dt, world.config,
            )
            payment_plans = []
            for payment_ragged, payment_state in payment_boundaries:
                payment_resident = a4.bind_a4_translation(
                    payment_ragged.to_torch(device=device),
                    payment_state.to_torch(device=device),
                )
                payment_plans.append(
                    a44.paid_replication_elongation_torch(
                        payment_resident, dt, world.config,
                    )
                )
            replicase_resident = a4.bind_a4_translation(
                replicase_ragged.to_torch(device=device),
                replicase_state.to_torch(device=device),
            )
            replicase_boundary_plan = a44.paid_replication_elongation_torch(
                replicase_resident, dt, replicase_boundary_config,
            )
            overflow_resident = a4.bind_a4_translation(
                overflow_ragged.to_torch(device=device),
                overflow_state.to_torch(device=device),
            )
            overflow_plan = a44.paid_replication_elongation_torch(
                overflow_resident, dt, world.config,
            )
            late_resident = a4.bind_a4_translation(
                late_ragged.to_torch(device=device),
                late_state.to_torch(device=device),
            )
            late_plan = a44.paid_replication_elongation_torch(
                late_resident, late_dt, world.config,
            )
            if not torch.are_deterministic_algorithms_enabled():
                raise AssertionError('A4.4b disabled deterministic algorithms')
            if any(getattr(plan, name).device.type != device
                   for name in a44._PLAN_ARRAY_FIELDS):
                raise AssertionError('%s A4.4b plan escaped device' % device)
            plan_ptrs = plan.data_ptrs()
            if device == 'cuda':
                torch.cuda.synchronize()
            v3.assert_recursive_close(
                expected_translation.state_dict(),
                translation.to_numpy().state_dict(),
                atol=2e-12, rtol=0.0,
                path='a44b.translation.%s' % device,
            )
            v3.assert_recursive_close(
                expected.state_dict(), plan.to_numpy().state_dict(),
                atol=2e-12, rtol=0.0, path='a44b.plan.%s' % device,
            )
            boundary_code = int(
                boundary_plan.scope_error_code.detach().cpu().numpy()[0]
            )
            division_code = int(
                division_plan.scope_error_code.detach().cpu().numpy()[0]
            )
            if (boundary_code != a44.SCOPE_FP64_DISCRETE_BOUNDARY
                    or division_code != a44.SCOPE_FP64_DISCRETE_BOUNDARY):
                raise AssertionError(
                    '%s fp64 integer guard codes differ: %d/%d' % (
                        device, boundary_code, division_code,
                    )
                )
            _assert_raises(
                a44.A4ReplicationScopeError, boundary_plan.to_numpy,
            )
            _assert_raises(
                a44.A4ReplicationScopeError, division_plan.to_numpy,
            )
            for payment_plan in payment_plans:
                payment_code = int(
                    payment_plan.scope_error_code.detach().cpu().numpy()[0]
                )
                if payment_code != a44.SCOPE_FP64_DISCRETE_BOUNDARY:
                    raise AssertionError(
                        '%s proof ATP guard code differs: %d' % (
                            device, payment_code,
                        )
                    )
                _assert_raises(
                    a44.A4ReplicationScopeError, payment_plan.to_numpy,
                )
            replicase_code = int(
                replicase_boundary_plan.scope_error_code.detach().cpu()
                .numpy()[0]
            )
            if replicase_code != a44.SCOPE_FP64_DISCRETE_BOUNDARY:
                raise AssertionError(
                    '%s actual replicase guard code differs: %d' % (
                        device, replicase_code,
                    )
                )
            if (int(replicase_boundary_plan.requested_symbols.detach().cpu()
                    .numpy()[0]) != 0
                    or int(replicase_boundary_plan.append_count.detach().cpu()
                           .numpy()[0]) != 0
                    or float(
                        replicase_boundary_plan.replication_fractional_after
                        .detach().cpu().numpy()[0]
                    ) != 0.0):
                raise AssertionError(
                    '%s replicase-boundary telemetry was speculative' % device
                )
            _assert_raises(
                a44.A4ReplicationScopeError,
                replicase_boundary_plan.to_numpy,
            )
            overflow_code = int(
                overflow_plan.scope_error_code.detach().cpu().numpy()[0]
            )
            if overflow_code != a44.SCOPE_FP64_DISCRETE_BOUNDARY:
                raise AssertionError(
                    '%s nonfinite kinetics guard code differs: %d' % (
                        device, overflow_code,
                    )
                )
            if (int(overflow_plan.requested_symbols.detach().cpu()
                    .numpy()[0]) != 0
                    or int(overflow_plan.append_count.detach().cpu()
                           .numpy()[0]) != 0
                    or float(
                        overflow_plan.replication_fractional_after
                        .detach().cpu().numpy()[0]
                    ) != 0.0):
                raise AssertionError(
                    '%s nonfinite kinetics telemetry was speculative' % device
                )
            _assert_raises(
                a44.A4ReplicationScopeError, overflow_plan.to_numpy,
            )
            late_code = int(
                late_plan.scope_error_code.detach().cpu().numpy()[0]
            )
            if late_code != a44.SCOPE_FP64_DISCRETE_BOUNDARY:
                raise AssertionError(
                    '%s late proof-payment guard code differs: %d' % (
                        device, late_code,
                    )
                )
            if (int(late_plan.requested_symbols.detach().cpu().numpy()[0])
                    != 0
                    or int(late_plan.append_count.detach().cpu().numpy()[0])
                    != 0
                    or float(
                        late_plan.replication_fractional_after.detach().cpu()
                        .numpy()[0]
                    ) != 0.0):
                raise AssertionError(
                    '%s late proof-payment telemetry was speculative' % device
                )
            _assert_raises(
                a44.A4ReplicationScopeError, late_plan.to_numpy,
            )
            v3.assert_recursive_close(
                control_expected.state_dict(), control_plan.state_dict(),
                atol=2e-12, rtol=0.0,
                path='a44b.nonambiguous_control.%s' % device,
            )
            pairwise_values = (
                np.clip(
                    np.asarray(boundary_state.membrane_oxidation[:1]),
                    0.0, 2.5,
                )
                * np.maximum(
                    np.asarray(boundary_state.membrane[:1]), 1e-9,
                )
            )
            pairwise_expected = np.sum(
                pairwise_values, axis=1, dtype=np.float64,
            )
            pairwise_actual = a44._torch_numpy_pairwise_sum_rows(
                torch.as_tensor(
                    pairwise_values, dtype=torch.float64, device=device,
                )
            ).detach().cpu().numpy()
            if not np.array_equal(
                    pairwise_expected.view(np.uint64),
                    pairwise_actual.view(np.uint64)):
                raise AssertionError(
                    '%s NumPy pairwise fp64 fold differs' % device
                )
            comparison_probe = torch.as_tensor(
                [1e-6, 1e-6 + 1e-10],
                dtype=torch.float64, device=device,
            )
            comparison_guard = a44._torch_fp64_comparison_boundary(
                comparison_probe, 1e-6,
            ).detach().cpu().numpy()
            if not np.array_equal(
                    comparison_guard, np.asarray([True, False])):
                raise AssertionError(
                    '%s derived comparison guard differs' % device
                )
            nonfinite_guard = a44._torch_fp64_integer_boundary(
                torch.as_tensor(
                    [float('nan'), 0.25],
                    dtype=torch.float64, device=device,
                ),
                torch.as_tensor(
                    [float('nan'), 0.0],
                    dtype=torch.float64, device=device,
                ),
            ).detach().cpu().numpy()
            if not np.array_equal(
                    nonfinite_guard, np.asarray([True, False])):
                raise AssertionError(
                    '%s nonfinite progress guard differs' % device
                )
            if (ragged_ptrs != resident_ragged.data_ptrs()
                    or state_ptrs != resident_state.data_ptrs()
                    or cache_ptrs != resident_binding.cache.data_ptrs()
                    or plan_ptrs != plan.data_ptrs()):
                raise AssertionError('%s resident pointer changed' % device)
            v3.assert_recursive_close(
                ragged.state_dict(), resident_ragged.to_numpy().state_dict(),
                atol=0.0, rtol=0.0,
                path='a44b.ragged_source.%s' % device,
            )
            v3.assert_recursive_close(
                state.state_dict(), resident_state.to_numpy().state_dict(),
                atol=0.0, rtol=0.0,
                path='a44b.state_source.%s' % device,
            )
    finally:
        torch.use_deterministic_algorithms(
            deterministic_before, warn_only=warn_before,
        )
    v3.assert_recursive_close(
        world_before, world.state_dict(), atol=0.0, rtol=0.0,
        path='a44b.device_world',
    )
    return ('deterministic=True direct gene decode/translation/replication '
            'NumPy/Torch %s; pairwise fold exact + two fp64 integer '
            'boundaries plus first/late proof-ATP, replicase, and nonfinite '
            'comparisons '
            'fail closed + control/pointers exact') % '/'.join(devices)


def test_a44b_capacity_scope_and_a3_authority():
    world, cells, roomy_config, dt = _paid_replication_b_fixture(seed=7904)
    single = [cells[0]]
    probe = a4.FullFidelityA4GenomeAdapter(roomy_config).pack_cells(single)
    common = {
        'max_cells': 1,
        'max_sequences': int(probe.sequence_count),
        'max_sequence_symbols': int(roomy_config.max_sequence_symbols),
        'max_proteins_per_cell': int(roomy_config.max_proteins_per_cell),
    }
    exact_config = a4.GPU068A4Config(
        max_symbols=int(probe.symbol_count) + 1, **common
    )
    exact_ragged, _, exact_binding = _paid_replication_binding(
        single, world.config, exact_config,
    )
    exact_plan = a44.paid_replication_elongation_numpy(
        exact_binding, dt, world.config,
    )
    if (int(exact_ragged.symbol_capacity) != int(exact_ragged.symbol_count) + 1
            or int(exact_plan.append_count[0]) != 1):
        raise AssertionError('A4.4b exact future capacity did not pass')
    full_config = a4.GPU068A4Config(
        max_symbols=int(probe.symbol_count), **common
    )
    _, _, full_binding = _paid_replication_binding(
        single, world.config, full_config,
    )
    _assert_replication_failure_atomic(
        a4.A4CapacityError, full_binding,
        lambda: a44.paid_replication_elongation_numpy(
            full_binding, dt, world.config,
        ),
        'a44b.capacity_plus_one',
    )

    negative_atp = copy.deepcopy(cells[0])
    negative_atp.pools[a4.a3.POOL_ATP] = -1e-12
    negative_atp.replication_fractional = 0.5
    negative_dt = 1e6
    _, negative_state, negative_binding = _paid_replication_binding(
        [negative_atp], world.config, roomy_config,
    )
    _assert_replication_failure_atomic(
        a44.A4ReplicationScopeError, negative_binding,
        lambda: a44.paid_replication_elongation_numpy(
            negative_binding, negative_dt, world.config,
        ),
        'a44b.negative_atp',
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    negative_ragged = negative_binding.ragged.to_torch(device)
    negative_resident = a4.bind_a4_translation(
        negative_ragged, negative_state.to_torch(device),
    )
    negative_plan = a44.paid_replication_elongation_torch(
        negative_resident, negative_dt, world.config,
    )
    if (int(negative_plan.scope_error_code[0])
            != int(a44.SCOPE_NEGATIVE_ATP)
            or int(negative_plan.requested_symbols[0]) != 0
            or float(negative_plan.replication_fractional_after[0]) != 0.0):
        raise AssertionError('negative ATP entered resident rate telemetry')
    _assert_raises(a44.A4ReplicationScopeError, negative_plan.to_numpy)

    zero_genome = copy.deepcopy(cells[0])
    zero_genome.genomes = []
    zero_genome.genome_lesions = []
    zero_genome._refresh_gene_cache()
    _, _, zero_binding = _paid_replication_binding(
        [zero_genome], world.config, roomy_config,
    )
    _assert_replication_failure_atomic(
        a44.A4ReplicationScopeError, zero_binding,
        lambda: a44.paid_replication_elongation_numpy(
            zero_binding, dt, world.config,
        ),
        'a44b.zero_genome_active_template',
    )

    mismatched = copy.deepcopy(world.config)
    mismatched.quiescence = not mismatched.quiescence
    _assert_replication_failure_atomic(
        a44.A4ReplicationScopeError, exact_binding,
        lambda: a44.paid_replication_elongation_numpy(
            exact_binding, dt, mismatched,
        ),
        'a44b.quiescence_metadata_mismatch',
    )
    mutation = copy.deepcopy(world.config)
    mutation.mutation = True
    _assert_replication_failure_atomic(
        a44.A4ReplicationScopeError, exact_binding,
        lambda: a44.paid_replication_elongation_numpy(
            exact_binding, dt, mutation,
        ),
        'a44b.mutation_scope',
    )
    external_only = _external_only_replication_cell(cells[0])
    no_external = _replication_model_config(world.config)
    _, _, no_external_binding = _paid_replication_binding(
        [external_only], no_external, roomy_config,
    )
    _assert_replication_failure_atomic(
        a44.A4ReplicationScopeError, no_external_binding,
        lambda: a44.paid_replication_elongation_numpy(
            no_external_binding, dt, no_external,
        ),
        'a44b.replicase_gate',
    )

    completing = copy.deepcopy(cells[0])
    completing.replication_copy = [
        int(value) for value in completing.replication_template[:-1]
    ]
    completing.replication_fractional = np.nextafter(1.0, 0.0)
    _, _, completing_binding = _paid_replication_binding(
        [completing], world.config, roomy_config,
    )
    _assert_replication_failure_atomic(
        a44.A4ReplicationScopeError, completing_binding,
        lambda: a44.paid_replication_elongation_numpy(
            completing_binding, dt, world.config,
        ),
        'a44b.completion',
    )

    completion_probe = a4.FullFidelityA4GenomeAdapter(
        roomy_config,
    ).pack_cells([completing])
    completion_capacity = a4.GPU068A4Config(
        max_cells=1,
        max_sequences=int(completion_probe.sequence_count),
        max_symbols=int(completion_probe.symbol_count),
        max_sequence_symbols=int(roomy_config.max_sequence_symbols),
        max_proteins_per_cell=int(roomy_config.max_proteins_per_cell),
    )
    completion_ragged, completion_state, completion_capacity_binding = (
        _paid_replication_binding(
            [completing], world.config, completion_capacity,
        )
    )
    _assert_replication_failure_atomic(
        a4.A4CapacityError, completion_capacity_binding,
        lambda: a44.paid_replication_elongation_numpy(
            completion_capacity_binding, dt, world.config,
        ),
        'a44b.completion_capacity_priority',
    )
    completion_resident = a4.bind_a4_translation(
        completion_ragged.to_torch(device),
        completion_state.to_torch(device),
    )
    completion_capacity_plan = a44.paid_replication_elongation_torch(
        completion_resident, dt, world.config,
    )
    _assert_raises(
        a4.A4CapacityError, completion_capacity_plan.to_numpy,
    )

    forged = exact_plan.clone()
    forged.cumulative_proofreading_atp_after[0] = -1e-12
    _assert_raises(
        a4.A4SchemaError,
        lambda: a44.validate_a4_paid_elongation_plan(forged),
    )
    forged_atp = exact_plan.clone()
    forged_atp.pools_after[0, a4.a3.POOL_ATP] = -1e-12
    _assert_raises(
        a4.A4SchemaError,
        lambda: a44.validate_a4_paid_elongation_plan(forged_atp),
    )
    expected_status = (
        'a4.6b2-pre-existing-active-all-row-completion-combined-pcg64-'
        'substitution-structural-material-fixed-resident-plan-'
        'not-arena-committed-not-integrated-cpu-authoritative'
    )
    if (a44.PORT_STATUS.get('genome_replication') != expected_status
            or a44.FULL_GPU_WORLD_STEP is not False
            or a44.PORT_STATUS.get('full_gpu_world_step') is not False):
        raise AssertionError('A4.6b2 CPU authority status differs')
    if (a4.PORT_STATUS.get('genome_replication')
            != 'cpu-authoritative-next-a4-slice'):
        raise AssertionError('A4 core replication authority changed')
    if (getattr(v3.a3_module(), 'FULL_GPU_WORLD_STEP', None) is not False
            or v3._event_order().count('replication_cpu') != 1):
        raise AssertionError('A3 CPU replication authority changed')
    return 'exact/+1 and completion+capacity priority; negative ATP telemetry/zero-genome/config/mutation/replicase/completion fail closed; A3 authority'


def _a45_substitution_fixture(seed=8002):
    world, cells, capacity, _ = _paid_replication_b_fixture(seed=seed)
    world.config.mutation = True
    world.config.mutation_rate = 0.5
    return world, cells[:3], capacity, 0.5


def _a45b_template_start_fixture(seed=8201, prime_uint32=True):
    """Two inactive starts around one active row, all non-completing."""
    world, cells, capacity, _ = _paid_replication_b_fixture(seed=seed)
    world.config.mutation = True
    world.config.mutation_rate = 0.45
    selected = [copy.deepcopy(cell) for cell in cells[:3]]
    for ci in (0, 2):
        cell = selected[ci]
        if len(cell.genomes) != 1 or len(cell.genomes[0]) == 0:
            raise AssertionError('A4.5b start fixture needs one nonempty genome')
        cell.replication_template = None
        cell.replication_copy = []
        cell.replication_template_lesion = 0.0
        cell.replication_fractional = 0.0
        cell.last_replication_symbols = 81 + ci
        cell.last_effective_error_rate = 0.75 + 0.01 * ci
    # The second start fixes the frozen missing-lesion fallback to exactly 0.
    selected[2].genome_lesions = []
    # Exercise the full PCG64 state, including a pre-existing cached uint32.
    if prime_uint32:
        world.rng.integers(0, 7)
    expected_cache = 1 if prime_uint32 else 0
    if int(world.rng.bit_generator.state['has_uint32']) != expected_cache:
        raise AssertionError('A4.5b fixture PCG64 uint32 cache differs')
    return world, selected, capacity, 0.5


def test_a45_substitution_rng_tape_formal066_oracle():
    world, cells, capacity, dt = _a45_substitution_fixture()
    world_before = v3.pickle_clone(world.state_dict())
    rng_before = copy.deepcopy(world.rng.bit_generator.state)
    ragged, state, binding = _paid_replication_binding(
        cells, world.config, capacity,
    )
    ragged_before = ragged.state_dict()
    state_before = state.state_dict()
    cache_before = binding.cache.state_dict()
    tape = a44.prepare_substitution_rng_tape(
        binding, dt, world.config, rng_before,
    )
    plan = a44.paid_replication_substitution_numpy(
        binding, dt, world.config, tape,
    )

    cpu_world = copy.deepcopy(world)
    cpu_before_cells = [copy.deepcopy(cell) for cell in cpu_world.cells[:3]]
    for cell in cpu_world.cells[:3]:
        cell._replicate_genome(cpu_world, dt, cpu_world.config)
    _assert_paid_replication_cpu_parity(
        plan, cpu_before_cells, cpu_world.cells[:3], 'a45.cpu_oracle',
    )
    if tape.rng_after_state != cpu_world.rng.bit_generator.state:
        raise AssertionError('A4.5a PCG64 after-state differs from Formal066')
    if not np.array_equal(tape.draw_count, plan.append_count):
        raise AssertionError('A4.5a threshold draw count differs from append')
    if (int(np.sum(tape.substitution_count)) <= 0
            or int(np.sum(tape.substitution_count))
            >= int(np.sum(tape.draw_count))):
        raise AssertionError('A4.5a fixture lacks mixed hit/miss draws')
    if not np.any(plan.requested_symbols[:3] > plan.append_count[:3]):
        raise AssertionError('A4.5a fixture lacks resource-stopped requests')
    for ci, before in enumerate(cpu_before_cells):
        offset = len(before.replication_copy)
        count = int(plan.append_count[ci])
        template = np.asarray(before.replication_template, dtype=np.uint8)
        for rank in range(count):
            old = int(template[offset + rank])
            new = int(plan.append_symbols[ci, rank])
            if bool(tape.replacement_mask[ci, rank]) != (new != old):
                raise AssertionError('A4.5a exclude-old mapping differs')

    # mutation_rate=0 still consumes one scalar random() per accepted symbol,
    # but never consumes the conditional bounded integer draw.
    zero_world, zero_cells, zero_capacity, zero_dt = _paid_replication_fixture(
        seed=8003,
    )
    zero_world.config.mutation = True
    zero_world.config.mutation_rate = 0.0
    zero_cells[0].replication_template_lesion = 0.0
    zero_cells[0].pools[a4.a3.POOL_REACTIVE] = 0.0
    _, _, zero_binding = _paid_replication_binding(
        [zero_cells[0]], zero_world.config, zero_capacity,
    )
    zero_before = copy.deepcopy(zero_world.rng.bit_generator.state)
    zero_tape = a44.prepare_substitution_rng_tape(
        zero_binding, zero_dt, zero_world.config, zero_before,
    )
    if (int(zero_tape.draw_count[0]) <= 0
            or int(zero_tape.substitution_count[0]) != 0
            or np.any(zero_tape.replacement_mask)):
        raise AssertionError('zero-error RNG consumption differs')
    replay = np.random.Generator(np.random.PCG64())
    replay.bit_generator.state = copy.deepcopy(zero_before)
    for _ in range(int(zero_tape.draw_count[0])):
        replay.random()
    if replay.bit_generator.state != zero_tape.rng_after_state:
        raise AssertionError('zero-error threshold-only RNG state differs')

    if world.rng.bit_generator.state != rng_before:
        raise AssertionError('A4.5a changed the live world RNG')
    v3.assert_recursive_close(
        world_before, world.state_dict(), atol=0.0, rtol=0.0,
        path='a45.source_world',
    )
    for label, expected, actual in (
        ('ragged', ragged_before, ragged.state_dict()),
        ('state', state_before, state.state_dict()),
        ('cache', cache_before, binding.cache.state_dict()),
    ):
        v3.assert_recursive_close(
            expected, actual, atol=0.0, rtol=0.0,
            path='a45.source_%s' % label,
        )
    return ('3 Formal066 rows: scalar random per paid symbol, conditional '
            'integer draws, suffix/events/PCG64 after-state exact')


def test_a45_substitution_rng_tape_numpy_torch_devices():
    if torch is None:
        raise AssertionError('PyTorch is required for A4.5a')
    if _REQUIRE_CUDA and not torch.cuda.is_available():
        raise AssertionError('CUDA required but unavailable; fallback forbidden')
    world, cells, capacity, dt = _a45_substitution_fixture(seed=8004)
    ragged, state, binding = _paid_replication_binding(
        cells, world.config, capacity,
    )
    rng_before = copy.deepcopy(world.rng.bit_generator.state)
    tape = a44.prepare_substitution_rng_tape(
        binding, dt, world.config, rng_before,
    )
    expected = a44.paid_replication_substitution_numpy(
        binding, dt, world.config, tape,
    )
    ragged_before = ragged.state_dict()
    state_before = state.state_dict()
    cache_before = binding.cache.state_dict()
    source = inspect.getsource(a44.paid_replication_substitution_torch)
    forbidden = (
        '.item(', '.cpu(', '.numpy(', '.tolist(', 'nonzero(',
        'masked_select(', 'unique(',
    )
    found = [token for token in forbidden if token in source]
    if found:
        raise AssertionError('resident substitution contains host op: %s' % found)
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    for device in devices:
        resident_ragged = ragged.to_torch(device=device)
        resident_state = state.to_torch(device=device)
        resident = a4.bind_a4_translation(resident_ragged, resident_state)
        resident_tape = tape.to_torch(device=device)
        ragged_ptrs = resident_ragged.data_ptrs()
        state_ptrs = resident_state.data_ptrs()
        cache_ptrs = resident.cache.data_ptrs()
        tape_ptrs = resident_tape.data_ptrs()
        plan = a44.paid_replication_substitution_torch(
            resident, dt, world.config, resident_tape,
        )
        if any(getattr(plan, name).device.type != device
               for name in a44._PLAN_ARRAY_FIELDS):
            raise AssertionError('%s A4.5a output escaped device' % device)
        v3.assert_recursive_close(
            expected.state_dict(), plan.to_numpy().state_dict(),
            atol=2e-12, rtol=0.0, path='a45.plan.%s' % device,
        )
        v3.assert_recursive_close(
            tape.state_dict(), resident_tape.to_numpy().state_dict(),
            atol=0.0, rtol=0.0, path='a45.tape.%s' % device,
        )
        if (ragged_ptrs != resident_ragged.data_ptrs()
                or state_ptrs != resident_state.data_ptrs()
                or cache_ptrs != resident.cache.data_ptrs()
                or tape_ptrs != resident_tape.data_ptrs()):
            raise AssertionError('%s A4.5a resident pointer changed' % device)
    if world.rng.bit_generator.state != rng_before:
        raise AssertionError('device parity changed the live RNG')
    for label, before, after in (
        ('ragged', ragged_before, ragged.state_dict()),
        ('state', state_before, state.state_dict()),
        ('cache', cache_before, binding.cache.state_dict()),
    ):
        v3.assert_recursive_close(
            before, after, atol=0.0, rtol=0.0,
            path='a45.device_source.%s' % label,
        )
    return 'NumPy/Torch %s fixed-tape suffix/events exact; source/RNG/pointers stable' % '/'.join(devices)


def test_a45_substitution_rng_tape_fail_closed_and_authority():
    if torch is None:
        raise AssertionError('PyTorch is required for A4.5a trust checks')
    world, cells, capacity, dt = _a45_substitution_fixture(seed=8005)
    ragged, state, binding = _paid_replication_binding(
        [cells[0]], world.config, capacity,
    )
    rng_before = copy.deepcopy(world.rng.bit_generator.state)
    tape = a44.prepare_substitution_rng_tape(
        binding, dt, world.config, rng_before,
    )

    forged = tape.clone()
    forged._factory_token = None
    _assert_raises(
        a4.A4SchemaError,
        lambda: a44.validate_a4_substitution_rng_tape(forged),
    )
    bad_after = tape.clone()
    bad_after.rng_after_state['state']['state'] ^= 1
    _assert_raises(
        a4.A4SchemaError,
        lambda: a44.validate_a4_substitution_rng_tape(bad_after),
    )
    bad_tail = tape.clone()
    bad_tail.uniform_draws[0, int(bad_tail.draw_count[0])] = 0.25
    _assert_raises(
        a4.A4SchemaError,
        lambda: a44.validate_a4_substitution_rng_tape(bad_tail),
    )
    _assert_raises(
        a44.A4ReplicationScopeError,
        lambda: a44.paid_replication_substitution_numpy(
            binding, dt + 1e-6, world.config, tape,
        ),
    )
    disabled = copy.deepcopy(world.config)
    disabled.mutation = False
    _assert_raises(
        a44.A4ReplicationScopeError,
        lambda: a44.prepare_substitution_rng_tape(
            binding, dt, disabled, rng_before,
        ),
    )
    noncanonical = copy.deepcopy(rng_before)
    noncanonical['bit_generator'] = 'PCG64DXSM'
    _assert_raises(
        a4.A4SchemaError,
        lambda: a44.prepare_substitution_rng_tape(
            binding, dt, world.config, noncanonical,
        ),
    )

    # An exact threshold comparison is deliberately CPU-authoritative.
    boundary_world, boundary_cells, boundary_capacity, boundary_dt = (
        _paid_replication_fixture(seed=8006)
    )
    boundary_world.config.mutation = True
    boundary_world.config.proofreading = False
    boundary_world.config.quiescence = False
    boundary_world.config.quiescence_effector = False
    boundary_cells[0].replication_template_lesion = 0.0
    boundary_cells[0].pools[a4.a3.POOL_REACTIVE] = 0.0
    boundary_state_before = copy.deepcopy(
        boundary_world.rng.bit_generator.state,
    )
    probe = np.random.Generator(np.random.PCG64())
    probe.bit_generator.state = copy.deepcopy(boundary_state_before)
    boundary_world.config.mutation_rate = float(probe.random())
    _, _, boundary_binding = _paid_replication_binding(
        [boundary_cells[0]], boundary_world.config, boundary_capacity,
    )
    _assert_raises(
        a44.A4ReplicationScopeError,
        lambda: a44.prepare_substitution_rng_tape(
            boundary_binding, boundary_dt, boundary_world.config,
            boundary_state_before,
        ),
    )
    if boundary_world.rng.bit_generator.state != boundary_state_before:
        raise AssertionError('ambiguous tape preparation changed live RNG')

    completing = copy.deepcopy(cells[0])
    completing.replication_copy = [
        int(value) for value in completing.replication_template[:-1]
    ]
    completing.replication_fractional = np.nextafter(1.0, 0.0)
    _, _, completing_binding = _paid_replication_binding(
        [completing], world.config, capacity,
    )
    _assert_raises(
        a44.A4ReplicationScopeError,
        lambda: a44.prepare_substitution_rng_tape(
            completing_binding, dt, world.config, rng_before,
        ),
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resident = a4.bind_a4_translation(
        ragged.to_torch(device=device), state.to_torch(device=device),
    )
    resident_tape = tape.to_torch(device=device)
    resident_tape.uniform_draws[0, 0] += 0.125
    _assert_raises(
        a4.A4SchemaError,
        lambda: a44.paid_replication_substitution_torch(
            resident, dt, world.config, resident_tape,
        ),
    )

    # Scalar metadata and the nested PCG64 states are part of the resident
    # attestation, not caller-editable labels around otherwise trusted arrays.
    for component in (
            'source_provenance', 'config_sha256', 'schedule_sha256',
            'rng_before_state', 'rng_after_state'):
        changed = tape.to_torch(device=device)
        if component in (
                'source_provenance', 'config_sha256', 'schedule_sha256'):
            setattr(
                changed, component,
                ('0' if getattr(changed, component)[0] != '0' else '1')
                + getattr(changed, component)[1:],
            )
        else:
            getattr(changed, component)['state']['state'] ^= 1
        _assert_raises(
            a4.A4SchemaError,
            lambda changed=changed: a44.paid_replication_substitution_torch(
                resident, dt, world.config, changed,
            ),
        )

    # A tape from the same ragged genome but a different physiology snapshot
    # must not be reusable merely because its append count happens to match.
    foreign_world, foreign_cells, foreign_capacity, foreign_dt = (
        _a45_substitution_fixture(seed=8123)
    )
    original = copy.deepcopy(foreign_cells[2])
    altered = copy.deepcopy(original)
    altered.pools[a4.a3.POOL_REACTIVE] = 1000.0
    original_ragged, original_state, _ = _paid_replication_binding(
        [original], foreign_world.config, foreign_capacity,
    )
    _, _, altered_binding = _paid_replication_binding(
        [altered], foreign_world.config, foreign_capacity,
    )
    foreign_tape = a44.prepare_substitution_rng_tape(
        altered_binding, foreign_dt, foreign_world.config,
        copy.deepcopy(foreign_world.rng.bit_generator.state),
    )
    original_binding = a4.bind_a4_translation(
        original_ragged.to_torch(device=device),
        original_state.to_torch(device=device),
    )
    foreign_result = a44.paid_replication_substitution_torch(
        original_binding, foreign_dt, foreign_world.config,
        foreign_tape.to_torch(device=device),
    )
    foreign_back = {
        'code': foreign_result.scope_error_code.detach().cpu().numpy(),
        'valid': foreign_result.scope_valid.detach().cpu().numpy(),
        'requested': foreign_result.requested_symbols.detach().cpu().numpy(),
        'append': foreign_result.append_count.detach().cpu().numpy(),
        'events': foreign_result.substitution_events.detach().cpu().numpy(),
    }
    if (int(foreign_back['code'][0])
            != int(a44.SCOPE_RNG_TAPE_MISMATCH)
            or bool(foreign_back['valid'][0])
            or any(int(foreign_back[name][0]) != 0
                   for name in ('requested', 'append', 'events'))):
        raise AssertionError('foreign physiology tape did not rollback as code 7')
    _assert_raises(a44.A4ReplicationScopeError, foreign_result.to_numpy)

    # A schedule mismatch on a row that has become independently out of scope
    # still invalidates later rows in the shared PCG64 chain.  The cause row
    # retains its primary code; otherwise-valid followers roll back as code 7.
    chain_world, chain_cells, chain_capacity, chain_dt = (
        _a45_substitution_fixture(seed=8124)
    )
    chain_ragged, chain_state, chain_binding = _paid_replication_binding(
        chain_cells[:2], chain_world.config, chain_capacity,
    )
    chain_tape = a44.prepare_substitution_rng_tape(
        chain_binding, chain_dt, chain_world.config,
        copy.deepcopy(chain_world.rng.bit_generator.state),
    )
    invalid_chain_state = chain_state.clone()
    invalid_chain_state.pools[0, int(a4.a3.POOL_ATP)] = -1e-12
    invalid_chain = a4.bind_a4_translation(
        chain_ragged.to_torch(device=device),
        invalid_chain_state.to_torch(device=device),
    )
    chain_result = a44.paid_replication_substitution_torch(
        invalid_chain, chain_dt, chain_world.config,
        chain_tape.to_torch(device=device),
    )
    chain_code = chain_result.scope_error_code.detach().cpu().numpy()
    chain_valid = chain_result.scope_valid.detach().cpu().numpy()
    chain_append = chain_result.append_count.detach().cpu().numpy()
    if (chain_code[:2].tolist() != [
            int(a44.SCOPE_NEGATIVE_ATP),
            int(a44.SCOPE_RNG_TAPE_MISMATCH),
            ] or np.any(chain_valid[:2]) or np.any(chain_append[:2] != 0)):
        raise AssertionError('shared RNG chain mismatch was not batch-atomic')
    _assert_raises(a44.A4ReplicationScopeError, chain_result.to_numpy)

    # If the current device error lies inside the registered comparison band
    # around a recorded draw, ambiguity (code 6) takes precedence over the
    # ordinary schedule mismatch and invalidates the complete RNG event.
    deterministic, _ = a44._substitution_config(world.config)
    base_plan = a44.paid_replication_elongation_numpy(
        binding, dt, deterministic,
    )
    ranks = [
        rank for rank in range(int(tape.draw_count[0]))
        if float(tape.uniform_draws[0, rank])
        > float(base_plan.last_effective_error_rate[0])
    ]
    if not ranks:
        raise AssertionError('A4.5a fixture lacks a tunable comparison draw')
    target = float(tape.uniform_draws[0, ranks[0]])
    tuned_state = state.clone()
    reactive_index = int(a4.a3.POOL_REACTIVE)
    original_reactive = float(tuned_state.pools[0, reactive_index])
    probe_state = state.clone()
    probe_state.pools[0, reactive_index] = original_reactive + 1.0
    probe_binding = a4.bind_a4_translation(ragged, probe_state)
    probe_error = float(a44.paid_replication_elongation_numpy(
        probe_binding, dt, deterministic,
    ).last_effective_error_rate[0])
    base_error = float(base_plan.last_effective_error_rate[0])
    slope = probe_error - base_error
    if not slope > 0.0:
        raise AssertionError('reactive/error tuning slope is not positive')
    tuned_state.pools[0, reactive_index] = (
        original_reactive + (target - base_error) / slope
    )
    tuned_binding = a4.bind_a4_translation(ragged, tuned_state)
    tuned_error = float(a44.paid_replication_elongation_numpy(
        tuned_binding, dt, deterministic,
    ).last_effective_error_rate[0])
    if not a44._numpy_fp64_comparison_boundary(target, tuned_error):
        raise AssertionError('failed to construct device comparison boundary')
    resident_tuned = a4.bind_a4_translation(
        ragged.to_torch(device=device), tuned_state.to_torch(device=device),
    )
    boundary_result = a44.paid_replication_substitution_torch(
        resident_tuned, dt, world.config, tape.to_torch(device=device),
    )
    boundary_back = {
        'code': boundary_result.scope_error_code.detach().cpu().numpy(),
        'valid': boundary_result.scope_valid.detach().cpu().numpy(),
        'requested': boundary_result.requested_symbols.detach().cpu().numpy(),
        'append': boundary_result.append_count.detach().cpu().numpy(),
        'events': boundary_result.substitution_events.detach().cpu().numpy(),
    }
    if (int(boundary_back['code'][0])
            != int(a44.SCOPE_FP64_DISCRETE_BOUNDARY)
            or bool(boundary_back['valid'][0])
            or any(int(boundary_back[name][0]) != 0
                   for name in ('requested', 'append', 'events'))):
        raise AssertionError('device decision boundary did not rollback as code 6')
    _assert_raises(a44.A4ReplicationScopeError, boundary_result.to_numpy)
    if (world.rng.bit_generator.state != rng_before
            or a44.FULL_GPU_WORLD_STEP is not False
            or a44.PORT_STATUS.get('full_gpu_world_step') is not False
            or v3._event_order().count('replication_cpu') != 1):
        raise AssertionError('A4.5a changed RNG or A3 CPU authority')
    return ('forged/state/tail/dt/config/PCG64/scalar-attestation/'
            'foreign-state-code7/device-boundary-code6/completion/resident '
            'mutation rejected; A3 replication_cpu authority retained')


def test_a45b_template_start_formal066_oracle_and_rng_order():
    world, cells, capacity, dt = _a45b_template_start_fixture()
    rng_before = copy.deepcopy(world.rng.bit_generator.state)
    source_before = [v3.pickle_clone(cell.state_dict()) for cell in cells]
    ragged, state, binding = _paid_replication_binding(
        cells, world.config, capacity,
    )
    ragged_before = ragged.state_dict()
    state_before = state.state_dict()
    cache_before = binding.cache.state_dict()
    tape = a44.prepare_substitution_rng_tape(
        binding, dt, world.config, rng_before,
    )
    plan = a44.paid_replication_substitution_numpy(
        binding, dt, world.config, tape,
    )
    expected_start = np.asarray((True, False, True), dtype=bool)
    expected_index = np.asarray((0, -1, 0), dtype=np.int64)
    if (not np.array_equal(plan.template_start_events[:3], expected_start)
            or not np.array_equal(tape.template_start_mask[:3], expected_start)
            or not np.array_equal(
                plan.selected_template_indices[:3], expected_index,
            )
            or not np.array_equal(
                tape.template_selection_indices[:3], expected_index,
            )):
        raise AssertionError('A4.5b template-start schedule differs')
    for ci in (0, 2):
        if (int(plan.template_storage_symbols[ci]) != len(cells[ci].genomes[0])
                or int(plan.append_count[ci]) <= 0
                or int(plan.append_count[ci]) >= len(cells[ci].genomes[0])):
            raise AssertionError('A4.5b start topology is not non-completing')

    cpu_world = copy.deepcopy(world)
    cpu_world.cells = [copy.deepcopy(cell) for cell in cells]
    cpu_before_cells = [copy.deepcopy(cell) for cell in cpu_world.cells]
    for cell in cpu_world.cells:
        cell._replicate_genome(cpu_world, dt, cpu_world.config)
    _assert_paid_replication_cpu_parity(
        plan, cpu_before_cells, cpu_world.cells, 'a45b.cpu_oracle',
    )
    if (float(cpu_world.cells[0].replication_template_lesion)
            != float(cells[0].genome_lesions[0])
            or float(cpu_world.cells[2].replication_template_lesion) != 0.0):
        raise AssertionError('A4.5b selected-lesion/fallback differs')
    if tape.rng_after_state != cpu_world.rng.bit_generator.state:
        raise AssertionError('A4.5b PCG64 after-state differs from Formal066')

    # Independently replay each cell's literal high-level call order.  The
    # index-zero selection is state-neutral on this NumPy/PCG64, including the
    # primed uint32 cache, but remains an explicit event in the tape.
    replay = np.random.Generator(np.random.PCG64())
    replay.bit_generator.state = copy.deepcopy(rng_before)
    for ci in range(3):
        if bool(tape.template_start_mask[ci]):
            selection_before = copy.deepcopy(replay.bit_generator.state)
            selected = int(replay.integers(0, 1))
            if (selected != 0
                    or replay.bit_generator.state != selection_before):
                raise AssertionError('A4.5b index-zero selection replay differs')
        for rank in range(int(tape.draw_count[ci])):
            uniform = float(replay.random())
            if (np.float64(uniform).view(np.uint64)
                    != np.float64(tape.uniform_draws[ci, rank]).view(np.uint64)):
                raise AssertionError('A4.5b threshold interleave differs')
            if bool(tape.replacement_mask[ci, rank]):
                raw = int(replay.integers(0, 7))
                if raw != int(tape.replacement_raw[ci, rank]):
                    raise AssertionError('A4.5b integer interleave differs')
    if replay.bit_generator.state != tape.rng_after_state:
        raise AssertionError('A4.5b explicit replay after-state differs')

    # Repeat one start from an ordinary PCG64 state with no cached uint32, so
    # both canonical cache states are bound by the direct Formal066 oracle.
    ordinary_world, ordinary_cells, ordinary_capacity, ordinary_dt = (
        _a45b_template_start_fixture(seed=8204, prime_uint32=False)
    )
    ordinary_before_rng = copy.deepcopy(
        ordinary_world.rng.bit_generator.state,
    )
    _, _, ordinary_binding = _paid_replication_binding(
        [ordinary_cells[0]], ordinary_world.config, ordinary_capacity,
    )
    ordinary_tape = a44.prepare_substitution_rng_tape(
        ordinary_binding, ordinary_dt, ordinary_world.config,
        ordinary_before_rng,
    )
    ordinary_plan = a44.paid_replication_substitution_numpy(
        ordinary_binding, ordinary_dt, ordinary_world.config, ordinary_tape,
    )
    ordinary_cpu = copy.deepcopy(ordinary_world)
    ordinary_cpu.cells = [copy.deepcopy(ordinary_cells[0])]
    ordinary_cpu_before = [copy.deepcopy(ordinary_cpu.cells[0])]
    ordinary_cpu.cells[0]._replicate_genome(
        ordinary_cpu, ordinary_dt, ordinary_cpu.config,
    )
    _assert_paid_replication_cpu_parity(
        ordinary_plan, ordinary_cpu_before, ordinary_cpu.cells,
        'a45b.ordinary_cache_cpu_oracle',
    )
    if (ordinary_tape.rng_after_state
            != ordinary_cpu.rng.bit_generator.state):
        raise AssertionError('A4.5b ordinary-cache PCG64 after-state differs')
    if world.rng.bit_generator.state != rng_before:
        raise AssertionError('A4.5b changed the live world RNG')
    for ci, (before, cell) in enumerate(zip(source_before, cells)):
        v3.assert_recursive_close(
            before, cell.state_dict(), atol=0.0, rtol=0.0,
            path='a45b.source_cell[%d]' % ci,
        )
    for label, before, after in (
            ('ragged', ragged_before, ragged.state_dict()),
            ('state', state_before, state.state_dict()),
            ('cache', cache_before, binding.cache.state_dict())):
        v3.assert_recursive_close(
            before, after, atol=0.0, rtol=0.0,
            path='a45b.source_%s' % label,
        )
    return ('inactive/active/inactive cell order: index-zero start -> scalar '
            'threshold -> conditional integer; Formal066 state and PCG64 exact')


def test_a45b_template_start_exact_capacity_and_zero_dt():
    world, cells, roomy, _ = _a45b_template_start_fixture(seed=8202)
    cell = copy.deepcopy(cells[0])
    rng_before = copy.deepcopy(world.rng.bit_generator.state)

    # Even dt=0 starts the frozen template and needs two sequence slots plus a
    # byte-exact template arena copy, while drawing no substitution threshold.
    ragged, _, binding = _paid_replication_binding(
        [cell], world.config, roomy,
    )
    zero_tape = a44.prepare_substitution_rng_tape(
        binding, 0.0, world.config, rng_before,
    )
    zero_plan = a44.paid_replication_substitution_numpy(
        binding, 0.0, world.config, zero_tape,
    )
    zero_q = int(ragged.sequence_count) + 2
    zero_s = int(ragged.symbol_count) + len(cell.genomes[0])
    if (not bool(zero_plan.template_start_events[0])
            or int(zero_plan.append_count[0]) != 0
            or int(zero_tape.draw_count[0]) != 0
            or zero_tape.rng_after_state != rng_before):
        raise AssertionError('A4.5b dt=0 start semantics differ')

    def capacity_config(sequences, symbols):
        return a4.GPU068A4Config(
            max_cells=1, max_sequences=int(sequences),
            max_symbols=int(symbols),
            max_sequence_symbols=a4.MAX_FROZEN_GENOME_SYMBOLS,
            max_proteins_per_cell=64,
        )

    exact_zero = capacity_config(zero_q, zero_s)
    exact_zero_ragged, _, exact_zero_binding = _paid_replication_binding(
        [cell], world.config, exact_zero,
    )
    exact_zero_plan = a44.paid_replication_substitution_numpy(
        exact_zero_binding, 0.0, world.config,
        a44.prepare_substitution_rng_tape(
            exact_zero_binding, 0.0, world.config, rng_before,
        ),
    )
    if (int(exact_zero_ragged.sequence_capacity) != zero_q
            or int(exact_zero_ragged.symbol_capacity) != zero_s
            or not bool(exact_zero_plan.template_start_events[0])):
        raise AssertionError('A4.5b dt=0 exact capacity was not accepted')
    short_q = capacity_config(zero_q - 1, zero_s)
    short_q_ragged, short_q_state, short_q_binding = _paid_replication_binding(
        [cell], world.config, short_q,
    )
    _assert_replication_failure_atomic(
        a4.A4CapacityError, short_q_binding,
        lambda: a44.prepare_substitution_rng_tape(
            short_q_binding, 0.0, world.config, rng_before,
        ),
        'a45b.sequence_capacity_short',
    )

    # A nonzero call additionally accounts for every paid copy symbol.
    paid_tape = a44.prepare_substitution_rng_tape(
        binding, 0.5, world.config, rng_before,
    )
    paid_plan = a44.paid_replication_substitution_numpy(
        binding, 0.5, world.config, paid_tape,
    )
    if int(paid_plan.append_count[0]) <= 0:
        raise AssertionError('A4.5b capacity fixture produced no paid append')
    paid_q = int(ragged.sequence_count) + 2
    paid_s = (
        int(ragged.symbol_count) + len(cell.genomes[0])
        + int(paid_plan.append_count[0])
    )
    exact_paid = capacity_config(paid_q, paid_s)
    exact_paid_ragged, exact_paid_state, exact_paid_binding = _paid_replication_binding(
        [cell], world.config, exact_paid,
    )
    exact_paid_tape = a44.prepare_substitution_rng_tape(
        exact_paid_binding, 0.5, world.config, rng_before,
    )
    exact_paid_plan = a44.paid_replication_substitution_numpy(
        exact_paid_binding, 0.5, world.config,
        exact_paid_tape,
    )
    if (int(exact_paid_ragged.symbol_capacity) != paid_s
            or int(exact_paid_plan.append_count[0])
            != int(paid_plan.append_count[0])):
        raise AssertionError('A4.5b paid exact capacity differs')
    short_s = capacity_config(paid_q, paid_s - 1)
    short_s_ragged, short_s_state, short_s_binding = _paid_replication_binding(
        [cell], world.config, short_s,
    )
    _assert_replication_failure_atomic(
        a4.A4CapacityError, short_s_binding,
        lambda: a44.prepare_substitution_rng_tape(
            short_s_binding, 0.5, world.config, rng_before,
        ),
        'a45b.symbol_capacity_short',
    )
    if torch is None:
        raise AssertionError('PyTorch is required for A4.5b capacity checks')
    if _REQUIRE_CUDA and not torch.cuda.is_available():
        raise AssertionError('CUDA required but unavailable; fallback forbidden')
    deterministic, _ = a44._substitution_config(world.config)
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    for device in devices:
        exact_resident = a4.bind_a4_translation(
            exact_paid_ragged.to_torch(device=device),
            exact_paid_state.to_torch(device=device),
        )
        exact_result = a44.paid_replication_substitution_torch(
            exact_resident, 0.5, world.config,
            exact_paid_tape.to_torch(device=device),
        )
        v3.assert_recursive_close(
            exact_paid_plan.state_dict(), exact_result.to_numpy().state_dict(),
            atol=2e-12, rtol=0.0,
            path='a45b.exact_capacity.%s' % device,
        )
        for label, short_ragged, short_state, case_dt in (
                ('sequence', short_q_ragged, short_q_state, 0.0),
                ('symbol', short_s_ragged, short_s_state, 0.5)):
            short_resident = a4.bind_a4_translation(
                short_ragged.to_torch(device=device),
                short_state.to_torch(device=device),
            )
            short_plan = a44._paid_replication_elongation_torch(
                short_resident, case_dt, deterministic,
                allow_template_start=True,
            )
            code = short_plan.scope_error_code.detach().cpu().numpy()
            valid = short_plan.scope_valid.detach().cpu().numpy()
            starts = short_plan.template_start_events.detach().cpu().numpy()
            indices = short_plan.selected_template_indices.detach().cpu().numpy()
            storage = short_plan.template_storage_symbols.detach().cpu().numpy()
            if (int(code[0]) != int(a44.SCOPE_CAPACITY)
                    or bool(valid[0]) or bool(starts[0])
                    or int(indices[0]) != -1 or int(storage[0]) != 0):
                raise AssertionError(
                    '%s %s capacity did not fail closed as code 4' % (
                        device, label,
                    )
                )
            _assert_raises(a4.A4CapacityError, short_plan.to_numpy)
    if world.rng.bit_generator.state != rng_before:
        raise AssertionError('A4.5b capacity checks changed live RNG')
    return ('dt=0 +2 sequences/template bytes exact; paid template+append '
            'symbols exact; each one-short capacity fails atomically')


def test_a45b_template_start_numpy_torch_trust_and_scope():
    if torch is None:
        raise AssertionError('PyTorch is required for A4.5b')
    if _REQUIRE_CUDA and not torch.cuda.is_available():
        raise AssertionError('CUDA required but unavailable; fallback forbidden')
    world, cells, capacity, dt = _a45b_template_start_fixture(seed=8203)
    ragged, state, binding = _paid_replication_binding(
        cells, world.config, capacity,
    )
    rng_before = copy.deepcopy(world.rng.bit_generator.state)
    tape = a44.prepare_substitution_rng_tape(
        binding, dt, world.config, rng_before,
    )
    expected = a44.paid_replication_substitution_numpy(
        binding, dt, world.config, tape,
    )
    deterministic, _ = a44._substitution_config(world.config)
    _assert_raises(
        a44.A4ReplicationScopeError,
        lambda: a44.paid_replication_elongation_numpy(
            binding, dt, deterministic,
        ),
    )
    _assert_raises(
        TypeError,
        lambda: a44.paid_replication_elongation_numpy(
            binding, dt, deterministic, allow_template_start=True,
        ),
    )
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    for device in devices:
        resident_ragged = ragged.to_torch(device=device)
        resident_state = state.to_torch(device=device)
        resident = a4.bind_a4_translation(resident_ragged, resident_state)
        resident_tape = tape.to_torch(device=device)
        pointers = (
            resident_ragged.data_ptrs(), resident_state.data_ptrs(),
            resident.cache.data_ptrs(), resident_tape.data_ptrs(),
        )
        result = a44.paid_replication_substitution_torch(
            resident, dt, world.config, resident_tape,
        )
        public_base = a44.paid_replication_elongation_torch(
            resident, dt, deterministic,
        )
        public_code = public_base.scope_error_code.detach().cpu().numpy()
        public_start = public_base.template_start_events.detach().cpu().numpy()
        if (int(public_code[0]) != int(a44.SCOPE_INACTIVE_TEMPLATE)
                or bool(public_start[0])):
            raise AssertionError(
                '%s public deterministic path minted a template start' % device
            )
        _assert_raises(a44.A4ReplicationScopeError, public_base.to_numpy)
        v3.assert_recursive_close(
            expected.state_dict(), result.to_numpy().state_dict(),
            atol=2e-12, rtol=0.0, path='a45b.device.%s' % device,
        )
        if pointers != (
                resident_ragged.data_ptrs(), resident_state.data_ptrs(),
                resident.cache.data_ptrs(), resident_tape.data_ptrs()):
            raise AssertionError('%s A4.5b resident pointer changed' % device)
        changed = tape.to_torch(device=device)
        changed.template_selection_indices[0] = 1
        _assert_raises(
            a4.A4SchemaError,
            lambda changed=changed: a44.paid_replication_substitution_torch(
                resident, dt, world.config, changed,
            ),
        )

    changed_host = tape.clone()
    changed_host.template_start_mask[0] = False
    changed_host.template_selection_indices[0] = -1
    _assert_raises(
        a4.A4SchemaError,
        lambda: a44.validate_a4_substitution_rng_tape(changed_host),
    )

    # Removing an index-zero selection call remains a fully replayable PCG64
    # tape on the recorded NumPy implementation because that high-level call is
    # state-neutral.  It is nevertheless the wrong biological schedule and
    # must become a batch-wide code 7 on every resident backend.
    alternate_values = tape.state_dict()
    alternate_values['template_start_mask'][0] = False
    alternate_values['template_selection_indices'][0] = -1
    alternate_values['schedule_sha256'] = '0' * 64
    alternate_draft = a44._make_rng_tape(**alternate_values)
    alternate_values['schedule_sha256'] = a44._rng_schedule_digest(
        alternate_draft,
    )
    alternate = a44.validate_a4_substitution_rng_tape(
        a44._make_rng_tape(**alternate_values),
    )
    for device in devices:
        alternate_resident = a4.bind_a4_translation(
            ragged.to_torch(device=device), state.to_torch(device=device),
        )
        alternate_result = a44.paid_replication_substitution_torch(
            alternate_resident, dt, world.config,
            alternate.to_torch(device=device),
        )
        code = alternate_result.scope_error_code.detach().cpu().numpy()
        valid = alternate_result.scope_valid.detach().cpu().numpy()
        requested = alternate_result.requested_symbols.detach().cpu().numpy()
        appended = alternate_result.append_count.detach().cpu().numpy()
        starts = alternate_result.template_start_events.detach().cpu().numpy()
        indices = (
            alternate_result.selected_template_indices.detach().cpu().numpy()
        )
        storage = (
            alternate_result.template_storage_symbols.detach().cpu().numpy()
        )
        if (code[:3].tolist() != [int(a44.SCOPE_RNG_TAPE_MISMATCH)] * 3
                or np.any(valid[:3]) or np.any(requested[:3] != 0)
                or np.any(appended[:3] != 0) or np.any(starts[:3])
                or np.any(indices[:3] != -1) or np.any(storage[:3] != 0)):
            raise AssertionError(
                '%s A4.5b alternate start schedule did not rollback' % device
            )
        _assert_raises(
            a44.A4ReplicationScopeError, alternate_result.to_numpy,
        )

    # Broader ordinary no-op batching and mutation-free starts remain explicit
    # CPU authority until prior telemetry and topology commit are represented.
    scope_cells = []
    zero = copy.deepcopy(cells[0])
    zero.genomes = []
    zero.genome_lesions = []
    zero.replication_template = None
    zero.replication_copy = []
    zero.replication_template_lesion = 0.0
    zero.replication_fractional = 0.0
    zero._refresh_gene_cache()
    scope_cells.append(('zero_genomes', zero, world.config, dt))
    two = copy.deepcopy(cells[0])
    two.replication_template = None
    two.replication_copy = []
    two.replication_template_lesion = 0.0
    two.replication_fractional = 0.0
    two.genomes = [
        np.asarray(two.genomes[0], dtype=np.uint8).copy(),
        np.asarray(two.genomes[0], dtype=np.uint8).copy(),
    ]
    two.genome_lesions = [0.1, 0.2]
    two._refresh_gene_cache()
    scope_cells.append(('two_genomes', two, world.config, dt))
    gated = _external_only_replication_cell(cells[0])
    gated.replication_template = None
    gated.replication_copy = []
    gated.replication_template_lesion = 0.0
    gated.replication_fractional = 0.0
    gated_config = copy.deepcopy(world.config)
    gated_config.external_replicase = False
    scope_cells.append(('replicase_gate', gated, gated_config, dt))
    for label, cell, model_config, case_dt in scope_cells:
        _, _, case_binding = _paid_replication_binding(
            [cell], model_config, capacity,
        )
        _assert_replication_failure_atomic(
            a44.A4ReplicationScopeError, case_binding,
            lambda case_binding=case_binding, model_config=model_config,
                    case_dt=case_dt: a44.prepare_substitution_rng_tape(
                        case_binding, case_dt, model_config, rng_before,
                    ),
            'a45b.scope.%s' % label,
        )
    mutation_off = copy.deepcopy(world.config)
    mutation_off.mutation = False
    _assert_raises(
        a44.A4ReplicationScopeError,
        lambda: a44.prepare_substitution_rng_tape(
            binding, dt, mutation_off, rng_before,
        ),
    )

    completing = copy.deepcopy(cells[0])
    completing.genomes = [np.asarray((3,), dtype=np.uint8)]
    completing.genome_lesions = [0.2]
    completing.replication_template = None
    completing.replication_copy = []
    completing.replication_template_lesion = 0.0
    completing.replication_fractional = 0.0
    completing._refresh_gene_cache()
    completing_config = copy.deepcopy(world.config)
    completing_config.external_replicase = True
    completion_ragged, completion_state, completion_binding = (
        _paid_replication_binding(
        [completing], completing_config, capacity,
        )
    )
    _assert_replication_failure_atomic(
        a44.A4ReplicationScopeError, completion_binding,
        lambda: a44.prepare_substitution_rng_tape(
            completion_binding, 100.0, completing_config, rng_before,
        ),
        'a45b.start_completion',
    )
    completion_deterministic, _ = a44._substitution_config(completing_config)
    for device in devices:
        completion_resident = a4.bind_a4_translation(
            completion_ragged.to_torch(device=device),
            completion_state.to_torch(device=device),
        )
        completion_plan = a44._paid_replication_elongation_torch(
            completion_resident, 100.0, completion_deterministic,
            allow_template_start=True,
        )
        code = completion_plan.scope_error_code.detach().cpu().numpy()
        valid = completion_plan.scope_valid.detach().cpu().numpy()
        starts = completion_plan.template_start_events.detach().cpu().numpy()
        indices = (
            completion_plan.selected_template_indices.detach().cpu().numpy()
        )
        storage = (
            completion_plan.template_storage_symbols.detach().cpu().numpy()
        )
        if (int(code[0]) != int(a44.SCOPE_COMPLETION)
                or bool(valid[0]) or bool(starts[0])
                or int(indices[0]) != -1 or int(storage[0]) != 0):
            raise AssertionError(
                '%s A4.5b start completion did not remain code 3' % device
            )
        _assert_raises(
            a44.A4ReplicationScopeError, completion_plan.to_numpy,
        )
    if (world.rng.bit_generator.state != rng_before
            or a44.FULL_GPU_WORLD_STEP is not False
            or a44.PORT_STATUS.get('full_gpu_world_step') is not False
            or v3._event_order().count('replication_cpu') != 1):
        raise AssertionError('A4.5b changed RNG or A3 CPU authority')
    return ('NumPy/Torch %s topology+tape exact; start trust tampering, '
            'len0/len2/gate/mutation-off/completion rejected; A3 authority' %
            '/'.join(devices))


def test_a46a_completion_formal066_oracle_and_conceptual_poststate():
    world, cells, capacity, dt = _paid_completion_fixture()
    world_before = v3.pickle_clone(world.state_dict())
    rng_before = v3.pickle_clone(world.rng.bit_generator.state)
    source_before = [v3.pickle_clone(cell.state_dict()) for cell in cells]
    ragged, state, binding = _paid_replication_binding(
        cells, world.config, capacity,
    )
    ragged_before = ragged.state_dict()
    state_before = state.state_dict()
    cache_before = binding.cache.state_dict()
    plan = a44.paid_replication_completion_plan(
        binding, dt, world.config,
    )

    cpu_cells = copy.deepcopy(cells)
    cpu_before = copy.deepcopy(cpu_cells)
    for cell in cpu_cells:
        cell._replicate_genome(world, dt, world.config)
    _assert_completion_cpu_parity(
        plan, cpu_before, cpu_cells, 'a46a.formal066', atol=0.0,
    )

    for ci, before in enumerate(cells):
        proof = _cpu_raw_repair_activity(
            before, a4.a3.REPAIR_PROOFREADING,
        )
        proof_fraction = proof / (0.75 + proof)
        length = int(plan.completed_lengths[ci])
        expected_lesion = float(before.replication_template_lesion) * (
            0.28 + 0.22 * (1.0 - proof_fraction)
        ) + float(plan.last_effective_error_rate[ci]) * length * 0.06
        v3.assert_recursive_close(
            expected_lesion, float(plan.new_genome_lesions[ci]),
            atol=0.0, rtol=0.0,
            path='a46a.lesion_grouping[%d]' % ci,
        )

    # Treat the direct CPU result only as a conceptual post-state.  The A4.6a
    # plan itself must not mutate/re-attest the authoritative resident arena.
    post_ragged = a4.FullFidelityA4GenomeAdapter(capacity).pack_cells(cpu_cells)
    if (int(post_ragged.sequence_count)
            != int(ragged.sequence_count)
            + int(np.sum(plan.topology_sequence_deltas[:2], dtype=np.int64))
            or int(post_ragged.symbol_count)
            != int(ragged.symbol_count)
            + int(np.sum(plan.topology_symbol_deltas[:2], dtype=np.int64))
            or int(post_ragged.lesion_count)
            != int(ragged.lesion_count) + 2):
        raise AssertionError('A4.6a conceptual topology delta differs')
    post_cache = a4.decode_a4_gene_cache_numpy(post_ragged)
    materialized = post_cache.materialize_gene_specs_host()
    for ci, cell in enumerate(cpu_cells):
        _assert_gene_specs_equal(
            cell.gene_specs, materialized[ci],
            'a46a.conceptual_cache[%d]' % ci,
        )

    for ci, (before, cell) in enumerate(zip(source_before, cells)):
        v3.assert_recursive_close(
            before, cell.state_dict(), atol=0.0, rtol=0.0,
            path='a46a.source_cell[%d]' % ci,
        )
    for label, before, after in (
            ('world', world_before, world.state_dict()),
            ('rng', rng_before, world.rng.bit_generator.state),
            ('ragged', ragged_before, ragged.state_dict()),
            ('state', state_before, state.state_dict()),
            ('cache', cache_before, binding.cache.state_dict())):
        v3.assert_recursive_close(
            before, after, atol=0.0, rtol=0.0,
            path='a46a.nonmutation.%s' % label,
        )
    return ('2 Formal066 completions: paid partial+suffix, lesion/cycle/reset, '
            'conceptual topology/cache exact; source and RNG unchanged')


def test_a46a_completion_numpy_torch_devices_and_nonmutation():
    if torch is None:
        raise AssertionError('PyTorch is required for A4.6a')
    if _REQUIRE_CUDA and not torch.cuda.is_available():
        raise AssertionError('CUDA required but unavailable; CPU fallback forbidden')
    world, cells, capacity, dt = _paid_completion_fixture(seed=8302)
    world_before = v3.pickle_clone(world.state_dict())
    rng_before = v3.pickle_clone(world.rng.bit_generator.state)
    ragged, state, binding = _paid_replication_binding(
        cells, world.config, capacity,
    )
    expected = a44.paid_replication_completion_plan(
        binding, dt, world.config,
    )
    source = '\n'.join(inspect.getsource(item) for item in (
        a44.paid_replication_completion_plan,
        a44._paid_replication_completion_torch,
        a44._paid_replication_elongation_torch,
        a44._torch_replicase,
        a44._validate_plan_metadata,
    ))
    forbidden = ('.item(', '.cpu(', '.numpy(', '.tolist(',
                 'nonzero(', 'masked_select(', 'unique(')
    found = [token for token in forbidden if token in source]
    if found:
        raise AssertionError('resident A4.6a contains host/dynamic op: %s' % found)

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    for device in devices:
        resident_ragged = ragged.to_torch(device=device)
        resident_state = state.to_torch(device=device)
        ragged_ptrs = resident_ragged.data_ptrs()
        state_ptrs = resident_state.data_ptrs()
        resident_binding = a4.bind_a4_translation(
            resident_ragged, resident_state,
        )
        cache_ptrs = resident_binding.cache.data_ptrs()
        plan = a44.paid_replication_completion_plan(
            resident_binding, dt, world.config,
        )
        plan_ptrs = plan.data_ptrs()
        if any(getattr(plan, name).device.type != device
               for name in a44._PLAN_ARRAY_FIELDS):
            raise AssertionError('%s A4.6a output escaped device' % device)
        if device == 'cuda':
            torch.cuda.synchronize()
        back = plan.to_numpy()
        v3.assert_recursive_close(
            expected.state_dict(), back.state_dict(),
            atol=2e-12, rtol=0.0, path='a46a.plan.%s' % device,
        )
        if (ragged_ptrs != resident_ragged.data_ptrs()
                or state_ptrs != resident_state.data_ptrs()
                or cache_ptrs != resident_binding.cache.data_ptrs()
                or plan_ptrs != plan.data_ptrs()):
            raise AssertionError('%s A4.6a resident pointer changed' % device)
        v3.assert_recursive_close(
            ragged.state_dict(), resident_ragged.to_numpy().state_dict(),
            atol=0.0, rtol=0.0,
            path='a46a.ragged_source.%s' % device,
        )
        v3.assert_recursive_close(
            state.state_dict(), resident_state.to_numpy().state_dict(),
            atol=0.0, rtol=0.0,
            path='a46a.state_source.%s' % device,
        )
    v3.assert_recursive_close(
        world_before, world.state_dict(), atol=0.0, rtol=0.0,
        path='a46a.device_world',
    )
    v3.assert_recursive_close(
        rng_before, world.rng.bit_generator.state, atol=0.0, rtol=0.0,
        path='a46a.device_rng',
    )
    return ('NumPy/Torch %s fp64 completion descriptor exact; fixed device '
            'outputs, pointers, inputs, world, and RNG unchanged' %
            '/'.join(devices))


def test_a46a_completion_scope_schema_and_a3_authority():
    world, cells, capacity, dt = _paid_completion_fixture(seed=8303)
    ragged, state, binding = _paid_replication_binding(
        cells, world.config, capacity,
    )
    valid = a44.paid_replication_completion_plan(
        binding, dt, world.config,
    )

    # Legacy public paths must continue to route completion to CPU authority.
    _assert_raises(
        a44.A4ReplicationScopeError,
        lambda: a44.paid_replication_elongation_numpy(
            binding, dt, world.config,
        ),
    )
    device = 'cuda' if torch is not None and torch.cuda.is_available() else 'cpu'
    resident = a4.bind_a4_translation(
        ragged.to_torch(device=device), state.to_torch(device=device),
    )
    legacy = a44.paid_replication_elongation_torch(
        resident, dt, world.config,
    )
    legacy_codes = legacy.scope_error_code.detach().cpu().numpy()[:2]
    if not np.all(legacy_codes == int(a44.SCOPE_COMPLETION)):
        raise AssertionError('legacy completion did not remain code 3')
    _assert_raises(a44.A4ReplicationScopeError, legacy.to_numpy)

    # A mixed completion/noncompletion batch is one atomic descriptor event.
    _, ordinary, _, ordinary_dt = _paid_replication_b_fixture(seed=8304)
    mixed_cells = [copy.deepcopy(cells[0]), copy.deepcopy(ordinary[1])]
    mixed_ragged, mixed_state, mixed_binding = _paid_replication_binding(
        mixed_cells, world.config, capacity,
    )
    _assert_raises(
        a44.A4ReplicationScopeError,
        lambda: a44.paid_replication_completion_plan(
            mixed_binding, ordinary_dt, world.config,
        ),
    )
    mixed_resident = a4.bind_a4_translation(
        mixed_ragged.to_torch(device=device),
        mixed_state.to_torch(device=device),
    )
    mixed_plan = a44.paid_replication_completion_plan(
        mixed_resident, ordinary_dt, world.config,
    )
    mixed_code = mixed_plan.scope_error_code.detach().cpu().numpy()[:2]
    mixed_valid = mixed_plan.scope_valid.detach().cpu().numpy()[:2]
    if (np.any(mixed_valid)
            or mixed_code.tolist() != [
                int(a44.SCOPE_NONCOMPLETION),
                int(a44.SCOPE_NONCOMPLETION),
            ]
            or bool(torch.any(mixed_plan.completion_events).detach().cpu())
            or int(torch.sum(mixed_plan.append_count).detach().cpu()) != 0
            or int(torch.sum(mixed_plan.completed_lengths).detach().cpu()) != 0):
        raise AssertionError('mixed completion batch did not roll back atomically')
    _assert_raises(a44.A4ReplicationScopeError, mixed_plan.to_numpy)

    mutation = copy.deepcopy(world.config)
    mutation.mutation = True
    _assert_raises(
        a44.A4ReplicationScopeError,
        lambda: a44.paid_replication_completion_plan(binding, dt, mutation),
    )
    inactive = copy.deepcopy(cells[0])
    inactive.replication_template = None
    inactive.replication_copy = []
    inactive.replication_template_lesion = 0.0
    inactive.replication_fractional = 0.0
    _, _, inactive_binding = _paid_replication_binding(
        [inactive], world.config, capacity,
    )
    _assert_raises(
        a44.A4ReplicationScopeError,
        lambda: a44.paid_replication_completion_plan(
            inactive_binding, dt, world.config,
        ),
    )

    corruptions = []
    def add(name, mutate):
        item = valid.clone()
        mutate(item)
        corruptions.append((name, item))

    add('completion_event', lambda x: x.completion_events.__setitem__(0, False))
    add('completed_length', lambda x: x.completed_lengths.__setitem__(0, 0))
    add('completed_tail', lambda x: x.completed_symbols.__setitem__(
        (0, int(x.completed_lengths[0])), 1,
    ))
    add('completed_suffix', lambda x: x.completed_symbols.__setitem__(
        (0, int(x.completed_lengths[0]) - 1),
        (int(x.completed_symbols[0, int(x.completed_lengths[0]) - 1]) + 1)
        % a4.ALPHABET_SIZE,
    ))
    add('new_lesion_negative', lambda x: x.new_genome_lesions.__setitem__(0, -1.0))
    add('new_lesion_nonfinite', lambda x: x.new_genome_lesions.__setitem__(0, np.nan))
    add('cycle_delta', lambda x: x.replication_cycle_deltas.__setitem__(0, 0))
    add('sequence_delta', lambda x: x.topology_sequence_deltas.__setitem__(0, 0))
    add('symbol_delta', lambda x: x.topology_symbol_deltas.__setitem__(
        0, int(x.topology_symbol_deltas[0]) + 1,
    ))
    add('fractional_reset', lambda x: x.replication_fractional_after.__setitem__(0, 0.5))
    for name, forged in corruptions:
        _assert_raises(
            a4.A4SchemaError,
            lambda forged=forged: a44.validate_a4_paid_elongation_plan(forged),
        )

    # A short empty-copy completion replaces template storage one-for-one.
    # Exact current symbol capacity must pass; no append may be double-counted.
    exact_cell = copy.deepcopy(cells[0])
    short = np.asarray(exact_cell.genomes[0][:64], dtype=np.uint8).copy()
    exact_cell.genomes = [short.copy()]
    exact_cell.genome_lesions = [0.125]
    exact_cell.replication_template = short.copy()
    exact_cell.replication_copy = []
    exact_cell.replication_template_lesion = 0.125
    exact_cell.replication_fractional = 0.0
    exact_cell.pools[a4.a3.POOL_NUCLEOTIDE] = 1.0
    exact_cell.pools[a4.a3.POOL_ATP] = 1.0
    exact_cell._refresh_gene_cache()
    exact_cell._sync_protein_pool()
    exact_model = _replication_model_config(
        world.config, external_replicase=True,
    )
    roomy = a4.GPU068A4Config(
        max_cells=1, max_sequences=3, max_symbols=256,
        max_sequence_symbols=64, max_proteins_per_cell=64,
    )
    probe = a4.FullFidelityA4GenomeAdapter(roomy).pack_cells([exact_cell])
    exact_capacity = a4.GPU068A4Config(
        max_cells=1, max_sequences=int(probe.sequence_count),
        max_symbols=int(probe.symbol_count), max_sequence_symbols=64,
        max_proteins_per_cell=64,
    )
    exact_ragged, _, exact_binding = _paid_replication_binding(
        [exact_cell], exact_model, exact_capacity,
    )
    exact_plan = a44.paid_replication_completion_plan(
        exact_binding, 20.0, exact_model,
    )
    if (int(exact_plan.completed_lengths[0]) != 64
            or int(exact_plan.topology_symbol_deltas[0]) != 0
            or int(exact_ragged.symbol_capacity)
            != int(exact_ragged.symbol_count)):
        raise AssertionError('exact current completion capacity did not pass')

    if (a44.FULL_GPU_WORLD_STEP is not False
            or a44.PORT_STATUS.get('full_gpu_world_step') is not False
            or 'not-integrated' not in a44.PORT_STATUS.get(
                'genome_replication', '')
            or v3._event_order().count('replication_cpu') != 1):
        raise AssertionError('A4.6a changed A3 replication authority')
    return ('legacy code3; mixed/noncompletion/mutation/start rollback; '
            '%d structural/schema corruptions rejected; exact current capacity; '
            'A3 replication_cpu retained' % len(corruptions))


def test_a46b1_completion_mutation_rng_tape_formal066_oracle():
    world, cells, capacity, dt = _a46b1_completion_mutation_fixture()
    world_before = v3.pickle_clone(world.state_dict())
    rng_before = copy.deepcopy(world.rng.bit_generator.state)
    source_before = [v3.pickle_clone(cell.state_dict()) for cell in cells]
    ragged, state, binding = _paid_replication_binding(
        cells, world.config, capacity,
    )
    ragged_before = ragged.state_dict()
    state_before = state.state_dict()
    cache_before = binding.cache.state_dict()
    tape = a44.prepare_completion_mutation_rng_tape(
        binding, dt, world.config, rng_before,
    )
    if a44.validate_a4_completion_mutation_rng_tape(
            tape, binding, dt, world.config) is not tape:
        raise AssertionError('A4.6b1 validator did not return its tape')

    expected_events = np.asarray([
        [4, 2, 16, 1, 1],
        [0, 2, 0, 1, 1],
        [3, 0, 0, 1, 1],
    ], dtype=np.int64)
    for label, actual, expected in (
            ('append_count', tape.append_count[:3], [1, 1, 1]),
            ('substitution_count', tape.substitution_count[:3], [0, 0, 1]),
            ('pre_length', tape.pre_structural_lengths[:3], [576, 640, 8]),
            ('post_length', tape.post_structural_lengths[:3], [577, 638, 32]),
            ('material_delta', tape.material_delta_symbols[:3], [1, -2, 24]),
            ('insertion_count', tape.insertion_count[:3], [4, 0, 3]),
            ('deletion_count', tape.deletion_count[:3], [2, 2, 0]),
            ('padding_count', tape.padding_count[:3], [0, 0, 21])):
        if not np.array_equal(
                np.asarray(actual), np.asarray(expected, dtype=np.int64)):
            raise AssertionError('A4.6b1 %s fixture drifted' % label)
    mass = float(a4.g2.MONOMER_MASS)
    expected_budget = np.asarray([
        min(
            int((
                float(cell.pools[a4.a3.POOL_NUCLEOTIDE])
                - int(tape.append_count[ci]) * mass
            ) / mass),
            int(a4.g2.MAX_GENOME_LENGTH),
        )
        for ci, cell in enumerate(cells)
    ], dtype=np.int64)
    if not np.array_equal(
            tape.nucleotide_budget_symbols[:3], expected_budget):
        raise AssertionError('A4.6b1 effective material budget differs')
    if not np.array_equal(
            tape.structural_event_counts[:3], expected_events):
        raise AssertionError('A4.6b1 structural event fixture drifted')
    if (not np.all(tape.append_draw_mask[:3, 0])
            or not np.all(tape.threshold_draw_mask[:3])
            or not np.all(tape.threshold_hit_mask[:3])
            or int(tape.duplication_gene_ordinal[0]) < 0
            or np.any(tape.duplication_gene_ordinal[1:3] != -1)):
        raise AssertionError('A4.6b1 conditional draw schedule drifted')
    if (int(tape.cell_count) != 3
            or bool(tape.cell_mask[3])
            or np.any(tape.structural_event_counts[3])
            or np.any(tape.append_draw_mask[3])
            or np.any(tape.threshold_draw_mask[3])
            or np.any(tape.padding_symbols[3])):
        raise AssertionError('A4.6b1 unused tape tail is not canonical')
    if (int(tape.rng_before_state['has_uint32']) != 1
            or int(tape.rng_before_state['uinteger']) != 3034332626):
        raise AssertionError('A4.6b1 primed PCG64 before-state drifted')

    cpu_world = copy.deepcopy(world)
    cpu_before = copy.deepcopy(cells)
    cpu_cells = copy.deepcopy(cells)
    for cell in cpu_cells:
        cell._replicate_genome(cpu_world, dt, cpu_world.config)
    if tape.rng_after_state != cpu_world.rng.bit_generator.state:
        raise AssertionError('A4.6b1 PCG64 after-state differs from Formal066')
    if (int(tape.rng_after_state['has_uint32']) != 1
            or int(tape.rng_after_state['uinteger']) != 2698505109):
        raise AssertionError('A4.6b1 PCG64 byte-vector cache drifted')

    for ci, (before, actual) in enumerate(zip(cpu_before, cpu_cells)):
        replayed = _a46b1_replay_tape_row(tape, before, ci)
        if (len(actual.genomes) != len(before.genomes) + 1
                or not np.array_equal(replayed, actual.genomes[-1])):
            raise AssertionError(
                'A4.6b1 cell[%d] final polymer differs' % ci
            )
        structural = np.asarray([
            int(actual.mutation_events[name])
            - int(before.mutation_events[name])
            for name in a44.STRUCTURAL_EVENT_NAMES
        ], dtype=np.int64)
        substitution = (
            int(actual.mutation_events['substitution'])
            - int(before.mutation_events['substitution'])
        )
        if (not np.array_equal(
                structural, tape.structural_event_counts[ci])
                or substitution != int(tape.substitution_count[ci])):
            raise AssertionError(
                'A4.6b1 cell[%d] mutation counters differ' % ci
            )
        expected_nucleotide = (
            float(before.pools[a4.a3.POOL_NUCLEOTIDE])
            - int(tape.append_count[ci]) * mass
            - int(tape.material_delta_symbols[ci]) * mass
        )
        v3.assert_recursive_close(
            expected_nucleotide,
            float(actual.pools[a4.a3.POOL_NUCLEOTIDE]),
            atol=2e-12, rtol=0.0,
            path='a46b1.material[%d]' % ci,
        )

    if world.rng.bit_generator.state != rng_before:
        raise AssertionError('A4.6b1 preparation advanced the live RNG')
    for ci, (before, cell) in enumerate(zip(source_before, cells)):
        v3.assert_recursive_close(
            before, cell.state_dict(), atol=0.0, rtol=0.0,
            path='a46b1.source_cell[%d]' % ci,
        )
    for label, before, after in (
            ('world', world_before, world.state_dict()),
            ('ragged', ragged_before, ragged.state_dict()),
            ('state', state_before, state.state_dict()),
            ('cache', cache_before, binding.cache.state_dict())):
        v3.assert_recursive_close(
            before, after, atol=0.0, rtol=0.0,
            path='a46b1.nonmutation.%s' % label,
        )
    return ('3 cell-interleaved Formal066 completions: substitution hit/miss, '
            'five structural operations, padding, material trim/refund, and '
            'full PCG64 uint32 cache exact')


def test_a46b1_completion_mutation_rng_tape_torch_roundtrip():
    if torch is None:
        raise AssertionError('PyTorch is required for A4.6b1')
    if _REQUIRE_CUDA and not torch.cuda.is_available():
        raise AssertionError('CUDA required but unavailable; CPU fallback forbidden')
    world, cells, capacity, dt = _a46b1_completion_mutation_fixture(
        seed=9912,
    )
    world_before = v3.pickle_clone(world.state_dict())
    rng_before = copy.deepcopy(world.rng.bit_generator.state)
    ragged, state, binding = _paid_replication_binding(
        cells, world.config, capacity,
    )
    ragged_before = ragged.state_dict()
    state_before = state.state_dict()
    cache_before = binding.cache.state_dict()
    tape = a44.prepare_completion_mutation_rng_tape(
        binding, dt, world.config, rng_before,
    )
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    for device in devices:
        resident = tape.to_torch(
            binding, dt, world.config, device=device,
        )
        if any(
                getattr(resident, name).device.type != device
                for name in a44._COMPLETION_TAPE_ARRAY_FIELDS):
            raise AssertionError('%s A4.6b1 tape escaped device' % device)
        pointers = resident.data_ptrs()
        if device == 'cuda':
            torch.cuda.synchronize()
        back = resident.to_numpy()
        a44.validate_a4_completion_mutation_rng_tape(
            back, binding, dt, world.config,
        )
        v3.assert_recursive_close(
            tape.state_dict(), back.state_dict(), atol=0.0, rtol=0.0,
            path='a46b1.tape_roundtrip.%s' % device,
        )
        if pointers != resident.data_ptrs():
            raise AssertionError('%s A4.6b1 tape pointer changed' % device)

        changed = tape.to_torch(
            binding, dt, world.config, device=device,
        )
        changed.append_uniform_draws[0, 0] += 0.125
        _assert_raises(a4.A4SchemaError, changed.to_numpy)
        stealth = tape.to_torch(
            binding, dt, world.config, device=device,
        )
        version_before = int(stealth.append_uniform_draws._version)
        stealth.append_uniform_draws.data[0, 0] = (
            stealth.append_uniform_draws.data[0, 0] + 0.25
        )
        if int(stealth.append_uniform_draws._version) != version_before:
            raise AssertionError(
                '%s stealth fixture unexpectedly changed Tensor version' %
                device
            )
        stealth_clone = stealth.clone()
        _assert_raises(a4.A4SchemaError, stealth.to_numpy)
        _assert_raises(a4.A4SchemaError, stealth_clone.to_numpy)
        changed_scalar = tape.to_torch(
            binding, dt, world.config, device=device,
        )
        changed_scalar.dt_hex = float(dt + 1e-12).hex()
        _assert_raises(a4.A4SchemaError, changed_scalar.to_numpy)

    if (world.rng.bit_generator.state != rng_before
            or a44.FULL_GPU_WORLD_STEP is not False):
        raise AssertionError('A4.6b1 upload changed RNG or authority')
    for label, before, after in (
            ('world', world_before, world.state_dict()),
            ('ragged', ragged_before, ragged.state_dict()),
            ('state', state_before, state.state_dict()),
            ('cache', cache_before, binding.cache.state_dict())):
        v3.assert_recursive_close(
            before, after, atol=0.0, rtol=0.0,
            path='a46b1.roundtrip_nonmutation.%s' % label,
        )
    return ('NumPy/Torch %s fixed-shape tape roundtrip exact; resident '
            'scalar/tensor/content/pointer attestation and source/RNG '
            'nonmutation' %
            '/'.join(devices))


def test_a46b1_completion_mutation_rng_tape_fail_closed_and_authority():
    world, cells, capacity, dt = _a46b1_completion_mutation_fixture(
        seed=9913,
    )
    rng_before = copy.deepcopy(world.rng.bit_generator.state)
    _, _, binding = _paid_replication_binding(
        cells, world.config, capacity,
    )
    tape = a44.prepare_completion_mutation_rng_tape(
        binding, dt, world.config, rng_before,
    )

    corruptions = []
    def add(name, mutate):
        values = tape.state_dict()
        mutate(values)
        corruptions.append((
            name, a44._make_completion_mutation_rng_tape(**values),
        ))

    add('append_uniform', lambda x: x['append_uniform_draws'].__setitem__(
        (0, 0), np.nextafter(x['append_uniform_draws'][0, 0], 1.0),
    ))
    add('replacement_raw', lambda x: x['replacement_raw'].__setitem__(
        (2, 0), (int(x['replacement_raw'][2, 0]) + 1) % 7,
    ))
    add('insertion_payload', lambda x: x['insertion_symbols'].__setitem__(
        (0, 0), (int(x['insertion_symbols'][0, 0]) + 1) % 8,
    ))
    add('deletion_position', lambda x: x['deletion_position'].__setitem__(
        0, int(x['deletion_position'][0]) + 1,
    ))
    add('duplication_start', lambda x: x['duplication_source_start'].__setitem__(
        0, int(x['duplication_source_start'][0]) + 1,
    ))
    add('inversion_right', lambda x: x['inversion_right'].__setitem__(
        0, int(x['inversion_right'][0]) - 1,
    ))
    add('transposition_position', lambda x: x['transposition_position'].__setitem__(
        0, int(x['transposition_position'][0]) + 1,
    ))
    add('padding_payload', lambda x: x['padding_symbols'].__setitem__(
        (2, 0), (int(x['padding_symbols'][2, 0]) + 1) % 8,
    ))
    add('material_delta', lambda x: x['material_delta_symbols'].__setitem__(
        0, int(x['material_delta_symbols'][0]) + 1,
    ))
    def change_after(values):
        nested = values['rng_after_state']['state']
        nested['state'] = (int(nested['state']) + 1) % (1 << 128)
    add('rng_after', change_after)
    add('schedule_digest', lambda x: x.__setitem__('schedule_sha256', 'f' * 64))
    for name, forged in corruptions:
        _assert_raises(
            a4.A4SchemaError,
            lambda forged=forged: a44.validate_a4_completion_mutation_rng_tape(
                forged, binding, dt, world.config,
            ),
        )

    changed_host = tape.clone()
    changed_host.padding_symbols[2, 0] ^= np.uint8(1)
    _assert_raises(
        a4.A4SchemaError,
        lambda: a44.validate_a4_completion_mutation_rng_tape(
            changed_host, binding, dt, world.config,
        ),
    )
    mutation_off = copy.deepcopy(world.config)
    mutation_off.mutation = False
    _assert_raises(
        a44.A4ReplicationScopeError,
        lambda: a44.prepare_completion_mutation_rng_tape(
            binding, dt, mutation_off, rng_before,
        ),
    )
    nonfinite = copy.deepcopy(world.config)
    nonfinite.structural_rate = float('nan')
    _assert_raises(
        a44.A4ReplicationScopeError,
        lambda: a44.prepare_completion_mutation_rng_tape(
            binding, dt, nonfinite, rng_before,
        ),
    )
    changed_config = copy.deepcopy(world.config)
    changed_config.structural_rate = 9.0
    _assert_raises(
        a44.A4ReplicationScopeError,
        lambda: a44.validate_a4_completion_mutation_rng_tape(
            tape, binding, dt, changed_config,
        ),
    )
    _assert_raises(
        a44.A4ReplicationScopeError,
        lambda: a44.validate_a4_completion_mutation_rng_tape(
            tape, binding, dt + 1e-12, world.config,
        ),
    )

    foreign_cells = copy.deepcopy(cells)
    foreign_cells[0].pools[a4.a3.POOL_NUCLEOTIDE] += 0.01
    _, _, foreign_binding = _paid_replication_binding(
        foreign_cells, world.config, capacity,
    )
    _assert_raises(
        a4.A4Error,
        lambda: a44.validate_a4_completion_mutation_rng_tape(
            tape, foreign_binding, dt, world.config,
        ),
    )

    # A large but CPU-convertible budget is outcome-equivalent to F and is
    # bounded in the fixed tape.  A finite pool whose raw quotient overflows to
    # infinity remains outside scope, matching the frozen CPU conversion
    # failure instead of inventing an A4-only successful state.
    bounded = copy.deepcopy(cells[0])
    bounded.pools[a4.a3.POOL_NUCLEOTIDE] = 0.75
    huge_capacity = a4.GPU068A4Config(
        max_cells=1, max_sequences=4, max_symbols=4096,
        max_sequence_symbols=a4.MAX_FROZEN_GENOME_SYMBOLS,
        max_proteins_per_cell=64,
    )
    _, _, bounded_binding = _paid_replication_binding(
        [bounded], world.config, huge_capacity,
    )
    bounded_tape = a44.prepare_completion_mutation_rng_tape(
        bounded_binding, dt, world.config, rng_before,
    )
    if int(bounded_tape.nucleotide_budget_symbols[0]) != int(
            a4.g2.MAX_GENOME_LENGTH):
        raise AssertionError('A4.6b1 large finite budget was not bounded to F')
    huge = copy.deepcopy(cells[0])
    huge.pools[a4.a3.POOL_NUCLEOTIDE] = 1e308
    _, _, huge_binding = _paid_replication_binding(
        [huge], world.config, huge_capacity,
    )
    _assert_raises(
        a44.A4ReplicationScopeError,
        lambda: a44.prepare_completion_mutation_rng_tape(
            huge_binding, dt, world.config, rng_before,
        ),
    )
    inactive = copy.deepcopy(cells[0])
    inactive.replication_template = None
    inactive.replication_copy = []
    inactive.replication_template_lesion = 0.0
    inactive.replication_fractional = 0.0
    _, _, inactive_binding = _paid_replication_binding(
        [inactive], world.config, capacity,
    )
    _assert_raises(
        a44.A4ReplicationScopeError,
        lambda: a44.prepare_completion_mutation_rng_tape(
            inactive_binding, dt, world.config, rng_before,
        ),
    )
    _, ordinary_cells, _, ordinary_dt = _paid_replication_b_fixture(
        seed=9914,
    )
    mixed = [copy.deepcopy(cells[0]), copy.deepcopy(ordinary_cells[1])]
    _, _, mixed_binding = _paid_replication_binding(
        mixed, world.config, capacity,
    )
    _assert_raises(
        a44.A4ReplicationScopeError,
        lambda: a44.prepare_completion_mutation_rng_tape(
            mixed_binding, ordinary_dt, world.config, rng_before,
        ),
    )

    # The tape carries no final genome, but preparation already knows the
    # frozen post-mutation length.  Per-sequence and aggregate future capacity
    # must fail closed; material-budget trimming is never reused as an arena
    # overflow clip.
    short = copy.deepcopy(cells[2])
    common = {
        'max_cells': 1, 'max_sequences': 3,
        'max_proteins_per_cell': 64,
    }
    current_capacity = a4.GPU068A4Config(
        max_symbols=23, max_sequence_symbols=32, **common
    )
    _, _, current_binding = _paid_replication_binding(
        [short], world.config, current_capacity,
    )
    _assert_replication_failure_atomic(
        a4.A4CapacityError, current_binding,
        lambda: a44.prepare_completion_mutation_rng_tape(
            current_binding, dt, world.config, rng_before,
        ),
        'a46b1.current_symbol_capacity',
    )
    exact_capacity = a4.GPU068A4Config(
        max_symbols=40, max_sequence_symbols=32, **common
    )
    exact_ragged, _, exact_binding = _paid_replication_binding(
        [short], world.config, exact_capacity,
    )
    exact = a44.prepare_completion_mutation_rng_tape(
        exact_binding, dt, world.config, rng_before,
    )
    if (int(exact_ragged.symbol_count) != 23
            or int(exact_ragged.symbol_capacity) != 40
            or int(exact.append_capacity) != 32
            or int(exact.post_structural_lengths[0]) != 32
            or int(exact.material_delta_symbols[0]) != 24):
        raise AssertionError('A4.6b1 exact future capacity differs')
    one_short_capacity = a4.GPU068A4Config(
        max_symbols=39, max_sequence_symbols=32, **common
    )
    _, _, one_short_binding = _paid_replication_binding(
        [short], world.config, one_short_capacity,
    )
    _assert_replication_failure_atomic(
        a4.A4CapacityError, one_short_binding,
        lambda: a44.prepare_completion_mutation_rng_tape(
            one_short_binding, dt, world.config, rng_before,
        ),
        'a46b1.one_short_symbol_capacity',
    )
    narrow_capacity = a4.GPU068A4Config(
        max_cells=1, max_sequences=3, max_symbols=64,
        max_sequence_symbols=31, max_proteins_per_cell=64,
    )
    _, _, narrow_binding = _paid_replication_binding(
        [short], world.config, narrow_capacity,
    )
    _assert_replication_failure_atomic(
        a4.A4CapacityError, narrow_binding,
        lambda: a44.prepare_completion_mutation_rng_tape(
            narrow_binding, dt, world.config, rng_before,
        ),
        'a46b1.sequence_capacity',
    )

    expected_replication = (
        'a4.6b2-pre-existing-active-all-row-completion-combined-pcg64-'
        'substitution-structural-material-fixed-resident-plan-'
        'not-arena-committed-not-integrated-cpu-authoritative'
    )
    expected_material = (
        'a4.6b2-binding-aware-attested-tape-applied-to-pure-resident-'
        'descriptor-not-live-rng-authority-cpu-authoritative'
    )
    if (a44.PORT_STATUS.get('genome_replication') != expected_replication
            or a44.PORT_STATUS.get('material_mutation') != expected_material
            or a44.FULL_GPU_WORLD_STEP is not False
            or v3._event_order().count('replication_cpu') != 1
            or world.rng.bit_generator.state != rng_before):
        raise AssertionError('A4.6b1 CPU/RNG/scheduler authority changed')
    return ('%d replay/attestation corruptions plus config/dt/source/inactive/'
            'mixed scope rejected; current/exact/one-short aggregate and '
            'per-sequence capacities fail closed; apply remains A4.6b2' %
            len(corruptions))


def test_a46b2_completion_mutation_apply_formal066_oracle():
    world, cells, capacity, dt = _a46b1_completion_mutation_fixture()
    world_before = v3.pickle_clone(world.state_dict())
    rng_before = copy.deepcopy(world.rng.bit_generator.state)
    source_before = [v3.pickle_clone(cell.state_dict()) for cell in cells]
    ragged, state, binding = _paid_replication_binding(
        cells, world.config, capacity,
    )
    ragged_before = ragged.state_dict()
    state_before = state.state_dict()
    cache_before = binding.cache.state_dict()
    tape = a44.prepare_completion_mutation_rng_tape(
        binding, dt, world.config, rng_before,
    )
    plan = a44.paid_replication_completion_mutation_numpy(
        binding, dt, world.config, tape,
    )

    cpu_world = copy.deepcopy(world)
    cpu_before = copy.deepcopy(cells)
    cpu_cells = copy.deepcopy(cells)
    for cell in cpu_cells:
        cell._replicate_genome(cpu_world, dt, cpu_world.config)
    if cpu_world.rng.bit_generator.state != tape.rng_after_state:
        raise AssertionError('A4.6b2 CPU RNG after-state differs from tape')

    post_ragged = a4.FullFidelityA4GenomeAdapter(capacity).pack_cells(
        cpu_cells,
    )
    post_state = a4.pack_a4_translation_state(
        cpu_cells, post_ragged, cpu_world.config, capacity,
    )
    for ci, (before, actual) in enumerate(zip(cpu_before, cpu_cells)):
        length = int(plan.final_lengths[ci])
        final = np.asarray(plan.final_symbols[ci, :length], dtype=np.uint8)
        if (len(actual.genomes) != len(before.genomes) + 1
                or not np.array_equal(final, actual.genomes[-1])):
            raise AssertionError(
                'A4.6b2 cell[%d] final polymer differs' % ci
            )
        v3.assert_recursive_close(
            np.asarray(plan.pools_after[ci], dtype=np.float64),
            np.asarray(actual.pools, dtype=np.float64),
            atol=2e-12, rtol=0.0,
            path='a46b2.cell[%d].pools' % ci,
        )
        v3.assert_recursive_close(
            float(plan.new_genome_lesions[ci]),
            float(actual.genome_lesions[-1]),
            atol=2e-12, rtol=0.0,
            path='a46b2.cell[%d].lesion' % ci,
        )
        for label, planned, observed in (
                ('last_symbols', plan.last_replication_symbols[ci],
                 actual.last_replication_symbols),
                ('cycle_delta', plan.replication_cycle_deltas[ci],
                 actual.replication_cycles - before.replication_cycles),
                ('genome_count', plan.genome_count_after[ci],
                 len(actual.genomes)),
                ('material_symbols', plan.genome_material_symbols_after[ci],
                 post_state.genome_material_symbols[ci])):
            if int(planned) != int(observed):
                raise AssertionError(
                    'A4.6b2 cell[%d] %s differs' % (ci, label)
                )
        for label, planned, observed in (
                ('effective_error', plan.last_effective_error_rate[ci],
                 actual.last_effective_error_rate),
                ('proof_atp', plan.cumulative_proofreading_atp_after[ci],
                 actual.cumulative_proofreading_atp),
                ('lesion_mean', plan.genome_lesion_mean_after[ci],
                 post_state.genome_lesion_mean[ci])):
            v3.assert_recursive_close(
                float(planned), float(observed), atol=2e-12, rtol=0.0,
                path='a46b2.cell[%d].%s' % (ci, label),
            )
        structural = np.asarray([
            int(actual.mutation_events[name])
            - int(before.mutation_events[name])
            for name in a44.STRUCTURAL_EVENT_NAMES
        ], dtype=np.int64)
        substitution = (
            int(actual.mutation_events['substitution'])
            - int(before.mutation_events['substitution'])
        )
        if (not np.array_equal(
                structural, plan.structural_event_counts[ci])
                or substitution != int(plan.substitution_events[ci])
                or bool(plan.replication_active_after[ci])
                or float(plan.replication_template_lesions_after[ci]) != 0.0
                or float(plan.replication_fractional_after[ci]) != 0.0):
            raise AssertionError(
                'A4.6b2 cell[%d] event/reset descriptor differs' % ci
            )
        pre_first = int(ragged.cell_sequence_offsets[ci])
        pre_last = int(ragged.cell_sequence_offsets[ci + 1])
        post_first = int(post_ragged.cell_sequence_offsets[ci])
        post_last = int(post_ragged.cell_sequence_offsets[ci + 1])
        pre_symbols = int(
            ragged.sequence_offsets[pre_last]
            - ragged.sequence_offsets[pre_first]
        )
        post_symbols = int(
            post_ragged.sequence_offsets[post_last]
            - post_ragged.sequence_offsets[post_first]
        )
        if (int(plan.topology_sequence_deltas[ci])
                != (post_last - post_first) - (pre_last - pre_first)
                or int(plan.topology_symbol_deltas[ci])
                != post_symbols - pre_symbols):
            raise AssertionError(
                'A4.6b2 cell[%d] topology delta differs' % ci
            )

    if world.rng.bit_generator.state != rng_before:
        raise AssertionError('A4.6b2 changed source world or live RNG')
    v3.assert_recursive_close(
        world_before, world.state_dict(), atol=0.0, rtol=0.0,
        path='a46b2.source_world',
    )
    for ci, (before, cell) in enumerate(zip(source_before, cells)):
        v3.assert_recursive_close(
            before, cell.state_dict(), atol=0.0, rtol=0.0,
            path='a46b2.source_cell[%d]' % ci,
        )
    for label, before, after in (
            ('ragged', ragged_before, ragged.state_dict()),
            ('state', state_before, state.state_dict()),
            ('cache', cache_before, binding.cache.state_dict())):
        v3.assert_recursive_close(
            before, after, atol=0.0, rtol=0.0,
            path='a46b2.nonmutation.%s' % label,
        )

    # A short completed polymer may be padded by the frozen mutator and then
    # trimmed back to its pre-structural length by a zero material budget.
    # MIN_GENOME_LENGTH is therefore not an unconditional post-mutation floor.
    short_world = copy.deepcopy(world)
    short_cell = copy.deepcopy(cells[2])
    short_cell.pools[a4.a3.POOL_NUCLEOTIDE] = float(a4.g2.MONOMER_MASS)
    short_rng_before = copy.deepcopy(short_world.rng.bit_generator.state)
    short_ragged, short_state, short_binding = _paid_replication_binding(
        [short_cell], short_world.config, capacity,
    )
    short_tape = a44.prepare_completion_mutation_rng_tape(
        short_binding, dt, short_world.config, short_rng_before,
    )
    short_plan = a44.paid_replication_completion_mutation_numpy(
        short_binding, dt, short_world.config, short_tape,
    )
    short_cpu_world = copy.deepcopy(short_world)
    short_cpu = copy.deepcopy(short_cell)
    short_cpu._replicate_genome(
        short_cpu_world, dt, short_cpu_world.config,
    )
    short_length = int(short_plan.final_lengths[0])
    if (int(short_tape.pre_structural_lengths[0]) != 8
            or int(short_tape.nucleotide_budget_symbols[0]) != 0
            or int(short_tape.padding_count[0]) <= 0
            or int(short_plan.material_delta_symbols[0]) != 0
            or short_length != 8
            or not np.array_equal(
                short_plan.final_symbols[0, :short_length],
                short_cpu.genomes[-1],
            )
            or short_cpu_world.rng.bit_generator.state
            != short_tape.rng_after_state):
        raise AssertionError(
            'A4.6b2 material-starved short completion differs'
        )
    short_devices = ['cpu']
    if torch is not None and torch.cuda.is_available():
        short_devices.append('cuda')
    elif _REQUIRE_CUDA:
        raise AssertionError('CUDA required but unavailable')
    for device in short_devices:
        short_resident_ragged = short_ragged.to_torch(device=device)
        short_resident_state = short_state.to_torch(device=device)
        short_resident_binding = a4.bind_a4_translation(
            short_resident_ragged, short_resident_state,
        )
        short_resident_tape = short_tape.to_torch(
            short_binding, dt, short_world.config, device=device,
        )
        short_back = a44.paid_replication_completion_mutation_torch(
            short_resident_binding, dt, short_world.config,
            short_resident_tape,
        ).to_numpy()
        v3.assert_recursive_close(
            short_plan.state_dict(), short_back.state_dict(),
            atol=2e-12, rtol=0.0,
            path='a46b2.short_material.%s' % device,
        )

    # The host PCG64 tape owns Python's float64 division/int budget.  CUDA may
    # evaluate an exact CPU integer quotient one ulp below the boundary; it
    # must validate the attested bucket rather than re-truncate to 18.
    boundary_world = copy.deepcopy(world)
    boundary_cell = copy.deepcopy(cells[0])
    boundary_cell.pools[a4.a3.POOL_NUCLEOTIDE] = (
        20.0 * float(a4.g2.MONOMER_MASS)
    )
    boundary_rng_before = copy.deepcopy(
        boundary_world.rng.bit_generator.state,
    )
    boundary_ragged, boundary_state, boundary_binding = (
        _paid_replication_binding(
            [boundary_cell], boundary_world.config, capacity,
        )
    )
    boundary_tape = a44.prepare_completion_mutation_rng_tape(
        boundary_binding, dt, boundary_world.config, boundary_rng_before,
    )
    if int(boundary_tape.nucleotide_budget_symbols[0]) != 19:
        raise AssertionError('A4.6b2 exact material budget fixture drifted')
    boundary_plan = a44.paid_replication_completion_mutation_numpy(
        boundary_binding, dt, boundary_world.config, boundary_tape,
    )
    boundary_cpu_world = copy.deepcopy(boundary_world)
    boundary_cpu = copy.deepcopy(boundary_cell)
    boundary_cpu._replicate_genome(
        boundary_cpu_world, dt, boundary_cpu_world.config,
    )
    boundary_length = int(boundary_plan.final_lengths[0])
    if (not np.array_equal(
            boundary_plan.final_symbols[0, :boundary_length],
            boundary_cpu.genomes[-1],
        ) or boundary_cpu_world.rng.bit_generator.state
            != boundary_tape.rng_after_state):
        raise AssertionError('A4.6b2 exact material budget differs')
    for device in short_devices:
        boundary_resident_ragged = boundary_ragged.to_torch(device=device)
        boundary_resident_state = boundary_state.to_torch(device=device)
        boundary_resident_binding = a4.bind_a4_translation(
            boundary_resident_ragged, boundary_resident_state,
        )
        boundary_resident_tape = boundary_tape.to_torch(
            boundary_binding, dt, boundary_world.config, device=device,
        )
        boundary_back = a44.paid_replication_completion_mutation_torch(
            boundary_resident_binding, dt, boundary_world.config,
            boundary_resident_tape,
        ).to_numpy()
        v3.assert_recursive_close(
            boundary_plan.state_dict(), boundary_back.state_dict(),
            atol=2e-12, rtol=0.0,
            path='a46b2.exact_material.%s' % device,
        )

    # Seven old lesions plus the new lesion cross NumPy's eight-element
    # reduction boundary.  The post-state mean must reduce the combined prefix
    # once; summing seven and then adding the eighth changes one fp64 bit here.
    lesion_world, lesion_cells, _, lesion_dt = (
        _a46b1_completion_mutation_fixture(seed=16001)
    )
    lesion_cell = copy.deepcopy(lesion_cells[0])
    lesion_root = np.asarray(lesion_cell.genomes[0], dtype=np.uint8).copy()
    lesion_cell.genomes = [lesion_root.copy() for _ in range(7)]
    lesion_cell.genome_lesions = [float.fromhex(value) for value in (
        '0x1.3f6d854147672p+18',
        '0x1.f8bc5d71f2247p-11',
        '0x1.8758cfd9c680cp+38',
        '0x1.36d1c260cd486p-29',
        '0x1.b70768dd2cda4p+38',
        '0x1.f4c3db9ec2057p+23',
        '0x1.521441496b18bp-22',
    )]
    lesion_cell.replication_template = lesion_root.copy()
    lesion_cell.replication_copy = [int(value) for value in lesion_root[:-1]]
    lesion_cell.replication_template_lesion = float.fromhex(
        '0x1.dc5dfd02182dbp-9'
    )
    lesion_cell._refresh_gene_cache()
    lesion_cell._sync_protein_pool()
    lesion_capacity = a4.GPU068A4Config(
        max_cells=1, max_sequences=10, max_symbols=8192,
        max_sequence_symbols=a4.MAX_FROZEN_GENOME_SYMBOLS,
        max_proteins_per_cell=64,
    )
    lesion_rng_before = copy.deepcopy(
        lesion_world.rng.bit_generator.state,
    )
    lesion_ragged, lesion_state, lesion_binding = _paid_replication_binding(
        [lesion_cell], lesion_world.config, lesion_capacity,
    )
    lesion_tape = a44.prepare_completion_mutation_rng_tape(
        lesion_binding, lesion_dt, lesion_world.config, lesion_rng_before,
    )
    lesion_plan = a44.paid_replication_completion_mutation_numpy(
        lesion_binding, lesion_dt, lesion_world.config, lesion_tape,
    )
    lesion_cpu_world = copy.deepcopy(lesion_world)
    lesion_cpu = copy.deepcopy(lesion_cell)
    lesion_cpu._replicate_genome(
        lesion_cpu_world, lesion_dt, lesion_cpu_world.config,
    )
    cpu_lesion_mean = float(np.mean(
        np.asarray(lesion_cpu.genome_lesions, dtype=np.float64),
        dtype=np.float64,
    ))
    if (float(lesion_plan.genome_lesion_mean_after[0]).hex()
            != cpu_lesion_mean.hex()):
        raise AssertionError('A4.6b2 combined lesion grouping differs')
    for device in short_devices:
        lesion_resident_ragged = lesion_ragged.to_torch(device=device)
        lesion_resident_state = lesion_state.to_torch(device=device)
        lesion_resident_binding = a4.bind_a4_translation(
            lesion_resident_ragged, lesion_resident_state,
        )
        lesion_resident_tape = lesion_tape.to_torch(
            lesion_binding, lesion_dt, lesion_world.config, device=device,
        )
        lesion_back = a44.paid_replication_completion_mutation_torch(
            lesion_resident_binding, lesion_dt, lesion_world.config,
            lesion_resident_tape,
        ).to_numpy()
        if (float(lesion_back.genome_lesion_mean_after[0]).hex()
                != cpu_lesion_mean.hex()):
            raise AssertionError(
                'A4.6b2 combined lesion grouping differs on %s' % device
            )

    # Recovering the inherited component by subtracting the A4.6a error term
    # loses low bits for large finite signals.  Recompute the frozen
    # template/proofreading term directly and preserve its grouping.
    cancel_world, cancel_cells, cancel_capacity, cancel_dt = (
        _a46b1_completion_mutation_fixture(seed=18031)
    )
    cancel_world.config.mutation_rate = 1.0088277385404213e-07
    cancel_cell = copy.deepcopy(cancel_cells[0])
    cancel_cell.replication_template_lesion = 312534361.6960746
    cancel_cell.genome_lesions = [1.0]
    cancel_cell.pools[a4.a3.POOL_REACTIVE] = 2311896.74665122
    cancel_rng_before = copy.deepcopy(
        cancel_world.rng.bit_generator.state,
    )
    cancel_ragged, cancel_state, cancel_binding = _paid_replication_binding(
        [cancel_cell], cancel_world.config, cancel_capacity,
    )
    cancel_tape = a44.prepare_completion_mutation_rng_tape(
        cancel_binding, cancel_dt, cancel_world.config, cancel_rng_before,
    )
    cancel_plan = a44.paid_replication_completion_mutation_numpy(
        cancel_binding, cancel_dt, cancel_world.config, cancel_tape,
    )
    cancel_cpu_world = copy.deepcopy(cancel_world)
    cancel_cpu = copy.deepcopy(cancel_cell)
    cancel_cpu._replicate_genome(
        cancel_cpu_world, cancel_dt, cancel_cpu_world.config,
    )
    cpu_new_lesion = float(cancel_cpu.genome_lesions[-1])
    if float(cancel_plan.new_genome_lesions[0]).hex() != cpu_new_lesion.hex():
        raise AssertionError('A4.6b2 inherited lesion grouping differs')
    for device in short_devices:
        cancel_resident_ragged = cancel_ragged.to_torch(device=device)
        cancel_resident_state = cancel_state.to_torch(device=device)
        cancel_resident_binding = a4.bind_a4_translation(
            cancel_resident_ragged, cancel_resident_state,
        )
        cancel_resident_tape = cancel_tape.to_torch(
            cancel_binding, cancel_dt, cancel_world.config, device=device,
        )
        cancel_back = a44.paid_replication_completion_mutation_torch(
            cancel_resident_binding, cancel_dt, cancel_world.config,
            cancel_resident_tape,
        ).to_numpy()
        v3.assert_recursive_close(
            float(cancel_back.new_genome_lesions[0]), cpu_new_lesion,
            atol=2e-12, rtol=0.0,
            path='a46b2.inherited_lesion.%s' % device,
        )
    return ('3 Formal066 completions plus material-starved short padding/trim '
            'exact integer budget, seven-to-eight lesion grouping, and large '
            'inherited-lesion cancellation: final polymer, paid/refunded '
            'material, lesion, counters, reset, derived physiology and '
            'topology exact')


def test_a46b2_completion_mutation_numpy_torch_devices():
    if torch is None:
        raise AssertionError('PyTorch is required for A4.6b2')
    world, cells, capacity, dt = _a46b1_completion_mutation_fixture(
        seed=9912,
    )
    rng_before = copy.deepcopy(world.rng.bit_generator.state)
    source_before = [v3.pickle_clone(cell.state_dict()) for cell in cells]
    ragged, state, binding = _paid_replication_binding(
        cells, world.config, capacity,
    )
    tape = a44.prepare_completion_mutation_rng_tape(
        binding, dt, world.config, rng_before,
    )
    expected = a44.paid_replication_completion_mutation_numpy(
        binding, dt, world.config, tape,
    )
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    elif _REQUIRE_CUDA:
        raise AssertionError('CUDA required but unavailable')
    for device in devices:
        resident_ragged = ragged.to_torch(device=device)
        resident_state = state.to_torch(device=device)
        resident_binding = a4.bind_a4_translation(
            resident_ragged, resident_state,
        )
        resident_tape = tape.to_torch(
            binding, dt, world.config, device=device,
        )
        ragged_ptrs = resident_ragged.data_ptrs()
        state_ptrs = resident_state.data_ptrs()
        cache_ptrs = resident_binding.cache.data_ptrs()
        tape_ptrs = resident_tape.data_ptrs()
        plan = a44.paid_replication_completion_mutation_torch(
            resident_binding, dt, world.config, resident_tape,
        )
        for name in a44._COMPLETION_PLAN_ARRAY_FIELDS:
            value = getattr(plan, name)
            if (not isinstance(value, torch.Tensor)
                    or value.device.type != device):
                raise AssertionError(
                    '%s A4.6b2 %s escaped device' % (device, name)
                )
        readback = plan.to_numpy()
        v3.assert_recursive_close(
            expected.state_dict(), readback.state_dict(),
            atol=2e-12, rtol=0.0,
            path='a46b2.device.%s' % device,
        )
        if (resident_ragged.data_ptrs() != ragged_ptrs
                or resident_state.data_ptrs() != state_ptrs
                or resident_binding.cache.data_ptrs() != cache_ptrs
                or resident_tape.data_ptrs() != tape_ptrs):
            raise AssertionError('%s A4.6b2 input pointer changed' % device)
    forbidden = ('.item(', '.cpu(', '.numpy(', '.tolist(',
                 'nonzero(', 'masked_select(', 'unique(')
    source = inspect.getsource(
        a44.paid_replication_completion_mutation_torch,
    )
    hits = [token for token in forbidden if token in source]
    if hits:
        raise AssertionError('A4.6b2 Torch path contains host op: %s' % hits)
    if world.rng.bit_generator.state != rng_before:
        raise AssertionError('A4.6b2 device apply advanced live RNG')
    for ci, (before, cell) in enumerate(zip(source_before, cells)):
        v3.assert_recursive_close(
            before, cell.state_dict(), atol=0.0, rtol=0.0,
            path='a46b2.device_source[%d]' % ci,
        )
    return 'NumPy/Torch %s fixed-shape final polymer+ledger parity' % '/'.join(
        devices,
    )


def test_a46b2_completion_mutation_fail_closed_and_authority():
    world, cells, capacity, dt = _a46b1_completion_mutation_fixture(
        seed=9913,
    )
    rng_before = copy.deepcopy(world.rng.bit_generator.state)
    ragged, state, binding = _paid_replication_binding(
        cells, world.config, capacity,
    )
    tape = a44.prepare_completion_mutation_rng_tape(
        binding, dt, world.config, rng_before,
    )
    plan = a44.paid_replication_completion_mutation_numpy(
        binding, dt, world.config, tape,
    )
    corruptions = []
    bad = plan.clone()
    bad.final_symbols[0, int(bad.final_lengths[0])] = 1
    corruptions.append(('final_tail', bad))
    bad = plan.clone()
    bad.final_lengths[0] = 0
    corruptions.append(('final_length', bad))
    bad = plan.clone()
    bad.material_delta_symbols[0] += 1
    corruptions.append(('material_delta', bad))
    bad = plan.clone()
    bad.new_genome_lesions[0] = -1.0
    corruptions.append(('lesion', bad))
    bad = plan.clone()
    bad.topology_symbol_deltas[0] += 1
    corruptions.append(('topology', bad))
    bad = plan.clone()
    bad.replication_active_after[0] = True
    corruptions.append(('active_reset', bad))
    bad = plan.clone()
    bad.genome_count_after[0] = 0
    corruptions.append(('genome_count', bad))
    bad = plan.clone()
    bad.genome_material_symbols_after[0] = 0
    corruptions.append(('genome_material', bad))
    bad = plan.clone()
    bad.genome_lesion_mean_after[0] = np.nan
    corruptions.append(('lesion_mean', bad))
    for label, corrupt in corruptions:
        _assert_raises(
            a4.A4SchemaError,
            lambda corrupt=corrupt: a44.validate_a4_completion_mutation_plan(
                corrupt,
            ),
        )

    if torch is None:
        raise AssertionError('PyTorch is required for A4.6b2 trust test')
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    elif _REQUIRE_CUDA:
        raise AssertionError('CUDA required but unavailable')
    for device in devices:
        resident_binding = a4.bind_a4_translation(
            ragged.to_torch(device=device),
            state.to_torch(device=device),
        )
        resident_tape = tape.to_torch(
            binding, dt, world.config, device=device,
        )
        before_version = int(resident_tape.insertion_symbols._version)
        resident_tape.insertion_symbols.data[0, 0] = (
            resident_tape.insertion_symbols.data[0, 0] + 1
        ) % int(a4.g2.ALPHABET_SIZE)
        if int(resident_tape.insertion_symbols._version) != before_version:
            raise AssertionError('%s .data unexpectedly changed version' % device)
        failed = a44.paid_replication_completion_mutation_torch(
            resident_binding, dt, world.config, resident_tape,
        )
        codes = failed.scope_error_code.detach().cpu().numpy()
        valid = failed.scope_valid.detach().cpu().numpy()
        final = failed.final_symbols.detach().cpu().numpy()
        cycles = failed.replication_cycle_deltas.detach().cpu().numpy()
        if (not np.all(codes[:len(cells)] == a44.SCOPE_RNG_TAPE_MISMATCH)
                or np.any(valid[:len(cells)])
                or np.any(final[:len(cells)])
                or np.any(cycles[:len(cells)])):
            raise AssertionError(
                '%s A4.6b2 hidden tape mutation did not roll back batch' %
                device
            )
        _assert_raises(a44.A4ReplicationScopeError, failed.to_numpy)
    if (world.rng.bit_generator.state != rng_before
            or a44.FULL_GPU_WORLD_STEP is not False
            or v3._event_order().count('replication_cpu') != 1):
        raise AssertionError('A4.6b2 CPU/RNG/scheduler authority changed')
    return ('%d plan corruptions and CPU/CUDA hidden tape mutation rejected; '
            'batch rollback, live RNG and A3 replication authority preserved' %
            len(corruptions))


def test_a47a_hydrolysis_literal_boundaries_and_zero_probability_draw():
    world, cell, capacity, dt = _a47_hydrolysis_fixture(boundaries=True)
    world_before = v3.pickle_clone(world.state_dict())
    cell_before = v3.pickle_clone(cell.state_dict())
    rng_before = copy.deepcopy(world.rng.bit_generator.state)
    ragged, state, binding = _paid_replication_binding(
        [cell], world.config, capacity,
    )
    ragged_before = ragged.state_dict()
    state_before = state.state_dict()
    cache_before = binding.cache.state_dict()

    tape = a47.prepare_genome_hydrolysis_rng_tape(
        binding, dt, rng_before,
    )
    if a47.validate_a4_hydrolysis_rng_tape(
            tape, binding, dt) is not tape:
        raise AssertionError('A4.7a validator did not return its tape')
    expected_slots = np.asarray(
        [True, True, True, False, False], dtype=bool,
    )
    expected_draws = np.asarray(
        [False, False, True, False, False], dtype=bool,
    )
    if (not np.array_equal(tape.genome_slot_mask, expected_slots)
            or not np.array_equal(tape.draw_mask, expected_draws)
            or np.any(tape.hit_mask)
            or np.any(tape.deletion_positions != -1)
            or int(tape.draw_count) != 1
            or int(tape.hit_count) != 0):
        raise AssertionError('A4.7a literal boundary schedule differs')

    replay_bitgen = np.random.PCG64()
    replay_bitgen.state = copy.deepcopy(rng_before)
    replay_rng = np.random.Generator(replay_bitgen)
    expected_uniform = float(replay_rng.random())
    if (float(tape.uniform_draws[2]).hex() != expected_uniform.hex()
            or np.any(tape.uniform_draws[[0, 1, 3, 4]] != 0.0)
            or tape.rng_after_state != replay_rng.bit_generator.state):
        raise AssertionError('A4.7a zero-probability RNG call differs')
    if (tape.rng_after_state != {
            'bit_generator': 'PCG64',
            'state': {
                'state': 126889499338173139678050600050083445318,
                'inc': 121863417007658695389390353187995180015,
            },
            'has_uint32': 1,
            'uinteger': 1123615560,
            }):
        raise AssertionError('A4.7a zero-dt PCG64 cache fixture drifted')

    if world.rng.bit_generator.state != rng_before:
        raise AssertionError('A4.7a boundary preparation advanced live RNG')
    for label, before, after in (
            ('world', world_before, world.state_dict()),
            ('cell', cell_before, cell.state_dict()),
            ('ragged', ragged_before, ragged.state_dict()),
            ('state', state_before, state.state_dict()),
            ('cache', cache_before, binding.cache.state_dict())):
        v3.assert_recursive_close(
            before, after, atol=0.0, rtol=0.0,
            path='a47a.boundary.%s' % label,
        )
    return ('lesion==0.75 and length==MIN skip draws; eligible dt=0 '
            'consumes exactly one random() with no integer/hit')


def test_a47a_hydrolysis_formal066_hit_miss_hit_oracle_and_purity():
    world, cell, capacity, dt = _a47_hydrolysis_fixture()
    world_before = v3.pickle_clone(world.state_dict())
    cell_before = v3.pickle_clone(cell.state_dict())
    rng_before = copy.deepcopy(world.rng.bit_generator.state)
    ragged, state, binding = _paid_replication_binding(
        [cell], world.config, capacity,
    )
    ragged_before = ragged.state_dict()
    state_before = state.state_dict()
    cache_before = binding.cache.state_dict()
    tape = a47.prepare_genome_hydrolysis_rng_tape(
        binding, dt, rng_before,
    )

    expected_uniforms = np.asarray([
        0.2984911434141233,
        0.8142257405942803,
        0.0919159421350969,
    ], dtype=np.float64)
    expected_hits = np.asarray(
        [True, False, True, False, False], dtype=bool,
    )
    expected_positions = np.asarray(
        [12, -1, 16, -1, -1], dtype=np.int64,
    )
    if (int(tape.draw_count) != 3 or int(tape.hit_count) != 2
            or not np.all(tape.draw_mask[:3])
            or np.any(tape.draw_mask[3:])
            or not np.array_equal(tape.hit_mask, expected_hits)
            or not np.array_equal(
                tape.deletion_positions, expected_positions,
            )
            or not np.array_equal(
                tape.uniform_draws[:3], expected_uniforms,
            )
            or np.any(tape.uniform_draws[3:] != 0.0)):
        raise AssertionError('A4.7a hit/miss/hit call schedule differs')
    expected_after = {
        'bit_generator': 'PCG64',
        'state': {
            'state': 94939001618465750405315260005713465823,
            'inc': 121863417007658695389390353187995180015,
        },
        'has_uint32': 1,
        'uinteger': 2577412133,
    }
    if tape.rng_after_state != expected_after:
        raise AssertionError('A4.7a conditional integer/cache order differs')

    manual = _a47_manual_apply_tape(cell, tape)
    cpu_world = copy.deepcopy(world)
    cpu_cell = copy.deepcopy(cell)
    cpu_world.cells = [cpu_cell]
    cpu_cell._decay_information_and_proteins(cpu_world, dt)
    if cpu_world.rng.bit_generator.state != tape.rng_after_state:
        raise AssertionError('A4.7a after-state differs from Formal066')
    if ([len(genome) for genome in cpu_cell.genomes] != [47, 48, 47]
            or int(cpu_cell.genome_damage_events)
            - int(cell.genome_damage_events) != 2):
        raise AssertionError('A4.7a Formal066 deletion topology differs')
    for genome_index, (expected, actual) in enumerate(zip(
            manual.genomes, cpu_cell.genomes)):
        if not np.array_equal(expected, actual):
            raise AssertionError(
                'A4.7a genome[%d] differs from Formal066' % genome_index
            )
    if ([float(value).hex() for value in cpu_cell.genome_lesions] != [
            '0x1.33b13b13b13b2p+9',
            '0x1.809d89d89d89ep+9',
            '0x1.33b13b13b13b2p+9',
            ]
            or [float(value).hex() for value in manual.genome_lesions]
            != [float(value).hex() for value in cpu_cell.genome_lesions]):
        raise AssertionError('A4.7a lesion attenuation differs')

    waste_index = int(a4.a3.POOL_WASTE)
    cpu_waste = float(cpu_cell.pools[waste_index])
    manual_waste = float(manual.pools[waste_index])
    naive_waste = 0.004 + 2.0 * float(a4.g2.MONOMER_MASS)
    if (cpu_waste.hex() != '0x1.719f7f8ca8198p-8'
            or manual_waste.hex() != cpu_waste.hex()
            or naive_waste.hex() == cpu_waste.hex()):
        raise AssertionError('A4.7a ordered waste ledger differs')
    _assert_gene_specs_equal(
        cpu_cell.gene_specs, manual.gene_specs, 'a47a.gene_cache',
    )
    remaining_roles = [
        int(spec['role']) for spec in cpu_cell.gene_specs.values()
    ]
    if remaining_roles != [int(a4.g2.ROLE_MEMBRANE)]:
        raise AssertionError('A4.7a final gene cache refresh differs')

    if world.rng.bit_generator.state != rng_before:
        raise AssertionError('A4.7a preparation advanced the live RNG')
    for label, before, after in (
            ('world', world_before, world.state_dict()),
            ('cell', cell_before, cell.state_dict()),
            ('ragged', ragged_before, ragged.state_dict()),
            ('state', state_before, state.state_dict()),
            ('cache', cache_before, binding.cache.state_dict())):
        v3.assert_recursive_close(
            before, after, atol=0.0, rtol=0.0,
            path='a47a.oracle_nonmutation.%s' % label,
        )
    return ('one-cell hit/miss/hit random->conditional-integer order, full '
            'PCG64 cache, polymer/waste/lesion/event/cache exact')


def test_a47a_hydrolysis_torch_trust_scope_and_a3_authority():
    if torch is None:
        raise AssertionError('PyTorch is required for A4.7a')
    world, cell, capacity, dt = _a47_hydrolysis_fixture()
    world_before = v3.pickle_clone(world.state_dict())
    cell_before = v3.pickle_clone(cell.state_dict())
    rng_before = copy.deepcopy(world.rng.bit_generator.state)
    ragged, state, binding = _paid_replication_binding(
        [cell], world.config, capacity,
    )
    tape = a47.prepare_genome_hydrolysis_rng_tape(
        binding, dt, rng_before,
    )

    forged = a47.A4HydrolysisRngTape(
        _factory_token=None, **tape.state_dict()
    )
    _assert_raises(
        a4.A4SchemaError,
        lambda: a47.validate_a4_hydrolysis_rng_tape(
            forged, binding, dt,
        ),
    )
    scalar_bad = tape.clone()
    scalar_bad.cell_id += 1
    _assert_raises(
        a4.A4SchemaError,
        lambda: a47.validate_a4_hydrolysis_rng_tape(
            scalar_bad, binding, dt,
        ),
    )
    dict_bad = tape.clone()
    dict_bad.rng_after_state['state']['state'] ^= 1
    _assert_raises(
        a4.A4SchemaError,
        lambda: a47.validate_a4_hydrolysis_rng_tape(
            dict_bad, binding, dt,
        ),
    )
    array_bad = tape.clone()
    array_bad.uniform_draws[0] = np.nextafter(
        array_bad.uniform_draws[0], np.float64(1.0),
    )
    _assert_raises(
        a4.A4SchemaError,
        lambda: a47.validate_a4_hydrolysis_rng_tape(
            array_bad, binding, dt,
        ),
    )
    tail_bad = tape.clone()
    tail_index = int(tape.sequence_count)
    if tail_index >= int(tape.sequence_capacity):
        raise AssertionError('A4.7a trust fixture lacks an unused tail')
    tail_bad.deletion_positions[tail_index] = 0
    _assert_raises(
        a4.A4SchemaError,
        lambda: a47.validate_a4_hydrolysis_rng_tape(
            tail_bad, binding, dt,
        ),
    )
    wrong_dt = float(np.nextafter(
        np.float64(dt), np.float64(np.inf),
    ))
    _assert_raises(
        a47.A4HydrolysisScopeError,
        lambda: a47.validate_a4_hydrolysis_rng_tape(
            tape, binding, wrong_dt,
        ),
    )

    stale_cell = copy.deepcopy(cell)
    stale_cell.genome_lesions[0] = np.nextafter(
        stale_cell.genome_lesions[0], np.float64(np.inf),
    )
    _, _, stale_binding = _paid_replication_binding(
        [stale_cell], world.config, capacity,
    )
    _assert_raises(
        a47.A4HydrolysisScopeError,
        lambda: a47.validate_a4_hydrolysis_rng_tape(
            tape, stale_binding, dt,
        ),
    )

    second = copy.deepcopy(cell)
    second.cell_id = int(cell.cell_id) + 1000000
    multi_capacity = a4.GPU068A4Config(
        max_cells=2, max_sequences=10,
        max_symbols=2 * sum(len(genome) for genome in cell.genomes),
        max_sequence_symbols=48, max_proteins_per_cell=64,
    )
    _, _, multi_binding = _paid_replication_binding(
        [cell, second], world.config, multi_capacity,
    )
    _assert_raises(
        a47.A4HydrolysisScopeError,
        lambda: a47.prepare_genome_hydrolysis_rng_tape(
            multi_binding, dt, rng_before,
        ),
    )

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    elif _REQUIRE_CUDA:
        raise AssertionError('CUDA required but unavailable')
    for device in devices:
        resident = tape.to_torch(binding, dt, device=device)
        pointers = resident.data_ptrs()
        for name in a47._TAPE_ARRAY_FIELDS:
            value = getattr(resident, name)
            if (not isinstance(value, torch.Tensor)
                    or value.device.type != device):
                raise AssertionError(
                    '%s A4.7a %s escaped device' % (device, name)
                )
        readback = resident.to_numpy()
        if a47.validate_a4_hydrolysis_rng_tape(
                readback, binding, dt) is not readback:
            raise AssertionError('%s A4.7a readback was not validated' % device)
        for name, expected in tape.state_dict().items():
            actual = getattr(readback, name)
            if isinstance(expected, np.ndarray):
                equal = np.array_equal(expected, actual)
            else:
                equal = expected == actual
            if not equal:
                raise AssertionError(
                    '%s A4.7a %s roundtrip differs' % (device, name)
                )
        if resident.data_ptrs() != pointers:
            raise AssertionError('%s A4.7a tape pointer changed' % device)

        hidden = tape.to_torch(binding, dt, device=device)
        version = int(hidden.uniform_draws._version)
        hidden.uniform_draws.data[0] = 0.0
        if int(hidden.uniform_draws._version) != version:
            raise AssertionError('%s .data unexpectedly changed version' % device)
        _assert_raises(a4.A4SchemaError, hidden.to_numpy)

        hidden_expected = tape.to_torch(binding, dt, device=device)
        expected_tensor = hidden_expected._resident_expected_arrays[
            'uniform_draws'
        ]
        expected_version = int(expected_tensor._version)
        expected_tensor.data[0] = 0.0
        if int(expected_tensor._version) != expected_version:
            raise AssertionError(
                '%s expected .data unexpectedly changed version' % device
            )
        _assert_raises(a4.A4SchemaError, hidden_expected.to_numpy)

    expected_status = (
        'a4.7a-single-cell-post-gain-literal-pcg64-event-tape-'
        'not-applied-not-live-rng-committed-not-integrated-cpu-authoritative'
    )
    if (world.rng.bit_generator.state != rng_before
            or a47.FULL_GPU_WORLD_STEP is not False
            or a47.PORT_STATUS.get('full_gpu_world_step') is not False
            or a47.PORT_STATUS.get('genome_symbol_hydrolysis_rng')
            != expected_status
            or a4.a3.PORT_STATUS.get('genome_symbol_hydrolysis_rng')
            != 'cpu-authoritative-explicit-hazard-plan'
            or v3._event_order().count('genome_hydrolysis_cpu_rng') != 1):
        raise AssertionError('A4.7a changed A3 hydrolysis authority')
    for label, before, after in (
            ('world', world_before, world.state_dict()),
            ('cell', cell_before, cell.state_dict())):
        v3.assert_recursive_close(
            before, after, atol=0.0, rtol=0.0,
            path='a47a.trust_nonmutation.%s' % label,
        )
    return ('NumPy/Torch %s explicit readback; factory/scalar/dict/array/'
            'tail/dt/stale/multicell/.data trust rejection; A3 authority retained' %
            '/'.join(devices))


def test_a47b_hydrolysis_apply_formal066_oracle_and_noop():
    world, cell, capacity, dt = _a47b_hydrolysis_fixture()
    rng_before = copy.deepcopy(world.rng.bit_generator.state)
    ragged, state, binding = _paid_replication_binding(
        [cell], world.config, capacity,
    )
    tape = a47.prepare_genome_hydrolysis_rng_tape(
        binding, dt, rng_before,
    )
    plan = a47.genome_hydrolysis_deletion_numpy(
        binding, dt, tape,
    )
    if a47.validate_a4_hydrolysis_deletion_plan(
            plan, binding, dt, tape) is not plan:
        raise AssertionError('A4.7b validator did not return its plan')
    cpu_world, actual, expected = _a47b_formal066_oracle(
        world, cell, dt,
    )
    if (not bool(plan.scope_valid[0])
            or int(plan.scope_error_code[0]) != 0
            or not np.array_equal(
                plan.final_symbols[:5], expected['final_symbols'],
            )
            or not np.array_equal(
                plan.final_lengths[:5], expected['final_lengths'],
            )
            or int(plan.symbol_count_after[0])
            != expected['final_symbol_count']
            or int(plan.topology_symbol_delta[0])
            != expected['topology_symbol_delta']
            or int(plan.genome_damage_event_delta[0])
            != expected['genome_damage_event_delta']
            or int(plan.genome_material_symbols_after[0])
            != expected['genome_material_symbols_after']
            or bool(plan.gene_cache_dirty[0]) is not True
            or int(plan.gene_cache_refresh_count[0]) != 2):
        raise AssertionError('A4.7b Formal066 discrete descriptor differs')
    if (not np.array_equal(
            plan.genome_lesions_after[:3],
            expected['genome_lesions_after'],
            )
            or np.any(plan.genome_lesions_after[3:] != 0.0)
            or not np.array_equal(
                plan.pools_after[0], expected['pools_after'],
            )
            or float(plan.genome_lesion_mean_after[0]).hex()
            != float(expected['genome_lesion_mean_after']).hex()):
        raise AssertionError('A4.7b Formal066 float descriptor differs')
    if ([int(value) for value in plan.final_lengths[:5]]
            != [47, 48, 47, 48, 7]
            or expected['final_offsets'].tolist()
            != [0, 47, 95, 142, 190, 197]
            or float(plan.pools_after[0, a4.a3.POOL_WASTE]).hex()
            != '0x1.719f7f8ca8198p-8'):
        raise AssertionError('A4.7b topology/ordered waste fixture drifted')

    template = np.asarray(cell.replication_template, dtype=np.uint8)
    copy_symbols = np.asarray(cell.replication_copy, dtype=np.uint8)
    if (not np.array_equal(plan.final_symbols[3, :48], template)
            or not np.array_equal(plan.final_symbols[4, :7], copy_symbols)):
        raise AssertionError('A4.7b touched active template/copy rows')
    final_genomes = [
        np.asarray(plan.final_symbols[index, :int(plan.final_lengths[index])],
                   dtype=np.uint8).copy()
        for index in range(3)
    ]
    conceptual_cache = a4._gene_cache_from_genomes(final_genomes)
    _assert_gene_specs_equal(
        actual.gene_specs, conceptual_cache, 'a47b.conceptual_cache',
    )
    if cpu_world.rng.bit_generator.state != tape.rng_after_state:
        raise AssertionError('A4.7b direct CPU RNG differs from tape')

    zero_world, zero_cell, zero_capacity, zero_dt = (
        _a47b_hydrolysis_fixture(zero_hit=True)
    )
    zero_ragged, _, zero_binding = _paid_replication_binding(
        [zero_cell], zero_world.config, zero_capacity,
    )
    zero_tape = a47.prepare_genome_hydrolysis_rng_tape(
        zero_binding, zero_dt,
        copy.deepcopy(zero_world.rng.bit_generator.state),
    )
    zero_plan = a47.genome_hydrolysis_deletion_numpy(
        zero_binding, zero_dt, zero_tape,
    )
    _, _, zero_expected = _a47b_formal066_oracle(
        zero_world, zero_cell, zero_dt,
    )
    source_lengths = np.diff(zero_ragged.sequence_offsets[:6])
    if (not np.array_equal(zero_plan.final_lengths[:5], source_lengths)
            or not np.array_equal(
                zero_plan.pools_after[0], zero_expected['pools_after'],
            )
            or not np.array_equal(
                zero_plan.genome_lesions_after[:3],
                zero_expected['genome_lesions_after'],
            )
            or int(zero_plan.symbol_count_after[0])
            != int(zero_ragged.symbol_count)
            or int(zero_plan.topology_symbol_delta[0]) != 0
            or int(zero_plan.genome_damage_event_delta[0]) != 0
            or bool(zero_plan.gene_cache_dirty[0])
            or int(zero_plan.gene_cache_refresh_count[0]) != 0):
        raise AssertionError('A4.7b zero-hit descriptor is not an exact no-op')
    return ('hit/miss/hit Formal066 deletion, sequential waste, lesion/event/'
            'topology/cache-dirty exact; active rows preserved; dt=0 no-op')


def test_a47b_hydrolysis_numpy_torch_devices_and_purity():
    if torch is None:
        raise AssertionError('PyTorch is required for A4.7b')
    world, cell, capacity, dt = _a47b_hydrolysis_fixture()
    world_before = v3.pickle_clone(world.state_dict())
    cell_before = v3.pickle_clone(cell.state_dict())
    rng_before = copy.deepcopy(world.rng.bit_generator.state)
    ragged, state, binding = _paid_replication_binding(
        [cell], world.config, capacity,
    )
    ragged_before = ragged.state_dict()
    state_before = state.state_dict()
    cache_before = binding.cache.state_dict()
    tape = a47.prepare_genome_hydrolysis_rng_tape(
        binding, dt, rng_before,
    )
    tape_before = tape.state_dict()
    expected = a47.genome_hydrolysis_deletion_numpy(
        binding, dt, tape,
    )
    expected_state = expected.state_dict()
    array_names = tuple(
        name for name, value in expected_state.items()
        if isinstance(value, np.ndarray)
    )

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    elif _REQUIRE_CUDA:
        raise AssertionError('CUDA required but unavailable')
    for device in devices:
        resident_ragged = ragged.to_torch(device=device)
        resident_state = state.to_torch(device=device)
        resident_binding = a4.bind_a4_translation(
            resident_ragged, resident_state,
        )
        resident_tape = tape.to_torch(
            binding, dt, device=device,
        )
        ragged_ptrs = resident_ragged.data_ptrs()
        state_ptrs = resident_state.data_ptrs()
        cache_ptrs = resident_binding.cache.data_ptrs()
        tape_ptrs = resident_tape.data_ptrs()
        resident_plan = a47.genome_hydrolysis_deletion_torch(
            resident_binding, dt, resident_tape,
        )
        for name in array_names:
            value = getattr(resident_plan, name)
            if (not isinstance(value, torch.Tensor)
                    or value.device.type != device):
                raise AssertionError(
                    '%s A4.7b %s escaped device' % (device, name)
                )
        readback = resident_plan.to_numpy()
        if a47.validate_a4_hydrolysis_deletion_plan(
                readback, binding, dt, tape) is not readback:
            raise AssertionError('%s A4.7b readback was not validated' % device)
        v3.assert_recursive_close(
            expected_state, readback.state_dict(), atol=0.0, rtol=0.0,
            path='a47b.device.%s' % device,
        )
        if (resident_ragged.data_ptrs() != ragged_ptrs
                or resident_state.data_ptrs() != state_ptrs
                or resident_binding.cache.data_ptrs() != cache_ptrs
                or resident_tape.data_ptrs() != tape_ptrs):
            raise AssertionError('%s A4.7b input pointer changed' % device)

    # Eight final lesions cross the NumPy reduction boundary.  Reducing the
    # first seven and then appending the eighth changes one fp64 bit, so the
    # plan must derive the mean from the conceptual final vector in one pass.
    mean_world, mean_cell, _, mean_dt = _a47_hydrolysis_fixture()
    mean_root = np.asarray(mean_cell.genomes[0], dtype=np.uint8).copy()
    mean_cell.genomes = [mean_root.copy() for _ in range(8)]
    mean_cell.genome_lesions = [float.fromhex(value) for value in (
        '0x1.ad76af3009b08p+10',
        '0x1.8b8030d222309p+13',
        '0x1.22aabaa71530cp+12',
        '0x1.f8640be991f30p+12',
        '0x1.2380e7928077cp+11',
        '0x1.3df56805c6afbp+11',
        '0x1.de49d4ca52714p+11',
        '0x1.32a89effea992p+11',
    )]
    mean_cell._refresh_gene_cache()
    mean_cell._sync_protein_pool()
    mean_capacity = a4.GPU068A4Config(
        max_cells=1, max_sequences=8, max_symbols=384,
        max_sequence_symbols=48, max_proteins_per_cell=64,
    )
    mean_ragged, mean_state, mean_binding = _paid_replication_binding(
        [mean_cell], mean_world.config, mean_capacity,
    )
    mean_tape = a47.prepare_genome_hydrolysis_rng_tape(
        mean_binding, mean_dt,
        copy.deepcopy(mean_world.rng.bit_generator.state),
    )
    mean_expected = a47.genome_hydrolysis_deletion_numpy(
        mean_binding, mean_dt, mean_tape,
    )
    _, _, mean_oracle = _a47b_formal066_oracle(
        mean_world, mean_cell, mean_dt,
    )
    final_lesions = mean_oracle['genome_lesions_after']
    cpu_mean = float(np.mean(final_lesions, dtype=np.float64))
    old_grouping = (
        float(np.sum(final_lesions[:7], dtype=np.float64))
        + float(final_lesions[7])
    ) / 8.0
    if (int(mean_expected.genome_damage_event_delta[0]) != 8
            or cpu_mean.hex() != '0x1.de203df070020p+11'
            or old_grouping.hex() != '0x1.de203df070021p+11'
            or float(mean_expected.genome_lesion_mean_after[0]).hex()
            != cpu_mean.hex()):
        raise AssertionError('A4.7b eight-lesion reduction fixture differs')
    for device in devices:
        mean_resident_binding = a4.bind_a4_translation(
            mean_ragged.to_torch(device=device),
            mean_state.to_torch(device=device),
        )
        mean_resident_tape = mean_tape.to_torch(
            mean_binding, mean_dt, device=device,
        )
        mean_back = a47.genome_hydrolysis_deletion_torch(
            mean_resident_binding, mean_dt, mean_resident_tape,
        ).to_numpy()
        v3.assert_recursive_close(
            float(mean_back.genome_lesion_mean_after[0]), cpu_mean,
            atol=2e-12, rtol=0.0,
            path='a47b.eight_lesion_mean.%s' % device,
        )

    # The final division for three extremely uneven but finite lesions is one
    # ULP lower on RTX CUDA than NumPy.  The pairwise numerator and every
    # non-mean field remain bit-exact; only this final quotient may be either
    # adjacent fp64 neighbour of the direct Formal066 mean.
    division_world = v3.make_world(seed=33003, cells=1)
    division_cell = division_world.cells[0]
    division_world.config.endogenous_damage = False
    division_cell.genomes = [
        _a47_gene_sequence(a4.g2.ROLE_ENERGY, 0, length=33),
        _a47_gene_sequence(a4.g2.ROLE_MEMBRANE, 0, length=34),
        _a47_gene_sequence(a4.g2.ROLE_TRANSPORTER, 16, length=35),
    ]
    division_cell.genome_lesions = [float.fromhex(value) for value in (
        '0x1.8000000000001p-1',
        '0x1.249ad2594c37dp+332',
        '0x1.249ad2594c37dp+333',
    )]
    division_cell.replication_template = None
    division_cell.replication_copy = []
    division_cell.replication_template_lesion = 0.0
    division_cell.replication_fractional = 0.0
    division_cell.membrane_oxidation[:] = 0.0
    division_cell._refresh_gene_cache()
    division_cell._sync_protein_pool()
    division_dt = 1000.0
    division_capacity = a4.GPU068A4Config(
        max_cells=1, max_sequences=3, max_symbols=102,
        max_sequence_symbols=35, max_proteins_per_cell=64,
    )
    division_ragged, division_state, division_binding = (
        _paid_replication_binding(
            [division_cell], division_world.config, division_capacity,
        )
    )
    division_tape = a47.prepare_genome_hydrolysis_rng_tape(
        division_binding, division_dt,
        copy.deepcopy(division_world.rng.bit_generator.state),
    )
    division_expected = a47.genome_hydrolysis_deletion_numpy(
        division_binding, division_dt, division_tape,
    )
    _, _, division_oracle = _a47b_formal066_oracle(
        division_world, division_cell, division_dt,
    )
    division_post = division_oracle['genome_lesions_after']
    division_mean = float(np.mean(division_post, dtype=np.float64))
    if (division_tape.hit_mask[:3].tolist() != [False, True, True]
            or [float(value).hex() for value in division_post] != [
                '0x1.8000000000001p-1',
                '0x1.d42aea2879f2fp+331',
                '0x1.d42aea2879f2fp+332',
            ]
            or float(np.sum(division_post, dtype=np.float64)).hex()
            != '0x1.5f202f9e5b763p+333'
            or division_mean.hex() != '0x1.d42aea2879f2fp+331'
            or float(division_expected.genome_lesion_mean_after[0]).hex()
            != division_mean.hex()):
        raise AssertionError('A4.7b three-lesion division fixture differs')
    division_expected_state = division_expected.state_dict()
    for device in devices:
        division_resident_binding = a4.bind_a4_translation(
            division_ragged.to_torch(device=device),
            division_state.to_torch(device=device),
        )
        division_resident_tape = division_tape.to_torch(
            division_binding, division_dt, device=device,
        )
        division_back = a47.genome_hydrolysis_deletion_torch(
            division_resident_binding, division_dt,
            division_resident_tape,
        ).to_numpy()
        if a47.validate_a4_hydrolysis_deletion_plan(
                division_back, division_binding, division_dt,
                division_tape) is not division_back:
            raise AssertionError(
                '%s A4.7b ULP readback was not validated' % device
            )
        division_actual_state = division_back.state_dict()
        for name, expected_value in division_expected_state.items():
            if name == 'genome_lesion_mean_after':
                continue
            v3.assert_recursive_close(
                expected_value, division_actual_state[name],
                atol=0.0, rtol=0.0,
                path='a47b.division_nonmean.%s.%s' % (device, name),
            )
        actual_mean = float(division_back.genome_lesion_mean_after[0])
        expected_bits = int(np.asarray(
            division_mean, dtype=np.float64,
        ).view(np.uint64))
        actual_bits = int(np.asarray(
            actual_mean, dtype=np.float64,
        ).view(np.uint64))
        if abs(actual_bits - expected_bits) > 1:
            raise AssertionError(
                '%s A4.7b lesion mean exceeds one ULP: %s vs %s' % (
                    device, actual_mean.hex(), division_mean.hex(),
                )
            )

    forbidden = (
        '.item(', '.cpu(', '.numpy(', '.tolist(',
        'nonzero(', 'masked_select(', 'unique(',
    )
    source = inspect.getsource(a47.genome_hydrolysis_deletion_torch)
    hits = [token for token in forbidden if token in source]
    if hits:
        raise AssertionError('A4.7b Torch path contains host op: %s' % hits)
    if world.rng.bit_generator.state != rng_before:
        raise AssertionError('A4.7b apply advanced live RNG')
    for label, before, after in (
            ('world', world_before, world.state_dict()),
            ('cell', cell_before, cell.state_dict()),
            ('ragged', ragged_before, ragged.state_dict()),
            ('state', state_before, state.state_dict()),
            ('cache', cache_before, binding.cache.state_dict()),
            ('tape', tape_before, tape.state_dict())):
        v3.assert_recursive_close(
            before, after, atol=0.0, rtol=0.0,
            path='a47b.nonmutation.%s' % label,
        )
    return ('NumPy/Torch %s final deletion descriptor exact; all outputs '
            'resident and source/tape/world/live RNG pure' % '/'.join(devices))


def test_a47b_hydrolysis_capacity_trust_rollback_and_authority():
    if torch is None:
        raise AssertionError('PyTorch is required for A4.7b trust test')
    world, cell, capacity, dt = _a47b_hydrolysis_fixture()
    world_before = v3.pickle_clone(world.state_dict())
    cell_before = v3.pickle_clone(cell.state_dict())
    rng_before = copy.deepcopy(world.rng.bit_generator.state)
    ragged, state, binding = _paid_replication_binding(
        [cell], world.config, capacity,
    )
    ragged_before = ragged.state_dict()
    state_before = state.state_dict()
    cache_before = binding.cache.state_dict()
    tape = a47.prepare_genome_hydrolysis_rng_tape(
        binding, dt, rng_before,
    )
    tape_before = tape.state_dict()
    plan = a47.genome_hydrolysis_deletion_numpy(
        binding, dt, tape,
    )

    if (int(ragged.cell_capacity) != 1
            or int(ragged.sequence_capacity) != 5
            or int(ragged.symbol_capacity) != 199
            or int(ragged.max_sequence_symbols) != 48):
        raise AssertionError('A4.7b exact source capacity drifted')
    exact_values = {
        'max_cells': 1,
        'max_sequences': 5,
        'max_symbols': 199,
        'max_sequence_symbols': 48,
        'max_proteins_per_cell': 64,
    }
    for label, field in (
            ('sequence', 'max_sequences'),
            ('symbol', 'max_symbols'),
            ('row_width', 'max_sequence_symbols')):
        values = dict(exact_values)
        values[field] -= 1
        short = a4.GPU068A4Config(**values)
        _assert_raises(
            a4.A4CapacityError,
            lambda short=short: a4.FullFidelityA4GenomeAdapter(
                short,
            ).pack_cells([cell]),
        )

    corruptions = []
    bad = plan.clone()
    bad.final_symbols[0, 0] = (
        int(bad.final_symbols[0, 0]) + 1
    ) % int(a4.g2.ALPHABET_SIZE)
    corruptions.append(('used_symbol', bad))
    bad = plan.clone()
    tail = int(bad.final_lengths[0])
    if tail >= int(bad.final_symbols.shape[1]):
        raise AssertionError('A4.7b trust fixture lacks final-symbol tail')
    bad.final_symbols[0, tail] = 1
    corruptions.append(('symbol_tail', bad))
    bad = plan.clone()
    bad.final_lengths[0] -= 1
    corruptions.append(('length', bad))
    bad = plan.clone()
    bad.genome_lesions_after[0] = np.nextafter(
        bad.genome_lesions_after[0], np.float64(np.inf),
    )
    corruptions.append(('lesion', bad))
    bad = plan.clone()
    bad.pools_after[0, a4.a3.POOL_WASTE] = np.nextafter(
        bad.pools_after[0, a4.a3.POOL_WASTE], np.float64(np.inf),
    )
    corruptions.append(('pool', bad))
    for label, field in (
            ('symbol_count', 'symbol_count_after'),
            ('topology', 'topology_symbol_delta'),
            ('damage_event', 'genome_damage_event_delta'),
            ('material', 'genome_material_symbols_after'),
            ('refresh_count', 'gene_cache_refresh_count')):
        bad = plan.clone()
        getattr(bad, field)[0] += 1
        corruptions.append((label, bad))
    bad = plan.clone()
    bad.genome_lesion_mean_after[0] = np.nextafter(
        np.nextafter(
            bad.genome_lesion_mean_after[0], np.float64(np.inf),
        ),
        np.float64(np.inf),
    )
    corruptions.append(('lesion_mean', bad))
    bad = plan.clone()
    bad.gene_cache_dirty[0] = False
    corruptions.append(('cache_dirty', bad))
    for label, corrupt in corruptions:
        _assert_raises(
            a4.A4SchemaError,
            lambda corrupt=corrupt: (
                a47.validate_a4_hydrolysis_deletion_plan(
                    corrupt, binding, dt, tape,
                )
            ),
        )

    wrong_dt = float(np.nextafter(
        np.float64(dt), np.float64(np.inf),
    ))
    _assert_raises(
        a47.A4HydrolysisScopeError,
        lambda: a47.validate_a4_hydrolysis_deletion_plan(
            plan, binding, wrong_dt, tape,
        ),
    )
    stale_cell = copy.deepcopy(cell)
    stale_cell.genome_lesions[0] = np.nextafter(
        stale_cell.genome_lesions[0], np.float64(np.inf),
    )
    _, _, stale_binding = _paid_replication_binding(
        [stale_cell], world.config, capacity,
    )
    _assert_raises(
        a47.A4HydrolysisScopeError,
        lambda: a47.validate_a4_hydrolysis_deletion_plan(
            plan, stale_binding, dt, tape,
        ),
    )
    foreign_state = copy.deepcopy(
        np.random.default_rng(10147).bit_generator.state,
    )
    foreign_tape = a47.prepare_genome_hydrolysis_rng_tape(
        binding, dt, foreign_state,
    )
    _assert_raises(
        a47.A4HydrolysisScopeError,
        lambda: a47.validate_a4_hydrolysis_deletion_plan(
            plan, binding, dt, foreign_tape,
        ),
    )

    second = copy.deepcopy(cell)
    second.cell_id = int(cell.cell_id) + 1000000
    multi_capacity = a4.GPU068A4Config(
        max_cells=2, max_sequences=10, max_symbols=398,
        max_sequence_symbols=48, max_proteins_per_cell=64,
    )
    _, _, multi_binding = _paid_replication_binding(
        [cell, second], world.config, multi_capacity,
    )
    _assert_raises(
        a47.A4HydrolysisScopeError,
        lambda: a47.prepare_genome_hydrolysis_rng_tape(
            multi_binding, dt, rng_before,
        ),
    )
    _assert_raises(
        a47.A4HydrolysisScopeError,
        lambda: a47.validate_a4_hydrolysis_deletion_plan(
            plan, multi_binding, dt, tape,
        ),
    )

    source_lengths = np.diff(
        ragged.sequence_offsets[:int(ragged.sequence_count) + 1]
    ).astype(np.int64, copy=False)
    source_rows = np.zeros(
        (int(ragged.sequence_capacity), int(ragged.max_sequence_symbols)),
        dtype=np.uint8,
    )
    for sequence_index, length in enumerate(source_lengths):
        start = int(ragged.sequence_offsets[sequence_index])
        source_rows[sequence_index, :int(length)] = (
            ragged.symbols[start:start + int(length)]
        )
    rollback_expected = {
        'final_symbols': source_rows,
        'final_lengths': source_lengths,
        'genome_lesions_after': ragged.genome_lesions.copy(),
        'pools_after': state.pools[:1].copy(),
        'symbol_count_after': np.asarray(
            [ragged.symbol_count], dtype=np.int64,
        ),
        'topology_symbol_delta': np.zeros((1,), dtype=np.int64),
        'genome_damage_event_delta': np.zeros((1,), dtype=np.int64),
        'genome_material_symbols_after': (
            state.genome_material_symbols[:1].copy()
        ),
        'genome_lesion_mean_after': state.genome_lesion_mean[:1].copy(),
        'gene_cache_dirty': np.zeros((1,), dtype=bool),
        'gene_cache_refresh_count': np.zeros((1,), dtype=np.int64),
    }
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    elif _REQUIRE_CUDA:
        raise AssertionError('CUDA required but unavailable')
    for device in devices:
        for target in ('actual', 'expected'):
            resident_binding = a4.bind_a4_translation(
                ragged.to_torch(device=device),
                state.to_torch(device=device),
            )
            resident_tape = tape.to_torch(
                binding, dt, device=device,
            )
            if target == 'actual':
                tensor = resident_tape.uniform_draws
            else:
                tensor = resident_tape._resident_expected_arrays[
                    'uniform_draws'
                ]
            version = int(tensor._version)
            tensor.data[0] = torch.remainder(
                tensor.data[0] + 0.125, 1.0,
            )
            if int(tensor._version) != version:
                raise AssertionError(
                    '%s %s .data unexpectedly changed version' % (
                        device, target,
                    )
                )
            failed = a47.genome_hydrolysis_deletion_torch(
                resident_binding, dt, resident_tape,
            )
            scope_valid = failed.scope_valid.detach().cpu().numpy()
            error_code = failed.scope_error_code.detach().cpu().numpy()
            if np.any(scope_valid) or not np.all(error_code != 0):
                raise AssertionError(
                    '%s %s tape tamper did not scope-fail' % (
                        device, target,
                    )
                )
            for name, expected_values in rollback_expected.items():
                values = getattr(failed, name).detach().cpu().numpy()
                v3.assert_recursive_close(
                    expected_values, values, atol=0.0, rtol=0.0,
                    path='a47b.rollback.%s.%s.%s' % (
                        device, target, name,
                    ),
                )
            _assert_raises(a47.A4HydrolysisScopeError, failed.to_numpy)

        resident_binding = a4.bind_a4_translation(
            ragged.to_torch(device=device),
            state.to_torch(device=device),
        )
        resident_tape = tape.to_torch(binding, dt, device=device)
        resident_plan = a47.genome_hydrolysis_deletion_torch(
            resident_binding, dt, resident_tape,
        )
        version = int(resident_plan.final_symbols._version)
        resident_plan.final_symbols.data[0, 0] = torch.remainder(
            resident_plan.final_symbols.data[0, 0].to(torch.int64) + 1,
            int(a4.g2.ALPHABET_SIZE),
        ).to(resident_plan.final_symbols.dtype)
        if int(resident_plan.final_symbols._version) != version:
            raise AssertionError(
                '%s plan .data unexpectedly changed version' % device
            )
        changed_readback = resident_plan.to_numpy()
        _assert_raises(
            a4.A4SchemaError,
            lambda changed_readback=changed_readback: (
                a47.validate_a4_hydrolysis_deletion_plan(
                    changed_readback, binding, dt, tape,
                )
            ),
        )

    if (world.rng.bit_generator.state != rng_before
            or a47.FULL_GPU_WORLD_STEP is not False
            or a47.PORT_STATUS.get('full_gpu_world_step') is not False
            or a4.a3.PORT_STATUS.get('genome_symbol_hydrolysis_rng')
            != 'cpu-authoritative-explicit-hazard-plan'
            or v3._event_order().count('genome_hydrolysis_cpu_rng') != 1):
        raise AssertionError('A4.7b changed A3 hydrolysis authority')
    for label, before, after in (
            ('world', world_before, world.state_dict()),
            ('cell', cell_before, cell.state_dict()),
            ('ragged', ragged_before, ragged.state_dict()),
            ('state', state_before, state.state_dict()),
            ('cache', cache_before, binding.cache.state_dict()),
            ('tape', tape_before, tape.state_dict())):
        v3.assert_recursive_close(
            before, after, atol=0.0, rtol=0.0,
            path='a47b.trust_nonmutation.%s' % label,
        )
    return ('exact Q/S/W and each one-short; %d descriptor fields/'
            'dt/source/tape/multicell trust rejection; CPU/CUDA public/private '
            'tape rollback plus binding-aware plan .data rejection; '
            'A3 authority retained' %
            len(corruptions))


def test_a48a_atomic_commit_formal066_hit_nohit_dt0():
    cases = []
    world, cell, capacity, dt = _a47b_hydrolysis_fixture()
    cases.append(('hit_miss_hit', world, cell, capacity, dt, 3, 2))

    world, cell, capacity, _ = _a47b_hydrolysis_fixture(zero_hit=True)
    cell.genome_lesions = [0.751, 1.0, 0.751]
    cell._refresh_gene_cache()
    cases.append(('no_hit', world, cell, capacity, 1.0, 2, 0))

    world, cell, capacity, dt = _a47b_hydrolysis_fixture(zero_hit=True)
    cases.append(('eligible_dt0', world, cell, capacity, dt, 1, 0))

    summaries = []
    for (label, world, cell, capacity, dt,
         expected_draws, expected_hits) in cases:
        source = _a48_binding(world, cell, capacity)
        rng_before = copy.deepcopy(world.rng.bit_generator.state)
        tape = a47.prepare_genome_hydrolysis_rng_tape(
            source, dt, rng_before,
        )
        expected_world, expected_cell, expected = _a47b_formal066_oracle(
            world, cell, dt,
        )
        if (int(tape.draw_count) != expected_draws
                or int(tape.hit_count) != expected_hits):
            raise AssertionError(label + ' fixture draw/hit schedule drifted')

        scheduler = a48.A4HydrolysisEventScheduler(capacity, 'cpu')
        _a48_begin_direct(scheduler, world, cell, dt)
        try:
            changed = scheduler.cpu_genome_hydrolysis(
                world, cell, (), dt,
            )
            entry = _a48_hydrolysis_entry(scheduler, cell)
            fresh = _a48_binding(world, cell, capacity)
            a4._require_translation_binding(fresh)
            sequence_count = len(expected['final_lengths'])
            if (int(fresh.ragged.sequence_count) != sequence_count
                    or int(fresh.ragged.symbol_count)
                    != int(expected['final_symbol_count'])
                    or not np.array_equal(
                        fresh.ragged.sequence_offsets[:sequence_count + 1],
                        expected['final_offsets'],
                    )):
                raise AssertionError(label + ' compact offsets differ')
            _a48_assert_committed_oracle(
                world, cell, expected_world, expected_cell,
                'a48a.' + label,
            )
            metadata = entry['metadata']
            expected_metadata = {
                'authority': (
                    'A4.8a-resident-plan-atomic-cpu-rng-commit'
                ),
                'enabled': True,
                'device': 'cpu',
                'legacy_hazard_count': 0,
                'eligible_genome_count': expected_draws,
                'draw_count': expected_draws,
                'hit_count': expected_hits,
                'source_provenance': source.state.source_provenance,
                'final_provenance': fresh.state.source_provenance,
                'mutation_count': expected_hits,
                'rng_draw_count': expected_draws + expected_hits,
                'gene_cache_refresh_count': expected_hits,
                'work_performed': expected_hits > 0,
            }
            if metadata != expected_metadata:
                raise AssertionError(label + ' receipt metadata differs')
            if entry['status'] != 'executed' or bool(changed) != (expected_hits > 0):
                raise AssertionError(label + ' executed/change status differs')
            if expected_hits:
                if source.state.source_provenance == fresh.state.source_provenance:
                    raise AssertionError(label + ' old binding remained current')
            elif source.state.source_provenance != fresh.state.source_provenance:
                raise AssertionError(label + ' no-hit source provenance changed')
            summaries.append('%s=%d/%d' % (
                label, expected_draws, expected_hits,
            ))
        finally:
            _a48_abort_direct(scheduler, cell)
    return ('Formal066 exact compact/lesion/waste/cache/counter/full-PCG64; '
            + ', '.join(summaries))


def test_a48a_candidate_cpu_cuda_purity_fresh_binding_and_one_shot():
    if torch is None:
        raise AssertionError('PyTorch unavailable')
    if _REQUIRE_CUDA and not torch.cuda.is_available():
        raise AssertionError('CUDA required but unavailable')
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    reference = None
    summaries = []
    for device in devices:
        world, cell, capacity, dt = _a47b_hydrolysis_fixture()
        scheduler = a48.A4HydrolysisEventScheduler(capacity, device)
        _a48_begin_direct(scheduler, world, cell, dt)
        before_cell = v3.pickle_clone(cell.state_dict())
        before_rng = copy.deepcopy(world.rng.bit_generator.state)
        try:
            candidate = scheduler._prepare_candidate(world, cell, dt)
            v3.assert_recursive_close(
                before_cell, cell.state_dict(), atol=0.0, rtol=0.0,
                path='a48a.%s.prepare_cell_purity' % device,
            )
            if world.rng.bit_generator.state != before_rng:
                raise AssertionError(device + ' candidate advanced live RNG')
            for owner in (
                    candidate.resident_binding.ragged,
                    candidate.resident_binding.state,
                    candidate.resident_binding.cache,
                    candidate.resident_tape, candidate.resident_plan):
                pointers = owner.data_ptrs()
                if (not pointers
                        or any(getattr(owner, name).device.type != device
                               for name in pointers)):
                    raise AssertionError(device + ' resident device differs')
            a4._require_translation_binding(candidate.fresh_binding)
            if (candidate.source_binding.state.source_provenance
                    == candidate.fresh_binding.state.source_provenance):
                raise AssertionError(device + ' hit candidate is not fresh')

            if reference is None:
                reference = (
                    candidate.tape.clone(), candidate.plan.clone(),
                    v3.pickle_clone(candidate.candidate_cell.state_dict()),
                )
            else:
                v3.assert_recursive_close(
                    reference[0].state_dict(), candidate.tape.state_dict(),
                    atol=0.0, rtol=0.0,
                    path='a48a.%s.tape_parity' % device,
                )
                _a48_assert_plan_equal(
                    reference[1], candidate.plan,
                    'a48a.%s.plan_parity' % device,
                )
                v3.assert_recursive_close(
                    reference[2], candidate.candidate_cell.state_dict(),
                    atol=0.0, rtol=0.0,
                    path='a48a.%s.candidate_cell_parity' % device,
                )

            scheduler._commit_candidate(
                world, cell, dt, (), candidate,
            )
            final_binding = _a48_binding(world, cell, capacity)
            if (a48._binding_identity(final_binding)
                    != a48._binding_identity(candidate.fresh_binding)):
                raise AssertionError(device + ' committed binding is not fresh')
            committed_cell = v3.pickle_clone(cell.state_dict())
            committed_rng = copy.deepcopy(world.rng.bit_generator.state)
            _assert_raises(
                a48.A4HydrolysisCommitError,
                lambda: scheduler._commit_candidate(
                    world, cell, dt, (), candidate,
                ),
            )
            v3.assert_recursive_close(
                committed_cell, cell.state_dict(), atol=0.0, rtol=0.0,
                path='a48a.%s.one_shot_cell' % device,
            )
            if world.rng.bit_generator.state != committed_rng:
                raise AssertionError(device + ' one-shot retry changed RNG')
            summaries.append(device)
        finally:
            _a48_abort_direct(scheduler, cell)
    return ('/'.join(summaries) +
            ' candidate exact parity, prepare purity, resident device, '
            'fresh binding and one-shot rejection')


def test_a48a_fail_closed_capacity_trust_order_and_publish_rollback():
    one_short = (
        ('Q', dict(max_sequences=4, max_symbols=199,
                   max_sequence_symbols=48)),
        ('S', dict(max_sequences=5, max_symbols=198,
                   max_sequence_symbols=48)),
        ('W', dict(max_sequences=5, max_symbols=199,
                   max_sequence_symbols=47)),
    )
    for label, values in one_short:
        world, cell, _, dt = _a47b_hydrolysis_fixture()
        capacity = a4.GPU068A4Config(
            max_cells=1, max_proteins_per_cell=64, **values
        )
        scheduler = a48.A4HydrolysisEventScheduler(capacity, 'cpu')
        _a48_begin_direct(scheduler, world, cell, dt)
        before_cell = v3.pickle_clone(cell.state_dict())
        before_rng = copy.deepcopy(world.rng.bit_generator.state)
        try:
            _assert_raises(
                a4.A4CapacityError,
                lambda: scheduler.cpu_genome_hydrolysis(
                    world, cell, (), dt,
                ),
            )
            v3.assert_recursive_close(
                before_cell, cell.state_dict(), 0.0, 0.0,
                'a48a.capacity_' + label,
            )
            if (world.rng.bit_generator.state != before_rng
                    or any(entry['event'] == 'genome_hydrolysis_cpu_rng'
                           for entry in scheduler._record(cell)[1]['events'])):
                raise AssertionError(label + ' preclaim failure was not atomic')
        finally:
            _a48_abort_direct(scheduler, cell)

    class TamperResidentPlan(a48.A4HydrolysisEventScheduler):
        def _candidate_ready(self, world, cell, dt, candidate):
            candidate.resident_plan.final_symbols.data[0, 0].bitwise_xor_(1)
            return candidate

    world, cell, capacity, dt = _a47b_hydrolysis_fixture()
    scheduler = TamperResidentPlan(capacity, 'cpu')
    _a48_begin_direct(scheduler, world, cell, dt)
    before_cell = v3.pickle_clone(cell.state_dict())
    before_rng = copy.deepcopy(world.rng.bit_generator.state)
    try:
        _assert_raises(
            (a48.A4HydrolysisCommitError, a4.A4SchemaError),
            lambda: scheduler.cpu_genome_hydrolysis(
                world, cell, (), dt,
            ),
        )
        v3.assert_recursive_close(
            before_cell, cell.state_dict(), 0.0, 0.0,
            'a48a.resident_data_tamper',
        )
        if (world.rng.bit_generator.state != before_rng
                or any(entry['event'] == 'genome_hydrolysis_cpu_rng'
                       for entry in scheduler._record(cell)[1]['events'])):
            raise AssertionError('resident .data tamper crossed claim boundary')
    finally:
        _a48_abort_direct(scheduler, cell)

    world, cell, capacity, dt = _a47b_hydrolysis_fixture()
    scheduler = a48.A4HydrolysisEventScheduler(capacity, 'cpu')
    _a48_begin_direct(scheduler, world, cell, dt)
    before_cell = v3.pickle_clone(cell.state_dict())
    before_rng = copy.deepcopy(world.rng.bit_generator.state)
    _assert_raises(
        a3s.A3SchedulerProtocolError,
        lambda: scheduler.cpu_genome_hydrolysis(
            world, cell, (), np.nextafter(dt, np.inf),
        ),
    )
    foreign_cell = copy.deepcopy(cell)
    _assert_raises(
        a3s.A3SchedulerProtocolError,
        lambda: scheduler.cpu_genome_hydrolysis(
            world, foreign_cell, (), dt,
        ),
    )
    v3.assert_recursive_close(
        before_cell, cell.state_dict(), 0.0, 0.0,
        'a48a.wrong_cell_dt',
    )
    if world.rng.bit_generator.state != before_rng:
        raise AssertionError('wrong cell/dt changed RNG')
    _a48_abort_direct(scheduler, cell)

    class TamperResidentTape(a48.A4HydrolysisEventScheduler):
        def _candidate_ready(self, world, cell, dt, candidate):
            candidate.resident_tape.uniform_draws.data[0].add_(0.125)
            return candidate

    class TamperHostSource(a48.A4HydrolysisEventScheduler):
        def _candidate_ready(self, world, cell, dt, candidate):
            candidate.source_binding.ragged.symbols[0] ^= np.uint8(1)
            return candidate

    class TamperSchedulerSettings(a48.A4HydrolysisEventScheduler):
        def _candidate_ready(self, world, cell, dt, candidate):
            self.a4_config = a4.GPU068A4Config(
                max_cells=1, max_sequences=4, max_symbols=199,
                max_sequence_symbols=48, max_proteins_per_cell=64,
            )
            self.a4_device = 'cuda'
            return candidate

    class TamperCandidateBeforeState(a48.A4HydrolysisEventScheduler):
        def _candidate_ready(self, world, cell, dt, candidate):
            bit_generator = np.random.PCG64()
            bit_generator.state = copy.deepcopy(candidate.rng_before_state)
            generator = np.random.Generator(bit_generator)
            generator.random()
            candidate.rng_before_state = copy.deepcopy(
                generator.bit_generator.state
            )
            return candidate

    for scheduler_type, label in (
            (TamperResidentTape, 'resident_tape_data'),
            (TamperHostSource, 'host_source'),
            (TamperSchedulerSettings, 'scheduler_config_device'),
            (TamperCandidateBeforeState, 'candidate_rng_before')):
        world, cell, capacity, dt = _a47b_hydrolysis_fixture()
        scheduler = scheduler_type(capacity, 'cpu')
        _a48_begin_direct(scheduler, world, cell, dt)
        before_cell = v3.pickle_clone(cell.state_dict())
        before_rng = copy.deepcopy(world.rng.bit_generator.state)
        try:
            _assert_raises(
                (a48.A4HydrolysisCommitError, a4.A4SchemaError),
                lambda: scheduler.cpu_genome_hydrolysis(
                    world, cell, (), dt,
                ),
            )
            v3.assert_recursive_close(
                before_cell, cell.state_dict(), 0.0, 0.0,
                'a48a.' + label,
            )
            if (world.rng.bit_generator.state != before_rng
                    or any(entry['event'] == 'genome_hydrolysis_cpu_rng'
                           for entry in scheduler._record(cell)[1]['events'])):
                raise AssertionError(label + ' crossed claim boundary')
        finally:
            _a48_abort_direct(scheduler, cell)

    class TamperCandidateAfterState(a48.A4HydrolysisEventScheduler):
        def _candidate_ready(self, world, cell, dt, candidate):
            bit_generator = np.random.PCG64()
            bit_generator.state = copy.deepcopy(candidate.rng_after_state)
            generator = np.random.Generator(bit_generator)
            generator.random()
            candidate.rng_after_state = copy.deepcopy(
                generator.bit_generator.state
            )
            return candidate

    world, cell, capacity, dt = _a47b_hydrolysis_fixture()
    scheduler = TamperCandidateAfterState(capacity, 'cpu')
    _a48_begin_direct(scheduler, world, cell, dt)
    before_cell = v3.pickle_clone(cell.state_dict())
    before_rng = copy.deepcopy(world.rng.bit_generator.state)
    try:
        _assert_raises(
            a48.A4HydrolysisCommitError,
            lambda: scheduler.cpu_genome_hydrolysis(
                world, cell, (), dt,
            ),
        )
        v3.assert_recursive_close(
            before_cell, cell.state_dict(), 0.0, 0.0,
            'a48a.candidate_rng_after_tamper',
        )
        if (world.rng.bit_generator.state != before_rng
                or any(entry['event'] == 'genome_hydrolysis_cpu_rng'
                       for entry in scheduler._record(cell)[1]['events'])):
            raise AssertionError(
                'candidate RNG after-state tamper crossed claim boundary'
            )
    finally:
        _a48_abort_direct(scheduler, cell)

    class FailAfterPublish(a48.A4HydrolysisEventScheduler):
        def _publish_candidate(self, world, cell, candidate):
            super(FailAfterPublish, self)._publish_candidate(
                world, cell, candidate,
            )
            cell.genomes[0][0] ^= np.uint8(1)
            raise RuntimeError('injected A4.8a publish failure')

    world, cell, capacity, dt = _a47b_hydrolysis_fixture()
    scheduler = FailAfterPublish(capacity, 'cpu')
    _a48_begin_direct(scheduler, world, cell, dt)
    before_cell = v3.pickle_clone(cell.state_dict())
    before_rng = copy.deepcopy(world.rng.bit_generator.state)
    object_ids = (
        id(cell.genomes), id(cell.genome_lesions), id(cell.pools),
        id(cell.gene_specs), id(world.rng),
    )
    caught = None
    try:
        scheduler.cpu_genome_hydrolysis(world, cell, (), dt)
    except RuntimeError as error:
        caught = error
    if caught is None:
        raise AssertionError('injected publish failure did not escape')
    v3.assert_recursive_close(
        before_cell, cell.state_dict(), 0.0, 0.0,
        'a48a.publish_rollback_cell',
    )
    if (world.rng.bit_generator.state != before_rng
            or object_ids != (
                id(cell.genomes), id(cell.genome_lesions), id(cell.pools),
                id(cell.gene_specs), id(world.rng),
            )):
        raise AssertionError('publish rollback did not restore identity/state')
    entry = _a48_hydrolysis_entry(scheduler, cell)
    receipt = _a48_abort_direct(scheduler, cell, caught)
    if (receipt['status'] != 'aborted'
            or entry['event'] != 'genome_hydrolysis_cpu_rng'):
        raise AssertionError('postclaim failure did not leave aborted receipt')

    world, cell, capacity, dt = _a47b_hydrolysis_fixture()
    scheduler = a48.A4HydrolysisEventScheduler(capacity, 'cpu')
    _a48_begin_direct(scheduler, world, cell, dt)
    scheduler.cpu_genome_hydrolysis(world, cell, (), dt)
    committed = v3.pickle_clone(cell.state_dict())
    committed_rng = copy.deepcopy(world.rng.bit_generator.state)
    _assert_raises(
        a3s.A3DuplicateEventError,
        lambda: scheduler.cpu_genome_hydrolysis(world, cell, (), dt),
    )
    v3.assert_recursive_close(
        committed, cell.state_dict(), 0.0, 0.0,
        'a48a.duplicate_atomic',
    )
    if world.rng.bit_generator.state != committed_rng:
        raise AssertionError('duplicate attempt changed RNG')
    _a48_abort_direct(scheduler, cell)

    world, cell, capacity, dt = _a47b_hydrolysis_fixture()
    scheduler = a48.A4HydrolysisEventScheduler(capacity, 'cpu')
    scheduler.begin_step(world, [cell])
    for event in a3s.WORLD_EVENT_ORDER[:5]:
        scheduler.claim_world(event, status='skipped')
    scheduler.claim(cell, 'surface_exchange', status='skipped')
    scheduler.reserve_metabolism_dispatch(world, cell, dt)
    scheduler.begin_a3_metabolism(world, cell, dt)
    before_cell = v3.pickle_clone(cell.state_dict())
    before_rng = copy.deepcopy(world.rng.bit_generator.state)
    _assert_raises(
        a3s.A3EventOrderError,
        lambda: scheduler.cpu_genome_hydrolysis(world, cell, (), dt),
    )
    v3.assert_recursive_close(
        before_cell, cell.state_dict(), 0.0, 0.0,
        'a48a.out_of_order_atomic',
    )
    if world.rng.bit_generator.state != before_rng:
        raise AssertionError('out-of-order attempt changed RNG')
    _a48_abort_direct(scheduler, cell)
    return ('Q/S/W one-short; wrong cell/dt/source/config/device/PCG64; '
            'resident tape+plan .data preclaim atomic; postclaim object/'
            'state/RNG rollback + aborted receipt; duplicate/out-of-order '
            'exact-once')


def test_a48a_world_interleave_lockstep_save_clone_and_a3_authority():
    capacity = a4.GPU068A4Config()
    for steps, seed in ((1, 9881), (10, 9882)):
        source = v3.make_world(seed=seed, cells=2)
        state = v3.pickle_clone(source.state_dict())
        cpu = a4.a3.s66.Formal066World.from_state(v3.pickle_clone(state))
        hybrid = _a48_hybrid_from_state(state, capacity, 'cpu')
        for _ in range(steps):
            cpu.step(0.1)
            hybrid.step(0.1)
        v3._assert_world_pair(
            cpu, hybrid, state_atol=v3.WORLD_FP64_ATOL,
            ledger_atol=v3.LEDGER_ATOL,
            label='a48a.lockstep_%d' % steps,
        )

    stressed = v3.make_world(seed=9886, cells=2)
    for cell in stressed.cells:
        cell.current_stress = 1.5
        cell.damage_trace[:] = 0.9
        cell.membrane_oxidation[:] = np.linspace(
            0.3, 1.8, len(cell.membrane_oxidation),
        )
        cell.pools[a4.a3.POOL_REACTIVE] += 0.16
        cell.pools[a4.a3.POOL_ATP] += 0.4
        cell._sync_damage_pool()
    state = v3.pickle_clone(stressed.state_dict())
    cpu = a4.a3.s66.Formal066World.from_state(v3.pickle_clone(state))
    hybrid = _a48_hybrid_from_state(state, capacity, 'cpu')
    for _ in range(3):
        cpu.step(0.1)
        hybrid.step(0.1)
    v3._assert_world_pair(
        cpu, hybrid, state_atol=v3.WORLD_FP64_ATOL,
        ledger_atol=v3.LEDGER_ATOL,
        label='a48a.repair_heavy_stress_3',
    )

    source = v3.make_world(seed=9883, cells=2)
    source.config.endogenous_damage = False
    for cell in source.cells:
        cell.genome_lesions = [
            0.5 / 0.00065 for _ in cell.genomes
        ]
        cell._refresh_gene_cache()
    state = v3.pickle_clone(source.state_dict())
    cpu = a4.a3.s66.Formal066World.from_state(v3.pickle_clone(state))
    hybrid = _a48_hybrid_from_state(state, capacity, 'cpu')
    cpu.step(0.1)
    original_bridge = a3s.A3EventScheduler.cpu_genome_hydrolysis

    def legacy_bridge_bomb(*args, **kwargs):
        raise AssertionError('legacy A3 hydrolysis bridge was called')

    a3s.A3EventScheduler.cpu_genome_hydrolysis = legacy_bridge_bomb
    try:
        receipt = hybrid.step(0.1)
    finally:
        a3s.A3EventScheduler.cpu_genome_hydrolysis = original_bridge
    v3._assert_world_pair(
        cpu, hybrid, state_atol=v3.WORLD_FP64_ATOL,
        ledger_atol=v3.LEDGER_ATOL,
        label='a48a.two_cell_interleave',
    )
    previous_motion = None
    for record in receipt['cells']:
        by_name = {entry['event']: entry for entry in record['events']}
        names = [entry['event'] for entry in record['events']]
        if names.count('genome_hydrolysis_cpu_rng') != 1:
            raise AssertionError('A4.8a integrated hydrolysis count differs')
        replication = by_name['replication_cpu']['ordinal']
        hydrolysis = by_name['genome_hydrolysis_cpu_rng']['ordinal']
        motion = by_name['motion']['ordinal']
        if not replication < hydrolysis < motion:
            raise AssertionError('per-cell RNG event order differs')
        if previous_motion is not None and previous_motion >= replication:
            raise AssertionError('two-cell RNG interleave was batched/reordered')
        previous_motion = motion
        if by_name['genome_hydrolysis_cpu_rng']['metadata'][
                'eligible_genome_count'] < 1:
            raise AssertionError('two-cell hydrolysis fixture drew no RNG')

    hybrid = _a48_hybrid_from_state(
        v3.make_world(seed=9884, cells=2).state_dict(), capacity, 'cpu',
    )
    for _ in range(2):
        hybrid.step(0.1)
    twin = hybrid.clone()
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, 'a48a.pkl')
        hybrid.save(path)
        restored = a48.Hybrid066WorldA4Hydrolysis.load(path)
        for value in (twin, restored):
            if (type(value) is not a48.Hybrid066WorldA4Hydrolysis
                    or type(value.scheduler)
                    is not a48.A4HydrolysisEventScheduler
                    or value.scheduler.a4_device != 'cpu'
                    or value.scheduler.a4_config != capacity):
                raise AssertionError('A4.8a save/clone authority was lost')
        for _ in range(3):
            hybrid.step(0.1)
            twin.step(0.1)
            restored.step(0.1)
        v3.assert_recursive_close(
            hybrid.world.state_dict(), twin.world.state_dict(),
            0.0, 0.0, 'a48a.clone_continuation',
        )
        v3.assert_recursive_close(
            hybrid.world.state_dict(), restored.world.state_dict(),
            0.0, 0.0, 'a48a.save_continuation',
        )

    active = _a48_hybrid_from_state(
        v3.make_world(seed=9885, cells=1).state_dict(), capacity, 'cpu',
    )
    active.scheduler.begin_step(active.world, active.world.cells)
    _assert_raises(a3s.A3SchedulerProtocolError, active.clone)
    active.scheduler.abort_step(RuntimeError('expected active-save rejection'))
    active.scheduler._a4_hydrolysis_commit_active = True
    try:
        _assert_raises(a3s.A3SchedulerProtocolError, active.state_dict)
    finally:
        active.scheduler._a4_hydrolysis_commit_active = False

    if (a4.a3.PORT_STATUS.get('genome_symbol_hydrolysis_rng')
            != 'cpu-authoritative-explicit-hazard-plan'
            or a48.FULL_GPU_WORLD_STEP is not False):
        raise AssertionError('A4.8a changed promoted A3/full-GPU authority')
    for relative, expected in PROMOTED_A3_SHA256.items():
        path = os.path.join(ROOT, *relative.split('/'))
        if _sha256(path) != expected:
            raise AssertionError(relative + ' changed from promoted A3 bytes')
    return ('1/10-step + repair-heavy 3-step + 2-cell hydrolysis-stress '
            'replication/hydrolysis/motion PCG64 lockstep; '
            'legacy bridge bomb; save/load/clone continuation; active reject; '
            'A3 baseline authority unchanged')


def test_a48b_translation_atomic_commit_formal066_oracle_and_gates():
    source_world, source_cells, capacity, source_dt = _paid_translation_fixture(
        seed=9981,
    )
    source_state = v3.pickle_clone(source_world.state_dict())
    cases = []
    for label, index in (
            ('rich', 0), ('atp_reserve', 1),
            ('mineral_exhaustion', 2), ('no_translator', 3)):
        world = a4.a3.s66.Formal066World.from_state(
            v3.pickle_clone(source_state),
        )
        cases.append((label, world, world.cells[index], source_dt))

    disabled = a4.a3.s66.Formal066World.from_state(
        v3.pickle_clone(source_state),
    )
    disabled.config.gene_expression = False
    cases.append(('disabled', disabled, disabled.cells[0], source_dt))

    no_genome = a4.a3.s66.Formal066World.from_state(
        v3.pickle_clone(source_state),
    )
    no_genome_cell = no_genome.cells[0]
    no_genome_cell.genomes = []
    no_genome_cell.genome_lesions = []
    no_genome_cell.replication_template = None
    no_genome_cell.replication_copy = []
    no_genome_cell.replication_template_lesion = 0.0
    no_genome_cell.replication_fractional = 0.0
    no_genome_cell._refresh_gene_cache()
    cases.append(('no_genome', no_genome, no_genome_cell, source_dt))

    zero_dt = a4.a3.s66.Formal066World.from_state(
        v3.pickle_clone(source_state),
    )
    cases.append(('eligible_dt0', zero_dt, zero_dt.cells[0], 0.0))

    residual = a4.a3.s66.Formal066World.from_state(
        v3.pickle_clone(source_state),
    )
    residual.cells[0].pools[a4.a3.POOL_FUEL] = 9.936293309191388e-08
    cases.append(('signed_residual', residual, residual.cells[0], 0.001))

    details = []
    for label, world, cell, dt in cases:
        expected = copy.deepcopy(cell)
        expected_pool_id = id(expected.pools)
        expected_active_id = id(expected.proteins)
        expected_damaged_id = id(expected.damaged_proteins)
        expected.translate(dt, world.config)
        expected_sync = (
            id(expected.proteins) != expected_active_id
            and id(expected.damaged_proteins) != expected_damaged_id
        )
        if id(expected.pools) != expected_pool_id:
            raise AssertionError(label + ' CPU oracle replaced pools')

        scheduler = a48b.A4TranslationEventScheduler(capacity, 'cpu')
        _a48b_begin_translation_direct(scheduler, world, cell, dt)
        source = _a48_binding(world, cell, capacity)
        before_rng_object = world.rng
        before_rng = copy.deepcopy(world.rng.bit_generator.state)
        pool_id = id(cell.pools)
        active_id = id(cell.proteins)
        damaged_id = id(cell.damaged_proteins)
        try:
            result = scheduler.cpu_translation(
                world, cell, dt, world.config,
            )
            if result is not None:
                raise AssertionError(label + ' translation return differs')
            _a48b_assert_cell_matches(
                expected, cell, 'a48b.formal066.' + label,
            )
            if id(cell.pools) != pool_id:
                raise AssertionError(label + ' replaced pools ndarray')
            actual_sync = (
                id(cell.proteins) != active_id
                and id(cell.damaged_proteins) != damaged_id
            )
            if actual_sync != expected_sync:
                raise AssertionError(label + ' dictionary identity branch differs')
            if (world.rng is not before_rng_object
                    or world.rng.bit_generator.state != before_rng):
                raise AssertionError(label + ' translation changed PCG64')

            final = _a48_binding(world, cell, capacity)
            entry = _a48b_translation_entry(scheduler, cell)
            metadata = entry['metadata']
            expected_metadata = {
                'authority': (
                    'A4.8b-resident-plan-atomic-cpu-cell-commit'
                ),
                'enabled': bool(world.config.gene_expression),
                'device': 'cpu',
                'genome_count': len(cell.genomes),
                'source_provenance': a4._translation_state_provenance(
                    source.state,
                ),
                'final_provenance': a4._translation_state_provenance(
                    final.state,
                ),
                'sync_performed': expected_sync,
                'amount': float(cell.last_translation),
                'work_performed': float(cell.last_translation) > 0.0,
                'rng_draw_count': 0,
            }
            if entry['status'] != 'executed' or metadata != expected_metadata:
                raise AssertionError(label + ' receipt metadata differs')
            details.append('%s:%s' % (
                label, 'sync' if expected_sync else 'early',
            ))
        finally:
            _a48_abort_direct(scheduler, cell)

    residual_pool = cases[-1][2].pools[a4.a3.POOL_FUEL]
    if not (-a4.TRANSLATION_LEDGER_ATOL <= residual_pool < 0.0):
        raise AssertionError('integrated signed payment residual was clipped')
    return ('8 direct Formal066 cases; discrete/order/genotype/PCG64 exact, '
            'fp64 <=2e-12, pools identity and early/sync dict identity; ' +
            ', '.join(details))


def test_a48b_translation_candidate_cpu_cuda_commit_purity_and_fresh_binding():
    if torch is None:
        raise AssertionError('PyTorch unavailable')
    if _REQUIRE_CUDA and not torch.cuda.is_available():
        raise AssertionError('CUDA required but unavailable')
    public_names = set(a48b.__all__)
    signature = inspect.signature(a48b.A4TranslationEventScheduler.cpu_translation)
    if (a48b.BUILD != 'SOMA-CELL 0.6.8-GPU A4.8b'
            or a48b.SCHEMA_VERSION
            != '0.6.8-GPU-A4.8b-translation-atomic-commit'
            or a48b.FULL_GPU_WORLD_STEP is not False
            or '_A4TranslationCommitCandidate' in public_names
            or tuple(signature.parameters)
            != ('self', 'world', 'cell', 'dt', 'config')
            or not issubclass(
                a48b.A4TranslationEventScheduler,
                a48.A4HydrolysisEventScheduler,
            )):
        raise AssertionError('A4.8b identity/public authority differs')
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    source_world, _, capacity, dt = _paid_translation_fixture(seed=9982)
    source_state = v3.pickle_clone(source_world.state_dict())
    reference_plan = None
    reference_cell = None
    details = []
    for device in devices:
        world = a4.a3.s66.Formal066World.from_state(
            v3.pickle_clone(source_state),
        )
        cell = world.cells[0]
        expected = copy.deepcopy(cell)
        expected.translate(dt, world.config)
        scheduler = a48b.A4TranslationEventScheduler(capacity, device)
        _a48b_begin_translation_direct(scheduler, world, cell, dt)
        before_cell = v3.pickle_clone(cell.state_dict())
        before_rng_object = world.rng
        before_rng = copy.deepcopy(world.rng.bit_generator.state)
        pool_id = id(cell.pools)
        active_id = id(cell.proteins)
        damaged_id = id(cell.damaged_proteins)
        try:
            candidate = scheduler._prepare_translation_candidate(
                world, cell, dt, world.config,
            )
            v3.assert_recursive_close(
                before_cell, cell.state_dict(), 0.0, 0.0,
                'a48b.%s.prepare_purity' % device,
            )
            if (world.rng is not before_rng_object
                    or world.rng.bit_generator.state != before_rng):
                raise AssertionError(device + ' prepare changed PCG64')
            for owner in (
                    candidate.resident_binding.ragged,
                    candidate.resident_binding.state,
                    candidate.resident_binding.cache,
                    candidate.resident_plan):
                pointers = owner.data_ptrs()
                if (not pointers
                        or any(getattr(owner, name).device.type != device
                               for name in pointers)):
                    raise AssertionError(device + ' resident device differs')
            _a48b_assert_cell_matches(
                expected, candidate.candidate_cell,
                'a48b.%s.candidate' % device,
            )
            _assert_translation_matches_cells(
                candidate.plan, [expected],
                'a48b.%s.resident_authority' % device,
            )
            v3.assert_recursive_close(
                candidate.host_replay.state_dict(),
                candidate.plan.state_dict(),
                atol=a48b.TRANSLATION_ORACLE_ATOL, rtol=0.0,
                path='a48b.%s.independent_replay' % device,
            )
            source_identity = a48._binding_identity(
                candidate.source_binding,
            )
            fresh_identity = a48._binding_identity(candidate.fresh_binding)
            if (source_identity[0] != fresh_identity[0]
                    or source_identity[2] != fresh_identity[2]
                    or source_identity[1] == fresh_identity[1]):
                raise AssertionError(
                    device + ' fresh physiology/genome/cache provenance differs'
                )
            if not candidate.sync_performed:
                raise AssertionError(device + ' rich fixture missed sync gate')

            if reference_plan is None:
                reference_plan = candidate.plan.clone()
                reference_cell = v3.pickle_clone(
                    candidate.candidate_cell.state_dict(),
                )
            else:
                v3.assert_recursive_close(
                    reference_plan.state_dict(), candidate.plan.state_dict(),
                    atol=a48b.TRANSLATION_ORACLE_ATOL, rtol=0.0,
                    path='a48b.%s.device_plan' % device,
                )
                v3.assert_recursive_close(
                    reference_cell, candidate.candidate_cell.state_dict(),
                    atol=a48b.TRANSLATION_ORACLE_ATOL, rtol=0.0,
                    path='a48b.%s.device_candidate' % device,
                )

            scheduler._commit_translation_candidate(
                world, cell, dt, world.config, candidate,
            )
            _a48b_assert_cell_matches(
                expected, cell, 'a48b.%s.commit' % device,
            )
            if (id(cell.pools) != pool_id
                    or id(cell.proteins) == active_id
                    or id(cell.damaged_proteins) == damaged_id):
                raise AssertionError(device + ' success object identity differs')
            if (world.rng is not before_rng_object
                    or world.rng.bit_generator.state != before_rng):
                raise AssertionError(device + ' commit changed PCG64')
            committed = v3.pickle_clone(cell.state_dict())
            committed_rng = copy.deepcopy(world.rng.bit_generator.state)
            _assert_raises(
                a48b.A4TranslationCommitError,
                lambda: scheduler._commit_translation_candidate(
                    world, cell, dt, world.config, candidate,
                ),
            )
            v3.assert_recursive_close(
                committed, cell.state_dict(), 0.0, 0.0,
                'a48b.%s.one_shot' % device,
            )
            if world.rng.bit_generator.state != committed_rng:
                raise AssertionError(device + ' one-shot changed PCG64')
            details.append(device)
        finally:
            _a48_abort_direct(scheduler, cell)
    return ('/'.join(details) +
            ' resident authority + independent replay, prepare purity, '
            'fresh binding, CPU identity branch, no RNG and one-shot')


def test_a48b_translation_capacity_trust_rollback_and_exact_once():
    source_world, _, roomy, dt = _paid_translation_fixture(seed=9983)
    source_state = v3.pickle_clone(source_world.state_dict())
    source_cell = source_world.cells[0]
    probe = a4.FullFidelityA4GenomeAdapter(roomy).pack_cells([source_cell])
    lengths = np.diff(
        probe.sequence_offsets[:int(probe.sequence_count) + 1],
    )
    specs = a4.decode_a4_gene_cache_numpy(probe).materialize_gene_specs_host()[0]
    protein_capacity = max(
        len(set(source_cell.proteins).union(specs)),
        len(set(source_cell.damaged_proteins).union(specs)),
    )
    exact_values = {
        'max_cells': 1,
        'max_sequences': int(probe.sequence_count),
        'max_symbols': int(probe.symbol_count),
        'max_sequence_symbols': int(np.max(lengths)),
        'max_proteins_per_cell': int(protein_capacity),
    }
    if any(exact_values[name] <= 1 for name in (
            'max_sequences', 'max_symbols', 'max_sequence_symbols',
            'max_proteins_per_cell')):
        raise AssertionError('A4.8b capacity fixture is too small')

    exact = a4.GPU068A4Config(**exact_values)
    exact_world = a4.a3.s66.Formal066World.from_state(
        v3.pickle_clone(source_state),
    )
    exact_cell = exact_world.cells[0]
    exact_scheduler = a48b.A4TranslationEventScheduler(exact, 'cpu')
    _a48b_begin_translation_direct(
        exact_scheduler, exact_world, exact_cell, dt,
    )
    try:
        exact_scheduler.cpu_translation(
            exact_world, exact_cell, dt, exact_world.config,
        )
    finally:
        _a48_abort_direct(exact_scheduler, exact_cell)

    for axis in (
            'max_sequences', 'max_symbols', 'max_sequence_symbols',
            'max_proteins_per_cell'):
        values = dict(exact_values)
        values[axis] -= 1
        capacity = a4.GPU068A4Config(**values)
        world = a4.a3.s66.Formal066World.from_state(
            v3.pickle_clone(source_state),
        )
        cell = world.cells[0]
        scheduler = a48b.A4TranslationEventScheduler(capacity, 'cpu')
        _a48b_begin_translation_direct(scheduler, world, cell, dt)
        before = v3.pickle_clone(cell.state_dict())
        before_rng = copy.deepcopy(world.rng.bit_generator.state)
        try:
            _assert_raises(
                a4.A4CapacityError,
                lambda: scheduler.cpu_translation(
                    world, cell, dt, world.config,
                ),
            )
            v3.assert_recursive_close(
                before, cell.state_dict(), 0.0, 0.0,
                'a48b.capacity_' + axis,
            )
            _, record = scheduler._record(cell)
            if (world.rng.bit_generator.state != before_rng
                    or any(entry['event'] == 'translation_cpu'
                           for entry in record['events'])):
                raise AssertionError(axis + ' one-short crossed claim')
        finally:
            _a48_abort_direct(scheduler, cell)

    class TamperHostSource(a48b.A4TranslationEventScheduler):
        def _translation_candidate_ready(
                self, world, cell, dt, config, candidate):
            candidate.source_binding.state.pools[0, a4.a3.POOL_FUEL] += 1e-8
            return candidate

    class TamperHostReplay(a48b.A4TranslationEventScheduler):
        def _translation_candidate_ready(
                self, world, cell, dt, config, candidate):
            candidate.host_replay.last_translation[0] += 1e-8
            return candidate

    class TamperResidentRagged(a48b.A4TranslationEventScheduler):
        def _translation_candidate_ready(
                self, world, cell, dt, config, candidate):
            candidate.resident_binding.ragged.symbols.data[0].bitwise_xor_(1)
            return candidate

    class TamperResidentState(a48b.A4TranslationEventScheduler):
        def _translation_candidate_ready(
                self, world, cell, dt, config, candidate):
            candidate.resident_binding.state.pools.data[
                0, a4.a3.POOL_FUEL
            ].add_(1e-8)
            return candidate

    class TamperResidentCache(a48b.A4TranslationEventScheduler):
        def _translation_candidate_ready(
                self, world, cell, dt, config, candidate):
            candidate.resident_binding.cache.copy_numbers.data[0].add_(1)
            return candidate

    class SwapResidentRaggedClone(a48b.A4TranslationEventScheduler):
        def _translation_candidate_ready(
                self, world, cell, dt, config, candidate):
            object.__setattr__(
                candidate.resident_binding, 'ragged',
                candidate.resident_binding.ragged.clone(),
            )
            return candidate

    class SwapResidentStateClone(a48b.A4TranslationEventScheduler):
        def _translation_candidate_ready(
                self, world, cell, dt, config, candidate):
            object.__setattr__(
                candidate.resident_binding, 'state',
                candidate.resident_binding.state.clone(),
            )
            return candidate

    class SwapResidentCacheClone(a48b.A4TranslationEventScheduler):
        def _translation_candidate_ready(
                self, world, cell, dt, config, candidate):
            object.__setattr__(
                candidate.resident_binding, 'cache',
                candidate.resident_binding.cache.clone(),
            )
            return candidate

    class TamperResidentPlan(a48b.A4TranslationEventScheduler):
        def _translation_candidate_ready(
                self, world, cell, dt, config, candidate):
            candidate.resident_plan.last_translation.data[0].add_(1e-8)
            return candidate

    class SwapResidentPlanAuthority(a48b.A4TranslationEventScheduler):
        def _translation_candidate_ready(
                self, world, cell, dt, config, candidate):
            candidate.resident_plan = candidate.plan
            return candidate

    class TamperSettings(a48b.A4TranslationEventScheduler):
        def _translation_candidate_ready(
                self, world, cell, dt, config, candidate):
            self.a4_config = a4.GPU068A4Config(
                max_cells=1, max_sequences=1, max_symbols=1,
                max_sequence_symbols=1, max_proteins_per_cell=1,
            )
            self.a4_device = 'cuda'
            return candidate

    class TamperCandidateCell(a48b.A4TranslationEventScheduler):
        def _translation_candidate_ready(
                self, world, cell, dt, config, candidate):
            candidate.candidate_cell.last_translation += 1e-8
            return candidate

    class TamperCandidateMembrane(a48b.A4TranslationEventScheduler):
        def _translation_candidate_ready(
                self, world, cell, dt, config, candidate):
            candidate.candidate_cell.membrane[0] = np.nextafter(
                candidate.candidate_cell.membrane[0], np.inf,
            )
            return candidate

    class TamperCandidateGeneSpecs(a48b.A4TranslationEventScheduler):
        def _translation_candidate_ready(
                self, world, cell, dt, config, candidate):
            fingerprint = next(iter(candidate.candidate_cell.gene_specs))
            candidate.candidate_cell.gene_specs[fingerprint]['promoter'] += 0.125
            return candidate

    class MutateUnpackedCandidate(a48b.A4TranslationEventScheduler):
        isolated = False

        def _translation_candidate_ready(
                self, world, cell, dt, config, candidate):
            self.isolated = (
                candidate.candidate_cell.last_repair_flux
                is not cell.last_repair_flux
            )
            candidate.candidate_cell.last_repair_flux[0] += 0.125
            return candidate

    class TamperFreshBinding(a48b.A4TranslationEventScheduler):
        def _translation_candidate_ready(
                self, world, cell, dt, config, candidate):
            candidate.fresh_binding.state.pools[0, a4.a3.POOL_FUEL] += 1e-8
            return candidate

    for scheduler_type, label in (
            (TamperHostSource, 'host_source'),
            (TamperHostReplay, 'host_replay'),
            (TamperResidentRagged, 'resident_ragged_data'),
            (TamperResidentState, 'resident_state_data'),
            (TamperResidentCache, 'resident_cache_data'),
            (SwapResidentRaggedClone, 'resident_ragged_member_clone'),
            (SwapResidentStateClone, 'resident_state_member_clone'),
            (SwapResidentCacheClone, 'resident_cache_member_clone'),
            (TamperResidentPlan, 'resident_plan_data'),
            (SwapResidentPlanAuthority, 'resident_plan_authority_swap'),
            (TamperSettings, 'config_device'),
            (TamperCandidateCell, 'candidate_cell'),
            (TamperCandidateMembrane, 'candidate_membrane'),
            (TamperCandidateGeneSpecs, 'candidate_gene_specs'),
            (TamperFreshBinding, 'fresh_binding')):
        world = a4.a3.s66.Formal066World.from_state(
            v3.pickle_clone(source_state),
        )
        cell = world.cells[0]
        scheduler = scheduler_type(roomy, 'cpu')
        _a48b_begin_translation_direct(scheduler, world, cell, dt)
        before = v3.pickle_clone(cell.state_dict())
        before_world = v3.pickle_clone(world.state_dict())
        before_rng = copy.deepcopy(world.rng.bit_generator.state)
        before_objects = (
            id(cell.pools), id(cell.proteins),
            id(cell.damaged_proteins), id(cell.gene_specs), id(world.rng),
        )
        before_receipt_count = len(scheduler._receipts)
        try:
            _assert_raises(
                (a48b.A4TranslationCommitError, a4.A4SchemaError),
                lambda: scheduler.cpu_translation(
                    world, cell, dt, world.config,
                ),
            )
            v3.assert_recursive_close(
                before, cell.state_dict(), 0.0, 0.0,
                'a48b.tamper_' + label,
            )
            v3.assert_recursive_close(
                before_world, world.state_dict(), 0.0, 0.0,
                'a48b.tamper_world_' + label,
            )
            _, record = scheduler._record(cell)
            if (world.rng.bit_generator.state != before_rng
                    or before_objects != (
                        id(cell.pools), id(cell.proteins),
                        id(cell.damaged_proteins), id(cell.gene_specs),
                        id(world.rng),
                    )
                    or len(scheduler._receipts) != before_receipt_count
                    or any(entry['event'] == 'translation_cpu'
                           for entry in record['events'])):
                raise AssertionError(label + ' crossed preclaim boundary')
        finally:
            _a48_abort_direct(scheduler, cell)

    if torch is not None and torch.cuda.is_available():
        for scheduler_type, label in (
                (TamperCandidateMembrane, 'candidate_membrane'),
                (SwapResidentRaggedClone, 'resident_ragged_member_clone'),
                (SwapResidentStateClone, 'resident_state_member_clone'),
                (SwapResidentCacheClone, 'resident_cache_member_clone'),
                (SwapResidentPlanAuthority, 'resident_plan_authority_swap')):
            cuda_world = a4.a3.s66.Formal066World.from_state(
                v3.pickle_clone(source_state),
            )
            cuda_cell = cuda_world.cells[0]
            cuda_scheduler = scheduler_type(roomy, 'cuda')
            _a48b_begin_translation_direct(
                cuda_scheduler, cuda_world, cuda_cell, dt,
            )
            cuda_before = v3.pickle_clone(cuda_cell.state_dict())
            cuda_world_before = v3.pickle_clone(cuda_world.state_dict())
            cuda_rng = copy.deepcopy(cuda_world.rng.bit_generator.state)
            cuda_objects = (
                id(cuda_cell.pools), id(cuda_cell.proteins),
                id(cuda_cell.damaged_proteins), id(cuda_cell.gene_specs),
                id(cuda_world.rng),
            )
            cuda_receipt_count = len(cuda_scheduler._receipts)
            try:
                _assert_raises(
                    (a48b.A4TranslationCommitError, a4.A4SchemaError),
                    lambda: cuda_scheduler.cpu_translation(
                        cuda_world, cuda_cell, dt, cuda_world.config,
                    ),
                )
                v3.assert_recursive_close(
                    cuda_before, cuda_cell.state_dict(), 0.0, 0.0,
                    'a48b.cuda_' + label,
                )
                v3.assert_recursive_close(
                    cuda_world_before, cuda_world.state_dict(), 0.0, 0.0,
                    'a48b.cuda_world_' + label,
                )
                _, cuda_record = cuda_scheduler._record(cuda_cell)
                if (cuda_world.rng.bit_generator.state != cuda_rng
                        or cuda_objects != (
                            id(cuda_cell.pools), id(cuda_cell.proteins),
                            id(cuda_cell.damaged_proteins),
                            id(cuda_cell.gene_specs), id(cuda_world.rng),
                        )
                        or len(cuda_scheduler._receipts)
                        != cuda_receipt_count
                        or any(entry['event'] == 'translation_cpu'
                               for entry in cuda_record['events'])):
                    raise AssertionError(
                        'CUDA %s tamper crossed preclaim boundary' % label
                    )
            finally:
                _a48_abort_direct(cuda_scheduler, cuda_cell)
    elif _REQUIRE_CUDA:
        raise AssertionError('CUDA required for A4.8b trust regression')

    isolation_devices = ['cpu']
    if torch is not None and torch.cuda.is_available():
        isolation_devices.append('cuda')
    for device in isolation_devices:
        isolation_world = a4.a3.s66.Formal066World.from_state(
            v3.pickle_clone(source_state),
        )
        isolation_cell = isolation_world.cells[0]
        isolation_before = v3.pickle_clone(isolation_cell.state_dict())
        isolation_world_before = v3.pickle_clone(
            isolation_world.state_dict(),
        )
        repair_flux_object = isolation_cell.last_repair_flux
        repair_flux_before = np.asarray(
            isolation_cell.last_repair_flux, dtype=np.float64,
        ).copy()
        isolation_rng = copy.deepcopy(
            isolation_world.rng.bit_generator.state,
        )
        isolation_objects = (
            id(isolation_cell.pools), id(isolation_cell.proteins),
            id(isolation_cell.damaged_proteins),
            id(isolation_cell.gene_specs), id(isolation_world.rng),
        )
        isolation_scheduler = MutateUnpackedCandidate(roomy, device)
        _a48b_begin_translation_direct(
            isolation_scheduler, isolation_world, isolation_cell, dt,
        )
        isolation_receipt_count = len(isolation_scheduler._receipts)
        try:
            _assert_raises(
                (a48b.A4TranslationCommitError, a4.A4SchemaError),
                lambda: isolation_scheduler.cpu_translation(
                    isolation_world, isolation_cell, dt,
                    isolation_world.config,
                ),
            )
            if not isolation_scheduler.isolated:
                raise AssertionError(
                    device + ' unpacked candidate array aliases live state'
                )
            v3.assert_recursive_close(
                isolation_before, isolation_cell.state_dict(), 0.0, 0.0,
                'a48b.%s.unpacked_candidate_reject' % device,
            )
            v3.assert_recursive_close(
                isolation_world_before, isolation_world.state_dict(),
                0.0, 0.0,
                'a48b.%s.unpacked_candidate_world' % device,
            )
            if (isolation_cell.last_repair_flux is not repair_flux_object
                    or not np.array_equal(
                        isolation_cell.last_repair_flux, repair_flux_before,
                    )
                    or isolation_world.rng.bit_generator.state
                    != isolation_rng
                    or isolation_objects != (
                        id(isolation_cell.pools),
                        id(isolation_cell.proteins),
                        id(isolation_cell.damaged_proteins),
                        id(isolation_cell.gene_specs),
                        id(isolation_world.rng),
                    )
                    or len(isolation_scheduler._receipts)
                    != isolation_receipt_count):
                raise AssertionError(
                    device + ' unpacked candidate mutation escaped isolation'
                )
            _, isolation_record = isolation_scheduler._record(
                isolation_cell,
            )
            if any(entry['event'] == 'translation_cpu'
                   for entry in isolation_record['events']):
                raise AssertionError(
                    device + ' unpacked candidate tamper crossed claim'
                )
        finally:
            _a48_abort_direct(isolation_scheduler, isolation_cell)

    stale_world = a4.a3.s66.Formal066World.from_state(
        v3.pickle_clone(source_state),
    )
    stale_cell = stale_world.cells[0]
    stale_fingerprint = next(iter(stale_cell.gene_specs))
    stale_cell.gene_specs[stale_fingerprint]['promoter'] += 0.125
    stale_scheduler = a48b.A4TranslationEventScheduler(roomy, 'cpu')
    _a48b_begin_translation_direct(
        stale_scheduler, stale_world, stale_cell, dt,
    )
    stale_before = v3.pickle_clone(stale_cell.state_dict())
    stale_rng = copy.deepcopy(stale_world.rng.bit_generator.state)
    try:
        _assert_raises(
            (a48b.A4TranslationCommitError, a4.A4SchemaError),
            lambda: stale_scheduler.cpu_translation(
                stale_world, stale_cell, dt, stale_world.config,
            ),
        )
        v3.assert_recursive_close(
            stale_before, stale_cell.state_dict(), 0.0, 0.0,
            'a48b.preexisting_stale_gene_specs',
        )
        _, stale_record = stale_scheduler._record(stale_cell)
        if (stale_world.rng.bit_generator.state != stale_rng
                or any(entry['event'] == 'translation_cpu'
                       for entry in stale_record['events'])):
            raise AssertionError('stale gene_specs crossed claim boundary')
    finally:
        _a48_abort_direct(stale_scheduler, stale_cell)

    class TamperLiveGeneSpecs(a48b.A4TranslationEventScheduler):
        tampered_state = None

        def _translation_candidate_ready(
                self, world, cell, dt, config, candidate):
            fingerprint = next(iter(cell.gene_specs))
            cell.gene_specs[fingerprint]['promoter'] += 0.125
            self.tampered_state = v3.pickle_clone(cell.state_dict())
            return candidate

    live_world = a4.a3.s66.Formal066World.from_state(
        v3.pickle_clone(source_state),
    )
    live_cell = live_world.cells[0]
    live_scheduler = TamperLiveGeneSpecs(roomy, 'cpu')
    _a48b_begin_translation_direct(live_scheduler, live_world, live_cell, dt)
    live_rng = copy.deepcopy(live_world.rng.bit_generator.state)
    try:
        _assert_raises(
            (a48b.A4TranslationCommitError, a4.A4SchemaError),
            lambda: live_scheduler.cpu_translation(
                live_world, live_cell, dt, live_world.config,
            ),
        )
        if live_scheduler.tampered_state is None:
            raise AssertionError('live gene_specs seam did not run')
        v3.assert_recursive_close(
            live_scheduler.tampered_state, live_cell.state_dict(),
            0.0, 0.0, 'a48b.live_gene_specs_no_extra_mutation',
        )
        _, live_record = live_scheduler._record(live_cell)
        if (live_world.rng.bit_generator.state != live_rng
                or any(entry['event'] == 'translation_cpu'
                       for entry in live_record['events'])):
            raise AssertionError('live gene_specs tamper crossed claim boundary')
    finally:
        _a48_abort_direct(live_scheduler, live_cell)

    wrong_world = a4.a3.s66.Formal066World.from_state(
        v3.pickle_clone(source_state),
    )
    wrong_cell = wrong_world.cells[0]
    wrong_scheduler = a48b.A4TranslationEventScheduler(roomy, 'cpu')
    _a48b_begin_translation_direct(
        wrong_scheduler, wrong_world, wrong_cell, dt,
    )
    wrong_before = v3.pickle_clone(wrong_cell.state_dict())
    wrong_rng = copy.deepcopy(wrong_world.rng.bit_generator.state)
    _assert_raises(
        a3s.A3SchedulerProtocolError,
        lambda: wrong_scheduler.cpu_translation(
            wrong_world, wrong_cell, np.nextafter(dt, np.inf),
            wrong_world.config,
        ),
    )
    foreign = copy.deepcopy(wrong_cell)
    _assert_raises(
        a3s.A3SchedulerProtocolError,
        lambda: wrong_scheduler.cpu_translation(
            wrong_world, foreign, dt, wrong_world.config,
        ),
    )
    v3.assert_recursive_close(
        wrong_before, wrong_cell.state_dict(), 0.0, 0.0,
        'a48b.wrong_cell_dt',
    )
    if wrong_world.rng.bit_generator.state != wrong_rng:
        raise AssertionError('wrong cell/dt changed PCG64')
    _a48_abort_direct(wrong_scheduler, wrong_cell)

    class FailAfterPublishNoGeneTouch(a48b.A4TranslationEventScheduler):
        def _publish_translation_candidate(self, world, cell, candidate):
            super(FailAfterPublishNoGeneTouch, self)._publish_translation_candidate(
                world, cell, candidate,
            )
            cell.pools[a4.a3.POOL_FUEL] += 1.0
            cell.proteins[999999] = 1.0
            cell.damaged_proteins[999998] = 1.0
            cell.last_translation += 1.0
            raise RuntimeError('injected A4.8b publish failure without gene touch')

    class FailAfterPublishGeneReplace(a48b.A4TranslationEventScheduler):
        def _publish_translation_candidate(self, world, cell, candidate):
            super(FailAfterPublishGeneReplace, self)._publish_translation_candidate(
                world, cell, candidate,
            )
            cell.pools[a4.a3.POOL_FUEL] += 1.0
            cell.proteins[999999] = 1.0
            cell.damaged_proteins[999998] = 1.0
            cell.last_translation += 1.0
            fingerprint = next(iter(cell.gene_specs))
            cell.gene_specs[fingerprint]['promoter'] += 0.125
            cell.gene_specs = copy.deepcopy(cell.gene_specs)
            raise RuntimeError('injected A4.8b publish failure')

    rollback_devices = ['cpu']
    if torch is not None and torch.cuda.is_available():
        rollback_devices.append('cuda')

    def run_publish_rollback(failure_type, failure_label, device):
        rollback_world = a4.a3.s66.Formal066World.from_state(
            v3.pickle_clone(source_state),
        )
        rollback_cell = rollback_world.cells[0]
        rollback_scheduler = failure_type(roomy, device)
        _a48b_begin_translation_direct(
            rollback_scheduler, rollback_world, rollback_cell, dt,
        )
        rollback_before = v3.pickle_clone(rollback_cell.state_dict())
        rollback_rng_object = rollback_world.rng
        rollback_rng = copy.deepcopy(rollback_world.rng.bit_generator.state)
        rollback_ids = (
            id(rollback_cell.pools), id(rollback_cell.proteins),
            id(rollback_cell.damaged_proteins),
            id(rollback_cell.gene_specs), id(rollback_world.rng),
        )
        rollback_nested_spec_ids = {
            int(fingerprint): id(spec)
            for fingerprint, spec in rollback_cell.gene_specs.items()
        }
        rollback_gene_order = list(rollback_cell.gene_specs)
        caught = None
        try:
            rollback_scheduler.cpu_translation(
                rollback_world, rollback_cell, dt, rollback_world.config,
            )
        except RuntimeError as error:
            caught = error
        if caught is None:
            raise AssertionError(
                '%s/%s injected translation publish failure escaped' % (
                    device, failure_label,
                )
            )
        v3.assert_recursive_close(
            rollback_before, rollback_cell.state_dict(), 0.0, 0.0,
            'a48b.%s.%s.publish_rollback' % (device, failure_label),
        )
        actual_nested_spec_ids = {
            int(fingerprint): id(spec)
            for fingerprint, spec in rollback_cell.gene_specs.items()
        }
        if (rollback_world.rng is not rollback_rng_object
                or rollback_world.rng.bit_generator.state != rollback_rng
                or rollback_ids != (
                    id(rollback_cell.pools), id(rollback_cell.proteins),
                    id(rollback_cell.damaged_proteins),
                    id(rollback_cell.gene_specs), id(rollback_world.rng),
                )
                or actual_nested_spec_ids != rollback_nested_spec_ids
                or list(rollback_cell.gene_specs) != rollback_gene_order):
            raise AssertionError(
                '%s/%s translation rollback identity/state differs' % (
                    device, failure_label,
                )
            )
        _a48b_translation_entry(rollback_scheduler, rollback_cell)
        receipt = _a48_abort_direct(
            rollback_scheduler, rollback_cell, caught,
        )
        if receipt['status'] != 'aborted':
            raise AssertionError(
                '%s/%s postclaim translation failure was not aborted' % (
                    device, failure_label,
                )
            )

    for failure_type, failure_label in (
            (FailAfterPublishNoGeneTouch, 'no_gene_touch'),
            (FailAfterPublishGeneReplace, 'nested_and_outer_gene_replace')):
        for device in rollback_devices:
            run_publish_rollback(failure_type, failure_label, device)

    duplicate_world = a4.a3.s66.Formal066World.from_state(
        v3.pickle_clone(source_state),
    )
    duplicate_cell = duplicate_world.cells[0]
    duplicate_scheduler = a48b.A4TranslationEventScheduler(roomy, 'cpu')
    _a48b_begin_translation_direct(
        duplicate_scheduler, duplicate_world, duplicate_cell, dt,
    )
    duplicate_scheduler.cpu_translation(
        duplicate_world, duplicate_cell, dt, duplicate_world.config,
    )
    duplicate_before = v3.pickle_clone(duplicate_cell.state_dict())
    duplicate_rng = copy.deepcopy(duplicate_world.rng.bit_generator.state)
    _assert_raises(
        a3s.A3DuplicateEventError,
        lambda: duplicate_scheduler.cpu_translation(
            duplicate_world, duplicate_cell, dt, duplicate_world.config,
        ),
    )
    v3.assert_recursive_close(
        duplicate_before, duplicate_cell.state_dict(), 0.0, 0.0,
        'a48b.duplicate',
    )
    if duplicate_world.rng.bit_generator.state != duplicate_rng:
        raise AssertionError('duplicate translation changed PCG64')
    _a48_abort_direct(duplicate_scheduler, duplicate_cell)

    order_world = a4.a3.s66.Formal066World.from_state(
        v3.pickle_clone(source_state),
    )
    order_cell = order_world.cells[0]
    order_scheduler = a48b.A4TranslationEventScheduler(roomy, 'cpu')
    order_scheduler.begin_step(order_world, [order_cell])
    metabolism_rank = a3s.WORLD_EVENT_ORDER.index('cell_metabolism_loop')
    for event in a3s.WORLD_EVENT_ORDER[:metabolism_rank]:
        order_scheduler.claim_world(event, status='skipped')
    order_scheduler.claim(order_cell, 'surface_exchange', status='skipped')
    order_scheduler.reserve_metabolism_dispatch(order_world, order_cell, dt)
    order_scheduler.begin_a3_metabolism(order_world, order_cell, dt)
    order_scheduler.claim(order_cell, 'gene_refresh', status='skipped')
    order_before = v3.pickle_clone(order_cell.state_dict())
    order_rng = copy.deepcopy(order_world.rng.bit_generator.state)
    _assert_raises(
        a3s.A3EventOrderError,
        lambda: order_scheduler.cpu_translation(
            order_world, order_cell, dt, order_world.config,
        ),
    )
    v3.assert_recursive_close(
        order_before, order_cell.state_dict(), 0.0, 0.0,
        'a48b.out_of_order',
    )
    if order_world.rng.bit_generator.state != order_rng:
        raise AssertionError('out-of-order translation changed PCG64')
    _a48_abort_direct(order_scheduler, order_cell)
    return ('Q/S/W/P exact + each one-short; host/resident member, storage, '
            'device, authority and full-candidate trust rejection; packed/'
            'unpacked non-alias; wrong cell/dt; preclaim atomic; postclaim '
            'nested gene-spec/value/object/RNG rollback; duplicate/out-of-order')


def test_a48b_world_lockstep_stress_interleave_save_clone_and_a3_authority():
    capacity = a4.GPU068A4Config()
    for steps, seed in ((1, 9991), (10, 9992)):
        source = v3.make_world(seed=seed, cells=1)
        state = v3.pickle_clone(source.state_dict())
        cpu = a4.a3.s66.Formal066World.from_state(v3.pickle_clone(state))
        hybrid = _a48b_hybrid_from_state(state, capacity, 'cpu')
        for _ in range(steps):
            cpu.step(0.1)
            hybrid.step(0.1)
        v3._assert_world_pair(
            cpu, hybrid, state_atol=v3.WORLD_FP64_ATOL,
            ledger_atol=v3.LEDGER_ATOL,
            label='a48b.lockstep_%d' % steps,
        )

    stressed = v3.make_world(seed=9993, cells=1)
    stressed.config.external_translator = True
    stressed.config.protein_repair = True
    stressed.config.quiescence = True
    stressed.config.quiescence_effector = True
    for ci, cell in enumerate(stressed.cells):
        cell.current_stress = 1.4 + 0.1 * ci
        cell.damage_trace[:] = 0.8 + 0.05 * ci
        cell.membrane_oxidation[:] = np.linspace(
            0.25, 1.75, len(cell.membrane_oxidation),
        )
        cell.pools[a4.a3.POOL_REACTIVE] += 0.14 + 0.01 * ci
        cell.pools[a4.a3.POOL_AGGREGATE] += 0.06
        cell.pools[a4.a3.POOL_ATP] += 0.45
        cell.behavioural_quiescence = 0.55 + 0.05 * ci
        cell._sync_damage_pool()
    stress_state = v3.pickle_clone(stressed.state_dict())
    stress_cpu = a4.a3.s66.Formal066World.from_state(
        v3.pickle_clone(stress_state),
    )
    stress_hybrid = _a48b_hybrid_from_state(
        stress_state, capacity, 'cpu',
    )
    for _ in range(3):
        stress_cpu.step(0.1)
        stress_hybrid.step(0.1)
    v3._assert_world_pair(
        stress_cpu, stress_hybrid, state_atol=v3.WORLD_FP64_ATOL,
        ledger_atol=v3.LEDGER_ATOL,
        label='a48b.translation_heavy_stress_3',
    )

    interleave_source = v3.make_world(seed=9994, cells=2)
    interleave_source.config.endogenous_damage = False
    for cell in interleave_source.cells:
        cell.genome_lesions = [
            0.5 / 0.00065 for _ in cell.genomes
        ]
        cell._refresh_gene_cache()
    interleave_state = v3.pickle_clone(interleave_source.state_dict())
    interleave_cpu = a4.a3.s66.Formal066World.from_state(
        v3.pickle_clone(interleave_state),
    )
    interleave_hybrid = _a48b_hybrid_from_state(
        interleave_state, capacity, 'cpu',
    )
    interleave_cpu.step(0.1)
    original_translation = a48.A4HydrolysisEventScheduler.cpu_translation

    def legacy_translation_bomb(*args, **kwargs):
        raise AssertionError('legacy CPU translation bridge was called')

    a48.A4HydrolysisEventScheduler.cpu_translation = legacy_translation_bomb
    try:
        receipt = interleave_hybrid.step(0.1)
    finally:
        a48.A4HydrolysisEventScheduler.cpu_translation = original_translation
    v3._assert_world_pair(
        interleave_cpu, interleave_hybrid,
        state_atol=v3.WORLD_FP64_ATOL,
        ledger_atol=v3.LEDGER_ATOL,
        label='a48b.two_cell_interleave',
    )
    previous_motion = None
    for record in receipt['cells']:
        by_name = {entry['event']: entry for entry in record['events']}
        names = [entry['event'] for entry in record['events']]
        if (names.count('translation_cpu') != 1
                or names.count('genome_hydrolysis_cpu_rng') != 1):
            raise AssertionError('A4.8b integrated event count differs')
        translation = by_name['translation_cpu']['ordinal']
        replication = by_name['replication_cpu']['ordinal']
        hydrolysis = by_name['genome_hydrolysis_cpu_rng']['ordinal']
        motion = by_name['motion']['ordinal']
        if not translation < replication < hydrolysis < motion:
            raise AssertionError('A4.8b per-cell event order differs')
        if previous_motion is not None and previous_motion >= translation:
            raise AssertionError('A4.8b two-cell chain was batched/reordered')
        previous_motion = motion
        if (by_name['translation_cpu']['metadata']['authority']
                != 'A4.8b-resident-plan-atomic-cpu-cell-commit'):
            raise AssertionError('A4.8b translation authority differs')
        if by_name['genome_hydrolysis_cpu_rng']['metadata'][
                'eligible_genome_count'] < 1:
            raise AssertionError('interleave fixture drew no hydrolysis RNG')

    persisted = _a48b_hybrid_from_state(
        v3.make_world(seed=9995, cells=1).state_dict(), capacity, 'cpu',
    )
    persisted.step(0.1)
    twin = persisted.clone()
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, 'a48b.pkl')
        persisted.save(path)
        restored = a48b.Hybrid066WorldA4Translation.load(path)
        for value in (twin, restored):
            if (type(value) is not a48b.Hybrid066WorldA4Translation
                    or type(value.scheduler)
                    is not a48b.A4TranslationEventScheduler
                    or value.scheduler.a4_device != 'cpu'
                    or value.scheduler.a4_config != capacity):
                raise AssertionError('A4.8b save/clone authority was lost')
        persisted.step(0.1)
        twin.step(0.1)
        restored.step(0.1)
        v3.assert_recursive_close(
            persisted.world.state_dict(), twin.world.state_dict(),
            0.0, 0.0, 'a48b.clone_continuation',
        )
        v3.assert_recursive_close(
            persisted.world.state_dict(), restored.world.state_dict(),
            0.0, 0.0, 'a48b.save_continuation',
        )

    active = _a48b_hybrid_from_state(
        v3.make_world(seed=9996, cells=1).state_dict(), capacity, 'cpu',
    )
    active.scheduler.begin_step(active.world, active.world.cells)
    _assert_raises(a3s.A3SchedulerProtocolError, active.clone)
    active.scheduler.abort_step(RuntimeError('expected active-save rejection'))
    active.scheduler._a4_translation_commit_active = True
    try:
        _assert_raises(a3s.A3SchedulerProtocolError, active.state_dict)
    finally:
        active.scheduler._a4_translation_commit_active = False

    boundary_details = []
    devices = ['cpu']
    if torch is not None and torch.cuda.is_available():
        devices.append('cuda')
    elif _REQUIRE_CUDA:
        raise AssertionError('CUDA required for A4.8b boundary authority')
    for device in devices:
        for (label, state, cell_id, dt,
             expected_symbols, expected_rng_change) in (
                _a48b_replication_boundary_cases(seed=10001)
        ):
            direct = a4.a3.s66.Formal066World.from_state(
                v3.pickle_clone(state),
            )
            integrated_hybrid = _a48b_hybrid_from_state(
                state, capacity, device,
            )
            integrated = integrated_hybrid.world
            direct_cell = next(
                cell for cell in direct.cells if int(cell.cell_id) == cell_id
            )
            integrated_cell = next(
                cell for cell in integrated.cells
                if int(cell.cell_id) == cell_id
            )
            direct_before_rng = copy.deepcopy(
                direct.rng.bit_generator.state,
            )
            integrated_before_rng = copy.deepcopy(
                integrated.rng.bit_generator.state,
            )
            direct_cell.translate(dt, direct.config)
            direct_cell._replicate_genome(direct, dt, direct.config)

            scheduler = integrated_hybrid.scheduler
            _a48b_begin_translation_direct(
                scheduler, integrated, integrated_cell, dt,
            )
            scheduler.cpu_translation(
                integrated, integrated_cell, dt, integrated.config,
            )
            if integrated.rng.bit_generator.state != integrated_before_rng:
                raise AssertionError(
                    '%s/%s translation consumed replication RNG' % (
                        device, label,
                    )
                )
            scheduler.cpu_replication(
                integrated, integrated_cell, dt, integrated.config,
            )
            _a48b_assert_cell_matches(
                direct_cell, integrated_cell,
                'a48b.boundary.%s.%s' % (device, label),
                atol=v3.WORLD_FP64_ATOL,
            )
            if direct.rng.bit_generator.state != integrated.rng.bit_generator.state:
                raise AssertionError(
                    '%s/%s post-translation replication PCG64 differs' % (
                        device, label,
                    )
                )
            if (int(integrated_cell.last_replication_symbols)
                    != expected_symbols):
                raise AssertionError(
                    '%s/%s discrete replication gate differs' % (
                        device, label,
                    )
                )
            actual_rng_change = (
                integrated.rng.bit_generator.state != integrated_before_rng
            )
            direct_rng_change = (
                direct.rng.bit_generator.state != direct_before_rng
            )
            if (actual_rng_change != expected_rng_change
                    or direct_rng_change != expected_rng_change):
                raise AssertionError(
                    '%s/%s RNG branch control differs' % (device, label)
                )
            replication_entry = [
                entry for entry in scheduler._record(integrated_cell)[1]['events']
                if entry['event'] == 'replication_cpu'
            ]
            if (len(replication_entry) != 1
                    or replication_entry[0]['metadata']['authority']
                    != 'frozen-0.6.6-wrapper-to-0.4'):
                raise AssertionError('A3 CPU replication authority differs')
            _a48_abort_direct(scheduler, integrated_cell)
            boundary_details.append(device + ':' + label)

    if (a48b.FULL_GPU_WORLD_STEP is not False
            or a48b.PORT_STATUS.get('genome_replication')
            != 'cpu-authoritative-scheduler-bridged'
            or a3s.PORT_STATUS.get('translation_replication_division')
            != 'CPU-authoritative-scheduler-bridged'):
        raise AssertionError('A4.8b changed A3/full-GPU authority')
    for relative, expected in PROMOTED_A3_SHA256.items():
        path = os.path.join(ROOT, *relative.split('/'))
        if _sha256(path) != expected:
            raise AssertionError(relative + ' changed from promoted A3 bytes')
    return ('1/10-step + translation-heavy stress + 2-cell nonbatched '
            'translation/replication/hydrolysis/motion lockstep; legacy '
            'translation bomb; save/load/clone + active reject; immediate '
            'A3 CPU replication boundaries and PCG64 exact: ' +
            ', '.join(boundary_details))


def _a48c_begin_replication_direct(scheduler, world, cell, dt):
    """Enter exactly the post-translation replication event boundary."""
    _a48b_begin_translation_direct(scheduler, world, cell, dt)
    scheduler.claim(
        cell, 'translation_cpu', status='skipped',
        metadata={'validation': 'A4.8c1'},
    )
    return scheduler


def _a48c_replication_entry(scheduler, cell):
    _, record = scheduler._record(cell)
    entries = [entry for entry in record['events']
               if entry['event'] == 'replication_cpu']
    if len(entries) != 1:
        raise AssertionError(
            'A4.8c1 replication event count is %d' % len(entries)
        )
    return copy.deepcopy(entries[0])


def _a48c_hybrid_from_state(state, a4_config=None, device='cpu'):
    world = a4.a3.s66.Formal066World.from_state(v3.pickle_clone(state))
    backend = a4.a3.TorchKernelBackendA3(
        v3._a3_config(device=device, precision='float64'),
    )
    return a48c.Hybrid066WorldA4Replication(
        world, backend=backend,
        a4_config=a4_config or a4.GPU068A4Config(),
        a4_device=device,
    )


def _a48c_reduce_world(world, cells):
    world.cells = list(cells)
    world.initial_total_material = world.total_material()
    world.last_step_material_residual = 0.0
    return world


def _a48c_identity_snapshot(world, cell):
    return {
        'rng': id(world.rng),
        'pools': id(cell.pools),
        'genomes': id(cell.genomes),
        'genome_items': tuple(id(value) for value in cell.genomes),
        'lesions': id(cell.genome_lesions),
        'template': id(cell.replication_template),
        'copy': id(cell.replication_copy),
        'events': id(cell.mutation_events),
        'proteins': id(cell.proteins),
        'damaged': id(cell.damaged_proteins),
        'gene_specs': id(cell.gene_specs),
        'gene_spec_items': tuple(
            (int(key), id(value)) for key, value in cell.gene_specs.items()
        ),
    }


def _a48c_assert_oracle(expected_world, expected_cell, actual_world,
                        actual_cell, label):
    _a48b_assert_cell_matches(
        expected_cell, actual_cell, label,
        atol=a48c.REPLICATION_ORACLE_ATOL,
    )
    if (expected_world.rng.bit_generator.state
            != actual_world.rng.bit_generator.state):
        raise AssertionError(label + ' full PCG64 state differs')
    v3.assert_recursive_close(
        float(expected_world.dissipated_energy),
        float(actual_world.dissipated_energy),
        atol=0.0, rtol=0.0, path=label + '.dissipated_energy',
    )
    if (expected_cell.replication_template is None
            or actual_cell.replication_template is None
            or len(expected_cell.replication_copy)
            >= len(expected_cell.replication_template)
            or len(actual_cell.replication_copy)
            >= len(actual_cell.replication_template)):
        raise AssertionError(label + ' escaped active noncompletion scope')


def _a48c_oracle_cases(seed=10101):
    """Fresh one-cell states spanning both supported c1 branches."""
    cases = []

    world, cells, _, dt = _paid_replication_b_fixture(seed=seed)
    _a48c_reduce_world(world, [cells[0]])
    cases.append((
        'deterministic-rich', v3.pickle_clone(world.state_dict()),
        int(cells[0].cell_id), dt, False, 'ordinary',
    ))

    world, cells, _, dt = _paid_replication_b_fixture(seed=seed + 5)
    cell = cells[1]
    _a48c_reduce_world(world, [cell])
    cases.append((
        'requested-zero', v3.pickle_clone(world.state_dict()),
        int(cell.cell_id), dt, False, 'requested-zero',
    ))

    for offset, label, mutation_rate, mode in (
            (1, 'substitution-miss', 0.0, 'miss'),
            (2, 'substitution-hit', 1.0, 'hit')):
        world, cells, _, dt = _a45_substitution_fixture(
            seed=seed + offset,
        )
        cell = cells[0]
        world.config.proofreading = False
        world.config.mutation_rate = float(mutation_rate)
        cell.replication_template_lesion = 0.0
        cell.pools[a4.a3.POOL_REACTIVE] = 0.0
        cell.genome_lesions = [0.0 for _ in cell.genomes]
        cell._refresh_gene_cache()
        _a48c_reduce_world(world, [cell])
        cases.append((
            label, v3.pickle_clone(world.state_dict()),
            int(cell.cell_id), dt, True, mode,
        ))

    world, cells, _, _ = _a45_substitution_fixture(seed=seed + 3)
    cell = cells[0]
    cell.replication_fractional = 0.375
    _a48c_reduce_world(world, [cell])
    cases.append((
        'dt-zero', v3.pickle_clone(world.state_dict()),
        int(cell.cell_id), 0.0, True, 'zero',
    ))

    world, cells, _, dt = _paid_replication_fixture(seed=seed + 4)
    world.config.mutation = True
    world.config.mutation_rate = 1.0
    cell = cells[3]
    _a48c_reduce_world(world, [cell])
    cases.append((
        'resource-stop', v3.pickle_clone(world.state_dict()),
        int(cell.cell_id), dt, True, 'zero',
    ))
    return cases


def test_a48c1_active_noncompletion_atomic_commit_formal066_oracle():
    if (_REQUIRE_CUDA
            and (torch is None or not torch.cuda.is_available())):
        raise AssertionError('CUDA required for A4.8c1 requested-zero')
    capacity = a4.GPU068A4Config()
    details = []

    class ObserveReplicationPlan(a48c.A4ReplicationEventScheduler):
        def _replication_candidate_ready(
                self, world, cell, dt, config, candidate):
            plan = candidate.plan
            self.plan_observation = {
                'requested_symbols': int(plan.requested_symbols[0]),
                'append_count': int(plan.append_count[0]),
                'replication_fractional_after': float(
                    plan.replication_fractional_after[0]
                ),
                'last_replication_symbols': int(
                    plan.last_replication_symbols[0]
                ),
                'last_effective_error_rate': float(
                    plan.last_effective_error_rate[0]
                ),
                'cumulative_proofreading_atp_after': float(
                    plan.cumulative_proofreading_atp_after[0]
                ),
            }
            return candidate

    for label, state, cell_id, dt, mutation_enabled, mode in (
            _a48c_oracle_cases()):
        devices = ['cpu']
        if (label == 'requested-zero' and torch is not None
                and torch.cuda.is_available()):
            devices.append('cuda')
        for device in devices:
            direct = a4.a3.s66.Formal066World.from_state(
                v3.pickle_clone(state),
            )
            integrated = a4.a3.s66.Formal066World.from_state(
                v3.pickle_clone(state),
            )
            direct_cell = next(
                cell for cell in direct.cells if int(cell.cell_id) == cell_id
            )
            cell = next(
                cell for cell in integrated.cells if int(cell.cell_id) == cell_id
            )
            direct_ids = _a48c_identity_snapshot(direct, direct_cell)
            integrated_ids = _a48c_identity_snapshot(integrated, cell)
            direct_rng_before = copy.deepcopy(
                direct.rng.bit_generator.state,
            )
            integrated_rng_before = copy.deepcopy(
                integrated.rng.bit_generator.state,
            )
            direct_events_before = int(
                direct_cell.mutation_events.get('substitution', 0)
            )
            direct_energy = float(direct.dissipated_energy)
            fractional_before = float(direct_cell.replication_fractional)
            last_before = int(direct_cell.last_replication_symbols)
            error_before = float(direct_cell.last_effective_error_rate)
            proof_before = float(direct_cell.cumulative_proofreading_atp)
            copy_before = list(direct_cell.replication_copy)
            pools_before = np.asarray(
                direct_cell.pools, dtype=np.float64,
            ).copy()
            direct_cell._replicate_genome(direct, dt, direct.config)
            appended = int(direct_cell.last_replication_symbols)
            substitution_delta = int(
                direct_cell.mutation_events.get('substitution', 0)
            ) - direct_events_before
            if mode == 'miss' and (
                    appended <= 0 or substitution_delta != 0):
                raise AssertionError(
                    label + ' did not consume miss thresholds'
                )
            if mode == 'hit' and (
                    appended <= 0 or substitution_delta != appended):
                raise AssertionError(
                    label + ' did not force every substitution'
                )
            if mode == 'zero' and appended != 0:
                raise AssertionError(label + ' unexpectedly copied a symbol')
            if mode == 'requested-zero':
                if (not float(dt) > 0.0
                        or appended != 0
                        or substitution_delta != 0
                        or direct_cell.replication_copy != copy_before
                        or not np.array_equal(
                            np.asarray(
                                direct_cell.pools, dtype=np.float64,
                            ).view(np.uint64),
                            pools_before.view(np.uint64),
                        )
                        or float(direct_cell.replication_fractional).hex()
                        == fractional_before.hex()
                        or last_before == 0
                        or error_before.hex()
                        == float(
                            direct_cell.last_effective_error_rate
                        ).hex()
                        or proof_before.hex()
                        != float(
                            direct_cell.cumulative_proofreading_atp
                        ).hex()):
                    raise AssertionError(
                        label + ' did not execute active-gate zero work'
                    )
            if float(direct.dissipated_energy) != direct_energy:
                raise AssertionError(
                    label + ' charged world energy for replication'
                )
            if _a48c_identity_snapshot(direct, direct_cell) != direct_ids:
                raise AssertionError(
                    label + ' frozen CPU object identity changed'
                )

            scheduler = ObserveReplicationPlan(capacity, device)
            _a48c_begin_replication_direct(
                scheduler, integrated, cell, dt,
            )
            try:
                scheduler.cpu_replication(
                    integrated, cell, dt, integrated.config,
                )
                _a48c_assert_oracle(
                    direct, direct_cell, integrated, cell,
                    'a48c1.oracle.%s.%s' % (device, label),
                )
                if (_a48c_identity_snapshot(integrated, cell)
                        != integrated_ids):
                    raise AssertionError(
                        label + ' integrated object identity changed'
                    )
                observation = scheduler.plan_observation
                entry = _a48c_replication_entry(scheduler, cell)
                metadata = entry['metadata']
                expected_branch = (
                    'active-noncompletion-substitution'
                    if mutation_enabled else
                    'active-noncompletion-deterministic'
                )
                expected_rng_calls = (
                    appended + substitution_delta
                    if mutation_enabled else 0
                )
                if (metadata.get('branch') != expected_branch
                        or bool(metadata.get('mutation_enabled'))
                        != mutation_enabled
                        or metadata.get('device') != device
                        or int(metadata.get('requested_symbols', -1))
                        != observation['requested_symbols']
                        or int(metadata.get('amount', -1)) != appended
                        or observation['append_count'] != appended
                        or bool(metadata.get('work_performed'))
                        != (appended > 0)
                        or int(metadata.get('substitution_events', -1))
                        != substitution_delta
                        or int(metadata.get('rng_call_count', -1))
                        != expected_rng_calls):
                    raise AssertionError(
                        '%s/%s receipt/plan metadata differs' % (
                            device, label,
                        )
                    )
                if mode == 'requested-zero':
                    if (observation['requested_symbols'] != 0
                            or observation['append_count'] != 0
                            or int(metadata['requested_symbols']) != 0
                            or int(metadata['amount']) != 0
                            or bool(metadata['work_performed'])):
                        raise AssertionError(
                            device + ' requested-zero plan/receipt differs'
                        )
                    v3.assert_recursive_close(
                        [
                            float(direct_cell.replication_fractional),
                            int(direct_cell.last_replication_symbols),
                            float(direct_cell.last_effective_error_rate),
                            float(direct_cell.cumulative_proofreading_atp),
                        ],
                        [
                            observation['replication_fractional_after'],
                            observation['last_replication_symbols'],
                            observation['last_effective_error_rate'],
                            observation[
                                'cumulative_proofreading_atp_after'
                            ],
                        ],
                        atol=a48c.REPLICATION_ORACLE_ATOL, rtol=0.0,
                        path='a48c1.%s.requested_zero_telemetry' % device,
                    )
                rng_changed = (
                    integrated.rng.bit_generator.state
                    != integrated_rng_before
                )
                expected_rng_change = mutation_enabled and appended > 0
                if rng_changed != expected_rng_change:
                    raise AssertionError(
                        label + ' live PCG64 branch differs'
                    )
                if ((not mutation_enabled)
                        and direct.rng.bit_generator.state
                        != direct_rng_before):
                    raise AssertionError(
                        label + ' mutation-off CPU changed RNG'
                    )
                details.append('%s@%s:%d/%d/%d' % (
                    label, device, observation['requested_symbols'],
                    appended, substitution_delta,
                ))
            finally:
                _a48_abort_direct(scheduler, cell)
    return ('Formal066 active/noncompletion resident commits preserve copy, '
            'paid ledger, object identity and full PCG64: ' +
            ', '.join(details))


def test_a48c1_active_noncompletion_candidate_cpu_cuda_purity_and_rng():
    if torch is None:
        raise AssertionError('PyTorch is required for A4.8c1')
    if _REQUIRE_CUDA and not torch.cuda.is_available():
        raise AssertionError('CUDA required but unavailable; fallback forbidden')
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    selected = [
        case for case in _a48c_oracle_cases(seed=10121)
        if case[0] in ('deterministic-rich', 'substitution-hit')
    ]
    details = []
    for device in devices:
        for label, state, cell_id, dt, mutation_enabled, _ in selected:
            direct = a4.a3.s66.Formal066World.from_state(
                v3.pickle_clone(state),
            )
            world = a4.a3.s66.Formal066World.from_state(
                v3.pickle_clone(state),
            )
            direct_cell = next(
                cell for cell in direct.cells if int(cell.cell_id) == cell_id
            )
            cell = next(
                cell for cell in world.cells if int(cell.cell_id) == cell_id
            )
            direct_cell._replicate_genome(direct, dt, direct.config)
            scheduler = a48c.A4ReplicationEventScheduler(
                a4.GPU068A4Config(), device,
            )
            _a48c_begin_replication_direct(scheduler, world, cell, dt)
            before_cell = v3.pickle_clone(cell.state_dict())
            before_rng = copy.deepcopy(world.rng.bit_generator.state)
            before_energy = float(world.dissipated_energy)
            try:
                candidate = scheduler._prepare_replication_candidate(
                    world, cell, dt, world.config,
                )
                v3.assert_recursive_close(
                    before_cell, cell.state_dict(), 0.0, 0.0,
                    'a48c1.%s.%s.prepare_cell' % (device, label),
                )
                if (world.rng.bit_generator.state != before_rng
                        or float(world.dissipated_energy) != before_energy
                        or any(entry['event'] == 'replication_cpu'
                               for entry in scheduler._record(cell)[1]['events'])):
                    raise AssertionError(
                        '%s/%s prepare crossed claim' % (device, label)
                    )
                expected_device = torch.device(device)
                if (candidate.resident_plan.pools_after.device.type
                        != expected_device.type
                        or candidate.resident_binding.state.pools.device.type
                        != expected_device.type
                        or candidate.resident_binding.ragged.symbols.device.type
                        != expected_device.type):
                    raise AssertionError(
                        '%s/%s resident artifact moved device' % (
                            device, label,
                        )
                    )
                if (candidate.candidate_cell is cell
                        or candidate.candidate_cell.pools is cell.pools
                        or candidate.candidate_cell.replication_copy
                        is cell.replication_copy):
                    raise AssertionError(
                        '%s/%s candidate aliases live biology' % (
                            device, label,
                        )
                    )
                a48c._plan_semantic_match(
                    candidate.plan, candidate.host_replay,
                )
                if mutation_enabled:
                    if (candidate.tape is None
                            or candidate.resident_tape is None
                            or candidate.rng_before_state != before_rng
                            or candidate.rng_after_state
                            != candidate.tape.rng_after_state):
                        raise AssertionError(
                            '%s/%s substitution tape/RNG differs' % (
                                device, label,
                            )
                        )
                elif (candidate.tape is not None
                        or candidate.resident_tape is not None
                        or candidate.rng_after_state != before_rng):
                    raise AssertionError(
                        '%s/%s deterministic candidate owns RNG work' % (
                            device, label,
                        )
                    )
                scheduler._commit_replication_candidate(
                    world, cell, dt, world.config, candidate,
                )
                _a48c_assert_oracle(
                    direct, direct_cell, world, cell,
                    'a48c1.%s.%s.commit' % (device, label),
                )
                committed = v3.pickle_clone(cell.state_dict())
                committed_rng = copy.deepcopy(world.rng.bit_generator.state)
                _assert_raises(
                    a48c.A4ReplicationCommitError,
                    lambda: scheduler._commit_replication_candidate(
                        world, cell, dt, world.config, candidate,
                    ),
                )
                v3.assert_recursive_close(
                    committed, cell.state_dict(), 0.0, 0.0,
                    'a48c1.%s.%s.one_shot' % (device, label),
                )
                if world.rng.bit_generator.state != committed_rng:
                    raise AssertionError(
                        '%s/%s one-shot retry changed RNG' % (device, label)
                    )
                details.append(device + ':' + label)
            finally:
                _a48_abort_direct(scheduler, cell)
    return ('/'.join(details) +
            ' prepare-pure resident candidate, explicit readback/NumPy '
            'association, fresh binding, full PCG64 and one-shot PASS')


def _a48c_capacity_for_cell(cell):
    sequences = list(cell.genomes) + [
        cell.replication_template, cell.replication_copy,
    ]
    q_value = len(sequences)
    s_value = sum(len(value) for value in sequences)
    w_value = max(len(value) for value in sequences)
    gene_keys = set(int(value) for value in cell.gene_specs)
    p_value = max(
        len(gene_keys.union(int(value) for value in cell.proteins)),
        len(gene_keys.union(
            int(value) for value in cell.damaged_proteins
        )),
    )
    return {
        'max_cells': 1,
        'max_sequences': q_value,
        'max_symbols': s_value,
        'max_sequence_symbols': w_value,
        'max_proteins_per_cell': p_value,
    }


def _a48c_assert_preclaim_failure(state, cell_id, dt, capacity,
                                  scheduler_type=None, label='scope'):
    world = a4.a3.s66.Formal066World.from_state(v3.pickle_clone(state))
    cell = next(
        item for item in world.cells if int(item.cell_id) == int(cell_id)
    )
    scheduler_type = scheduler_type or a48c.A4ReplicationEventScheduler
    scheduler = scheduler_type(capacity, 'cpu')
    _a48c_begin_replication_direct(scheduler, world, cell, dt)
    before = v3.pickle_clone(cell.state_dict())
    before_rng = copy.deepcopy(world.rng.bit_generator.state)
    before_energy = float(world.dissipated_energy)
    try:
        _assert_raises(
            (a4.A4Error, a3s.A3SchedulerError),
            lambda: scheduler.cpu_replication(
                world, cell, dt, world.config,
            ),
        )
        v3.assert_recursive_close(
            before, cell.state_dict(), 0.0, 0.0,
            'a48c1.preclaim.' + label,
        )
        if (world.rng.bit_generator.state != before_rng
                or float(world.dissipated_energy) != before_energy
                or any(entry['event'] == 'replication_cpu'
                       for entry in scheduler._record(cell)[1]['events'])):
            raise AssertionError(label + ' crossed preclaim boundary')
    finally:
        _a48_abort_direct(scheduler, cell)


def test_a48c1_active_noncompletion_capacity_trust_rollback_scope_and_exact_once():
    base_case = _a48c_oracle_cases(seed=10141)[0]
    _, state, cell_id, dt, _, _ = base_case
    direct = a4.a3.s66.Formal066World.from_state(v3.pickle_clone(state))
    direct_cell = next(
        cell for cell in direct.cells if int(cell.cell_id) == cell_id
    )
    direct_cell._replicate_genome(direct, dt, direct.config)
    exact_values = _a48c_capacity_for_cell(direct_cell)
    exact = a4.GPU068A4Config(**exact_values)

    # The exact final arena is accepted. S-1 is specifically post-append
    # failure; Q/W/P one-short freeze every other fixed dimension.
    world = a4.a3.s66.Formal066World.from_state(v3.pickle_clone(state))
    cell = next(item for item in world.cells if int(item.cell_id) == cell_id)
    scheduler = a48c.A4ReplicationEventScheduler(exact, 'cpu')
    _a48c_begin_replication_direct(scheduler, world, cell, dt)
    try:
        scheduler.cpu_replication(world, cell, dt, world.config)
        _a48c_assert_oracle(
            direct, direct_cell, world, cell, 'a48c1.capacity_exact',
        )
    finally:
        _a48_abort_direct(scheduler, cell)

    for label, key in (
            ('Q', 'max_sequences'), ('S', 'max_symbols'),
            ('W', 'max_sequence_symbols'),
            ('P', 'max_proteins_per_cell')):
        values = dict(exact_values)
        values[key] -= 1
        _a48c_assert_preclaim_failure(
            state, cell_id, dt, a4.GPU068A4Config(**values),
            label='capacity_' + label,
        )

    mutation_case = next(
        value for value in _a48c_oracle_cases(seed=10151)
        if value[0] == 'substitution-hit'
    )
    _, mutation_state, mutation_cell_id, mutation_dt, _, _ = mutation_case

    if torch is None:
        raise AssertionError('PyTorch is required for A4.8c1 trust tests')
    if _REQUIRE_CUDA and not torch.cuda.is_available():
        raise AssertionError('CUDA required for A4.8c1 trust tests')
    resident_devices = ['cpu']
    if torch.cuda.is_available():
        resident_devices.append('cuda')
    trust_details = []

    def assert_candidate_tamper_failure(
            case_state, case_cell_id, case_dt, scheduler_type, label,
            devices=('cpu',), require_isolation=False):
        """A4.8b-style full preclaim audit for a protected c1 seam."""
        for device in devices:
            tamper_world = a4.a3.s66.Formal066World.from_state(
                v3.pickle_clone(case_state),
            )
            tamper_cell = next(
                item for item in tamper_world.cells
                if int(item.cell_id) == int(case_cell_id)
            )
            tamper_scheduler = scheduler_type(
                a4.GPU068A4Config(), device,
            )
            _a48c_begin_replication_direct(
                tamper_scheduler, tamper_world, tamper_cell, case_dt,
            )
            before_world = v3.pickle_clone(tamper_world.state_dict())
            before_rng = copy.deepcopy(
                tamper_world.rng.bit_generator.state,
            )
            before_objects = _a48c_identity_snapshot(
                tamper_world, tamper_cell,
            )
            before_config_object = id(tamper_world.config)
            before_receipts = len(tamper_scheduler._receipts)
            try:
                _assert_raises(
                    (a48c.A4ReplicationCommitError,
                     a4.A4SchemaError, a3s.A3SchedulerError),
                    lambda: tamper_scheduler.cpu_replication(
                        tamper_world, tamper_cell, case_dt,
                        tamper_world.config,
                    ),
                )
                expected_world = getattr(
                    tamper_scheduler, 'expected_world_state',
                    before_world,
                )
                expected_rng = getattr(
                    tamper_scheduler, 'expected_rng_state', before_rng,
                )
                v3.assert_recursive_close(
                    expected_world, tamper_world.state_dict(), 0.0, 0.0,
                    'a48c1.%s.%s.preclaim_world' % (device, label),
                )
                _, record = tamper_scheduler._record(tamper_cell)
                if (tamper_world.rng.bit_generator.state != expected_rng
                        or _a48c_identity_snapshot(
                            tamper_world, tamper_cell,
                        ) != before_objects
                        or id(tamper_world.config) != before_config_object
                        or len(tamper_scheduler._receipts) != before_receipts
                        or any(entry['event'] == 'replication_cpu'
                               for entry in record['events'])):
                    raise AssertionError(
                        '%s/%s tamper crossed preclaim boundary' % (
                            device, label,
                        )
                    )
                if (require_isolation
                        and not bool(getattr(
                            tamper_scheduler, 'isolated', False,
                        ))):
                    raise AssertionError(
                        '%s/%s retained artifact aliases live state' % (
                            device, label,
                        )
                    )
                trust_details.append(device + ':' + label)
            finally:
                _a48_abort_direct(tamper_scheduler, tamper_cell)

    class TamperPackedHostSource(a48c.A4ReplicationEventScheduler):
        def _replication_candidate_ready(
                self, world, cell, dt, config, candidate):
            retained = candidate.source_binding.state.pools
            self.isolated = (
                retained is not cell.pools
                and not np.shares_memory(retained, cell.pools)
            )
            retained[0, a4.a3.POOL_NUCLEOTIDE] += 1e-8
            return candidate

    class TamperHostOracle(a48c.A4ReplicationEventScheduler):
        def _replication_candidate_ready(
                self, world, cell, dt, config, candidate):
            candidate.host_replay.last_replication_symbols[0] += 1
            return candidate

    class TamperResidentRaggedValue(a48c.A4ReplicationEventScheduler):
        def _replication_candidate_ready(
                self, world, cell, dt, config, candidate):
            candidate.resident_binding.ragged.symbols.data[0].bitwise_xor_(1)
            return candidate

    class TamperResidentStateValue(a48c.A4ReplicationEventScheduler):
        def _replication_candidate_ready(
                self, world, cell, dt, config, candidate):
            candidate.resident_binding.state.pools.data[
                0, a4.a3.POOL_NUCLEOTIDE
            ].add_(1e-8)
            return candidate

    class TamperResidentCacheValue(a48c.A4ReplicationEventScheduler):
        def _replication_candidate_ready(
                self, world, cell, dt, config, candidate):
            candidate.resident_binding.cache.copy_numbers.data[0].add_(1)
            return candidate

    class TamperResidentPlanValue(a48c.A4ReplicationEventScheduler):
        def _replication_candidate_ready(
                self, world, cell, dt, config, candidate):
            candidate.resident_plan.append_symbols.data[0, 0].bitwise_xor_(1)
            return candidate

    class SwapResidentTensorPointer(a48c.A4ReplicationEventScheduler):
        def _replication_candidate_ready(
                self, world, cell, dt, config, candidate):
            candidate.resident_binding.state.pools = (
                candidate.resident_binding.state.pools.clone()
            )
            return candidate

    class SwapResidentWholeMember(a48c.A4ReplicationEventScheduler):
        def _replication_candidate_ready(
                self, world, cell, dt, config, candidate):
            object.__setattr__(
                candidate.resident_binding, 'ragged',
                candidate.resident_binding.ragged.clone(),
            )
            return candidate

    class TamperResidentPrivateSnapshot(
            a48c.A4ReplicationEventScheduler):
        def _replication_candidate_ready(
                self, world, cell, dt, config, candidate):
            name = next(iter(
                candidate.resident_artifacts['data_ptrs']['plan'],
            ))
            candidate.resident_artifacts['data_ptrs']['plan'][name] += 1
            return candidate

    class TamperSchedulerCapacityDevice(
            a48c.A4ReplicationEventScheduler):
        def _replication_candidate_ready(
                self, world, cell, dt, config, candidate):
            self.a4_config = a4.GPU068A4Config(
                max_cells=1, max_sequences=1, max_symbols=1,
                max_sequence_symbols=1, max_proteins_per_cell=1,
            )
            self.a4_device = (
                'cuda' if self.a4_device == 'cpu' else 'cpu'
            )
            return candidate

    class MoveResidentTensorDevice(a48c.A4ReplicationEventScheduler):
        def _replication_candidate_ready(
                self, world, cell, dt, config, candidate):
            tensor = candidate.resident_binding.ragged.symbols
            target = 'cuda' if tensor.device.type == 'cpu' else 'cpu'
            candidate.resident_binding.ragged.symbols = tensor.to(target)
            return candidate

    class TamperCandidateCell(a48c.A4ReplicationEventScheduler):
        def _replication_candidate_ready(
                self, world, cell, dt, config, candidate):
            candidate.candidate_cell.replication_copy[-1] ^= 1
            return candidate

    class TamperFreshBinding(a48c.A4ReplicationEventScheduler):
        def _replication_candidate_ready(
                self, world, cell, dt, config, candidate):
            candidate.fresh_binding.ragged.symbols[-1] ^= np.uint8(1)
            return candidate

    class MutateUnpackedCandidate(a48c.A4ReplicationEventScheduler):
        def _replication_candidate_ready(
                self, world, cell, dt, config, candidate):
            retained = candidate.candidate_cell.last_repair_flux
            self.isolated = (
                retained is not cell.last_repair_flux
                and not np.shares_memory(retained, cell.last_repair_flux)
            )
            retained[0] += 0.125
            return candidate

    class TamperLiveConfig(a48c.A4ReplicationEventScheduler):
        def _replication_candidate_ready(
                self, world, cell, dt, config, candidate):
            world.config.mutation_rate = float(np.nextafter(
                float(world.config.mutation_rate), np.inf,
            ))
            self.expected_world_state = v3.pickle_clone(
                world.state_dict(),
            )
            return candidate

    class TamperLiveSource(a48c.A4ReplicationEventScheduler):
        def _replication_candidate_ready(
                self, world, cell, dt, config, candidate):
            cell.replication_fractional = float(np.nextafter(
                float(cell.replication_fractional), np.inf,
            ))
            self.expected_world_state = v3.pickle_clone(
                world.state_dict(),
            )
            return candidate

    class TamperLiveRng(a48c.A4ReplicationEventScheduler):
        def _replication_candidate_ready(
                self, world, cell, dt, config, candidate):
            world.rng.random()
            self.expected_rng_state = copy.deepcopy(
                world.rng.bit_generator.state,
            )
            self.expected_world_state = v3.pickle_clone(
                world.state_dict(),
            )
            return candidate

    deterministic_cases = (
        (TamperPackedHostSource, 'host_source_packed_alias', True),
        (TamperHostOracle, 'host_oracle', False),
        (TamperResidentRaggedValue, 'resident_ragged_data_value', False),
        (TamperResidentStateValue, 'resident_state_data_value', False),
        (TamperResidentCacheValue, 'resident_cache_data_value', False),
        (TamperResidentPlanValue, 'resident_plan_data_value', False),
        (SwapResidentTensorPointer, 'resident_tensor_pointer', False),
        (SwapResidentWholeMember, 'resident_whole_member', False),
        (TamperResidentPrivateSnapshot, 'resident_private_snapshot', False),
        (MutateUnpackedCandidate, 'unpacked_candidate_alias', True),
    )
    for scheduler_type, label, require_isolation in deterministic_cases:
        assert_candidate_tamper_failure(
            state, cell_id, dt, scheduler_type, label,
            devices=resident_devices,
            require_isolation=require_isolation,
        )
    for scheduler_type, label in (
            (TamperSchedulerCapacityDevice, 'scheduler_capacity_device'),
            (TamperCandidateCell, 'candidate_cell'),
            (TamperFreshBinding, 'fresh_binding'),
            (TamperLiveConfig, 'live_config'),
            (TamperLiveSource, 'live_source')):
        assert_candidate_tamper_failure(
            state, cell_id, dt, scheduler_type, label,
        )
    if torch.cuda.is_available():
        assert_candidate_tamper_failure(
            state, cell_id, dt, MoveResidentTensorDevice,
            'resident_tensor_device', devices=resident_devices,
        )

    class TamperHostTape(a48c.A4ReplicationEventScheduler):
        def _replication_candidate_ready(
                self, world, cell, dt, config, candidate):
            candidate.tape.uniform_draws[0, 0] = np.nextafter(
                candidate.tape.uniform_draws[0, 0], np.inf,
            )
            return candidate

    class TamperCandidateRng(a48c.A4ReplicationEventScheduler):
        def _replication_candidate_ready(
                self, world, cell, dt, config, candidate):
            generator = np.random.Generator(np.random.PCG64())
            generator.bit_generator.state = copy.deepcopy(
                candidate.rng_after_state,
            )
            generator.random()
            candidate.rng_after_state = copy.deepcopy(
                generator.bit_generator.state,
            )
            return candidate

    class TamperResidentTapeValue(a48c.A4ReplicationEventScheduler):
        def _replication_candidate_ready(
                self, world, cell, dt, config, candidate):
            candidate.resident_tape.uniform_draws.data[0, 0].add_(0.125)
            return candidate

    for scheduler_type, label, devices in (
            (TamperHostTape, 'host_tape', ('cpu',)),
            (TamperCandidateRng, 'candidate_rng', ('cpu',)),
            (TamperLiveRng, 'live_rng', ('cpu',)),
            (TamperResidentTapeValue, 'resident_tape_data_value',
             resident_devices)):
        assert_candidate_tamper_failure(
            mutation_state, mutation_cell_id, mutation_dt,
            scheduler_type, label, devices=devices,
        )

    # Wrong cell/dt/config arguments are rejected against one prepared
    # private candidate without consuming it or crossing the claim.
    world = a4.a3.s66.Formal066World.from_state(v3.pickle_clone(state))
    cell = next(item for item in world.cells if int(item.cell_id) == cell_id)
    scheduler = a48c.A4ReplicationEventScheduler(
        a4.GPU068A4Config(), 'cpu',
    )
    _a48c_begin_replication_direct(scheduler, world, cell, dt)
    before = v3.pickle_clone(cell.state_dict())
    before_rng = copy.deepcopy(world.rng.bit_generator.state)
    try:
        candidate = scheduler._prepare_replication_candidate(
            world, cell, dt, world.config,
        )
        wrong_inputs = (
            ('dt', cell, np.nextafter(dt, np.inf), world.config),
            ('cell', copy.deepcopy(cell), dt, world.config),
            ('config', cell, dt, copy.deepcopy(world.config)),
        )
        for wrong_label, wrong_cell, wrong_dt, wrong_config in wrong_inputs:
            _assert_raises(
                a48c.A4ReplicationCommitError,
                lambda wrong_cell=wrong_cell, wrong_dt=wrong_dt,
                       wrong_config=wrong_config:
                scheduler._commit_replication_candidate(
                    world, wrong_cell, wrong_dt, wrong_config, candidate,
                ),
            )
            if candidate.consumed:
                raise AssertionError(
                    'wrong %s consumed the private candidate' % wrong_label
                )
        v3.assert_recursive_close(
            before, cell.state_dict(), 0.0, 0.0,
            'a48c1.wrong_cell_dt_config',
        )
        if (world.rng.bit_generator.state != before_rng
                or any(entry['event'] == 'replication_cpu'
                       for entry in scheduler._record(cell)[1]['events'])):
            raise AssertionError(
                'wrong cell/dt/config crossed the claim boundary'
            )
    finally:
        _a48_abort_direct(scheduler, cell)

    # Scope exclusions must never invoke the inherited CPU bridge.
    scope_cases = []
    start_world, start_cells, _, start_dt = _a45b_template_start_fixture(
        seed=10161,
    )
    _a48c_reduce_world(start_world, [start_cells[0]])
    scope_cases.append((
        'inactive_start', start_world.state_dict(),
        int(start_cells[0].cell_id), start_dt,
    ))
    completion_world, completion_cells, _, completion_dt = (
        _paid_completion_fixture(seed=10162)
    )
    _a48c_reduce_world(completion_world, [completion_cells[0]])
    scope_cases.append((
        'completion_off', completion_world.state_dict(),
        int(completion_cells[0].cell_id), completion_dt,
    ))
    mutation_world, mutation_cells, _, mutation_completion_dt = (
        _a46b1_completion_mutation_fixture(seed=10163)
    )
    _a48c_reduce_world(mutation_world, [mutation_cells[0]])
    scope_cases.append((
        'completion_on', mutation_world.state_dict(),
        int(mutation_cells[0].cell_id), mutation_completion_dt,
    ))
    disabled = a4.a3.s66.Formal066World.from_state(v3.pickle_clone(state))
    disabled.config.genome_replication = False
    scope_cases.append(('disabled', disabled.state_dict(), cell_id, dt))
    no_genome = a4.a3.s66.Formal066World.from_state(v3.pickle_clone(state))
    empty = no_genome.cells[0]
    empty.genomes = []
    empty.genome_lesions = []
    empty.replication_template = None
    empty.replication_copy = []
    empty.replication_template_lesion = 0.0
    empty.replication_fractional = 0.0
    empty._refresh_gene_cache()
    scope_cases.append(('no_genome', no_genome.state_dict(), cell_id, dt))
    gate = a4.a3.s66.Formal066World.from_state(v3.pickle_clone(state))
    gate.config.external_replicase = False
    gate_cell = gate.cells[0]
    for fingerprint, spec in list(gate_cell.gene_specs.items()):
        if int(spec['role']) == int(a4.g2.ROLE_REPLICASE):
            gate_cell.proteins.pop(fingerprint, None)
    gate_cell._sync_protein_pool()
    scope_cases.append(('replicase_gate', gate.state_dict(), cell_id, dt))
    negative = a4.a3.s66.Formal066World.from_state(v3.pickle_clone(state))
    negative.cells[0].pools[a4.a3.POOL_ATP] = -1e-12
    scope_cases.append(('negative_atp', negative.state_dict(), cell_id, dt))
    boundary = a4.a3.s66.Formal066World.from_state(v3.pickle_clone(state))
    boundary_dt = float.fromhex('0x1.c46204851f076p-4')
    scope_cases.append(('fp64_code6', boundary.state_dict(), cell_id, boundary_dt))

    legacy_called = []
    original_cpu_replication = a48b.A4TranslationEventScheduler.cpu_replication

    def legacy_replication_bomb(*args, **kwargs):
        legacy_called.append(True)
        raise AssertionError('legacy CPU replication fallback was called')

    a48b.A4TranslationEventScheduler.cpu_replication = legacy_replication_bomb
    try:
        for label, scope_state, scope_cell_id, scope_dt in scope_cases:
            _a48c_assert_preclaim_failure(
                scope_state, scope_cell_id, scope_dt,
                a4.GPU068A4Config(), label='scope_' + label,
            )
    finally:
        a48b.A4TranslationEventScheduler.cpu_replication = (
            original_cpu_replication
        )
    if legacy_called:
        raise AssertionError('an excluded scope used CPU fallback')

    # Duplicate and out-of-order calls remain atomic under the inherited
    # scheduler protocol.
    duplicate = a4.a3.s66.Formal066World.from_state(v3.pickle_clone(state))
    duplicate_cell = duplicate.cells[0]
    duplicate_scheduler = a48c.A4ReplicationEventScheduler(
        a4.GPU068A4Config(), 'cpu',
    )
    _a48c_begin_replication_direct(
        duplicate_scheduler, duplicate, duplicate_cell, dt,
    )
    duplicate_scheduler.cpu_replication(
        duplicate, duplicate_cell, dt, duplicate.config,
    )
    duplicate_before = v3.pickle_clone(duplicate_cell.state_dict())
    duplicate_rng = copy.deepcopy(duplicate.rng.bit_generator.state)
    _assert_raises(
        a3s.A3DuplicateEventError,
        lambda: duplicate_scheduler.cpu_replication(
            duplicate, duplicate_cell, dt, duplicate.config,
        ),
    )
    v3.assert_recursive_close(
        duplicate_before, duplicate_cell.state_dict(), 0.0, 0.0,
        'a48c1.duplicate',
    )
    if duplicate.rng.bit_generator.state != duplicate_rng:
        raise AssertionError('duplicate replication changed RNG')
    _a48_abort_direct(duplicate_scheduler, duplicate_cell)

    order = a4.a3.s66.Formal066World.from_state(v3.pickle_clone(state))
    order_cell = order.cells[0]
    order_scheduler = a48c.A4ReplicationEventScheduler(
        a4.GPU068A4Config(), 'cpu',
    )
    _a48b_begin_translation_direct(order_scheduler, order, order_cell, dt)
    order_before = v3.pickle_clone(order_cell.state_dict())
    order_rng = copy.deepcopy(order.rng.bit_generator.state)
    _assert_raises(
        a3s.A3EventOrderError,
        lambda: order_scheduler.cpu_replication(
            order, order_cell, dt, order.config,
        ),
    )
    v3.assert_recursive_close(
        order_before, order_cell.state_dict(), 0.0, 0.0,
        'a48c1.out_of_order',
    )
    if order.rng.bit_generator.state != order_rng:
        raise AssertionError('out-of-order replication changed RNG')
    _a48_abort_direct(order_scheduler, order_cell)

    class FailAfterPublish(a48c.A4ReplicationEventScheduler):
        def _publish_replication_candidate(self, world, cell, candidate):
            super(FailAfterPublish, self)._publish_replication_candidate(
                world, cell, candidate,
            )
            cell.genomes[0][0] ^= np.uint8(1)
            cell.replication_template[0] ^= np.uint8(1)
            first = next(iter(cell.gene_specs))
            cell.gene_specs[first]['promoter'] += 0.25
            cell.proteins[next(iter(cell.proteins))] += 0.125
            world.dissipated_energy += 1.0
            world.rng.random()
            raise RuntimeError('injected A4.8c1 publish failure')

    rollback = a4.a3.s66.Formal066World.from_state(
        v3.pickle_clone(mutation_state),
    )
    rollback_cell = rollback.cells[0]
    rollback_scheduler = FailAfterPublish(
        a4.GPU068A4Config(), 'cpu',
    )
    _a48c_begin_replication_direct(
        rollback_scheduler, rollback, rollback_cell, mutation_dt,
    )
    rollback_before = v3.pickle_clone(rollback_cell.state_dict())
    rollback_rng = copy.deepcopy(rollback.rng.bit_generator.state)
    rollback_energy = float(rollback.dissipated_energy)
    rollback_ids = _a48c_identity_snapshot(rollback, rollback_cell)
    caught = None
    try:
        rollback_scheduler.cpu_replication(
            rollback, rollback_cell, mutation_dt, rollback.config,
        )
    except RuntimeError as error:
        caught = error
    if caught is None:
        raise AssertionError('injected replication publish failure did not escape')
    v3.assert_recursive_close(
        rollback_before, rollback_cell.state_dict(), 0.0, 0.0,
        'a48c1.publish_rollback',
    )
    if (rollback.rng.bit_generator.state != rollback_rng
            or float(rollback.dissipated_energy) != rollback_energy
            or _a48c_identity_snapshot(rollback, rollback_cell)
            != rollback_ids):
        raise AssertionError('postclaim rollback lost identity/state')
    entry = _a48c_replication_entry(rollback_scheduler, rollback_cell)
    receipt = _a48_abort_direct(
        rollback_scheduler, rollback_cell, caught,
    )
    if (entry['event'] != 'replication_cpu'
            or receipt['status'] != 'aborted'):
        raise AssertionError('postclaim failure did not leave aborted receipt')
    return ('final Q/S/W/P exact + each one-short; %d explicit %s '
            'source/config/RNG/oracle/resident/pointer/device/alias trust '
            'checks; wrong cell/dt/config; excluded start/completion/no-op/'
            'code6 without fallback; duplicate/order; postclaim nested '
            'identity/RNG rollback' % (
                len(trust_details), '/'.join(resident_devices),
            ))


def test_a48c1_world_lockstep_interleave_save_clone_successor_and_authority():
    capacity = a4.GPU068A4Config()
    dt = None
    for steps, seed in ((1, 10181), (10, 10182)):
        source, cells, _, dt = _paid_replication_b_fixture(seed=seed)
        _a48c_reduce_world(source, [cells[0]])
        state = v3.pickle_clone(source.state_dict())
        cpu = a4.a3.s66.Formal066World.from_state(v3.pickle_clone(state))
        hybrid = _a48c_hybrid_from_state(state, capacity, 'cpu')
        for index in range(steps):
            for label, cell in (
                    ('cpu', cpu.cells[0]),
                    ('hybrid', hybrid.world.cells[0])):
                if (cell.replication_template is None
                        or len(cell.replication_copy)
                        >= len(cell.replication_template)):
                    raise AssertionError(
                        '%s left c1 scope before step %d' % (label, index)
                    )
            cpu.step(dt)
            hybrid.step(dt)
        v3._assert_world_pair(
            cpu, hybrid, state_atol=v3.WORLD_FP64_ATOL,
            ledger_atol=v3.LEDGER_ATOL,
            label='a48c1.lockstep_%d' % steps,
        )

    interleave_source, interleave_cells, _, interleave_dt = (
        _a45_substitution_fixture(seed=10183)
    )
    interleave_source.config.mutation_rate = 1.0
    interleave_source.config.endogenous_damage = False
    for cell in interleave_cells[:2]:
        cell.genome_lesions = [
            0.5 / 0.00065 for _ in cell.genomes
        ]
        cell._refresh_gene_cache()
    _a48c_reduce_world(interleave_source, interleave_cells[:2])
    interleave_state = v3.pickle_clone(interleave_source.state_dict())
    devices = ['cpu']
    if torch is not None and torch.cuda.is_available():
        devices.append('cuda')
    elif _REQUIRE_CUDA:
        raise AssertionError('CUDA required for A4.8c1 world interleave')
    interleave_details = []
    original_cpu_replication = a48b.A4TranslationEventScheduler.cpu_replication

    def legacy_replication_bomb(*args, **kwargs):
        raise AssertionError('legacy CPU replication bridge was called')

    for device in devices:
        cpu = a4.a3.s66.Formal066World.from_state(
            v3.pickle_clone(interleave_state),
        )
        hybrid = _a48c_hybrid_from_state(
            interleave_state, capacity, device,
        )
        cpu.step(interleave_dt)
        a48b.A4TranslationEventScheduler.cpu_replication = (
            legacy_replication_bomb
        )
        try:
            receipt = hybrid.step(interleave_dt)
        finally:
            a48b.A4TranslationEventScheduler.cpu_replication = (
                original_cpu_replication
            )
        v3._assert_world_pair(
            cpu, hybrid, state_atol=v3.WORLD_FP64_ATOL,
            ledger_atol=v3.LEDGER_ATOL,
            label='a48c1.interleave.' + device,
        )
        previous_motion = None
        for record in receipt['cells']:
            by_name = {entry['event']: entry for entry in record['events']}
            names = [entry['event'] for entry in record['events']]
            if (names.count('translation_cpu') != 1
                    or names.count('replication_cpu') != 1
                    or names.count('genome_hydrolysis_cpu_rng') != 1):
                raise AssertionError(
                    device + ' integrated event count differs'
                )
            translation = by_name['translation_cpu']['ordinal']
            replication = by_name['replication_cpu']['ordinal']
            surface = by_name['surface_assembly']['ordinal']
            hydrolysis = by_name['genome_hydrolysis_cpu_rng']['ordinal']
            motion = by_name['motion']['ordinal']
            if not translation < replication < surface < hydrolysis < motion:
                raise AssertionError(device + ' per-cell event order differs')
            if previous_motion is not None and previous_motion >= translation:
                raise AssertionError(device + ' cells were batched/reordered')
            previous_motion = motion
            metadata = by_name['replication_cpu']['metadata']
            if (metadata.get('authority')
                    != 'A4.8c1-resident-plan-atomic-cpu-cell-rng-commit'
                    or metadata.get('branch')
                    != 'active-noncompletion-substitution'
                    or int(metadata.get('amount', 0)) <= 0
                    or int(metadata.get('rng_call_count', 0)) <= 0
                    or by_name['genome_hydrolysis_cpu_rng']['metadata'].get(
                        'eligible_genome_count', 0
                    ) < 1):
                raise AssertionError(device + ' interleave work differs')
        interleave_details.append(device)

    persisted_source, persisted_cells, _, persisted_dt = (
        _paid_replication_b_fixture(seed=10184)
    )
    _a48c_reduce_world(persisted_source, [persisted_cells[0]])
    persisted = _a48c_hybrid_from_state(
        persisted_source.state_dict(), capacity, 'cpu',
    )
    persisted.step(persisted_dt)
    if (persisted.world.cells[0].replication_template is None
            or len(persisted.world.cells[0].replication_copy)
            >= len(persisted.world.cells[0].replication_template)):
        raise AssertionError('persisted c1 fixture completed unexpectedly')
    twin = persisted.clone()
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, 'a48c1.pkl')
        persisted.save(path)
        restored = a48c.Hybrid066WorldA4Replication.load(path)
        for value in (twin, restored):
            if (type(value) is not a48c.Hybrid066WorldA4Replication
                    or type(value.scheduler)
                    is not a48c.A4ReplicationEventScheduler
                    or value.scheduler.a4_device != 'cpu'
                    or value.scheduler.a4_config != capacity):
                raise AssertionError('A4.8c1 save/clone authority was lost')
        persisted.step(persisted_dt)
        twin.step(persisted_dt)
        restored.step(persisted_dt)
        v3.assert_recursive_close(
            persisted.world.state_dict(), twin.world.state_dict(),
            v3.WORLD_FP64_ATOL, 0.0, 'a48c1.clone_continuation',
        )
        v3.assert_recursive_close(
            persisted.world.state_dict(), restored.world.state_dict(),
            v3.WORLD_FP64_ATOL, 0.0, 'a48c1.save_continuation',
        )

    active_source, active_cells, _, _ = _paid_replication_b_fixture(
        seed=10185,
    )
    _a48c_reduce_world(active_source, [active_cells[0]])
    active = _a48c_hybrid_from_state(
        active_source.state_dict(), capacity, 'cpu',
    )
    active.scheduler.begin_step(active.world, active.world.cells)
    _assert_raises(a3s.A3SchedulerProtocolError, active.clone)
    active.scheduler.abort_step(RuntimeError('expected active-save rejection'))
    active.scheduler._a4_replication_commit_active = True
    try:
        _assert_raises(a3s.A3SchedulerProtocolError, active.state_dict)
    finally:
        active.scheduler._a4_replication_commit_active = False

    # Freeze the immediate frozen successor gate. One paid symbol leaves ATP
    # at exact 0.045, its immediate neighbours, then surface assembly must take
    # the same zero/positive branch after CPU or CUDA resident replication.
    surface_details = []
    surface_targets = (
        ('nextbelow', np.nextafter(np.float64(0.045), 0.0), False),
        ('exact', np.float64(0.045), False),
        ('nextabove', np.nextafter(np.float64(0.045), np.inf), True),
    )
    atp_cost = float(a4.g2.REPLICATION_ATP_PER_SYMBOL)
    for device in devices:
        for label, target, expected_work in surface_targets:
            source, cells, _, surface_dt = _paid_replication_fixture(
                seed=10190,
            )
            cell = cells[0]
            cell.pools[a4.a3.POOL_ATP] = float(target + atp_cost)
            cell.pools[a4.a3.POOL_MEM_PRECURSOR] = 0.1
            cell.pools[a4.a3.POOL_TRANSPORTER_PRECURSOR] = 0.0
            _a48c_reduce_world(source, [cell])
            state = v3.pickle_clone(source.state_dict())
            direct = a4.a3.s66.Formal066World.from_state(
                v3.pickle_clone(state),
            )
            integrated = a4.a3.s66.Formal066World.from_state(
                v3.pickle_clone(state),
            )
            direct_cell = direct.cells[0]
            integrated_cell = integrated.cells[0]
            direct_cell._replicate_genome(
                direct, surface_dt, direct.config,
            )
            if (int(direct_cell.last_replication_symbols) != 1
                    or np.float64(
                        direct_cell.pools[a4.a3.POOL_ATP]
                    ).view(np.uint64) != target.view(np.uint64)):
                raise AssertionError(label + ' frozen ATP seam drifted')
            scheduler = a48c.A4ReplicationEventScheduler(capacity, device)
            _a48c_begin_replication_direct(
                scheduler, integrated, integrated_cell, surface_dt,
            )
            try:
                scheduler.cpu_replication(
                    integrated, integrated_cell, surface_dt,
                    integrated.config,
                )
                _a48c_assert_oracle(
                    direct, direct_cell, integrated, integrated_cell,
                    'a48c1.surface_pre.%s.%s' % (device, label),
                )
                scheduler.claim(
                    integrated_cell, 'surface_assembly',
                    metadata={'validation': 'A4.8c1-successor'},
                )
                direct_cell._assemble_surface(surface_dt, direct.config)
                integrated_cell._assemble_surface(
                    surface_dt, integrated.config,
                )
                _a48b_assert_cell_matches(
                    direct_cell, integrated_cell,
                    'a48c1.surface_post.%s.%s' % (device, label),
                    atol=a48c.REPLICATION_ORACLE_ATOL,
                )
                actual_work = float(integrated_cell.last_assembly) > 0.0
                if actual_work != expected_work:
                    raise AssertionError(
                        '%s/%s successor ATP branch differs' % (
                            device, label,
                        )
                    )
                surface_details.append(device + ':' + label)
            finally:
                _a48_abort_direct(scheduler, integrated_cell)

    signature = inspect.signature(
        a48c.A4ReplicationEventScheduler.cpu_replication,
    )
    expected_status = (
        'a4.8c1-pre-existing-active-noncompletion-mutation-off-on-'
        'resident-plan-cpu-cell-pcg64-atomic-commit-'
        'inactive-completion-early-noop-fail-closed'
    )
    if (tuple(signature.parameters)
            != ('self', 'world', 'cell', 'dt', 'config')
            or '_A4ReplicationCommitCandidate' in set(a48c.__all__)
            or not issubclass(
                a48c.A4ReplicationEventScheduler,
                a48b.A4TranslationEventScheduler,
            )
            or not issubclass(
                a48c.Hybrid066WorldA4Replication,
                a48b.Hybrid066WorldA4Translation,
            )
            or a48c.PORT_STATUS.get('genome_replication') != expected_status
            or a48c.FULL_GPU_WORLD_STEP is not False
            or a48c.PORT_STATUS.get('full_gpu_world_step') is not False):
        raise AssertionError('A4.8c1 API/scope/authority differs')
    for relative, expected in PROMOTED_A3_SHA256.items():
        path = os.path.join(ROOT, *relative.split('/'))
        if _sha256(path) != expected:
            raise AssertionError(relative + ' changed from promoted A3 bytes')
    return ('1/10-step active lockstep; %s two-cell nonbatched '
            'translation/replication/surface/hydrolysis/motion PCG64 order; '
            'legacy bridge bomb; save/load/clone + active reject; successor '
            'ATP gates %s; promoted A3 and full_gpu=false' % (
                '/'.join(interleave_details), '/'.join(surface_details),
            ))


def _a48c2_completion_case(seed=11101, row=0):
    world, cells, _, dt = _paid_completion_fixture(seed=seed)
    cell = cells[int(row)]
    world.config.mutation = False
    _a48c_reduce_world(world, [cell])
    return (
        v3.pickle_clone(world.state_dict()), int(cell.cell_id), float(dt),
    )


def _a48c2_hybrid_from_state(state, capacity=None, device='cpu'):
    world = a4.a3.s66.Formal066World.from_state(v3.pickle_clone(state))
    backend = a4.a3.TorchKernelBackendA3(
        v3._a3_config(device=device, precision='float64'),
    )
    return a48c2.Hybrid066WorldA4ReplicationCompletion(
        world, backend=backend,
        a4_config=capacity or a4.GPU068A4Config(),
        a4_device=device,
    )


def _a48c2_assert_completion_oracle(expected_world, expected_cell,
                                     actual_world, actual_cell, label):
    _a48b_assert_cell_matches(
        expected_cell, actual_cell, label,
        atol=a48c2.REPLICATION_ORACLE_ATOL,
    )
    if (expected_world.rng.bit_generator.state
            != actual_world.rng.bit_generator.state
            or actual_cell.replication_template is not None
            or actual_cell.replication_copy
            or float(actual_cell.replication_template_lesion) != 0.0
            or float(actual_cell.replication_fractional) != 0.0):
        raise AssertionError(label + ' completion/RNG state differs')
    v3.assert_recursive_close(
        float(expected_world.dissipated_energy),
        float(actual_world.dissipated_energy),
        atol=0.0, rtol=0.0, path=label + '.dissipated_energy',
    )


def _a48c2_capacity_for_transition(before, after):
    def dimensions(cell):
        sequences = list(cell.genomes)
        if cell.replication_template is not None:
            sequences.extend((
                cell.replication_template, cell.replication_copy,
            ))
        q_value = len(sequences)
        s_value = sum(len(value) for value in sequences)
        w_value = max([len(value) for value in sequences] or [1])
        gene_keys = set(int(value) for value in cell.gene_specs)
        p_value = max(
            len(gene_keys.union(int(value) for value in cell.proteins)),
            len(gene_keys.union(
                int(value) for value in cell.damaged_proteins
            )),
            1,
        )
        return q_value, s_value, w_value, p_value

    left = dimensions(before)
    right = dimensions(after)
    return {
        'max_cells': 1,
        'max_sequences': max(left[0], right[0]),
        'max_symbols': max(left[1], right[1]),
        'max_sequence_symbols': max(left[2], right[2]),
        'max_proteins_per_cell': max(left[3], right[3]),
    }


def test_a48c2_mutation_free_completion_atomic_commit_formal066_oracle():
    if torch is None:
        raise AssertionError('PyTorch is required for A4.8c2')
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    elif _REQUIRE_CUDA:
        raise AssertionError('CUDA required but unavailable; fallback forbidden')

    class ObserveCompletion(a48c2.A4ReplicationCompletionEventScheduler):
        def _replication_completion_candidate_ready(
                self, world, cell, dt, config, candidate):
            self.observed_plan = candidate.plan.clone()
            return candidate

    details = []
    for device in devices:
        for row in (0, 1):
            state, cell_id, dt = _a48c2_completion_case(
                seed=11101 + row, row=row,
            )
            direct = a4.a3.s66.Formal066World.from_state(
                v3.pickle_clone(state),
            )
            world = a4.a3.s66.Formal066World.from_state(
                v3.pickle_clone(state),
            )
            direct_cell = direct.cells[0]
            cell = world.cells[0]
            partial = np.asarray(cell.replication_copy, dtype=np.uint8).copy()
            template = np.asarray(
                cell.replication_template, dtype=np.uint8,
            ).copy()
            cycles_before = int(cell.replication_cycles)
            rng_before = copy.deepcopy(world.rng.bit_generator.state)
            identities = _a48c_identity_snapshot(world, cell)

            direct_cell._replicate_genome(direct, dt, direct.config)
            if (direct_cell.replication_template is not None
                    or len(direct_cell.genomes) != 2):
                raise AssertionError('Formal066 completion fixture drifted')

            scheduler = ObserveCompletion(a4.GPU068A4Config(), device)
            _a48c_begin_replication_direct(scheduler, world, cell, dt)
            try:
                scheduler.cpu_replication(world, cell, dt, world.config)
                _a48c2_assert_completion_oracle(
                    direct, direct_cell, world, cell,
                    'a48c2.oracle.%s.%d' % (device, row),
                )
                plan = scheduler.observed_plan
                length = int(plan.completed_lengths[0])
                appended = int(plan.append_count[0])
                completed = np.asarray(
                    plan.completed_symbols[0, :length], dtype=np.uint8,
                )
                suffix = np.asarray(
                    plan.append_symbols[0, :appended], dtype=np.uint8,
                )
                if (appended <= 0
                        or not np.array_equal(completed[:len(partial)], partial)
                        or not np.array_equal(completed[len(partial):], suffix)
                        or not np.array_equal(suffix, template[len(partial):])
                        or int(cell.replication_cycles) != cycles_before + 1
                        or world.rng.bit_generator.state != rng_before):
                    raise AssertionError(
                        '%s row %d completion payload/ledger differs' % (
                            device, row,
                        )
                    )
                current_ids = _a48c_identity_snapshot(world, cell)
                for key in ('rng', 'pools', 'genomes', 'lesions', 'copy',
                            'events', 'proteins', 'damaged', 'gene_specs'):
                    if current_ids[key] != identities[key]:
                        raise AssertionError(
                            '%s row %d changed %s object identity' % (
                                device, row, key,
                            )
                        )
                if current_ids['genome_items'][:len(
                        identities['genome_items'])] != identities['genome_items']:
                    raise AssertionError('existing genome identity changed')
                entry = _a48c_replication_entry(scheduler, cell)
                metadata = entry['metadata']
                if (metadata.get('authority') != (
                        'A4.8c2-resident-completion-plan-atomic-cpu-cell-commit')
                        or metadata.get('branch')
                        != 'active-completion-deterministic'
                        or int(metadata.get('completion_events', 0)) != 1
                        or int(metadata.get('rng_call_count', -1)) != 0
                        or int(metadata.get('amount', 0)) != appended):
                    raise AssertionError('A4.8c2 completion receipt differs')
                details.append('%s:%d:%d' % (device, row, appended))
            finally:
                _a48_abort_direct(scheduler, cell)
    return ('Formal066 mutation-free completion payload/lesion/cycle/reset/'
            'cache/novel-path, object identity and PCG64 no-advance exact: ' +
            ', '.join(details))


def test_a48c2_completion_candidate_cpu_cuda_purity_and_fresh_binding():
    if torch is None:
        raise AssertionError('PyTorch is required for A4.8c2')
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    elif _REQUIRE_CUDA:
        raise AssertionError('CUDA required but unavailable; fallback forbidden')
    details = []
    for device in devices:
        state, cell_id, dt = _a48c2_completion_case(seed=11121)
        world = a4.a3.s66.Formal066World.from_state(v3.pickle_clone(state))
        cell = world.cells[0]
        scheduler = a48c2.A4ReplicationCompletionEventScheduler(
            a4.GPU068A4Config(), device,
        )
        _a48c_begin_replication_direct(scheduler, world, cell, dt)
        before = v3.pickle_clone(cell.state_dict())
        before_rng = copy.deepcopy(world.rng.bit_generator.state)
        before_energy = float(world.dissipated_energy)
        try:
            candidate = scheduler._prepare_completion_candidate(
                world, cell, dt, world.config,
            )
            v3.assert_recursive_close(
                before, cell.state_dict(), 0.0, 0.0,
                'a48c2.prepare_purity.' + device,
            )
            if (world.rng.bit_generator.state != before_rng
                    or float(world.dissipated_energy) != before_energy
                    or candidate.candidate_cell is cell
                    or candidate.plan is candidate.host_replay
                    or candidate.resident_plan.pools_after.device.type != device
                    or candidate.resident_binding.ragged.symbols.device.type
                    != device):
                raise AssertionError(device + ' candidate authority/purity differs')
            a4._require_translation_binding(candidate.fresh_binding)
            a48c._plan_semantic_match(
                candidate.plan, candidate.host_replay,
            )
            direct = a4.a3.s66.Formal066World.from_state(
                v3.pickle_clone(state),
            )
            direct.cells[0]._replicate_genome(direct, dt, direct.config)
            _a48b_assert_cell_matches(
                direct.cells[0], candidate.candidate_cell,
                'a48c2.candidate.' + device,
                atol=a48c2.REPLICATION_ORACLE_ATOL,
            )
            scheduler._commit_completion_candidate(
                world, cell, dt, world.config, candidate,
            )
            _a48c2_assert_completion_oracle(
                direct, direct.cells[0], world, cell,
                'a48c2.commit.' + device,
            )
            committed = v3.pickle_clone(cell.state_dict())
            committed_rng = copy.deepcopy(world.rng.bit_generator.state)
            _assert_raises(
                (a4.A4Error, a3s.A3SchedulerError),
                lambda: scheduler._commit_completion_candidate(
                    world, cell, dt, world.config, candidate,
                ),
            )
            v3.assert_recursive_close(
                committed, cell.state_dict(), 0.0, 0.0,
                'a48c2.one_shot.' + device,
            )
            if world.rng.bit_generator.state != committed_rng:
                raise AssertionError(device + ' reused candidate changed RNG')
            details.append(device)
        finally:
            _a48_abort_direct(scheduler, cell)
    return ('CPU/CUDA resident authority, independent NumPy oracle, prepare '
            'purity, fresh binding and one-shot candidate PASS: ' +
            '/'.join(details))


def test_a48c2_completion_capacity_trust_rollback_scope_and_exact_once():
    state, cell_id, dt = _a48c2_completion_case(seed=11141)
    source = a4.a3.s66.Formal066World.from_state(v3.pickle_clone(state))
    direct = a4.a3.s66.Formal066World.from_state(v3.pickle_clone(state))
    direct.cells[0]._replicate_genome(direct, dt, direct.config)
    exact_values = _a48c2_capacity_for_transition(
        source.cells[0], direct.cells[0],
    )
    exact = a4.GPU068A4Config(**exact_values)

    for device in (['cpu', 'cuda'] if torch is not None
                   and torch.cuda.is_available() else ['cpu']):
        world = a4.a3.s66.Formal066World.from_state(v3.pickle_clone(state))
        cell = world.cells[0]
        scheduler = a48c2.A4ReplicationCompletionEventScheduler(exact, device)
        _a48c_begin_replication_direct(scheduler, world, cell, dt)
        try:
            scheduler.cpu_replication(world, cell, dt, world.config)
            _a48c2_assert_completion_oracle(
                direct, direct.cells[0], world, cell,
                'a48c2.capacity_exact.' + device,
            )
        finally:
            _a48_abort_direct(scheduler, cell)

    for label, key in (
            ('Q', 'max_sequences'), ('S', 'max_symbols'),
            ('W', 'max_sequence_symbols'), ('P', 'max_proteins_per_cell')):
        values = dict(exact_values)
        values[key] -= 1
        _a48c_assert_preclaim_failure(
            state, cell_id, dt, a4.GPU068A4Config(**values),
            scheduler_type=a48c2.A4ReplicationCompletionEventScheduler,
            label='a48c2.capacity_' + label,
        )

    class TamperResidentPlan(a48c2.A4ReplicationCompletionEventScheduler):
        def _replication_completion_candidate_ready(
                self, world, cell, dt, config, candidate):
            candidate.resident_plan.completed_symbols.data[0, 0].bitwise_xor_(1)
            return candidate

    class TamperHostReplay(a48c2.A4ReplicationCompletionEventScheduler):
        def _replication_completion_candidate_ready(
                self, world, cell, dt, config, candidate):
            candidate.host_replay.completed_symbols[0, 0] ^= np.uint8(1)
            return candidate

    class TamperCandidate(a48c2.A4ReplicationCompletionEventScheduler):
        def _replication_completion_candidate_ready(
                self, world, cell, dt, config, candidate):
            candidate.candidate_cell.membrane[0] += 0.125
            return candidate

    class SwapResidentCache(a48c2.A4ReplicationCompletionEventScheduler):
        def _replication_completion_candidate_ready(
                self, world, cell, dt, config, candidate):
            object.__setattr__(
                candidate.resident_binding, 'cache',
                candidate.resident_binding.cache.clone(),
            )
            return candidate

    trust_types = (
        TamperResidentPlan, TamperHostReplay, TamperCandidate,
        SwapResidentCache,
    )
    for scheduler_type in trust_types:
        _a48c_assert_preclaim_failure(
            state, cell_id, dt, exact,
            scheduler_type=scheduler_type,
            label='a48c2.trust.' + scheduler_type.__name__,
        )

    # Wrong dt is checked against the prepared one-shot candidate itself.
    world = a4.a3.s66.Formal066World.from_state(v3.pickle_clone(state))
    cell = world.cells[0]
    scheduler = a48c2.A4ReplicationCompletionEventScheduler(exact, 'cpu')
    _a48c_begin_replication_direct(scheduler, world, cell, dt)
    try:
        candidate = scheduler._prepare_completion_candidate(
            world, cell, dt, world.config,
        )
        before = v3.pickle_clone(cell.state_dict())
        before_rng = copy.deepcopy(world.rng.bit_generator.state)
        _assert_raises(
            (a4.A4Error, a3s.A3SchedulerError),
            lambda: scheduler._commit_completion_candidate(
                world, cell, np.nextafter(dt, np.inf),
                world.config, candidate,
            ),
        )
        v3.assert_recursive_close(
            before, cell.state_dict(), 0.0, 0.0, 'a48c2.wrong_dt',
        )
        if world.rng.bit_generator.state != before_rng:
            raise AssertionError('wrong-dt rejection changed RNG')
    finally:
        _a48_abort_direct(scheduler, cell)

    class FailingPublish(a48c2.A4ReplicationCompletionEventScheduler):
        def _publish_replication_completion_candidate(
                self, world, cell, candidate):
            super(FailingPublish, self)._publish_replication_completion_candidate(
                world, cell, candidate,
            )
            cell.age += 9.0
            world.rng.random()
            raise RuntimeError('injected A4.8c2 publish failure')

    world = a4.a3.s66.Formal066World.from_state(v3.pickle_clone(state))
    cell = world.cells[0]
    scheduler = FailingPublish(exact, 'cpu')
    _a48c_begin_replication_direct(scheduler, world, cell, dt)
    before = v3.pickle_clone(cell.state_dict())
    before_ids = _a48c_identity_snapshot(world, cell)
    before_rng = copy.deepcopy(world.rng.bit_generator.state)
    try:
        _assert_raises(
            RuntimeError,
            lambda: scheduler.cpu_replication(world, cell, dt, world.config),
        )
        v3.assert_recursive_close(
            before, cell.state_dict(), 0.0, 0.0, 'a48c2.rollback',
        )
        after_ids = _a48c_identity_snapshot(world, cell)
        if after_ids != before_ids or world.rng.bit_generator.state != before_rng:
            raise AssertionError('A4.8c2 publish rollback identity/RNG differs')
    finally:
        _a48_abort_direct(scheduler, cell)

    for label, mutate in (
            ('mutation_on', lambda world, cell: setattr(
                world.config, 'mutation', True)),
            ('inactive', lambda world, cell: (
                setattr(cell, 'replication_template', None),
                setattr(cell, 'replication_copy', []))),
            ('disabled', lambda world, cell: setattr(
                world.config, 'genome_replication', False))):
        scoped = a4.a3.s66.Formal066World.from_state(v3.pickle_clone(state))
        scoped_cell = scoped.cells[0]
        mutate(scoped, scoped_cell)
        scoped.initial_total_material = scoped.total_material()
        scoped_state = v3.pickle_clone(scoped.state_dict())
        _a48c_assert_preclaim_failure(
            scoped_state, int(scoped_cell.cell_id), dt, exact,
            scheduler_type=a48c2.A4ReplicationCompletionEventScheduler,
            label='a48c2.scope.' + label,
        )
    return ('Q/S/W/P exact and one-short; resident/host/candidate trust; '
            'wrong-dt; declared identity/RNG rollback; mutation/inactive/disabled '
            'scope fail-closed PASS')


def test_a48c2_inherited_c1_save_clone_successor_and_authority():
    devices = ['cpu']
    if torch is not None and torch.cuda.is_available():
        devices.append('cuda')
    elif _REQUIRE_CUDA:
        raise AssertionError('CUDA required but unavailable; fallback forbidden')

    # Noncompletion remains explicitly delegated to the inherited c1 bridge.
    source, cells, _, dt = _paid_replication_b_fixture(seed=11161)
    _a48c_reduce_world(source, [cells[0]])
    noncompletion_state = v3.pickle_clone(source.state_dict())
    delegated = []
    for device in devices:
        direct = a4.a3.s66.Formal066World.from_state(
            v3.pickle_clone(noncompletion_state),
        )
        world = a4.a3.s66.Formal066World.from_state(
            v3.pickle_clone(noncompletion_state),
        )
        direct.cells[0]._replicate_genome(direct, dt, direct.config)
        scheduler = a48c2.A4ReplicationCompletionEventScheduler(
            a4.GPU068A4Config(), device,
        )
        _a48c_begin_replication_direct(scheduler, world, world.cells[0], dt)
        try:
            scheduler.cpu_replication(
                world, world.cells[0], dt, world.config,
            )
            _a48c_assert_oracle(
                direct, direct.cells[0], world, world.cells[0],
                'a48c2.delegate.' + device,
            )
            entry = _a48c_replication_entry(scheduler, world.cells[0])
            if entry['metadata'].get('authority') != (
                    'A4.8c1-resident-plan-atomic-cpu-cell-rng-commit'):
                raise AssertionError(device + ' did not delegate to A4.8c1')
            delegated.append(device)
        finally:
            _a48_abort_direct(scheduler, world.cells[0])

    # Freeze wrapper/scheduler persistence before the one supported completion.
    completion_state, _, completion_dt = _a48c2_completion_case(seed=11162)
    capacity = a4.GPU068A4Config()
    hybrid = _a48c2_hybrid_from_state(completion_state, capacity, 'cpu')
    twin = hybrid.clone()
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, 'a48c2.pkl')
        hybrid.save(path)
        restored = a48c2.Hybrid066WorldA4ReplicationCompletion.load(path)
        for value in (hybrid, twin, restored):
            if (type(value) is not a48c2.Hybrid066WorldA4ReplicationCompletion
                    or type(value.scheduler)
                    is not a48c2.A4ReplicationCompletionEventScheduler
                    or value.scheduler.a4_device != 'cpu'
                    or value.scheduler.a4_config != capacity):
                raise AssertionError('A4.8c2 save/clone authority was lost')
        cpu = a4.a3.s66.Formal066World.from_state(
            v3.pickle_clone(completion_state),
        )
        cpu.step(completion_dt)
        receipts = []
        for value in (hybrid, twin, restored):
            receipts.append(value.step(completion_dt))
            v3._assert_world_pair(
                cpu, value, state_atol=v3.WORLD_FP64_ATOL,
                ledger_atol=v3.LEDGER_ATOL,
                label='a48c2.persisted_completion',
            )
        for receipt in receipts:
            record = receipt['cells'][0]
            names = [entry['event'] for entry in record['events']]
            by_name = {entry['event']: entry for entry in record['events']}
            if (names.count('replication_cpu') != 1
                    or not names.index('translation_cpu')
                    < names.index('replication_cpu')
                    < names.index('surface_assembly')
                    or by_name['replication_cpu']['metadata'].get('authority')
                    != ('A4.8c2-resident-completion-plan-atomic-'
                        'cpu-cell-commit')):
                raise AssertionError('A4.8c2 successor order/authority differs')

    active = _a48c2_hybrid_from_state(completion_state, capacity, 'cpu')
    active.scheduler.begin_step(active.world, active.world.cells)
    _assert_raises(a3s.A3SchedulerProtocolError, active.clone)
    active.scheduler.abort_step(RuntimeError('expected active-save rejection'))
    active.scheduler._a4_replication_completion_commit_active = True
    try:
        _assert_raises(a3s.A3SchedulerProtocolError, active.state_dict)
    finally:
        active.scheduler._a4_replication_completion_commit_active = False

    expected_status = (
        'a4.8c2-mutation-free-pre-existing-active-completion-'
        'resident-descriptor-cpu-cell-atomic-commit-'
        'a4.8c1-active-noncompletion-inherited-'
        'mutation-completion-inactive-early-noop-fail-closed'
    )
    if (a48c2.PORT_STATUS.get('genome_replication') != expected_status
            or a48c2.FULL_GPU_WORLD_STEP is not False
            or not issubclass(
                a48c2.A4ReplicationCompletionEventScheduler,
                a48c.A4ReplicationEventScheduler,
            )
            or not issubclass(
                a48c2.Hybrid066WorldA4ReplicationCompletion,
                a48c.Hybrid066WorldA4Replication,
            )):
        raise AssertionError('A4.8c2 scope/API/full-GPU authority differs')
    for relative, expected in PROMOTED_A3_SHA256.items():
        if _sha256(os.path.join(ROOT, *relative.split('/'))) != expected:
            raise AssertionError(relative + ' changed from promoted A3 bytes')
    return ('inherited c1 delegation %s; completion save/load/clone type; '
            'translation<replication<surface successor order; active save '
            'reject; promoted A3 byte authority/full_gpu=false PASS' %
            '/'.join(delegated))


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
    test_a44_paid_replication_formal066_cpu_oracle_and_nonmutation,
    test_a44_requested_zero_and_exact_resource_boundaries,
    test_a44_paid_replication_numpy_torch_devices_and_scope_readback,
    test_a44_capacity_scope_fail_closed_and_a3_authority,
    test_a44b_feature_matrix_formal066_cpu_oracle,
    test_a44b_proof_ledger_resource_error_and_nonmutation,
    test_a44b_numpy_torch_deterministic_devices_and_pointers,
    test_a44b_capacity_scope_and_a3_authority,
    test_a45_substitution_rng_tape_formal066_oracle,
    test_a45_substitution_rng_tape_numpy_torch_devices,
    test_a45_substitution_rng_tape_fail_closed_and_authority,
    test_a45b_template_start_formal066_oracle_and_rng_order,
    test_a45b_template_start_exact_capacity_and_zero_dt,
    test_a45b_template_start_numpy_torch_trust_and_scope,
    test_a46a_completion_formal066_oracle_and_conceptual_poststate,
    test_a46a_completion_numpy_torch_devices_and_nonmutation,
    test_a46a_completion_scope_schema_and_a3_authority,
    test_a46b1_completion_mutation_rng_tape_formal066_oracle,
    test_a46b1_completion_mutation_rng_tape_torch_roundtrip,
    test_a46b1_completion_mutation_rng_tape_fail_closed_and_authority,
    test_a46b2_completion_mutation_apply_formal066_oracle,
    test_a46b2_completion_mutation_numpy_torch_devices,
    test_a46b2_completion_mutation_fail_closed_and_authority,
    test_a47a_hydrolysis_literal_boundaries_and_zero_probability_draw,
    test_a47a_hydrolysis_formal066_hit_miss_hit_oracle_and_purity,
    test_a47a_hydrolysis_torch_trust_scope_and_a3_authority,
    test_a47b_hydrolysis_apply_formal066_oracle_and_noop,
    test_a47b_hydrolysis_numpy_torch_devices_and_purity,
    test_a47b_hydrolysis_capacity_trust_rollback_and_authority,
    test_a48a_atomic_commit_formal066_hit_nohit_dt0,
    test_a48a_candidate_cpu_cuda_purity_fresh_binding_and_one_shot,
    test_a48a_fail_closed_capacity_trust_order_and_publish_rollback,
    test_a48a_world_interleave_lockstep_save_clone_and_a3_authority,
    test_a48b_translation_atomic_commit_formal066_oracle_and_gates,
    test_a48b_translation_candidate_cpu_cuda_commit_purity_and_fresh_binding,
    test_a48b_translation_capacity_trust_rollback_and_exact_once,
    test_a48b_world_lockstep_stress_interleave_save_clone_and_a3_authority,
    test_a48c1_active_noncompletion_atomic_commit_formal066_oracle,
    test_a48c1_active_noncompletion_candidate_cpu_cuda_purity_and_rng,
    test_a48c1_active_noncompletion_capacity_trust_rollback_scope_and_exact_once,
    test_a48c1_world_lockstep_interleave_save_clone_successor_and_authority,
    test_a48c2_mutation_free_completion_atomic_commit_formal066_oracle,
    test_a48c2_completion_candidate_cpu_cuda_purity_and_fresh_binding,
    test_a48c2_completion_capacity_trust_rollback_scope_and_exact_once,
    test_a48c2_inherited_c1_save_clone_successor_and_authority,
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
        'build': a48c2.BUILD,
        'schema': {
            'ragged_genome': a4.SCHEMA_VERSION,
            'gene_cache': a4.GENE_CACHE_SCHEMA_VERSION,
            'translation_state': a4.TRANSLATION_SCHEMA_VERSION,
            'paid_replication_elongation': a44.SCHEMA_VERSION,
            'substitution_rng_tape': a44.RNG_TAPE_SCHEMA_VERSION,
            'completion_mutation_rng_tape': (
                a44.COMPLETION_MUTATION_RNG_TAPE_SCHEMA_VERSION
            ),
            'completion_mutation_plan': (
                a44.COMPLETION_MUTATION_PLAN_SCHEMA_VERSION
            ),
            'genome_hydrolysis_rng_tape': a47.SCHEMA_VERSION,
            'genome_hydrolysis_deletion_plan': (
                a47.DELETION_PLAN_SCHEMA_VERSION
            ),
            'genome_hydrolysis_atomic_commit': a48.SCHEMA_VERSION,
            'paid_translation_atomic_commit': a48b.SCHEMA_VERSION,
            'active_noncompletion_replication_atomic_commit': (
                a48c.SCHEMA_VERSION
            ),
            'mutation_free_active_completion_atomic_commit': (
                a48c2.SCHEMA_VERSION
            ),
        },
        'development_slice': (
            'A4.8c2-mutation-free-pre-existing-active-completion-'
            'atomic-commit-bridge'
        ),
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
            '%s VALIDATION' % a48c2.BUILD,
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
