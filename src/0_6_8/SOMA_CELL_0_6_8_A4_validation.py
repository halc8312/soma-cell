# coding: utf-8
"""Focused validation for SOMA-CELL 0.6.8-GPU A4.1 through A4.6a slices."""
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
import SOMA_CELL_0_6_8_gpu_a4_replication as a44
import SOMA_CELL_0_6_8_A3_validation as v3

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

RESULT_JSON = 'SOMA_CELL_0_6_8_GPU_A4_6A_VALIDATION_RESULTS.json'
RESULT_CSV = 'soma_cell_0_6_8_gpu_a4_6a_validation.csv'
RESULT_TXT = 'SOMA_CELL_0_6_8_GPU_A4_6A_VALIDATION_RESULTS.txt'

SOURCE_PATHS = (
    'src/0_6_8/SOMA_CELL_0_6_8_gpu_a4.py',
    'src/0_6_8/SOMA_CELL_0_6_8_gpu_a4_replication.py',
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
    )
    missing = [name for name in replication_required if not hasattr(a44, name)]
    if missing:
        raise AssertionError('missing A4.6a API: %s' % missing)
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
    if a44.BUILD != 'SOMA-CELL 0.6.8-GPU A4.6a':
        raise AssertionError('A4.6a build identity differs')
    if a44.SCHEMA_VERSION != (
            '0.6.8-GPU-A4.6a-replication-completion-plan'):
        raise AssertionError('A4.6a schema identity differs')
    if a44.RNG_TAPE_SCHEMA_VERSION != (
            '0.6.8-GPU-A4.5b-template-start-rng-tape'):
        raise AssertionError('A4.5b RNG tape schema identity differs')
    if a44.FULL_GPU_WORLD_STEP is not False:
        raise AssertionError('A4.6a must not claim full GPU world-step')
    if (a44.SCOPE_FP64_DISCRETE_BOUNDARY != 6
            or a44.FP64_DISCRETE_GUARD_EPS != 4096.0):
        raise AssertionError('A4.5b fp64 discrete guard contract differs')
    if a44.SCOPE_RNG_TAPE_MISMATCH != 7:
        raise AssertionError('A4.5b RNG tape scope code differs')
    if a44.SCOPE_NONCOMPLETION != 8:
        raise AssertionError('A4.6a noncompletion scope code differs')
    return '%s / %s + %s + %s / full_gpu=false' % (
        a44.BUILD, a4.SCHEMA_VERSION,
        a4.GENE_CACHE_SCHEMA_VERSION + ' + ' + a4.TRANSLATION_SCHEMA_VERSION,
        a44.SCHEMA_VERSION,
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
        'a4.6a-mutation-free-active-template-all-row-completion-payload-'
        'ledger-topology-descriptor-not-arena-committed-not-integrated-'
        'cpu-authoritative'
    )
    if a44.PORT_STATUS.get('genome_replication') != expected_status:
        raise AssertionError('A4.6a CPU authority status differs')
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
        'a4.6a-mutation-free-active-template-all-row-completion-payload-'
        'ledger-topology-descriptor-not-arena-committed-not-integrated-'
        'cpu-authoritative'
    )
    if (a44.PORT_STATUS.get('genome_replication') != expected_status
            or a44.FULL_GPU_WORLD_STEP is not False
            or a44.PORT_STATUS.get('full_gpu_world_step') is not False):
        raise AssertionError('A4.6a CPU authority status differs')
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
        'build': a44.BUILD,
        'schema': {
            'ragged_genome': a4.SCHEMA_VERSION,
            'gene_cache': a4.GENE_CACHE_SCHEMA_VERSION,
            'translation_state': a4.TRANSLATION_SCHEMA_VERSION,
            'paid_replication_elongation': a44.SCHEMA_VERSION,
            'substitution_rng_tape': a44.RNG_TAPE_SCHEMA_VERSION,
        },
        'development_slice': (
            'A4.6a-mutation-free-completion-payload-ledger-descriptor'
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
            '%s VALIDATION' % a44.BUILD,
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
