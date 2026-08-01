# coding: utf-8
"""Deterministic validation for SOMA-CELL 0.6-P1.

P1 adds one gene-built material neural compartment on the frozen P0 body port.
The suite verifies exact P0 regression when the cassette is absent, material
construction and dismantling, equal-cost dummy control, physical sensing and
paid effectors, local prediction, deterministic checkpointing, and the P1
acceptance comparison in a moving non-stationary resource field.
"""
from __future__ import division

import csv
import hashlib
import os
import pickle
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P0_DIR = os.path.abspath(os.path.join(HERE, '..', '0_6_p0'))
P0_SOURCE_PATH = os.path.join(P0_DIR, 'SOMA_CELL_0_6_P0_pythonista.py')
if not os.path.isfile(P0_SOURCE_PATH):
    P0_SOURCE_PATH = os.path.join(HERE, 'SOMA_CELL_0_6_P0_pythonista.py')
BASELINE_DIR = os.path.abspath(os.path.join(HERE, '..', 'baseline'))
for candidate in (HERE, P0_DIR, BASELINE_DIR):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import SOMA_CELL_0_6_P0_pythonista as p0
import SOMA_CELL_0_6_P1_pythonista as p1

s5 = p1.s5
CSV_PATH = os.path.join(HERE, 'soma_cell_0_6_p1_validation.csv')
TXT_PATH = os.path.join(HERE, 'SOMA_CELL_0_6_P1_VALIDATION_RESULTS.txt')
P0_SHA256 = '0568ff6b64c7409aeaf11c906699f150a08a520ea2bd7b717801abd66f4980c9'


def exact_equal(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (
            isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
            and a.dtype == b.dtype and a.shape == b.shape
            and np.array_equal(a, b)
        )
    if isinstance(a, dict) or isinstance(b, dict):
        return (
            isinstance(a, dict) and isinstance(b, dict)
            and set(a) == set(b)
            and all(exact_equal(a[key], b[key]) for key in a)
        )
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return (
            type(a) is type(b) and len(a) == len(b)
            and all(exact_equal(x, y) for x, y in zip(a, b))
        )
    if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
        return float(a) == float(b)
    return a == b


def state_hash(value):
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def first_cell(world):
    living = world.living_cells()
    if not living:
        raise AssertionError('expected a living cell')
    return living[0]


def assert_close(a, b, tolerance, label):
    if abs(float(a) - float(b)) > float(tolerance):
        raise AssertionError('{}: {} != {} (tol {})'.format(label, a, b, tolerance))


def high_budget_config(mode=p1.TISSUE_NEURON, **kwargs):
    values = dict(
        p1_tissue_mode=mode,
        p1_environment=p1.ENV_STABLE_PATCH,
        p1_effectors=False,
        sensorimotor=False,
        environmental_damage=False,
        neural_atp_reserve=0.0,
        neural_fuel_reserve=0.0,
        neural_mineral_reserve=0.0,
        neural_membrane_reserve=0.0,
        neural_atp_rate=4.0,
        neural_protein_rate=4.0,
        neural_membrane_rate=4.0,
        neural_signal_rate=4.0,
    )
    values.update(kwargs)
    return p1.P1Config(**values)


def develop(world, max_steps=240):
    dt = 1.0 / p1.SIM_HZ
    for step in range(max_steps):
        world.step(dt)
        cell = first_cell(world)
        if cell.p1_tissue is not None and cell.p1_tissue.mature:
            return step + 1, cell.p1_tissue
    raise AssertionError('P1 tissue did not mature')


def test_build_and_frozen_p0():
    with open(P0_SOURCE_PATH, 'rb') as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    assert digest == P0_SHA256
    assert p1.BUILD == 'SOMA-CELL 0.6-P1.0'
    assert p1.P1_SCHEMA_VERSION.startswith('0.6-P1')
    assert p0.PORT_SCHEMA_VERSION == '0.6-P0.2'
    return 'build={}, P0={}, sha={}'.format(p1.BUILD, p0.PORT_SCHEMA_VERSION, digest)


def test_no_cassette_exact_p0_lockstep():
    observations = []
    for seed in (101, 202, 303):
        control = p0.P0World(seed=seed, initial_cells=1, config=p0.P0Config())
        candidate = p1.P1World(
            seed=seed, initial_cells=1,
            config=p1.P1Config(
                p1_tissue_mode=p1.TISSUE_NONE,
                p1_install_gene=False,
                p1_environment=p1.ENV_NATIVE,
            ),
        )
        for _ in range(240):
            control.step(1.0 / p0.SIM_HZ)
            candidate.step(1.0 / p1.SIM_HZ)
        projected = p1.p0_projection(candidate)
        assert exact_equal(control.state_dict(), projected)
        assert control.rng.bit_generator.state == candidate.rng.bit_generator.state
        observations.append('{}:{}'.format(seed, state_hash(projected)[:12]))
    return '3 seeds x 240 steps exact: ' + ', '.join(observations)


def test_costless_tissue_is_rejected():
    try:
        p1.P1Config(p1_neural_cost=False)
        raise AssertionError('unmetered tissue was accepted')
    except ValueError as exc:
        assert 'unmetered' in str(exc)
    return 'unmetered neural tissue fails closed'


def test_cassette_is_material_and_required():
    no_gene = p1.P1World(
        seed=310, initial_cells=1,
        config=p1.P1Config(
            p1_tissue_mode=p1.TISSUE_NEURON,
            p1_install_gene=False,
            p1_environment=p1.ENV_STABLE_PATCH,
        ),
    )
    for _ in range(120):
        no_gene.step(1.0 / p1.SIM_HZ)
    assert first_cell(no_gene).p1_tissue is None

    base = p0.P0World(seed=311, initial_cells=1, config=p0.P0Config())
    candidate = p1.P1World(
        seed=311, initial_cells=1,
        config=p1.P1Config(p1_tissue_mode=p1.TISSUE_NONE),
    )
    base_cell = first_cell(base)
    cell = first_cell(candidate)
    gene_length = len(cell.genomes[0]) - len(base_cell.genomes[0])
    assert gene_length > 0
    expected_extra = 2.0 * gene_length * s5.MONOMER_MASS + 0.012
    actual_extra = candidate.total_material() - base.total_material()
    assert_close(actual_extra, expected_extra, 3e-12, 'founder cassette material')
    assert p1.one_neuron_specs(cell)
    return 'gene symbols={}, explicit founder matter={:.9g}'.format(gene_length, actual_extra)


def test_gene_driven_tissue_development():
    world = p1.P1World(seed=320, initial_cells=1, config=high_budget_config())
    steps, tissue = develop(world)
    cell = first_cell(world)
    assert tissue.mature and tissue.development >= 0.80
    assert p1.one_neuron_gene_activity(cell) > 0.020
    status = world.port_for(cell.cell_id).attachment_status(p1.P1_TISSUE_ID)
    material = np.asarray(status['tissue_material'], dtype=float)
    assert material[p0.TISSUE_FUNCTIONAL_PROTEIN] >= p1.MATURE_FUNCTIONAL_PROTEIN
    assert material[p0.TISSUE_MEMBRANE] >= p1.MATURE_MEMBRANE
    assert material[p0.TISSUE_SIGNAL] >= p1.MATURE_SIGNAL
    return 'mature in {} steps; material={}'.format(steps, np.round(material, 7).tolist())


def test_equal_material_dummy_before_action_divergence():
    neuron = p1.P1World(seed=330, initial_cells=1, config=high_budget_config(mode=p1.TISSUE_NEURON))
    dummy = p1.P1World(seed=330, initial_cells=1, config=high_budget_config(mode=p1.TISSUE_DUMMY))
    for _ in range(80):
        neuron.step(1.0 / p1.SIM_HZ)
        dummy.step(1.0 / p1.SIM_HZ)
    nc, dc = first_cell(neuron), first_cell(dummy)
    ns = neuron.port_for(nc.cell_id).attachment_status(p1.P1_TISSUE_ID)
    ds = dummy.port_for(dc.cell_id).attachment_status(p1.P1_TISSUE_ID)
    assert np.array_equal(ns['tissue_material'], ds['tissue_material'])
    assert np.array_equal(ns['stores'], ds['stores'])
    assert_close(neuron.summary()['neural_budget_spent'], dummy.summary()['neural_budget_spent'], 1e-15, 'equal paid schedule')
    # Attachment kind metadata differs by design; the chemical body and world
    # must nevertheless remain physically identical while effectors are off.
    for name in ('pools', 'membrane', 'membrane_oxidation', 'transporters',
                 'pos', 'vel', 'proteins', 'surface_flux', 'repair_polarity'):
        assert exact_equal(getattr(nc, name), getattr(dc, name))
    assert exact_equal(nc.genomes, dc.genomes)
    assert exact_equal(neuron.field.state_dict(), dummy.field.state_dict())
    return 'equal material/stores/cost with effectors disabled; spent={:.9g}'.format(neuron.summary()['neural_budget_spent'])


def test_physical_gradient_controls_neuron_direction():
    world = p1.P1World(seed=340, initial_cells=1, config=high_budget_config())
    _, tissue = develop(world)
    cell = first_cell(world)
    port = world.port_for(cell.cell_id)
    original = world.field.pos.copy()
    fuel_indices = np.where(world.field.kind == s5.PARTICLE_FUEL)[0]
    world.field.pos[fuel_indices] = (cell.pos + np.asarray([0.055, 0.0])) % 1.0
    tissue.pre_step(port, 1.0 / p1.SIM_HZ, world.config, p1.one_neuron_gene_activity(cell))
    right = tissue.last_direction.copy()
    world.field.pos[fuel_indices] = (cell.pos + np.asarray([-0.055, 0.0])) % 1.0
    tissue.pre_step(port, 1.0 / p1.SIM_HZ, world.config, p1.one_neuron_gene_activity(cell))
    left = tissue.last_direction.copy()
    world.field.pos = original
    assert right[0] > 0.65 and left[0] < -0.65
    return 'right dx={:.4f}, left dx={:.4f}'.format(right[0], left[0])


def test_dummy_direction_is_sensor_independent():
    world = p1.P1World(seed=350, initial_cells=1, config=high_budget_config(mode=p1.TISSUE_DUMMY))
    _, tissue = develop(world)
    tissue.phase = 1.234
    _, _, g1 = tissue._dummy_sensor()
    direction1 = tissue._choose_direction(g1)
    # Radically alter physical chemistry; the dummy's oscillator is unchanged.
    world.field.pos = (world.field.pos + np.asarray([0.41, 0.27])) % 1.0
    _, _, g2 = tissue._dummy_sensor()
    direction2 = tissue._choose_direction(g2)
    assert np.array_equal(direction1, direction2)
    return 'direction={} unchanged after physical field displacement'.format(np.round(direction1, 6).tolist())


def test_equal_magnitude_action_has_near_equal_paid_cost():
    world = p1.P1World(seed=360, initial_cells=1, config=high_budget_config(p1_effectors=True))
    _, _ = develop(world)
    neuron = world.clone()
    dummy = world.clone()
    dummy.cells[0].p1_tissue.mode = p1.TISSUE_DUMMY
    nr = neuron.cells[0].p1_tissue.pre_step(
        neuron.port_for(neuron.cells[0].cell_id), 1.0 / p1.SIM_HZ,
        neuron.config, p1.one_neuron_gene_activity(neuron.cells[0]),
    )
    dr = dummy.cells[0].p1_tissue.pre_step(
        dummy.port_for(dummy.cells[0].cell_id), 1.0 / p1.SIM_HZ,
        dummy.config, p1.one_neuron_gene_activity(dummy.cells[0]),
    )
    assert_close(np.linalg.norm(nr['direction']), 1.0, 1e-12, 'neuron direction norm')
    assert_close(np.linalg.norm(dr['direction']), 1.0, 1e-12, 'dummy direction norm')
    ncost = nr['effector']['atp_spent'] + nr['effector']['signal_spent']
    dcost = dr['effector']['atp_spent'] + dr['effector']['signal_spent']
    assert abs(ncost - dcost) < 2e-6
    return 'equal command norm; paid cost neuron={:.8g}, dummy={:.8g}'.format(ncost, dcost)


def test_local_predictor_reduces_replay_error():
    world = p1.P1World(
        seed=370, initial_cells=1,
        config=p1.P1Config(
            p1_tissue_mode=p1.TISSUE_NEURON,
            p1_environment=p1.ENV_MOVING_PATCH,
            p1_effectors=False,
        ),
    )
    values = []
    for _ in range(420):
        world.step(1.0 / p1.SIM_HZ)
        tissue = first_cell(world).p1_tissue
        if tissue is not None and tissue.mature and tissue.predictor_updates:
            values.append(tissue.last_prediction_rms)
    assert len(values) > 300
    early = float(np.mean(values[:60]))
    late = float(np.mean(values[-60:]))
    assert late < early * 0.90
    return 'prediction RMS early={:.6f}, late={:.6f}, updates={}'.format(early, late, len(values))


def test_no_effector_has_no_neural_motor_flux():
    summary = p1.run_headless_trial(
        seed=380, seconds=20.0, initial_cells=1,
        config=p1.P1Config(
            p1_tissue_mode=p1.TISSUE_NO_EFFECTOR,
            p1_environment=p1.ENV_MOVING_PATCH,
        ),
    )
    assert summary['p1_predictor_updates'] > 0
    assert summary['p1_motor_force_total'] == 0.0
    assert summary['neural_effector_calls'] == 0
    return 'predictor updates={}, motor force=0'.format(summary['p1_predictor_updates'])


def test_activity_wear_turnover_is_conservative():
    world = p1.P1World(seed=390, initial_cells=1, config=high_budget_config(p1_effectors=False))
    _, tissue = develop(world)
    cell = first_cell(world)
    state = cell.neural_attachments[p1.P1_TISSUE_ID]
    shift = min(0.020, float(state.tissue_material[p0.TISSUE_FUNCTIONAL_PROTEIN]))
    state.tissue_material[p0.TISSUE_FUNCTIONAL_PROTEIN] -= shift
    state.tissue_material[p0.TISSUE_DAMAGED_PROTEIN] += shift
    before = world.total_material()
    tissue.pre_step(world.port_for(cell.cell_id), 1.0 / p1.SIM_HZ, world.config, p1.one_neuron_gene_activity(cell))
    assert tissue.failed
    assert p1.P1_TISSUE_ID not in cell.neural_attachments
    assert_close(before, world.total_material(), 3e-12, 'turnover material')
    return 'damaged mass={:.6g} returned; ledger={:+.3e}'.format(shift, world.matter_ledger_residual())


def test_host_death_routes_neural_matter_to_corpse():
    world = p1.P1World(seed=400, initial_cells=1, config=high_budget_config())
    _, _ = develop(world)
    cell = first_cell(world)
    tissue_mass = cell.neural_attachments[p1.P1_TISSUE_ID].material_mass()
    cell.alive = False
    cell.death_reason = 'p1-validation'
    world._handle_divisions_and_deaths()
    assert not world.cells and len(world.corpses) == 1
    assert world.p0_returned_material >= tissue_mass - 2e-12
    assert abs(world.matter_ledger_residual()) < 8e-5
    return 'neural matter={:.6g}, corpse={:.6g}'.format(tissue_mass, world.corpses[0].material_mass())


def test_division_recycles_state_and_gene_remains_material():
    config = high_budget_config(mutation=False, external_replicase=True)
    world = p1.P1World(seed=410, initial_cells=1, config=config)
    _, _ = develop(world)
    cell = first_cell(world)
    cell.pools[s5.POOL_NUCLEOTIDE] = max(cell.pools[s5.POOL_NUCLEOTIDE], 4.0)
    cell.pools[s5.POOL_ATP] = max(cell.pools[s5.POOL_ATP], 4.0)
    steps = 0
    while len(cell.genomes) < 2 and steps < 9000:
        cell._replicate_genome(world, 0.1, world.config)
        steps += 1
    assert len(cell.genomes) >= 2
    tissue_mass = cell.neural_attachments[p1.P1_TISSUE_ID].material_mass()
    cell.division_progress = 1.0
    cell.septum_mass = max(cell.septum_mass, 0.12)
    daughters = cell.split(world)
    assert daughters is not None and len(daughters) == 2
    assert all(d.p1_tissue is None for d in daughters)
    assert all(not d.neural_attachments for d in daughters)
    assert all(p1.one_neuron_specs(d) for d in daughters)
    assert world.p0_returned_material >= tissue_mass - 2e-12
    assert abs(world.division_residual) < 3e-8
    return 'copy steps={}, tissue recycled={:.6g}, daughters retain gene not state'.format(steps, tissue_mass)


def test_save_restore_exact():
    world = p1.P1World(
        seed=420, initial_cells=1,
        config=p1.P1Config(p1_environment=p1.ENV_MOVING_PATCH),
    )
    for _ in range(180):
        world.step(1.0 / p1.SIM_HZ)
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, 'p1.pkl')
        world.save(path)
        restored = p1.P1World.load(path)
    assert exact_equal(world.state_dict(), restored.state_dict())
    for _ in range(120):
        world.step(1.0 / p1.SIM_HZ)
        restored.step(1.0 / p1.SIM_HZ)
    assert exact_equal(world.state_dict(), restored.state_dict())
    return 'checkpoint and 120-step continuation exact'


def test_logger_is_noninterfering():
    observed = p1.P1World(seed=430, initial_cells=1, config=p1.P1Config(p1_environment=p1.ENV_MOVING_PATCH))
    control = observed.clone()
    with tempfile.TemporaryDirectory() as directory:
        logger = p1.LongRunLogger(observed, path=os.path.join(directory, 'p1.csv'))
        for step in range(120):
            observed.step(1.0 / p1.SIM_HZ)
            control.step(1.0 / p1.SIM_HZ)
            if step % 15 == 0:
                logger.log(observed, force=True)
        assert os.path.getsize(logger.path) > 0
    assert exact_equal(observed.state_dict(), control.state_dict())
    return 'logger-neutral 120-step continuation'


def test_moving_patch_information_value_vs_equal_dummy():
    observations = []
    for seed in (101,):
        neuron = p1.run_headless_trial(
            seed=seed, seconds=35.0,
            config=p1.P1Config(
                p1_tissue_mode=p1.TISSUE_NEURON,
                p1_environment=p1.ENV_MOVING_PATCH,
            ),
        )
        dummy = p1.run_headless_trial(
            seed=seed, seconds=35.0,
            config=p1.P1Config(
                p1_tissue_mode=p1.TISSUE_DUMMY,
                p1_environment=p1.ENV_MOVING_PATCH,
            ),
        )
        margin_gain = neuron['mean_margin_over_life'] - dummy['mean_margin_over_life']
        uptake_gain = neuron['uptake_during_trial'] - dummy['uptake_during_trial']
        assert margin_gain > 0.04
        assert uptake_gain > 0.50
        assert neuron['p1_patch_switches'] >= 2
        observations.append('{}:+{:.4f}/+{:.3f}'.format(seed, margin_gain, uptake_gain))
    return 'margin/uptake gains: ' + ', '.join(observations)


def test_finite_and_material_ledger_stress():
    world = p1.P1World(seed=440, initial_cells=2, config=p1.P1Config(p1_environment=p1.ENV_MOVING_PATCH))
    maximum = 0.0
    for _ in range(360):
        world.step(1.0 / p1.SIM_HZ)
        maximum = max(maximum, abs(world.matter_ledger_residual()))
        assert world.finite()
    port_max = max([
        c.neural_budget_ledger.maximum_material_residual
        for c in world.living_cells()
    ] or [0.0])
    assert maximum < 8e-5
    assert port_max < 3e-10
    return '360 steps finite; world max={:.3e}, port max={:.3e}'.format(maximum, port_max)


TESTS = (
    ('build_and_frozen_p0', test_build_and_frozen_p0),
    ('no_cassette_exact_p0_lockstep', test_no_cassette_exact_p0_lockstep),
    ('costless_tissue_is_rejected', test_costless_tissue_is_rejected),
    ('cassette_is_material_and_required', test_cassette_is_material_and_required),
    ('gene_driven_tissue_development', test_gene_driven_tissue_development),
    ('equal_material_dummy_before_action_divergence', test_equal_material_dummy_before_action_divergence),
    ('physical_gradient_controls_neuron_direction', test_physical_gradient_controls_neuron_direction),
    ('dummy_direction_is_sensor_independent', test_dummy_direction_is_sensor_independent),
    ('equal_magnitude_action_has_near_equal_paid_cost', test_equal_magnitude_action_has_near_equal_paid_cost),
    ('local_predictor_reduces_replay_error', test_local_predictor_reduces_replay_error),
    ('no_effector_has_no_neural_motor_flux', test_no_effector_has_no_neural_motor_flux),
    ('activity_wear_turnover_is_conservative', test_activity_wear_turnover_is_conservative),
    ('host_death_routes_neural_matter_to_corpse', test_host_death_routes_neural_matter_to_corpse),
    ('division_recycles_state_and_gene_remains_material', test_division_recycles_state_and_gene_remains_material),
    ('save_restore_exact', test_save_restore_exact),
    ('logger_is_noninterfering', test_logger_is_noninterfering),
    ('moving_patch_information_value_vs_equal_dummy', test_moving_patch_information_value_vs_equal_dummy),
    ('finite_and_material_ledger_stress', test_finite_and_material_ledger_stress),
)


def run_validation():
    rows = []
    for name, function in TESTS:
        try:
            observed = function()
            passed = True
            error = ''
            print('[PASS] {}'.format(name), flush=True)
        except Exception as exc:
            observed = ''
            passed = False
            error = '{}: {}'.format(type(exc).__name__, exc)
            print('[FAIL] {} — {}'.format(name, error), flush=True)
            traceback.print_exc()
        rows.append({'test': name, 'passed': int(passed), 'observed': observed, 'error': error})

    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=('test', 'passed', 'observed', 'error'))
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(row['passed'] for row in rows)
    lines = [
        'SOMA-CELL 0.6-P1 deterministic validation',
        'build: {}'.format(p1.BUILD),
        'P1 schema: {}'.format(p1.P1_SCHEMA_VERSION),
        'P0 port schema: {}'.format(p0.PORT_SCHEMA_VERSION),
        'result: {}/{} PASS'.format(passed, len(rows)),
        '',
    ]
    for row in rows:
        lines.append('[{}] {}'.format('PASS' if row['passed'] else 'FAIL', row['test']))
        if row['observed']:
            lines.append('  observed: {}'.format(row['observed']))
        if row['error']:
            lines.append('  error: {}'.format(row['error']))
    with open(TXT_PATH, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')

    if passed != len(rows):
        raise SystemExit('validation failed: {}/{} PASS'.format(passed, len(rows)))
    return rows


if __name__ == '__main__':
    run_validation()
