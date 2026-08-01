# coding: utf-8
"""Deterministic contract validation for SOMA-CELL 0.6-P0.

P0 contains no information-processing neuron.  These checks validate the
boundary around the frozen SOMA-CELL 0.5 chemical body: read-only sensing,
finite physical budgets, effectors that act through existing body physics,
conservative tissue dismantling, exact baseline lockstep, and deterministic
checkpointing.
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
BASELINE_DIR = os.path.abspath(os.path.join(HERE, '..', 'baseline'))
# The repository keeps frozen dependencies in ../baseline, while the
# Pythonista release is intentionally flat.  Support both without guessing.
for candidate in (BASELINE_DIR, HERE):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import SOMA_CELL_0_5_pythonista as s5
import SOMA_CELL_0_6_P0_pythonista as p0

FROZEN_05_PATH = os.path.join(HERE, 'SOMA_CELL_0_5_pythonista.py')
if not os.path.isfile(FROZEN_05_PATH):
    FROZEN_05_PATH = os.path.join(BASELINE_DIR, 'SOMA_CELL_0_5_pythonista.py')

CSV_PATH = os.path.join(HERE, 'soma_cell_0_6_p0_validation.csv')
TXT_PATH = os.path.join(HERE, 'SOMA_CELL_0_6_P0_VALIDATION_RESULTS.txt')
FROZEN_05_SHA256 = '5f37ac140e5a5e5cbf0e2f48597e3366ea3710a112f35940de35aae1e1931459'


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
        raise AssertionError('expected at least one living cell')
    return living[0]


def assert_close(a, b, tolerance, label):
    if abs(float(a) - float(b)) > float(tolerance):
        raise AssertionError('{}: {} != {} (tol {})'.format(label, a, b, tolerance))


def baseline_world(seed, initial_cells=2):
    return s5.EcologicalWorld(
        seed=seed,
        initial_cells=initial_cells,
        config=p0.P0Config().baseline_config(),
    )


def p0_world(seed, initial_cells=2, config=None):
    return p0.P0World(
        seed=seed,
        initial_cells=initial_cells,
        config=config if config is not None else p0.P0Config(),
    )


def high_budget_config(**kwargs):
    values = dict(
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
    return p0.P0Config(**values)


def run_exact_lockstep(seed, steps, attach_null=False, initial_cells=1):
    control = baseline_world(seed, initial_cells=initial_cells)
    candidate = p0_world(seed, initial_cells=initial_cells)
    if attach_null:
        for cell in candidate.living_cells():
            candidate.attach_null_tissue(cell.cell_id, 'null-{}'.format(cell.cell_id))
    dt = 1.0 / p0.SIM_HZ
    for _ in range(int(steps)):
        control.step(dt)
        candidate.step(dt)
    return control, candidate


def test_build_and_frozen_baseline():
    with open(FROZEN_05_PATH, 'rb') as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    assert digest == FROZEN_05_SHA256
    assert p0.BUILD == 'SOMA-CELL 0.6-P0.0'
    assert p0.PORT_SCHEMA_VERSION.startswith('0.6-P0')
    return 'build={}, baseline_sha256={}'.format(p0.BUILD, digest)


def test_unattached_exact_lockstep():
    observations = []
    for seed in (101, 202, 303):
        control, candidate = run_exact_lockstep(seed, 240, attach_null=False, initial_cells=1)
        projected = p0.baseline_projection(candidate)
        assert exact_equal(control.state_dict(), projected)
        assert control.rng.bit_generator.state == candidate.rng.bit_generator.state
        observations.append('{}:{}'.format(seed, state_hash(projected)[:12]))
    return '3 seeds x 240 steps exact: ' + ', '.join(observations)


def test_null_attachment_physical_lockstep():
    observations = []
    for seed in (111, 222, 333):
        control, candidate = run_exact_lockstep(seed, 240, attach_null=True, initial_cells=1)
        assert exact_equal(control.state_dict(), p0.baseline_projection(candidate))
        assert control.rng.bit_generator.state == candidate.rng.bit_generator.state
        assert candidate.summary()['neural_attachments'] == 1
        assert candidate.summary()['neural_material_escrow'] == 0.0
        observations.append('{}:{}'.format(seed, candidate.summary()['neural_attachments']))
    return 'null metadata is physically inert: ' + ', '.join(observations)


def test_sensor_readonly_rng_neutral_no_privileged_labels():
    world = p0_world(404, initial_cells=1)
    cell = first_cell(world)
    port = world.port_for(cell.cell_id)
    port.attach('sensor-probe')
    before = world.state_dict()
    rng_before = pickle.dumps(world.rng.bit_generator.state, protocol=4)
    frame = None
    for _ in range(64):
        frame = port.raw_sensor_fluxes('sensor-probe')
    assert exact_equal(before, world.state_dict())
    assert rng_before == pickle.dumps(world.rng.bit_generator.state, protocol=4)
    try:
        frame['cell_id'] = 99
        raise AssertionError('top-level sensor mapping was mutable')
    except TypeError:
        pass
    try:
        frame['membrane']['material'][0] = 99.0
        raise AssertionError('sensor array was mutable')
    except ValueError:
        pass
    forbidden = {
        'reward', 'fitness', 'correct_action', 'food_direction',
        'autopoietic_margin', 'position', 'world_age', 'cell_age',
    }
    def keys_recursive(value):
        result = set()
        if hasattr(value, 'items'):
            for key, item in value.items():
                result.add(str(key))
                result.update(keys_recursive(item))
        return result
    observed = keys_recursive(frame)
    assert not (forbidden & observed)
    return 'state/RNG unchanged; immutable; no privileged keys ({})'.format(
        ','.join(sorted(forbidden))
    )


def test_port_surface_hides_mutable_body_and_attachment_state():
    world = p0_world(4041, initial_cells=1)
    cell = first_cell(world)
    port = world.port_for(cell.cell_id)
    status = port.attach('privacy-probe', kind='contract-probe')
    assert not hasattr(port, 'cell')
    assert not hasattr(port, 'world')
    assert not hasattr(port, 'attachment')
    assert status['tissue_id'] == 'privacy-probe'
    assert status['kind'] == 'contract-probe'
    try:
        status['active'] = False
        raise AssertionError('attachment status mapping was mutable')
    except TypeError:
        pass
    try:
        status['stores'][0] = 99.0
        raise AssertionError('attachment stores escaped as mutable state')
    except ValueError:
        pass
    assert np.all(cell.neural_attachments['privacy-probe'].stores == 0.0)
    return 'no public body/world/state backdoor; immutable diagnostic attachment status'


def test_sensor_schema_physical_shapes():
    world = p0_world(405, initial_cells=1)
    cell = first_cell(world)
    port = world.port_for(cell.cell_id)
    port.attach('sensor-shape')
    frame = port.raw_sensor_fluxes('sensor-shape')
    assert frame['velocity'].shape == (2,)
    assert frame['membrane']['material'].shape == (s5.MEMBRANE_SEGMENTS,)
    assert frame['membrane']['closure'].shape == (s5.MEMBRANE_SEGMENTS,)
    assert frame['membrane']['transporters'].shape == (
        s5.MEMBRANE_SEGMENTS, p0.s4.CHANNEL_COUNT
    )
    assert frame['external']['particle_profiles'].shape == (
        len(p0.s4.PARTICLE_NAMES), s5.MEMBRANE_SEGMENTS
    )
    assert frame['external']['ligand_profiles'].shape == (
        p0.s4.LIGAND_COUNT, s5.MEMBRANE_SEGMENTS
    )
    assert frame['external']['stress_profile'].shape == (s5.MEMBRANE_SEGMENTS,)
    assert frame['internal']['pools'].shape == (s5.POOL_COUNT,)
    return 'membrane={}, transporters={}, ligands={}, pools={}'.format(
        frame['membrane']['material'].shape,
        frame['membrane']['transporters'].shape,
        frame['external']['ligand_profiles'].shape,
        frame['internal']['pools'].shape,
    )


def test_port_disabled_fails_closed():
    world = p0_world(406, initial_cells=1, config=p0.P0Config(neural_body_port=False))
    cell = first_cell(world)
    port = world.port_for(cell.cell_id)
    failures = 0
    for action in (
        lambda: port.attach('x'),
        lambda: port.raw_sensor_fluxes(),
    ):
        try:
            action()
        except RuntimeError:
            failures += 1
    assert failures == 2
    assert not cell.neural_attachments
    return '2/2 port operations rejected while disabled'


def test_invalid_budget_requests_fail_closed():
    world = p0_world(505, initial_cells=1)
    cell = first_cell(world)
    port = world.port_for(cell.cell_id)
    port.attach('budget-invalid')
    body_before = cell.pools.copy()
    stores_before = cell.neural_attachments['budget-invalid'].stores.copy()
    material_before = world.total_material()
    failures = 0
    cases = (
        {'free_energy': 1.0},
        {'atp': -1.0},
        {'protein': float('nan')},
    )
    for request in cases:
        try:
            port.allocate_budget('budget-invalid', request, 0.1)
        except (ValueError, TypeError):
            failures += 1
    try:
        port.allocate_budget('budget-invalid', {'atp': 0.1}, float('nan'))
    except ValueError:
        failures += 1
    assert failures == 4
    assert np.array_equal(body_before, cell.pools)
    assert np.array_equal(stores_before, cell.neural_attachments['budget-invalid'].stores)
    assert_close(material_before, world.total_material(), 0.0, 'invalid budget matter')
    return '4/4 malformed requests rejected without physical mutation'


def test_budget_capped_reserves_and_nonnegative():
    world = p0_world(506, initial_cells=1)
    cell = first_cell(world)
    port = world.port_for(cell.cell_id)
    port.attach('budget-cap')
    dt = 0.1
    grant = port.allocate_budget(
        'budget-cap', {name: 999.0 for name in p0.BUDGET_NAMES}, dt
    )
    limits = {
        'atp': world.config.neural_atp_rate * dt,
        'protein': world.config.neural_protein_rate * dt,
        'membrane': world.config.neural_membrane_rate * dt,
        'signal': world.config.neural_signal_rate * dt,
    }
    for name, value in grant.items():
        assert value <= limits[name] + 1e-15
    assert cell.pools[s5.POOL_ATP] >= world.config.neural_atp_reserve - 1e-12
    assert cell.pools[s5.POOL_FUEL] >= world.config.neural_fuel_reserve - 1e-12
    assert cell.pools[s5.POOL_MINERAL] >= world.config.neural_mineral_reserve - 1e-12
    assert cell.pools[s5.POOL_MEM_PRECURSOR] >= world.config.neural_membrane_reserve - 1e-12
    assert np.all(cell.pools >= -1e-12)
    return 'grant={} reserves maintained'.format(grant)


def test_budget_transfer_and_return_conserve_material():
    world = p0_world(507, initial_cells=1, config=high_budget_config())
    cell = first_cell(world)
    port = world.port_for(cell.cell_id)
    port.attach('budget-roundtrip')
    state = cell.neural_attachments['budget-roundtrip']
    physical_before = cell.pools.copy()
    total_before = world.total_material()
    grant = port.allocate_budget(
        'budget-roundtrip',
        {'atp': 0.09, 'protein': 0.04, 'membrane': 0.025, 'signal': 0.018},
        0.1,
    )
    assert_close(total_before, world.total_material(), 2e-12, 'allocation material')
    assert sum(grant.values()) > 0.0
    returned = port.return_unused_budget('budget-roundtrip')
    assert np.array_equal(physical_before, cell.pools)
    assert np.all(state.stores == 0.0)
    assert_close(total_before, world.total_material(), 2e-12, 'return material')
    return 'allocated/returned={} residual={:+.3e}'.format(
        returned, world.total_material() - total_before
    )


def test_multiple_tissues_share_finite_body_budget():
    world = p0_world(508, initial_cells=1, config=high_budget_config())
    cell = first_cell(world)
    port = world.port_for(cell.cell_id)
    port.attach('a')
    port.attach('b')
    initial = cell.pools.copy()
    first = port.allocate_budget('a', {name: 99.0 for name in p0.BUDGET_NAMES}, 0.1)
    second = port.allocate_budget('b', {name: 99.0 for name in p0.BUDGET_NAMES}, 0.1)
    assert np.all(cell.pools >= -1e-12)
    assert first['atp'] + second['atp'] <= initial[s5.POOL_ATP] + 1e-12
    material = sum(
        attachment.material_mass() for attachment in cell.neural_attachments.values()
    )
    assert material > 0.0
    assert_close(world.total_material(), world.initial_total_material, 2e-12, 'shared budget matter')
    return 'a={}, b={}, remaining_ATP={:.6g}'.format(first, second, cell.pools[s5.POOL_ATP])


def test_material_commit_is_paid_conservative_and_cannot_mint():
    world = p0_world(509, initial_cells=1, config=high_budget_config())
    cell = first_cell(world)
    port = world.port_for(cell.cell_id)
    port.attach('assembly')
    state = cell.neural_attachments['assembly']
    port.allocate_budget(
        'assembly',
        {'atp': 0.10, 'protein': 0.05, 'membrane': 0.03, 'signal': 0.02},
        0.1,
    )
    total_before = world.total_material()
    atp_before = state.atp()
    substrate_before = state.material_mass()
    built = port.commit_material(
        'assembly', protein=999.0, membrane=999.0, signal=999.0,
        damaged_fraction=0.20, aggregate_fraction=0.10,
    )
    assert_close(total_before, world.total_material(), 2e-12, 'assembly material')
    assert state.atp() < atp_before
    assert state.material_mass() <= substrate_before + 1e-12
    assert sum(built[key] for key in (
        'functional_protein', 'damaged_protein', 'aggregate'
    )) <= 0.05 + 1e-12
    try:
        port.commit_material('assembly', protein=-1.0)
        raise AssertionError('negative material request was accepted')
    except ValueError:
        pass
    return 'built={} ATP spent={:.6g}'.format(built, atp_before - state.atp())


def test_forbidden_effector_rewrites_fail_closed():
    world = p0_world(601, initial_cells=1, config=high_budget_config())
    cell = first_cell(world)
    port = world.port_for(cell.cell_id)
    port.attach('effector-forbidden')
    state = cell.neural_attachments['effector-forbidden']
    port.allocate_budget('effector-forbidden', {'atp': 0.08, 'signal': 0.05}, 0.1)
    position = cell.pos.copy()
    membrane = cell.membrane.copy()
    pools = cell.pools.copy()
    genomes = [genome.copy() for genome in cell.genomes]
    stores = state.stores.copy()
    rejected = 0
    for request in (
        {'position': [0.2, 0.2]}, {'membrane': [1.0]},
        {'dna': [1, 2, 3]}, {'atp': 5.0}, {'teleport': [1, 0]},
    ):
        try:
            port.apply_effector_fluxes('effector-forbidden', request, 0.1)
        except ValueError:
            rejected += 1
    assert rejected == 5
    assert np.array_equal(position, cell.pos)
    assert np.array_equal(membrane, cell.membrane)
    assert np.array_equal(pools, cell.pools)
    assert all(np.array_equal(a, b) for a, b in zip(genomes, cell.genomes))
    assert np.array_equal(stores, state.stores)
    return '5/5 direct or unknown rewrites rejected'


def test_motor_effector_is_paid_bounded_and_indirect():
    world = p0_world(602, initial_cells=1, config=high_budget_config())
    cell = first_cell(world)
    port = world.port_for(cell.cell_id)
    port.attach('motor')
    state = cell.neural_attachments['motor']
    port.allocate_budget('motor', {'atp': 0.10, 'signal': 0.08}, 0.1)
    position_before = cell.pos.copy()
    flux_before = cell.surface_flux.copy()
    material_before = world.total_material()
    atp_before = state.atp()
    report = port.apply_effector_fluxes('motor', {'motor': [9.0, 0.0]}, 0.1)
    assert report['motor_force'] > 0.0
    assert report['atp_spent'] > 0.0
    assert report['signal_spent'] > 0.0
    assert state.atp() < atp_before
    assert np.array_equal(position_before, cell.pos)
    assert cell.surface_flux[0] > flux_before[0]
    assert_close(material_before, world.total_material(), 2e-12, 'motor material')
    cell.update_motion(world.rng, 0.1)
    displacement = float(np.linalg.norm(p0.wrapped_delta(position_before, cell.pos)))
    assert displacement > 0.0
    return 'force={:.6g}, ATP={:.6g}, indirect displacement={:.6g}'.format(
        report['motor_force'], report['atp_spent'], displacement
    )


def test_transporter_polarity_conserves_channel_material():
    world = p0_world(603, initial_cells=1, config=high_budget_config())
    cell = first_cell(world)
    port = world.port_for(cell.cell_id)
    port.attach('polarity')
    port.allocate_budget('polarity', {'atp': 0.10, 'signal': 0.08}, 0.1)
    before = cell.transporters.copy()
    channel_before = np.sum(before, axis=0)
    total_before = world.total_material()
    report = port.apply_effector_fluxes(
        'polarity', {'transporter_polarity': [1.0, 0.35]}, 0.1
    )
    channel_after = np.sum(cell.transporters, axis=0)
    assert report['transporter_movement'] > 0.0
    assert not np.array_equal(before, cell.transporters)
    assert np.allclose(channel_before, channel_after, atol=3e-14, rtol=0.0)
    assert_close(total_before, world.total_material(), 2e-12, 'polarity material')
    return 'movement={:.6g}, max channel residual={:.3e}'.format(
        report['transporter_movement'],
        float(np.max(np.abs(channel_after - channel_before))),
    )


def test_repair_and_quiescence_are_paid_physical_requests():
    world = p0_world(604, initial_cells=1, config=high_budget_config())
    cell = first_cell(world)
    port = world.port_for(cell.cell_id)
    port.attach('regulation')
    state = cell.neural_attachments['regulation']
    port.allocate_budget('regulation', {'atp': 0.10, 'signal': 0.08}, 0.1)
    atp_before = state.atp()
    repair_before = cell.repair_polarity.copy()
    quiescence_before = float(cell.behavioural_quiescence)
    report = port.apply_effector_fluxes(
        'regulation',
        {'repair_polarity': [0.2, 1.0], 'quiescence': 0.75},
        0.1,
    )
    assert report['repair_polarity_gain'] > 0.0
    assert report['quiescence'] > quiescence_before
    assert not np.array_equal(repair_before, cell.repair_polarity)
    assert state.atp() < atp_before
    return 'repair gain={:.6g}, quiescence={:.6g}, ATP={:.6g}'.format(
        report['repair_polarity_gain'], report['quiescence'], report['atp_spent']
    )


def test_dead_tissue_return_is_conservative():
    world = p0_world(605, initial_cells=1, config=high_budget_config())
    cell = first_cell(world)
    port = world.port_for(cell.cell_id)
    port.attach('dead-tissue')
    state = cell.neural_attachments['dead-tissue']
    port.allocate_budget(
        'dead-tissue', {'atp': 0.09, 'protein': 0.04, 'membrane': 0.025, 'signal': 0.018}, 0.1
    )
    port.commit_material(
        'dead-tissue', protein=0.02, membrane=0.01, signal=0.006,
        damaged_fraction=0.25, aggregate_fraction=0.10,
    )
    material = state.material_mass()
    atp = state.atp()
    total_before = world.total_material()
    result = port.return_dead_tissue('dead-tissue', reason='validation')
    assert 'dead-tissue' not in cell.neural_attachments
    assert_close(result['material'], material, 2e-12, 'returned tissue material')
    assert_close(result['atp_dissipated'], atp, 2e-12, 'returned tissue ATP')
    assert_close(total_before, world.total_material(), 2e-12, 'dead tissue material')
    assert abs(result['residual']) <= 2e-12
    return 'material={:.6g}, ATP dissipated={:.6g}, residual={:+.3e}'.format(
        material, atp, result['residual']
    )


def test_host_death_routes_tissue_into_spatial_corpse():
    world = p0_world(606, initial_cells=1, config=high_budget_config())
    cell = first_cell(world)
    port = world.port_for(cell.cell_id)
    port.attach('host-death')
    state = cell.neural_attachments['host-death']
    port.allocate_budget(
        'host-death', {'atp': 0.08, 'protein': 0.035, 'membrane': 0.02, 'signal': 0.016}, 0.1
    )
    port.commit_material('host-death', protein=0.018, membrane=0.01, signal=0.005)
    tissue_mass = state.material_mass()
    cell.alive = False
    cell.death_reason = 'validation-host-death'
    world._handle_divisions_and_deaths()
    assert len(world.corpses) == 1
    assert not world.cells
    assert world.p0_tissue_returns == 1
    assert world.p0_returned_material >= tissue_mass - 2e-12
    assert world.corpses[0].material_mass() > tissue_mass
    assert abs(world.matter_ledger_residual()) < 8e-5
    return 'tissue={:.6g}, corpse={:.6g}, ledger={:+.3e}'.format(
        tissue_mass, world.corpses[0].material_mass(), world.matter_ledger_residual()
    )


def test_division_recycles_tissue_without_free_inheritance():
    config = high_budget_config(mutation=False, external_replicase=True)
    world = p0_world(607, initial_cells=1, config=config)
    cell = first_cell(world)
    cell.pools[s5.POOL_NUCLEOTIDE] = max(cell.pools[s5.POOL_NUCLEOTIDE], 3.0)
    cell.pools[s5.POOL_ATP] = max(cell.pools[s5.POOL_ATP], 3.0)
    steps = 0
    while len(cell.genomes) < 2 and steps < 6000:
        cell._replicate_genome(world, 0.1, world.config)
        steps += 1
    assert len(cell.genomes) >= 2
    port = world.port_for(cell.cell_id)
    port.attach('pre-division')
    state = cell.neural_attachments['pre-division']
    port.allocate_budget(
        'pre-division', {'atp': 0.08, 'protein': 0.035, 'membrane': 0.02, 'signal': 0.016}, 0.1
    )
    port.commit_material('pre-division', protein=0.018, membrane=0.01, signal=0.005)
    tissue_mass = state.material_mass()
    cell.division_progress = 1.0
    cell.septum_mass = max(cell.septum_mass, 0.12)
    daughters = cell.split(world)
    assert daughters is not None and len(daughters) == 2
    assert all(not daughter.neural_attachments for daughter in daughters)
    assert world.p0_tissue_returns == 1
    assert world.p0_returned_material >= tissue_mass - 2e-12
    assert abs(world.division_residual) < 3e-8
    return 'replication steps={}, tissue recycled={:.6g}, division residual={:+.3e}'.format(
        steps, tissue_mass, world.division_residual
    )


def test_pure_05_state_migrates_without_physical_change():
    baseline = baseline_world(701, initial_cells=2)
    for _ in range(120):
        baseline.step(1.0 / p0.SIM_HZ)
    migrated = p0.P0World.from_state(baseline.state_dict())
    assert exact_equal(baseline.state_dict(), p0.baseline_projection(migrated))
    assert migrated.summary()['neural_attachments'] == 0
    return 'baseline={}, migrated={}'.format(
        state_hash(baseline.state_dict())[:16],
        state_hash(p0.baseline_projection(migrated))[:16],
    )


def test_save_restore_exact_with_attachment_state():
    world = p0_world(702, initial_cells=1, config=high_budget_config())
    cell = first_cell(world)
    port = world.port_for(cell.cell_id)
    port.attach('checkpoint')
    port.allocate_budget(
        'checkpoint', {'atp': 0.08, 'protein': 0.035, 'membrane': 0.02, 'signal': 0.016}, 0.1
    )
    port.commit_material('checkpoint', protein=0.015, membrane=0.008, signal=0.004)
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, 'p0.pkl')
        world.save(path)
        restored = p0.P0World.load(path)
    assert exact_equal(world.state_dict(), restored.state_dict())
    for _ in range(180):
        world.step(1.0 / p0.SIM_HZ)
        restored.step(1.0 / p0.SIM_HZ)
    assert exact_equal(world.state_dict(), restored.state_dict())
    return 'save and 180-step continuation are exact'


def test_logger_noninterfering_and_budget_stress_finite():
    template = p0_world(703, initial_cells=2, config=high_budget_config())
    for cell in template.living_cells():
        port = template.port_for(cell.cell_id)
        port.attach('stress-{}'.format(cell.cell_id))
    observed = template.clone()
    control = template.clone()
    with tempfile.TemporaryDirectory() as directory:
        log_path = os.path.join(directory, 'longrun.csv')
        logger = p0.LongRunLogger(observed, path=log_path)
        for step in range(240):
            for world in (observed, control):
                for cell in list(world.living_cells()):
                    tissue_id = 'stress-{}'.format(cell.cell_id)
                    if tissue_id not in cell.neural_attachments:
                        world.port_for(cell.cell_id).attach(tissue_id)
                    port = world.port_for(cell.cell_id)
                    if step % 12 == 0:
                        port.allocate_budget(
                            tissue_id,
                            {'atp': 0.012, 'protein': 0.003, 'membrane': 0.0015, 'signal': 0.003},
                            0.05,
                        )
                    if step % 12 == 1:
                        port.apply_effector_fluxes(
                            tissue_id,
                            {
                                'motor': [np.cos(step * 0.13), np.sin(step * 0.13)],
                                'transporter_polarity': [np.sin(step * 0.07), np.cos(step * 0.07)],
                            },
                            0.05,
                        )
                    if step % 12 == 2:
                        port.return_unused_budget(tissue_id)
                world.step(0.05)
            if step % 15 == 0:
                logger.log(observed, fps=20.0, sim_rate=1.0, force=True)
        assert os.path.exists(log_path) and os.path.getsize(log_path) > 0
    assert exact_equal(observed.state_dict(), control.state_dict())
    assert observed.finite()
    assert abs(observed.matter_ledger_residual()) < 8e-5
    maximum = max(
        cell.neural_budget_ledger.maximum_material_residual
        for cell in observed.living_cells()
    )
    assert maximum < 3e-10
    return 'logger-neutral; finite 240-step budget/effector stress; max port residual={:.3e}'.format(maximum)


TESTS = (
    ('build_and_frozen_baseline', test_build_and_frozen_baseline),
    ('unattached_exact_lockstep', test_unattached_exact_lockstep),
    ('null_attachment_physical_lockstep', test_null_attachment_physical_lockstep),
    ('sensor_readonly_rng_neutral_no_privileged_labels', test_sensor_readonly_rng_neutral_no_privileged_labels),
    ('port_surface_hides_mutable_body_and_attachment_state', test_port_surface_hides_mutable_body_and_attachment_state),
    ('sensor_schema_physical_shapes', test_sensor_schema_physical_shapes),
    ('port_disabled_fails_closed', test_port_disabled_fails_closed),
    ('invalid_budget_requests_fail_closed', test_invalid_budget_requests_fail_closed),
    ('budget_capped_reserves_and_nonnegative', test_budget_capped_reserves_and_nonnegative),
    ('budget_transfer_and_return_conserve_material', test_budget_transfer_and_return_conserve_material),
    ('multiple_tissues_share_finite_body_budget', test_multiple_tissues_share_finite_body_budget),
    ('material_commit_is_paid_conservative_and_cannot_mint', test_material_commit_is_paid_conservative_and_cannot_mint),
    ('forbidden_effector_rewrites_fail_closed', test_forbidden_effector_rewrites_fail_closed),
    ('motor_effector_is_paid_bounded_and_indirect', test_motor_effector_is_paid_bounded_and_indirect),
    ('transporter_polarity_conserves_channel_material', test_transporter_polarity_conserves_channel_material),
    ('repair_and_quiescence_are_paid_physical_requests', test_repair_and_quiescence_are_paid_physical_requests),
    ('dead_tissue_return_is_conservative', test_dead_tissue_return_is_conservative),
    ('host_death_routes_tissue_into_spatial_corpse', test_host_death_routes_tissue_into_spatial_corpse),
    ('division_recycles_tissue_without_free_inheritance', test_division_recycles_tissue_without_free_inheritance),
    ('pure_05_state_migrates_without_physical_change', test_pure_05_state_migrates_without_physical_change),
    ('save_restore_exact_with_attachment_state', test_save_restore_exact_with_attachment_state),
    ('logger_noninterfering_and_budget_stress_finite', test_logger_noninterfering_and_budget_stress_finite),
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
        rows.append({
            'test': name,
            'passed': int(passed),
            'observed': observed,
            'error': error,
        })

    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=('test', 'passed', 'observed', 'error')
        )
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(row['passed'] for row in rows)
    lines = [
        'SOMA-CELL 0.6-P0 deterministic contract validation',
        'build: {}'.format(p0.BUILD),
        'port schema: {}'.format(p0.PORT_SCHEMA_VERSION),
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
