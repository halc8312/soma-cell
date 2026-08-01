# coding: utf-8
"""Deterministic mechanism validation for SOMA-CELL 0.5.

The suite tests a narrow claim: death can remain a spatially conserved
transformation of matter and hereditary polymers, and extracellular material
DNA can be taken up, digested, restricted, integrated, expressed and inherited
without free Python-level copying.  It does not treat HGT, corpse feeding or a
mobile element as proof of life or open-ended evolution.
"""
from __future__ import division

import csv
import os
import tempfile

import numpy as np

import SOMA_CELL_0_5_pythonista as soma

BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, 'soma_cell_0_5_validation.csv')
TXT_PATH = os.path.join(BASE_DIR, 'SOMA_CELL_0_5_VALIDATION_RESULTS.txt')


def _fmt(value):
    if isinstance(value, (float, np.floating)):
        return '{:.12g}'.format(float(value))
    return str(value)


def _record(rows, name, passed, observed, criterion):
    status = 'PASS' if bool(passed) else 'FAIL'
    print('[{}] {}'.format(status, name), flush=True)
    rows.append({
        'test': name,
        'pass': status,
        'observed': _fmt(observed),
        'criterion': str(criterion),
    })


def _state_equal(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return np.array_equal(np.asarray(a), np.asarray(b))
        except Exception:
            return False
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_state_equal(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_state_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
        return float(a) == float(b)
    return a == b


def _field_material(field):
    return float(np.sum(field.amount))


def _reaction_gene(sequence, reaction):
    sequence = np.asarray(sequence, dtype=np.uint8)
    for spec in soma.g2.parse_genes(sequence):
        if spec['role'] == soma.ROLE_GENERIC and int(spec.get('reaction', -1)) == int(reaction):
            start = int(spec['start'])
            return sequence[start:start + soma.g2.GENE_SPAN].copy()
    raise AssertionError('reaction gene not found')


def _contains_subsequence(sequence, target):
    sequence = np.asarray(sequence, dtype=np.uint8)
    target = np.asarray(target, dtype=np.uint8)
    if len(target) == 0 or len(sequence) < len(target):
        return False
    return any(np.array_equal(sequence[i:i + len(target)], target)
               for i in range(len(sequence) - len(target) + 1))


def _remove_ecology_protein(cell, kind):
    removed = 0.0
    for fingerprint, spec in list(cell.ecology_specs()):
        if int(spec['parameter']) % soma.ECOLOGY_COUNT == int(kind):
            amount = float(cell.proteins.pop(fingerprint, 0.0))
            removed += amount
    cell.pools[soma.POOL_WASTE] += removed
    cell._sync_protein_pool()
    return removed


def _give_ecology_protein_by_redistribution(cell, kind, amount=0.030):
    """Move existing protein mass to a selected encoded ecology protein."""
    target = None
    for fingerprint, spec in cell.ecology_specs():
        if int(spec['parameter']) % soma.ECOLOGY_COUNT == int(kind):
            target = fingerprint
            break
    if target is None:
        raise AssertionError('target ecology gene absent')
    remaining = float(amount)
    for fingerprint in list(cell.proteins):
        if fingerprint == target:
            continue
        take = min(remaining, max(0.0, cell.proteins[fingerprint] - 1e-8))
        cell.proteins[fingerprint] -= take
        remaining -= take
        if remaining <= 1e-12:
            break
    moved = amount - remaining
    cell.proteins[target] = cell.proteins.get(target, 0.0) + moved
    cell._sync_protein_pool()
    return moved


def _add_mobile_gene_materially(cell, world):
    sequence = soma.mobile_element_sequence()
    fragment = soma.DNAFragment(
        sequence, cell.pos, origin_lineage=-77, origin_cell=-77,
        origin_hash='controlled-mobile', mobile=True,
    )
    world.edna.fragments.append(fragment)
    world.initial_total_material = world.total_material()
    fragment = world.edna.fragments.pop()
    ok = cell.integrate_fragment(fragment, world, force=True)
    if not ok:
        raise AssertionError('controlled mobile integration failed')
    return sequence


def _reset_ledger(world):
    world.initial_total_material = world.total_material()
    world.field.injected_material = 0.0
    world.field.dissipated_material = 0.0


def run_validation():
    rows = []

    # 1. The finite material genome contains the full ecology apparatus but no
    # built-in selfish mobile element.
    sequence = soma.founding_genome_05(complete_alt=True)
    counts = soma.ecology_gene_counts(sequence)
    apparatus_ok = bool(
        len(sequence) <= soma.g2.MAX_GENOME_LENGTH
        and np.all(counts[:soma.ECO_MOBILE] >= 1)
        and counts[soma.ECO_MOBILE] == 0
        and soma.g2.sequence_has_novel_path(sequence)
    )
    _record(
        rows, 'material_genome_encodes_ecology_without_seeded_mobile_element',
        apparatus_ok,
        'length={}, counts={}'.format(len(sequence), counts.tolist()),
        'competence through vesicle genes are present, mobile-element gene is absent',
    )

    # 2. Death creates a spatial corpse whose non-ATP matter exactly replaces
    # the living cell in the material ledger.
    corpse_world = soma.EcologicalWorld(seed=2, initial_cells=1)
    corpse_cell = corpse_world.cells[0]
    mass_before = corpse_cell.material_mass()
    corpse_cell.alive = False
    corpse_cell.death_reason = 'controlled-lysis'
    corpse_world._handle_divisions_and_deaths()
    corpse_mass = corpse_world.corpses[0].material_mass() if corpse_world.corpses else 0.0
    corpse_residual = corpse_world.matter_ledger_residual()
    _record(
        rows, 'death_becomes_spatial_conserved_corpse',
        len(corpse_world.cells) == 0 and len(corpse_world.corpses) == 1
        and abs(corpse_mass - mass_before) < 1e-10
        and abs(corpse_residual) < 2e-8,
        'cell={:.12g}, corpse={:.12g}, ledger={:.3e}'.format(
            mass_before, corpse_mass, corpse_residual
        ),
        'dead-cell material persists in one spatial corpse without free creation or loss',
    )

    # 3. The explicit immediate-recycling comparator removes corpse persistence.
    recycle_world = soma.EcologicalWorld(
        seed=2, initial_cells=1,
        config=soma.EcologyConfig(immediate_dead_recycling=True),
    )
    recycle_world.cells[0].alive = False
    recycle_world.cells[0].death_reason = 'controlled-lysis'
    field_before = _field_material(recycle_world.field)
    recycle_world._handle_divisions_and_deaths()
    field_after = _field_material(recycle_world.field)
    _record(
        rows, 'immediate_recycling_ablation_has_no_corpse_or_edna',
        len(recycle_world.corpses) == 0 and len(recycle_world.edna.fragments) == 0
        and field_after > field_before,
        'corpses={}, eDNA={}, field_gain={:.9g}'.format(
            len(recycle_world.corpses), len(recycle_world.edna.fragments),
            field_after - field_before,
        ),
        'the comparator returns dead matter anonymously to particles instead of retaining a body',
    )

    # 4. Corpse hydrolysis transfers matter to the shared field and releases
    # material genome fragments while conserving the cumulative ledger.
    decay_world = soma.EcologicalWorld(
        seed=4, initial_cells=1,
        config=soma.EcologyConfig(corpse_decay_scale=18.0, dna_decay_scale=0.1),
    )
    decay_world.cells[0].alive = False
    decay_world.cells[0].death_reason = 'decay-assay'
    decay_world._handle_divisions_and_deaths()
    body_before = decay_world.corpses[0].material_mass()
    field_before = _field_material(decay_world.field)
    for _ in range(600):
        decay_world._step_corpses(0.1)
        if decay_world.edna.fragments:
            break
    body_after = sum(c.material_mass() for c in decay_world.corpses)
    field_after = _field_material(decay_world.field)
    _record(
        rows, 'corpse_decomposition_releases_matter_and_hereditary_fragments',
        body_after < body_before and field_after > field_before
        and len(decay_world.edna.fragments) > 0
        and abs(decay_world.matter_ledger_residual()) < 4e-5,
        'corpse {:.8g}->{:.8g}, field +{:.8g}, fragments={}, ledger={:.3e}'.format(
            body_before, body_after, field_after - field_before,
            len(decay_world.edna.fragments), decay_world.matter_ledger_residual(),
        ),
        'dead-body hydrolysis and genome release move conserved matter into environmental pools',
    )

    # 5. Extracellular DNA erodes to mineral monomer rather than disappearing.
    edna_world = soma.EcologicalWorld(
        seed=5, initial_cells=0,
        config=soma.EcologyConfig(dna_decay_scale=80.0),
    )
    fragment_sequence = soma.founding_genome_05()[:64]
    edna_world.edna.add_fragment(fragment_sequence, [0.4, 0.4])
    _reset_ledger(edna_world)
    mineral_before = _field_material(edna_world.field)
    edna_before = edna_world.edna.material_total()
    for _ in range(2000):
        edna_world.edna.step(edna_world, 0.1)
        if not edna_world.edna.fragments:
            break
    mineral_after = _field_material(edna_world.field)
    _record(
        rows, 'edna_decay_returns_polymer_matter_to_environment',
        edna_world.edna.material_total() == 0.0
        and mineral_after > mineral_before
        and edna_world.edna.decayed_symbols == len(fragment_sequence)
        and abs(edna_world.matter_ledger_residual()) < 2e-8,
        'eDNA {:.8g}->0, field +{:.8g}, symbols={}, ledger={:.3e}'.format(
            edna_before, mineral_after - mineral_before,
            edna_world.edna.decayed_symbols, edna_world.matter_ledger_residual(),
        ),
        'every eroded DNA symbol becomes material debris in the shared field',
    )

    # 6. Uptake requires competence, proximity and enough ATP.
    uptake_sequence = soma.mobile_element_sequence()
    uptake_off = soma.EcologicalWorld(
        seed=6, initial_cells=1,
        config=soma.EcologyConfig(competence=False, hgt_rate_scale=1e6),
    )
    uptake_off.edna.add_fragment(uptake_sequence, uptake_off.cells[0].pos, mobile=True)
    for _ in range(20):
        uptake_off.cells[0].process_edna(uptake_off, 0.1)
    no_competence = uptake_off.edna.uptaken_fragments == 0

    uptake_low = soma.EcologicalWorld(
        seed=6, initial_cells=1,
        config=soma.EcologyConfig(hgt_rate_scale=1e6),
    )
    uptake_low.cells[0].pools[soma.POOL_ATP] = 0.001
    uptake_low.edna.add_fragment(uptake_sequence, uptake_low.cells[0].pos, mobile=True)
    for _ in range(20):
        uptake_low.cells[0].process_edna(uptake_low, 0.1)
    no_atp = uptake_low.edna.uptaken_fragments == 0

    uptake_on = soma.EcologicalWorld(
        seed=6, initial_cells=1,
        config=soma.EcologyConfig(
            recombination=False, hgt_rate_scale=1e6, dna_digestion=True,
        ),
    )
    uptake_on.cells[0].pools[soma.POOL_ATP] = 0.8
    uptake_on.edna.add_fragment(uptake_sequence, uptake_on.cells[0].pos, mobile=True)
    for _ in range(20):
        uptake_on.cells[0].process_edna(uptake_on, 0.1)
        if uptake_on.edna.uptaken_fragments:
            break
    _record(
        rows, 'dna_uptake_requires_competence_and_atp',
        no_competence and no_atp and uptake_on.edna.uptaken_fragments == 1,
        'no_comp={}, no_ATP={}, enabled_uptake={}'.format(
            no_competence, no_atp, uptake_on.edna.uptaken_fragments
        ),
        'the same nearby polymer is imported only with competence enabled and an ATP budget',
    )

    # 7. With recombination off, imported DNA is physically salvaged as
    # nucleotide material and the genome remains unchanged.
    digest_cell = uptake_on.cells[0]
    genome_length_before = len(digest_cell.genomes[0])
    nucleotide_after = float(digest_cell.pools[soma.POOL_NUCLEOTIDE])
    expected_gain = len(uptake_sequence) * soma.MONOMER_MASS
    _record(
        rows, 'recombination_ablation_digests_imported_dna',
        len(digest_cell.genomes[0]) == genome_length_before
        and uptake_on.hgt_digestions == 1
        and digest_cell.hgt_integrations == 0
        and nucleotide_after >= expected_gain - 1e-9,
        'genome={}, digestions={}, nucleotide={:.8g}'.format(
            len(digest_cell.genomes[0]), uptake_on.hgt_digestions, nucleotide_after
        ),
        'imported polymer becomes nucleotide pool rather than hereditary sequence when recombination is disabled',
    )

    # 8. A missing reaction gene can be inserted as the actual environmental
    # polymer, with no material ledger jump.
    integration_world = soma.EcologicalWorld(seed=8, initial_cells=1)
    recipient = integration_world.cells[0]
    recipient.genomes = [soma.founding_genome_05(complete_alt=False)]
    recipient.genome_lesions = [0.0]
    recipient._refresh_gene_cache()
    recipient.proteins = {
        fingerprint: amount for fingerprint, amount in recipient.proteins.items()
        if fingerprint in recipient.gene_specs
    }
    recipient._sync_protein_pool()
    recipient._clean_control_state()
    target_gene = _reaction_gene(
        soma.founding_genome_05(True), soma.g2.REACTION_INTERMEDIATE_TO_WASTE
    )
    integration_world.edna.add_fragment(
        target_gene, recipient.pos, origin_lineage=91, origin_cell=91,
        origin_hash='target-reaction', mobile=False,
    )
    _reset_ledger(integration_world)
    before_length = len(recipient.genomes[0])
    fragment = integration_world.edna.fragments.pop()
    integration_ok = recipient.integrate_fragment(fragment, integration_world, force=True)
    after_length = len(recipient.genomes[0])
    _record(
        rows, 'physical_hgt_inserts_exact_foreign_symbols_conservatively',
        integration_ok and after_length == before_length + len(target_gene)
        and soma.g2.sequence_has_novel_path(recipient.genomes[0])
        and abs(integration_world.matter_ledger_residual()) < 2e-8,
        'length {}->{}, novel={}, ledger={:.3e}'.format(
            before_length, after_length,
            soma.g2.sequence_has_novel_path(recipient.genomes[0]),
            integration_world.matter_ledger_residual(),
        ),
        'the environmental polymer becomes recipient genome matter without free sequence copying',
    )

    # 9. The transferred sequence is not merely stored: ordinary material
    # translation creates its enzyme and restores the reaction path.
    reaction_before = recipient.reaction_activity(soma.g2.REACTION_INTERMEDIATE_TO_WASTE)
    recipient.pools[soma.POOL_FUEL] = max(recipient.pools[soma.POOL_FUEL], 2.0)
    recipient.pools[soma.POOL_MINERAL] = max(recipient.pools[soma.POOL_MINERAL], 2.0)
    recipient.pools[soma.POOL_ATP] = max(recipient.pools[soma.POOL_ATP], 2.0)
    for _ in range(2400):
        recipient.translate(0.05, integration_world.config)
        if recipient.reaction_activity(soma.g2.REACTION_INTERMEDIATE_TO_WASTE) > 0.05:
            break
    reaction_after = recipient.reaction_activity(soma.g2.REACTION_INTERMEDIATE_TO_WASTE)
    _record(
        rows, 'transferred_gene_is_expressed_by_material_translation',
        reaction_before < 0.01 and reaction_after > 0.05 and recipient.has_novel_path(),
        'reaction {:.8g}->{:.8g}, path={}'.format(
            reaction_before, reaction_after, recipient.has_novel_path()
        ),
        'fuel, mineral, ATP and translator activity turn the integrated sequence into a functional catalyst',
    )

    # 10. The horizontally acquired gene survives physical genome replication
    # and material division, rather than existing only in the recipient object.
    inheritance_world = integration_world
    inheritance_world.config.mutation = False
    inheritance_world.config.external_replicase = True
    recipient.pools[soma.POOL_NUCLEOTIDE] = 3.0
    recipient.pools[soma.POOL_ATP] = 3.0
    replication_steps = 0
    while len(recipient.genomes) < 2 and replication_steps < 6000:
        recipient._replicate_genome(inheritance_world, 0.1, inheritance_world.config)
        replication_steps += 1
    recipient.division_progress = 1.0
    recipient.septum_mass = max(recipient.septum_mass, 0.12)
    daughters = recipient.split(inheritance_world)
    inherited = bool(
        daughters is not None and len(daughters) == 2
        and all(any(soma.g2.sequence_has_novel_path(g) for g in d.genomes) for d in daughters)
    )
    _record(
        rows, 'hgt_gene_persists_through_replication_and_division',
        inherited and abs(inheritance_world.division_residual) < 3e-8,
        'replication_steps={}, daughters={}, division_residual={:.3e}'.format(
            replication_steps, 0 if daughters is None else len(daughters),
            inheritance_world.division_residual,
        ),
        'an acquired sequence is copied with monomer/ATP cost and inherited by both material daughters',
    )

    # 11. Gene-derived restriction lowers the actual integration frequency for
    # the same foreign mobile fragment.
    def integration_frequency(restriction_enabled, trials=120):
        successes = 0
        for seed in range(1000, 1000 + trials):
            world = soma.EcologicalWorld(
                seed=seed, initial_cells=1,
                config=soma.EcologyConfig(
                    restriction=restriction_enabled, dna_digestion=True,
                ),
            )
            cell = world.cells[0]
            cell.pools[soma.POOL_ATP] = 0.9
            fragment = soma.DNAFragment(
                soma.mobile_element_sequence(), cell.pos,
                origin_lineage=-10, origin_cell=-10, mobile=True,
            )
            if cell.integrate_fragment(fragment, world, force=False):
                successes += 1
        return successes

    unrestricted = integration_frequency(False)
    restricted = integration_frequency(True)
    _record(
        rows, 'restriction_system_reduces_foreign_integration_frequency',
        restricted < unrestricted and unrestricted - restricted >= 8,
        'unrestricted={}/120, restricted={}/120'.format(unrestricted, restricted),
        'the encoded restriction apparatus suppresses actual foreign-sequence integration',
    )

    # 12. Necrophagy transfers physical corpse matter and pays ATP.
    necro_world = soma.EcologicalWorld(
        seed=12, initial_cells=1,
        config=soma.EcologyConfig(necrophagy_rate_scale=30.0),
    )
    scavenger = necro_world.cells[0]
    donor = soma.EcologicalProtoCell(99, necro_world.rng, position=scavenger.pos, bootstrap=True)
    corpse = soma.Corpse.from_cell(0, donor)
    corpse.pos = scavenger.pos.copy()
    necro_world.corpses.append(corpse)
    _reset_ledger(necro_world)
    corpse_before = corpse.material_mass()
    atp_before = float(scavenger.pools[soma.POOL_ATP])
    fuel_mineral_before = float(scavenger.pools[soma.POOL_FUEL] + scavenger.pools[soma.POOL_MINERAL])
    scavenger.process_corpse_contact(necro_world, 0.8)
    _record(
        rows, 'necrophagy_transfers_corpse_matter_with_atp_cost',
        corpse.material_mass() < corpse_before
        and scavenger.pools[soma.POOL_ATP] < atp_before
        and scavenger.pools[soma.POOL_FUEL] + scavenger.pools[soma.POOL_MINERAL] > fuel_mineral_before
        and scavenger.last_necrophagy_mass > 0.0
        and abs(necro_world.matter_ledger_residual()) < 3e-8,
        'consumed={:.8g}, ATP {:.8g}->{:.8g}, ledger={:.3e}'.format(
            scavenger.last_necrophagy_mass, atp_before,
            scavenger.pools[soma.POOL_ATP], necro_world.matter_ledger_residual(),
        ),
        'corpse substrate becomes intracellular fuel/mineral while the scavenger pays an energetic handling cost',
    )

    # 13. Detox apparatus reduces toxic co-uptake for equal corpse feeding.
    detox_template = soma.EcologicalWorld(
        seed=13, initial_cells=1,
        config=soma.EcologyConfig(necrophagy_rate_scale=18.0),
    )
    donor = soma.EcologicalProtoCell(98, detox_template.rng, position=detox_template.cells[0].pos, bootstrap=True)
    donor.pools[soma.POOL_WASTE] += 0.50
    detox_template.corpses = [soma.Corpse.from_cell(0, donor)]
    detox_template.corpses[0].pos = detox_template.cells[0].pos.copy()
    with_detox = detox_template.clone()
    without_detox = detox_template.clone()
    _remove_ecology_protein(without_detox.cells[0], soma.ECO_DETOX)
    # Reset each ledger after the controlled protein-to-waste redistribution.
    _reset_ledger(with_detox)
    _reset_ledger(without_detox)
    with_detox.cells[0].process_corpse_contact(with_detox, 0.8)
    without_detox.cells[0].process_corpse_contact(without_detox, 0.8)
    _record(
        rows, 'detox_apparatus_reduces_necrotoxin_uptake',
        with_detox.cells[0].last_necrophagy_mass > 0.0
        and without_detox.cells[0].last_necrophagy_mass > 0.0
        and with_detox.cells[0].necrotoxin_uptake < without_detox.cells[0].necrotoxin_uptake,
        'detox={:.8g}, no_detox={:.8g}, feed={:.8g}/{:.8g}'.format(
            with_detox.cells[0].necrotoxin_uptake,
            without_detox.cells[0].necrotoxin_uptake,
            with_detox.cells[0].last_necrophagy_mass,
            without_detox.cells[0].last_necrophagy_mass,
        ),
        'for comparable corpse consumption, encoded detoxification lowers reactive matter entering the cell',
    )

    # 14. A mobile element copies its own physical sequence into eDNA using
    # nucleotide material and ATP.
    mobile_world = soma.EcologicalWorld(
        seed=14, initial_cells=1,
        config=soma.EcologyConfig(mobile_rate_scale=1e7),
    )
    mobile_cell = mobile_world.cells[0]
    mobile_sequence = _add_mobile_gene_materially(mobile_cell, mobile_world)
    moved = _give_ecology_protein_by_redistribution(mobile_cell, soma.ECO_MOBILE, 0.030)
    mobile_cell.pools[soma.POOL_NUCLEOTIDE] = max(mobile_cell.pools[soma.POOL_NUCLEOTIDE], 0.8)
    mobile_cell.pools[soma.POOL_ATP] = max(mobile_cell.pools[soma.POOL_ATP], 0.8)
    _reset_ledger(mobile_world)
    nuc_before = float(mobile_cell.pools[soma.POOL_NUCLEOTIDE])
    atp_before = float(mobile_cell.pools[soma.POOL_ATP])
    for _ in range(30):
        mobile_cell.export_mobile_element(mobile_world, 0.1)
        if mobile_world.mobile_exports:
            break
    exported_mobile = [f for f in mobile_world.edna.fragments if f.mobile]
    _record(
        rows, 'mobile_element_export_is_material_and_atp_paid',
        moved > 0.0 and len(exported_mobile) >= 1
        and np.array_equal(exported_mobile[-1].sequence, mobile_sequence)
        and mobile_cell.pools[soma.POOL_NUCLEOTIDE] < nuc_before
        and mobile_cell.pools[soma.POOL_ATP] < atp_before
        and abs(mobile_world.matter_ledger_residual()) < 3e-8,
        'exports={}, nucleotide {:.8g}->{:.8g}, ATP {:.8g}->{:.8g}, ledger={:.3e}'.format(
            mobile_world.mobile_exports, nuc_before,
            mobile_cell.pools[soma.POOL_NUCLEOTIDE], atp_before,
            mobile_cell.pools[soma.POOL_ATP], mobile_world.matter_ledger_residual(),
        ),
        'a selfish gene buds an extracellular copy only by consuming real monomer and energy',
    )

    # 15. The exported mobile fragment can infect a separate recipient and
    # become part of its genome.
    exported = exported_mobile[-1]
    mobile_world.edna.fragments.remove(exported)
    recipient_world = soma.EcologicalWorld(seed=15, initial_cells=1)
    recipient_mobile = recipient_world.cells[0]
    recipient_world.edna.fragments.append(exported)
    _reset_ledger(recipient_world)
    exported = recipient_world.edna.fragments.pop()
    infected = recipient_mobile.integrate_fragment(exported, recipient_world, force=True)
    recipient_counts = soma.ecology_gene_counts(recipient_mobile.genomes[0])
    _record(
        rows, 'mobile_element_can_enter_a_separate_material_genome',
        infected and recipient_counts[soma.ECO_MOBILE] >= 1
        and abs(recipient_world.matter_ledger_residual()) < 3e-8,
        'infected={}, mobile_genes={}, ledger={:.3e}'.format(
            infected, recipient_counts[soma.ECO_MOBILE],
            recipient_world.matter_ledger_residual(),
        ),
        'a physically exported polymer becomes a heritable recipient sequence rather than a copied metadata flag',
    )

    # 16. A gene travels through the complete donor -> corpse -> extracellular
    # fragment -> recipient route.
    route_world = soma.EcologicalWorld(
        seed=16, initial_cells=2,
        config=soma.EcologyConfig(corpse_decay_scale=40.0, dna_decay_scale=0.01),
    )
    donor, route_recipient = route_world.cells
    route_recipient.genomes = [soma.founding_genome_05(complete_alt=False)]
    route_recipient.genome_lesions = [0.0]
    route_recipient._refresh_gene_cache()
    route_recipient.proteins = {
        fingerprint: amount for fingerprint, amount in route_recipient.proteins.items()
        if fingerprint in route_recipient.gene_specs
    }
    route_recipient._sync_protein_pool()
    _reset_ledger(route_world)
    donor.alive = False
    donor.death_reason = 'donor-lysis'
    route_world._handle_divisions_and_deaths()
    target = _reaction_gene(soma.founding_genome_05(True), soma.g2.REACTION_INTERMEDIATE_TO_WASTE)
    matching = None
    for _ in range(1500):
        route_world._step_corpses(0.1)
        for fragment in route_world.edna.fragments:
            if _contains_subsequence(fragment.sequence, target):
                matching = fragment
                break
        if matching is not None:
            break
    routed = False
    if matching is not None:
        route_world.edna.fragments.remove(matching)
        routed = route_recipient.integrate_fragment(matching, route_world, force=True)
    _record(
        rows, 'donor_corpse_can_transfer_a_functional_gene_to_recipient',
        matching is not None and routed
        and soma.g2.sequence_has_novel_path(route_recipient.genomes[0])
        and abs(route_world.matter_ledger_residual()) < 5e-5,
        'fragment_found={}, integrated={}, novel={}, ledger={:.3e}'.format(
            matching is not None, routed,
            soma.g2.sequence_has_novel_path(route_recipient.genomes[0]),
            route_world.matter_ledger_residual(),
        ),
        'the actual donor polymer survives corpse release and becomes recipient hereditary material',
    )

    # 17. Removing eDNA persistence recycles genome mass but prevents a DNA pool.
    no_edna_world = soma.EcologicalWorld(
        seed=17, initial_cells=1,
        config=soma.EcologyConfig(
            extracellular_dna=False, corpse_decay_scale=80.0,
        ),
    )
    no_edna_world.cells[0].alive = False
    no_edna_world.cells[0].death_reason = 'no-edna-assay'
    no_edna_world._handle_divisions_and_deaths()
    for _ in range(2000):
        no_edna_world._step_corpses(0.1)
        if no_edna_world.corpse_genomes_recycled > 0:
            break
    _record(
        rows, 'edna_ablation_recycles_genome_matter_without_information_persistence',
        no_edna_world.corpse_genomes_recycled > 0
        and len(no_edna_world.edna.fragments) == 0
        and abs(no_edna_world.matter_ledger_residual()) < 5e-5,
        'recycled_genomes={}, eDNA={}, ledger={:.3e}'.format(
            no_edna_world.corpse_genomes_recycled,
            len(no_edna_world.edna.fragments),
            no_edna_world.matter_ledger_residual(),
        ),
        'the ablation removes environmental heredity but returns its material to mineral debris',
    )

    # 18. The full 0.5 ecology pays its costs and still reaches physical G1.
    division_world = soma.EcologicalWorld(seed=101, initial_cells=1)
    dt = 1.0 / soma.SIM_HZ
    first_division_age = None
    max_residual = 0.0
    for _ in range(int(430 * soma.SIM_HZ)):
        division_world.step(dt)
        max_residual = max(max_residual, abs(division_world.matter_ledger_residual()))
        if division_world.divisions > 0:
            first_division_age = division_world.age
            break
    _record(
        rows, 'full_ecological_cell_reaches_material_g1_division',
        first_division_age is not None and division_world.divisions >= 1
        and max_residual < 8e-5,
        'division_age={}, cells={}, max_ledger={:.3e}'.format(
            first_division_age, len(division_world.living_cells()), max_residual
        ),
        'corpse/HGT apparatus costs do not prevent one complete material growth-replication-division cycle',
    )

    # 19. Save/restore is exact even with a corpse and extracellular fragments.
    persistence_world = soma.EcologicalWorld(
        seed=19, initial_cells=2,
        config=soma.EcologyConfig(corpse_decay_scale=30.0, dna_decay_scale=0.2),
    )
    persistence_world.cells[0].alive = False
    persistence_world.cells[0].death_reason = 'persistence-assay'
    persistence_world._handle_divisions_and_deaths()
    for _ in range(200):
        persistence_world.step(0.05)
        if persistence_world.edna.fragments:
            break
    with tempfile.TemporaryDirectory() as tmp:
        save_path = os.path.join(tmp, 'world.pkl')
        persistence_world.save(save_path)
        restored_world = soma.EcologicalWorld.load(save_path)
        saved_equal = _state_equal(persistence_world.state_dict(), restored_world.state_dict())
        for _ in range(120):
            persistence_world.step(0.05)
            restored_world.step(0.05)
        continued_equal = _state_equal(
            persistence_world.state_dict(), restored_world.state_dict()
        )
    _record(
        rows, 'save_restore_preserves_corpse_edna_and_future_exactly',
        saved_equal and continued_equal,
        'saved_equal={}, continued_equal={}'.format(saved_equal, continued_equal),
        'pickle restoration resumes the identical corpse, DNA, cell and RNG trajectory',
    )

    # 20. Logging observes without consuming RNG or modifying ecology.
    log_template = soma.EcologicalWorld(seed=20, initial_cells=2)
    observed = log_template.clone()
    control = log_template.clone()
    with tempfile.TemporaryDirectory() as tmp:
        logger = soma.LongRunLogger(observed, path=os.path.join(tmp, 'log.csv'))
        for step in range(180):
            observed.step(0.05)
            control.step(0.05)
            if step % 7 == 0:
                logger.log(observed, fps=17.0, sim_rate=0.9, force=True)
        logging_equal = _state_equal(observed.state_dict(), control.state_dict())
    finite_ok = observed.finite() and np.isfinite(observed.matter_ledger_residual())
    _record(
        rows, 'logger_is_noninterfering_and_state_remains_finite',
        logging_equal and finite_ok and abs(observed.matter_ledger_residual()) < 8e-5,
        'state_equal={}, finite={}, ledger={:.3e}'.format(
            logging_equal, finite_ok, observed.matter_ledger_residual()
        ),
        'long-run telemetry consumes no ecological randomness and changes no simulated state',
    )

    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=('test', 'pass', 'observed', 'criterion'))
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(item['pass'] == 'PASS' for item in rows)
    lines = [
        'SOMA-CELL 0.5 deterministic validation',
        'build: {}'.format(soma.BUILD),
        'result: {}/{} PASS'.format(passed, len(rows)),
        '',
    ]
    for item in rows:
        lines.extend([
            '[{}] {}'.format(item['pass'], item['test']),
            '  observed: {}'.format(item['observed']),
            '  criterion: {}'.format(item['criterion']),
        ])
    with open(TXT_PATH, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')

    if passed != len(rows):
        raise SystemExit('validation failed: {}/{} PASS'.format(passed, len(rows)))
    return rows


if __name__ == '__main__':
    run_validation()
