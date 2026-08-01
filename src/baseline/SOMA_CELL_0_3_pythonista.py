# coding: utf-8
"""
SOMA-CELL 0.3 — Evolvable Damage Economy / 進化可能な損傷・修復・生活史

Pythonista 3 / CPython + NumPy research prototype.

0.3 keeps the material membrane, metabolism, material genome, internal
replicase, translation and mutation grammar of SOMA-CELL 0.2.  It adds a
materially paid damage economy instead of a chronological death timer:

* metabolic and environmental flux converts active proteins into damaged
  proteins, damaged proteins into aggregates, and functional membrane into an
  oxidised state;
* genome molecules carry repairable lesion loads that alter replication error;
* regulator genes encode antioxidant, chaperone, protease, genome-repair,
  damage-segregation, proofreading, quiescence and membrane-repair functions;
* repair consumes ATP and sometimes recyclable matter, competing directly with
  growth, translation and genome copying;
* division can asymmetrically partition damage, producing one cleaner and one
  dirtier daughter while conserving damaged matter;
* there is no maximum age or division-count death rule.  Aging, negligible
  senescence, rejuvenation and collapse are outcomes of flux, repair and
  partitioning.

The model remains a coarse-grained artificial chemistry.  The reaction grammar,
repair-function vocabulary and physical laws are human-designed.  It is not a
claim of biological life or open-ended evolution.
"""

from __future__ import division

import csv
import gc
import hashlib
import json
import math
import os
import pickle
import time

import numpy as np

import SOMA_CELL_0_1_pythonista as base
import SOMA_CELL_0_2_pythonista as g2


BUILD = 'SOMA-CELL 0.3.0'
SAVE_VERSION = 3
BASE_DIR = os.path.dirname(__file__)
SAVE_FILE = os.path.join(BASE_DIR, 'soma_cell_0_3.pkl')
LOG_FILE = os.path.join(BASE_DIR, 'soma_cell_0_3_longrun.csv')
REPORT_FILE = os.path.join(BASE_DIR, 'soma_cell_0_3_report.txt')
SESSION_FILE = os.path.join(BASE_DIR, 'soma_cell_0_3_sessions.csv')

SIM_HZ = 20.0
AUTO_SAVE_INTERVAL = 30.0
LOG_INTERVAL = 10.0
MAX_CELLS = 14

# Re-export 0.2 material and genome constants.
PARTICLE_FUEL = g2.PARTICLE_FUEL
PARTICLE_MINERAL = g2.PARTICLE_MINERAL
PARTICLE_WASTE = g2.PARTICLE_WASTE
PARTICLE_ALT = g2.PARTICLE_ALT
PARTICLE_NAMES = g2.PARTICLE_NAMES

POOL_FUEL = g2.POOL_FUEL
POOL_MINERAL = g2.POOL_MINERAL
POOL_ATP = g2.POOL_ATP
POOL_MEM_PRECURSOR = g2.POOL_MEM_PRECURSOR
POOL_CATALYST = g2.POOL_CATALYST
POOL_TRANSPORTER_PRECURSOR = g2.POOL_TRANSPORTER_PRECURSOR
POOL_WASTE = g2.POOL_WASTE
POOL_ALT = g2.POOL_ALT
POOL_INTERMEDIATE = g2.POOL_INTERMEDIATE
POOL_NUCLEOTIDE = g2.POOL_NUCLEOTIDE

# New material pools.  Damaged protein retains molecular identity in a dict;
# the aggregate and reactive pools are explicit matter classes.
POOL_DAMAGED_PROTEIN = 10
POOL_AGGREGATE = 11
POOL_REACTIVE = 12
POOL_COUNT = 13

CHANNEL_FUEL = g2.CHANNEL_FUEL
CHANNEL_MINERAL = g2.CHANNEL_MINERAL
CHANNEL_WASTE = g2.CHANNEL_WASTE
CHANNEL_ALT = g2.CHANNEL_ALT
CHANNEL_COUNT = g2.CHANNEL_COUNT

MEMBRANE_SEGMENTS = g2.MEMBRANE_SEGMENTS
BASE_RADIUS = g2.BASE_RADIUS
MIN_RADIUS = g2.MIN_RADIUS
MAX_RADIUS = g2.MAX_RADIUS
INITIAL_MEMBRANE_MASS = g2.INITIAL_MEMBRANE_MASS
MONOMER_MASS = g2.MONOMER_MASS
REPLICATION_ATP_PER_SYMBOL = g2.REPLICATION_ATP_PER_SYMBOL

ROLE_ENERGY = g2.ROLE_ENERGY
ROLE_MEMBRANE = g2.ROLE_MEMBRANE
ROLE_TRANSPORTER = g2.ROLE_TRANSPORTER
ROLE_REPLICASE = g2.ROLE_REPLICASE
ROLE_TRANSLATOR = g2.ROLE_TRANSLATOR
ROLE_NUCLEOTIDE = g2.ROLE_NUCLEOTIDE
ROLE_GENERIC = g2.ROLE_GENERIC
ROLE_REGULATOR = g2.ROLE_REGULATOR

# The regulator parameter selects a repair/life-history function.  This reuses
# the 0.2 alphabet and remains backwards compatible with 0.2 genomes.
REPAIR_ANTIOXIDANT = 0
REPAIR_CHAPERONE = 1
REPAIR_PROTEASE = 2
REPAIR_GENOME = 3
REPAIR_SEGREGATION = 4
REPAIR_PROOFREADING = 5
REPAIR_QUIESCENCE = 6
REPAIR_MEMBRANE = 7
REPAIR_NAMES = (
    'antioxidant', 'chaperone', 'protease', 'genome-repair',
    'damage-segregation', 'proofreading', 'quiescence', 'membrane-repair',
)

clamp = g2.clamp
wrapped_delta = g2.wrapped_delta
torus_distance = g2.torus_distance
unit_vector = g2.unit_vector
finite_array = g2.finite_array
circular_smooth = g2.circular_smooth
_atomic_pickle = g2._atomic_pickle
_memory_peak_mb_estimate = g2._memory_peak_mb_estimate


def founding_genome_03():
    """0.2 viable core plus one gene for every repair function.

    The 0.2 core already contains regulator parameter 0.  Parameters 1..7 are
    appended.  Mutation can delete, duplicate or alter any of them.
    """
    genes = [g2.founding_genome()]
    settings = (
        (REPAIR_CHAPERONE, 5, 5, 4),
        (REPAIR_PROTEASE, 4, 4, 4),
        (REPAIR_GENOME, 4, 5, 6),
        (REPAIR_SEGREGATION, 3, 4, 4),
        (REPAIR_PROOFREADING, 4, 4, 6),
        (REPAIR_QUIESCENCE, 3, 4, 4),
        (REPAIR_MEMBRANE, 4, 5, 4),
    )
    for parameter, promoter, efficiency, fidelity in settings:
        genes.append(g2.make_gene(
            ROLE_REGULATOR,
            parameter=parameter,
            promoter=promoter,
            efficiency=efficiency,
            fidelity=fidelity,
            regulator=parameter,
        ))
    return np.concatenate(genes).astype(np.uint8)


def _sequence_hash(sequence):
    return hashlib.sha1(np.asarray(sequence, dtype=np.uint8).tobytes()).hexdigest()[:12]


def regulator_trait_vector(sequence):
    """Decode repair functions from one genome into bounded 0..1 traits."""
    raw = np.zeros(8, dtype=float)
    for spec in g2.parse_genes(sequence):
        if spec['role'] != ROLE_REGULATOR:
            continue
        kind = int(spec['parameter']) % 8
        raw[kind] += spec['promoter'] * spec['efficiency'] * spec.get('copy_number', 1)
    return raw / (1.65 + raw)


def essential_gene_score(sequence):
    """Geometric-like viability score for accelerated assays."""
    activity = np.zeros(6, dtype=float)
    roles = (
        ROLE_ENERGY, ROLE_MEMBRANE, ROLE_TRANSPORTER,
        ROLE_REPLICASE, ROLE_TRANSLATOR, ROLE_NUCLEOTIDE,
    )
    for spec in g2.parse_genes(sequence):
        if spec['role'] in roles:
            index = roles.index(spec['role'])
            activity[index] += spec['promoter'] * spec['efficiency']
    bounded = activity / (1.25 + activity)
    return float(np.prod(np.clip(bounded, 1e-4, 1.0)) ** (1.0 / len(bounded)))


class LifeHistoryConfig(g2.GeneticConfig):
    """Ablation and environment switches for the damage economy."""

    def __init__(
        self,
        membrane_synthesis=True,
        catalyst_synthesis=True,
        transport=True,
        targeted_repair=True,
        division=True,
        waste_export=True,
        external_inflow=True,
        environmental_damage=True,
        gene_expression=True,
        genome_replication=True,
        mutation=True,
        variable_length=True,
        gene_duplication=True,
        generic_reactions=True,
        composition_inheritance=True,
        external_replicase=False,
        external_translator=False,
        mutation_rate=0.0025,
        structural_rate=0.08,
        alt_niche=False,
        endogenous_damage=True,
        protein_repair=True,
        genome_repair=True,
        membrane_repair=True,
        damage_segregation=True,
        proofreading=True,
        quiescence=True,
        stress_mode='pulsed',
        damage_rate_scale=1.0,
        repair_cost_scale=1.0,
        forced_symmetric_damage=False,
    ):
        super(LifeHistoryConfig, self).__init__(
            membrane_synthesis=membrane_synthesis,
            catalyst_synthesis=catalyst_synthesis,
            transport=transport,
            targeted_repair=targeted_repair,
            division=division,
            waste_export=waste_export,
            external_inflow=external_inflow,
            environmental_damage=environmental_damage,
            gene_expression=gene_expression,
            genome_replication=genome_replication,
            mutation=mutation,
            variable_length=variable_length,
            gene_duplication=gene_duplication,
            generic_reactions=generic_reactions,
            composition_inheritance=composition_inheritance,
            external_replicase=external_replicase,
            external_translator=external_translator,
            mutation_rate=mutation_rate,
            structural_rate=structural_rate,
            alt_niche=alt_niche,
        )
        self.endogenous_damage = bool(endogenous_damage)
        self.protein_repair = bool(protein_repair)
        self.genome_repair = bool(genome_repair)
        self.membrane_repair = bool(membrane_repair)
        self.damage_segregation = bool(damage_segregation)
        self.proofreading = bool(proofreading)
        self.quiescence = bool(quiescence)
        self.stress_mode = str(stress_mode)
        self.damage_rate_scale = float(damage_rate_scale)
        self.repair_cost_scale = float(repair_cost_scale)
        self.forced_symmetric_damage = bool(forced_symmetric_damage)

    @classmethod
    def from_state(cls, state):
        return cls(**dict(state))


class DamageProtoCell(g2.GeneticProtoCell):
    """Materially hereditary protocell with damage and paid repair."""

    def __init__(self, cell_id, rng, position=None, generation=0, lineage=0,
                 bootstrap=True):
        super(DamageProtoCell, self).__init__(
            cell_id, rng, position=position, generation=generation,
            lineage=lineage, bootstrap=False,
        )
        if len(self.pools) < POOL_COUNT:
            pools = np.zeros((POOL_COUNT,), dtype=float)
            pools[:len(self.pools)] = self.pools
            self.pools = pools
        self.pools[POOL_DAMAGED_PROTEIN] = 0.0
        self.pools[POOL_AGGREGATE] = 0.0
        self.pools[POOL_REACTIVE] = 0.010

        self.damaged_proteins = {}
        self.membrane_oxidation = np.zeros(MEMBRANE_SEGMENTS, dtype=float)
        self.genome_lesions = []
        self.replication_template_lesion = 0.0
        self.current_stress = 0.0
        self._defer_damage_viability = False

        self.last_damage_generated = 0.0
        self.last_repair_atp = 0.0
        self.last_repair_flux = np.zeros(8, dtype=float)
        self.cumulative_damage_generated = 0.0
        self.cumulative_repair_atp = 0.0
        self.cumulative_recycled_damage = 0.0
        self.cumulative_oxidant_neutralised = 0.0
        self.cumulative_genome_repairs = 0.0
        self.cumulative_proofreading_atp = 0.0
        self.cumulative_segregation_atp = 0.0
        self.proteostatic_collapse_timer = 0.0
        self.genomic_collapse_timer = 0.0
        self.reactive_crisis_timer = 0.0
        self.birth_damage = 0.0
        self.pole_age = 0
        self.parent_damage_at_birth = 0.0
        self.last_effective_error_rate = 0.0
        self.last_quiescence = 0.0
        self.last_segregation_strength = 0.0
        self.last_division_damage_pair = None
        self.functional_age_ema = 0.0
        self.functional_age_slope = 0.0
        self._previous_functional_age = 0.0

        if bootstrap:
            genome = founding_genome_03()
            self.genomes = [genome]
            self.genome_lesions = [0.0]
            self._refresh_gene_cache()
            initial_by_role = {
                ROLE_ENERGY: 0.072,
                ROLE_MEMBRANE: 0.043,
                ROLE_TRANSPORTER: 0.038,
                ROLE_REPLICASE: 0.046,
                ROLE_TRANSLATOR: 0.058,
                ROLE_NUCLEOTIDE: 0.038,
                ROLE_GENERIC: 0.026,
                ROLE_REGULATOR: 0.010,
            }
            for fingerprint, spec in self.gene_specs.items():
                self.proteins[fingerprint] = (
                    self.proteins.get(fingerprint, 0.0)
                    + initial_by_role[spec['role']]
                )
            self.pools[POOL_FUEL] = max(self.pools[POOL_FUEL], 0.44)
            self.pools[POOL_MINERAL] = max(self.pools[POOL_MINERAL], 0.42)
            self.pools[POOL_ATP] = max(self.pools[POOL_ATP], 0.52)
            self.pools[POOL_NUCLEOTIDE] = max(self.pools[POOL_NUCLEOTIDE], 0.29)
            self._sync_protein_pool()
            self._sync_damage_pool()

    # ------------------------------------------------------------------
    # Damage bookkeeping and functional state
    # ------------------------------------------------------------------

    def _sync_damage_pool(self):
        clean = {}
        for fingerprint, amount in self.damaged_proteins.items():
            amount = float(amount)
            if np.isfinite(amount) and amount > 1e-11:
                clean[int(fingerprint)] = amount
        self.damaged_proteins = clean
        self.pools[POOL_DAMAGED_PROTEIN] = float(sum(clean.values()))
        self.pools[POOL_AGGREGATE] = max(0.0, float(self.pools[POOL_AGGREGATE]))
        self.pools[POOL_REACTIVE] = max(0.0, float(self.pools[POOL_REACTIVE]))

    def material_mass(self):
        # 0.2 sums every non-ATP pool, so the three new matter classes are
        # already part of the ledger.
        return super(DamageProtoCell, self).material_mass()

    def osmolyte(self):
        return float(
            super(DamageProtoCell, self).osmolyte()
            + 0.18 * self.pools[POOL_DAMAGED_PROTEIN]
            + 0.08 * self.pools[POOL_AGGREGATE]
            + 0.70 * self.pools[POOL_REACTIVE]
        )

    def closure_array(self):
        base_closure = super(DamageProtoCell, self).closure_array()
        impairment = np.exp(-1.65 * np.clip(self.membrane_oxidation, 0.0, 2.5))
        return np.clip(base_closure * impairment, 0.0, 1.0)

    def raw_repair_activity(self, kind):
        total = 0.0
        for fingerprint, amount in self.proteins.items():
            spec = self.gene_specs.get(fingerprint)
            if (
                spec is not None
                and spec['role'] == ROLE_REGULATOR
                and int(spec['parameter']) % 8 == int(kind)
            ):
                total += amount * spec['efficiency'] * spec['promoter']
        aggregate_inhibition = 1.0 / (1.0 + 2.6 * self.aggregate_concentration())
        return float(total / 0.014 * aggregate_inhibition)

    def role_activity(self, role):
        value = super(DamageProtoCell, self).role_activity(role)
        if role == ROLE_REGULATOR:
            return value
        return float(value * self.proteostasis_factor() * self.genome_function_factor())

    def protein_material(self):
        return float(
            self.pools[POOL_CATALYST]
            + self.pools[POOL_DAMAGED_PROTEIN]
            + self.pools[POOL_AGGREGATE]
        )

    def aggregate_concentration(self):
        volume = max(0.20, (self.radius / BASE_RADIUS) ** 2)
        return float(self.pools[POOL_AGGREGATE] / volume)

    def reactive_concentration(self):
        volume = max(0.20, (self.radius / BASE_RADIUS) ** 2)
        return float(self.pools[POOL_REACTIVE] / volume)

    def mean_genome_lesion(self):
        if not self.genome_lesions:
            return 1.0
        return float(np.mean(self.genome_lesions))

    def genome_function_factor(self):
        return float(1.0 / (1.0 + 0.85 * self.mean_genome_lesion()))

    def proteostasis_factor(self):
        active = max(0.0, self.pools[POOL_CATALYST])
        damaged = self.pools[POOL_DAMAGED_PROTEIN]
        aggregate = self.pools[POOL_AGGREGATE]
        functional_fraction = active / max(1e-9, active + damaged + aggregate)
        toxicity = 1.0 / (1.0 + 3.6 * self.aggregate_concentration())
        return float(clamp(functional_fraction * toxicity, 0.02, 1.0))

    def damage_burden(self):
        protein_total = max(0.08, self.protein_material())
        protein_damage = (
            self.pools[POOL_DAMAGED_PROTEIN] + 1.8 * self.pools[POOL_AGGREGATE]
        ) / protein_total
        membrane_damage = float(np.average(
            np.clip(self.membrane_oxidation, 0.0, 2.5),
            weights=np.maximum(self.membrane, 1e-9),
        ))
        reactive = self.reactive_concentration()
        genome = self.mean_genome_lesion()
        return float(clamp(
            0.36 * protein_damage
            + 0.24 * membrane_damage
            + 0.20 * reactive
            + 0.20 * genome,
            0.0, 4.0,
        ))

    def functional_age(self):
        # This is an observable damage index, not a death timer.
        return float(self.damage_burden() * 100.0)

    def reaction_loop_strength(self):
        return float(
            super(DamageProtoCell, self).reaction_loop_strength()
            * self.proteostasis_factor()
            * self.genome_function_factor()
            * math.exp(-0.45 * self.reactive_concentration())
        )

    def _protein_need(self, spec):
        if spec['role'] != ROLE_REGULATOR:
            return super(DamageProtoCell, self)._protein_need(spec)
        kind = int(spec['parameter']) % 8
        damage = self.damage_burden()
        if kind == REPAIR_ANTIOXIDANT:
            need = 0.030 + 2.8 * self.reactive_concentration() + 0.6 * self.current_stress
        elif kind == REPAIR_CHAPERONE:
            need = 0.025 + 3.0 * self.pools[POOL_DAMAGED_PROTEIN]
        elif kind == REPAIR_PROTEASE:
            need = 0.022 + 2.1 * self.pools[POOL_DAMAGED_PROTEIN] + 3.3 * self.pools[POOL_AGGREGATE]
        elif kind == REPAIR_GENOME:
            need = 0.022 + 1.5 * self.mean_genome_lesion()
        elif kind == REPAIR_SEGREGATION:
            need = 0.016 + damage * (0.4 + 1.4 * self.division_progress)
        elif kind == REPAIR_PROOFREADING:
            need = 0.022 + 0.5 * self.mean_genome_lesion() + (0.6 if self.replication_template is not None else 0.0)
        elif kind == REPAIR_QUIESCENCE:
            need = 0.016 + 1.5 * max(0.0, damage - 0.20) + 0.7 * self.current_stress
        else:
            need = 0.022 + 2.2 * float(np.mean(self.membrane_oxidation)) + 1.4 * (1.0 - self.closure())
        return clamp(need, 0.012, 2.2)

    def quiescence_level(self, config):
        if not config.quiescence:
            return 0.0
        signal = self.raw_repair_activity(REPAIR_QUIESCENCE)
        need = max(0.0, self.damage_burden() - 0.12) + 0.35 * self.current_stress
        return float(clamp((signal / (0.8 + signal)) * need * 1.45, 0.0, 0.82))

    # ------------------------------------------------------------------
    # Translation, replication and damage generation
    # ------------------------------------------------------------------

    def translate(self, dt, config):
        self.last_translation = 0.0
        if not config.gene_expression or not self.genomes:
            return
        translator = self.role_activity(ROLE_TRANSLATOR)
        if config.external_translator:
            translator += 0.75
        if translator <= 1e-5 or not self.gene_specs:
            return
        total_weight = 0.0
        weighted = []
        for fingerprint, spec in self.gene_specs.items():
            weight = spec['promoter'] * spec.get('copy_number', 1) * self._protein_need(spec)
            weighted.append((fingerprint, spec, weight))
            total_weight += weight
        if total_weight <= 0.0:
            return
        quiescence = self.quiescence_level(config)
        self.last_quiescence = quiescence
        translation_capacity = dt * 0.0060 * translator * (1.0 - 0.72 * quiescence)
        reactive = self.reactive_concentration()
        chaperone = self.raw_repair_activity(REPAIR_CHAPERONE) if config.protein_repair else 0.0
        misfold_fraction = clamp(
            0.010
            + 0.055 * reactive
            + 0.040 * self.current_stress
            + 0.020 * self.aggregate_concentration()
            - 0.010 * chaperone,
            0.004, 0.42,
        )
        for fingerprint, spec, weight in weighted:
            desired = translation_capacity * weight / total_weight
            desired = min(
                desired,
                self.pools[POOL_FUEL] / 0.64,
                self.pools[POOL_MINERAL] / 0.36,
                max(0.0, self.pools[POOL_ATP] - 0.042) / 0.52,
            )
            if desired <= 0.0:
                continue
            self.pools[POOL_FUEL] -= 0.64 * desired
            self.pools[POOL_MINERAL] -= 0.36 * desired
            self.pools[POOL_ATP] -= 0.52 * desired
            misfolded = desired * misfold_fraction
            active = desired - misfolded
            self.proteins[fingerprint] = self.proteins.get(fingerprint, 0.0) + active
            self.damaged_proteins[fingerprint] = self.damaged_proteins.get(fingerprint, 0.0) + misfolded
            self.last_translation += desired / max(dt, 1e-9)
        self._sync_protein_pool()
        self._sync_damage_pool()

    def _replicate_genome(self, world, dt, config):
        """0.2 copying with evolvable proofreading and lesion-dependent errors."""
        self.last_replication_symbols = 0
        if not config.genome_replication or not self.genomes:
            return
        replicase = self.role_activity(ROLE_REPLICASE)
        if config.external_replicase:
            replicase += 0.85
        if replicase <= 1e-6:
            return
        if self.replication_template is None:
            if len(self.genomes) >= 3:
                return
            template_index = int(world.rng.integers(0, len(self.genomes)))
            self.replication_template = self.genomes[template_index].copy()
            self.replication_template_lesion = float(
                self.genome_lesions[template_index]
                if template_index < len(self.genome_lesions) else 0.0
            )
            self.replication_copy = []
            self.replication_fractional = 0.0

        template = self.replication_template
        if template is None:
            return
        proofreading = self.raw_repair_activity(REPAIR_PROOFREADING) if config.proofreading else 0.0
        proof_fraction = proofreading / (0.75 + proofreading)
        quiescence = self.quiescence_level(config)
        sat_nucleotide = self.pools[POOL_NUCLEOTIDE] / (0.055 + self.pools[POOL_NUCLEOTIDE])
        sat_atp = self.pools[POOL_ATP] / (0.10 + self.pools[POOL_ATP])
        speed = 10.0 * replicase * sat_nucleotide * sat_atp
        speed *= (1.0 - 0.34 * proof_fraction) * (1.0 - 0.78 * quiescence)
        self.replication_fractional += speed * dt
        requested = int(self.replication_fractional)
        self.replication_fractional -= requested

        lesion_error = 0.0012 * self.replication_template_lesion
        reactive_error = 0.0010 * self.reactive_concentration()
        raw_error = max(0.0, config.mutation_rate + lesion_error + reactive_error)
        effective_error = raw_error * (1.0 - 0.82 * proof_fraction)
        self.last_effective_error_rate = effective_error
        extra_atp = 0.00075 * proof_fraction

        for _ in range(requested):
            index = len(self.replication_copy)
            if index >= len(template):
                break
            atp_per_symbol = REPLICATION_ATP_PER_SYMBOL + extra_atp
            if (
                self.pools[POOL_NUCLEOTIDE] < MONOMER_MASS
                or self.pools[POOL_ATP] < atp_per_symbol + 0.022
            ):
                break
            symbol = int(template[index])
            if config.mutation and world.rng.random() < effective_error:
                new = int(world.rng.integers(0, g2.ALPHABET_SIZE - 1))
                if new >= symbol:
                    new += 1
                symbol = new % g2.ALPHABET_SIZE
                self.mutation_events['substitution'] += 1
            self.replication_copy.append(symbol)
            self.pools[POOL_NUCLEOTIDE] -= MONOMER_MASS
            self.pools[POOL_ATP] -= atp_per_symbol
            self.cumulative_proofreading_atp += extra_atp
            self.last_replication_symbols += 1

        if len(self.replication_copy) < len(template):
            return

        copied = np.asarray(self.replication_copy, dtype=np.uint8)
        structural_config = LifeHistoryConfig.from_state(config.state_dict())
        structural_config.mutation_rate = 0.0
        budget_symbols = int(self.pools[POOL_NUCLEOTIDE] / MONOMER_MASS)
        copied, delta_symbols, events = g2.mutate_sequence(
            copied, world.rng, structural_config, nucleotide_budget=budget_symbols
        )
        if delta_symbols > 0:
            self.pools[POOL_NUCLEOTIDE] -= delta_symbols * MONOMER_MASS
        elif delta_symbols < 0:
            self.pools[POOL_NUCLEOTIDE] += (-delta_symbols) * MONOMER_MASS
        for name, count in events.items():
            self.mutation_events[name] += int(count)
        self.genomes.append(copied)
        inherited_lesion = self.replication_template_lesion * (0.28 + 0.22 * (1.0 - proof_fraction))
        new_lesion = inherited_lesion + effective_error * len(copied) * 0.06
        self.genome_lesions.append(float(new_lesion))
        self.replication_cycles += 1
        self.replication_template = None
        self.replication_template_lesion = 0.0
        self.replication_copy = []
        self.replication_fractional = 0.0
        self._refresh_gene_cache()
        if self.novel_path_first_age is None and g2.sequence_has_novel_path(copied):
            self.novel_path_first_age = float(self.age)

    def _damage_active_proteins(self, amount):
        amount = min(max(0.0, amount), self.pools[POOL_CATALYST])
        if amount <= 0.0 or not self.proteins:
            return 0.0
        total = max(1e-12, sum(self.proteins.values()))
        moved = 0.0
        for fingerprint in list(self.proteins.keys()):
            share = amount * self.proteins[fingerprint] / total
            share = min(share, self.proteins[fingerprint])
            self.proteins[fingerprint] -= share
            self.damaged_proteins[fingerprint] = self.damaged_proteins.get(fingerprint, 0.0) + share
            moved += share
        self._sync_protein_pool()
        self._sync_damage_pool()
        return moved

    def _aggregate_damaged_proteins(self, amount):
        amount = min(max(0.0, amount), self.pools[POOL_DAMAGED_PROTEIN])
        if amount <= 0.0 or not self.damaged_proteins:
            return 0.0
        total = max(1e-12, sum(self.damaged_proteins.values()))
        moved = 0.0
        for fingerprint in list(self.damaged_proteins.keys()):
            share = amount * self.damaged_proteins[fingerprint] / total
            share = min(share, self.damaged_proteins[fingerprint])
            self.damaged_proteins[fingerprint] -= share
            moved += share
        self.pools[POOL_AGGREGATE] += moved
        self._sync_damage_pool()
        return moved

    def _decay_information_and_proteins(self, world, dt):
        """Convert functional material into repairable damage before loss."""
        if not self.alive:
            return
        volume = max(0.20, (self.radius / BASE_RADIUS) ** 2)
        waste_stress = self.pools[POOL_WASTE] / volume
        reactive = self.reactive_concentration()
        flux = self.last_catalysis + 0.35 * self.last_translation
        config = world.config
        rate_scale = config.damage_rate_scale if config.endogenous_damage else 0.0
        protein_rate = rate_scale * (
            0.00055
            + 0.0020 * waste_stress
            + 0.0038 * reactive
            + 0.0012 * self.current_stress
            + 0.00045 * flux
        )
        damage_amount = self.pools[POOL_CATALYST] * protein_rate * dt
        moved = self._damage_active_proteins(damage_amount)

        aggregation_rate = rate_scale * (
            0.0020
            + 0.0080 * reactive
            + 0.0040 * self.current_stress
            + 0.0030 * self.aggregate_concentration()
        )
        aggregate_amount = self.pools[POOL_DAMAGED_PROTEIN] * aggregation_rate * dt
        aggregated = self._aggregate_damaged_proteins(aggregate_amount)

        # A fraction of newly produced inert waste becomes chemically reactive;
        # no material is created.  External stress changes chemical state.
        convertible = self.pools[POOL_WASTE]
        reactive_amount = min(
            convertible,
            rate_scale * dt * (
                0.015 * self.last_catalysis
                + 0.006 * float(np.sum(np.abs(self.last_generic_flux)))
                + 0.0015 * self.current_stress
            ),
        )
        self.pools[POOL_WASTE] -= reactive_amount
        self.pools[POOL_REACTIVE] += reactive_amount

        # Membrane oxidation changes functional state, not matter.  Strongly
        # oxidised material later hydrolyses into waste.
        local_driver = (
            0.55 * self.damage_trace
            + 0.35 * np.roll(self.damage_trace, 1)
            + 0.35 * np.roll(self.damage_trace, -1)
            + 0.12
        )
        local_driver /= max(1e-9, float(np.mean(local_driver)))
        oxidation_gain = rate_scale * dt * (
            0.00045 + 0.0022 * reactive + 0.0014 * self.current_stress
        ) * local_driver
        self.membrane_oxidation += oxidation_gain
        self.membrane_oxidation = np.clip(self.membrane_oxidation, 0.0, 3.0)

        severe = np.maximum(self.membrane_oxidation - 1.25, 0.0)
        hydrolysis = np.minimum(self.membrane, self.membrane * severe * 0.0014 * dt)
        if float(np.sum(hydrolysis)) > 0.0:
            self.membrane -= hydrolysis
            self.pools[POOL_WASTE] += float(np.sum(hydrolysis))
            self.membrane_oxidation *= np.where(self.membrane > 1e-9, 1.0, 0.0)

        # Lesions are state defects.  Severe lesions sometimes become physical
        # hydrolysis, returning one monomer to waste.
        while len(self.genome_lesions) < len(self.genomes):
            self.genome_lesions.append(0.0)
        lesion_gain = rate_scale * dt * (
            0.0010 + 0.0050 * reactive + 0.0024 * self.current_stress
        )
        for index in range(len(self.genome_lesions)):
            self.genome_lesions[index] += lesion_gain
            if (
                self.genome_lesions[index] > 0.75
                and len(self.genomes[index]) > g2.MIN_GENOME_LENGTH
                and world.rng.random() < dt * 0.00065 * self.genome_lesions[index]
            ):
                sequence = self.genomes[index]
                position = int(world.rng.integers(0, len(sequence)))
                self.genomes[index] = np.concatenate([sequence[:position], sequence[position + 1:]])
                self.pools[POOL_WASTE] += MONOMER_MASS
                self.genome_lesions[index] *= 0.80
                self.genome_damage_events += 1
                self._refresh_gene_cache()

        self.last_damage_generated = float(moved + aggregated + reactive_amount)
        self.cumulative_damage_generated += self.last_damage_generated
        self._sync_damage_pool()

    # ------------------------------------------------------------------
    # Paid repair economy
    # ------------------------------------------------------------------

    def _repair_damage(self, world, dt, config):
        self.last_repair_atp = 0.0
        self.last_repair_flux[:] = 0.0
        cost_scale = max(0.05, config.repair_cost_scale)

        # 1) Antioxidant: reactive matter becomes inert waste.  Reducing fuel is
        # spent and itself becomes waste, preserving matter.
        antioxidant = self.raw_repair_activity(REPAIR_ANTIOXIDANT) if config.protein_repair else 0.0
        desired = dt * 0.030 * antioxidant * (
            self.pools[POOL_REACTIVE] / (0.018 + self.pools[POOL_REACTIVE])
        )
        neutralised = min(
            self.pools[POOL_REACTIVE], desired,
            self.pools[POOL_FUEL] / 0.18,
            max(0.0, self.pools[POOL_ATP] - 0.025) / (0.20 * cost_scale),
        )
        if neutralised > 0.0:
            self.pools[POOL_REACTIVE] -= neutralised
            fuel_spent = 0.18 * neutralised
            self.pools[POOL_FUEL] -= fuel_spent
            self.pools[POOL_WASTE] += neutralised + fuel_spent
            atp = 0.20 * cost_scale * neutralised
            self.pools[POOL_ATP] -= atp
            self.last_repair_atp += atp
            self.last_repair_flux[REPAIR_ANTIOXIDANT] = neutralised / max(dt, 1e-9)
            self.cumulative_oxidant_neutralised += neutralised

        # 2) Chaperone: repair identity-preserving damaged proteins.
        if config.protein_repair:
            chaperone = self.raw_repair_activity(REPAIR_CHAPERONE)
            repair_amount = min(
                self.pools[POOL_DAMAGED_PROTEIN],
                dt * 0.023 * chaperone * (
                    self.pools[POOL_DAMAGED_PROTEIN]
                    / (0.020 + self.pools[POOL_DAMAGED_PROTEIN])
                ),
                max(0.0, self.pools[POOL_ATP] - 0.025) / (0.72 * cost_scale),
            )
            if repair_amount > 0.0 and self.damaged_proteins:
                total = max(1e-12, sum(self.damaged_proteins.values()))
                repaired = 0.0
                for fingerprint in list(self.damaged_proteins.keys()):
                    share = min(
                        self.damaged_proteins[fingerprint],
                        repair_amount * self.damaged_proteins[fingerprint] / total,
                    )
                    self.damaged_proteins[fingerprint] -= share
                    self.proteins[fingerprint] = self.proteins.get(fingerprint, 0.0) + share
                    repaired += share
                atp = 0.72 * cost_scale * repaired
                self.pools[POOL_ATP] -= atp
                self.last_repair_atp += atp
                self.last_repair_flux[REPAIR_CHAPERONE] = repaired / max(dt, 1e-9)

            # 3) Protease/recycling: damaged proteins and aggregates are broken
            # into reusable matter plus inert waste.
            protease = self.raw_repair_activity(REPAIR_PROTEASE)
            substrate = self.pools[POOL_DAMAGED_PROTEIN] + 0.65 * self.pools[POOL_AGGREGATE]
            recycle = min(
                substrate,
                dt * 0.016 * protease * substrate / (0.025 + substrate),
                max(0.0, self.pools[POOL_ATP] - 0.025) / (0.48 * cost_scale),
            )
            recycled = 0.0
            if recycle > 0.0:
                from_damaged = min(self.pools[POOL_DAMAGED_PROTEIN], recycle)
                if from_damaged > 0.0 and self.damaged_proteins:
                    total = max(1e-12, sum(self.damaged_proteins.values()))
                    for fingerprint in list(self.damaged_proteins.keys()):
                        share = min(
                            self.damaged_proteins[fingerprint],
                            from_damaged * self.damaged_proteins[fingerprint] / total,
                        )
                        self.damaged_proteins[fingerprint] -= share
                        recycled += share
                remaining = max(0.0, recycle - from_damaged)
                from_aggregate = min(self.pools[POOL_AGGREGATE], remaining)
                self.pools[POOL_AGGREGATE] -= from_aggregate
                recycled += from_aggregate
                self.pools[POOL_FUEL] += 0.58 * recycled
                self.pools[POOL_MINERAL] += 0.32 * recycled
                self.pools[POOL_WASTE] += 0.10 * recycled
                atp = 0.48 * cost_scale * recycled
                self.pools[POOL_ATP] -= atp
                self.last_repair_atp += atp
                self.last_repair_flux[REPAIR_PROTEASE] = recycled / max(dt, 1e-9)
                self.cumulative_recycled_damage += recycled

        # 4) Genome repair: lesions are repaired with ATP, not a free reset.
        if config.genome_repair and self.genome_lesions:
            activity = self.raw_repair_activity(REPAIR_GENOME)
            total_lesion = float(sum(self.genome_lesions))
            repair = min(
                total_lesion,
                dt * 0.018 * activity * total_lesion / (0.05 + total_lesion),
                max(0.0, self.pools[POOL_ATP] - 0.025) / (0.95 * cost_scale),
            )
            if repair > 0.0:
                for index in range(len(self.genome_lesions)):
                    share = repair * self.genome_lesions[index] / max(total_lesion, 1e-12)
                    self.genome_lesions[index] = max(0.0, self.genome_lesions[index] - share)
                atp = 0.95 * cost_scale * repair
                self.pools[POOL_ATP] -= atp
                self.last_repair_atp += atp
                self.last_repair_flux[REPAIR_GENOME] = repair / max(dt, 1e-9)
                self.cumulative_genome_repairs += repair

        # 5) Membrane repair: reversible chemistry first, then replacement of
        # severely oxidised local material using explicit precursor.
        if config.membrane_repair:
            activity = self.raw_repair_activity(REPAIR_MEMBRANE)
            damage_total = float(np.sum(self.membrane_oxidation * self.membrane))
            chemical = min(
                damage_total,
                dt * 0.012 * activity * damage_total / (0.012 + damage_total),
                max(0.0, self.pools[POOL_ATP] - 0.025) / (0.60 * cost_scale),
            )
            if chemical > 0.0 and damage_total > 0.0:
                weighted = self.membrane_oxidation * self.membrane
                fraction = min(1.0, chemical / max(1e-12, float(np.sum(weighted))))
                self.membrane_oxidation *= (1.0 - fraction)
                atp = 0.60 * cost_scale * chemical
                self.pools[POOL_ATP] -= atp
                self.last_repair_atp += atp
                self.last_repair_flux[REPAIR_MEMBRANE] = chemical / max(dt, 1e-9)

            severe_weight = np.maximum(self.membrane_oxidation - 0.95, 0.0) * self.membrane
            severe_total = float(np.sum(severe_weight))
            replace = min(
                severe_total,
                self.pools[POOL_MEM_PRECURSOR],
                dt * 0.008 * activity,
                max(0.0, self.pools[POOL_ATP] - 0.025) / (0.72 * cost_scale),
            )
            if replace > 0.0 and severe_total > 0.0:
                weights = severe_weight / severe_total
                old = np.minimum(self.membrane, weights * replace)
                old_total = float(np.sum(old))
                self.membrane -= old
                self.pools[POOL_WASTE] += old_total
                new_total = min(old_total, self.pools[POOL_MEM_PRECURSOR])
                self.membrane += weights * new_total
                self.pools[POOL_MEM_PRECURSOR] -= new_total
                self.membrane_oxidation *= np.maximum(0.0, 1.0 - 0.85 * weights)
                atp = 0.72 * cost_scale * new_total
                self.pools[POOL_ATP] -= atp
                self.last_repair_atp += atp

        self.pools[POOL_ATP] = max(0.0, self.pools[POOL_ATP])
        self.cumulative_repair_atp += self.last_repair_atp
        self._sync_protein_pool()
        self._sync_damage_pool()

    def metabolism(self, world, dt, config):
        if not self.alive:
            return
        self._defer_damage_viability = True
        super(DamageProtoCell, self).metabolism(world, dt, config)
        if not self.alive:
            self._defer_damage_viability = False
            return
        self._repair_damage(world, dt, config)
        self._defer_damage_viability = False
        self._update_damage_viability(dt)

        current = self.functional_age()
        alpha = 1.0 - math.exp(-0.15 * dt)
        old_ema = self.functional_age_ema
        self.functional_age_ema += alpha * (current - self.functional_age_ema)
        slope = (self.functional_age_ema - old_ema) / max(dt, 1e-9)
        self.functional_age_slope += alpha * (slope - self.functional_age_slope)
        self._previous_functional_age = current

    def update_viability(self, dt):
        super(DamageProtoCell, self).update_viability(dt)
        if not self._defer_damage_viability:
            self._update_damage_viability(dt)

    def _update_damage_viability(self, dt):
        if not self.alive:
            return
        proteostasis = self.proteostasis_factor()
        genome = self.genome_function_factor()
        reactive = self.reactive_concentration()
        if proteostasis < 0.075 and self.pools[POOL_AGGREGATE] > 0.16:
            self.proteostatic_collapse_timer += dt
        else:
            self.proteostatic_collapse_timer = max(0.0, self.proteostatic_collapse_timer - 0.45 * dt)
        if genome < 0.28 and self.mean_genome_lesion() > 2.0:
            self.genomic_collapse_timer += dt
        else:
            self.genomic_collapse_timer = max(0.0, self.genomic_collapse_timer - 0.35 * dt)
        if reactive > 0.75 and self.closure() < 0.65:
            self.reactive_crisis_timer += dt
        else:
            self.reactive_crisis_timer = max(0.0, self.reactive_crisis_timer - 0.50 * dt)
        if self.proteostatic_collapse_timer > 18.0:
            self.alive = False
            self.death_reason = 'proteostatic closure lost'
        elif self.genomic_collapse_timer > 22.0:
            self.alive = False
            self.death_reason = 'hereditary information damaged'
        elif self.reactive_crisis_timer > 10.0:
            self.alive = False
            self.death_reason = 'reactive boundary collapse'

    def ready_for_division(self):
        if not super(DamageProtoCell, self).ready_for_division():
            return False
        # No age ceiling.  Damage acts through function and an explicit
        # segregation/repair burden, not a clock.
        return bool(
            self.proteostasis_factor() > 0.22
            and self.genome_function_factor() > 0.35
            and self.reactive_concentration() < 0.75
        )

    # ------------------------------------------------------------------
    # Division budget for the larger 0.3 hereditary/repair apparatus
    # ------------------------------------------------------------------

    def ready_for_division(self):
        """Material checkpoint, not a chronological reproduction timer.

        The 0.3 founder carries almost twice the genome and a repair proteome
        absent from 0.2.  Requiring the 0.1 membrane surplus unchanged would
        make the added hereditary machinery a permanent sterility tax.  The
        checkpoint still requires two genomes, a nearly closed boundary,
        explicit membrane surplus, precursor, catalyst, ATP and total matter.
        """
        if (
            not self.alive
            or len(self.genomes) < 2
            or self.replication_template is not None
        ):
            return False
        return bool(
            self.closure() > 0.955
            and self.worst_gap() < 0.26
            and float(np.sum(self.membrane)) > INITIAL_MEMBRANE_MASS * 1.27
            and self.pools[POOL_CATALYST] > 0.46
            and self.pools[POOL_ATP] > 0.018
            and self.pools[POOL_MEM_PRECURSOR] > 0.16
            and self.material_mass() > 5.25
            and self.age > 80.0
        )

    def update_division(self, dt, config):
        """Build a material septum with a paced ATP budget.

        Division still consumes precursor and ATP and can regress when the
        chemistry cannot pay.  The lower per-unit assembly cost represents a
        coarse-grained cytokinetic chemistry rebalanced for the larger 0.3
        hereditary apparatus.  This coefficient is model-wide, not yet an
        evolvable gene.  Assembly is still paid through the ATP ledger.
        """
        if not config.division:
            returned = min(self.septum_mass, 0.006 * dt)
            self.septum_mass -= returned
            self.pools[POOL_MEM_PRECURSOR] += returned
            self.division_progress = clamp(self.septum_mass / 0.12, 0.0, 1.0)
            return
        if self.ready_for_division() or self.division_progress > 0.0:
            reserve = 0.012
            atp_cost = 0.38
            precursor_use = min(
                self.pools[POOL_MEM_PRECURSOR],
                0.015 * dt,
                max(0.0, self.pools[POOL_ATP] - reserve) / atp_cost,
            )
            if precursor_use > 0.0 and self.closure() > 0.90:
                self.pools[POOL_MEM_PRECURSOR] -= precursor_use
                self.pools[POOL_ATP] -= atp_cost * precursor_use
                self.septum_mass += precursor_use
            else:
                returned = min(self.septum_mass, 0.0045 * dt)
                self.septum_mass -= returned
                self.pools[POOL_MEM_PRECURSOR] += returned
            self.division_progress = clamp(self.septum_mass / 0.12, 0.0, 1.0)
        else:
            returned = min(self.septum_mass, 0.0035 * dt)
            self.septum_mass -= returned
            self.pools[POOL_MEM_PRECURSOR] += returned
            self.division_progress = clamp(self.septum_mass / 0.12, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Division and asymmetric damage inheritance
    # ------------------------------------------------------------------

    def _damage_partition_fractions(self, world):
        config = world.config
        if not config.damage_segregation or config.forced_symmetric_damage:
            return (0.5, 0.5), 0.0
        activity = self.raw_repair_activity(REPAIR_SEGREGATION)
        requested = activity / (0.75 + activity)
        burden = self.damage_burden()
        atp_cost = 0.030 * requested * burden * config.repair_cost_scale
        affordable = 1.0
        if atp_cost > 0.0:
            affordable = min(1.0, max(0.0, self.pools[POOL_ATP] - 0.030) / atp_cost)
        effective = clamp(requested * affordable, 0.0, 0.92)
        paid = min(max(0.0, self.pools[POOL_ATP] - 0.030), atp_cost * affordable)
        self.pools[POOL_ATP] -= paid
        world.dissipated_energy += paid
        self.cumulative_segregation_atp += paid
        dirty = 0.5 + 0.44 * effective
        self.last_segregation_strength = effective
        return (dirty, 1.0 - dirty), effective

    @staticmethod
    def _split_dict_mass(source, first_fraction, total_factor):
        first = {}
        second = {}
        for key, value in source.items():
            retained = float(value) * total_factor
            first[int(key)] = retained * first_fraction
            second[int(key)] = retained * (1.0 - first_fraction)
        return first, second

    def split(self, world):
        if not self.can_split() or len(self.genomes) < 2:
            return None
        parent_damage = self.damage_burden()
        parent_lesions = list(self.genome_lesions)
        parent_membrane_oxidation = self.membrane_oxidation.copy()
        parent_damaged = dict(self.damaged_proteins)
        parent_aggregate = float(self.pools[POOL_AGGREGATE])
        parent_reactive = float(self.pools[POOL_REACTIVE])

        damage_fractions, effective = self._damage_partition_fractions(world)
        daughters = g2.GeneticProtoCell.split(self, world)
        if daughters is None:
            return None
        remaining_factor = 0.988

        # Match inherited genome molecules to parental lesion loads by sequence
        # hash, retaining duplicate occurrences as queues.
        lesion_queues = {}
        for genome, lesion in zip(self.genomes, parent_lesions):
            lesion_queues.setdefault(_sequence_hash(genome), []).append(float(lesion))
        for queue in lesion_queues.values():
            queue.sort(reverse=True)

        damaged_pair = self._split_dict_mass(
            parent_damaged, damage_fractions[0], remaining_factor
        )
        aggregate_pair = (
            parent_aggregate * remaining_factor * damage_fractions[0],
            parent_aggregate * remaining_factor * damage_fractions[1],
        )
        reactive_pair = (
            parent_reactive * remaining_factor * damage_fractions[0],
            parent_reactive * remaining_factor * damage_fractions[1],
        )

        for index, daughter in enumerate(daughters):
            daughter.__class__ = DamageProtoCell
            if len(daughter.pools) < POOL_COUNT:
                pools = np.zeros((POOL_COUNT,), dtype=float)
                pools[:len(daughter.pools)] = daughter.pools
                daughter.pools = pools
            daughter.damaged_proteins = damaged_pair[index]
            daughter.pools[POOL_AGGREGATE] = aggregate_pair[index]
            daughter.pools[POOL_REACTIVE] = reactive_pair[index]
            daughter.membrane_oxidation = np.roll(
                parent_membrane_oxidation,
                0 if index == 0 else MEMBRANE_SEGMENTS // 2,
            ).copy()
            # Concentrate oxidation into the dirty daughter, dilute it in the
            # rejuvenated daughter while keeping state burden approximately
            # conserved under the two mass fractions.
            state_scale = 1.0 + (0.72 * effective if index == 0 else -0.62 * effective)
            daughter.membrane_oxidation = np.clip(
                daughter.membrane_oxidation * state_scale, 0.0, 3.0
            )
            daughter.genome_lesions = []
            for genome in daughter.genomes:
                key = _sequence_hash(genome)
                queue = lesion_queues.get(key, [])
                lesion = queue.pop(0) if queue else float(np.mean(parent_lesions) if parent_lesions else 0.0)
                lesion *= (1.0 + 0.52 * effective if index == 0 else 1.0 - 0.46 * effective)
                daughter.genome_lesions.append(max(0.0, lesion))
            daughter.replication_template_lesion = 0.0
            daughter.current_stress = float(self.current_stress)
            daughter._defer_damage_viability = False
            daughter.last_damage_generated = 0.0
            daughter.last_repair_atp = 0.0
            daughter.last_repair_flux = np.zeros(8, dtype=float)
            daughter.cumulative_damage_generated = float(self.cumulative_damage_generated)
            daughter.cumulative_repair_atp = float(self.cumulative_repair_atp)
            daughter.cumulative_recycled_damage = float(self.cumulative_recycled_damage)
            daughter.cumulative_oxidant_neutralised = float(self.cumulative_oxidant_neutralised)
            daughter.cumulative_genome_repairs = float(self.cumulative_genome_repairs)
            daughter.cumulative_proofreading_atp = float(self.cumulative_proofreading_atp)
            daughter.cumulative_segregation_atp = float(self.cumulative_segregation_atp)
            daughter.proteostatic_collapse_timer = 0.0
            daughter.genomic_collapse_timer = 0.0
            daughter.reactive_crisis_timer = 0.0
            daughter.parent_damage_at_birth = parent_damage
            daughter.birth_damage = daughter.damage_burden()
            daughter.pole_age = self.pole_age + 1 if index == 0 else 0
            daughter.last_effective_error_rate = 0.0
            daughter.last_quiescence = 0.0
            daughter.last_segregation_strength = effective
            daughter.last_division_damage_pair = None
            daughter.functional_age_ema = daughter.functional_age()
            daughter.functional_age_slope = 0.0
            daughter._previous_functional_age = daughter.functional_age()
            daughter._sync_damage_pool()

        pair = tuple(float(d.damage_burden()) for d in daughters)
        self.last_division_damage_pair = pair
        world.last_damage_partition = pair
        if effective > 0.05:
            world.damage_segregation_events += 1
            if pair[1] + 0.03 < parent_damage:
                world.rejuvenation_events += 1
        world.damage_partition_log.append({
            'age': float(world.age),
            'parent': float(parent_damage),
            'dirty': float(pair[0]),
            'clean': float(pair[1]),
            'strength': float(effective),
        })
        world.damage_partition_log = world.damage_partition_log[-128:]
        return daughters

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self):
        state = super(DamageProtoCell, self).state_dict()
        state.update({
            'cell_class': 'DamageProtoCell',
            'damaged_proteins': [
                (int(key), float(value)) for key, value in sorted(self.damaged_proteins.items())
            ],
            'membrane_oxidation': self.membrane_oxidation.copy(),
            'genome_lesions': list(float(value) for value in self.genome_lesions),
            'replication_template_lesion': float(self.replication_template_lesion),
            'current_stress': float(self.current_stress),
            'last_damage_generated': float(self.last_damage_generated),
            'last_repair_atp': float(self.last_repair_atp),
            'last_repair_flux': self.last_repair_flux.copy(),
            'cumulative_damage_generated': float(self.cumulative_damage_generated),
            'cumulative_repair_atp': float(self.cumulative_repair_atp),
            'cumulative_recycled_damage': float(self.cumulative_recycled_damage),
            'cumulative_oxidant_neutralised': float(self.cumulative_oxidant_neutralised),
            'cumulative_genome_repairs': float(self.cumulative_genome_repairs),
            'cumulative_proofreading_atp': float(self.cumulative_proofreading_atp),
            'cumulative_segregation_atp': float(self.cumulative_segregation_atp),
            'proteostatic_collapse_timer': float(self.proteostatic_collapse_timer),
            'genomic_collapse_timer': float(self.genomic_collapse_timer),
            'reactive_crisis_timer': float(self.reactive_crisis_timer),
            'birth_damage': float(self.birth_damage),
            'pole_age': int(self.pole_age),
            'parent_damage_at_birth': float(self.parent_damage_at_birth),
            'last_effective_error_rate': float(self.last_effective_error_rate),
            'last_quiescence': float(self.last_quiescence),
            'last_segregation_strength': float(self.last_segregation_strength),
            'last_division_damage_pair': self.last_division_damage_pair,
            'functional_age_ema': float(self.functional_age_ema),
            'functional_age_slope': float(self.functional_age_slope),
            '_previous_functional_age': float(self._previous_functional_age),
        })
        return state

    @classmethod
    def from_state(cls, rng, state):
        parent = g2.GeneticProtoCell.from_state(rng, state)
        parent.__class__ = cls
        cell = parent
        if len(cell.pools) < POOL_COUNT:
            pools = np.zeros((POOL_COUNT,), dtype=float)
            pools[:len(cell.pools)] = cell.pools
            cell.pools = pools
        cell.damaged_proteins = {
            int(key): float(value) for key, value in state.get('damaged_proteins', [])
        }
        cell.membrane_oxidation = np.asarray(
            state.get('membrane_oxidation', np.zeros(MEMBRANE_SEGMENTS)), dtype=float
        ).copy()
        cell.genome_lesions = [
            float(value) for value in state.get('genome_lesions', [0.0] * len(cell.genomes))
        ]
        while len(cell.genome_lesions) < len(cell.genomes):
            cell.genome_lesions.append(0.0)
        cell.replication_template_lesion = float(state.get('replication_template_lesion', 0.0))
        cell.current_stress = float(state.get('current_stress', 0.0))
        cell._defer_damage_viability = False
        cell.last_damage_generated = float(state.get('last_damage_generated', 0.0))
        cell.last_repair_atp = float(state.get('last_repair_atp', 0.0))
        cell.last_repair_flux = np.asarray(
            state.get('last_repair_flux', np.zeros(8)), dtype=float
        ).copy()
        for name in (
            'cumulative_damage_generated', 'cumulative_repair_atp',
            'cumulative_recycled_damage', 'cumulative_oxidant_neutralised',
            'cumulative_genome_repairs', 'cumulative_proofreading_atp',
            'cumulative_segregation_atp', 'proteostatic_collapse_timer',
            'genomic_collapse_timer', 'reactive_crisis_timer', 'birth_damage',
            'parent_damage_at_birth', 'last_effective_error_rate',
            'last_quiescence', 'last_segregation_strength', 'functional_age_ema',
            'functional_age_slope', '_previous_functional_age',
        ):
            setattr(cell, name, float(state.get(name, 0.0)))
        cell.pole_age = int(state.get('pole_age', 0))
        pair = state.get('last_division_damage_pair')
        cell.last_division_damage_pair = None if pair is None else tuple(float(v) for v in pair)
        cell._sync_damage_pool()
        return cell


class DamageWorld(g2.GeneticWorld):
    """Shared physical world with environmental stress and damage histories."""

    def __init__(self, seed=101, initial_cells=1, config=None):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.config = config if config is not None else LifeHistoryConfig()
        self.age = 0.0
        self.field = g2.GeneticParticleField(
            self.rng, initial=True, alt_niche=self.config.alt_niche
        )
        self.cells = []
        self.next_cell_id = 0
        for index in range(int(initial_cells)):
            angle = 2.0 * math.pi * index / max(1, int(initial_cells))
            position = np.array([
                0.5 + 0.12 * math.cos(angle),
                0.5 + 0.12 * math.sin(angle),
            ]) % 1.0
            self.cells.append(DamageProtoCell(
                self.next_cell_id, self.rng, position=position,
                generation=0, lineage=index, bootstrap=True,
            ))
            self.next_cell_id += 1
        self.births = 0
        self.divisions = 0
        self.deaths = 0
        self.last_deaths = []
        self.last_births = []
        self.dissipated_energy = 0.0
        self.division_parent_material = 0.0
        self.division_daughter_material = 0.0
        self.division_shed_material = 0.0
        self.division_residual = 0.0
        self.released_dead_material = 0.0
        self.manual_injections = 0
        self.manual_punctures = 0
        self.damage_events = 0
        self.external_protein_assistance = 0.0
        self.first_novel_path_age = None
        self.novel_path_lineages = set()
        self.current_stress = 0.0
        self.stress_pulses = 0
        self._last_stress_high = False
        self.damage_segregation_events = 0
        self.rejuvenation_events = 0
        self.last_damage_partition = None
        self.damage_partition_log = []
        self.initial_total_material = self.total_material()
        self.last_step_material_residual = 0.0

    def stress_level(self, age=None):
        age = self.age if age is None else float(age)
        mode = self.config.stress_mode
        if mode == 'stable':
            return 0.08
        if mode == 'chronic':
            return 0.62
        if mode == 'alternating':
            return 0.78 if (age % 100.0) >= 58.0 else 0.10
        # Default pulsed: short oxidative episodes with long recovery periods.
        phase = age % 120.0
        if 72.0 <= phase < 90.0:
            x = (phase - 72.0) / 18.0
            return float(0.12 + 0.92 * math.sin(math.pi * x) ** 2)
        return 0.10

    def step(self, dt):
        self.current_stress = self.stress_level(self.age)
        high = self.current_stress > 0.55
        if high and not self._last_stress_high:
            self.stress_pulses += 1
        self._last_stress_high = high
        for cell in self.living_cells():
            # Slight spatial heterogeneity prevents every cell from receiving an
            # exactly identical insult while preserving a shared pulse.
            spatial = 0.88 + 0.12 * math.sin(2.0 * math.pi * cell.pos[0] + 0.7)
            cell.current_stress = self.current_stress * spatial
        super(DamageWorld, self).step(dt)

    def _release_dead_cell(self, cell):
        total = cell.material_mass()
        releases = (
            (PARTICLE_FUEL, cell.pools[POOL_FUEL]),
            (PARTICLE_MINERAL, cell.pools[POOL_MINERAL]),
            (PARTICLE_ALT, cell.pools[POOL_ALT]),
            (
                PARTICLE_WASTE,
                cell.pools[POOL_MEM_PRECURSOR]
                + cell.pools[POOL_CATALYST]
                + cell.pools[POOL_TRANSPORTER_PRECURSOR]
                + cell.pools[POOL_WASTE]
                + cell.pools[POOL_INTERMEDIATE]
                + cell.pools[POOL_NUCLEOTIDE]
                + cell.pools[POOL_DAMAGED_PROTEIN]
                + cell.pools[POOL_AGGREGATE]
                + cell.pools[POOL_REACTIVE]
                + float(np.sum(cell.membrane))
                + float(np.sum(cell.transporters))
                + float(cell.septum_mass)
                + cell.genome_mass(),
            ),
        )
        for kind, amount in releases:
            if amount <= 1e-9:
                continue
            count = max(1, min(16, int(math.ceil(amount / 0.035))))
            positions = (
                cell.pos[None, :]
                + self.rng.normal(0.0, cell.radius * 0.45, (count, 2))
            ) % 1.0
            amounts = np.full((count,), amount / count, dtype=float)
            self.field.add_many(kind, positions, amounts, count_as_injection=False)
        self.dissipated_energy += float(cell.pools[POOL_ATP])
        self.released_dead_material += total
        self.deaths += 1
        self.last_deaths.append((
            cell.cell_id, self.age, cell.death_reason,
            cell.age, cell.damage_burden(), cell.pole_age,
        ))
        self.last_deaths = self.last_deaths[-16:]

    def finite(self):
        if not super(DamageWorld, self).finite():
            return False
        for cell in self.cells:
            if not finite_array(cell.membrane_oxidation):
                return False
            if not finite_array(cell.last_repair_flux):
                return False
            if any(not np.isfinite(value) for value in cell.genome_lesions):
                return False
            if any(not np.isfinite(value) for value in cell.damaged_proteins.values()):
                return False
        return True

    def summary(self):
        summary = super(DamageWorld, self).summary()
        alive = self.living_cells()
        if alive:
            summary.update({
                'mean_damage': float(np.mean([cell.damage_burden() for cell in alive])),
                'max_damage': float(np.max([cell.damage_burden() for cell in alive])),
                'mean_functional_age': float(np.mean([cell.functional_age() for cell in alive])),
                'mean_damage_slope': float(np.mean([cell.functional_age_slope for cell in alive])),
                'mean_proteostasis': float(np.mean([cell.proteostasis_factor() for cell in alive])),
                'mean_genome_lesion': float(np.mean([cell.mean_genome_lesion() for cell in alive])),
                'mean_membrane_oxidation': float(np.mean([
                    np.mean(cell.membrane_oxidation) for cell in alive
                ])),
                'mean_reactive': float(np.mean([cell.reactive_concentration() for cell in alive])),
                'mean_aggregate': float(np.mean([cell.aggregate_concentration() for cell in alive])),
                'mean_repair_atp': float(np.mean([cell.last_repair_atp for cell in alive])),
                'repair_atp_total': float(sum(cell.cumulative_repair_atp for cell in alive)),
                'damage_generated_total': float(sum(
                    cell.cumulative_damage_generated for cell in alive
                )),
                'mean_quiescence': float(np.mean([cell.last_quiescence for cell in alive])),
                'mean_segregation': float(np.mean([
                    cell.last_segregation_strength for cell in alive
                ])),
                'max_pole_age': int(max(cell.pole_age for cell in alive)),
                'mean_birth_damage': float(np.mean([cell.birth_damage for cell in alive])),
                'oldest_cell_age': float(max(cell.age for cell in alive)),
            })
        else:
            summary.update({
                'mean_damage': 0.0, 'max_damage': 0.0,
                'mean_functional_age': 0.0, 'mean_damage_slope': 0.0,
                'mean_proteostasis': 0.0, 'mean_genome_lesion': 0.0,
                'mean_membrane_oxidation': 0.0, 'mean_reactive': 0.0,
                'mean_aggregate': 0.0, 'mean_repair_atp': 0.0,
                'repair_atp_total': 0.0, 'damage_generated_total': 0.0,
                'mean_quiescence': 0.0, 'mean_segregation': 0.0,
                'max_pole_age': 0, 'mean_birth_damage': 0.0,
                'oldest_cell_age': 0.0,
            })
        summary.update({
            'build': BUILD,
            'stress': float(self.current_stress),
            'stress_mode': self.config.stress_mode,
            'stress_pulses': int(self.stress_pulses),
            'damage_segregation_events': int(self.damage_segregation_events),
            'rejuvenation_events': int(self.rejuvenation_events),
            'last_damage_partition': self.last_damage_partition,
        })
        return summary

    def state_dict(self):
        state = super(DamageWorld, self).state_dict()
        state.update({
            'save_version': SAVE_VERSION,
            'build': BUILD,
            'config': self.config.state_dict(),
            'cells': [cell.state_dict() for cell in self.cells],
            'current_stress': float(self.current_stress),
            'stress_pulses': int(self.stress_pulses),
            '_last_stress_high': bool(self._last_stress_high),
            'damage_segregation_events': int(self.damage_segregation_events),
            'rejuvenation_events': int(self.rejuvenation_events),
            'last_damage_partition': self.last_damage_partition,
            'damage_partition_log': list(self.damage_partition_log),
        })
        return state

    @classmethod
    def from_state(cls, state):
        world = cls(
            seed=int(state['seed']), initial_cells=0,
            config=LifeHistoryConfig.from_state(state.get('config', {})),
        )
        world.rng.bit_generator.state = state['rng_state']
        world.age = float(state['age'])
        world.field = g2.GeneticParticleField.from_state(world.rng, state['field'])
        world.cells = [DamageProtoCell.from_state(world.rng, item) for item in state['cells']]
        world.rng.bit_generator.state = state['rng_state']
        for name in (
            'next_cell_id', 'births', 'divisions', 'deaths', 'manual_injections',
            'manual_punctures', 'damage_events', 'stress_pulses',
            'damage_segregation_events', 'rejuvenation_events',
        ):
            setattr(world, name, int(state.get(name, getattr(world, name))))
        for name in (
            'dissipated_energy', 'division_parent_material', 'division_daughter_material',
            'division_shed_material', 'division_residual', 'released_dead_material',
            'external_protein_assistance', 'initial_total_material',
            'last_step_material_residual', 'current_stress',
        ):
            setattr(world, name, float(state.get(name, getattr(world, name))))
        world.first_novel_path_age = state.get('first_novel_path_age')
        world.novel_path_lineages = set(state.get('novel_path_lineages', []))
        world.last_deaths = list(state.get('last_deaths', []))
        world.last_births = list(state.get('last_births', []))
        world._last_stress_high = bool(state.get('_last_stress_high', False))
        pair = state.get('last_damage_partition')
        world.last_damage_partition = None if pair is None else tuple(float(v) for v in pair)
        world.damage_partition_log = list(state.get('damage_partition_log', []))
        return world

    def save(self, path=SAVE_FILE):
        _atomic_pickle(path, self.state_dict())

    @classmethod
    def load(cls, path=SAVE_FILE):
        with open(path, 'rb') as handle:
            return cls.from_state(pickle.load(handle))

    def clone(self):
        return DamageWorld.from_state(self.state_dict())


# ---------------------------------------------------------------------------
# Accelerated life-history assay using the same genome grammar and mutator
# ---------------------------------------------------------------------------


def simulate_life_history(sequence, environment='stable', rng=None, steps=240):
    """Evaluate one genotype in an age-structured coarse assay.

    This assay deliberately omits membrane particles and diffusion.  It uses
    the exact 0.2/0.3 genome decoder and repair genes, then asks how repair,
    proofreading, quiescence and damage segregation trade reproduction against
    persistence under stable or pulsed damage.  It is a control assay, not a
    claim that the spatial protocell evolved for hundreds of generations.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    traits = regulator_trait_vector(sequence)
    essential = essential_gene_score(sequence)
    if essential < 0.08:
        return {
            'fitness': 0.0, 'offspring': 0.0, 'survived': 0.0,
            'final_damage': 2.0, 'mean_damage': 2.0,
            'repair_investment': 0.0, 'segregation': traits[REPAIR_SEGREGATION],
            'proofreading': traits[REPAIR_PROOFREADING],
        }

    biomass = 1.0
    damage = 0.055
    aggregate = 0.012
    genome_damage = 0.020
    energy = 0.45
    offspring = 0.0
    mean_damage = 0.0
    repair_spent = 0.0
    alive = True
    pole_age = 0

    for tick in range(int(steps)):
        if environment == 'pulsed':
            phase = tick % 64
            stress = 1.0 if 38 <= phase < 48 else 0.08
            resource = 0.86 + 0.10 * math.sin(2.0 * math.pi * tick / 77.0)
        elif environment == 'chronic':
            stress = 0.55
            resource = 0.82
        else:
            stress = 0.07
            resource = 1.0

        anti, chap, prot, grepair, segregation, proof, quiescence_gene, _ = traits
        sensed = clamp(damage + 0.7 * aggregate + 0.5 * genome_damage + 0.35 * stress, 0.0, 2.0)
        quiescence = quiescence_gene * clamp((sensed - 0.16) * 1.3, 0.0, 0.78)
        repair_request = clamp(
            0.22 * anti + 0.25 * chap + 0.22 * prot + 0.18 * grepair,
            0.0, 0.82,
        ) * clamp(0.35 + 1.4 * sensed, 0.0, 1.4)
        repair_cost = 0.0018 + 0.0135 * repair_request + 0.0032 * proof
        repair_spent += repair_cost

        growth_flux = (
            0.030 * essential * resource
            * (1.0 - 0.78 * quiescence)
            * math.exp(-1.35 * damage - 1.8 * aggregate)
        )
        energy += 0.038 * essential * resource - repair_cost - 0.010 * growth_flux
        energy = clamp(energy, -0.25, 1.4)

        damage_gain = (
            0.0030 + 0.100 * growth_flux + 0.028 * stress
            + 0.010 * max(0.0, -energy)
        )
        damage_repair = 0.030 * (0.52 * anti + 0.48 * chap) * repair_request
        damage = max(0.0, damage + damage_gain - damage_repair)
        aggregate += 0.030 * damage * (1.0 - 0.62 * chap) - 0.020 * prot * repair_request
        aggregate = max(0.0, aggregate)
        genome_damage += 0.006 * stress + 0.004 * growth_flux - 0.020 * grepair * repair_request
        genome_damage = max(0.0, genome_damage)

        error_load = (0.006 + 0.035 * genome_damage) * (1.0 - 0.78 * proof)
        growth_flux *= math.exp(-2.4 * error_load)
        biomass += growth_flux

        if damage + 1.4 * aggregate + 0.7 * genome_damage > 1.55 or energy < -0.18:
            alive = False
            break

        if biomass >= 2.0:
            segregation_effect = 0.44 * segregation
            dirty = clamp(damage * (1.0 + segregation_effect), 0.0, 2.0)
            clean = clamp(damage * (1.0 - 0.86 * segregation_effect), 0.0, 2.0)
            aggregate_dirty = aggregate * (1.0 + 1.15 * segregation_effect)
            aggregate_clean = aggregate * (1.0 - 0.90 * segregation_effect)
            child_quality = math.exp(-2.3 * clean - 2.7 * aggregate_clean - 1.2 * genome_damage)
            offspring += child_quality
            biomass = 1.0
            damage = dirty
            aggregate = aggregate_dirty
            pole_age += 1
            energy *= 0.54

        mean_damage += damage + aggregate

    longevity_bonus = 0.20 * (tick + 1) / float(steps)
    fitness = offspring + longevity_bonus + (0.12 if alive else 0.0)
    # Genome bulk and excessive repair expression carry a small synthesis cost.
    fitness *= math.exp(-0.0009 * max(0, len(sequence) - 128))
    return {
        'fitness': float(max(0.0, fitness)),
        'offspring': float(offspring),
        'survived': float(alive),
        'final_damage': float(damage + aggregate),
        'mean_damage': float(mean_damage / max(1, tick + 1)),
        'repair_investment': float(repair_spent / max(1, tick + 1)),
        'segregation': float(traits[REPAIR_SEGREGATION]),
        'proofreading': float(traits[REPAIR_PROOFREADING]),
        'repair_trait': float(np.mean(traits[:4])),
        'quiescence_trait': float(traits[REPAIR_QUIESCENCE]),
        'pole_age': int(pole_age),
    }


def run_life_history_assay(seed=101, generations=60, population=72,
                           environment='stable', mutation=True,
                           variable_length=True, damage_segregation=True):
    """Evolve 0.3 genomes in a controlled life-history assay."""
    rng = np.random.default_rng(int(seed))
    config = LifeHistoryConfig(
        mutation=mutation,
        variable_length=variable_length,
        gene_duplication=variable_length,
        mutation_rate=0.0040,
        structural_rate=0.10,
    )
    founder = founding_genome_03()
    genomes = [founder.copy() for _ in range(int(population))]
    history = []
    first_adaptive_generation = None

    for generation in range(int(generations)):
        results = []
        for index, genome in enumerate(genomes):
            local_rng = np.random.default_rng(
                (int(seed) * 1000003 + generation * 1009 + index * 17) & 0xFFFFFFFF
            )
            result = simulate_life_history(genome, environment=environment, rng=local_rng)
            if not damage_segregation:
                # Counterfactual assay: remove the offspring-quality advantage
                # of segregation while retaining expression cost.
                result['fitness'] *= math.exp(-0.22 * result['segregation'])
            results.append(result)
        fitness = np.asarray([item['fitness'] for item in results], dtype=float)
        traits = np.asarray([regulator_trait_vector(genome) for genome in genomes])
        record = {
            'generation': generation,
            'mean_fitness': float(np.mean(fitness)),
            'best_fitness': float(np.max(fitness)),
            'mean_repair_trait': float(np.mean(traits[:, :4])),
            'mean_segregation_trait': float(np.mean(traits[:, REPAIR_SEGREGATION])),
            'mean_proofreading_trait': float(np.mean(traits[:, REPAIR_PROOFREADING])),
            'mean_quiescence_trait': float(np.mean(traits[:, REPAIR_QUIESCENCE])),
            'mean_genome_length': float(np.mean([len(genome) for genome in genomes])),
        }
        history.append(record)
        if first_adaptive_generation is None and record['best_fitness'] > 1.35:
            first_adaptive_generation = generation

        shifted = fitness - float(np.max(fitness))
        weights = np.exp(shifted / 0.16)
        weights += 1e-12
        weights /= float(np.sum(weights))
        elite_indices = np.argsort(fitness)[-4:]
        next_genomes = [genomes[int(index)].copy() for index in elite_indices]
        while len(next_genomes) < int(population):
            parent = genomes[int(rng.choice(len(genomes), p=weights))]
            budget = 64
            child, _, _ = g2.mutate_sequence(
                parent, rng, config, nucleotide_budget=budget
            )
            next_genomes.append(child)
        genomes = next_genomes[:int(population)]

    final_results = [
        simulate_life_history(
            genome, environment=environment,
            rng=np.random.default_rng(int(seed) + 500000 + index),
        )
        for index, genome in enumerate(genomes)
    ]
    final_fitness = np.asarray([item['fitness'] for item in final_results])
    best_index = int(np.argmax(final_fitness))
    best = genomes[best_index]
    best_traits = regulator_trait_vector(best)
    return {
        'seed': int(seed),
        'environment': str(environment),
        'generations': int(generations),
        'population': int(population),
        'mutation': bool(mutation),
        'variable_length': bool(variable_length),
        'damage_segregation': bool(damage_segregation),
        'first_adaptive_generation': first_adaptive_generation,
        'final_mean_fitness': float(np.mean(final_fitness)),
        'final_best_fitness': float(np.max(final_fitness)),
        'final_mean_repair_trait': float(np.mean([
            np.mean(regulator_trait_vector(genome)[:4]) for genome in genomes
        ])),
        'final_mean_segregation_trait': float(np.mean([
            regulator_trait_vector(genome)[REPAIR_SEGREGATION] for genome in genomes
        ])),
        'final_mean_proofreading_trait': float(np.mean([
            regulator_trait_vector(genome)[REPAIR_PROOFREADING] for genome in genomes
        ])),
        'best_repair_trait': float(np.mean(best_traits[:4])),
        'best_segregation_trait': float(best_traits[REPAIR_SEGREGATION]),
        'best_proofreading_trait': float(best_traits[REPAIR_PROOFREADING]),
        'best_quiescence_trait': float(best_traits[REPAIR_QUIESCENCE]),
        'best_genome_length': int(len(best)),
        'best_hash': _sequence_hash(best),
        'history': history,
        'best_sequence': best.copy(),
    }


# ---------------------------------------------------------------------------
# Logging and reports
# ---------------------------------------------------------------------------


LOG_FIELDS = (
    'wall_time', 'session_id', 'reason', 'age', 'cells', 'divisions', 'deaths',
    'max_generation', 'stress', 'stress_pulses', 'mean_damage', 'max_damage',
    'mean_functional_age', 'mean_damage_slope', 'mean_proteostasis',
    'mean_genome_lesion', 'mean_membrane_oxidation', 'mean_reactive',
    'mean_aggregate', 'mean_repair_atp', 'repair_atp_total',
    'damage_generated_total', 'mean_quiescence', 'mean_segregation',
    'damage_segregation_events', 'rejuvenation_events', 'max_pole_age',
    'oldest_cell_age', 'mean_atp', 'mean_closure', 'mean_loop',
    'mean_genome_length', 'genome_types', 'mutation_total', 'total_material',
    'matter_residual', 'division_residual', 'fps', 'sim_rate', 'peak_mb',
)


class LongRunLogger(object):
    def __init__(self, world, path=LOG_FILE):
        self.path = path
        self.session_id = '{}-{}'.format(int(time.time()), int(world.seed))
        self.last_age = -1e9
        self.status = 'READY'

    def log(self, world, fps=0.0, sim_rate=0.0, reason='interval', force=False):
        if not force and world.age - self.last_age < LOG_INTERVAL:
            return
        row = world.summary()
        row.update({
            'wall_time': time.time(),
            'session_id': self.session_id,
            'reason': reason,
            'fps': float(fps),
            'sim_rate': float(sim_rate),
            'peak_mb': float(_memory_peak_mb_estimate()),
        })
        exists = os.path.exists(self.path) and os.path.getsize(self.path) > 0
        try:
            with open(self.path, 'a', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS, extrasaction='ignore')
                if not exists:
                    writer.writeheader()
                writer.writerow({name: row.get(name, '') for name in LOG_FIELDS})
            self.last_age = float(world.age)
            self.status = 'OK'
        except Exception:
            self.status = 'ERR'


def generate_report(log_path=LOG_FILE, report_path=REPORT_FILE, session_path=SESSION_FILE):
    if not os.path.exists(log_path):
        return 'NO LOG'
    try:
        with open(log_path, 'r', newline='') as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return 'EMPTY'
        grouped = {}
        for row in rows:
            grouped.setdefault(row.get('session_id', 'unknown'), []).append(row)
        session_rows = []
        for session_id, items in grouped.items():
            first, last = items[0], items[-1]
            session_rows.append({
                'session_id': session_id,
                'start_age': first.get('age', ''),
                'end_age': last.get('age', ''),
                'final_cells': last.get('cells', ''),
                'final_generation': last.get('max_generation', ''),
                'final_mean_damage': last.get('mean_damage', ''),
                'max_damage': max(float(item.get('max_damage') or 0.0) for item in items),
                'repair_atp_total': last.get('repair_atp_total', ''),
                'segregation_events': last.get('damage_segregation_events', ''),
                'rejuvenation_events': last.get('rejuvenation_events', ''),
                'matter_residual': last.get('matter_residual', ''),
            })
        with open(session_path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(session_rows[0].keys()))
            writer.writeheader()
            writer.writerows(session_rows)
        last = rows[-1]
        text = [
            BUILD + ' long-run report',
            'sessions: {}'.format(len(grouped)),
            'last age: {}'.format(last.get('age')),
            'last cells: {}'.format(last.get('cells')),
            'last generation: {}'.format(last.get('max_generation')),
            'mean / max damage: {} / {}'.format(
                last.get('mean_damage'), last.get('max_damage')
            ),
            'repair ATP total: {}'.format(last.get('repair_atp_total')),
            'damage segregation / rejuvenation: {} / {}'.format(
                last.get('damage_segregation_events'), last.get('rejuvenation_events')
            ),
            'matter residual: {}'.format(last.get('matter_residual')),
            'No fixed lifespan is used; chronological age alone cannot kill a cell.',
        ]
        with open(report_path, 'w') as handle:
            handle.write('\n'.join(text) + '\n')
        return 'OK'
    except Exception:
        return 'WRITE ERR'


def run_headless_trial(seed=101, seconds=240.0, initial_cells=1, config=None):
    world = DamageWorld(seed=seed, initial_cells=initial_cells, config=config)
    dt = 1.0 / SIM_HZ
    for _ in range(int(round(float(seconds) * SIM_HZ))):
        world.step(dt)
        if not world.living_cells():
            break
    return world.summary()


# ---------------------------------------------------------------------------
# Pythonista Scene
# ---------------------------------------------------------------------------


try:
    from scene import Scene, run, LANDSCAPE
    from scene import background, fill, stroke, stroke_weight, ellipse, line, rect, text

    class SomaCellDamageScene(Scene):
        def setup(self):
            self.paused = False
            self.accumulator = 0.0
            self.last_wall = time.time()
            self.last_save_age = 0.0
            self.last_touch_wall = -10.0
            self.fps = 0.0
            self.sim_rate = 0.0
            self.telemetry_wall = time.time()
            self.telemetry_age = 0.0
            self.telemetry_frames = 0
            self.save_status = 'NEW'
            self.report_status = 'WAIT'
            try:
                if os.path.exists(SAVE_FILE):
                    self.world = DamageWorld.load(SAVE_FILE)
                    self.save_status = 'LOAD'
                else:
                    self.world = DamageWorld(seed=101, initial_cells=1)
            except Exception:
                self.world = DamageWorld(seed=101, initial_cells=1)
                self.save_status = 'RECOVER'
            self.logger = LongRunLogger(self.world)
            self.logger.log(self.world, reason='start', force=True)

        def _world_rect(self):
            width, height = float(self.size.w), float(self.size.h)
            return 34.0, 76.0, width - 68.0, height - 160.0

        def _screen(self, position):
            left, bottom, width, height = self._world_rect()
            position = np.asarray(position, dtype=float) % 1.0
            return left + position[0] * width, bottom + position[1] * height

        def _unit_position(self, point):
            left, bottom, width, height = self._world_rect()
            return np.array([
                clamp((point.x - left) / max(width, 1.0), 0.0, 1.0),
                clamp((point.y - bottom) / max(height, 1.0), 0.0, 1.0),
            ])

        def update(self):
            now = time.time()
            wall_dt = clamp(now - self.last_wall, 0.0, 0.20)
            self.last_wall = now
            if not self.paused:
                self.accumulator += wall_dt
                fixed = 1.0 / SIM_HZ
                steps = 0
                while self.accumulator >= fixed and steps < 7:
                    self.world.step(fixed)
                    self.accumulator -= fixed
                    steps += 1
                if steps >= 7:
                    self.accumulator = min(self.accumulator, fixed)
            self.telemetry_frames += 1
            elapsed = now - self.telemetry_wall
            if elapsed >= 1.0:
                self.fps = self.telemetry_frames / elapsed
                self.sim_rate = (self.world.age - self.telemetry_age) / elapsed
                self.telemetry_wall = now
                self.telemetry_age = self.world.age
                self.telemetry_frames = 0
            self.logger.log(self.world, self.fps, self.sim_rate)
            if self.world.age - self.last_save_age >= AUTO_SAVE_INTERVAL:
                try:
                    self.world.save(SAVE_FILE)
                    self.save_status = 'OK'
                    self.last_save_age = self.world.age
                    gc.collect()
                except Exception:
                    self.save_status = 'ERR'
            if not self.world.living_cells() and self.report_status == 'WAIT':
                self.logger.log(self.world, self.fps, self.sim_rate, reason='extinct', force=True)
                self.report_status = generate_report()

        def _draw_particles(self):
            colours = (
                (0.28, 0.90, 0.46),
                (0.96, 0.69, 0.20),
                (0.92, 0.25, 0.29),
                (0.20, 0.72, 1.00),
            )
            for index in range(len(self.world.field.amount)):
                x, y = self._screen(self.world.field.pos[index])
                amount = float(self.world.field.amount[index])
                radius = 1.5 + 8.0 * math.sqrt(clamp(amount / 0.045, 0.0, 1.7))
                colour = colours[int(self.world.field.kind[index])]
                fill(colour[0], colour[1], colour[2], 0.78)
                ellipse(x - radius, y - radius, radius * 2.0, radius * 2.0)

        def _draw_cell(self, cell):
            left, bottom, width, height = self._world_rect()
            centre_x, centre_y = self._screen(cell.pos)
            scale = min(width, height)
            screen_radius = cell.radius * scale
            closure = cell.closure_array()
            damage = clamp(cell.damage_burden() / 1.0, 0.0, 1.0)
            atp = clamp(cell.pools[POOL_ATP] / 0.65, 0.0, 1.0)
            fill(0.08 + 0.34 * damage, 0.22 + 0.32 * atp, 0.38 - 0.20 * damage, 0.32)
            ellipse(centre_x - screen_radius, centre_y - screen_radius,
                    screen_radius * 2.0, screen_radius * 2.0)

            # Aggregates are explicit internal matter.
            aggregate_count = int(clamp(cell.pools[POOL_AGGREGATE] / 0.008, 0.0, 12.0))
            for index in range(aggregate_count):
                angle = 2.399963 * index + 0.3 * cell.cell_id
                radial = screen_radius * (0.12 + 0.045 * (index % 5))
                x = centre_x + math.cos(angle) * radial
                y = centre_y + math.sin(angle) * radial
                r = 1.2 + 2.2 * damage
                fill(1.0, 0.28, 0.20, 0.88)
                ellipse(x - r, y - r, 2 * r, 2 * r)

            # Material genome rings, lesion-dependent colour.
            lesion = clamp(cell.mean_genome_lesion() / 1.5, 0.0, 1.0)
            for genome_index, genome in enumerate(cell.genomes[:4]):
                ring_radius = screen_radius * (0.18 + 0.07 * genome_index)
                stroke(0.42 + 0.50 * lesion, 1.0 - 0.55 * lesion, 0.92 - 0.55 * lesion, 0.82)
                stroke_weight(1.2)
                segments = max(8, min(28, len(genome) // 7))
                previous = None
                for index in range(segments + 1):
                    angle = 2.0 * math.pi * index / segments + 0.3 * genome_index
                    point = (
                        centre_x + math.cos(angle) * ring_radius,
                        centre_y + math.sin(angle) * ring_radius,
                    )
                    if previous is not None:
                        line(previous[0], previous[1], point[0], point[1])
                    previous = point

            points = cell.boundary_points()
            for index in range(MEMBRANE_SEGMENTS):
                nxt = (index + 1) % MEMBRANE_SEGMENTS
                if closure[index] < 0.055 and closure[nxt] < 0.055:
                    continue
                p0 = points[index]
                p1 = p0 + wrapped_delta(p0, points[nxt])
                x0, y0 = self._screen(p0)
                x1 = left + p1[0] * width
                y1 = bottom + p1[1] * height
                local = 0.5 * (closure[index] + closure[nxt])
                oxidation = clamp(0.5 * (
                    cell.membrane_oxidation[index] + cell.membrane_oxidation[nxt]
                ), 0.0, 1.0)
                stroke(
                    0.34 + 0.62 * oxidation,
                    0.78 - 0.52 * oxidation,
                    0.95 - 0.58 * oxidation,
                    0.30 + 0.70 * local,
                )
                stroke_weight(0.6 + 3.2 * local)
                line(x0, y0, x1, y1)

            if cell.division_progress > 0.02:
                normal = unit_vector(cell.division_axis + math.pi * 0.5)
                length = screen_radius * 0.65 * cell.division_progress
                stroke(0.98, 0.92, 0.55, 0.72)
                stroke_weight(1.0 + 2.0 * cell.division_progress)
                line(centre_x - normal[0] * length, centre_y - normal[1] * length,
                     centre_x + normal[0] * length, centre_y + normal[1] * length)

            fill(0.91, 0.97, 1.0)
            text('#{} G{} P{} D{:.2f} F{:.0f}'.format(
                cell.cell_id, cell.generation, cell.pole_age,
                cell.damage_burden(), cell.functional_age()
            ), x=centre_x, y=centre_y - screen_radius - 10,
                font_size=8, alignment=5)

        def draw(self):
            background(0.012, 0.022, 0.035)
            left, bottom, width, height = self._world_rect()
            fill(0.022, 0.045, 0.060)
            rect(left, bottom, width, height)
            self._draw_particles()
            for cell in self.world.living_cells():
                self._draw_cell(cell)
            summary = self.world.summary()
            fill(0.92, 0.98, 1.0)
            text(BUILD, x=24, y=self.size.h - 25, font_size=18, alignment=4)
            fill(0.64, 0.78, 0.86)
            text('SCENE ACTIVE | SAVE {} | LOG {} | REPORT {} | {:.1f} fps | x{:.2f}'.format(
                self.save_status, self.logger.status, self.report_status,
                self.fps, self.sim_rate
            ), x=self.size.w - 72, y=self.size.h - 25, font_size=9, alignment=6)
            fill(0.84, 0.92, 0.97)
            text('age {:.1f}s cells {} div {} deaths {} G{} stress {:.2f} pulses {}'.format(
                summary['age'], summary['cells'], summary['divisions'], summary['deaths'],
                summary['max_generation'], summary['stress'], summary['stress_pulses']
            ), x=24, y=54, font_size=10, alignment=4)
            text('damage mean/max {:.3f}/{:.3f} proteostasis {:.3f} lesions {:.3f} reactive {:.3f}'.format(
                summary['mean_damage'], summary['max_damage'],
                summary['mean_proteostasis'], summary['mean_genome_lesion'],
                summary['mean_reactive']
            ), x=24, y=38, font_size=9, alignment=4)
            text('repair ATP {:.4f} segregation {} rejuvenation {} pole {} oldest {:.1f}s'.format(
                summary['mean_repair_atp'], summary['damage_segregation_events'],
                summary['rejuvenation_events'], summary['max_pole_age'],
                summary['oldest_cell_age']
            ), x=24, y=23, font_size=9, alignment=4)
            text('seal {:.3f} ATP {:.3f} loop {:.3f} ledger {:+.2e} | tap puncture/inject | top pause | double reset'.format(
                summary['mean_closure'], summary['mean_atp'], summary['mean_loop'],
                summary['matter_residual']
            ), x=24, y=9, font_size=8, alignment=4)
            if summary['cells'] == 0:
                fill(1.0, 0.38, 0.32)
                text('DAMAGE–REPAIR LOOP EXTINCT — double tap to reseed',
                     x=self.size.w * 0.5, y=self.size.h * 0.52,
                     font_size=17, alignment=5)
            if self.paused:
                fill(1.0, 0.92, 0.45)
                text('PAUSED', x=self.size.w * 0.5,
                     y=self.size.h - 26, font_size=13, alignment=5)

        def touch_began(self, touch):
            now = time.time()
            if now - self.last_touch_wall < 0.42:
                try:
                    if os.path.exists(SAVE_FILE):
                        os.remove(SAVE_FILE)
                except Exception:
                    pass
                self.world = DamageWorld(seed=101, initial_cells=1)
                self.logger = LongRunLogger(self.world)
                self.logger.log(self.world, reason='reset', force=True)
                self.report_status = 'WAIT'
                self.save_status = 'NEW'
                self.accumulator = 0.0
                self.paused = False
                self.last_touch_wall = -10.0
                gc.collect()
                return
            self.last_touch_wall = now
            if touch.location.y > self.size.h - 60:
                self.paused = not self.paused
                self.logger.log(
                    self.world, self.fps, self.sim_rate,
                    reason='pause' if self.paused else 'resume', force=True
                )
                return
            position = self._unit_position(touch.location)
            if not self.world.puncture_nearest(position):
                self.world.inject_cloud(position)
                self.world.field.add_many(
                    PARTICLE_ALT,
                    (position[None, :] + self.world.rng.normal(0.0, 0.024, (6, 2))) % 1.0,
                    np.full(6, 0.020),
                    count_as_injection=True,
                )

        def stop(self):
            try:
                self.world.save(SAVE_FILE)
                self.save_status = 'OK'
            except Exception:
                self.save_status = 'ERR'
            self.logger.log(self.world, self.fps, self.sim_rate, reason='stop', force=True)
            self.report_status = generate_report()

except ImportError:
    Scene = None


if __name__ == '__main__':
    if Scene is None:
        print(run_headless_trial(seed=101, seconds=240.0, initial_cells=1))
    else:
        run(SomaCellDamageScene(), LANDSCAPE, show_fps=False)
