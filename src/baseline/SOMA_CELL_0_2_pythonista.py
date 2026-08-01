# coding: utf-8
"""
SOMA-CELL 0.2 — Material Heredity / 物質遺伝子と進化可能な反応化学

Pythonista 3 / CPython + NumPy research prototype.

This module extends SOMA-CELL 0.1 rather than replacing its membrane physics.
The new layer adds a material, variable-length genome whose copied monomers,
translated catalysts, and replication apparatus are all paid for from the
same intracellular matter and ATP ledger as the membrane.

What is genuinely new in 0.2
----------------------------
* A cell contains one or more circular symbol polymers.  Polymer mass is part
  of the matter ledger, can leak/degrade, and must be copied before division.
* Genes are delimited sequences.  Their products determine energy extraction,
  membrane-precursor synthesis, transporter synthesis, nucleotide synthesis,
  genome replication, translation, regulation, and generic reaction edges.
* Replication requires an internally maintained replicase protein, free genome
  monomers, and ATP.  A Python-level ``child.genome = parent.genome.copy()`` is
  not used to create an inheritable copy.
* Copying can introduce substitutions, insertions, deletions, duplications,
  inversions, and transpositions.  Insertions/duplications consume material;
  deletions return polymer material to the monomer pool.
* Daughters must each physically receive a complete genome molecule.
* Generic reaction genes can create reaction edges absent from the founding
  population.  A controlled accelerated assay tests whether such an edge can
  arise, spread under selection, and lose its benefit after knockout.

This remains a coarse-grained artificial chemistry.  The alphabet, decoding
physics, reaction grammar, and environmental laws are human-designed.  It is
not claimed to be open-ended evolution or biological life.
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


BUILD = 'SOMA-CELL 0.2.0'
SAVE_VERSION = 2
BASE_DIR = os.path.dirname(__file__)
SAVE_FILE = os.path.join(BASE_DIR, 'soma_cell_0_2.pkl')
LOG_FILE = os.path.join(BASE_DIR, 'soma_cell_0_2_longrun.csv')
REPORT_FILE = os.path.join(BASE_DIR, 'soma_cell_0_2_report.txt')
SESSION_FILE = os.path.join(BASE_DIR, 'soma_cell_0_2_sessions.csv')

SIM_HZ = 20.0
AUTO_SAVE_INTERVAL = 30.0
LOG_INTERVAL = 10.0
MAX_CELLS = 14

# Re-export the 0.1 physical constants used by the subclass.
PARTICLE_FUEL = base.PARTICLE_FUEL
PARTICLE_MINERAL = base.PARTICLE_MINERAL
PARTICLE_WASTE = base.PARTICLE_WASTE
PARTICLE_ALT = 3
PARTICLE_NAMES = ('fuel', 'mineral', 'waste', 'alt-substrate')

POOL_FUEL = base.POOL_FUEL
POOL_MINERAL = base.POOL_MINERAL
POOL_ATP = base.POOL_ATP
POOL_MEM_PRECURSOR = base.POOL_MEM_PRECURSOR
POOL_CATALYST = base.POOL_CATALYST
POOL_TRANSPORTER_PRECURSOR = base.POOL_TRANSPORTER_PRECURSOR
POOL_WASTE = base.POOL_WASTE
POOL_ALT = 7
POOL_INTERMEDIATE = 8
POOL_NUCLEOTIDE = 9
POOL_COUNT = 10

CHANNEL_FUEL = base.CHANNEL_FUEL
CHANNEL_MINERAL = base.CHANNEL_MINERAL
CHANNEL_WASTE = base.CHANNEL_WASTE
CHANNEL_ALT = 3
CHANNEL_COUNT = 4

MEMBRANE_SEGMENTS = base.MEMBRANE_SEGMENTS
BASE_RADIUS = base.BASE_RADIUS
MIN_RADIUS = base.MIN_RADIUS
MAX_RADIUS = base.MAX_RADIUS
INITIAL_MEMBRANE_MASS = base.INITIAL_MEMBRANE_MASS

# Genome physics.
ALPHABET_SIZE = 8
START_MARKER = (7, 7)
STOP_MARKER = (0, 0)
GENE_PAYLOAD = 12
GENE_SPAN = 2 + GENE_PAYLOAD + 2
MIN_GENOME_LENGTH = 32
MAX_GENOME_LENGTH = 384
MONOMER_MASS = 0.00082
REPLICATION_ATP_PER_SYMBOL = 0.0012

# Protein roles encoded by the first payload symbol.
ROLE_ENERGY = 0
ROLE_MEMBRANE = 1
ROLE_TRANSPORTER = 2
ROLE_REPLICASE = 3
ROLE_TRANSLATOR = 4
ROLE_NUCLEOTIDE = 5
ROLE_GENERIC = 6
ROLE_REGULATOR = 7
ROLE_NAMES = (
    'energy', 'membrane', 'transporter', 'replicase',
    'translator', 'nucleotide', 'generic', 'regulator',
)

# Generic reaction edges.  Material is preserved; chemical potential gaps
# generate or consume ATP with a deliberately lossy efficiency.
REACTION_FUEL_TO_INTERMEDIATE = 0
REACTION_ALT_TO_INTERMEDIATE = 1
REACTION_INTERMEDIATE_TO_WASTE = 2
REACTION_WASTE_TO_INTERMEDIATE = 3
REACTION_NAMES = (
    'fuel->intermediate',
    'alt->intermediate',
    'intermediate->waste',
    'waste->intermediate',
)
REACTION_SOURCE = (
    POOL_FUEL, POOL_ALT, POOL_INTERMEDIATE, POOL_WASTE,
)
REACTION_PRODUCT = (
    POOL_INTERMEDIATE, POOL_INTERMEDIATE, POOL_WASTE, POOL_INTERMEDIATE,
)
REACTION_POTENTIAL = {
    POOL_FUEL: 3.00,
    POOL_ALT: 2.00,
    POOL_INTERMEDIATE: 2.40,
    POOL_WASTE: 0.10,
}

clamp = base.clamp
wrapped_delta = base.wrapped_delta
torus_distance = base.torus_distance
unit_vector = base.unit_vector
safe_div = base.safe_div
finite_array = base.finite_array
circular_smooth = base.circular_smooth
_atomic_pickle = base._atomic_pickle
_memory_peak_mb_estimate = base._memory_peak_mb_estimate


def _sequence_hash(sequence):
    raw = np.asarray(sequence, dtype=np.uint8).tobytes()
    return hashlib.sha1(raw).hexdigest()[:12]


def _gene_fingerprint(payload):
    value = 0
    for symbol in payload:
        value = value * ALPHABET_SIZE + int(symbol)
    return int(value)


def make_gene(role, parameter=0, promoter=6, efficiency=5, regulator=3,
              fidelity=5, localisation=0, spare=None):
    """Create one delimited gene in the fixed artificial decoding grammar."""
    payload = [
        int(role) % 8,
        int(parameter) % 8,
        int(regulator) % 8,
        int(promoter) % 8,
        int(efficiency) % 8,
        int(fidelity) % 8,
        int(localisation) % 8,
        3,
        4,
        2,
        5,
        1,
    ]
    if spare is not None:
        for index, value in enumerate(list(spare)[:4]):
            payload[8 + index] = int(value) % 8
    return np.asarray(START_MARKER + tuple(payload) + STOP_MARKER, dtype=np.uint8)


def founding_genome():
    """Genome with a viable core and an incomplete alternative pathway.

    The founding sequence contains ``alt -> intermediate`` but no active
    ``intermediate -> waste`` reaction.  Thus alternative substrate is not a
    profitable energy source until a heritable mutation creates the missing
    edge.
    """
    genes = [
        make_gene(ROLE_ENERGY, parameter=0, promoter=7, efficiency=6),
        make_gene(ROLE_MEMBRANE, parameter=0, promoter=6, efficiency=5),
        make_gene(ROLE_TRANSPORTER, parameter=0, promoter=6, efficiency=5),
        make_gene(ROLE_REPLICASE, parameter=0, promoter=6, efficiency=5),
        make_gene(ROLE_TRANSLATOR, parameter=0, promoter=7, efficiency=5),
        make_gene(ROLE_NUCLEOTIDE, parameter=0, promoter=6, efficiency=5),
        make_gene(ROLE_GENERIC, parameter=REACTION_ALT_TO_INTERMEDIATE,
                  promoter=3, efficiency=4),
        make_gene(ROLE_REGULATOR, parameter=0, promoter=4, efficiency=4),
    ]
    return np.concatenate(genes).astype(np.uint8)


def parse_genes(sequence):
    """Return decoded gene records from a possibly damaged variable sequence."""
    sequence = np.asarray(sequence, dtype=np.uint8).reshape(-1)
    records = []
    length = len(sequence)
    index = 0
    while index + GENE_SPAN <= length:
        if (int(sequence[index]), int(sequence[index + 1])) != START_MARKER:
            index += 1
            continue
        stop_index = index + 2 + GENE_PAYLOAD
        if (int(sequence[stop_index]), int(sequence[stop_index + 1])) != STOP_MARKER:
            index += 1
            continue
        payload = tuple(int(value) for value in sequence[index + 2:stop_index])
        role = payload[0] % 8
        record = {
            'start': int(index),
            'payload': payload,
            'fingerprint': _gene_fingerprint(payload),
            'role': role,
            'role_name': ROLE_NAMES[role],
            'parameter': payload[1] % 8,
            'promoter': 0.18 + 1.22 * (payload[3] / 7.0),
            'efficiency': 0.52 + 0.96 * (payload[4] / 7.0),
            'fidelity': 0.45 + 0.54 * (payload[5] / 7.0),
            'localisation': payload[6] % 4,
            'regulator': payload[2] % 8,
        }
        if role == ROLE_GENERIC:
            record['reaction'] = payload[1] % len(REACTION_NAMES)
            record['reaction_name'] = REACTION_NAMES[record['reaction']]
        records.append(record)
        index += GENE_SPAN
    return records


def genome_gene_counts(sequence):
    counts = np.zeros(8, dtype=int)
    reactions = np.zeros(len(REACTION_NAMES), dtype=int)
    for record in parse_genes(sequence):
        counts[record['role']] += 1
        if record['role'] == ROLE_GENERIC:
            reactions[record['reaction']] += 1
    return counts, reactions


def sequence_has_novel_path(sequence):
    _, reactions = genome_gene_counts(sequence)
    return bool(
        reactions[REACTION_ALT_TO_INTERMEDIATE] > 0
        and reactions[REACTION_INTERMEDIATE_TO_WASTE] > 0
    )


class GeneticConfig(base.CellConfig):
    """Physical and hereditary ablation switches."""

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
    ):
        super(GeneticConfig, self).__init__(
            membrane_synthesis=membrane_synthesis,
            catalyst_synthesis=catalyst_synthesis,
            transport=transport,
            targeted_repair=targeted_repair,
            division=division,
            waste_export=waste_export,
            external_inflow=external_inflow,
            environmental_damage=environmental_damage,
        )
        self.gene_expression = bool(gene_expression)
        self.genome_replication = bool(genome_replication)
        self.mutation = bool(mutation)
        self.variable_length = bool(variable_length)
        self.gene_duplication = bool(gene_duplication)
        self.generic_reactions = bool(generic_reactions)
        self.composition_inheritance = bool(composition_inheritance)
        self.external_replicase = bool(external_replicase)
        self.external_translator = bool(external_translator)
        self.mutation_rate = float(mutation_rate)
        self.structural_rate = float(structural_rate)
        self.alt_niche = bool(alt_niche)

    @classmethod
    def from_state(cls, state):
        return cls(**dict(state))


def mutate_sequence(sequence, rng, config, nucleotide_budget=None):
    """Copy a sequence with physical mutation operations.

    Returns ``(mutated, mass_delta_symbols, event_counts)``.  Positive length
    changes require the caller to supply extra monomer material; negative
    changes return material.
    """
    original = np.asarray(sequence, dtype=np.uint8).copy()
    candidate = original.copy()
    events = {
        'substitution': 0,
        'insertion': 0,
        'deletion': 0,
        'duplication': 0,
        'inversion': 0,
        'transposition': 0,
    }
    if not config.mutation:
        return candidate, 0, events

    rate = max(0.0, float(config.mutation_rate))
    if rate > 0.0 and len(candidate):
        mask = rng.random(len(candidate)) < rate
        indices = np.where(mask)[0]
        for index in indices:
            old = int(candidate[index])
            new = int(rng.integers(0, ALPHABET_SIZE - 1))
            if new >= old:
                new += 1
            candidate[index] = new % ALPHABET_SIZE
        events['substitution'] = int(len(indices))

    structural = max(0.0, float(config.structural_rate))
    if config.variable_length and len(candidate) > 0:
        if rng.random() < structural * 0.55 and len(candidate) < MAX_GENOME_LENGTH:
            count = int(rng.integers(1, min(5, MAX_GENOME_LENGTH - len(candidate)) + 1))
            position = int(rng.integers(0, len(candidate) + 1))
            insert = rng.integers(0, ALPHABET_SIZE, count, dtype=np.uint8)
            candidate = np.concatenate([candidate[:position], insert, candidate[position:]])
            events['insertion'] += count
        if rng.random() < structural * 0.50 and len(candidate) > MIN_GENOME_LENGTH:
            count = int(rng.integers(1, min(5, len(candidate) - MIN_GENOME_LENGTH) + 1))
            position = int(rng.integers(0, len(candidate) - count + 1))
            candidate = np.concatenate([candidate[:position], candidate[position + count:]])
            events['deletion'] += count
        if (
            config.gene_duplication
            and rng.random() < structural * 0.42
            and len(candidate) + GENE_SPAN <= MAX_GENOME_LENGTH
        ):
            genes = parse_genes(candidate)
            if genes:
                gene = genes[int(rng.integers(0, len(genes)))]
                start = gene['start']
                fragment = candidate[start:start + GENE_SPAN].copy()
                position = int(rng.integers(0, len(candidate) + 1))
                candidate = np.concatenate([candidate[:position], fragment, candidate[position:]])
                events['duplication'] += int(len(fragment))

    if len(candidate) >= 4 and rng.random() < structural * 0.28:
        left = int(rng.integers(0, len(candidate) - 2))
        right = int(rng.integers(left + 2, min(len(candidate), left + 28) + 1))
        candidate[left:right] = candidate[left:right][::-1]
        events['inversion'] += 1

    if len(candidate) >= 8 and rng.random() < structural * 0.20:
        count = int(rng.integers(2, min(12, len(candidate) // 3) + 1))
        start = int(rng.integers(0, len(candidate) - count + 1))
        fragment = candidate[start:start + count].copy()
        remainder = np.concatenate([candidate[:start], candidate[start + count:]])
        position = int(rng.integers(0, len(remainder) + 1))
        candidate = np.concatenate([remainder[:position], fragment, remainder[position:]])
        events['transposition'] += 1

    if len(candidate) < MIN_GENOME_LENGTH:
        padding = rng.integers(0, ALPHABET_SIZE, MIN_GENOME_LENGTH - len(candidate), dtype=np.uint8)
        candidate = np.concatenate([candidate, padding])
    if len(candidate) > MAX_GENOME_LENGTH:
        candidate = candidate[:MAX_GENOME_LENGTH]

    delta = int(len(candidate) - len(original))
    if nucleotide_budget is not None and delta > int(nucleotide_budget):
        # Structural expansion cannot conjure monomers.  Trim the excess from
        # the end; substitutions and mass-neutral rearrangements remain.
        candidate = candidate[:len(original) + int(nucleotide_budget)]
        delta = int(len(candidate) - len(original))
    return candidate.astype(np.uint8), delta, events


class GeneticParticleField(base.ParticleField):
    """External matter with a fourth, alternative substrate species."""

    def __init__(self, rng, initial=True, alt_niche=False):
        self.alt_niche = bool(alt_niche)
        self.recycled_alt_buffer = 0.0
        super(GeneticParticleField, self).__init__(rng, initial=False)
        if initial:
            self.seed_initial()

    def seed_initial(self):
        if self.alt_niche:
            specification = (
                (PARTICLE_FUEL, 34, 0.025),
                (PARTICLE_MINERAL, 76, 0.028),
                (PARTICLE_WASTE, 12, 0.018),
                (PARTICLE_ALT, 102, 0.031),
            )
        else:
            specification = (
                (PARTICLE_FUEL, 82, 0.030),
                (PARTICLE_MINERAL, 72, 0.028),
                (PARTICLE_WASTE, 12, 0.018),
                (PARTICLE_ALT, 58, 0.028),
            )
        for kind, count, mean_amount in specification:
            positions = self.rng.random((count, 2))
            amounts = np.clip(
                self.rng.normal(mean_amount, mean_amount * 0.18, count),
                mean_amount * 0.45,
                mean_amount * 1.55,
            )
            self.add_many(kind, positions, amounts, count_as_injection=False)

    def _enforce_limit(self):
        limit = 440
        if len(self.amount) <= limit:
            return
        order = np.argsort(self.amount)
        remove = order[:len(self.amount) - limit]
        keep = np.ones(len(self.amount), dtype=bool)
        keep[remove] = False
        for index in remove:
            amount = float(self.amount[index])
            kind = int(self.kind[index])
            if kind == PARTICLE_MINERAL:
                self.recycled_mineral_buffer += amount
            elif kind == PARTICLE_FUEL:
                self.recycled_fuel_buffer += amount
            elif kind == PARTICLE_ALT:
                self.recycled_alt_buffer += amount
            else:
                self.recycled_fuel_buffer += 0.55 * amount
                self.recycled_mineral_buffer += 0.35 * amount
                self.dissipated_material += 0.10 * amount
        self.pos = self.pos[keep]
        self.kind = self.kind[keep]
        self.amount = self.amount[keep]

    def material_total(self):
        return float(
            np.sum(self.amount)
            + self.recycled_fuel_buffer
            + self.recycled_mineral_buffer
            + self.recycled_alt_buffer
        )

    def totals_by_kind(self):
        return [float(np.sum(self.amount[self.kind == kind])) for kind in range(4)]

    def step_diffusion(self, dt):
        if len(self.amount) == 0:
            return
        diffusion = np.asarray([0.010, 0.008, 0.006, 0.009], dtype=float)[self.kind]
        noise = self.rng.normal(0.0, 1.0, self.pos.shape)
        self.pos = (self.pos + noise * np.sqrt(2.0 * diffusion[:, None] * dt)) % 1.0

    def recycle_waste(self, dt):
        super(GeneticParticleField, self).recycle_waste(dt)
        buffer_value = float(self.recycled_alt_buffer)
        while buffer_value >= 0.025 and len(self.amount) < 440:
            self.add_particle(PARTICLE_ALT, self.rng.random(2), 0.025, count_as_injection=False)
            buffer_value -= 0.025
        self.recycled_alt_buffer = buffer_value

    def external_inflow(self, dt, enabled=True):
        if not enabled:
            return
        fuel_rate = 0.0065 if self.alt_niche else 0.0160
        alt_rate = 0.0180 if self.alt_niche else 0.0090
        for kind, rate in (
            (PARTICLE_FUEL, fuel_rate),
            (PARTICLE_MINERAL, 0.0060),
            (PARTICLE_ALT, alt_rate),
        ):
            expected = rate * dt
            if self.rng.random() < expected / 0.020:
                self.add_particle(kind, self.rng.random(2), 0.020, count_as_injection=True)

    def state_dict(self):
        state = super(GeneticParticleField, self).state_dict()
        state['alt_niche'] = bool(self.alt_niche)
        state['recycled_alt_buffer'] = float(self.recycled_alt_buffer)
        return state

    @classmethod
    def from_state(cls, rng, state):
        field = cls(rng, initial=False, alt_niche=state.get('alt_niche', False))
        field.pos = np.asarray(state['pos'], dtype=float).copy()
        field.kind = np.asarray(state['kind'], dtype=np.int16).copy()
        field.amount = np.asarray(state['amount'], dtype=float).copy()
        field.recycled_fuel_buffer = float(state.get('recycled_fuel_buffer', 0.0))
        field.recycled_mineral_buffer = float(state.get('recycled_mineral_buffer', 0.0))
        field.recycled_alt_buffer = float(state.get('recycled_alt_buffer', 0.0))
        field.injected_material = float(state.get('injected_material', 0.0))
        field.dissipated_material = float(state.get('dissipated_material', 0.0))
        return field


class GeneticProtoCell(base.ProtoCell):
    """0.1 material cell whose catalytic specificity is genome-derived."""

    def __init__(self, cell_id, rng, position=None, generation=0, lineage=0,
                 bootstrap=True):
        super(GeneticProtoCell, self).__init__(
            cell_id, rng, position=position, generation=generation, lineage=lineage
        )
        if self.transporters.shape[1] < CHANNEL_COUNT:
            extra = np.zeros((MEMBRANE_SEGMENTS, CHANNEL_COUNT - self.transporters.shape[1]))
            extra[:, 0] = base.INITIAL_SEGMENT_MASS * 0.045
            self.transporters = np.concatenate([self.transporters, extra], axis=1)
        if len(self.pools) < POOL_COUNT:
            extended = np.zeros((POOL_COUNT,), dtype=float)
            extended[:len(self.pools)] = self.pools
            self.pools = extended
        self.pools[POOL_ALT] = 0.025
        self.pools[POOL_INTERMEDIATE] = 0.015
        self.pools[POOL_NUCLEOTIDE] = 0.205
        self.pools[POOL_ATP] = 0.46
        self.alt_contact_trace = np.zeros((MEMBRANE_SEGMENTS,), dtype=float)
        self.last_uptake_alt = 0.0

        self.genomes = []
        self.proteins = {}
        self.gene_specs = {}
        self.replication_template = None
        self.replication_copy = []
        self.replication_fractional = 0.0
        self.replication_cycles = 0
        self.mutation_events = {
            'substitution': 0,
            'insertion': 0,
            'deletion': 0,
            'duplication': 0,
            'inversion': 0,
            'transposition': 0,
        }
        self.genome_damage_events = 0
        self.information_silence_timer = 0.0
        self.novel_path_first_age = None
        self.novel_path_flux = 0.0
        self.last_translation = 0.0
        self.last_replication_symbols = 0
        self.last_generic_flux = np.zeros(len(REACTION_NAMES), dtype=float)

        if bootstrap:
            genome = founding_genome()
            self.genomes = [genome]
            self._refresh_gene_cache()
            initial_by_role = {
                ROLE_ENERGY: 0.072,
                ROLE_MEMBRANE: 0.043,
                ROLE_TRANSPORTER: 0.038,
                ROLE_REPLICASE: 0.046,
                ROLE_TRANSLATOR: 0.058,
                ROLE_NUCLEOTIDE: 0.038,
                ROLE_GENERIC: 0.026,
                ROLE_REGULATOR: 0.024,
            }
            for fingerprint, spec in self.gene_specs.items():
                self.proteins[fingerprint] = self.proteins.get(fingerprint, 0.0) + initial_by_role[spec['role']]
            self._sync_protein_pool()

    # ---- Genome / protein bookkeeping -----------------------------------------

    def genome_mass(self):
        complete = sum(len(genome) for genome in self.genomes) * MONOMER_MASS
        partial = len(self.replication_copy) * MONOMER_MASS
        return float(complete + partial)

    def material_mass(self):
        pool_material = float(np.sum(self.pools) - self.pools[POOL_ATP])
        return float(
            pool_material
            + np.sum(self.membrane)
            + np.sum(self.transporters)
            + self.septum_mass
            + self.genome_mass()
        )

    def osmolyte(self):
        # Osmotic pressure depends on particle count, not bulk macromolecular
        # mass.  Proteins, precursor aggregates, and genome polymers therefore
        # contribute far less than small soluble metabolites.
        small = (
            self.pools[POOL_FUEL]
            + self.pools[POOL_MINERAL]
            + self.pools[POOL_ATP]
            + self.pools[POOL_WASTE]
            + self.pools[POOL_ALT]
            + self.pools[POOL_INTERMEDIATE]
            + self.pools[POOL_NUCLEOTIDE]
        )
        aggregate = 0.25 * (
            self.pools[POOL_MEM_PRECURSOR]
            + self.pools[POOL_TRANSPORTER_PRECURSOR]
        )
        proteins = 0.12 * self.pools[POOL_CATALYST]
        polymer = 0.03 * self.genome_mass()
        return float(small + aggregate + proteins + polymer)

    def _refresh_gene_cache(self):
        cache = {}
        for genome in self.genomes:
            for spec in parse_genes(genome):
                fingerprint = spec['fingerprint']
                if fingerprint not in cache:
                    item = dict(spec)
                    item['copy_number'] = 1
                    cache[fingerprint] = item
                else:
                    cache[fingerprint]['copy_number'] += 1
        self.gene_specs = cache

    def _sync_protein_pool(self):
        clean = {}
        for key, value in self.proteins.items():
            value = float(value)
            if np.isfinite(value) and value > 1e-10:
                clean[int(key)] = value
        self.proteins = clean
        self.pools[POOL_CATALYST] = float(sum(clean.values()))

    def role_activity(self, role):
        total = 0.0
        for fingerprint, amount in self.proteins.items():
            spec = self.gene_specs.get(fingerprint)
            if spec is not None and spec['role'] == role:
                total += amount * spec['efficiency']
        return float(total / 0.040)

    def reaction_activity(self, reaction):
        total = 0.0
        for fingerprint, amount in self.proteins.items():
            spec = self.gene_specs.get(fingerprint)
            if (
                spec is not None
                and spec['role'] == ROLE_GENERIC
                and spec.get('reaction') == reaction
            ):
                total += amount * spec['efficiency']
        return float(total / 0.032)

    def active_gene_count(self):
        return int(sum(item.get('copy_number', 1) for item in self.gene_specs.values()))

    def mean_genome_length(self):
        if not self.genomes:
            return 0.0
        return float(np.mean([len(genome) for genome in self.genomes]))

    def has_novel_path(self):
        return bool(
            self.reaction_activity(REACTION_ALT_TO_INTERMEDIATE) > 0.04
            and self.reaction_activity(REACTION_INTERMEDIATE_TO_WASTE) > 0.04
        )

    def reaction_loop_strength(self):
        physical = super(GeneticProtoCell, self).reaction_loop_strength()
        information = min(1.0, len(self.genomes))
        translation = self.role_activity(ROLE_TRANSLATOR)
        replication = self.role_activity(ROLE_REPLICASE)
        info_factor = information * (translation / (0.25 + translation))
        heredity_factor = 0.35 + 0.65 * (replication / (0.25 + replication))
        return float(physical * info_factor * heredity_factor)

    # ---- Surface exchange ------------------------------------------------------

    def surface_exchange(self, field, dt, config):
        super(GeneticProtoCell, self).surface_exchange(field, dt, config)
        self.last_uptake_alt = 0.0
        if not self.alive or len(field.amount) == 0:
            return
        indices = np.where(field.kind == PARTICLE_ALT)[0]
        if not len(indices):
            return
        deltas = (field.pos[indices] - self.pos[None, :] + 0.5) % 1.0 - 0.5
        distances = np.linalg.norm(deltas, axis=1)
        closure_values = self.closure_array()
        for local_index, particle_index in enumerate(indices):
            distance = float(distances[local_index])
            if abs(distance - self.radius) > 0.016:
                continue
            amount = float(field.amount[particle_index])
            if amount <= 1e-9:
                continue
            delta = deltas[local_index]
            if distance <= 1e-10:
                delta = np.array([1.0, 0.0], dtype=float)
                distance = 1e-10
            segment = self.segment_for_delta(delta)
            closure = float(closure_values[segment])
            gap = 1.0 - closure
            self.alt_contact_trace[segment] += amount * dt * 3.0
            facilitated = 0.0
            powered = 0.0
            if config.transport:
                density = self.transporters[segment, CHANNEL_ALT] / max(self.membrane[segment], 1e-8)
                saturation = amount / (0.018 + amount)
                shortage = 1.0 / (1.0 + (self.pools[POOL_ALT] / 0.32) ** 4)
                facilitated = 0.58 * density * closure * saturation * shortage * dt
                if self.pools[POOL_ATP] > 0.030:
                    powered = min(
                        0.36 * density * closure * saturation * shortage * dt,
                        max(0.0, amount - facilitated),
                        max(0.0, self.pools[POOL_ATP] - 0.028) / 0.045,
                    )
            transported = min(amount, facilitated + powered)
            passive = min(max(0.0, amount - transported), 0.065 * gap ** 2.2 * amount * dt)
            transfer = max(0.0, transported + passive)
            if transfer <= 0.0:
                continue
            field.amount[particle_index] -= transfer
            self.pools[POOL_ALT] += transfer
            self.pools[POOL_ATP] -= 0.045 * min(powered, transfer)
            self.last_uptake_alt += transfer
            self.surface_flux += (delta / distance) * transfer

    # ---- Gene expression and chemistry ----------------------------------------

    def _protein_need(self, spec):
        role = spec['role']
        if role == ROLE_ENERGY:
            return clamp((0.34 - self.pools[POOL_ATP]) * 3.0 + 0.35, 0.12, 1.8)
        if role == ROLE_MEMBRANE:
            return clamp((1.02 - self.closure()) * 4.0 + self.tension() * 2.2 + 0.20, 0.10, 2.0)
        if role == ROLE_TRANSPORTER:
            return clamp(
                0.35 + max(0.0, 0.25 - self.pools[POOL_FUEL])
                + max(0.0, 0.23 - self.pools[POOL_MINERAL])
                + max(0.0, 0.18 - self.pools[POOL_ALT])
                + self.pools[POOL_WASTE],
                0.10, 1.8,
            )
        if role == ROLE_REPLICASE:
            return 1.6 if len(self.genomes) < 2 else 0.28
        if role == ROLE_TRANSLATOR:
            return clamp(1.2 - self.role_activity(ROLE_TRANSLATOR) * 0.22, 0.22, 1.2)
        if role == ROLE_NUCLEOTIDE:
            return clamp((0.22 - self.pools[POOL_NUCLEOTIDE]) * 5.0 + 0.18, 0.08, 1.7)
        if role == ROLE_GENERIC:
            reaction = spec.get('reaction', 0)
            source = REACTION_SOURCE[reaction]
            return clamp(0.18 + self.pools[source] * 2.2, 0.10, 1.5)
        return clamp(0.22 + np.mean(self.damage_trace) * 3.0, 0.10, 1.2)

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
            weight = (
                spec['promoter']
                * spec.get('copy_number', 1)
                * self._protein_need(spec)
            )
            weighted.append((fingerprint, spec, weight))
            total_weight += weight
        if total_weight <= 0.0:
            return
        translation_capacity = dt * 0.0060 * translator
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
            self.proteins[fingerprint] = self.proteins.get(fingerprint, 0.0) + desired
            self.last_translation += desired / max(dt, 1e-9)
        self._sync_protein_pool()

    def _apply_generic_reactions(self, dt, config):
        self.last_generic_flux[:] = 0.0
        self.novel_path_flux = 0.0
        if not config.generic_reactions:
            return
        for reaction in range(len(REACTION_NAMES)):
            activity = self.reaction_activity(reaction)
            if activity <= 1e-7:
                continue
            source = REACTION_SOURCE[reaction]
            product = REACTION_PRODUCT[reaction]
            available = float(self.pools[source])
            saturation = available / (0.045 + available)
            desired = dt * 0.030 * activity * saturation
            # Uphill precursor-forming reactions are self-limited by product
            # accumulation.  This prevents an incomplete pathway from draining
            # the entire ATP pool before a downstream catalyst exists.
            if reaction in (REACTION_ALT_TO_INTERMEDIATE, REACTION_WASTE_TO_INTERMEDIATE):
                product_inhibition = 1.0 / (1.0 + (self.pools[product] / 0.060) ** 3)
                desired *= product_inhibition
            source_potential = REACTION_POTENTIAL[source]
            product_potential = REACTION_POTENTIAL[product]
            delta = source_potential - product_potential
            if delta >= 0.0:
                amount = min(available, desired)
                if amount > 0.0:
                    self.pools[source] -= amount
                    self.pools[product] += amount
                    self.pools[POOL_ATP] += amount * delta * 0.68
            else:
                atp_cost = (-delta) * 1.28
                amount = min(
                    available,
                    desired,
                    max(0.0, self.pools[POOL_ATP] - 0.040) / max(atp_cost, 1e-9),
                )
                if amount > 0.0:
                    self.pools[source] -= amount
                    self.pools[product] += amount
                    self.pools[POOL_ATP] -= amount * atp_cost
            self.last_generic_flux[reaction] = amount / max(dt, 1e-9)
            if reaction == REACTION_INTERMEDIATE_TO_WASTE:
                self.novel_path_flux += amount / max(dt, 1e-9)
            if amount > 1e-8:
                self.reaction_events += 1

    def _synthesise_precursors(self, dt, config):
        # Genome-derived energy catalyst: fuel matter becomes waste while ATP
        # carries the released chemical potential.
        energy_activity = self.role_activity(ROLE_ENERGY)
        volume = max(0.20, (self.radius / BASE_RADIUS) ** 2)
        fuel = self.pools[POOL_FUEL] / volume
        waste = self.pools[POOL_WASTE] / volume
        inhibition = 1.0 / (1.0 + 2.8 * waste)
        cat_rate = 0.17 * energy_activity * fuel / (0.14 + fuel) * inhibition
        amount = min(self.pools[POOL_FUEL], cat_rate * dt)
        self.pools[POOL_FUEL] -= amount
        self.pools[POOL_WASTE] += amount
        self.pools[POOL_ATP] += 2.45 * amount
        self.last_catalysis = amount / max(dt, 1e-9)

        membrane_activity = self.role_activity(ROLE_MEMBRANE)
        need_membrane = clamp(
            (1.03 - self.closure()) * 2.5
            + self.tension() * 2.2
            + max(0.0, 0.17 - self.pools[POOL_MEM_PRECURSOR]) * 1.5,
            0.0, 2.5,
        )
        desired = dt * 0.070 * membrane_activity * need_membrane
        amount = min(
            desired,
            self.pools[POOL_FUEL] / 0.62,
            self.pools[POOL_MINERAL] / 0.38,
            max(0.0, self.pools[POOL_ATP] - 0.044) / 0.42,
        ) if config.membrane_synthesis else 0.0
        self.pools[POOL_FUEL] -= 0.62 * amount
        self.pools[POOL_MINERAL] -= 0.38 * amount
        self.pools[POOL_ATP] -= 0.42 * amount
        self.pools[POOL_MEM_PRECURSOR] += amount

        transport_activity = self.role_activity(ROLE_TRANSPORTER)
        need_transport = clamp(
            0.35 + max(0.0, 0.22 - self.pools[POOL_FUEL])
            + max(0.0, 0.20 - self.pools[POOL_MINERAL])
            + max(0.0, 0.18 - self.pools[POOL_ALT])
            + 1.2 * self.pools[POOL_WASTE],
            0.0, 1.8,
        )
        desired = dt * 0.017 * transport_activity * need_transport
        amount = min(
            desired,
            self.pools[POOL_FUEL] / 0.65,
            self.pools[POOL_MINERAL] / 0.35,
            max(0.0, self.pools[POOL_ATP] - 0.044) / 0.70,
        ) if config.transport else 0.0
        self.pools[POOL_FUEL] -= 0.65 * amount
        self.pools[POOL_MINERAL] -= 0.35 * amount
        self.pools[POOL_ATP] -= 0.70 * amount
        self.pools[POOL_TRANSPORTER_PRECURSOR] += amount

        nucleotide_activity = self.role_activity(ROLE_NUCLEOTIDE)
        nucleotide_need = clamp((0.24 - self.pools[POOL_NUCLEOTIDE]) * 5.0 + 0.16, 0.0, 1.7)
        desired = dt * 0.014 * nucleotide_activity * nucleotide_need
        amount = min(
            desired,
            self.pools[POOL_FUEL] / 0.58,
            self.pools[POOL_MINERAL] / 0.42,
            max(0.0, self.pools[POOL_ATP] - 0.044) / 0.66,
        )
        self.pools[POOL_FUEL] -= 0.58 * amount
        self.pools[POOL_MINERAL] -= 0.42 * amount
        self.pools[POOL_ATP] -= 0.66 * amount
        self.pools[POOL_NUCLEOTIDE] += amount

    def _assemble_surface(self, dt, config):
        self.last_assembly = 0.0
        if config.membrane_synthesis and self.pools[POOL_MEM_PRECURSOR] > 1e-9:
            required = self.required_segment_mass()
            deficit = np.maximum(required * 1.06 - self.membrane, 0.0)
            if config.targeted_repair:
                weights = deficit * 8.0 + self.damage_trace * 2.5 + 0.02
            else:
                weights = np.ones(MEMBRANE_SEGMENTS, dtype=float)
            if float(np.sum(deficit)) < 1e-5:
                weights += 0.18
            weights = np.maximum(weights, 1e-8)
            weights /= float(np.sum(weights))
            assembly_rate = 0.070 * max(0.05, self.role_activity(ROLE_MEMBRANE)) * (
                0.45 + 1.4 * (1.0 - self.closure()) + 0.6 * self.tension()
            )
            assembly = min(
                self.pools[POOL_MEM_PRECURSOR],
                assembly_rate * dt,
                max(0.0, self.pools[POOL_ATP] - 0.045) / 0.34,
            )
            if assembly > 0.0:
                self.membrane += weights * assembly
                self.pools[POOL_MEM_PRECURSOR] -= assembly
                self.pools[POOL_ATP] -= 0.34 * assembly
                self.last_assembly = assembly / max(dt, 1e-9)

        if self.pools[POOL_TRANSPORTER_PRECURSOR] > 1e-9:
            channel_need = np.asarray([
                max(0.08, 0.34 - self.pools[POOL_FUEL]),
                max(0.08, 0.29 - self.pools[POOL_MINERAL]),
                max(0.05, self.pools[POOL_WASTE] * 1.4),
                max(0.08, 0.26 - self.pools[POOL_ALT]),
            ], dtype=float)
            channel_need /= float(np.sum(channel_need))
            insert = min(
                self.pools[POOL_TRANSPORTER_PRECURSOR],
                0.034 * max(0.05, self.role_activity(ROLE_TRANSPORTER)) * dt,
                max(0.0, self.pools[POOL_ATP] - 0.045) / 0.40,
            )
            if insert > 0.0:
                for channel in range(CHANNEL_COUNT):
                    if channel == CHANNEL_FUEL:
                        local = self.contact_trace[:, 0] + 0.03
                    elif channel == CHANNEL_MINERAL:
                        local = self.contact_trace[:, 1] + 0.03
                    elif channel == CHANNEL_ALT:
                        local = self.alt_contact_trace + 0.03
                    else:
                        local = self.closure_array() + self.damage_trace + 0.03
                    local = np.maximum(local, 1e-8)
                    local /= float(np.sum(local))
                    self.transporters[:, channel] += local * insert * channel_need[channel]
                self.pools[POOL_TRANSPORTER_PRECURSOR] -= insert
                self.pools[POOL_ATP] -= 0.40 * insert

    def _replicate_genome(self, world, dt, config):
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
            template = self.genomes[int(world.rng.integers(0, len(self.genomes)))]
            self.replication_template = template.copy()
            self.replication_copy = []
            self.replication_fractional = 0.0

        template = self.replication_template
        if template is None:
            return
        sat_nucleotide = self.pools[POOL_NUCLEOTIDE] / (0.055 + self.pools[POOL_NUCLEOTIDE])
        sat_atp = self.pools[POOL_ATP] / (0.10 + self.pools[POOL_ATP])
        speed = 10.0 * replicase * sat_nucleotide * sat_atp
        self.replication_fractional += speed * dt
        requested = int(self.replication_fractional)
        self.replication_fractional -= requested
        for _ in range(requested):
            index = len(self.replication_copy)
            if index >= len(template):
                break
            if (
                self.pools[POOL_NUCLEOTIDE] < MONOMER_MASS
                or self.pools[POOL_ATP] < REPLICATION_ATP_PER_SYMBOL + 0.022
            ):
                break
            symbol = int(template[index])
            if config.mutation and world.rng.random() < config.mutation_rate:
                new = int(world.rng.integers(0, ALPHABET_SIZE - 1))
                if new >= symbol:
                    new += 1
                symbol = new % ALPHABET_SIZE
                self.mutation_events['substitution'] += 1
            self.replication_copy.append(symbol)
            self.pools[POOL_NUCLEOTIDE] -= MONOMER_MASS
            self.pools[POOL_ATP] -= REPLICATION_ATP_PER_SYMBOL
            self.last_replication_symbols += 1

        if len(self.replication_copy) < len(template):
            return

        copied = np.asarray(self.replication_copy, dtype=np.uint8)
        # Substitutions were already applied physically during copying.  Apply
        # only structural changes at completion to avoid double substitution.
        structural_config = GeneticConfig.from_state(config.state_dict())
        structural_config.mutation_rate = 0.0
        budget_symbols = int(self.pools[POOL_NUCLEOTIDE] / MONOMER_MASS)
        copied, delta_symbols, events = mutate_sequence(
            copied, world.rng, structural_config, nucleotide_budget=budget_symbols
        )
        if delta_symbols > 0:
            self.pools[POOL_NUCLEOTIDE] -= delta_symbols * MONOMER_MASS
        elif delta_symbols < 0:
            self.pools[POOL_NUCLEOTIDE] += (-delta_symbols) * MONOMER_MASS
        for name, count in events.items():
            self.mutation_events[name] += int(count)
        self.genomes.append(copied)
        self.replication_cycles += 1
        self.replication_template = None
        self.replication_copy = []
        self.replication_fractional = 0.0
        self._refresh_gene_cache()
        if self.novel_path_first_age is None and sequence_has_novel_path(copied):
            self.novel_path_first_age = float(self.age)

    def _decay_information_and_proteins(self, world, dt):
        volume = max(0.20, (self.radius / BASE_RADIUS) ** 2)
        waste_stress = self.pools[POOL_WASTE] / volume
        decay_rate = 0.0010 + 0.0022 * waste_stress
        lost = 0.0
        for fingerprint in list(self.proteins.keys()):
            amount = self.proteins[fingerprint] * decay_rate * dt
            self.proteins[fingerprint] -= amount
            lost += amount
        self.pools[POOL_WASTE] += lost
        self._sync_protein_pool()

        # Rare hydrolysis makes the information carrier a vulnerable material.
        probability = dt * (0.000018 + 0.000060 * waste_stress)
        if self.genomes and world.rng.random() < probability:
            genome_index = int(world.rng.integers(0, len(self.genomes)))
            genome = self.genomes[genome_index]
            if len(genome) > MIN_GENOME_LENGTH:
                index = int(world.rng.integers(0, len(genome)))
                self.genomes[genome_index] = np.concatenate([genome[:index], genome[index + 1:]])
                self.pools[POOL_WASTE] += MONOMER_MASS
                self.genome_damage_events += 1
                self._refresh_gene_cache()

    def metabolism(self, world, dt, config):
        if not self.alive:
            return
        self._refresh_gene_cache()
        self._sync_protein_pool()
        self._apply_generic_reactions(dt, config)
        self._synthesise_precursors(dt, config)

        maintenance = dt * (
            0.0042
            + 0.010 * self.pools[POOL_CATALYST]
            + 0.009 * float(np.sum(self.transporters))
            + 0.0025 * float(np.sum(self.membrane))
            + 0.008 * self.tension()
            + 0.000030 * sum(len(genome) for genome in self.genomes)
        )
        paid = min(self.pools[POOL_ATP], maintenance)
        self.pools[POOL_ATP] -= paid
        maintenance_shortfall = max(0.0, maintenance - paid)

        self.translate(dt, config)
        self._replicate_genome(world, dt, config)
        self._assemble_surface(dt, config)
        self._decay_information_and_proteins(world, dt)

        for channel in range(CHANNEL_COUNT):
            self.transporters[:, channel] = circular_smooth(
                self.transporters[:, channel], min(0.49, 0.10 * dt)
            )
        self.membrane = circular_smooth(self.membrane, min(0.49, 0.075 * dt))

        volume = max(0.20, (self.radius / BASE_RADIUS) ** 2)
        waste_stress = self.pools[POOL_WASTE] / volume
        tension = self.tension()
        membrane_decay_rate = 0.00032 + 0.0010 * waste_stress + 0.0012 * tension
        if maintenance_shortfall > 0.0:
            membrane_decay_rate += 0.015 * maintenance_shortfall / max(dt, 1e-9)
        membrane_loss = np.minimum(
            self.membrane,
            self.membrane * membrane_decay_rate * dt + self.damage_trace * 0.0008 * dt,
        )
        self.membrane -= membrane_loss
        self.pools[POOL_WASTE] += float(np.sum(membrane_loss))

        transporter_decay = np.minimum(
            self.transporters,
            self.transporters * (0.00075 + 0.0010 * waste_stress + 0.0015 * maintenance_shortfall) * dt,
        )
        self.transporters -= transporter_decay
        self.pools[POOL_WASTE] += float(np.sum(transporter_decay))

        self.pools[POOL_ATP] *= math.exp(-0.018 * dt)
        self.pools[POOL_ATP] = clamp(self.pools[POOL_ATP], 0.0, 1.8)
        self.contact_trace *= math.exp(-0.80 * dt)
        self.alt_contact_trace *= math.exp(-0.80 * dt)
        self.damage_trace *= math.exp(-0.55 * dt)
        self.damage_trace = np.clip(self.damage_trace, 0.0, 1.5)
        self.pools = np.maximum(self.pools, 0.0)

        self.export_waste(world.field, dt, config)
        self.leak(world, dt)
        self.update_radius(dt)
        self.update_motion(world.rng, dt)
        self.update_division(dt, config)
        self.update_viability(dt)
        self.age += dt

    def leak(self, world, dt):
        closure = self.closure_array()
        gap_weights = np.maximum(1.0 - closure, 0.0) ** 2.4
        gap_strength = float(np.mean(gap_weights))
        pressure = self.tension()
        leak_fraction = clamp(
            (0.065 * gap_strength + 0.080 * gap_strength * pressure) * dt,
            0.0, 0.18,
        )
        if leak_fraction <= 0.0:
            self.last_leak = 0.0
            return
        # Let 0.1 handle all inherited pools and ATP, then leak 0.2 pools using
        # the same opening fraction.  Protein composition must lose the same
        # fraction as the aggregate catalyst pool, otherwise the next sync would
        # recreate leaked protein matter.
        catalyst_before = float(self.pools[POOL_CATALYST])
        super(GeneticProtoCell, self).leak(world, dt)
        catalyst_after = float(self.pools[POOL_CATALYST])
        if catalyst_before > 1e-12 and catalyst_after < catalyst_before:
            scale = clamp(catalyst_after / catalyst_before, 0.0, 1.0)
            for fingerprint in list(self.proteins.keys()):
                self.proteins[fingerprint] *= scale
            self._sync_protein_pool()
        segment = int(np.argmax(gap_weights))
        angle = 2.0 * math.pi * (segment + 0.5) / MEMBRANE_SEGMENTS
        position = (self.pos + unit_vector(angle) * (self.radius + 0.004)) % 1.0
        extra_total = 0.0
        for pool_index, particle_kind in (
            (POOL_ALT, PARTICLE_ALT),
            (POOL_INTERMEDIATE, PARTICLE_WASTE),
            (POOL_NUCLEOTIDE, PARTICLE_WASTE),
        ):
            amount = self.pools[pool_index] * leak_fraction
            if amount <= 0.0:
                continue
            self.pools[pool_index] -= amount
            world.field.add_particle(
                particle_kind,
                (position + world.rng.normal(0.0, 0.003, 2)) % 1.0,
                amount,
                count_as_injection=False,
            )
            extra_total += amount
        # A partial polymer can escape/fragment through a large gap.
        if self.replication_copy and gap_strength > 0.08:
            count = min(len(self.replication_copy), max(1, int(len(self.replication_copy) * leak_fraction)))
            del self.replication_copy[-count:]
            amount = count * MONOMER_MASS
            world.field.add_particle(PARTICLE_WASTE, position, amount, count_as_injection=False)
            extra_total += amount
        self.last_leak += extra_total

    # ---- Division / viability --------------------------------------------------

    def ready_for_division(self):
        if not self.alive:
            return False
        return bool(
            len(self.genomes) >= 2
            and self.replication_template is None
            and self.closure() > 0.962
            and self.worst_gap() < 0.24
            and float(np.sum(self.membrane)) > INITIAL_MEMBRANE_MASS * 1.48
            and self.pools[POOL_CATALYST] > 0.25
            and self.pools[POOL_ATP] > 0.040
            and self.pools[POOL_MEM_PRECURSOR] > 0.055
            and self.material_mass() > 4.75
            and self.age > 48.0
        )

    def split(self, world):
        if not self.can_split() or len(self.genomes) < 2:
            return None
        before = self.material_mass()
        genome_mass = sum(len(genome) for genome in self.genomes) * MONOMER_MASS
        divisible_before = max(0.0, before - genome_mass)
        shed_fraction = 0.012
        shed_mass = divisible_before * shed_fraction
        remaining_factor = 1.0 - shed_fraction

        axis = unit_vector(self.division_axis)
        offset = axis * max(0.024, self.radius * 0.46)
        asymmetry = clamp(float(world.rng.normal(0.5, 0.025)), 0.44, 0.56)
        fractions = (asymmetry, 1.0 - asymmetry)

        genome_order = list(range(len(self.genomes)))
        world.rng.shuffle(genome_order)
        genome_assignments = [[self.genomes[genome_order[0]].copy()], [self.genomes[genome_order[1]].copy()]]
        for genome_index in genome_order[2:]:
            target = int(world.rng.integers(0, 2))
            genome_assignments[target].append(self.genomes[genome_index].copy())

        daughters = []
        parent_atp = float(self.pools[POOL_ATP])
        world.dissipated_energy += parent_atp * shed_fraction
        protein_items = list(self.proteins.items())
        for daughter_index, fraction in enumerate(fractions):
            daughter = GeneticProtoCell(
                world.next_cell_id + daughter_index,
                world.rng,
                position=(self.pos + (1.0 if daughter_index == 0 else -1.0) * offset) % 1.0,
                generation=self.generation + 1,
                lineage=self.lineage,
                bootstrap=False,
            )
            phase = 0 if daughter_index == 0 else MEMBRANE_SEGMENTS // 2
            daughter.membrane = np.roll(self.membrane, phase) * fraction * remaining_factor
            daughter.membrane += self.septum_mass * fraction * remaining_factor / MEMBRANE_SEGMENTS
            daughter.transporters = np.roll(self.transporters, phase, axis=0) * fraction * remaining_factor
            daughter.pools = self.pools * fraction * remaining_factor
            daughter.pools[POOL_ATP] *= 0.94
            world.dissipated_energy += float(self.pools[POOL_ATP] * fraction * remaining_factor * 0.06)
            daughter.genomes = genome_assignments[daughter_index]
            if world.config.composition_inheritance:
                daughter.proteins = {
                    fingerprint: amount * fraction * remaining_factor
                    for fingerprint, amount in protein_items
                }
            else:
                # Removing compositional heredity must not remove matter.
                # Inherited protein mass is hydrolysed to generic waste rather
                # than silently disappearing from the material ledger.
                inherited_protein_mass = float(daughter.pools[POOL_CATALYST])
                daughter.proteins = {}
                daughter.pools[POOL_WASTE] += inherited_protein_mass
            daughter._refresh_gene_cache()
            if not daughter.proteins:
                # A tiny physical bootstrap is retained only when the explicit
                # composition-inheritance ablation is disabled.  This is logged
                # as external assistance by the world.
                if world.config.external_translator or world.config.external_replicase:
                    for fingerprint, spec in daughter.gene_specs.items():
                        if spec['role'] in (ROLE_TRANSLATOR, ROLE_REPLICASE):
                            daughter.proteins[fingerprint] = 0.010
                            world.external_protein_assistance += 0.010
            daughter._sync_protein_pool()
            daughter.radius = max(MIN_RADIUS, self.radius * math.sqrt(fraction))
            daughter.vel = self.vel + world.rng.normal(0.0, 0.002, 2)
            daughter.division_progress = 0.0
            daughter.septum_mass = 0.0
            daughter.division_axis = float(world.rng.uniform(0.0, 2.0 * math.pi))
            daughter.age = 0.0
            daughter.mutation_events = dict(self.mutation_events)
            daughters.append(daughter)

        world.field.add_particle(PARTICLE_WASTE, self.pos, shed_mass, count_as_injection=False)
        after = sum(d.material_mass() for d in daughters) + shed_mass
        residual = before - after
        world.division_parent_material += before
        world.division_daughter_material += sum(d.material_mass() for d in daughters)
        world.division_shed_material += shed_mass
        world.division_residual += residual
        world.next_cell_id += 2
        return daughters

    def update_viability(self, dt):
        super(GeneticProtoCell, self).update_viability(dt)
        if not self.alive:
            return
        translator = self.role_activity(ROLE_TRANSLATOR)
        if not self.genomes or translator < 0.025:
            self.information_silence_timer += dt
        else:
            self.information_silence_timer = max(0.0, self.information_silence_timer - 1.5 * dt)
        if self.information_silence_timer > 75.0:
            self.alive = False
            self.death_reason = 'informational_collapse'

    # ---- Serialization ---------------------------------------------------------

    def state_dict(self):
        state = super(GeneticProtoCell, self).state_dict()
        state.update({
            'cell_class': 'GeneticProtoCell',
            'alt_contact_trace': self.alt_contact_trace.copy(),
            'last_uptake_alt': float(self.last_uptake_alt),
            'genomes': [genome.copy() for genome in self.genomes],
            'proteins': [(int(key), float(value)) for key, value in sorted(self.proteins.items())],
            'replication_template': None if self.replication_template is None else self.replication_template.copy(),
            'replication_copy': list(int(value) for value in self.replication_copy),
            'replication_fractional': float(self.replication_fractional),
            'replication_cycles': int(self.replication_cycles),
            'mutation_events': dict(self.mutation_events),
            'genome_damage_events': int(self.genome_damage_events),
            'information_silence_timer': float(self.information_silence_timer),
            'novel_path_first_age': self.novel_path_first_age,
            'novel_path_flux': float(self.novel_path_flux),
            'last_translation': float(self.last_translation),
            'last_replication_symbols': int(self.last_replication_symbols),
            'last_generic_flux': self.last_generic_flux.copy(),
        })
        return state

    @classmethod
    def from_state(cls, rng, state):
        cell = cls(
            int(state['cell_id']),
            rng,
            position=np.asarray(state['pos'], dtype=float),
            generation=int(state.get('generation', 0)),
            lineage=int(state.get('lineage', 0)),
            bootstrap=False,
        )
        # Restore inherited 0.1 state explicitly.
        for name in (
            'pos', 'vel', 'membrane', 'transporters', 'pools',
            'contact_trace', 'damage_trace', 'surface_flux',
        ):
            setattr(cell, name, np.asarray(state[name], dtype=float).copy())
        for name in (
            'age', 'radius', 'division_progress', 'septum_mass', 'division_axis',
            'lysis_timer', 'network_silence_timer', 'low_energy_timer',
            'last_catalysis', 'last_assembly', 'last_export', 'last_leak',
        ):
            setattr(cell, name, float(state.get(name, getattr(cell, name))))
        cell.last_uptake = np.asarray(state.get('last_uptake', [0.0, 0.0]), dtype=float).copy()
        cell.alive = bool(state.get('alive', True))
        cell.death_reason = str(state.get('death_reason', ''))
        cell.reaction_events = int(state.get('reaction_events', 0))
        cell.punctures_survived = int(state.get('punctures_survived', 0))
        cell.repair_reference = state.get('repair_reference')

        cell.alt_contact_trace = np.asarray(
            state.get('alt_contact_trace', np.zeros(MEMBRANE_SEGMENTS)), dtype=float
        ).copy()
        cell.last_uptake_alt = float(state.get('last_uptake_alt', 0.0))
        cell.genomes = [np.asarray(item, dtype=np.uint8).copy() for item in state.get('genomes', [])]
        cell.proteins = {int(key): float(value) for key, value in state.get('proteins', [])}
        template = state.get('replication_template')
        cell.replication_template = None if template is None else np.asarray(template, dtype=np.uint8).copy()
        cell.replication_copy = [int(value) for value in state.get('replication_copy', [])]
        cell.replication_fractional = float(state.get('replication_fractional', 0.0))
        cell.replication_cycles = int(state.get('replication_cycles', 0))
        cell.mutation_events = dict(state.get('mutation_events', cell.mutation_events))
        cell.genome_damage_events = int(state.get('genome_damage_events', 0))
        cell.information_silence_timer = float(state.get('information_silence_timer', 0.0))
        cell.novel_path_first_age = state.get('novel_path_first_age')
        cell.novel_path_flux = float(state.get('novel_path_flux', 0.0))
        cell.last_translation = float(state.get('last_translation', 0.0))
        cell.last_replication_symbols = int(state.get('last_replication_symbols', 0))
        cell.last_generic_flux = np.asarray(
            state.get('last_generic_flux', np.zeros(len(REACTION_NAMES))), dtype=float
        ).copy()
        cell._refresh_gene_cache()
        cell._sync_protein_pool()
        return cell


class GeneticWorld(base.SomaCellWorld):
    """Shared chemical world containing materially hereditary protocells."""

    def __init__(self, seed=101, initial_cells=1, config=None):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.config = config if config is not None else GeneticConfig()
        self.age = 0.0
        self.field = GeneticParticleField(self.rng, initial=True, alt_niche=self.config.alt_niche)
        self.cells = []
        self.next_cell_id = 0
        for index in range(int(initial_cells)):
            angle = 2.0 * math.pi * index / max(1, int(initial_cells))
            position = np.array([
                0.5 + 0.12 * math.cos(angle),
                0.5 + 0.12 * math.sin(angle),
            ]) % 1.0
            self.cells.append(GeneticProtoCell(
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
        self.initial_total_material = self.total_material()
        self.last_step_material_residual = 0.0

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
                + float(np.sum(cell.membrane))
                + float(np.sum(cell.transporters))
                + float(cell.septum_mass)
                + cell.genome_mass(),
            ),
        )
        for kind, amount in releases:
            if amount <= 1e-9:
                continue
            count = max(1, min(14, int(math.ceil(amount / 0.035))))
            positions = (cell.pos[None, :] + self.rng.normal(0.0, cell.radius * 0.45, (count, 2))) % 1.0
            amounts = np.full((count,), amount / count, dtype=float)
            self.field.add_many(kind, positions, amounts, count_as_injection=False)
        self.dissipated_energy += float(cell.pools[POOL_ATP])
        self.released_dead_material += total
        self.deaths += 1
        self.last_deaths.append((cell.cell_id, self.age, cell.death_reason))
        self.last_deaths = self.last_deaths[-8:]

    def _handle_divisions_and_deaths(self):
        new_cells = []
        survivors = []
        for cell in self.cells:
            if not cell.alive:
                self._release_dead_cell(cell)
                continue
            if cell.can_split() and len(self.cells) + len(new_cells) < MAX_CELLS:
                daughters = cell.split(self)
                if daughters is not None:
                    new_cells.extend(daughters)
                    self.births += 2
                    self.divisions += 1
                    self.last_births.append((cell.cell_id, tuple(d.cell_id for d in daughters), self.age))
                    self.last_births = self.last_births[-8:]
                    continue
            survivors.append(cell)
        self.cells = survivors + new_cells

    def step(self, dt):
        super(GeneticWorld, self).step(dt)
        for cell in self.living_cells():
            if cell.has_novel_path():
                self.novel_path_lineages.add(cell.lineage)
                if self.first_novel_path_age is None:
                    self.first_novel_path_age = float(self.age)

    def finite(self):
        if not super(GeneticWorld, self).finite():
            return False
        for cell in self.cells:
            if not finite_array(cell.alt_contact_trace) or not finite_array(cell.last_generic_flux):
                return False
            if any(not np.isfinite(value) for value in cell.proteins.values()):
                return False
            for genome in cell.genomes:
                if genome.dtype.kind not in 'ui' or len(genome) > MAX_GENOME_LENGTH:
                    return False
        return True

    def summary(self):
        alive = self.living_cells()
        if alive:
            mean_closure = float(np.mean([cell.closure() for cell in alive]))
            min_closure = float(np.min([cell.closure() for cell in alive]))
            mean_atp = float(np.mean([cell.pools[POOL_ATP] for cell in alive]))
            mean_catalyst = float(np.mean([cell.pools[POOL_CATALYST] for cell in alive]))
            mean_loop = float(np.mean([cell.reaction_loop_strength() for cell in alive]))
            mean_membrane = float(np.mean([np.sum(cell.membrane) for cell in alive]))
            max_generation = max(cell.generation for cell in alive)
            division_progress = float(np.max([cell.division_progress for cell in alive]))
        else:
            mean_closure = min_closure = mean_atp = mean_catalyst = 0.0
            mean_loop = mean_membrane = division_progress = 0.0
            max_generation = 0
        lengths = [len(genome) for cell in alive for genome in cell.genomes]
        hashes = [_sequence_hash(genome) for cell in alive for genome in cell.genomes]
        mutation_totals = {name: 0 for name in (
            'substitution', 'insertion', 'deletion', 'duplication', 'inversion', 'transposition'
        )}
        for cell in alive:
            for name in mutation_totals:
                mutation_totals[name] += int(cell.mutation_events.get(name, 0))
        fuel, mineral, waste, alt = self.field.totals_by_kind()
        summary = {
            'build': BUILD,
            'seed': self.seed,
            'age': float(self.age),
            'cells': len(alive),
            'births': int(self.births),
            'divisions': int(self.divisions),
            'deaths': int(self.deaths),
            'max_generation': int(max_generation),
            'mean_closure': mean_closure,
            'min_closure': min_closure,
            'mean_atp': mean_atp,
            'mean_catalyst': mean_catalyst,
            'mean_loop': mean_loop,
            'mean_membrane': mean_membrane,
            'division_progress': division_progress,
            'particles': int(len(self.field.amount)),
            'external_fuel': fuel,
            'external_mineral': mineral,
            'external_waste': waste,
            'external_alt': alt,
            'cell_material': self.total_cell_material(),
            'external_material': self.field.material_total(),
            'total_material': self.total_material(),
            'matter_residual': self.matter_ledger_residual(),
            'division_residual': float(self.division_residual),
            'dissipated_energy': float(self.dissipated_energy),
            'damage_events': int(self.damage_events),
            'manual_punctures': int(self.manual_punctures),
            'manual_injections': int(self.manual_injections),
            'mean_genome_length': float(np.mean(lengths)) if lengths else 0.0,
            'max_genome_length': int(max(lengths)) if lengths else 0,
            'genome_copies': int(len(lengths)),
            'genome_types': int(len(set(hashes))),
            'mean_gene_count': float(np.mean([cell.active_gene_count() for cell in alive])) if alive else 0.0,
            'mean_translator': float(np.mean([cell.role_activity(ROLE_TRANSLATOR) for cell in alive])) if alive else 0.0,
            'mean_replicase': float(np.mean([cell.role_activity(ROLE_REPLICASE) for cell in alive])) if alive else 0.0,
            'replication_cycles': int(sum(cell.replication_cycles for cell in alive)),
            'novel_path_cells': int(sum(cell.has_novel_path() for cell in alive)),
            'novel_path_lineages': int(len(self.novel_path_lineages)),
            'first_novel_path_age': self.first_novel_path_age,
            'external_protein_assistance': float(self.external_protein_assistance),
            'mutation_total': int(sum(mutation_totals.values())),
        }
        for name, value in mutation_totals.items():
            summary['mutation_' + name] = int(value)
        return summary

    def state_dict(self):
        return {
            'save_version': SAVE_VERSION,
            'build': BUILD,
            'seed': self.seed,
            'rng_state': self.rng.bit_generator.state,
            'config': self.config.state_dict(),
            'age': float(self.age),
            'field': self.field.state_dict(),
            'cells': [cell.state_dict() for cell in self.cells],
            'next_cell_id': int(self.next_cell_id),
            'births': int(self.births),
            'divisions': int(self.divisions),
            'deaths': int(self.deaths),
            'last_deaths': list(self.last_deaths),
            'last_births': list(self.last_births),
            'dissipated_energy': float(self.dissipated_energy),
            'division_parent_material': float(self.division_parent_material),
            'division_daughter_material': float(self.division_daughter_material),
            'division_shed_material': float(self.division_shed_material),
            'division_residual': float(self.division_residual),
            'released_dead_material': float(self.released_dead_material),
            'manual_injections': int(self.manual_injections),
            'manual_punctures': int(self.manual_punctures),
            'damage_events': int(self.damage_events),
            'external_protein_assistance': float(self.external_protein_assistance),
            'first_novel_path_age': self.first_novel_path_age,
            'novel_path_lineages': sorted(self.novel_path_lineages),
            'initial_total_material': float(self.initial_total_material),
            'last_step_material_residual': float(self.last_step_material_residual),
        }

    @classmethod
    def from_state(cls, state):
        world = cls(
            seed=int(state['seed']), initial_cells=0,
            config=GeneticConfig.from_state(state.get('config', {})),
        )
        world.rng.bit_generator.state = state['rng_state']
        world.age = float(state['age'])
        world.field = GeneticParticleField.from_state(world.rng, state['field'])
        world.cells = [GeneticProtoCell.from_state(world.rng, item) for item in state['cells']]
        world.rng.bit_generator.state = state['rng_state']
        for name in (
            'next_cell_id', 'births', 'divisions', 'deaths', 'manual_injections',
            'manual_punctures', 'damage_events',
        ):
            setattr(world, name, int(state.get(name, getattr(world, name))))
        for name in (
            'dissipated_energy', 'division_parent_material', 'division_daughter_material',
            'division_shed_material', 'division_residual', 'released_dead_material',
            'external_protein_assistance', 'initial_total_material', 'last_step_material_residual',
        ):
            setattr(world, name, float(state.get(name, getattr(world, name))))
        world.first_novel_path_age = state.get('first_novel_path_age')
        world.novel_path_lineages = set(state.get('novel_path_lineages', []))
        world.last_deaths = list(state.get('last_deaths', []))
        world.last_births = list(state.get('last_births', []))
        return world

    def save(self, path=SAVE_FILE):
        _atomic_pickle(path, self.state_dict())

    @classmethod
    def load(cls, path=SAVE_FILE):
        with open(path, 'rb') as handle:
            return cls.from_state(pickle.load(handle))

    def clone(self):
        return GeneticWorld.from_state(self.state_dict())


# ---------------------------------------------------------------------------
# Accelerated evolutionary assay using the exact same genome grammar/mutator
# ---------------------------------------------------------------------------


def genome_alt_niche_fitness(sequence):
    genes = parse_genes(sequence)
    role_strength = np.zeros(8, dtype=float)
    reaction_strength = np.zeros(len(REACTION_NAMES), dtype=float)
    for gene in genes:
        strength = gene['promoter'] * gene['efficiency']
        role_strength[gene['role']] += strength
        if gene['role'] == ROLE_GENERIC:
            reaction_strength[gene['reaction']] += strength
    essential = min(
        role_strength[ROLE_MEMBRANE],
        role_strength[ROLE_TRANSLATOR],
        role_strength[ROLE_REPLICASE],
        role_strength[ROLE_NUCLEOTIDE],
    )
    essential_score = essential / (0.55 + essential)
    base_energy = role_strength[ROLE_ENERGY] / (1.0 + role_strength[ROLE_ENERGY])
    alt_path = math.sqrt(
        max(0.0, reaction_strength[REACTION_ALT_TO_INTERMEDIATE])
        * max(0.0, reaction_strength[REACTION_INTERMEDIATE_TO_WASTE])
    )
    alt_score = alt_path / (0.75 + alt_path)
    length_cost = 0.00075 * max(0, len(sequence) - len(founding_genome()))
    broken_cost = 0.04 * max(0, 7 - len(genes))
    return float(max(1e-6, 0.10 + 0.40 * essential_score + 0.16 * base_energy + 1.55 * alt_score - length_cost - broken_cost))


def knockout_novel_path(sequence):
    sequence = np.asarray(sequence, dtype=np.uint8).copy()
    for gene in parse_genes(sequence):
        if gene['role'] == ROLE_GENERIC and gene.get('reaction') == REACTION_INTERMEDIATE_TO_WASTE:
            # Change the reaction parameter to the founding fuel edge while
            # leaving length and all other payload symbols unchanged.
            sequence[gene['start'] + 3] = REACTION_FUEL_TO_INTERMEDIATE
    return sequence


def run_evolution_assay(seed=101, generations=70, population=96, mutation=True,
                        variable_length=True, duplication=True):
    rng = np.random.default_rng(int(seed))
    config = GeneticConfig(
        mutation=mutation,
        variable_length=variable_length,
        gene_duplication=duplication,
        mutation_rate=0.013 if mutation else 0.0,
        structural_rate=0.11 if mutation else 0.0,
        alt_niche=True,
    )
    genomes = [founding_genome().copy() for _ in range(int(population))]
    history = []
    best = genomes[0].copy()
    best_fitness = genome_alt_niche_fitness(best)
    first_novel_generation = None
    for generation in range(int(generations) + 1):
        fitness = np.asarray([genome_alt_niche_fitness(genome) for genome in genomes], dtype=float)
        novel = np.asarray([sequence_has_novel_path(genome) for genome in genomes], dtype=bool)
        best_index = int(np.argmax(fitness))
        if float(fitness[best_index]) > best_fitness:
            best_fitness = float(fitness[best_index])
            best = genomes[best_index].copy()
        frequency = float(np.mean(novel))
        if first_novel_generation is None and np.any(novel):
            first_novel_generation = int(generation)
        history.append({
            'generation': int(generation),
            'mean_fitness': float(np.mean(fitness)),
            'max_fitness': float(np.max(fitness)),
            'novel_frequency': frequency,
            'mean_length': float(np.mean([len(genome) for genome in genomes])),
            'genome_types': int(len({_sequence_hash(genome) for genome in genomes})),
        })
        if generation >= generations:
            break
        # Tournament selection avoids a numerical fitness-normalisation trick.
        offspring = []
        while len(offspring) < population:
            contestants = rng.integers(0, population, 4)
            parent_index = int(contestants[np.argmax(fitness[contestants])])
            parent = genomes[parent_index]
            child, _, _ = mutate_sequence(parent, rng, config, nucleotide_budget=MAX_GENOME_LENGTH)
            offspring.append(child)
        genomes = offspring
    knockout = knockout_novel_path(best)
    return {
        'seed': int(seed),
        'generations': int(generations),
        'population': int(population),
        'mutation': bool(mutation),
        'variable_length': bool(variable_length),
        'duplication': bool(duplication),
        'first_novel_generation': first_novel_generation,
        'final_novel_frequency': float(history[-1]['novel_frequency']),
        'initial_mean_fitness': float(history[0]['mean_fitness']),
        'final_mean_fitness': float(history[-1]['mean_fitness']),
        'best_fitness': float(best_fitness),
        'knockout_fitness': float(genome_alt_niche_fitness(knockout)),
        'best_length': int(len(best)),
        'best_gene_count': int(len(parse_genes(best))),
        'best_hash': _sequence_hash(best),
        'history': history,
        'best_genome': best,
    }


# ---------------------------------------------------------------------------
# Logging / reports
# ---------------------------------------------------------------------------


LOG_FIELDS = (
    'session_id', 'reason', 'wall_time', 'age', 'cells', 'births', 'divisions',
    'deaths', 'max_generation', 'mean_closure', 'mean_atp', 'mean_loop',
    'genome_copies', 'genome_types', 'mean_genome_length', 'mean_gene_count',
    'replication_cycles', 'novel_path_cells', 'mutation_total', 'external_fuel',
    'external_mineral', 'external_alt', 'external_waste', 'total_material',
    'matter_residual', 'division_residual', 'fps', 'sim_rate', 'peak_mb',
)


class LongRunLogger(object):
    def __init__(self, world, path=LOG_FILE):
        self.path = path
        self.session_id = '{}-{}'.format(int(time.time()), world.seed)
        self.last_age = -1e9
        self.status = 'READY'

    def log(self, world, fps=0.0, sim_rate=0.0, reason='interval', force=False):
        if not force and world.age - self.last_age < LOG_INTERVAL:
            return
        summary = world.summary()
        row = {field: '' for field in LOG_FIELDS}
        row.update(summary)
        row.update({
            'session_id': self.session_id,
            'reason': reason,
            'wall_time': time.time(),
            'fps': float(fps),
            'sim_rate': float(sim_rate),
            'peak_mb': _memory_peak_mb_estimate(),
        })
        try:
            exists = os.path.exists(self.path) and os.path.getsize(self.path) > 0
            with open(self.path, 'a', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
                if not exists:
                    writer.writeheader()
                writer.writerow({key: row.get(key, '') for key in LOG_FIELDS})
            self.status = 'OK'
            self.last_age = world.age
        except Exception:
            self.status = 'ERR'


def generate_report(log_path=LOG_FILE, report_path=REPORT_FILE, session_path=SESSION_FILE):
    if not os.path.exists(log_path):
        return 'NO LOG'
    try:
        with open(log_path, 'r', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        sessions = {}
        for row in rows:
            sessions.setdefault(row['session_id'], []).append(row)
        lines = [BUILD + ' LONG-RUN REPORT', '=' * 72, '']
        session_rows = []
        for session_id, items in sessions.items():
            first, last = items[0], items[-1]
            def number(item, key, default=0.0):
                try:
                    return float(item.get(key, default))
                except Exception:
                    return float(default)
            summary = {
                'session_id': session_id,
                'start_age': number(first, 'age'),
                'end_age': number(last, 'age'),
                'final_cells': int(number(last, 'cells')),
                'births': int(number(last, 'births')),
                'deaths': int(number(last, 'deaths')),
                'max_generation': int(number(last, 'max_generation')),
                'final_genome_types': int(number(last, 'genome_types')),
                'final_novel_path_cells': int(number(last, 'novel_path_cells')),
                'max_mutations': int(max(number(item, 'mutation_total') for item in items)),
                'max_abs_matter_residual': max(abs(number(item, 'matter_residual')) for item in items),
            }
            session_rows.append(summary)
            lines.extend([
                'Session {}'.format(session_id),
                '  simulated: {:.1f}s'.format(summary['end_age'] - summary['start_age']),
                '  cells / births / deaths: {} / {} / {}'.format(
                    summary['final_cells'], summary['births'], summary['deaths']),
                '  max generation: G{}'.format(summary['max_generation']),
                '  genome types: {}'.format(summary['final_genome_types']),
                '  novel-path cells: {}'.format(summary['final_novel_path_cells']),
                '  max mutation count: {}'.format(summary['max_mutations']),
                '  max |matter residual|: {:.3e}'.format(summary['max_abs_matter_residual']),
                '',
            ])
        with open(report_path, 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(lines))
        if session_rows:
            with open(session_path, 'w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(session_rows[0].keys()))
                writer.writeheader()
                writer.writerows(session_rows)
        return 'OK'
    except Exception:
        return 'WRITE ERR'


def run_headless_trial(seed=101, seconds=180.0, initial_cells=1, config=None):
    world = GeneticWorld(seed=seed, initial_cells=initial_cells, config=config)
    steps = int(round(float(seconds) * SIM_HZ))
    dt = 1.0 / SIM_HZ
    for _ in range(steps):
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

    class SomaCellGeneticScene(Scene):
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
                    self.world = GeneticWorld.load(SAVE_FILE)
                    self.save_status = 'LOAD'
                else:
                    self.world = GeneticWorld(seed=101, initial_cells=1)
            except Exception:
                self.world = GeneticWorld(seed=101, initial_cells=1)
                self.save_status = 'RECOVER'
            self.logger = LongRunLogger(self.world)
            self.logger.log(self.world, reason='start', force=True)

        def _world_rect(self):
            width, height = float(self.size.w), float(self.size.h)
            return 34.0, 70.0, width - 68.0, height - 152.0

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
            atp = clamp(cell.pools[POOL_ATP] / 0.65, 0.0, 1.0)
            alt = clamp(cell.pools[POOL_ALT] / 0.30, 0.0, 1.0)
            fill(0.08, 0.24 + 0.30 * atp, 0.34 + 0.35 * alt, 0.30)
            ellipse(centre_x - screen_radius, centre_y - screen_radius,
                    screen_radius * 2.0, screen_radius * 2.0)

            # Genome molecules are material rings inside the cytoplasm.
            for genome_index, genome in enumerate(cell.genomes[:4]):
                ring_radius = screen_radius * (0.18 + 0.07 * genome_index)
                stroke(0.42, 1.0, 0.92, 0.82)
                stroke_weight(1.2)
                segments = max(8, min(28, len(genome) // 5))
                previous = None
                first = None
                for index in range(segments + 1):
                    angle = 2.0 * math.pi * index / segments + 0.3 * genome_index
                    point = (centre_x + math.cos(angle) * ring_radius,
                             centre_y + math.sin(angle) * ring_radius)
                    if first is None:
                        first = point
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
                stroke(0.34 + 0.48 * local, 0.78 + 0.20 * local, 0.95, 0.30 + 0.70 * local)
                stroke_weight(0.6 + 3.2 * local)
                line(x0, y0, x1, y1)
                if index % 2 == 0 and local > 0.25:
                    channel = int(np.argmax(cell.transporters[index]))
                    colours = (
                        (0.28, 0.95, 0.45), (1.00, 0.70, 0.24),
                        (1.00, 0.30, 0.55), (0.25, 0.78, 1.00),
                    )
                    colour = colours[channel]
                    fill(colour[0], colour[1], colour[2], 0.86)
                    ellipse(x0 - 1.3, y0 - 1.3, 2.6, 2.6)

            if cell.division_progress > 0.02:
                normal = unit_vector(cell.division_axis + math.pi * 0.5)
                length = screen_radius * 0.65 * cell.division_progress
                stroke(0.98, 0.92, 0.55, 0.72)
                stroke_weight(1.0 + 2.0 * cell.division_progress)
                line(centre_x - normal[0] * length, centre_y - normal[1] * length,
                     centre_x + normal[0] * length, centre_y + normal[1] * length)

            fill(0.91, 0.97, 1.0)
            text('#{} G{} DNA{} L{:.0f}{}'.format(
                cell.cell_id, cell.generation, len(cell.genomes), cell.mean_genome_length(),
                ' NEW' if cell.has_novel_path() else ''
            ), x=centre_x, y=centre_y - screen_radius - 10, font_size=8, alignment=5)

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
                self.save_status, self.logger.status, self.report_status, self.fps, self.sim_rate
            ), x=self.size.w - 72, y=self.size.h - 25, font_size=9, alignment=6)
            fill(0.84, 0.92, 0.97)
            text('age {:.1f}s cells {} divisions {} deaths {} G{} DNA {} types {}'.format(
                summary['age'], summary['cells'], summary['divisions'], summary['deaths'],
                summary['max_generation'], summary['genome_copies'], summary['genome_types']
            ), x=24, y=49, font_size=11, alignment=4)
            text('genes {:.1f} length {:.1f} repl {} mutations {} new-path {}'.format(
                summary['mean_gene_count'], summary['mean_genome_length'],
                summary['replication_cycles'], summary['mutation_total'],
                summary['novel_path_cells']
            ), x=24, y=31, font_size=10, alignment=4)
            text('seal {:.3f} ATP {:.3f} loop {:.3f} matter {:.3f} ledger {:+.2e} | tap puncture/inject | top pause | double reset'.format(
                summary['mean_closure'], summary['mean_atp'], summary['mean_loop'],
                summary['total_material'], summary['matter_residual']
            ), x=24, y=14, font_size=8, alignment=4)
            if summary['cells'] == 0:
                fill(1.0, 0.38, 0.32)
                text('MATERIAL–INFORMATION LOOP EXTINCT — double tap to reseed',
                     x=self.size.w * 0.5, y=self.size.h * 0.52, font_size=17, alignment=5)
            if self.paused:
                fill(1.0, 0.92, 0.45)
                text('PAUSED', x=self.size.w * 0.5, y=self.size.h - 26, font_size=13, alignment=5)

        def touch_began(self, touch):
            now = time.time()
            if now - self.last_touch_wall < 0.42:
                try:
                    if os.path.exists(SAVE_FILE):
                        os.remove(SAVE_FILE)
                except Exception:
                    pass
                self.world = GeneticWorld(seed=101, initial_cells=1)
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
                self.logger.log(self.world, self.fps, self.sim_rate,
                                reason='pause' if self.paused else 'resume', force=True)
                return
            position = self._unit_position(touch.location)
            if not self.world.puncture_nearest(position):
                self.world.inject_cloud(position)
                # Add a little alternative substrate to manual clouds.
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
        print(run_headless_trial(seed=101, seconds=180.0, initial_cells=1))
    else:
        run(SomaCellGeneticScene(), LANDSCAPE, show_fps=False)
