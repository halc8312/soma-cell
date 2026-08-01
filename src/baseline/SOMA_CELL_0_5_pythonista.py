# coding: utf-8
"""
SOMA-CELL 0.5 — Thanatochemical Ecology and Horizontal Heredity
死体化学・水平遺伝・共有化学生態系

Pythonista 3 / CPython + NumPy research prototype.

This module extends SOMA-CELL 0.4.  Dead cells are no longer flattened
immediately into anonymous particles.  Their membrane, protein, metabolites,
toxins and material genomes persist as spatial corpses.  Corpse matter can be
hydrolysed, eaten, detoxified or released into the geochemical field.  Genome
polymers enter an extracellular-DNA pool, diffuse, fragment and decay.  A
living cell can pay ATP to import such a fragment, digest it as nucleotide
material, reject it, or integrate it into a complete material genome through
gene-derived competence and recombination machinery.  A gene-derived mobile
element can also copy its own physical sequence into the extracellular pool,
creating a deliberately limited parasitic information cycle.

The model does not claim that death has an intrinsic meaning, that horizontal
gene transfer is open-ended, or that this simulation is biological life.
Matter classes, genetic decoding grammar, reaction rules and the integration
law remain human-designed.  The falsifiable question is narrower: can death
become a conserved ecological transformation of both matter and hereditary
information, and can an initially absent functional gene reach a recipient
through the shared chemical environment without free Python-level copying?
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
import SOMA_CELL_0_3_pythonista as g3
import SOMA_CELL_0_4_pythonista as s4

# 0.4's founding genome already occupies 368 symbols.  0.5 adds a finite
# ecology apparatus.  The ceiling is still explicit and small enough for an
# iPhone, but no longer prevents a viable 0.5 founder from existing.
g2.MAX_GENOME_LENGTH = max(int(g2.MAX_GENOME_LENGTH), 640)

BUILD = 'SOMA-CELL 0.5.0'
SAVE_VERSION = 5
BASE_DIR = os.path.dirname(__file__)
SAVE_FILE = os.path.join(BASE_DIR, 'soma_cell_0_5.pkl')
LOG_FILE = os.path.join(BASE_DIR, 'soma_cell_0_5_longrun.csv')
REPORT_FILE = os.path.join(BASE_DIR, 'soma_cell_0_5_report.txt')
SESSION_FILE = os.path.join(BASE_DIR, 'soma_cell_0_5_sessions.csv')

SIM_HZ = 18.0
AUTO_SAVE_INTERVAL = 30.0
LOG_INTERVAL = 10.0
MAX_CELLS = 16
MAX_CORPSES = 32
MAX_EDNA_FRAGMENTS = 256

# Re-export matter and geometry constants.
PARTICLE_FUEL = s4.PARTICLE_FUEL
PARTICLE_MINERAL = s4.PARTICLE_MINERAL
PARTICLE_WASTE = s4.PARTICLE_WASTE
PARTICLE_ALT = s4.PARTICLE_ALT
PARTICLE_NAMES = s4.PARTICLE_NAMES

POOL_FUEL = s4.POOL_FUEL
POOL_MINERAL = s4.POOL_MINERAL
POOL_ATP = s4.POOL_ATP
POOL_MEM_PRECURSOR = s4.POOL_MEM_PRECURSOR
POOL_CATALYST = s4.POOL_CATALYST
POOL_TRANSPORTER_PRECURSOR = s4.POOL_TRANSPORTER_PRECURSOR
POOL_WASTE = s4.POOL_WASTE
POOL_ALT = s4.POOL_ALT
POOL_INTERMEDIATE = s4.POOL_INTERMEDIATE
POOL_NUCLEOTIDE = s4.POOL_NUCLEOTIDE
POOL_DAMAGED_PROTEIN = s4.POOL_DAMAGED_PROTEIN
POOL_AGGREGATE = s4.POOL_AGGREGATE
POOL_REACTIVE = s4.POOL_REACTIVE
POOL_COUNT = s4.POOL_COUNT

MEMBRANE_SEGMENTS = s4.MEMBRANE_SEGMENTS
BASE_RADIUS = s4.BASE_RADIUS
MONOMER_MASS = s4.MONOMER_MASS
ROLE_REGULATOR = s4.ROLE_REGULATOR
ROLE_GENERIC = g2.ROLE_GENERIC

# Localisation 3 was explicitly reserved in 0.4.  It now denotes proteins that
# interact with corpses, extracellular genomes and mobile information.
LOC_ECOLOGY = s4.LOC_RESERVED

ECO_COMPETENCE = 0
ECO_RECOMBINASE = 1
ECO_NUCLEASE = 2
ECO_RESTRICTION = 3
ECO_NECROPHAGE = 4
ECO_DETOX = 5
ECO_VESICLE = 6
ECO_MOBILE = 7
ECOLOGY_NAMES = (
    'competence', 'recombinase', 'nuclease', 'restriction',
    'necrophage', 'detox', 'vesicle', 'mobile-element',
)
ECOLOGY_COUNT = len(ECOLOGY_NAMES)

# Corpse matter compartments.  They are explicit material, not scores.
CORPSE_LABILE = 0
CORPSE_MINERAL = 1
CORPSE_PROTEIN = 2
CORPSE_STRUCTURAL = 3
CORPSE_TOXIC = 4
CORPSE_POOL_COUNT = 5
CORPSE_POOL_NAMES = ('labile', 'mineral', 'protein', 'structural', 'toxic')

clamp = s4.clamp
wrapped_delta = s4.wrapped_delta
torus_distance = s4.torus_distance
finite_array = s4.finite_array
_atomic_pickle = s4._atomic_pickle
_memory_peak_mb_estimate = s4._memory_peak_mb_estimate


def _sequence_hash(sequence):
    return hashlib.sha1(np.asarray(sequence, dtype=np.uint8).tobytes()).hexdigest()[:12]


def make_ecology_gene(kind, promoter=5, efficiency=5, fidelity=5,
                       activity=5, selectivity=5, cost=4, threshold=3,
                       regulator=0):
    """Encode one ecological protein in the existing finite gene grammar."""
    return g2.make_gene(
        ROLE_REGULATOR,
        parameter=int(kind),
        regulator=int(regulator),
        promoter=int(promoter),
        efficiency=int(efficiency),
        fidelity=int(fidelity),
        localisation=LOC_ECOLOGY,
        spare=(activity, selectivity, cost, threshold),
    )


def ecology_gene_counts(sequence):
    counts = np.zeros(ECOLOGY_COUNT, dtype=int)
    for spec in g2.parse_genes(sequence):
        if spec['role'] == ROLE_REGULATOR and spec['localisation'] == LOC_ECOLOGY:
            counts[int(spec['parameter']) % ECOLOGY_COUNT] += 1
    return counts


def _remove_reaction_gene(sequence, reaction):
    """Remove complete genes for one generic reaction, returning their mass as length."""
    sequence = np.asarray(sequence, dtype=np.uint8).reshape(-1)
    remove = np.zeros(len(sequence), dtype=bool)
    for spec in g2.parse_genes(sequence):
        if spec['role'] == ROLE_GENERIC and int(spec.get('reaction', -1)) == int(reaction):
            start = int(spec['start'])
            remove[start:start + g2.GENE_SPAN] = True
    return sequence[~remove].copy()


def founding_genome_05(complete_alt=True):
    """0.4 chemistry plus gene-derived corpse and DNA ecology.

    The founder deliberately lacks a mobile-element gene.  Mobile elements are
    therefore foreign information in the controlled infection assay rather
    than a built-in universal trait.
    """
    sequence = s4.founding_genome_04()
    if not complete_alt:
        sequence = _remove_reaction_gene(sequence, g2.REACTION_INTERMEDIATE_TO_WASTE)
    ecology = [
        make_ecology_gene(ECO_COMPETENCE, activity=5, selectivity=5, cost=4),
        make_ecology_gene(ECO_RECOMBINASE, activity=4, selectivity=6, cost=5),
        make_ecology_gene(ECO_NUCLEASE, activity=4, selectivity=4, cost=3),
        make_ecology_gene(ECO_RESTRICTION, activity=3, selectivity=6, cost=4),
        make_ecology_gene(ECO_NECROPHAGE, activity=5, selectivity=4, cost=4),
        make_ecology_gene(ECO_DETOX, activity=4, selectivity=5, cost=4),
        make_ecology_gene(ECO_VESICLE, activity=2, selectivity=4, cost=6),
    ]
    result = np.concatenate([sequence] + ecology).astype(np.uint8)
    if len(result) > g2.MAX_GENOME_LENGTH:
        raise ValueError('0.5 founding genome exceeds material genome limit')
    return result


def mobile_element_sequence(promoter=7, efficiency=6):
    """A one-gene selfish element used only when it physically enters a world."""
    return make_ecology_gene(
        ECO_MOBILE, promoter=promoter, efficiency=efficiency, fidelity=4,
        activity=7, selectivity=2, cost=2, threshold=1,
    )


def _ecology_parameters(spec):
    payload = spec['payload']
    activity = 0.20 + 1.80 * (float(payload[8]) / 7.0)
    selectivity = 0.20 + 1.80 * (float(payload[9]) / 7.0)
    cost = 0.45 + 1.15 * (float(payload[10]) / 7.0)
    threshold = 0.02 + 0.30 * (float(payload[11]) / 7.0)
    return activity, selectivity, cost, threshold


def _homology_score(sequence_a, sequence_b, k=5):
    """Small deterministic k-mer overlap score in [0, 1]."""
    a = np.asarray(sequence_a, dtype=np.uint8).reshape(-1)
    b = np.asarray(sequence_b, dtype=np.uint8).reshape(-1)
    if len(a) < k or len(b) < k:
        return 0.0
    set_a = set(tuple(int(v) for v in a[i:i + k]) for i in range(len(a) - k + 1))
    set_b = set(tuple(int(v) for v in b[i:i + k]) for i in range(len(b) - k + 1))
    if not set_a or not set_b:
        return 0.0
    return float(len(set_a & set_b) / max(1, min(len(set_a), len(set_b))))


class EcologyConfig(s4.SensorimotorConfig):
    """Ablations for corpse chemistry, extracellular DNA and HGT."""

    def __init__(
        self,
        corpse_chemistry=True,
        corpse_nutrition=True,
        corpse_toxicity=True,
        extracellular_dna=True,
        competence=True,
        recombination=True,
        restriction=True,
        dna_digestion=True,
        mobile_elements=True,
        vesicle_export=True,
        immediate_dead_recycling=False,
        seed_mobile_fragments=False,
        dna_decay_scale=1.0,
        corpse_decay_scale=1.0,
        hgt_rate_scale=1.0,
        necrophagy_rate_scale=1.0,
        mobile_rate_scale=1.0,
        **kwargs
    ):
        super(EcologyConfig, self).__init__(**kwargs)
        self.corpse_chemistry = bool(corpse_chemistry)
        self.corpse_nutrition = bool(corpse_nutrition)
        self.corpse_toxicity = bool(corpse_toxicity)
        self.extracellular_dna = bool(extracellular_dna)
        self.competence = bool(competence)
        self.recombination = bool(recombination)
        self.restriction = bool(restriction)
        self.dna_digestion = bool(dna_digestion)
        self.mobile_elements = bool(mobile_elements)
        self.vesicle_export = bool(vesicle_export)
        self.immediate_dead_recycling = bool(immediate_dead_recycling)
        self.seed_mobile_fragments = bool(seed_mobile_fragments)
        self.dna_decay_scale = float(dna_decay_scale)
        self.corpse_decay_scale = float(corpse_decay_scale)
        self.hgt_rate_scale = float(hgt_rate_scale)
        self.necrophagy_rate_scale = float(necrophagy_rate_scale)
        self.mobile_rate_scale = float(mobile_rate_scale)

    @classmethod
    def from_state(cls, state):
        return cls(**dict(state))


class DNAFragment(object):
    """One material extracellular polymer with provenance and decay state."""

    def __init__(self, sequence, position, origin_lineage=-1, origin_cell=-1,
                 origin_hash='', lesion=0.0, mobile=False, age=0.0,
                 decay_progress=0.0):
        self.sequence = np.asarray(sequence, dtype=np.uint8).reshape(-1).copy()
        self.pos = np.asarray(position, dtype=float).reshape(2).copy() % 1.0
        self.origin_lineage = int(origin_lineage)
        self.origin_cell = int(origin_cell)
        self.origin_hash = str(origin_hash or _sequence_hash(self.sequence))
        self.lesion = float(max(0.0, lesion))
        self.mobile = bool(mobile)
        self.age = float(age)
        self.decay_progress = float(decay_progress)

    def material_mass(self):
        return float(len(self.sequence) * MONOMER_MASS)

    def state_dict(self):
        return {
            'sequence': self.sequence.copy(), 'pos': self.pos.copy(),
            'origin_lineage': self.origin_lineage,
            'origin_cell': self.origin_cell,
            'origin_hash': self.origin_hash,
            'lesion': self.lesion, 'mobile': self.mobile,
            'age': self.age, 'decay_progress': self.decay_progress,
        }

    @classmethod
    def from_state(cls, state):
        payload = dict(state)
        if 'position' not in payload and 'pos' in payload:
            payload['position'] = payload.pop('pos')
        return cls(**payload)


class ExtracellularDNAField(object):
    """Diffusing, degradable material genome fragments."""

    def __init__(self, rng):
        self.rng = rng
        self.fragments = []
        self.released_fragments = 0
        self.released_symbols = 0
        self.decayed_symbols = 0
        self.uptaken_fragments = 0
        self.integrated_fragments = 0
        self.digested_fragments = 0

    def material_total(self):
        return float(sum(fragment.material_mass() for fragment in self.fragments))

    def add_fragment(self, sequence, position, origin_lineage=-1, origin_cell=-1,
                     origin_hash='', lesion=0.0, mobile=False):
        sequence = np.asarray(sequence, dtype=np.uint8).reshape(-1)
        if len(sequence) == 0:
            return None
        fragment = DNAFragment(
            sequence, position, origin_lineage=origin_lineage,
            origin_cell=origin_cell, origin_hash=origin_hash,
            lesion=lesion, mobile=mobile,
        )
        self.fragments.append(fragment)
        self.released_fragments += 1
        self.released_symbols += len(sequence)
        return fragment

    def add_genome_as_fragments(self, sequence, position, origin_lineage=-1,
                                origin_cell=-1, lesion=0.0, intact_probability=0.18):
        """Cleave without overlap, preferentially at gene boundaries."""
        sequence = np.asarray(sequence, dtype=np.uint8).reshape(-1)
        if len(sequence) == 0:
            return 0
        origin_hash = _sequence_hash(sequence)
        if self.rng.random() < intact_probability:
            self.add_fragment(
                sequence, position, origin_lineage, origin_cell,
                origin_hash, lesion, mobile=False,
            )
            return 1
        cuts = [0]
        cursor = 0
        # Gene-sized pieces preserve some intact genes but never duplicate a
        # symbol.  Non-gene tails are included in the last piece.
        while cursor < len(sequence):
            genes = int(self.rng.integers(1, 4))
            step = genes * g2.GENE_SPAN
            cursor = min(len(sequence), cursor + step)
            cuts.append(cursor)
        count = 0
        for start, end in zip(cuts[:-1], cuts[1:]):
            if end <= start:
                continue
            jitter = self.rng.normal(0.0, 0.012, 2)
            self.add_fragment(
                sequence[start:end], (np.asarray(position) + jitter) % 1.0,
                origin_lineage, origin_cell, origin_hash, lesion,
                mobile=False,
            )
            count += 1
        return count

    def _return_symbols_to_environment(self, world, position, count):
        if count <= 0:
            return
        amount = float(count) * MONOMER_MASS
        # Degraded nucleic polymer becomes mineral-rich monomer debris.
        world.field.add_particle(
            PARTICLE_MINERAL, position, amount, count_as_injection=False
        )
        self.decayed_symbols += int(count)

    def step(self, world, dt):
        if not self.fragments:
            return
        kept = []
        for fragment in self.fragments:
            fragment.age += dt
            diffusion = 0.0016 / (1.0 + 0.025 * len(fragment.sequence))
            fragment.pos = (
                fragment.pos
                + world.rng.normal(0.0, math.sqrt(2.0 * diffusion * dt), 2)
            ) % 1.0
            fragment.lesion += dt * (
                0.0018 + 0.0035 * world.current_stress
            ) * world.config.dna_decay_scale
            rate = (
                0.010 + 0.010 * fragment.lesion
                + 0.00012 * max(0.0, fragment.age - 70.0)
            ) * world.config.dna_decay_scale
            fragment.decay_progress += rate * dt * max(1.0, len(fragment.sequence) / 16.0)
            remove = int(fragment.decay_progress)
            fragment.decay_progress -= remove
            if remove > 0 and len(fragment.sequence) > 0:
                remove = min(remove, len(fragment.sequence))
                # End erosion keeps the remaining sequence a real subset.
                if world.rng.random() < 0.5:
                    fragment.sequence = fragment.sequence[remove:].copy()
                else:
                    fragment.sequence = fragment.sequence[:-remove].copy()
                self._return_symbols_to_environment(world, fragment.pos, remove)
            if len(fragment.sequence) >= 4:
                kept.append(fragment)
            elif len(fragment.sequence) > 0:
                self._return_symbols_to_environment(
                    world, fragment.pos, len(fragment.sequence)
                )
        self.fragments = kept
        self._enforce_cap(world)

    def _enforce_cap(self, world):
        while len(self.fragments) > MAX_EDNA_FRAGMENTS:
            index = max(
                range(len(self.fragments)),
                key=lambda i: self.fragments[i].age + 0.03 * len(self.fragments[i].sequence),
            )
            fragment = self.fragments.pop(index)
            self._return_symbols_to_environment(
                world, fragment.pos, len(fragment.sequence)
            )

    def state_dict(self):
        return {
            'fragments': [fragment.state_dict() for fragment in self.fragments],
            'released_fragments': int(self.released_fragments),
            'released_symbols': int(self.released_symbols),
            'decayed_symbols': int(self.decayed_symbols),
            'uptaken_fragments': int(self.uptaken_fragments),
            'integrated_fragments': int(self.integrated_fragments),
            'digested_fragments': int(self.digested_fragments),
        }

    @classmethod
    def from_state(cls, rng, state):
        field = cls(rng)
        field.fragments = [DNAFragment.from_state(item) for item in state.get('fragments', [])]
        for name in (
            'released_fragments', 'released_symbols', 'decayed_symbols',
            'uptaken_fragments', 'integrated_fragments', 'digested_fragments',
        ):
            setattr(field, name, int(state.get(name, 0)))
        return field


class Corpse(object):
    """A spatial dead body whose matter and genomes decay along separate paths."""

    def __init__(self, corpse_id, position, radius, pools, genomes,
                 genome_lesions=None, origin_cell=-1, origin_lineage=-1,
                 origin_generation=0, death_reason='', age=0.0,
                 release_buffer=None, dna_release_progress=0.0,
                 initial_structural=None):
        self.corpse_id = int(corpse_id)
        self.pos = np.asarray(position, dtype=float).reshape(2).copy() % 1.0
        self.radius = float(max(0.012, radius))
        self.pools = np.asarray(pools, dtype=float).reshape(CORPSE_POOL_COUNT).copy()
        self.genomes = [np.asarray(seq, dtype=np.uint8).reshape(-1).copy() for seq in genomes]
        self.genome_lesions = list(
            float(v) for v in (
                genome_lesions if genome_lesions is not None
                else [0.0] * len(self.genomes)
            )
        )
        while len(self.genome_lesions) < len(self.genomes):
            self.genome_lesions.append(0.0)
        self.origin_cell = int(origin_cell)
        self.origin_lineage = int(origin_lineage)
        self.origin_generation = int(origin_generation)
        self.death_reason = str(death_reason)
        self.age = float(age)
        self.release_buffer = (
            np.zeros(4, dtype=float) if release_buffer is None
            else np.asarray(release_buffer, dtype=float).reshape(4).copy()
        )
        self.dna_release_progress = float(dna_release_progress)
        self.initial_structural = float(
            max(1e-9, self.pools[CORPSE_STRUCTURAL])
            if initial_structural is None else initial_structural
        )
        self.initial_mass = self.material_mass()

    @classmethod
    def from_cell(cls, corpse_id, cell):
        pools = np.zeros(CORPSE_POOL_COUNT, dtype=float)
        pools[CORPSE_LABILE] = (
            cell.pools[POOL_FUEL]
            + cell.pools[POOL_ALT]
            + cell.pools[POOL_INTERMEDIATE]
        )
        pools[CORPSE_MINERAL] = (
            cell.pools[POOL_MINERAL] + cell.pools[POOL_NUCLEOTIDE]
        )
        pools[CORPSE_PROTEIN] = (
            cell.pools[POOL_CATALYST]
            + cell.pools[POOL_TRANSPORTER_PRECURSOR]
            + cell.pools[POOL_DAMAGED_PROTEIN]
            + cell.pools[POOL_AGGREGATE]
        )
        pools[CORPSE_STRUCTURAL] = (
            cell.pools[POOL_MEM_PRECURSOR]
            + float(np.sum(cell.membrane))
            + float(np.sum(cell.transporters))
            + float(cell.septum_mass)
        )
        pools[CORPSE_TOXIC] = cell.pools[POOL_WASTE] + cell.pools[POOL_REACTIVE]
        genomes = [seq.copy() for seq in cell.genomes]
        lesions = list(float(v) for v in cell.genome_lesions)
        if len(cell.replication_copy) > 0:
            genomes.append(np.asarray(cell.replication_copy, dtype=np.uint8).copy())
            lesions.append(float(getattr(cell, 'replication_template_lesion', 0.0)))
        corpse = cls(
            corpse_id, cell.pos, cell.radius, pools, genomes,
            genome_lesions=lesions, origin_cell=cell.cell_id,
            origin_lineage=cell.lineage, origin_generation=cell.generation,
            death_reason=cell.death_reason,
        )
        expected = cell.material_mass()
        if abs(corpse.material_mass() - expected) > 5e-10:
            raise AssertionError('corpse material classification mismatch')
        return corpse

    def genome_mass(self):
        return float(sum(len(seq) for seq in self.genomes) * MONOMER_MASS)

    def material_mass(self):
        return float(np.sum(self.pools) + np.sum(self.release_buffer) + self.genome_mass())

    def exposure(self):
        structural_fraction = self.pools[CORPSE_STRUCTURAL] / self.initial_structural
        return float(clamp(
            0.05 + 0.55 * (1.0 - structural_fraction) + 0.012 * self.age,
            0.0, 1.0,
        ))

    def _hydrolyse(self, source, amount, fuel_fraction, mineral_fraction):
        amount = min(max(0.0, amount), self.pools[source])
        if amount <= 0.0:
            return 0.0
        self.pools[source] -= amount
        self.release_buffer[PARTICLE_FUEL] += amount * fuel_fraction
        self.release_buffer[PARTICLE_MINERAL] += amount * mineral_fraction
        remainder = amount * max(0.0, 1.0 - fuel_fraction - mineral_fraction)
        self.release_buffer[PARTICLE_WASTE] += remainder
        return amount

    def _flush_buffers(self, world, force=False):
        for kind in range(4):
            amount = float(self.release_buffer[kind])
            threshold = 0.010 if kind != PARTICLE_WASTE else 0.007
            if amount < threshold and not force:
                continue
            if amount <= 0.0:
                continue
            count = max(1, min(5, int(math.ceil(amount / 0.026))))
            positions = (
                self.pos[None, :]
                + world.rng.normal(0.0, max(0.008, self.radius * 0.40), (count, 2))
            ) % 1.0
            world.field.add_many(
                kind, positions, np.full(count, amount / count),
                count_as_injection=False,
            )
            self.release_buffer[kind] = 0.0

    def step(self, world, dt):
        self.age += dt
        if not world.config.corpse_chemistry:
            return
        exposure = self.exposure()
        scale = world.config.corpse_decay_scale
        self._hydrolyse(
            CORPSE_LABILE,
            self.pools[CORPSE_LABILE] * (0.020 + 0.040 * exposure) * dt * scale,
            0.88, 0.12,
        )
        self._hydrolyse(
            CORPSE_MINERAL,
            self.pools[CORPSE_MINERAL] * (0.014 + 0.022 * exposure) * dt * scale,
            0.0, 1.0,
        )
        self._hydrolyse(
            CORPSE_PROTEIN,
            self.pools[CORPSE_PROTEIN] * (0.006 + 0.026 * exposure) * dt * scale,
            0.58, 0.32,
        )
        self._hydrolyse(
            CORPSE_STRUCTURAL,
            self.pools[CORPSE_STRUCTURAL] * (0.003 + 0.014 * exposure) * dt * scale,
            0.24, 0.54,
        )
        toxic = min(
            self.pools[CORPSE_TOXIC],
            self.pools[CORPSE_TOXIC] * (0.015 + 0.045 * exposure) * dt * scale,
        )
        self.pools[CORPSE_TOXIC] -= toxic
        self.release_buffer[PARTICLE_WASTE] += toxic

        if self.genomes:
            self.dna_release_progress += dt * scale * (0.035 + 0.25 * exposure)
            while self.dna_release_progress >= 1.0 and self.genomes:
                self.dna_release_progress -= 1.0
                index = int(world.rng.integers(0, len(self.genomes)))
                sequence = self.genomes.pop(index)
                lesion = self.genome_lesions.pop(index)
                if world.config.extracellular_dna:
                    count = world.edna.add_genome_as_fragments(
                        sequence, self.pos,
                        origin_lineage=self.origin_lineage,
                        origin_cell=self.origin_cell,
                        lesion=lesion,
                    )
                    world.corpse_genomes_released += 1
                    world.corpse_fragments_released += count
                else:
                    # eDNA ablation removes information persistence, not matter.
                    self.release_buffer[PARTICLE_MINERAL] += len(sequence) * MONOMER_MASS
                    world.corpse_genomes_recycled += 1

        self._flush_buffers(world, force=self.material_mass() < 0.016)

    def empty(self):
        return bool(self.material_mass() < 1e-8)

    def state_dict(self):
        return {
            'corpse_id': self.corpse_id, 'position': self.pos.copy(),
            'radius': self.radius, 'pools': self.pools.copy(),
            'genomes': [seq.copy() for seq in self.genomes],
            'genome_lesions': list(self.genome_lesions),
            'origin_cell': self.origin_cell, 'origin_lineage': self.origin_lineage,
            'origin_generation': self.origin_generation,
            'death_reason': self.death_reason, 'age': self.age,
            'release_buffer': self.release_buffer.copy(),
            'dna_release_progress': self.dna_release_progress,
            'initial_structural': self.initial_structural,
        }

    @classmethod
    def from_state(cls, state):
        return cls(**dict(state))


class EcologicalProtoCell(s4.SensorimotorProtoCell):
    """A 0.4 cell with gene-derived corpse use and horizontal heredity."""

    def __init__(self, cell_id, rng, position=None, generation=0, lineage=0,
                 bootstrap=True, complete_alt=True):
        super(EcologicalProtoCell, self).__init__(
            cell_id, rng, position=position, generation=generation,
            lineage=lineage, bootstrap=bootstrap,
        )
        self._init_ecology_state()
        if bootstrap:
            self.genomes = [founding_genome_05(complete_alt=complete_alt)]
            self.genome_lesions = [0.0]
            self.replication_template = None
            self.replication_copy = []
            self.replication_fractional = 0.0
            self._refresh_gene_cache()
            # The 0.4 bootstrap proteins were made for its original genome.
            # When 0.5 deliberately replaces that genome (including the
            # incomplete-path founder used in HGT assays), proteins whose genes
            # are absent must not survive as free hidden information.
            self.proteins = {
                fingerprint: amount
                for fingerprint, amount in self.proteins.items()
                if fingerprint in self.gene_specs
            }
            self._sync_protein_pool()
            initial = {
                ECO_COMPETENCE: 0.016, ECO_RECOMBINASE: 0.013,
                ECO_NUCLEASE: 0.012, ECO_RESTRICTION: 0.010,
                ECO_NECROPHAGE: 0.016, ECO_DETOX: 0.013,
                ECO_VESICLE: 0.006, ECO_MOBILE: 0.0,
            }
            for fingerprint, spec in self.ecology_specs():
                kind = int(spec['parameter']) % ECOLOGY_COUNT
                self.proteins[fingerprint] = self.proteins.get(fingerprint, 0.0) + initial[kind]
            # Ecological machinery is a material tax; modest starting reserves
            # keep the founder viable without granting free ongoing support.
            self.pools[POOL_FUEL] = max(self.pools[POOL_FUEL], 0.47)
            self.pools[POOL_MINERAL] = max(self.pools[POOL_MINERAL], 0.45)
            self.pools[POOL_ATP] = max(self.pools[POOL_ATP], 0.56)
            self.pools[POOL_NUCLEOTIDE] = max(self.pools[POOL_NUCLEOTIDE], 0.34)
            self._sync_protein_pool()
            self._sync_damage_pool()
            self.previous_autopoietic_margin = self.autopoietic_margin()
            self.last_margin = self.previous_autopoietic_margin

    def _init_ecology_state(self):
        self.last_edna_signal = 0.0
        self.last_corpse_signal = 0.0
        self.last_necrotoxin_signal = 0.0
        self.last_dna_uptake_mass = 0.0
        self.last_necrophagy_mass = 0.0
        self.last_hgt_outcome = 'none'
        self.hgt_attempts = 0
        self.hgt_integrations = 0
        self.hgt_rejections = 0
        self.hgt_digestions = 0
        self.hgt_symbols = 0
        self.hgt_novel_path_events = 0
        self.foreign_lineages = {}
        self.necrophagy_mass = 0.0
        self.necrotoxin_uptake = 0.0
        self.ecology_atp = 0.0
        self.mobile_exports = 0
        self.mobile_symbols_exported = 0
        self.last_mobile_export_age = -1e9
        self.hgt_marker_hashes = set()

    def ecology_specs(self):
        return [
            (fingerprint, spec)
            for fingerprint, spec in self.gene_specs.items()
            if spec['role'] == ROLE_REGULATOR and spec['localisation'] == LOC_ECOLOGY
        ]

    def ecology_activity(self, kind):
        total = 0.0
        for fingerprint, spec in self.ecology_specs():
            if int(spec['parameter']) % ECOLOGY_COUNT != int(kind):
                continue
            amount = max(0.0, float(self.proteins.get(fingerprint, 0.0)))
            activity, selectivity, cost, threshold = _ecology_parameters(spec)
            total += (
                amount * spec['promoter'] * spec['efficiency']
                * activity / 0.015
            )
        return float(total * self.proteostasis_factor() * self.genome_function_factor())

    def _protein_need(self, spec):
        if spec['role'] == ROLE_REGULATOR and spec['localisation'] == LOC_ECOLOGY:
            kind = int(spec['parameter']) % ECOLOGY_COUNT
            if kind == ECO_COMPETENCE:
                need = 0.003 + 1.4 * self.last_edna_signal
            elif kind == ECO_RECOMBINASE:
                need = 0.002 + 0.9 * self.last_edna_signal + 0.3 * self.mean_genome_lesion()
            elif kind == ECO_NUCLEASE:
                need = 0.002 + 0.8 * self.last_edna_signal
            elif kind == ECO_RESTRICTION:
                need = 0.002 + 0.6 * self.last_edna_signal + 0.25 * self.current_stress
            elif kind == ECO_NECROPHAGE:
                need = 0.003 + 1.2 * self.last_corpse_signal
            elif kind == ECO_DETOX:
                need = 0.003 + 1.5 * self.last_necrotoxin_signal + 0.7 * self.reactive_concentration()
            elif kind == ECO_VESICLE:
                need = 0.0015 + 0.2 * self.last_edna_signal
            elif kind == ECO_MOBILE:
                # A selfish element must be capable of expressing itself after
                # horizontal acquisition; founders pay no cost because the
                # gene is absent.  Its expression still competes for ordinary
                # translation capacity and material.
                need = 0.012 + 0.55 * self.last_edna_signal
            else:
                need = 0.001 + 0.4 * self.last_edna_signal
            return clamp(need, 0.001, 1.8)
        return super(EcologicalProtoCell, self)._protein_need(spec)

    def _digest_fragment(self, fragment):
        mass = fragment.material_mass()
        self.pools[POOL_NUCLEOTIDE] += mass
        self.hgt_digestions += 1
        self.last_hgt_outcome = 'digested'
        return mass

    def _dispose_failed_fragment(self, fragment, world, outcome='not-integrated'):
        self.hgt_rejections += 1
        if world.config.dna_digestion:
            mass = self._digest_fragment(fragment)
            world.edna.digested_fragments += 1
            world.hgt_digestions += 1
            self.last_hgt_outcome = outcome + '+digested'
            return mass
        fragment.pos = (self.pos + world.rng.normal(0.0, 0.010, 2)) % 1.0
        world.edna.fragments.append(fragment)
        self.last_hgt_outcome = outcome + '+released'
        return 0.0

    def _best_target_genome(self, fragment):
        if not self.genomes:
            return None, 0.0
        scores = [_homology_score(fragment.sequence, genome) for genome in self.genomes]
        index = int(np.argmax(scores))
        return index, float(scores[index])

    def integrate_fragment(self, fragment, world, force=False):
        """Integrate physical fragment symbols; unused symbols become monomers."""
        if not self.genomes or len(fragment.sequence) < 4:
            self._dispose_failed_fragment(fragment, world, 'no-target')
            return False
        target_index, homology = self._best_target_genome(fragment)
        recombinase = self.ecology_activity(ECO_RECOMBINASE)
        restriction = self.ecology_activity(ECO_RESTRICTION) if world.config.restriction else 0.0
        foreignness = 1.0 - homology
        mobile_bonus = 0.55 if fragment.mobile else 0.0
        probability = (
            (recombinase / (0.65 + recombinase))
            * (0.10 + 0.90 * homology + mobile_bonus)
            * math.exp(-0.55 * restriction * foreignness)
        )
        probability = clamp(probability, 0.0, 0.97)
        if not force and world.rng.random() >= probability:
            outcome = 'restricted' if restriction > 0.2 else 'not-integrated'
            self._dispose_failed_fragment(fragment, world, outcome)
            return False

        genome = self.genomes[target_index]
        room = max(0, g2.MAX_GENOME_LENGTH - len(genome))
        used = min(room, len(fragment.sequence))
        if used < 4:
            self._dispose_failed_fragment(fragment, world, 'no-genome-room')
            return False
        cost = 0.00058 * used * (1.0 + 0.20 * foreignness)
        available = max(0.0, self.pools[POOL_ATP] - 0.020)
        if available < cost and not force:
            self._dispose_failed_fragment(fragment, world, 'ATP-limited')
            return False
        paid = min(cost, available) if not force else min(cost, self.pools[POOL_ATP])
        self.pools[POOL_ATP] -= paid
        world.dissipated_energy += paid
        self.ecology_atp += paid

        before_novel = any(g2.sequence_has_novel_path(genome) for genome in self.genomes)
        # Homologous DNA is placed near the best matching 5-mer; mobile DNA
        # inserts at a random position.  No recipient material is overwritten.
        position = int(world.rng.integers(0, len(genome) + 1))
        if homology > 0.0 and not fragment.mobile:
            k = 5
            lookup = {}
            for i in range(max(0, len(genome) - k + 1)):
                lookup.setdefault(tuple(int(v) for v in genome[i:i + k]), i)
            candidates = []
            for j in range(max(0, len(fragment.sequence) - k + 1)):
                key = tuple(int(v) for v in fragment.sequence[j:j + k])
                if key in lookup:
                    candidates.append(lookup[key])
            if candidates:
                raw_position = int(candidates[int(world.rng.integers(0, len(candidates)))])
                # Insert at a gene boundary so a complete transferred gene is
                # not made nonfunctional merely by splitting an existing gene.
                position = int(round(raw_position / float(g2.GENE_SPAN))) * g2.GENE_SPAN
                position = max(0, min(len(genome), position))
        insert = fragment.sequence[:used].copy()
        self.genomes[target_index] = np.concatenate([
            genome[:position], insert, genome[position:]
        ]).astype(np.uint8)
        if used < len(fragment.sequence):
            self.pools[POOL_NUCLEOTIDE] += (len(fragment.sequence) - used) * MONOMER_MASS
        lesion = float(fragment.lesion) * used / max(1, len(fragment.sequence))
        self.genome_lesions[target_index] = float(
            self.genome_lesions[target_index] + 0.35 * lesion
        )
        self._refresh_gene_cache()
        self._clean_control_state()
        self.hgt_integrations += 1
        self.hgt_symbols += used
        self.hgt_marker_hashes.add(fragment.origin_hash)
        self.foreign_lineages[fragment.origin_lineage] = (
            self.foreign_lineages.get(fragment.origin_lineage, 0) + used
        )
        self.last_hgt_outcome = 'integrated-mobile' if fragment.mobile else 'integrated'
        after_novel = any(g2.sequence_has_novel_path(genome) for genome in self.genomes)
        if not before_novel and after_novel:
            self.hgt_novel_path_events += 1
            world.hgt_novel_path_events += 1
        return True

    def process_edna(self, world, dt):
        self.last_edna_signal = 0.0
        self.last_dna_uptake_mass = 0.0
        if not world.config.extracellular_dna or not world.edna.fragments:
            return
        nearby = []
        sense_radius = self.radius + 0.115
        for index, fragment in enumerate(world.edna.fragments):
            distance = torus_distance(self.pos, fragment.pos)
            if distance < sense_radius:
                signal = fragment.material_mass() * math.exp(-distance / 0.045)
                self.last_edna_signal += signal / 0.012
                nearby.append((distance, index, fragment))
        if not nearby or not world.config.competence:
            return
        competence = self.ecology_activity(ECO_COMPETENCE)
        if competence <= 1e-8:
            return
        nearby.sort(key=lambda item: item[0])
        distance, index, fragment = nearby[0]
        rate = (
            0.42 * competence * world.config.hgt_rate_scale
            * fragment.material_mass() / (0.010 + fragment.material_mass())
            * math.exp(-distance / 0.050)
        )
        if world.rng.random() >= 1.0 - math.exp(-rate * dt):
            return
        cost = 0.00042 * len(fragment.sequence)
        if self.pools[POOL_ATP] < cost + 0.020:
            return
        # Transfer the actual fragment object out of the environmental pool.
        fragment = world.edna.fragments.pop(index)
        self.pools[POOL_ATP] -= cost
        world.dissipated_energy += cost
        self.ecology_atp += cost
        self.last_dna_uptake_mass = fragment.material_mass()
        world.edna.uptaken_fragments += 1
        world.hgt_attempts += 1
        self.hgt_attempts += 1
        nuclease = self.ecology_activity(ECO_NUCLEASE)
        recombinase = self.ecology_activity(ECO_RECOMBINASE)
        if not world.config.recombination or recombinase <= 1e-8:
            self._dispose_failed_fragment(fragment, world, 'recombination-off')
            return
        integrated = self.integrate_fragment(fragment, world)
        if integrated:
            world.edna.integrated_fragments += 1
            world.hgt_integrations += 1
        # Nuclease increases the likelihood that non-integrated foreign DNA is
        # salvaged as monomer; integration path already accounts for all matter.
        _ = nuclease

    def process_corpse_contact(self, world, dt):
        self.last_corpse_signal = 0.0
        self.last_necrotoxin_signal = 0.0
        self.last_necrophagy_mass = 0.0
        if not world.corpses:
            return
        nearby = []
        for corpse in world.corpses:
            distance = torus_distance(self.pos, corpse.pos)
            if distance < self.radius + corpse.radius + 0.035:
                signal = corpse.material_mass() * math.exp(-distance / 0.060)
                self.last_corpse_signal += signal / 0.30
                self.last_necrotoxin_signal += (
                    corpse.pools[CORPSE_TOXIC] * math.exp(-distance / 0.045) / 0.08
                )
                nearby.append((distance, corpse))
        if not nearby or not world.config.corpse_nutrition:
            return
        activity = self.ecology_activity(ECO_NECROPHAGE)
        if activity <= 1e-8:
            return
        nearby.sort(key=lambda item: item[0])
        distance, corpse = nearby[0]
        capacity = (
            dt * 0.012 * activity * world.config.necrophagy_rate_scale
            * math.exp(-distance / 0.050)
        )
        cost_factor = 0.22
        capacity = min(capacity, max(0.0, self.pools[POOL_ATP] - 0.020) / cost_factor)
        if capacity <= 1e-10:
            return
        removed = np.zeros(CORPSE_POOL_COUNT, dtype=float)
        remaining = capacity
        for source in (CORPSE_LABILE, CORPSE_PROTEIN, CORPSE_MINERAL, CORPSE_STRUCTURAL):
            take = min(remaining, corpse.pools[source])
            corpse.pools[source] -= take
            removed[source] += take
            remaining -= take
            if remaining <= 1e-12:
                break
        consumed = float(np.sum(removed))
        if consumed <= 0.0:
            return
        paid = min(self.pools[POOL_ATP], consumed * cost_factor)
        self.pools[POOL_ATP] -= paid
        world.dissipated_energy += paid
        self.ecology_atp += paid
        self.pools[POOL_FUEL] += (
            removed[CORPSE_LABILE]
            + 0.62 * removed[CORPSE_PROTEIN]
            + 0.26 * removed[CORPSE_STRUCTURAL]
        )
        self.pools[POOL_MINERAL] += (
            removed[CORPSE_MINERAL]
            + 0.38 * removed[CORPSE_PROTEIN]
            + 0.74 * removed[CORPSE_STRUCTURAL]
        )
        detox = self.ecology_activity(ECO_DETOX)
        toxic_fraction = (
            0.10 * (1.0 / (1.0 + 0.8 * detox))
            if world.config.corpse_toxicity else 0.0
        )
        toxic = min(corpse.pools[CORPSE_TOXIC], consumed * toxic_fraction)
        corpse.pools[CORPSE_TOXIC] -= toxic
        self.pools[POOL_REACTIVE] += toxic
        self.necrotoxin_uptake += toxic
        self.last_necrotoxin_signal += toxic / 0.02
        self.necrophagy_mass += consumed
        self.last_necrophagy_mass = consumed
        world.necrophagy_mass += consumed
        world.necrotoxin_uptake += toxic

    def _mobile_gene_segments(self):
        segments = []
        for genome in self.genomes:
            for spec in g2.parse_genes(genome):
                if (
                    spec['role'] == ROLE_REGULATOR
                    and spec['localisation'] == LOC_ECOLOGY
                    and int(spec['parameter']) % ECOLOGY_COUNT == ECO_MOBILE
                ):
                    start = int(spec['start'])
                    segments.append(genome[start:start + g2.GENE_SPAN].copy())
        return segments

    def export_mobile_element(self, world, dt):
        if not world.config.mobile_elements or not world.config.vesicle_export:
            return
        mobile = self.ecology_activity(ECO_MOBILE)
        vesicle = self.ecology_activity(ECO_VESICLE)
        if mobile <= 1e-8 or vesicle <= 1e-8:
            return
        segments = self._mobile_gene_segments()
        if not segments:
            return
        rate = 0.018 * mobile * vesicle / (1.0 + vesicle) * world.config.mobile_rate_scale
        if world.rng.random() >= 1.0 - math.exp(-rate * dt):
            return
        segment = segments[int(world.rng.integers(0, len(segments)))]
        material = len(segment) * MONOMER_MASS
        atp_cost = 0.0015 * len(segment)
        if (
            self.pools[POOL_NUCLEOTIDE] < material
            or self.pools[POOL_ATP] < atp_cost + 0.022
        ):
            return
        self.pools[POOL_NUCLEOTIDE] -= material
        self.pools[POOL_ATP] -= atp_cost
        world.dissipated_energy += atp_cost
        self.ecology_atp += atp_cost
        offset = world.rng.normal(0.0, self.radius * 0.25, 2)
        world.edna.add_fragment(
            segment, (self.pos + offset) % 1.0,
            origin_lineage=self.lineage, origin_cell=self.cell_id,
            origin_hash=_sequence_hash(segment), lesion=0.0, mobile=True,
        )
        self.mobile_exports += 1
        self.mobile_symbols_exported += len(segment)
        self.last_mobile_export_age = self.age
        world.mobile_exports += 1
        world.mobile_symbols_exported += len(segment)
        # Copying and budding a selfish element generates chemical wear.
        converted = min(self.pools[POOL_WASTE], material * 0.06)
        self.pools[POOL_WASTE] -= converted
        self.pools[POOL_REACTIVE] += converted

    def metabolism(self, world, dt, config):
        super(EcologicalProtoCell, self).metabolism(world, dt, config)
        if self.alive:
            self.export_mobile_element(world, dt)

    def split(self, world):
        daughters = super(EcologicalProtoCell, self).split(world)
        if daughters is None:
            return None
        for daughter in daughters:
            daughter.__class__ = EcologicalProtoCell
            daughter._init_ecology_state()
            daughter._refresh_gene_cache()
            daughter.previous_autopoietic_margin = daughter.autopoietic_margin()
            daughter.last_margin = daughter.previous_autopoietic_margin
        return daughters

    def state_dict(self):
        state = super(EcologicalProtoCell, self).state_dict()
        state.update({
            'cell_class': 'EcologicalProtoCell',
            'last_edna_signal': self.last_edna_signal,
            'last_corpse_signal': self.last_corpse_signal,
            'last_necrotoxin_signal': self.last_necrotoxin_signal,
            'last_dna_uptake_mass': self.last_dna_uptake_mass,
            'last_necrophagy_mass': self.last_necrophagy_mass,
            'last_hgt_outcome': self.last_hgt_outcome,
            'hgt_attempts': self.hgt_attempts,
            'hgt_integrations': self.hgt_integrations,
            'hgt_rejections': self.hgt_rejections,
            'hgt_digestions': self.hgt_digestions,
            'hgt_symbols': self.hgt_symbols,
            'hgt_novel_path_events': self.hgt_novel_path_events,
            'foreign_lineages': sorted((int(k), int(v)) for k, v in self.foreign_lineages.items()),
            'necrophagy_mass': self.necrophagy_mass,
            'necrotoxin_uptake': self.necrotoxin_uptake,
            'ecology_atp': self.ecology_atp,
            'mobile_exports': self.mobile_exports,
            'mobile_symbols_exported': self.mobile_symbols_exported,
            'last_mobile_export_age': self.last_mobile_export_age,
            'hgt_marker_hashes': sorted(self.hgt_marker_hashes),
        })
        return state

    @classmethod
    def from_state(cls, rng, state):
        parent = s4.SensorimotorProtoCell.from_state(rng, state)
        parent.__class__ = cls
        cell = parent
        cell._init_ecology_state()
        for name in (
            'last_edna_signal', 'last_corpse_signal', 'last_necrotoxin_signal',
            'last_dna_uptake_mass', 'last_necrophagy_mass', 'necrophagy_mass',
            'necrotoxin_uptake', 'ecology_atp', 'last_mobile_export_age',
        ):
            setattr(cell, name, float(state.get(name, getattr(cell, name))))
        for name in (
            'hgt_attempts', 'hgt_integrations', 'hgt_rejections',
            'hgt_digestions', 'hgt_symbols', 'hgt_novel_path_events',
            'mobile_exports', 'mobile_symbols_exported',
        ):
            setattr(cell, name, int(state.get(name, getattr(cell, name))))
        cell.last_hgt_outcome = str(state.get('last_hgt_outcome', 'none'))
        cell.foreign_lineages = {
            int(k): int(v) for k, v in state.get('foreign_lineages', [])
        }
        cell.hgt_marker_hashes = set(state.get('hgt_marker_hashes', []))
        return cell


class EcologicalWorld(s4.SensorimotorWorld):
    """Shared patch world with corpses, extracellular DNA and limited HGT."""

    def __init__(self, seed=101, initial_cells=3, config=None):
        self.corpses = []
        self.edna = None
        super(EcologicalWorld, self).__init__(
            seed=seed, initial_cells=0,
            config=config if config is not None else EcologyConfig(),
        )
        self.config = config if config is not None else EcologyConfig()
        self.edna = ExtracellularDNAField(self.rng)
        self.cells = []
        self.next_cell_id = 0
        count = int(initial_cells)
        for index in range(count):
            angle = 2.0 * math.pi * index / max(1, count)
            radial = 0.0 if count == 1 else 0.10
            position = np.array([
                0.50 + radial * math.cos(angle),
                0.50 + radial * math.sin(angle),
            ]) % 1.0
            self.cells.append(EcologicalProtoCell(
                self.next_cell_id, self.rng, position=position,
                generation=0, lineage=index, bootstrap=True,
            ))
            self.next_cell_id += 1
        self.next_corpse_id = 0
        self.corpse_genomes_released = 0
        self.corpse_fragments_released = 0
        self.corpse_genomes_recycled = 0
        self.necrophagy_mass = 0.0
        self.necrotoxin_uptake = 0.0
        self.hgt_attempts = 0
        self.hgt_integrations = 0
        self.hgt_digestions = 0
        self.hgt_novel_path_events = 0
        self.mobile_exports = 0
        self.mobile_symbols_exported = 0
        self.corpse_recycled_mass = 0.0
        self.last_hgt_event = None
        self.last_necrophagy_event = None
        # Environmental mobile fragments are an explicit post-baseline
        # material injection.  Establish the baseline first so their mass is
        # not counted both in initial material and injected material.
        self.initial_total_material = self.total_material()
        self._seed_mobile_if_requested()
        self.last_step_material_residual = 0.0

    def _seed_mobile_if_requested(self):
        if not self.config.seed_mobile_fragments or not self.config.extracellular_dna:
            return
        for _ in range(4):
            position = (
                np.asarray([0.50, 0.50])
                + self.rng.normal(0.0, 0.10, 2)
            ) % 1.0
            sequence = mobile_element_sequence()
            # Environmental seeding is an explicit material injection.
            self.edna.add_fragment(
                sequence, position, origin_lineage=-99, origin_cell=-99,
                origin_hash='seed-mobile', mobile=True,
            )
            self.field.injected_material += len(sequence) * MONOMER_MASS

    def total_material(self):
        base_total = super(EcologicalWorld, self).total_material()
        corpse_total = float(sum(corpse.material_mass() for corpse in getattr(self, 'corpses', [])))
        edna_total = 0.0 if getattr(self, 'edna', None) is None else self.edna.material_total()
        return float(base_total + corpse_total + edna_total)

    def _particle_interactions(self, dt):
        super(EcologicalWorld, self)._particle_interactions(dt)
        for cell in self.living_cells():
            cell.process_corpse_contact(self, dt)
            cell.process_edna(self, dt)

    def _release_dead_cell_immediate(self, cell):
        # Comparator only: reproduce 0.4's anonymous immediate recycling.
        return s4.SensorimotorWorld._release_dead_cell(self, cell)

    def _release_dead_cell(self, cell):
        if self.config.immediate_dead_recycling or not self.config.corpse_chemistry:
            return self._release_dead_cell_immediate(cell)
        total = cell.material_mass()
        corpse = Corpse.from_cell(self.next_corpse_id, cell)
        self.next_corpse_id += 1
        self.corpses.append(corpse)
        self.dissipated_energy += float(cell.pools[POOL_ATP])
        self.released_dead_material += total
        self.deaths += 1
        self.last_deaths.append((
            cell.cell_id, self.age, cell.death_reason,
            cell.age, cell.damage_burden(), cell.pole_age,
        ))
        self.last_deaths = self.last_deaths[-16:]
        while len(self.corpses) > MAX_CORPSES:
            oldest = max(range(len(self.corpses)), key=lambda i: self.corpses[i].age)
            corpse = self.corpses.pop(oldest)
            corpse._flush_buffers(self, force=True)
            # Remaining complex body is conservatively returned as waste.
            amount = corpse.material_mass()
            if amount > 0.0:
                self.field.add_particle(
                    PARTICLE_WASTE, corpse.pos, amount,
                    count_as_injection=False,
                )
                self.corpse_recycled_mass += amount

    def _step_corpses(self, dt):
        kept = []
        for corpse in self.corpses:
            corpse.step(self, dt)
            if corpse.empty():
                continue
            kept.append(corpse)
        self.corpses = kept

    def step(self, dt):
        dt = clamp(dt, 1e-5, 0.10)
        # Existing corpse and extracellular information dynamics occur before
        # the living-cell step.  Every transfer remains visible to the base
        # cumulative material ledger through total_material().
        self._step_corpses(dt)
        if self.edna is not None:
            self.edna.step(self, dt)
        super(EcologicalWorld, self).step(dt)
        if not self.finite():
            raise FloatingPointError('non-finite SOMA-CELL 0.5 state')

    def finite(self):
        if not super(EcologicalWorld, self).finite():
            return False
        for corpse in self.corpses:
            if not finite_array(corpse.pos) or not finite_array(corpse.pools):
                return False
            if not finite_array(corpse.release_buffer):
                return False
        if self.edna is not None:
            for fragment in self.edna.fragments:
                if not finite_array(fragment.pos) or not finite_array(fragment.sequence):
                    return False
                if not np.isfinite(fragment.lesion) or not np.isfinite(fragment.age):
                    return False
        return True

    def summary(self):
        summary = super(EcologicalWorld, self).summary()
        alive = self.living_cells()
        summary.update({
            'build': BUILD,
            'corpses': len(self.corpses),
            'corpse_material': float(sum(c.material_mass() for c in self.corpses)),
            'corpse_toxic_material': float(sum(c.pools[CORPSE_TOXIC] for c in self.corpses)),
            'corpse_genome_material': float(sum(c.genome_mass() for c in self.corpses)),
            'edna_fragments': 0 if self.edna is None else len(self.edna.fragments),
            'edna_material': 0.0 if self.edna is None else self.edna.material_total(),
            'edna_released_symbols': 0 if self.edna is None else self.edna.released_symbols,
            'edna_decayed_symbols': 0 if self.edna is None else self.edna.decayed_symbols,
            'hgt_attempts': int(self.hgt_attempts),
            'hgt_integrations': int(self.hgt_integrations),
            'hgt_digestions': int(self.hgt_digestions),
            'hgt_novel_path_events': int(self.hgt_novel_path_events),
            'necrophagy_mass': float(self.necrophagy_mass),
            'necrotoxin_uptake': float(self.necrotoxin_uptake),
            'mobile_exports': int(self.mobile_exports),
            'mobile_symbols_exported': int(self.mobile_symbols_exported),
            'corpse_genomes_released': int(self.corpse_genomes_released),
            'corpse_fragments_released': int(self.corpse_fragments_released),
            'corpse_genomes_recycled': int(self.corpse_genomes_recycled),
            'ecology_atp_total': float(sum(cell.ecology_atp for cell in alive)),
            'mean_hgt_integrations': float(np.mean([
                cell.hgt_integrations for cell in alive
            ])) if alive else 0.0,
            'mean_necrophagy': float(np.mean([
                cell.necrophagy_mass for cell in alive
            ])) if alive else 0.0,
            'cells_with_foreign_information': int(sum(
                bool(cell.hgt_marker_hashes) for cell in alive
            )),
            'cells_with_mobile_gene': int(sum(
                ecology_gene_counts(cell.genomes[0])[ECO_MOBILE] > 0
                if cell.genomes else False for cell in alive
            )),
        })
        return summary

    def state_dict(self):
        state = super(EcologicalWorld, self).state_dict()
        state.update({
            'save_version': SAVE_VERSION,
            'build': BUILD,
            'config': self.config.state_dict(),
            'cells': [cell.state_dict() for cell in self.cells],
            'corpses': [corpse.state_dict() for corpse in self.corpses],
            'edna': self.edna.state_dict(),
            'next_corpse_id': self.next_corpse_id,
            'corpse_genomes_released': self.corpse_genomes_released,
            'corpse_fragments_released': self.corpse_fragments_released,
            'corpse_genomes_recycled': self.corpse_genomes_recycled,
            'necrophagy_mass': self.necrophagy_mass,
            'necrotoxin_uptake': self.necrotoxin_uptake,
            'hgt_attempts': self.hgt_attempts,
            'hgt_integrations': self.hgt_integrations,
            'hgt_digestions': self.hgt_digestions,
            'hgt_novel_path_events': self.hgt_novel_path_events,
            'mobile_exports': self.mobile_exports,
            'mobile_symbols_exported': self.mobile_symbols_exported,
            'corpse_recycled_mass': self.corpse_recycled_mass,
            'last_hgt_event': self.last_hgt_event,
            'last_necrophagy_event': self.last_necrophagy_event,
        })
        return state

    @classmethod
    def from_state(cls, state):
        world = cls(
            seed=int(state['seed']), initial_cells=0,
            config=EcologyConfig.from_state(state.get('config', {})),
        )
        world.rng.bit_generator.state = state['rng_state']
        world.age = float(state['age'])
        world.field = s4.SensorimotorParticleField.from_state(world.rng, state['field'])
        world.field._world_ref = world
        world.cells = [EcologicalProtoCell.from_state(world.rng, item) for item in state['cells']]
        world.corpses = [Corpse.from_state(item) for item in state.get('corpses', [])]
        world.edna = ExtracellularDNAField.from_state(world.rng, state.get('edna', {}))
        world.rng.bit_generator.state = state['rng_state']
        for name in (
            'next_cell_id', 'births', 'divisions', 'deaths', 'manual_injections',
            'manual_punctures', 'damage_events', 'stress_pulses',
            'damage_segregation_events', 'rejuvenation_events', 'switch_events',
            'next_corpse_id', 'corpse_genomes_released',
            'corpse_fragments_released', 'corpse_genomes_recycled',
            'hgt_attempts', 'hgt_integrations',
            'hgt_digestions', 'hgt_novel_path_events', 'mobile_exports',
            'mobile_symbols_exported',
        ):
            setattr(world, name, int(state.get(name, getattr(world, name))))
        for name in (
            'dissipated_energy', 'division_parent_material', 'division_daughter_material',
            'division_shed_material', 'division_residual', 'released_dead_material',
            'external_protein_assistance', 'initial_total_material',
            'last_step_material_residual', 'current_stress',
            'fuel_contamination_total', 'necrophagy_mass',
            'necrotoxin_uptake', 'corpse_recycled_mass',
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
        world._switch_recorded = bool(state.get('_switch_recorded', False))
        world.last_patch_distances = np.asarray(
            state.get('last_patch_distances', np.zeros(4)), dtype=float
        ).copy()
        world.last_hgt_event = state.get('last_hgt_event')
        world.last_necrophagy_event = state.get('last_necrophagy_event')
        return world

    def save(self, path=SAVE_FILE):
        _atomic_pickle(path, self.state_dict())

    @classmethod
    def load(cls, path=SAVE_FILE):
        with open(path, 'rb') as handle:
            return cls.from_state(pickle.load(handle))

    def clone(self):
        return EcologicalWorld.from_state(self.state_dict())


def run_headless_trial(seed=101, seconds=240.0, initial_cells=3, config=None):
    world = EcologicalWorld(
        seed=seed, initial_cells=initial_cells,
        config=config if config is not None else EcologyConfig(),
    )
    dt = 1.0 / SIM_HZ
    margin_integral = 0.0
    living_time = 0.0
    for _ in range(int(round(float(seconds) * SIM_HZ))):
        world.step(dt)
        summary = world.summary()
        if summary['cells'] > 0:
            margin_integral += summary['mean_autopoietic_margin'] * dt
            living_time += dt
    result = world.summary()
    result.update({
        'requested_seconds': float(seconds),
        'completed_seconds': float(world.age),
        'mean_margin_over_life': float(margin_integral / max(dt, living_time)),
    })
    return result


LOG_FIELDS = (
    'wall_time', 'session_id', 'reason', 'build', 'seed', 'age', 'cells',
    'births', 'divisions', 'deaths', 'max_generation', 'mean_autopoietic_margin',
    'mean_atp', 'mean_closure', 'mean_loop', 'mean_damage', 'corpses',
    'corpse_material', 'corpse_toxic_material', 'corpse_genome_material',
    'edna_fragments', 'edna_material', 'hgt_attempts', 'hgt_integrations',
    'hgt_digestions', 'hgt_novel_path_events', 'necrophagy_mass',
    'necrotoxin_uptake', 'mobile_exports', 'cells_with_foreign_information',
    'cells_with_mobile_gene', 'ecology_atp_total', 'total_material',
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
            'wall_time': time.time(), 'session_id': self.session_id,
            'reason': reason, 'fps': float(fps), 'sim_rate': float(sim_rate),
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
        sessions = []
        for session_id, items in grouped.items():
            last = items[-1]
            sessions.append({
                'session_id': session_id,
                'start_age': items[0].get('age', ''),
                'end_age': last.get('age', ''),
                'final_cells': last.get('cells', ''),
                'final_generation': last.get('max_generation', ''),
                'corpses': last.get('corpses', ''),
                'edna_fragments': last.get('edna_fragments', ''),
                'hgt_integrations': last.get('hgt_integrations', ''),
                'necrophagy_mass': last.get('necrophagy_mass', ''),
                'mobile_exports': last.get('mobile_exports', ''),
                'matter_residual': last.get('matter_residual', ''),
            })
        with open(session_path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(sessions[0].keys()))
            writer.writeheader()
            writer.writerows(sessions)
        last = rows[-1]
        lines = [
            BUILD + ' long-run report',
            'sessions: {}'.format(len(grouped)),
            'last age / cells / generation: {} / {} / {}'.format(
                last.get('age'), last.get('cells'), last.get('max_generation')
            ),
            'corpses / corpse matter: {} / {}'.format(
                last.get('corpses'), last.get('corpse_material')
            ),
            'eDNA fragments / matter: {} / {}'.format(
                last.get('edna_fragments'), last.get('edna_material')
            ),
            'HGT attempts / integrations / digestions: {} / {} / {}'.format(
                last.get('hgt_attempts'), last.get('hgt_integrations'),
                last.get('hgt_digestions')
            ),
            'necrophagy / necrotoxin: {} / {}'.format(
                last.get('necrophagy_mass'), last.get('necrotoxin_uptake')
            ),
            'mobile exports: {}'.format(last.get('mobile_exports')),
            'matter residual: {}'.format(last.get('matter_residual')),
            'Death-derived matter and DNA are explicit conserved pools; HGT is not free copying.',
        ]
        with open(report_path, 'w') as handle:
            handle.write('\n'.join(lines) + '\n')
        return 'OK'
    except Exception:
        return 'WRITE ERR'


try:
    from scene import Scene, run, LANDSCAPE
    from scene import background, fill, stroke, stroke_weight, ellipse, line, rect, text

    class SomaCellEcologyScene(Scene):
        def setup(self):
            self.paused = False
            self.accumulator = 0.0
            self.last_wall = time.time()
            self.last_touch_wall = -10.0
            self.last_save_age = 0.0
            self.save_status = 'NEW'
            self.report_status = 'WAIT'
            self.fps = 0.0
            self.sim_rate = 0.0
            self.telemetry_wall = time.time()
            self.telemetry_age = 0.0
            self.telemetry_frames = 0
            try:
                if os.path.exists(SAVE_FILE):
                    self.world = EcologicalWorld.load(SAVE_FILE)
                    self.save_status = 'LOAD'
                else:
                    self.world = EcologicalWorld(seed=101, initial_cells=3)
            except Exception:
                self.world = EcologicalWorld(seed=101, initial_cells=3)
                self.save_status = 'RECOVER'
            self.logger = LongRunLogger(self.world)
            self.logger.log(self.world, reason='start', force=True)

        def _world_rect(self):
            width, height = float(self.size.w), float(self.size.h)
            return 34.0, 82.0, width - 68.0, height - 170.0

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
                (0.28, 0.90, 0.46), (0.96, 0.69, 0.20),
                (0.92, 0.25, 0.29), (0.20, 0.72, 1.00),
            )
            for index in range(len(self.world.field.amount)):
                x, y = self._screen(self.world.field.pos[index])
                amount = float(self.world.field.amount[index])
                radius = 1.3 + 6.8 * math.sqrt(clamp(amount / 0.045, 0.0, 1.7))
                colour = colours[int(self.world.field.kind[index])]
                fill(colour[0], colour[1], colour[2], 0.68)
                ellipse(x - radius, y - radius, 2 * radius, 2 * radius)

        def _draw_corpses(self):
            left, bottom, width, height = self._world_rect()
            scale = min(width, height)
            for corpse in self.world.corpses:
                x, y = self._screen(corpse.pos)
                radius = max(5.0, corpse.radius * scale * (0.72 + 0.28 * corpse.exposure()))
                toxic = clamp(corpse.pools[CORPSE_TOXIC] / 0.18, 0.0, 1.0)
                fill(0.30 + 0.35 * toxic, 0.20, 0.12 + 0.25 * toxic, 0.55)
                ellipse(x - radius, y - radius, 2 * radius, 2 * radius)
                stroke(0.70, 0.48, 0.24, 0.75)
                stroke_weight(1.2)
                line(x - radius * 0.7, y - radius * 0.6,
                     x + radius * 0.7, y + radius * 0.6)
                line(x - radius * 0.7, y + radius * 0.6,
                     x + radius * 0.7, y - radius * 0.6)

        def _draw_edna(self):
            for fragment in self.world.edna.fragments:
                x, y = self._screen(fragment.pos)
                size = 2.0 + min(6.0, len(fragment.sequence) / 20.0)
                if fragment.mobile:
                    stroke(1.0, 0.25, 0.92, 0.95)
                else:
                    stroke(0.74, 0.42, 1.0, 0.78)
                stroke_weight(1.2)
                line(x - size, y - 1.5, x + size, y + 1.5)
                line(x - size, y + 1.5, x + size, y - 1.5)

        def _draw_cell(self, cell):
            # Reuse the proven 0.4 membrane drawing by a compact local version.
            left, bottom, width, height = self._world_rect()
            cx, cy = self._screen(cell.pos)
            scale = min(width, height)
            sr = cell.radius * scale
            closure = cell.closure_array()
            damage = clamp(cell.damage_burden(), 0.0, 1.0)
            margin = clamp(cell.autopoietic_margin(), 0.0, 1.0)
            foreign = clamp(cell.hgt_integrations / 3.0, 0.0, 1.0)
            fill(0.08 + 0.30 * damage, 0.22 + 0.48 * margin,
                 0.42 + 0.30 * foreign - 0.20 * damage, 0.34)
            ellipse(cx - sr, cy - sr, 2 * sr, 2 * sr)
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
                stroke(0.34 + 0.58 * oxidation, 0.78 - 0.48 * oxidation,
                       0.95 - 0.52 * oxidation, 0.30 + 0.70 * local)
                stroke_weight(0.6 + 3.2 * local)
                line(x0, y0, x1, y1)
            command = cell.last_motor_command
            if float(np.linalg.norm(command)) > 1e-5:
                stroke(0.98, 0.94, 0.42, 0.90)
                stroke_weight(2.0)
                line(cx, cy, cx + command[0] * sr * 1.7, cy + command[1] * sr * 1.7)
            if cell.hgt_integrations > 0:
                fill(0.82, 0.38, 1.0, 0.95)
                ellipse(cx - 3, cy + sr * 0.48 - 3, 6, 6)
            if ecology_gene_counts(cell.genomes[0])[ECO_MOBILE] > 0 if cell.genomes else False:
                fill(1.0, 0.22, 0.85, 0.95)
                ellipse(cx + sr * 0.38 - 3, cy + sr * 0.34 - 3, 6, 6)
            fill(0.91, 0.97, 1.0)
            text('#{} G{} M{:.2f} H{}'.format(
                cell.cell_id, cell.generation,
                cell.autopoietic_margin(), cell.hgt_integrations,
            ), x=cx, y=cy - sr - 10, font_size=8, alignment=5)

        def draw(self):
            background(0.010, 0.018, 0.030)
            left, bottom, width, height = self._world_rect()
            fill(0.020, 0.040, 0.055)
            rect(left, bottom, width, height)
            self._draw_particles()
            self._draw_corpses()
            self._draw_edna()
            for cell in self.world.living_cells():
                self._draw_cell(cell)
            s = self.world.summary()
            fill(0.92, 0.98, 1.0)
            text(BUILD, x=24, y=self.size.h - 25, font_size=18, alignment=4)
            fill(0.64, 0.78, 0.86)
            text('SCENE ACTIVE | SAVE {} | LOG {} | REPORT {} | {:.1f} fps | x{:.2f}'.format(
                self.save_status, self.logger.status, self.report_status,
                self.fps, self.sim_rate,
            ), x=self.size.w - 72, y=self.size.h - 25, font_size=9, alignment=6)
            fill(0.84, 0.92, 0.97)
            text('age {:.1f}s cells {} corpses {} DNA {} div {} deaths {} G{}'.format(
                s['age'], s['cells'], s['corpses'], s['edna_fragments'],
                s['divisions'], s['deaths'], s['max_generation'],
            ), x=24, y=62, font_size=10, alignment=4)
            text('margin {:.3f} ATP {:.3f} corpseM {:.3f} eDNA {:.3f} ledger {:+.2e}'.format(
                s['mean_autopoietic_margin'], s['mean_atp'],
                s['corpse_material'], s['edna_material'], s['matter_residual'],
            ), x=24, y=46, font_size=9, alignment=4)
            text('HGT try/int/digest {}/{}/{} novel {} foreign cells {}'.format(
                s['hgt_attempts'], s['hgt_integrations'], s['hgt_digestions'],
                s['hgt_novel_path_events'], s['cells_with_foreign_information'],
            ), x=24, y=30, font_size=9, alignment=4)
            text('necrophagy {:.3f} toxin {:.3f} mobile exports {} ecology ATP {:.3f}'.format(
                s['necrophagy_mass'], s['necrotoxin_uptake'],
                s['mobile_exports'], s['ecology_atp_total'],
            ), x=24, y=14, font_size=9, alignment=4)
            if s['cells'] == 0:
                fill(1.0, 0.38, 0.32)
                text('CHEMICAL POPULATION EXTINCT — bodies and DNA still decay',
                     x=self.size.w * 0.5, y=self.size.h * 0.52,
                     font_size=15, alignment=5)
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
                self.world = EcologicalWorld(seed=101, initial_cells=3)
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
                    reason='pause' if self.paused else 'resume', force=True,
                )
                return
            position = self._unit_position(touch.location)
            if not self.world.puncture_nearest(position):
                self.world.inject_cloud(position)

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
        print(run_headless_trial(seed=101, seconds=20.0, initial_cells=3))
    else:
        run(SomaCellEcologyScene(), LANDSCAPE, show_fps=False)
