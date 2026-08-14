# coding: utf-8
"""SOMA-CELL 0.6.8-GPU A3 full-fidelity metabolism/damage kernels.

The frozen 0.2/0.3/0.4/0.5/0.6.6 Python implementations remain the semantic
authority.  This module packs their variable protein dictionaries without
sorting them, retains aggregate molecular composition, and supplies paired
NumPy/Torch fp64 transcriptions of the deterministic metabolism and damage
phases.  Genome copying, lesion-triggered symbol deletion, division, death,
eDNA/HGT and neural state remain CPU-authoritative.

All public pure kernels clone their input.  Capacity and schema validation is
performed before a CPU cell is changed; overflow raises ``A3CapacityError``.
In particular, no variable-length collection is clipped to a tensor capacity.
"""
from __future__ import division

import copy
import math
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, fields

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import SOMA_CELL_0_6_8_gpu_a2 as a2

try:
    import torch
except Exception:  # pragma: no cover - NumPy reference remains available
    torch = None

a1 = a2.a1
s66 = a2.s66
s5 = a2.s5
s4 = a2.s4
g3 = s4.g3
g2 = a2.g2

BUILD = 'SOMA-CELL 0.6.8-GPU A3'
BUILD_ID = BUILD
BUILD_LONG = BUILD + ' | full-fidelity gene metabolism and damage/repair kernels'
SCHEMA_VERSION = '0.6.8-GPU-A3.0'
SAVE_VERSION = 1
FULL_GPU_WORLD_STEP = False

MEMBRANE_SEGMENTS = a2.MEMBRANE_SEGMENTS
CHANNEL_COUNT = a2.CHANNEL_COUNT
POOL_COUNT = int(g3.POOL_COUNT)
GENERIC_REACTION_COUNT = len(g2.REACTION_NAMES)
REPAIR_COUNT = len(g3.REPAIR_NAMES)

ACTIVE_MASS_THRESHOLD = 1e-10
DAMAGED_MASS_THRESHOLD = 1e-11
AGGREGATE_MASS_THRESHOLD = 0.0

POOL_FUEL = int(g3.POOL_FUEL)
POOL_MINERAL = int(g3.POOL_MINERAL)
POOL_ATP = int(g3.POOL_ATP)
POOL_MEM_PRECURSOR = int(g3.POOL_MEM_PRECURSOR)
POOL_CATALYST = int(g3.POOL_CATALYST)
POOL_TRANSPORTER_PRECURSOR = int(g3.POOL_TRANSPORTER_PRECURSOR)
POOL_WASTE = int(g3.POOL_WASTE)
POOL_ALT = int(g3.POOL_ALT)
POOL_INTERMEDIATE = int(g3.POOL_INTERMEDIATE)
POOL_NUCLEOTIDE = int(g3.POOL_NUCLEOTIDE)
POOL_DAMAGED_PROTEIN = int(g3.POOL_DAMAGED_PROTEIN)
POOL_AGGREGATE = int(g3.POOL_AGGREGATE)
POOL_REACTIVE = int(g3.POOL_REACTIVE)

ROLE_ENERGY = int(g2.ROLE_ENERGY)
ROLE_MEMBRANE = int(g2.ROLE_MEMBRANE)
ROLE_TRANSPORTER = int(g2.ROLE_TRANSPORTER)
ROLE_REPLICASE = int(g2.ROLE_REPLICASE)
ROLE_TRANSLATOR = int(g2.ROLE_TRANSLATOR)
ROLE_NUCLEOTIDE = int(g2.ROLE_NUCLEOTIDE)
ROLE_GENERIC = int(g2.ROLE_GENERIC)
ROLE_REGULATOR = int(g2.ROLE_REGULATOR)

REPAIR_ANTIOXIDANT = int(g3.REPAIR_ANTIOXIDANT)
REPAIR_CHAPERONE = int(g3.REPAIR_CHAPERONE)
REPAIR_PROTEASE = int(g3.REPAIR_PROTEASE)
REPAIR_GENOME = int(g3.REPAIR_GENOME)
REPAIR_SEGREGATION = int(g3.REPAIR_SEGREGATION)
REPAIR_PROOFREADING = int(g3.REPAIR_PROOFREADING)
REPAIR_QUIESCENCE = int(g3.REPAIR_QUIESCENCE)
REPAIR_MEMBRANE = int(g3.REPAIR_MEMBRANE)
LOC_REPAIR = int(s4.LOC_REPAIR)

PORT_STATUS = dict(a2.PORT_STATUS)
PORT_STATUS.update({
    'packed_damage_schema': 'implemented-strict-capacity-order-preserving',
    'gene_coded_metabolism': 'numpy-torch-fp64-pure-kernels',
    'translation': 'pure-kernel-validated-hybrid-cpu-authoritative',
    'damage_repair': 'numpy-torch-fp64-pure-kernels',
    'aggregate_composition': 'typed-plus-unresolved-legacy-mass',
    'genome_replication': 'cpu-authoritative',
    'genome_symbol_hydrolysis_rng': 'cpu-authoritative-explicit-hazard-plan',
    'division': 'cpu-authoritative-a3-segregation-plan',
    'death_corpse_edna_hgt': 'cpu-authoritative',
    'neural_causal_system': 'cpu-authoritative',
})


class A3Error(RuntimeError):
    """Base class for fail-closed A3 errors."""


class A3CapacityError(A3Error):
    """A variable-length CPU state cannot fit the configured packed schema."""


class A3SchemaError(A3Error):
    """Packed state is malformed, non-finite, or violates a material ledger."""


@dataclass
class GPU068A3Config(a2.GPU068A2Config):
    max_protein_species: int = 256
    max_genome_copies: int = 3
    strict_capacity: bool = True
    strict_material_ledger: bool = True
    ledger_atol: float = 2e-10

    def __post_init__(self):
        if int(self.max_protein_species) <= 0:
            raise ValueError('max_protein_species must be positive')
        if int(self.max_genome_copies) < 3:
            raise ValueError('max_genome_copies must be >= 3 (frozen 0.3 permits three)')
        if int(self.max_genome_symbols) < int(g2.MAX_GENOME_LENGTH):
            raise ValueError('max_genome_symbols is below the frozen genome limit')
        if str(self.precision).lower() not in ('float64', 'fp64', 'double', 'float32', 'fp32', 'single'):
            raise ValueError('A3 supports fp64 reference and an explicitly labelled fp32 candidate')

    @classmethod
    def from_state(cls, state):
        state = dict(state or {})
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in state.items() if key in allowed})


_ARRAY_FIELDS = (
    'pools', 'membrane', 'membrane_oxidation', 'transporters',
    'contact_trace', 'alt_contact_trace', 'damage_trace',
    'genome_lengths', 'genome_lesions', 'genome_mask',
    'genome_hydrolysis_hazard', 'species_fingerprint', 'species_mask',
    'active_mass', 'damaged_mass', 'aggregate_mass', 'active_order',
    'damaged_order', 'aggregate_order', 'gene_order', 'gene_role', 'gene_parameter',
    'gene_reaction', 'gene_promoter', 'gene_efficiency', 'gene_copy_number',
    'gene_localisation', 'last_generic_flux', 'last_repair_flux',
)


@dataclass
class A3PackedState:
    """Fixed-capacity, single-cell packed state used by every A3 pure kernel.

    ``species_fingerprint`` is a union table.  ``active_order``,
    ``damaged_order`` and ``gene_order`` contain union-table indices and retain
    the independent Python insertion orders.  The three amount vectors never
    imply an order by themselves.  Typed aggregate mass is stored per union
    slot, while ``aggregate_unresolved`` preserves legacy scalar aggregate for
    which the frozen source recorded no molecular identity.
    """

    schema_version: str
    capacity: int
    genome_capacity: int
    pools: object
    membrane: object
    membrane_oxidation: object
    transporters: object
    contact_trace: object
    alt_contact_trace: object
    damage_trace: object
    radius: object
    alive: bool
    genome_lengths: object
    genome_lesions: object
    genome_mask: object
    genome_count: int
    genome_lesion_count: int
    replication_template_length: int
    replication_copy_length: int
    replication_template_lesion: object
    replication_active: bool
    genome_hydrolysis_hazard: object
    species_fingerprint: object
    species_mask: object
    species_count: int
    active_mass: object
    damaged_mass: object
    aggregate_mass: object
    aggregate_unresolved: object
    active_order: object
    active_count: int
    damaged_order: object
    damaged_count: int
    aggregate_order: object
    aggregate_count: int
    gene_order: object
    gene_count: int
    gene_role: object
    gene_parameter: object
    gene_reaction: object
    gene_promoter: object
    gene_efficiency: object
    gene_copy_number: object
    gene_localisation: object
    current_stress: object
    division_progress: object
    total_genome_symbols: int
    last_catalysis: object
    last_translation: object
    last_generic_flux: object
    novel_path_flux: object
    last_assembly: object
    maintenance_shortfall: object
    last_damage_generated: object
    last_repair_atp: object
    last_repair_flux: object
    cumulative_damage_generated: object
    cumulative_repair_atp: object
    cumulative_recycled_damage: object
    cumulative_oxidant_neutralised: object
    cumulative_genome_repairs: object
    cumulative_segregation_atp: object
    last_quiescence: object
    last_segregation_strength: object
    supplemental_atp_spent: object
    reaction_events: int
    gene_expression: bool = True
    generic_reactions: bool = True
    membrane_synthesis: bool = True
    transport_enabled: bool = True
    targeted_repair: bool = True
    external_translator: bool = False
    protein_repair: bool = True
    genome_repair: bool = True
    membrane_repair: bool = True
    endogenous_damage: bool = True
    damage_segregation: bool = True
    forced_symmetric_damage: bool = False
    quiescence: bool = True
    damage_rate_scale: float = 1.0
    repair_cost_scale: float = 1.0

    def clone(self):
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in _ARRAY_FIELDS or _is_tensor(value):
                value = _clone_value(value)
            else:
                value = copy.deepcopy(value)
            values[item.name] = value
        return A3PackedState(**values)

    def with_updates(self, **changes):
        out = self.clone()
        unknown = set(changes).difference(item.name for item in fields(self))
        if unknown:
            raise TypeError('unknown A3PackedState fields: %s' % sorted(unknown))
        for key, value in changes.items():
            setattr(out, key, _clone_value(value))
        return out

    def validate(self, config=None):
        return validate_a3_state(self, config=config)

    def to_torch(self, device='cpu', dtype=None):
        if torch is None:
            raise RuntimeError('PyTorch is unavailable')
        dtype = dtype or torch.float64
        if dtype not in (torch.float64, torch.float32):
            raise ValueError('A3 Torch state requires float64 reference or float32 candidate')
        out = self.clone()
        for name in _ARRAY_FIELDS:
            value = getattr(out, name)
            if _is_tensor(value):
                if value.dtype == torch.bool:
                    converted = value.to(device=device, dtype=torch.bool)
                elif value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                    converted = value.to(device=device, dtype=torch.int64)
                else:
                    converted = value.to(device=device, dtype=dtype)
            else:
                array = np.asarray(value)
                if array.dtype == np.bool_:
                    converted = torch.as_tensor(array, dtype=torch.bool, device=device)
                elif np.issubdtype(array.dtype, np.integer):
                    converted = torch.as_tensor(array, dtype=torch.int64, device=device)
                else:
                    converted = torch.as_tensor(array, dtype=dtype, device=device)
            setattr(out, name, converted.clone())
        for name in _SCALAR_FLOAT_FIELDS:
            value = getattr(out, name)
            setattr(out, name, torch.as_tensor(_number(value), dtype=dtype, device=device))
        return out

    def to_numpy(self):
        out = self.clone()
        for name in _ARRAY_FIELDS:
            value = getattr(out, name)
            if _is_tensor(value):
                value = value.detach().cpu().numpy().copy()
            array = np.asarray(value)
            if array.dtype == np.bool_:
                value = array.astype(bool, copy=True)
            elif np.issubdtype(array.dtype, np.integer):
                value = array.astype(np.int64, copy=True)
            else:
                value = array.astype(np.float64, copy=True)
            setattr(out, name, value)
        for name in _SCALAR_FLOAT_FIELDS:
            setattr(out, name, float(_number(getattr(out, name))))
        return out


_SCALAR_FLOAT_FIELDS = (
    'radius', 'replication_template_lesion', 'aggregate_unresolved',
    'current_stress', 'division_progress', 'last_catalysis',
    'last_translation', 'novel_path_flux', 'last_assembly',
    'maintenance_shortfall', 'last_damage_generated', 'last_repair_atp',
    'cumulative_damage_generated', 'cumulative_repair_atp',
    'cumulative_recycled_damage', 'cumulative_oxidant_neutralised',
    'cumulative_genome_repairs', 'cumulative_segregation_atp',
    'last_quiescence', 'last_segregation_strength',
    'supplemental_atp_spent',
)


def _is_tensor(value):
    return torch is not None and isinstance(value, torch.Tensor)


def _clone_value(value):
    if _is_tensor(value):
        return value.clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


def _number(value):
    if _is_tensor(value):
        return value.detach().item()
    if isinstance(value, np.ndarray):
        return value.item()
    return value


def _flag(state, config, name):
    if config is None:
        return bool(getattr(state, name))
    if isinstance(config, dict) and name in config:
        return bool(config[name])
    if hasattr(config, name):
        return bool(getattr(config, name))
    return bool(getattr(state, name))


def _rate(state, config, name):
    if config is None:
        return float(getattr(state, name))
    if isinstance(config, dict) and name in config:
        return float(config[name])
    if hasattr(config, name):
        return float(getattr(config, name))
    return float(getattr(state, name))


def _finite_array(value):
    if _is_tensor(value):
        return bool(torch.isfinite(value).all().detach().item())
    return bool(np.isfinite(np.asarray(value)).all())


def _shape(value):
    return tuple(value.shape)


def _dtype_class(value):
    if _is_tensor(value):
        if value.dtype == torch.bool:
            return 'bool'
        if value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            return 'integer'
        if value.dtype.is_floating_point:
            return 'float'
        return 'other'
    kind = np.asarray(value).dtype.kind
    if kind == 'b':
        return 'bool'
    if kind in 'iu':
        return 'integer'
    if kind in 'fc':
        return 'float'
    return 'other'


def _ordered_total(values, order, count):
    total = values.new_zeros(()) if _is_tensor(values) else np.float64(0.0)
    for position in range(int(count)):
        total = total + values[int(_number(order[position]))]
    return total


def _sum_fixed(values):
    total = values.new_zeros(()) if _is_tensor(values) else np.float64(0.0)
    for index in range(int(values.shape[0])):
        total = total + values[index]
    return total


def _clip(value, low, high):
    if _is_tensor(value):
        return value.clamp(float(low), float(high))
    return np.clip(value, low, high)


def _maximum(value, other):
    if _is_tensor(value):
        other = torch.as_tensor(other, dtype=value.dtype, device=value.device)
        return torch.maximum(value, other)
    return np.maximum(value, other)


def _minimum(value, other):
    if _is_tensor(value):
        other = torch.as_tensor(other, dtype=value.dtype, device=value.device)
        return torch.minimum(value, other)
    return np.minimum(value, other)


def _exp(value):
    return torch.exp(value) if _is_tensor(value) else np.exp(value)


def _roll(value, shift, dim=0):
    return torch.roll(value, int(shift), dims=int(dim)) if _is_tensor(value) else np.roll(value, int(shift), axis=int(dim))


def _zeros_like(value):
    return torch.zeros_like(value) if _is_tensor(value) else np.zeros_like(value)


def _scalar_like(state, value=0.0):
    if _is_tensor(state.pools):
        return state.pools.new_tensor(float(value))
    return np.float64(value)


def _assign_scalar(state, name, value):
    if _is_tensor(state.pools):
        setattr(state, name, torch.as_tensor(value, dtype=state.pools.dtype, device=state.pools.device))
    else:
        setattr(state, name, float(_number(value)))


def _synchronise_material_pools(state):
    state.pools[POOL_CATALYST] = _ordered_total(state.active_mass, state.active_order, state.active_count)
    state.pools[POOL_DAMAGED_PROTEIN] = _ordered_total(state.damaged_mass, state.damaged_order, state.damaged_count)
    state.pools[POOL_AGGREGATE] = (
        _ordered_total(state.aggregate_mass, state.aggregate_order, state.aggregate_count)
        + state.aggregate_unresolved
    )
    return state


def _compact_order(state, mass_name, order_name, count_name, threshold):
    mass = getattr(state, mass_name)
    order = getattr(state, order_name)
    count = int(getattr(state, count_name))
    kept = []
    for position in range(count):
        slot = int(_number(order[position]))
        if _number(mass[slot]) > threshold and math.isfinite(float(_number(mass[slot]))):
            kept.append(slot)
        else:
            mass[slot] = 0.0
    order[:] = -1
    for position, slot in enumerate(kept):
        order[position] = slot
    setattr(state, count_name, len(kept))


def _compact_species_union(state):
    """Remove empty union slots while preserving every independent order."""
    order_specs = (
        ('active_order', 'active_count'), ('damaged_order', 'damaged_count'),
        ('aggregate_order', 'aggregate_count'), ('gene_order', 'gene_count'),
    )
    used = set()
    for order_name, count_name in order_specs:
        order = getattr(state, order_name)
        for position in range(int(getattr(state, count_name))):
            used.add(int(_number(order[position])))
    retained = [slot for slot in range(int(state.species_count)) if slot in used]
    if retained == list(range(int(state.species_count))):
        return state

    mapping = {old: new for new, old in enumerate(retained)}
    slot_fields = (
        'species_fingerprint', 'active_mass', 'damaged_mass',
        'aggregate_mass', 'gene_role', 'gene_parameter', 'gene_reaction',
        'gene_promoter', 'gene_efficiency', 'gene_copy_number',
        'gene_localisation',
    )
    old_slots = {name: _clone_value(getattr(state, name)) for name in slot_fields}
    old_orders = {name: _clone_value(getattr(state, name))
                  for name, _ in order_specs}
    state.species_fingerprint[:] = -1
    state.species_mask[:] = False
    state.active_mass[:] = 0.0
    state.damaged_mass[:] = 0.0
    state.aggregate_mass[:] = 0.0
    state.gene_role[:] = -1
    state.gene_parameter[:] = -1
    state.gene_reaction[:] = -1
    state.gene_promoter[:] = 0.0
    state.gene_efficiency[:] = 0.0
    state.gene_copy_number[:] = 0
    state.gene_localisation[:] = -1
    for new_slot, old_slot in enumerate(retained):
        state.species_mask[new_slot] = True
        for name in slot_fields:
            getattr(state, name)[new_slot] = old_slots[name][old_slot]
    for order_name, count_name in order_specs:
        order = getattr(state, order_name)
        count = int(getattr(state, count_name))
        order[:] = -1
        for position in range(count):
            order[position] = mapping[int(_number(old_orders[order_name][position]))]
    state.species_count = len(retained)
    return state


def _append_order_slot(state, order_name, count_name, slot):
    order = getattr(state, order_name)
    count = int(getattr(state, count_name))
    for position in range(count):
        if int(_number(order[position])) == int(slot):
            return
    if count >= int(state.capacity):
        raise A3CapacityError('%s overflow' % order_name)
    order[count] = int(slot)
    setattr(state, count_name, count + 1)


def _model_value(model_config, name, default):
    if model_config is None:
        return default
    if isinstance(model_config, dict):
        return model_config.get(name, default)
    return getattr(model_config, name, default)


@dataclass(frozen=True)
class AggregateCompositionState:
    """Serializable sidecar for identity-preserving aggregate material."""

    items: tuple
    unresolved: float

    def __post_init__(self):
        seen = set()
        for fingerprint, amount in self.items:
            fingerprint = _strict_fingerprint(fingerprint, 'aggregate fingerprint')
            amount = float(amount)
            if fingerprint in seen:
                raise A3SchemaError('duplicate aggregate fingerprint %d' % fingerprint)
            if not math.isfinite(amount) or amount <= AGGREGATE_MASS_THRESHOLD:
                raise A3SchemaError('invalid aggregate composition item')
            seen.add(fingerprint)
        if not math.isfinite(float(self.unresolved)) or float(self.unresolved) < 0.0:
            raise A3SchemaError('invalid unresolved aggregate mass')

    @property
    def typed_total(self):
        return float(sum(float(amount) for _, amount in self.items))

    @property
    def total(self):
        return self.typed_total + float(self.unresolved)

    def state_dict(self):
        return {
            'schema_version': SCHEMA_VERSION,
            'items': [(int(key), float(value)) for key, value in self.items],
            'unresolved': float(self.unresolved),
        }

    @classmethod
    def from_state(cls, state):
        state = dict(state or {})
        return cls(
            tuple((key, float(value)) for key, value in state.get('items', ())),
            float(state.get('unresolved', 0.0)),
        )


def aggregate_composition_state(cell):
    raw = getattr(cell, '_soma068a3_aggregate_composition', ())
    if isinstance(raw, AggregateCompositionState):
        items = raw.items
        unresolved_default = raw.unresolved
    elif isinstance(raw, dict):
        items = tuple(raw.items())
        unresolved_default = 0.0
    else:
        items = tuple(raw or ())
        unresolved_default = 0.0
    typed_items = []
    for key, value in items:
        fingerprint = _strict_fingerprint(key, 'aggregate fingerprint')
        amount = float(value)
        if not math.isfinite(amount) or amount <= AGGREGATE_MASS_THRESHOLD:
            raise A3SchemaError('aggregate composition contains zero/negative/non-finite mass')
        typed_items.append((fingerprint, amount))
    typed = tuple(typed_items)
    if hasattr(cell, '_soma068a3_aggregate_unresolved'):
        unresolved = float(cell._soma068a3_aggregate_unresolved)
    elif isinstance(raw, AggregateCompositionState):
        unresolved = float(unresolved_default)
    elif raw:
        unresolved = max(0.0, float(cell.pools[POOL_AGGREGATE]) - sum(value for _, value in typed))
    else:
        # Frozen 0.3-0.6 saves only a scalar.  Its identity is unknowable and
        # must not be invented from the damaged/active dictionaries.
        unresolved = float(cell.pools[POOL_AGGREGATE])
    scalar_total = float(cell.pools[POOL_AGGREGATE])
    sidecar_total = float(sum(value for _, value in typed) + unresolved)
    delta = scalar_total - sidecar_total
    if delta > 0.0:
        # CPU-authoritative neural return and mechanism faults may add scalar
        # aggregate between A3 calls.  It has no molecular provenance, so the
        # only honest representation is new unresolved mass.
        unresolved += delta
    elif delta < -2e-10:
        # A scalar loss outside A3 must be accompanied by an explicit sidecar
        # update.  Guessing which typed molecules disappeared would corrupt
        # composition and reduction order.
        raise A3SchemaError(
            'aggregate scalar is below typed/unresolved sidecar by %.17g' % (-delta)
        )
    result = AggregateCompositionState(typed, unresolved)
    return result


def apply_aggregate_composition_state(cell, state):
    """Attach an A3 sidecar after validating it against the scalar pool."""
    if not isinstance(state, AggregateCompositionState):
        state = AggregateCompositionState.from_state(state)
    if not math.isclose(state.total, float(cell.pools[POOL_AGGREGATE]), rel_tol=0.0, abs_tol=2e-10):
        raise A3SchemaError('aggregate sidecar does not equal POOL_AGGREGATE')
    cell._soma068a3_aggregate_composition = OrderedDict(state.items)
    cell._soma068a3_aggregate_unresolved = float(state.unresolved)
    return cell


def _ordered_mapping_items(mapping, label, minimum_exclusive=None):
    if mapping is None:
        return []
    if not hasattr(mapping, 'items'):
        raise A3SchemaError('%s is not a mapping' % label)
    out = []
    seen = set()
    for key, value in mapping.items():
        key = _strict_fingerprint(key, '%s fingerprint' % label)
        value = float(value)
        if key in seen:
            raise A3SchemaError('%s contains invalid/duplicate fingerprint' % label)
        if not math.isfinite(value) or value < 0.0:
            raise A3SchemaError('%s contains negative/non-finite mass' % label)
        if minimum_exclusive is not None and value <= float(minimum_exclusive):
            raise A3SchemaError(
                '%s contains non-lossless mass %.17g at/below %.17g' %
                (label, value, float(minimum_exclusive))
            )
        seen.add(key)
        out.append((key, value))
    return out


def _strict_fingerprint(value, label='fingerprint'):
    """Accept signed integer fingerprints, including reserved negative keys."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise A3SchemaError('%s must be an integer, got %r' % (label, value))
    integer = int(value)
    if integer < -(1 << 63) or integer > (1 << 63) - 1:
        raise A3CapacityError('%s is outside signed int64' % label)
    return integer


def _zero_state(config=None):
    config = config if isinstance(config, GPU068A3Config) else GPU068A3Config.from_state(config or {})
    capacity = int(config.max_protein_species)
    genomes = int(config.max_genome_copies)
    z_species_f = np.zeros(capacity, dtype=np.float64)
    z_species_i = np.full(capacity, -1, dtype=np.int64)
    return A3PackedState(
        schema_version=SCHEMA_VERSION,
        capacity=capacity,
        genome_capacity=genomes,
        pools=np.zeros(POOL_COUNT, dtype=np.float64),
        membrane=np.zeros(MEMBRANE_SEGMENTS, dtype=np.float64),
        membrane_oxidation=np.zeros(MEMBRANE_SEGMENTS, dtype=np.float64),
        transporters=np.zeros((MEMBRANE_SEGMENTS, CHANNEL_COUNT), dtype=np.float64),
        contact_trace=np.zeros((MEMBRANE_SEGMENTS, 2), dtype=np.float64),
        alt_contact_trace=np.zeros(MEMBRANE_SEGMENTS, dtype=np.float64),
        damage_trace=np.zeros(MEMBRANE_SEGMENTS, dtype=np.float64),
        radius=float(s5.BASE_RADIUS),
        alive=True,
        genome_lengths=np.zeros(genomes, dtype=np.int64),
        genome_lesions=np.zeros(genomes, dtype=np.float64),
        genome_mask=np.zeros(genomes, dtype=bool),
        genome_count=0,
        genome_lesion_count=0,
        replication_template_length=0,
        replication_copy_length=0,
        replication_template_lesion=0.0,
        replication_active=False,
        genome_hydrolysis_hazard=np.zeros(genomes, dtype=np.float64),
        species_fingerprint=z_species_i.copy(),
        species_mask=np.zeros(capacity, dtype=bool),
        species_count=0,
        active_mass=z_species_f.copy(),
        damaged_mass=z_species_f.copy(),
        aggregate_mass=z_species_f.copy(),
        aggregate_unresolved=0.0,
        active_order=z_species_i.copy(),
        active_count=0,
        damaged_order=z_species_i.copy(),
        damaged_count=0,
        aggregate_order=z_species_i.copy(),
        aggregate_count=0,
        gene_order=z_species_i.copy(),
        gene_count=0,
        gene_role=z_species_i.copy(),
        gene_parameter=z_species_i.copy(),
        gene_reaction=z_species_i.copy(),
        gene_promoter=z_species_f.copy(),
        gene_efficiency=z_species_f.copy(),
        gene_copy_number=np.zeros(capacity, dtype=np.int64),
        gene_localisation=z_species_i.copy(),
        current_stress=0.0,
        division_progress=0.0,
        total_genome_symbols=0,
        last_catalysis=0.0,
        last_translation=0.0,
        last_generic_flux=np.zeros(GENERIC_REACTION_COUNT, dtype=np.float64),
        novel_path_flux=0.0,
        last_assembly=0.0,
        maintenance_shortfall=0.0,
        last_damage_generated=0.0,
        last_repair_atp=0.0,
        last_repair_flux=np.zeros(REPAIR_COUNT, dtype=np.float64),
        cumulative_damage_generated=0.0,
        cumulative_repair_atp=0.0,
        cumulative_recycled_damage=0.0,
        cumulative_oxidant_neutralised=0.0,
        cumulative_genome_repairs=0.0,
        cumulative_segregation_atp=0.0,
        last_quiescence=0.0,
        last_segregation_strength=0.0,
        supplemental_atp_spent=0.0,
        reaction_events=0,
    )


def empty_a3_state(config=None):
    """Return a valid empty synthetic state for focused kernel tests."""
    return _zero_state(config)


def make_synthetic_a3_state(config=None):
    """Return a materially consistent, non-biological smoke-test state."""
    out = _zero_state(config)
    out.pools[:] = np.asarray((0.44, 0.42, 0.52, 0.12, 0.0, 0.08, 0.02,
                              0.16, 0.04, 0.29, 0.0, 0.0, 0.01), dtype=np.float64)
    out.membrane[:] = float(g2.INITIAL_MEMBRANE_MASS) / MEMBRANE_SEGMENTS
    out.genome_count = 1
    out.genome_lesion_count = 1
    out.genome_mask[0] = True
    out.genome_lengths[0] = int(g2.MIN_GENOME_LENGTH)
    out.total_genome_symbols = int(g2.MIN_GENOME_LENGTH)
    return validate_a3_state(out)


def _require_capacity(observed, capacity, label):
    if int(observed) > int(capacity):
        raise A3CapacityError('%s overflow: observed=%d capacity=%d' % (label, observed, capacity))


class FullFidelityA3Adapter:
    """Fail-closed adapter between one frozen CPU cell and ``A3PackedState``."""

    def __init__(self, config=None):
        self.config = config if isinstance(config, GPU068A3Config) else GPU068A3Config.from_state(config or {})

    def validate_cell(self, cell):
        # pack_cell is read-only and performs all representability checks.
        self.pack_cell(cell)
        return True

    def pack_cell(self, cell, model_config=None):
        if not hasattr(cell, 'pools'):
            raise A3SchemaError('cell lacks material pools')
        pools = np.asarray(cell.pools, dtype=np.float64)
        if pools.shape != (POOL_COUNT,):
            raise A3SchemaError('expected %d pools, got %r' % (POOL_COUNT, pools.shape))
        if not np.isfinite(pools).all() or np.any(pools < -self.config.ledger_atol):
            raise A3SchemaError('cell pools contain negative/non-finite values')

        active_items = _ordered_mapping_items(
            getattr(cell, 'proteins', {}), 'proteins', ACTIVE_MASS_THRESHOLD,
        )
        damaged_items = _ordered_mapping_items(
            getattr(cell, 'damaged_proteins', {}), 'damaged_proteins',
            DAMAGED_MASS_THRESHOLD,
        )
        gene_items = _ordered_mapping_items(
            OrderedDict((key, 0.0) for key in getattr(cell, 'gene_specs', {}).keys()),
            'gene_specs',
        )
        aggregate_state = aggregate_composition_state(cell)
        aggregate_items = list(aggregate_state.items)

        fingerprints = []
        slot_by_fingerprint = {}
        for collection in (active_items, damaged_items, gene_items, aggregate_items):
            for fingerprint, _ in collection:
                if fingerprint not in slot_by_fingerprint:
                    slot_by_fingerprint[fingerprint] = len(fingerprints)
                    fingerprints.append(fingerprint)
        _require_capacity(len(fingerprints), self.config.max_protein_species, 'protein species union')
        _require_capacity(len(active_items), self.config.max_protein_species, 'active protein order')
        _require_capacity(len(damaged_items), self.config.max_protein_species, 'damaged protein order')
        _require_capacity(len(gene_items), self.config.max_protein_species, 'gene order')

        genomes = list(getattr(cell, 'genomes', ()))
        _require_capacity(len(genomes), self.config.max_genome_copies, 'complete genomes')
        for index, genome in enumerate(genomes):
            _require_capacity(len(genome), self.config.max_genome_symbols, 'genome[%d] symbols' % index)
        template = getattr(cell, 'replication_template', None)
        replication_copy = getattr(cell, 'replication_copy', ())
        if template is not None:
            _require_capacity(len(template), self.config.max_genome_symbols, 'replication template symbols')
        _require_capacity(len(replication_copy), self.config.max_genome_symbols, 'replication copy symbols')

        lesions = list(getattr(cell, 'genome_lesions', ()))
        if len(lesions) > len(genomes):
            raise A3SchemaError('genome_lesions cannot exceed complete genome count')
        if any((not math.isfinite(float(value)) or float(value) < 0.0) for value in lesions):
            raise A3SchemaError('invalid genome lesion')

        active_total = sum(value for _, value in active_items)
        damaged_total = sum(value for _, value in damaged_items)
        atol = float(self.config.ledger_atol)
        if self.config.strict_material_ledger:
            if not math.isclose(active_total, float(pools[POOL_CATALYST]), rel_tol=0.0, abs_tol=atol):
                raise A3SchemaError('protein dictionary != POOL_CATALYST')
            if not math.isclose(damaged_total, float(pools[POOL_DAMAGED_PROTEIN]), rel_tol=0.0, abs_tol=atol):
                raise A3SchemaError('damaged dictionary != POOL_DAMAGED_PROTEIN')
            if not math.isclose(aggregate_state.total, float(pools[POOL_AGGREGATE]), rel_tol=0.0, abs_tol=atol):
                raise A3SchemaError('aggregate composition != POOL_AGGREGATE')

        out = _zero_state(self.config)
        out.pools[:] = pools
        for name, expected in (
            ('membrane', (MEMBRANE_SEGMENTS,)),
            ('membrane_oxidation', (MEMBRANE_SEGMENTS,)),
            ('transporters', (MEMBRANE_SEGMENTS, CHANNEL_COUNT)),
            ('contact_trace', (MEMBRANE_SEGMENTS, 2)),
            ('alt_contact_trace', (MEMBRANE_SEGMENTS,)),
            ('damage_trace', (MEMBRANE_SEGMENTS,)),
        ):
            if not hasattr(cell, name):
                raise A3SchemaError('cell lacks %s' % name)
            value = np.asarray(getattr(cell, name), dtype=np.float64)
            if value.shape != expected or not np.isfinite(value).all():
                raise A3SchemaError('invalid %s shape/finiteness' % name)
            setattr(out, name, value.copy())
        if (np.any(out.membrane < -atol) or np.any(out.membrane_oxidation < -atol)
                or np.any(out.transporters < -atol)):
            raise A3SchemaError('negative surface state')

        out.radius = float(cell.radius)
        out.alive = bool(cell.alive)
        out.species_count = len(fingerprints)
        for slot, fingerprint in enumerate(fingerprints):
            out.species_fingerprint[slot] = int(fingerprint)
            out.species_mask[slot] = True
        for position, (fingerprint, amount) in enumerate(active_items):
            slot = slot_by_fingerprint[fingerprint]
            out.active_order[position] = slot
            out.active_mass[slot] = amount
        out.active_count = len(active_items)
        for position, (fingerprint, amount) in enumerate(damaged_items):
            slot = slot_by_fingerprint[fingerprint]
            out.damaged_order[position] = slot
            out.damaged_mass[slot] = amount
        out.damaged_count = len(damaged_items)
        for fingerprint, amount in aggregate_items:
            slot = slot_by_fingerprint[fingerprint]
            out.aggregate_order[out.aggregate_count] = slot
            out.aggregate_count += 1
            out.aggregate_mass[slot] = amount
        out.aggregate_unresolved = float(aggregate_state.unresolved)

        specs = getattr(cell, 'gene_specs', {})
        for position, (fingerprint, _) in enumerate(gene_items):
            slot = slot_by_fingerprint[fingerprint]
            spec = specs[fingerprint]
            out.gene_order[position] = slot
            out.gene_role[slot] = int(spec['role'])
            out.gene_parameter[slot] = int(spec.get('parameter', 0))
            out.gene_reaction[slot] = int(spec.get('reaction', -1))
            out.gene_promoter[slot] = float(spec.get('promoter', 0.0))
            out.gene_efficiency[slot] = float(spec.get('efficiency', 0.0))
            out.gene_copy_number[slot] = int(spec.get('copy_number', 1))
            out.gene_localisation[slot] = int(spec.get('localisation', -1))
        out.gene_count = len(gene_items)

        out.genome_count = len(genomes)
        for index, genome in enumerate(genomes):
            out.genome_mask[index] = True
            out.genome_lengths[index] = len(genome)
            out.genome_lesions[index] = float(lesions[index]) if index < len(lesions) else 0.0
        out.genome_lesion_count = len(lesions)
        out.total_genome_symbols = int(sum(len(genome) for genome in genomes))
        out.replication_template_length = 0 if template is None else len(template)
        out.replication_copy_length = len(replication_copy)
        out.replication_template_lesion = float(getattr(cell, 'replication_template_lesion', 0.0))
        out.replication_active = template is not None

        scalar_defaults = {
            'current_stress': 0.0, 'division_progress': 0.0,
            'last_catalysis': 0.0, 'last_translation': 0.0,
            'novel_path_flux': 0.0, 'last_assembly': 0.0,
            'maintenance_shortfall': 0.0, 'last_damage_generated': 0.0,
            'last_repair_atp': 0.0, 'cumulative_damage_generated': 0.0,
            'cumulative_repair_atp': 0.0, 'cumulative_recycled_damage': 0.0,
            'cumulative_oxidant_neutralised': 0.0, 'cumulative_genome_repairs': 0.0,
            'cumulative_segregation_atp': 0.0, 'last_quiescence': 0.0,
            'last_segregation_strength': 0.0,
        }
        for name, default in scalar_defaults.items():
            setattr(out, name, float(getattr(cell, name, default)))
        generic_flux = np.asarray(getattr(cell, 'last_generic_flux', np.zeros(GENERIC_REACTION_COUNT)), dtype=np.float64)
        repair_flux = np.asarray(getattr(cell, 'last_repair_flux', np.zeros(REPAIR_COUNT)), dtype=np.float64)
        if generic_flux.shape != (GENERIC_REACTION_COUNT,) or repair_flux.shape != (REPAIR_COUNT,):
            raise A3SchemaError('invalid telemetry vector shape')
        out.last_generic_flux = generic_flux.copy()
        out.last_repair_flux = repair_flux.copy()
        out.reaction_events = int(getattr(cell, 'reaction_events', 0))

        for name, default in (
            ('gene_expression', True), ('generic_reactions', True),
            ('membrane_synthesis', True), ('targeted_repair', True),
            ('external_translator', False), ('protein_repair', True),
            ('genome_repair', True), ('membrane_repair', True),
            ('endogenous_damage', True), ('damage_segregation', True),
            ('forced_symmetric_damage', False), ('quiescence', True),
        ):
            setattr(out, name, bool(_model_value(model_config, name, default)))
        out.transport_enabled = bool(_model_value(model_config, 'transport', True))
        out.damage_rate_scale = float(_model_value(model_config, 'damage_rate_scale', 1.0))
        out.repair_cost_scale = float(_model_value(model_config, 'repair_cost_scale', 1.0))
        return validate_a3_state(out, self.config)

    def unpack_cell(self, state, cell):
        """Atomically apply a validated result to a CPU cell."""
        validate_a3_state(state, self.config)
        source = state.to_numpy()
        current_lengths = tuple(len(genome) for genome in getattr(cell, 'genomes', ()))
        packed_lengths = tuple(int(source.genome_lengths[index]) for index in range(source.genome_count))
        if current_lengths != packed_lengths:
            raise A3SchemaError('stale packed state: complete genome lengths changed')
        if ((getattr(cell, 'replication_template', None) is not None) != bool(source.replication_active)
                or (0 if getattr(cell, 'replication_template', None) is None else len(cell.replication_template))
                != int(source.replication_template_length)
                or len(getattr(cell, 'replication_copy', ())) != int(source.replication_copy_length)):
            raise A3SchemaError('stale packed state: replication state changed')

        fingerprints = [int(source.species_fingerprint[index]) for index in range(source.species_count)]
        active = OrderedDict()
        damaged = OrderedDict()
        for position in range(source.active_count):
            slot = int(source.active_order[position])
            amount = float(source.active_mass[slot])
            if amount > ACTIVE_MASS_THRESHOLD:
                active[fingerprints[slot]] = amount
        for position in range(source.damaged_count):
            slot = int(source.damaged_order[position])
            amount = float(source.damaged_mass[slot])
            if amount > DAMAGED_MASS_THRESHOLD:
                damaged[fingerprints[slot]] = amount
        aggregate_items = tuple(
            (fingerprints[int(source.aggregate_order[position])],
             float(source.aggregate_mass[int(source.aggregate_order[position])]))
            for position in range(source.aggregate_count)
            if (float(source.aggregate_mass[int(source.aggregate_order[position])])
                > AGGREGATE_MASS_THRESHOLD)
        )
        aggregate = AggregateCompositionState(aggregate_items, float(source.aggregate_unresolved))

        # All conversions and checks above complete before the first assignment.
        cell.pools = source.pools.copy()
        cell.membrane = source.membrane.copy()
        cell.membrane_oxidation = source.membrane_oxidation.copy()
        cell.transporters = source.transporters.copy()
        cell.contact_trace = source.contact_trace.copy()
        cell.alt_contact_trace = source.alt_contact_trace.copy()
        cell.damage_trace = source.damage_trace.copy()
        cell.radius = float(source.radius)
        cell.alive = bool(source.alive)
        cell.proteins = dict(active)
        cell.damaged_proteins = dict(damaged)
        cell.genome_lesions = [
            float(source.genome_lesions[index])
            for index in range(source.genome_lesion_count)
        ]
        cell.replication_template_lesion = float(source.replication_template_lesion)
        for name in (
            'current_stress', 'division_progress', 'last_catalysis',
            'last_translation', 'novel_path_flux', 'last_assembly',
            'maintenance_shortfall',
            'last_damage_generated', 'last_repair_atp',
            'cumulative_damage_generated', 'cumulative_repair_atp',
            'cumulative_recycled_damage', 'cumulative_oxidant_neutralised',
            'cumulative_genome_repairs', 'cumulative_segregation_atp',
            'last_quiescence', 'last_segregation_strength',
        ):
            setattr(cell, name, float(getattr(source, name)))
        cell.last_generic_flux = source.last_generic_flux.copy()
        cell.last_repair_flux = source.last_repair_flux.copy()
        cell.reaction_events = int(source.reaction_events)
        cell._soma068a3_aggregate_composition = OrderedDict(aggregate.items)
        cell._soma068a3_aggregate_unresolved = float(aggregate.unresolved)
        return cell


def _packed_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise A3SchemaError('%s must be an integer scalar' % name)
    return int(value)


def validate_a3_state(state, config=None):
    """Validate the complete lossless packed representation, including tails."""
    if not isinstance(state, A3PackedState):
        raise A3SchemaError('expected A3PackedState')
    config = config if isinstance(config, GPU068A3Config) else GPU068A3Config.from_state(config or {})
    if state.schema_version != SCHEMA_VERSION:
        raise A3SchemaError('schema version mismatch')

    integer_scalars = (
        'capacity', 'genome_capacity', 'species_count', 'active_count',
        'damaged_count', 'aggregate_count', 'gene_count', 'genome_count',
        'genome_lesion_count', 'replication_template_length',
        'replication_copy_length', 'total_genome_symbols', 'reaction_events',
    )
    integers = {name: _packed_integer(getattr(state, name), name)
                for name in integer_scalars}
    capacity = integers['capacity']
    genome_capacity = integers['genome_capacity']
    if capacity <= 0 or genome_capacity <= 0:
        raise A3CapacityError('packed capacities must be positive')
    if capacity > int(config.max_protein_species):
        raise A3CapacityError('state protein capacity exceeds configured capacity')
    if genome_capacity > int(config.max_genome_copies):
        raise A3CapacityError('state genome capacity exceeds configured capacity')

    float_fields = (
        'pools', 'membrane', 'membrane_oxidation', 'transporters',
        'contact_trace', 'alt_contact_trace', 'damage_trace',
        'genome_lesions', 'genome_hydrolysis_hazard', 'active_mass',
        'damaged_mass', 'aggregate_mass', 'gene_promoter',
        'gene_efficiency', 'last_generic_flux', 'last_repair_flux',
    )
    integer_fields = (
        'species_fingerprint', 'active_order', 'damaged_order',
        'aggregate_order', 'gene_order', 'gene_role', 'gene_parameter',
        'gene_reaction', 'gene_copy_number', 'gene_localisation',
        'genome_lengths',
    )
    bool_fields = ('species_mask', 'genome_mask')
    if set(float_fields + integer_fields + bool_fields) != set(_ARRAY_FIELDS):
        raise A3SchemaError('internal packed array classification is incomplete')

    pools = state.pools
    torch_backend = _is_tensor(pools)
    numpy_backend = isinstance(pools, np.ndarray)
    if not torch_backend and not numpy_backend:
        raise A3SchemaError('pools must be a NumPy array or Torch tensor')
    if torch_backend:
        if pools.dtype not in (torch.float64, torch.float32):
            raise A3SchemaError('Torch packed floats must be float64 or float32')
        float_dtype = pools.dtype
        device = pools.device
        for name in _ARRAY_FIELDS:
            value = getattr(state, name)
            if not _is_tensor(value):
                raise A3SchemaError('%s uses a mixed non-Torch backend' % name)
            if value.device != device:
                raise A3SchemaError('%s is on a different Torch device' % name)
            expected_dtype = (float_dtype if name in float_fields else
                              torch.int64 if name in integer_fields else torch.bool)
            if value.dtype != expected_dtype:
                raise A3SchemaError('%s has a non-canonical Torch dtype' % name)
        for name in _SCALAR_FLOAT_FIELDS:
            value = getattr(state, name)
            if (not _is_tensor(value) or value.ndim != 0 or
                    value.device != device or value.dtype != float_dtype):
                raise A3SchemaError('%s must be a same-device/dtype Torch scalar' % name)
    else:
        if pools.dtype not in (np.dtype(np.float64), np.dtype(np.float32)):
            raise A3SchemaError('NumPy packed floats must be float64 or float32')
        float_dtype = pools.dtype
        for name in _ARRAY_FIELDS:
            value = getattr(state, name)
            if not isinstance(value, np.ndarray):
                raise A3SchemaError('%s uses a mixed non-NumPy backend' % name)
            expected_dtype = (float_dtype if name in float_fields else
                              np.dtype(np.int64) if name in integer_fields else np.dtype(bool))
            if value.dtype != expected_dtype:
                raise A3SchemaError('%s has a non-canonical NumPy dtype' % name)
        for name in _SCALAR_FLOAT_FIELDS:
            if _is_tensor(getattr(state, name)):
                raise A3SchemaError('%s mixes a Torch scalar into NumPy state' % name)

    expected_shapes = {
        'pools': (POOL_COUNT,), 'membrane': (MEMBRANE_SEGMENTS,),
        'membrane_oxidation': (MEMBRANE_SEGMENTS,),
        'transporters': (MEMBRANE_SEGMENTS, CHANNEL_COUNT),
        'contact_trace': (MEMBRANE_SEGMENTS, 2),
        'alt_contact_trace': (MEMBRANE_SEGMENTS,),
        'damage_trace': (MEMBRANE_SEGMENTS,),
        'genome_lengths': (genome_capacity,),
        'genome_lesions': (genome_capacity,),
        'genome_mask': (genome_capacity,),
        'genome_hydrolysis_hazard': (genome_capacity,),
        'last_generic_flux': (GENERIC_REACTION_COUNT,),
        'last_repair_flux': (REPAIR_COUNT,),
    }
    for name in (
        'species_fingerprint', 'species_mask', 'active_mass',
        'damaged_mass', 'aggregate_mass', 'active_order', 'damaged_order',
        'aggregate_order', 'gene_order', 'gene_role', 'gene_parameter',
        'gene_reaction', 'gene_promoter', 'gene_efficiency',
        'gene_copy_number', 'gene_localisation',
    ):
        expected_shapes[name] = (capacity,)
    for name, expected in expected_shapes.items():
        value = getattr(state, name)
        if _shape(value) != tuple(expected):
            raise A3SchemaError('%s shape %r != %r' % (name, _shape(value), expected))
        if name not in bool_fields and not _finite_array(value):
            raise A3SchemaError('%s contains non-finite values' % name)

    for name in (
        'alive', 'replication_active', 'gene_expression', 'generic_reactions',
        'membrane_synthesis', 'transport_enabled', 'targeted_repair',
        'external_translator', 'protein_repair', 'genome_repair',
        'membrane_repair', 'endogenous_damage', 'damage_segregation',
        'forced_symmetric_damage', 'quiescence',
    ):
        if not isinstance(getattr(state, name), (bool, np.bool_)):
            raise A3SchemaError('%s must be a boolean scalar' % name)
    for name in ('damage_rate_scale', 'repair_cost_scale'):
        value = float(getattr(state, name))
        if not math.isfinite(value) or value < 0.0:
            raise A3SchemaError('%s must be finite and nonnegative' % name)
    for name in _SCALAR_FLOAT_FIELDS:
        value = float(_number(getattr(state, name)))
        if not math.isfinite(value):
            raise A3SchemaError('%s is non-finite' % name)
    if float(_number(state.radius)) <= 0.0:
        raise A3SchemaError('radius must be positive')
    if float(_number(state.aggregate_unresolved)) < 0.0:
        raise A3SchemaError('aggregate_unresolved must be nonnegative')
    if float(_number(state.replication_template_lesion)) < 0.0:
        raise A3SchemaError('replication_template_lesion must be nonnegative')
    if integers['reaction_events'] < 0:
        raise A3SchemaError('reaction_events must be nonnegative')

    for count_name, limit in (
        ('species_count', capacity), ('active_count', capacity),
        ('damaged_count', capacity), ('aggregate_count', capacity),
        ('gene_count', capacity), ('genome_count', genome_capacity),
        ('genome_lesion_count', genome_capacity),
    ):
        count = integers[count_name]
        if count < 0 or count > int(limit):
            raise A3CapacityError('%s out of range' % count_name)
    species_count = integers['species_count']
    genome_count = integers['genome_count']
    genome_lesion_count = integers['genome_lesion_count']
    if genome_lesion_count > genome_count:
        raise A3SchemaError('genome_lesion_count exceeds genome_count')

    for slot in range(capacity):
        expected = slot < species_count
        if bool(_number(state.species_mask[slot])) != expected:
            raise A3SchemaError('species_mask is not the exact count prefix')
    for index in range(genome_capacity):
        expected = index < genome_count
        if bool(_number(state.genome_mask[index])) != expected:
            raise A3SchemaError('genome_mask is not the exact count prefix')

    fingerprints = []
    for slot in range(species_count):
        fingerprint = int(_number(state.species_fingerprint[slot]))
        if fingerprint in fingerprints:
            raise A3SchemaError('invalid/duplicate species fingerprint')
        fingerprints.append(fingerprint)
    for slot in range(species_count, capacity):
        if int(_number(state.species_fingerprint[slot])) != -1:
            raise A3SchemaError('unused species_fingerprint must be -1')

    order_sets = {}
    for order_name, count_name in (
        ('active_order', 'active_count'), ('damaged_order', 'damaged_count'),
        ('aggregate_order', 'aggregate_count'), ('gene_order', 'gene_count'),
    ):
        count = integers[count_name]
        order = getattr(state, order_name)
        seen = set()
        for position in range(count):
            slot = int(_number(order[position]))
            if slot < 0 or slot >= species_count or slot in seen:
                raise A3SchemaError('%s contains invalid/duplicate slot' % order_name)
            seen.add(slot)
        for position in range(count, capacity):
            if int(_number(order[position])) != -1:
                raise A3SchemaError('%s unused tail must be -1' % order_name)
        order_sets[order_name] = seen

    for mass_name, order_name, threshold in (
        ('active_mass', 'active_order', ACTIVE_MASS_THRESHOLD),
        ('damaged_mass', 'damaged_order', DAMAGED_MASS_THRESHOLD),
        ('aggregate_mass', 'aggregate_order', AGGREGATE_MASS_THRESHOLD),
    ):
        mass = getattr(state, mass_name)
        ordered = order_sets[order_name]
        nonzero = set()
        for slot in range(species_count):
            amount = float(_number(mass[slot]))
            if amount != 0.0:
                nonzero.add(slot)
            if slot in ordered and amount <= float(threshold):
                raise A3SchemaError('%s ordered mass is at/below unpack threshold' % mass_name)
        if nonzero != ordered:
            raise A3SchemaError('%s nonzero slots are not exactly covered by %s' %
                                (mass_name, order_name))
        for slot in range(species_count, capacity):
            if float(_number(mass[slot])) != 0.0:
                raise A3SchemaError('%s outside species_count must be zero' % mass_name)

    gene_slots = order_sets['gene_order']
    for slot in range(capacity):
        role = int(_number(state.gene_role[slot]))
        parameter = int(_number(state.gene_parameter[slot]))
        reaction = int(_number(state.gene_reaction[slot]))
        promoter = float(_number(state.gene_promoter[slot]))
        efficiency = float(_number(state.gene_efficiency[slot]))
        copy_number = int(_number(state.gene_copy_number[slot]))
        localisation = int(_number(state.gene_localisation[slot]))
        if slot not in gene_slots:
            if ((role, parameter, reaction, copy_number, localisation) != (-1, -1, -1, 0, -1)
                    or promoter != 0.0 or efficiency != 0.0):
                raise A3SchemaError('non-gene slot contains non-default gene metadata')
            continue
        if role < ROLE_ENERGY or role > ROLE_REGULATOR:
            raise A3SchemaError('gene_role is outside the frozen grammar')
        if parameter < 0 or parameter > 7:
            raise A3SchemaError('gene_parameter is outside the frozen grammar')
        if role == ROLE_GENERIC:
            if reaction < 0 or reaction >= GENERIC_REACTION_COUNT:
                raise A3SchemaError('generic gene reaction is invalid')
        elif reaction != -1:
            raise A3SchemaError('non-generic gene reaction must be -1')
        if promoter < 0.18 or promoter > 1.40:
            raise A3SchemaError('gene_promoter is outside the frozen grammar')
        if efficiency < 0.52 or efficiency > 1.48:
            raise A3SchemaError('gene_efficiency is outside the frozen grammar')
        if copy_number < 1:
            raise A3SchemaError('gene_copy_number must be positive')
        if localisation < 0 or localisation > 3:
            raise A3SchemaError('gene_localisation is outside the frozen grammar')

    represented_species = (
        order_sets['active_order'] | order_sets['damaged_order'] |
        order_sets['aggregate_order'] | gene_slots
    )
    if represented_species != set(range(species_count)):
        raise A3SchemaError('species union contains an unrepresented or missing slot')

    genome_symbol_total = 0
    for index in range(genome_capacity):
        length = int(_number(state.genome_lengths[index]))
        lesion = float(_number(state.genome_lesions[index]))
        hazard = float(_number(state.genome_hydrolysis_hazard[index]))
        if index < genome_count:
            if length <= 0 or length > int(config.max_genome_symbols):
                raise A3CapacityError('genome_lengths[%d] exceeds configured capacity' % index)
            genome_symbol_total += length
        elif length != 0:
            raise A3SchemaError('genome_lengths outside genome_count must be zero')
        if index >= genome_lesion_count and lesion != 0.0:
            raise A3SchemaError('genome_lesions outside genome_lesion_count must be zero')
        if index >= genome_count and hazard != 0.0:
            raise A3SchemaError('genome hazard outside genome_count must be zero')
        if lesion < 0.0:
            raise A3SchemaError('genome_lesions contains negative values')
        if hazard < 0.0 or hazard > 1.0:
            raise A3SchemaError('genome hydrolysis hazard must be a probability')
    if integers['total_genome_symbols'] != genome_symbol_total:
        raise A3SchemaError('total_genome_symbols does not equal complete genome lengths')

    template_length = integers['replication_template_length']
    copy_length = integers['replication_copy_length']
    template_lesion = float(_number(state.replication_template_lesion))
    if bool(state.replication_active):
        if (template_length <= 0 or template_length > int(config.max_genome_symbols)
                or copy_length < 0 or copy_length > template_length):
            raise A3SchemaError('active replication lengths are inconsistent')
    elif template_length != 0 or copy_length != 0 or template_lesion != 0.0:
        raise A3SchemaError('inactive replication must have zero lengths and lesion')

    for name in ('pools', 'membrane', 'membrane_oxidation', 'transporters'):
        value = getattr(state, name)
        if _number(value.min()) < -float(config.ledger_atol):
            raise A3SchemaError('%s contains negative values' % name)

    active_total = float(_number(_ordered_total(
        state.active_mass, state.active_order, state.active_count,
    )))
    damaged_total = float(_number(_ordered_total(
        state.damaged_mass, state.damaged_order, state.damaged_count,
    )))
    aggregate_total = float(_number(_ordered_total(
        state.aggregate_mass, state.aggregate_order, state.aggregate_count,
    ))) + float(_number(state.aggregate_unresolved))
    atol = float(config.ledger_atol)
    # fp32 is a separately labelled performance/discrepancy candidate.  Its
    # packed reductions can differ by several ulps from a scalar pool rounded
    # once.  Keep strict fp64 tolerances while allowing only bounded fp32
    # representation error after the fp64 gate.
    if ((torch_backend and float_dtype == torch.float32) or
            (numpy_backend and float_dtype == np.dtype(np.float32))):
        atol = max(atol, 5e-6)
    if config.strict_material_ledger:
        if not math.isclose(active_total, float(_number(state.pools[POOL_CATALYST])),
                            rel_tol=0.0, abs_tol=atol):
            raise A3SchemaError('active mass ledger mismatch')
        if not math.isclose(damaged_total, float(_number(state.pools[POOL_DAMAGED_PROTEIN])),
                            rel_tol=0.0, abs_tol=atol):
            raise A3SchemaError('damaged mass ledger mismatch')
        if not math.isclose(aggregate_total, float(_number(state.pools[POOL_AGGREGATE])),
                            rel_tol=0.0, abs_tol=atol):
            raise A3SchemaError('aggregate mass ledger mismatch')
    return state


def pack_a3_cell(cell, config=None, model_config=None):
    return FullFidelityA3Adapter(config).pack_cell(cell, model_config=model_config)


def unpack_a3_cell(state, cell, config=None):
    return FullFidelityA3Adapter(config).unpack_cell(state, cell)


# ---------------------------------------------------------------------------
# Shared scalar physiology.  Reductions explicitly follow packed order.
# ---------------------------------------------------------------------------


def _scalar(state, value):
    if _is_tensor(state.pools):
        if _is_tensor(value):
            return value.to(dtype=state.pools.dtype, device=state.pools.device)
        return state.pools.new_tensor(float(value))
    return np.float64(_number(value))


def _scalar_min(state, *values):
    result = _scalar(state, values[0])
    for value in values[1:]:
        result = _minimum(result, _scalar(state, value))
    return result


def _scalar_max(state, *values):
    result = _scalar(state, values[0])
    for value in values[1:]:
        result = _maximum(result, _scalar(state, value))
    return result


def _genome_mass(state):
    symbols = int(state.total_genome_symbols) + int(state.replication_copy_length)
    return _scalar(state, symbols * float(g2.MONOMER_MASS))


def _closure_array(state):
    if _is_tensor(state.pools):
        return a2._closure_torch_exact(state.membrane, state.membrane_oxidation, state.radius)
    return a2._closure_numpy_exact(state.membrane, state.membrane_oxidation, state.radius)


def _closure(state):
    values = _closure_array(state)
    return _sum_fixed(values) / float(MEMBRANE_SEGMENTS)


def _tension(state):
    if _is_tensor(state.pools):
        return a2._tension_torch_exact(state.pools, state.membrane, _genome_mass(state))
    return np.float64(a2._tension_numpy_exact(state.pools, state.membrane, _genome_mass(state)))


def _volume(state):
    ratio = state.radius / float(s5.BASE_RADIUS)
    return _scalar_max(state, 0.20, ratio * ratio)


def _aggregate_concentration(state):
    return state.pools[POOL_AGGREGATE] / _volume(state)


def _reactive_concentration(state):
    return state.pools[POOL_REACTIVE] / _volume(state)


def _mean_genome_lesion(state):
    if int(state.genome_lesion_count) == 0:
        return _scalar(state, 1.0)
    total = _scalar(state, 0.0)
    for index in range(int(state.genome_lesion_count)):
        total = total + state.genome_lesions[index]
    return total / float(state.genome_lesion_count)


def _genome_function_factor(state):
    return 1.0 / (1.0 + 0.85 * _mean_genome_lesion(state))


def _proteostasis_factor(state):
    active = _scalar_max(state, 0.0, state.pools[POOL_CATALYST])
    damaged = state.pools[POOL_DAMAGED_PROTEIN]
    aggregate = state.pools[POOL_AGGREGATE]
    functional = active / _scalar_max(state, 1e-9, active + damaged + aggregate)
    toxicity = 1.0 / (1.0 + 3.6 * _aggregate_concentration(state))
    return _clip(functional * toxicity, 0.02, 1.0)


def _damage_burden(state):
    protein_total = _scalar_max(
        state, 0.08,
        state.pools[POOL_CATALYST] + state.pools[POOL_DAMAGED_PROTEIN] + state.pools[POOL_AGGREGATE],
    )
    protein_damage = (state.pools[POOL_DAMAGED_PROTEIN] + 1.8 * state.pools[POOL_AGGREGATE]) / protein_total
    weights = _maximum(state.membrane, 1e-9)
    weighted_oxidation = _sum_fixed(_clip(state.membrane_oxidation, 0.0, 2.5) * weights)
    membrane_damage = weighted_oxidation / _scalar_max(state, 1e-12, _sum_fixed(weights))
    burden = (0.36 * protein_damage + 0.24 * membrane_damage
              + 0.20 * _reactive_concentration(state) + 0.20 * _mean_genome_lesion(state))
    return _clip(burden, 0.0, 4.0)


def _gene_slot_exists(state, slot):
    return 0 <= int(slot) < int(state.species_count) and int(_number(state.gene_role[slot])) >= 0


def _role_activity(state, role):
    total = _scalar(state, 0.0)
    for position in range(int(state.active_count)):
        slot = int(_number(state.active_order[position]))
        if _gene_slot_exists(state, slot) and int(_number(state.gene_role[slot])) == int(role):
            total = total + state.active_mass[slot] * state.gene_efficiency[slot]
    total = total / 0.040
    if int(role) != ROLE_REGULATOR:
        total = total * _proteostasis_factor(state) * _genome_function_factor(state)
    return total


def _reaction_activity(state, reaction):
    total = _scalar(state, 0.0)
    for position in range(int(state.active_count)):
        slot = int(_number(state.active_order[position]))
        if (_gene_slot_exists(state, slot)
                and int(_number(state.gene_role[slot])) == ROLE_GENERIC
                and int(_number(state.gene_reaction[slot])) == int(reaction)):
            total = total + state.active_mass[slot] * state.gene_efficiency[slot]
    return total / 0.032


def _raw_repair_activity(state, kind):
    total = _scalar(state, 0.0)
    for position in range(int(state.active_count)):
        slot = int(_number(state.active_order[position]))
        if (_gene_slot_exists(state, slot)
                and int(_number(state.gene_role[slot])) == ROLE_REGULATOR
                and int(_number(state.gene_localisation[slot])) == LOC_REPAIR
                and int(_number(state.gene_parameter[slot])) % REPAIR_COUNT == int(kind)):
            total = total + (state.active_mass[slot] * state.gene_efficiency[slot]
                             * state.gene_promoter[slot])
    return total / 0.014 / (1.0 + 2.6 * _aggregate_concentration(state))


def _quiescence_level(state, config=None):
    if not _flag(state, config, 'quiescence'):
        return _scalar(state, 0.0)
    signal = _raw_repair_activity(state, REPAIR_QUIESCENCE)
    need = _scalar_max(state, 0.0, _damage_burden(state) - 0.12) + 0.35 * state.current_stress
    return _clip((signal / (0.8 + signal)) * need * 1.45, 0.0, 0.82)


def _protein_need(state, slot):
    role = int(_number(state.gene_role[slot]))
    if role == ROLE_ENERGY:
        return _clip((0.34 - state.pools[POOL_ATP]) * 3.0 + 0.35, 0.12, 1.8)
    if role == ROLE_MEMBRANE:
        return _clip((1.02 - _closure(state)) * 4.0 + _tension(state) * 2.2 + 0.20, 0.10, 2.0)
    if role == ROLE_TRANSPORTER:
        return _clip(
            0.35 + _scalar_max(state, 0.0, 0.25 - state.pools[POOL_FUEL])
            + _scalar_max(state, 0.0, 0.23 - state.pools[POOL_MINERAL])
            + _scalar_max(state, 0.0, 0.18 - state.pools[POOL_ALT])
            + state.pools[POOL_WASTE], 0.10, 1.8,
        )
    if role == ROLE_REPLICASE:
        return _scalar(state, 1.6 if int(state.genome_count) < 2 else 0.28)
    if role == ROLE_TRANSLATOR:
        return _clip(1.2 - _role_activity(state, ROLE_TRANSLATOR) * 0.22, 0.22, 1.2)
    if role == ROLE_NUCLEOTIDE:
        return _clip((0.22 - state.pools[POOL_NUCLEOTIDE]) * 5.0 + 0.18, 0.08, 1.7)
    if role == ROLE_GENERIC:
        reaction = int(_number(state.gene_reaction[slot]))
        reaction = max(0, min(GENERIC_REACTION_COUNT - 1, reaction))
        source = int(g2.REACTION_SOURCE[reaction])
        return _clip(0.18 + state.pools[source] * 2.2, 0.10, 1.5)
    if role != ROLE_REGULATOR:
        return _clip(0.22 + _sum_fixed(state.damage_trace) / MEMBRANE_SEGMENTS * 3.0, 0.10, 1.2)
    kind = int(_number(state.gene_parameter[slot])) % REPAIR_COUNT
    damage = _damage_burden(state)
    if kind == REPAIR_ANTIOXIDANT:
        need = 0.030 + 2.8 * _reactive_concentration(state) + 0.6 * state.current_stress
    elif kind == REPAIR_CHAPERONE:
        need = 0.025 + 3.0 * state.pools[POOL_DAMAGED_PROTEIN]
    elif kind == REPAIR_PROTEASE:
        need = 0.022 + 2.1 * state.pools[POOL_DAMAGED_PROTEIN] + 3.3 * state.pools[POOL_AGGREGATE]
    elif kind == REPAIR_GENOME:
        need = 0.022 + 1.5 * _mean_genome_lesion(state)
    elif kind == REPAIR_SEGREGATION:
        need = 0.016 + damage * (0.4 + 1.4 * state.division_progress)
    elif kind == REPAIR_PROOFREADING:
        need = 0.022 + 0.5 * _mean_genome_lesion(state) + (0.6 if state.replication_active else 0.0)
    elif kind == REPAIR_QUIESCENCE:
        need = 0.016 + 1.5 * _scalar_max(state, 0.0, damage - 0.20) + 0.7 * state.current_stress
    else:
        mean_oxidation = _sum_fixed(state.membrane_oxidation) / MEMBRANE_SEGMENTS
        need = 0.022 + 2.2 * mean_oxidation + 1.4 * (1.0 - _closure(state))
    return _clip(need, 0.012, 2.2)


def _kernel_input(state, backend):
    validate_a3_state(state)
    if backend == 'numpy':
        if _is_tensor(state.pools):
            raise TypeError('NumPy kernel requires NumPy A3PackedState')
    elif backend == 'torch':
        if torch is None or not _is_tensor(state.pools):
            raise TypeError('Torch kernel requires Torch A3PackedState')
        if not state.pools.dtype.is_floating_point:
            raise TypeError('Torch packed state must use floating point')
    else:
        raise ValueError('unknown backend')
    return state.clone()


# ---------------------------------------------------------------------------
# Pure gene-coded metabolism phases.
# ---------------------------------------------------------------------------


def _generic_metabolism(state, dt, config=None):
    state.last_generic_flux[:] = 0.0
    _assign_scalar(state, 'novel_path_flux', 0.0)
    if not _flag(state, config, 'generic_reactions'):
        return state
    dt = float(dt)
    for reaction in range(GENERIC_REACTION_COUNT):
        activity = _reaction_activity(state, reaction)
        amount = _scalar(state, 0.0)
        if _number(activity) > 1e-7:
            source = int(g2.REACTION_SOURCE[reaction])
            product = int(g2.REACTION_PRODUCT[reaction])
            available = state.pools[source]
            saturation = available / (0.045 + available)
            desired = dt * 0.030 * activity * saturation
            if reaction in (int(g2.REACTION_ALT_TO_INTERMEDIATE), int(g2.REACTION_WASTE_TO_INTERMEDIATE)):
                desired = desired / (1.0 + (state.pools[product] / 0.060) ** 3)
            delta = float(g2.REACTION_POTENTIAL[source] - g2.REACTION_POTENTIAL[product])
            if delta >= 0.0:
                amount = _scalar_min(state, available, desired)
                if _number(amount) > 0.0:
                    state.pools[source] = state.pools[source] - amount
                    state.pools[product] = state.pools[product] + amount
                    state.pools[POOL_ATP] = state.pools[POOL_ATP] + amount * delta * 0.68
            else:
                atp_cost = (-delta) * 1.28
                amount = _scalar_min(
                    state, available, desired,
                    _scalar_max(state, 0.0, state.pools[POOL_ATP] - 0.040) / max(atp_cost, 1e-9),
                )
                if _number(amount) > 0.0:
                    state.pools[source] = state.pools[source] - amount
                    state.pools[product] = state.pools[product] + amount
                    state.pools[POOL_ATP] = state.pools[POOL_ATP] - amount * atp_cost
            state.last_generic_flux[reaction] = amount / max(dt, 1e-9)
            if reaction == int(g2.REACTION_INTERMEDIATE_TO_WASTE):
                state.novel_path_flux = state.novel_path_flux + amount / max(dt, 1e-9)
            if _number(amount) > 1e-8:
                state.reaction_events += 1
    return state


def generic_metabolism_numpy(state, dt, config=None):
    return _generic_metabolism(_kernel_input(state, 'numpy'), dt, config)


def generic_metabolism_torch(state, dt, config=None):
    return _generic_metabolism(_kernel_input(state, 'torch'), dt, config)


def _precursor_synthesis(state, dt, config=None):
    dt = float(dt)
    energy_activity = _role_activity(state, ROLE_ENERGY)
    volume = _volume(state)
    fuel = state.pools[POOL_FUEL] / volume
    waste = state.pools[POOL_WASTE] / volume
    inhibition = 1.0 / (1.0 + 2.8 * waste)
    catalytic_rate = 0.17 * energy_activity * fuel / (0.14 + fuel) * inhibition
    amount = _scalar_min(state, state.pools[POOL_FUEL], catalytic_rate * dt)
    state.pools[POOL_FUEL] = state.pools[POOL_FUEL] - amount
    state.pools[POOL_WASTE] = state.pools[POOL_WASTE] + amount
    state.pools[POOL_ATP] = state.pools[POOL_ATP] + 2.45 * amount
    _assign_scalar(state, 'last_catalysis', amount / max(dt, 1e-9))

    membrane_activity = _role_activity(state, ROLE_MEMBRANE)
    need = _clip((1.03 - _closure(state)) * 2.5 + _tension(state) * 2.2
                 + _scalar_max(state, 0.0, 0.17 - state.pools[POOL_MEM_PRECURSOR]) * 1.5,
                 0.0, 2.5)
    desired = dt * 0.070 * membrane_activity * need
    amount = _scalar_min(
        state, desired, state.pools[POOL_FUEL] / 0.62,
        state.pools[POOL_MINERAL] / 0.38,
        _scalar_max(state, 0.0, state.pools[POOL_ATP] - 0.044) / 0.42,
    ) if _flag(state, config, 'membrane_synthesis') else _scalar(state, 0.0)
    state.pools[POOL_FUEL] = state.pools[POOL_FUEL] - 0.62 * amount
    state.pools[POOL_MINERAL] = state.pools[POOL_MINERAL] - 0.38 * amount
    state.pools[POOL_ATP] = state.pools[POOL_ATP] - 0.42 * amount
    state.pools[POOL_MEM_PRECURSOR] = state.pools[POOL_MEM_PRECURSOR] + amount

    transport_activity = _role_activity(state, ROLE_TRANSPORTER)
    need = _clip(
        0.35 + _scalar_max(state, 0.0, 0.22 - state.pools[POOL_FUEL])
        + _scalar_max(state, 0.0, 0.20 - state.pools[POOL_MINERAL])
        + _scalar_max(state, 0.0, 0.18 - state.pools[POOL_ALT])
        + 1.2 * state.pools[POOL_WASTE], 0.0, 1.8,
    )
    desired = dt * 0.017 * transport_activity * need
    amount = _scalar_min(
        state, desired, state.pools[POOL_FUEL] / 0.65,
        state.pools[POOL_MINERAL] / 0.35,
        _scalar_max(state, 0.0, state.pools[POOL_ATP] - 0.044) / 0.70,
    ) if _flag(state, config, 'transport_enabled') else _scalar(state, 0.0)
    state.pools[POOL_FUEL] = state.pools[POOL_FUEL] - 0.65 * amount
    state.pools[POOL_MINERAL] = state.pools[POOL_MINERAL] - 0.35 * amount
    state.pools[POOL_ATP] = state.pools[POOL_ATP] - 0.70 * amount
    state.pools[POOL_TRANSPORTER_PRECURSOR] = state.pools[POOL_TRANSPORTER_PRECURSOR] + amount

    nucleotide_activity = _role_activity(state, ROLE_NUCLEOTIDE)
    need = _clip((0.24 - state.pools[POOL_NUCLEOTIDE]) * 5.0 + 0.16, 0.0, 1.7)
    desired = dt * 0.014 * nucleotide_activity * need
    amount = _scalar_min(
        state, desired, state.pools[POOL_FUEL] / 0.58,
        state.pools[POOL_MINERAL] / 0.42,
        _scalar_max(state, 0.0, state.pools[POOL_ATP] - 0.044) / 0.66,
    )
    state.pools[POOL_FUEL] = state.pools[POOL_FUEL] - 0.58 * amount
    state.pools[POOL_MINERAL] = state.pools[POOL_MINERAL] - 0.42 * amount
    state.pools[POOL_ATP] = state.pools[POOL_ATP] - 0.66 * amount
    state.pools[POOL_NUCLEOTIDE] = state.pools[POOL_NUCLEOTIDE] + amount
    return state


def precursor_synthesis_numpy(state, dt, config=None):
    return _precursor_synthesis(_kernel_input(state, 'numpy'), dt, config)


def precursor_synthesis_torch(state, dt, config=None):
    return _precursor_synthesis(_kernel_input(state, 'torch'), dt, config)


def _maintenance(state, dt):
    maintenance = float(dt) * (
        0.0042 + 0.010 * state.pools[POOL_CATALYST]
        + 0.009 * _sum_fixed(state.transporters.reshape(-1))
        + 0.0025 * _sum_fixed(state.membrane)
        + 0.008 * _tension(state)
        + 0.000030 * int(state.total_genome_symbols)
    )
    paid = _scalar_min(state, state.pools[POOL_ATP], maintenance)
    state.pools[POOL_ATP] = state.pools[POOL_ATP] - paid
    _assign_scalar(state, 'maintenance_shortfall', _scalar_max(state, 0.0, maintenance - paid))
    return state


def maintenance_numpy(state, dt, config=None):
    return _maintenance(_kernel_input(state, 'numpy'), dt)


def maintenance_torch(state, dt, config=None):
    return _maintenance(_kernel_input(state, 'torch'), dt)


def _translation(state, dt, config=None):
    _assign_scalar(state, 'last_translation', 0.0)
    if not _flag(state, config, 'gene_expression') or int(state.genome_count) == 0:
        return state
    translator = _role_activity(state, ROLE_TRANSLATOR)
    if _flag(state, config, 'external_translator'):
        translator = translator + 0.75
    if _number(translator) <= 1e-5 or int(state.gene_count) == 0:
        return state
    weights = []
    total_weight = _scalar(state, 0.0)
    for position in range(int(state.gene_count)):
        slot = int(_number(state.gene_order[position]))
        weight = (state.gene_promoter[slot] * state.gene_copy_number[slot]
                  * _protein_need(state, slot))
        weights.append((slot, weight))
        total_weight = total_weight + weight
    if _number(total_weight) <= 0.0:
        return state
    quiescence = _quiescence_level(state, config)
    _assign_scalar(state, 'last_quiescence', quiescence)
    capacity = float(dt) * 0.0060 * translator * (1.0 - 0.72 * quiescence)
    chaperone = _raw_repair_activity(state, REPAIR_CHAPERONE) if _flag(state, config, 'protein_repair') else _scalar(state, 0.0)
    misfold_fraction = _clip(
        0.010 + 0.055 * _reactive_concentration(state) + 0.040 * state.current_stress
        + 0.020 * _aggregate_concentration(state) - 0.010 * chaperone,
        0.004, 0.42,
    )
    dt = float(dt)
    for slot, weight in weights:
        desired = capacity * weight / total_weight
        desired = _scalar_min(
            state, desired, state.pools[POOL_FUEL] / 0.64,
            state.pools[POOL_MINERAL] / 0.36,
            _scalar_max(state, 0.0, state.pools[POOL_ATP] - 0.042) / 0.52,
        )
        if _number(desired) <= 0.0:
            continue
        was_active = _number(state.active_mass[slot]) > 1e-10
        was_damaged = _number(state.damaged_mass[slot]) > 1e-11
        state.pools[POOL_FUEL] = state.pools[POOL_FUEL] - 0.64 * desired
        state.pools[POOL_MINERAL] = state.pools[POOL_MINERAL] - 0.36 * desired
        state.pools[POOL_ATP] = state.pools[POOL_ATP] - 0.52 * desired
        misfolded = desired * misfold_fraction
        active = desired - misfolded
        state.active_mass[slot] = state.active_mass[slot] + active
        state.damaged_mass[slot] = state.damaged_mass[slot] + misfolded
        if not was_active and _number(state.active_mass[slot]) > 1e-10:
            _append_order_slot(state, 'active_order', 'active_count', slot)
        if not was_damaged and _number(state.damaged_mass[slot]) > 1e-11:
            _append_order_slot(state, 'damaged_order', 'damaged_count', slot)
        state.last_translation = state.last_translation + desired / max(dt, 1e-9)
    _synchronise_material_pools(state)
    return state


def translation_numpy(state, dt, config=None):
    return _translation(_kernel_input(state, 'numpy'), dt, config)


def translation_torch(state, dt, config=None):
    return _translation(_kernel_input(state, 'torch'), dt, config)


def _surface_assembly(state, dt, config=None):
    _assign_scalar(state, 'last_assembly', 0.0)
    dt = float(dt)
    if _flag(state, config, 'membrane_synthesis') and _number(state.pools[POOL_MEM_PRECURSOR]) > 1e-9:
        required = float(g2.base.MEMBRANE_DENSITY) * (2.0 * math.pi * state.radius / MEMBRANE_SEGMENTS)
        deficit = _maximum(required * 1.06 - state.membrane, 0.0)
        if _flag(state, config, 'targeted_repair'):
            weights = deficit * 8.0 + state.damage_trace * 2.5 + 0.02
        else:
            weights = state.membrane * 0.0 + 1.0
        if _number(_sum_fixed(deficit)) < 1e-5:
            weights = weights + 0.18
        weights = _maximum(weights, 1e-8)
        weights = weights / _sum_fixed(weights)
        rate = 0.070 * _scalar_max(state, 0.05, _role_activity(state, ROLE_MEMBRANE)) * (
            0.45 + 1.4 * (1.0 - _closure(state)) + 0.6 * _tension(state)
        )
        assembly = _scalar_min(
            state, state.pools[POOL_MEM_PRECURSOR], rate * dt,
            _scalar_max(state, 0.0, state.pools[POOL_ATP] - 0.045) / 0.34,
        )
        if _number(assembly) > 0.0:
            state.membrane = state.membrane + weights * assembly
            state.pools[POOL_MEM_PRECURSOR] = state.pools[POOL_MEM_PRECURSOR] - assembly
            state.pools[POOL_ATP] = state.pools[POOL_ATP] - 0.34 * assembly
            _assign_scalar(state, 'last_assembly', assembly / max(dt, 1e-9))

    if _number(state.pools[POOL_TRANSPORTER_PRECURSOR]) > 1e-9:
        channel_need_values = (
            _scalar_max(state, 0.08, 0.34 - state.pools[POOL_FUEL]),
            _scalar_max(state, 0.08, 0.29 - state.pools[POOL_MINERAL]),
            _scalar_max(state, 0.05, state.pools[POOL_WASTE] * 1.4),
            _scalar_max(state, 0.08, 0.26 - state.pools[POOL_ALT]),
        )
        if _is_tensor(state.pools):
            channel_need = torch.stack(channel_need_values)
        else:
            channel_need = np.asarray(channel_need_values, dtype=np.float64)
        channel_need = channel_need / _sum_fixed(channel_need)
        insert = _scalar_min(
            state, state.pools[POOL_TRANSPORTER_PRECURSOR],
            0.034 * _scalar_max(state, 0.05, _role_activity(state, ROLE_TRANSPORTER)) * dt,
            _scalar_max(state, 0.0, state.pools[POOL_ATP] - 0.045) / 0.40,
        )
        if _number(insert) > 0.0:
            closure = _closure_array(state)
            for channel in range(CHANNEL_COUNT):
                if channel == int(g2.CHANNEL_FUEL):
                    local = state.contact_trace[:, 0] + 0.03
                elif channel == int(g2.CHANNEL_MINERAL):
                    local = state.contact_trace[:, 1] + 0.03
                elif channel == int(g2.CHANNEL_ALT):
                    local = state.alt_contact_trace + 0.03
                else:
                    local = closure + state.damage_trace + 0.03
                local = _maximum(local, 1e-8)
                local = local / _sum_fixed(local)
                state.transporters[:, channel] = state.transporters[:, channel] + local * insert * channel_need[channel]
            state.pools[POOL_TRANSPORTER_PRECURSOR] = state.pools[POOL_TRANSPORTER_PRECURSOR] - insert
            state.pools[POOL_ATP] = state.pools[POOL_ATP] - 0.40 * insert
    return state


def surface_assembly_numpy(state, dt, config=None):
    return _surface_assembly(_kernel_input(state, 'numpy'), dt, config)


def surface_assembly_torch(state, dt, config=None):
    return _surface_assembly(_kernel_input(state, 'torch'), dt, config)


# ---------------------------------------------------------------------------
# Damage generation, aggregation composition, and ordinary housekeeping.
# ---------------------------------------------------------------------------


def _damage_generation(state, dt, config=None):
    if not bool(state.alive):
        state.genome_hydrolysis_hazard[:] = 0.0
        _assign_scalar(state, 'last_damage_generated', 0.0)
        return state
    dt = float(dt)
    rate_scale = _rate(state, config, 'damage_rate_scale') if _flag(state, config, 'endogenous_damage') else 0.0
    volume = _volume(state)
    waste_stress = state.pools[POOL_WASTE] / volume
    reactive = _reactive_concentration(state)
    flux = state.last_catalysis + 0.35 * state.last_translation

    protein_rate = rate_scale * (
        0.00055 + 0.0020 * waste_stress + 0.0038 * reactive
        + 0.0012 * state.current_stress + 0.00045 * flux
    )
    damage_amount = _scalar_min(
        state, _scalar_max(state, 0.0, state.pools[POOL_CATALYST]),
        state.pools[POOL_CATALYST] * protein_rate * dt,
    )
    moved = _scalar(state, 0.0)
    if _number(damage_amount) > 0.0 and int(state.active_count) > 0:
        total = _scalar_max(state, 1e-12, _ordered_total(state.active_mass, state.active_order, state.active_count))
        for position in range(int(state.active_count)):
            slot = int(_number(state.active_order[position]))
            share = _scalar_min(state, state.active_mass[slot], damage_amount * state.active_mass[slot] / total)
            was_damaged = _number(state.damaged_mass[slot]) > 1e-11
            state.active_mass[slot] = state.active_mass[slot] - share
            state.damaged_mass[slot] = state.damaged_mass[slot] + share
            if not was_damaged and _number(state.damaged_mass[slot]) > 1e-11:
                _append_order_slot(state, 'damaged_order', 'damaged_count', slot)
            moved = moved + share
        _compact_order(
            state, 'active_mass', 'active_order', 'active_count',
            ACTIVE_MASS_THRESHOLD,
        )
        _synchronise_material_pools(state)

    aggregation_rate = rate_scale * (
        0.0020 + 0.0080 * reactive + 0.0040 * state.current_stress
        + 0.0030 * _aggregate_concentration(state)
    )
    aggregate_amount = _scalar_min(
        state, _scalar_max(state, 0.0, state.pools[POOL_DAMAGED_PROTEIN]),
        state.pools[POOL_DAMAGED_PROTEIN] * aggregation_rate * dt,
    )
    aggregated = _scalar(state, 0.0)
    if _number(aggregate_amount) > 0.0 and int(state.damaged_count) > 0:
        total = _scalar_max(state, 1e-12, _ordered_total(state.damaged_mass, state.damaged_order, state.damaged_count))
        for position in range(int(state.damaged_count)):
            slot = int(_number(state.damaged_order[position]))
            share = _scalar_min(state, state.damaged_mass[slot], aggregate_amount * state.damaged_mass[slot] / total)
            was_aggregate = _number(state.aggregate_mass[slot]) > 0.0
            state.damaged_mass[slot] = state.damaged_mass[slot] - share
            state.aggregate_mass[slot] = state.aggregate_mass[slot] + share
            if not was_aggregate and _number(state.aggregate_mass[slot]) > 0.0:
                _append_order_slot(state, 'aggregate_order', 'aggregate_count', slot)
            aggregated = aggregated + share
        _compact_order(
            state, 'damaged_mass', 'damaged_order', 'damaged_count',
            DAMAGED_MASS_THRESHOLD,
        )
        _synchronise_material_pools(state)

    absolute_flux = _scalar(state, 0.0)
    for reaction in range(GENERIC_REACTION_COUNT):
        absolute_flux = absolute_flux + _maximum(state.last_generic_flux[reaction], -state.last_generic_flux[reaction])
    reactive_amount = _scalar_min(
        state, state.pools[POOL_WASTE],
        rate_scale * dt * (
            0.015 * state.last_catalysis + 0.006 * absolute_flux
            + 0.0015 * state.current_stress
        ),
    )
    state.pools[POOL_WASTE] = state.pools[POOL_WASTE] - reactive_amount
    state.pools[POOL_REACTIVE] = state.pools[POOL_REACTIVE] + reactive_amount

    local_driver = (
        0.55 * state.damage_trace + 0.35 * _roll(state.damage_trace, 1)
        + 0.35 * _roll(state.damage_trace, -1) + 0.12
    )
    local_driver = local_driver / _scalar_max(
        state, 1e-9, _sum_fixed(local_driver) / MEMBRANE_SEGMENTS,
    )
    gain = rate_scale * dt * (
        0.00045 + 0.0022 * reactive + 0.0014 * state.current_stress
    ) * local_driver
    state.membrane_oxidation = _clip(state.membrane_oxidation + gain, 0.0, 3.0)

    severe = _maximum(state.membrane_oxidation - 1.25, 0.0)
    hydrolysis = _minimum(state.membrane, state.membrane * severe * 0.0014 * dt)
    if _number(_sum_fixed(hydrolysis)) > 0.0:
        state.membrane = state.membrane - hydrolysis
        state.pools[POOL_WASTE] = state.pools[POOL_WASTE] + _sum_fixed(hydrolysis)
        if _is_tensor(state.pools):
            state.membrane_oxidation = state.membrane_oxidation * (state.membrane > 1e-9).to(state.pools.dtype)
        else:
            state.membrane_oxidation = state.membrane_oxidation * np.where(state.membrane > 1e-9, 1.0, 0.0)

    # This is the frozen 0.3 ``while len(genome_lesions) < len(genomes)``
    # location.  Packing and pre-damage role activity preserve the shorter
    # legacy list; zero-extension occurs only here, immediately before gain.
    while int(state.genome_lesion_count) < int(state.genome_count):
        state.genome_lesions[int(state.genome_lesion_count)] = 0.0
        state.genome_lesion_count += 1
    lesion_gain = rate_scale * dt * (
        0.0010 + 0.0050 * reactive + 0.0024 * state.current_stress
    )
    state.genome_hydrolysis_hazard[:] = 0.0
    for index in range(int(state.genome_count)):
        state.genome_lesions[index] = state.genome_lesions[index] + lesion_gain
        if (_number(state.genome_lesions[index]) > 0.75
                and int(_number(state.genome_lengths[index])) > int(g2.MIN_GENOME_LENGTH)):
            state.genome_hydrolysis_hazard[index] = dt * 0.00065 * state.genome_lesions[index]

    total_generated = moved + aggregated + reactive_amount
    _assign_scalar(state, 'last_damage_generated', total_generated)
    state.cumulative_damage_generated = state.cumulative_damage_generated + total_generated
    _compact_species_union(state)
    _synchronise_material_pools(state)
    return state


def damage_numpy(state, dt, config=None):
    return _damage_generation(_kernel_input(state, 'numpy'), dt, config)


def damage_torch(state, dt, config=None):
    return _damage_generation(_kernel_input(state, 'torch'), dt, config)


def genome_hydrolysis_plan(state):
    """Return CPU-RNG work records without consuming random numbers."""
    validate_a3_state(state)
    records = []
    for index in range(int(state.genome_count)):
        probability = float(_number(state.genome_hydrolysis_hazard[index]))
        if probability <= 0.0:
            continue
        records.append({
            'genome_index': int(index),
            'probability': probability,
            'lesion_after_gain': float(_number(state.genome_lesions[index])),
            'genome_length': int(_number(state.genome_lengths[index])),
            'minimum_length': int(g2.MIN_GENOME_LENGTH),
            'monomer_mass': float(g2.MONOMER_MASS),
        })
    return tuple(records)


def _circular_smooth(values, amount):
    amount = max(0.0, min(0.49, float(amount)))
    return (1.0 - 2.0 * amount) * values + amount * _roll(values, 1) + amount * _roll(values, -1)


def _housekeeping(state, dt, config=None):
    dt = float(dt)
    for channel in range(CHANNEL_COUNT):
        state.transporters[:, channel] = _circular_smooth(
            state.transporters[:, channel], min(0.49, 0.10 * dt),
        )
    state.membrane = _circular_smooth(state.membrane, min(0.49, 0.075 * dt))

    waste_stress = state.pools[POOL_WASTE] / _volume(state)
    membrane_decay_rate = 0.00032 + 0.0010 * waste_stress + 0.0012 * _tension(state)
    if _number(state.maintenance_shortfall) > 0.0:
        membrane_decay_rate = membrane_decay_rate + 0.015 * state.maintenance_shortfall / max(dt, 1e-9)
    membrane_loss = _minimum(
        state.membrane,
        state.membrane * membrane_decay_rate * dt + state.damage_trace * 0.0008 * dt,
    )
    state.membrane = state.membrane - membrane_loss
    state.pools[POOL_WASTE] = state.pools[POOL_WASTE] + _sum_fixed(membrane_loss)

    transporter_loss = _minimum(
        state.transporters,
        state.transporters * (
            0.00075 + 0.0010 * waste_stress + 0.0015 * state.maintenance_shortfall
        ) * dt,
    )
    state.transporters = state.transporters - transporter_loss
    state.pools[POOL_WASTE] = state.pools[POOL_WASTE] + _sum_fixed(transporter_loss.reshape(-1))

    state.pools[POOL_ATP] = _clip(state.pools[POOL_ATP] * math.exp(-0.018 * dt), 0.0, 1.8)
    state.contact_trace = state.contact_trace * math.exp(-0.80 * dt)
    state.alt_contact_trace = state.alt_contact_trace * math.exp(-0.80 * dt)
    state.damage_trace = _clip(state.damage_trace * math.exp(-0.55 * dt), 0.0, 1.5)
    state.pools = _maximum(state.pools, 0.0)
    return state


def housekeeping_numpy(state, dt, config=None):
    return _housekeeping(_kernel_input(state, 'numpy'), dt, config)


def housekeeping_torch(state, dt, config=None):
    return _housekeeping(_kernel_input(state, 'torch'), dt, config)


# ---------------------------------------------------------------------------
# Paid repair in the exact frozen order.
# ---------------------------------------------------------------------------


def _repair(state, dt, config=None):
    dt = float(dt)
    _assign_scalar(state, 'last_repair_atp', 0.0)
    state.last_repair_flux[:] = 0.0
    cost_scale = max(0.05, _rate(state, config, 'repair_cost_scale'))

    antioxidant = _raw_repair_activity(state, REPAIR_ANTIOXIDANT) if _flag(state, config, 'protein_repair') else _scalar(state, 0.0)
    desired = dt * 0.030 * antioxidant * (
        state.pools[POOL_REACTIVE] / (0.018 + state.pools[POOL_REACTIVE])
    )
    neutralised = _scalar_min(
        state, state.pools[POOL_REACTIVE], desired,
        state.pools[POOL_FUEL] / 0.18,
        _scalar_max(state, 0.0, state.pools[POOL_ATP] - 0.025) / (0.20 * cost_scale),
    )
    if _number(neutralised) > 0.0:
        state.pools[POOL_REACTIVE] = state.pools[POOL_REACTIVE] - neutralised
        fuel_spent = 0.18 * neutralised
        state.pools[POOL_FUEL] = state.pools[POOL_FUEL] - fuel_spent
        state.pools[POOL_WASTE] = state.pools[POOL_WASTE] + neutralised + fuel_spent
        atp = 0.20 * cost_scale * neutralised
        state.pools[POOL_ATP] = state.pools[POOL_ATP] - atp
        state.last_repair_atp = state.last_repair_atp + atp
        state.last_repair_flux[REPAIR_ANTIOXIDANT] = neutralised / max(dt, 1e-9)
        state.cumulative_oxidant_neutralised = state.cumulative_oxidant_neutralised + neutralised

    if _flag(state, config, 'protein_repair'):
        chaperone = _raw_repair_activity(state, REPAIR_CHAPERONE)
        damaged_pool = state.pools[POOL_DAMAGED_PROTEIN]
        repair_amount = _scalar_min(
            state, damaged_pool,
            dt * 0.023 * chaperone * damaged_pool / (0.020 + damaged_pool),
            _scalar_max(state, 0.0, state.pools[POOL_ATP] - 0.025) / (0.72 * cost_scale),
        )
        if _number(repair_amount) > 0.0 and int(state.damaged_count) > 0:
            total = _scalar_max(state, 1e-12, _ordered_total(state.damaged_mass, state.damaged_order, state.damaged_count))
            repaired = _scalar(state, 0.0)
            for position in range(int(state.damaged_count)):
                slot = int(_number(state.damaged_order[position]))
                share = _scalar_min(state, state.damaged_mass[slot], repair_amount * state.damaged_mass[slot] / total)
                was_active = _number(state.active_mass[slot]) > 1e-10
                state.damaged_mass[slot] = state.damaged_mass[slot] - share
                state.active_mass[slot] = state.active_mass[slot] + share
                if not was_active and _number(state.active_mass[slot]) > 1e-10:
                    _append_order_slot(state, 'active_order', 'active_count', slot)
                repaired = repaired + share
            atp = 0.72 * cost_scale * repaired
            state.pools[POOL_ATP] = state.pools[POOL_ATP] - atp
            state.last_repair_atp = state.last_repair_atp + atp
            state.last_repair_flux[REPAIR_CHAPERONE] = repaired / max(dt, 1e-9)
            # Frozen 0.3 deliberately leaves the scalar protein/damage pools
            # stale between chaperone and protease.  Protease activity observes
            # the updated dictionaries, while ``substrate``/``from_damaged``
            # below observe the pre-chaperone POOL_DAMAGED_PROTEIN.  Sync only
            # once at the repair tail to preserve that exact event semantics.

        protease = _raw_repair_activity(state, REPAIR_PROTEASE)
        substrate = state.pools[POOL_DAMAGED_PROTEIN] + 0.65 * state.pools[POOL_AGGREGATE]
        recycle = _scalar_min(
            state, substrate,
            dt * 0.016 * protease * substrate / (0.025 + substrate),
            _scalar_max(state, 0.0, state.pools[POOL_ATP] - 0.025) / (0.48 * cost_scale),
        )
        recycled = _scalar(state, 0.0)
        if _number(recycle) > 0.0:
            from_damaged = _scalar_min(state, state.pools[POOL_DAMAGED_PROTEIN], recycle)
            if _number(from_damaged) > 0.0 and int(state.damaged_count) > 0:
                total = _scalar_max(state, 1e-12, _ordered_total(state.damaged_mass, state.damaged_order, state.damaged_count))
                for position in range(int(state.damaged_count)):
                    slot = int(_number(state.damaged_order[position]))
                    share = _scalar_min(state, state.damaged_mass[slot], from_damaged * state.damaged_mass[slot] / total)
                    state.damaged_mass[slot] = state.damaged_mass[slot] - share
                    recycled = recycled + share
                _compact_order(
                    state, 'damaged_mass', 'damaged_order', 'damaged_count',
                    DAMAGED_MASS_THRESHOLD,
                )
                _synchronise_material_pools(state)
            remaining = _scalar_max(state, 0.0, recycle - from_damaged)
            aggregate_before = state.pools[POOL_AGGREGATE]
            from_aggregate = _scalar_min(state, aggregate_before, remaining)
            if _number(from_aggregate) > 0.0:
                fraction_remaining = _scalar_max(
                    state, 0.0, (aggregate_before - from_aggregate)
                    / _scalar_max(state, 1e-12, aggregate_before),
                )
                for position in range(int(state.aggregate_count)):
                    slot = int(_number(state.aggregate_order[position]))
                    state.aggregate_mass[slot] = state.aggregate_mass[slot] * fraction_remaining
                state.aggregate_unresolved = state.aggregate_unresolved * fraction_remaining
                recycled = recycled + from_aggregate
                _compact_order(
                    state, 'aggregate_mass', 'aggregate_order',
                    'aggregate_count', AGGREGATE_MASS_THRESHOLD,
                )
            state.pools[POOL_FUEL] = state.pools[POOL_FUEL] + 0.58 * recycled
            state.pools[POOL_MINERAL] = state.pools[POOL_MINERAL] + 0.32 * recycled
            state.pools[POOL_WASTE] = state.pools[POOL_WASTE] + 0.10 * recycled
            atp = 0.48 * cost_scale * recycled
            state.pools[POOL_ATP] = state.pools[POOL_ATP] - atp
            state.last_repair_atp = state.last_repair_atp + atp
            state.last_repair_flux[REPAIR_PROTEASE] = recycled / max(dt, 1e-9)
            state.cumulative_recycled_damage = state.cumulative_recycled_damage + recycled
            _synchronise_material_pools(state)

    if _flag(state, config, 'genome_repair') and int(state.genome_count) > 0:
        activity = _raw_repair_activity(state, REPAIR_GENOME)
        total_lesion = _scalar(state, 0.0)
        for index in range(int(state.genome_count)):
            total_lesion = total_lesion + state.genome_lesions[index]
        repair_amount = _scalar_min(
            state, total_lesion,
            dt * 0.018 * activity * total_lesion / (0.05 + total_lesion),
            _scalar_max(state, 0.0, state.pools[POOL_ATP] - 0.025) / (0.95 * cost_scale),
        )
        if _number(repair_amount) > 0.0:
            for index in range(int(state.genome_count)):
                share = repair_amount * state.genome_lesions[index] / _scalar_max(state, total_lesion, 1e-12)
                state.genome_lesions[index] = _scalar_max(state, 0.0, state.genome_lesions[index] - share)
            atp = 0.95 * cost_scale * repair_amount
            state.pools[POOL_ATP] = state.pools[POOL_ATP] - atp
            state.last_repair_atp = state.last_repair_atp + atp
            state.last_repair_flux[REPAIR_GENOME] = repair_amount / max(dt, 1e-9)
            state.cumulative_genome_repairs = state.cumulative_genome_repairs + repair_amount

    if _flag(state, config, 'membrane_repair'):
        activity = _raw_repair_activity(state, REPAIR_MEMBRANE)
        damage_total = _sum_fixed(state.membrane_oxidation * state.membrane)
        chemical = _scalar_min(
            state, damage_total,
            dt * 0.012 * activity * damage_total / (0.012 + damage_total),
            _scalar_max(state, 0.0, state.pools[POOL_ATP] - 0.025) / (0.60 * cost_scale),
        )
        if _number(chemical) > 0.0 and _number(damage_total) > 0.0:
            weighted = state.membrane_oxidation * state.membrane
            fraction = _scalar_min(state, 1.0, chemical / _scalar_max(state, 1e-12, _sum_fixed(weighted)))
            state.membrane_oxidation = state.membrane_oxidation * (1.0 - fraction)
            atp = 0.60 * cost_scale * chemical
            state.pools[POOL_ATP] = state.pools[POOL_ATP] - atp
            state.last_repair_atp = state.last_repair_atp + atp
            state.last_repair_flux[REPAIR_MEMBRANE] = chemical / max(dt, 1e-9)

        severe_weight = _maximum(state.membrane_oxidation - 0.95, 0.0) * state.membrane
        severe_total = _sum_fixed(severe_weight)
        replace = _scalar_min(
            state, severe_total, state.pools[POOL_MEM_PRECURSOR], dt * 0.008 * activity,
            _scalar_max(state, 0.0, state.pools[POOL_ATP] - 0.025) / (0.72 * cost_scale),
        )
        if _number(replace) > 0.0 and _number(severe_total) > 0.0:
            weights = severe_weight / severe_total
            old = _minimum(state.membrane, weights * replace)
            old_total = _sum_fixed(old)
            state.membrane = state.membrane - old
            state.pools[POOL_WASTE] = state.pools[POOL_WASTE] + old_total
            new_total = _scalar_min(state, old_total, state.pools[POOL_MEM_PRECURSOR])
            state.membrane = state.membrane + weights * new_total
            state.pools[POOL_MEM_PRECURSOR] = state.pools[POOL_MEM_PRECURSOR] - new_total
            state.membrane_oxidation = state.membrane_oxidation * _maximum(1.0 - 0.85 * weights, 0.0)
            atp = 0.72 * cost_scale * new_total
            state.pools[POOL_ATP] = state.pools[POOL_ATP] - atp
            state.last_repair_atp = state.last_repair_atp + atp

    state.pools[POOL_ATP] = _scalar_max(state, 0.0, state.pools[POOL_ATP])
    state.cumulative_repair_atp = state.cumulative_repair_atp + state.last_repair_atp
    _compact_order(
        state, 'active_mass', 'active_order', 'active_count',
        ACTIVE_MASS_THRESHOLD,
    )
    _compact_order(
        state, 'damaged_mass', 'damaged_order', 'damaged_count',
        DAMAGED_MASS_THRESHOLD,
    )
    _compact_species_union(state)
    _synchronise_material_pools(state)
    return state


def repair_numpy(state, dt, config=None):
    return _repair(_kernel_input(state, 'numpy'), dt, config)


def repair_torch(state, dt, config=None):
    return _repair(_kernel_input(state, 'torch'), dt, config)


def _supplemental_066(state, dt):
    need = _scalar_max(state, 0.0, 0.095 - state.pools[POOL_MEM_PRECURSOR])
    amount = _scalar_min(
        state, need, 0.024 * float(dt),
        _scalar_max(state, 0.0, state.pools[POOL_MINERAL] - 0.08),
        _scalar_max(state, 0.0, state.pools[POOL_ATP] - 0.018) / 0.30,
    )
    state.pools[POOL_MINERAL] = state.pools[POOL_MINERAL] - amount
    state.pools[POOL_MEM_PRECURSOR] = state.pools[POOL_MEM_PRECURSOR] + amount
    state.pools[POOL_ATP] = state.pools[POOL_ATP] - 0.30 * amount
    _assign_scalar(state, 'supplemental_atp_spent', 0.30 * amount)
    return state


def supplemental_066_numpy(state, dt, config=None):
    return _supplemental_066(_kernel_input(state, 'numpy'), dt)


def supplemental_066_torch(state, dt, config=None):
    return _supplemental_066(_kernel_input(state, 'torch'), dt)


# Friendly aliases matching the milestone wording.
gene_coded_generic_numpy = generic_metabolism_numpy
gene_coded_generic_torch = generic_metabolism_torch
gene_coded_synthesis_numpy = precursor_synthesis_numpy
gene_coded_synthesis_torch = precursor_synthesis_torch
paid_translation_numpy = translation_numpy
paid_translation_torch = translation_torch


# ---------------------------------------------------------------------------
# Independent combined kernel and damage-segregation plan.
# ---------------------------------------------------------------------------


def _combined_metabolism_damage(state, dt, config=None):
    """Run the independent deterministic A3 chain on an already packed cell.

    This function includes the pure translation transcription for independent
    parity.  The hybrid backend intentionally uses the CPU translation and
    replication bridges in their canonical locations instead.
    """
    state = _generic_metabolism(state, dt, config)
    state = _precursor_synthesis(state, dt, config)
    state = _maintenance(state, dt)
    state = _translation(state, dt, config)
    state = _surface_assembly(state, dt, config)
    state = _damage_generation(state, dt, config)
    state = _housekeeping(state, dt, config)
    state = _repair(state, dt, config)
    state = _supplemental_066(state, dt)
    return state


def metabolism_damage_numpy(state, dt, config=None):
    out = _kernel_input(state, 'numpy')
    return _combined_metabolism_damage(out, dt, config)


def metabolism_damage_torch(state, dt, config=None):
    out = _kernel_input(state, 'torch')
    return _combined_metabolism_damage(out, dt, config)


def _segregation_plan(state, config=None):
    source = state
    out = state.clone()
    typed_first = _zeros_like(out.aggregate_mass)
    typed_second = _zeros_like(out.aggregate_mass)
    damaged_first = _zeros_like(out.damaged_mass)
    damaged_second = _zeros_like(out.damaged_mass)
    enabled = _flag(source, config, 'damage_segregation')
    forced_symmetric = _flag(source, config, 'forced_symmetric_damage')
    if not enabled or forced_symmetric:
        effective = _scalar(source, 0.0)
        paid = _scalar(source, 0.0)
        first_fraction = _scalar(source, 0.5)
    else:
        activity = _raw_repair_activity(source, REPAIR_SEGREGATION)
        requested = activity / (0.75 + activity)
        burden = _damage_burden(source)
        cost_scale = _rate(source, config, 'repair_cost_scale')
        atp_cost = 0.030 * requested * burden * cost_scale
        if _number(atp_cost) > 0.0:
            affordable = _scalar_min(
                source, 1.0,
                _scalar_max(source, 0.0, source.pools[POOL_ATP] - 0.030) / atp_cost,
            )
        else:
            affordable = _scalar(source, 1.0)
        effective = _clip(requested * affordable, 0.0, 0.92)
        paid = _scalar_min(
            source, _scalar_max(source, 0.0, source.pools[POOL_ATP] - 0.030),
            atp_cost * affordable,
        )
        first_fraction = 0.5 + 0.44 * effective
    second_fraction = 1.0 - first_fraction
    out.pools[POOL_ATP] = out.pools[POOL_ATP] - paid
    out.cumulative_segregation_atp = out.cumulative_segregation_atp + paid
    _assign_scalar(out, 'last_segregation_strength', effective)
    retained = 0.988
    for position in range(int(out.damaged_count)):
        slot = int(_number(out.damaged_order[position]))
        amount = out.damaged_mass[slot] * retained
        damaged_first[slot] = amount * first_fraction
        damaged_second[slot] = amount * second_fraction
    for position in range(int(out.aggregate_count)):
        slot = int(_number(out.aggregate_order[position]))
        amount = out.aggregate_mass[slot] * retained
        typed_first[slot] = amount * first_fraction
        typed_second[slot] = amount * second_fraction
    unresolved = out.aggregate_unresolved * retained
    reactive = out.pools[POOL_REACTIVE] * retained
    oxidation_first_scale = 1.0 + 0.72 * effective
    oxidation_second_scale = 1.0 - 0.62 * effective
    return {
        'state': out,
        'fractions': (first_fraction, second_fraction),
        'effective': effective,
        'paid': paid,
        'damaged_first': damaged_first,
        'damaged_second': damaged_second,
        'aggregate_first': typed_first,
        'aggregate_second': typed_second,
        'aggregate_unresolved_first': unresolved * first_fraction,
        'aggregate_unresolved_second': unresolved * second_fraction,
        'reactive_first': reactive * first_fraction,
        'reactive_second': reactive * second_fraction,
        'oxidation_first_scale': oxidation_first_scale,
        'oxidation_second_scale': oxidation_second_scale,
        'remaining_factor': _scalar(source, retained),
    }


def segregation_plan_numpy(state, config=None):
    return _segregation_plan(_kernel_input(state, 'numpy'), config=config)


def segregation_plan_torch(state, config=None):
    return _segregation_plan(_kernel_input(state, 'torch'), config=config)


def packed_state_dict(state):
    """Serialize a packed state including aggregate order/composition."""
    validate_a3_state(state)
    state = state.to_numpy() if _is_tensor(state.pools) else state.clone()
    result = {'schema_version': SCHEMA_VERSION}
    for item in fields(state):
        value = getattr(state, item.name)
        if isinstance(value, np.ndarray):
            result[item.name] = value.copy()
        else:
            result[item.name] = copy.deepcopy(value)
    return result


def packed_state_from_dict(payload, config=None):
    payload = dict(payload or {})
    if payload.get('schema_version') != SCHEMA_VERSION:
        raise A3SchemaError('packed save schema mismatch')
    names = {item.name for item in fields(A3PackedState)}
    unknown = set(payload).difference(names)
    if unknown:
        raise A3SchemaError('unknown packed save fields: %s' % sorted(unknown))
    missing = names.difference(payload)
    if missing:
        raise A3SchemaError('missing packed save fields: %s' % sorted(missing))
    state = A3PackedState(**{name: copy.deepcopy(payload[name]) for name in names})
    return validate_a3_state(state, config)


def aggregate_sidecar_state(cell):
    """Small helper for Hybrid save/clone wrappers."""
    return aggregate_composition_state(cell).state_dict()


def restore_aggregate_sidecar(cell, payload):
    return apply_aggregate_composition_state(cell, AggregateCompositionState.from_state(payload))


# ---------------------------------------------------------------------------
# Correctness-first hybrid backend.  The scheduler owns the outer event chain.
# ---------------------------------------------------------------------------


class TorchKernelBackendA3(a2.TorchKernelBackendA2):
    """A2 backend extended with strict full-fidelity A3 metabolism phases."""

    def __init__(self, config=None):
        config = config if isinstance(config, GPU068A3Config) else GPU068A3Config.from_state(config or {})
        super(TorchKernelBackendA3, self).__init__(config)
        self.config = config
        self.adapter = FullFidelityA3Adapter(config)
        self.metabolism_calls = 0
        self.segregation_calls = 0
        self.capacity_failures = 0

    @staticmethod
    def _require_scheduler(scheduler):
        required = (
            'begin_a3_metabolism', 'claim', 'cpu_translation',
            'cpu_replication', 'cpu_genome_hydrolysis',
            'run_post_housekeeping_cpu', 'run_post_repair_cpu',
            'end_a3_metabolism',
        )
        missing = [name for name in required if not callable(getattr(scheduler, name, None))]
        if missing:
            raise A3Error('incomplete A3 scheduler; missing %s' % ', '.join(missing))

    @staticmethod
    def _claim(scheduler, cell, event, status='executed', metadata=None):
        return scheduler.claim(cell, event, status, {} if metadata is None else dict(metadata))

    def _torch_state(self, cell, model_config):
        try:
            packed = self.adapter.pack_cell(cell, model_config=model_config)
        except A3CapacityError:
            self.capacity_failures += 1
            raise
        return packed.to_torch(device=self.device, dtype=self.dtype)

    def _plan_phase(self, cell, model_config, function, dt):
        """Compute an uncommitted phase result for claim-before-mutation use."""
        packed = self._torch_state(cell, model_config)
        self._sync_if_cuda()
        started = time.perf_counter()
        result = function(packed, dt, model_config)
        self._sync_if_cuda()
        self.kernel_seconds += time.perf_counter() - started
        return result

    def _commit_phase(self, result, cell):
        self.adapter.unpack_cell(result, cell)
        return result

    def metabolism_inplace(self, world, cell, dt, scheduler):
        """Execute one authoritative A3 chain; never call the CPU monolith."""
        self._require_scheduler(scheduler)
        model_config = world.config
        begun = scheduler.begin_a3_metabolism(world, cell, dt)
        try:
            if not begun:
                return None
            # Canonical refresh/sync remains CPU because parse/cache is variable
            # genome Python state.  No metabolism has changed before it passes.
            # The refresh methods mutate caches/pools.  The scheduler claim is
            # deliberately first so duplicate/order failures are atomic.
            self._claim(scheduler, cell, 'gene_refresh')
            cell._refresh_gene_cache()
            cell._sync_protein_pool()
            cell._sync_damage_pool()

            plan = self._plan_phase(cell, model_config, generic_metabolism_torch, dt)
            self._claim(scheduler, cell, 'generic_reactions')
            self._commit_phase(plan, cell)
            plan = self._plan_phase(cell, model_config, precursor_synthesis_torch, dt)
            self._claim(scheduler, cell, 'precursor_synthesis')
            self._commit_phase(plan, cell)
            plan = self._plan_phase(cell, model_config, maintenance_torch, dt)
            self._claim(scheduler, cell, 'maintenance')
            self._commit_phase(plan, cell)

            # Translation and replication preserve their authoritative CPU
            # ordering/RNG today.  The pure translation kernel remains tested.
            scheduler.cpu_translation(world, cell, dt, model_config)
            if not cell.alive:
                return None
            scheduler.cpu_replication(world, cell, dt, model_config)
            if not cell.alive:
                return None

            plan = self._plan_phase(cell, model_config, surface_assembly_torch, dt)
            self._claim(scheduler, cell, 'surface_assembly')
            self._commit_phase(plan, cell)

            result = self._plan_phase(cell, model_config, damage_torch, dt)
            self._claim(scheduler, cell, 'protein_damage')
            self._claim(scheduler, cell, 'aggregation')
            self._claim(scheduler, cell, 'reactive_byproduct')
            self._claim(scheduler, cell, 'membrane_oxidation')
            self._claim(scheduler, cell, 'genome_lesion_gain')
            self._commit_phase(result, cell)
            hazards = genome_hydrolysis_plan(result)
            scheduler.cpu_genome_hydrolysis(world, cell, hazards, dt)

            # Smoothing and decay share one pure phase but retain distinct
            # scheduler receipts in their authoritative order.
            plan = self._plan_phase(cell, model_config, housekeeping_torch, dt)
            self._claim(scheduler, cell, 'transporter_smoothing')
            self._claim(scheduler, cell, 'membrane_smoothing')
            self._claim(scheduler, cell, 'ordinary_decay')
            self._commit_phase(plan, cell)

            scheduler.run_post_housekeeping_cpu(world, cell, dt, model_config)
            if not cell.alive:
                return None

            plan = self._plan_phase(cell, model_config, repair_torch, dt)
            for event in ('repair_antioxidant', 'repair_chaperone', 'repair_protease',
                          'repair_genome', 'repair_membrane'):
                self._claim(scheduler, cell, event)
            self._commit_phase(plan, cell)

            post_repair_alive = scheduler.run_post_repair_cpu(world, cell, dt, model_config)
            if not post_repair_alive:
                self._claim(
                    scheduler, cell, 'formal066_supplemental', status='skipped',
                    metadata={'reason': 'dead_after_damage_viability'},
                )
                return None

            # Calculate a private pure plan first; claim its exact executed/no-op
            # status before any CPU object or world ledger is committed.
            result = self._plan_phase(cell, model_config, supplemental_066_torch, dt)
            spent = float(_number(result.supplemental_atp_spent))
            self._claim(
                scheduler, cell, 'formal066_supplemental',
                metadata={
                    'enabled': True, 'atp_spent': spent, 'amount': spent,
                    'work_performed': spent > 0.0,
                },
            )
            self.adapter.unpack_cell(result, cell)
            if spent > 0.0:
                world.dissipated_energy += spent
            self.metabolism_calls += 1
            return cell
        finally:
            cell._defer_damage_viability = False
            scheduler.end_a3_metabolism(world, cell, dt)

    def segregation_plan_inplace(self, world, cell, scheduler=None):
        """Debit the paid plan after the wrapper has claimed ``segregation_plan``."""
        if scheduler is None:
            raise A3Error('segregation wrapper/scheduler is required')
        model_config = world.config
        packed = self._torch_state(cell, model_config)
        plan = segregation_plan_torch(packed, model_config)
        paid = float(_number(plan['paid']))
        effective = float(_number(plan['effective']))
        fractions = tuple(float(_number(value)) for value in plan['fractions'])
        self.adapter.unpack_cell(plan['state'], cell)
        world.dissipated_energy += paid
        cached = {}
        for key, value in plan.items():
            if key == 'state':
                continue
            if _is_tensor(value):
                cached[key] = value.detach().cpu().numpy().copy() if value.ndim else float(value.detach().cpu())
            elif isinstance(value, tuple):
                cached[key] = tuple(float(_number(item)) for item in value)
            else:
                cached[key] = copy.deepcopy(value)
        cell._soma068a3_last_segregation_plan = cached
        self.segregation_calls += 1
        return fractions, effective

    def stats(self):
        result = super(TorchKernelBackendA3, self).stats()
        result.update({
            'metabolism_calls': self.metabolism_calls,
            'segregation_calls': self.segregation_calls,
            'capacity_failures': self.capacity_failures,
            'a3_full_gpu_world_step': False,
        })
        return result


def environment_report():
    report = a2.environment_report()
    report.update({
        'build': BUILD,
        'build_id': BUILD_ID,
        'schema': SCHEMA_VERSION,
        'port_status': dict(PORT_STATUS),
        'full_gpu_world_step': False,
        'reference_precision': 'float64',
        'fp32_status': 'candidate-only-after-fp64-correctness',
    })
    return report


__all__ = [
    'BUILD', 'BUILD_ID', 'BUILD_LONG', 'SCHEMA_VERSION', 'SAVE_VERSION',
    'FULL_GPU_WORLD_STEP', 'PORT_STATUS', 'A3Error', 'A3CapacityError',
    'A3SchemaError', 'GPU068A3Config', 'A3PackedState',
    'AggregateCompositionState', 'FullFidelityA3Adapter',
    'empty_a3_state', 'make_synthetic_a3_state', 'validate_a3_state',
    'pack_a3_cell', 'unpack_a3_cell', 'packed_state_dict',
    'packed_state_from_dict', 'aggregate_sidecar_state',
    'restore_aggregate_sidecar', 'generic_metabolism_numpy',
    'generic_metabolism_torch', 'precursor_synthesis_numpy',
    'precursor_synthesis_torch', 'maintenance_numpy', 'maintenance_torch',
    'translation_numpy', 'translation_torch', 'surface_assembly_numpy',
    'surface_assembly_torch', 'damage_numpy', 'damage_torch',
    'genome_hydrolysis_plan', 'housekeeping_numpy', 'housekeeping_torch',
    'repair_numpy', 'repair_torch', 'supplemental_066_numpy',
    'supplemental_066_torch', 'metabolism_damage_numpy',
    'metabolism_damage_torch', 'segregation_plan_numpy',
    'segregation_plan_torch', 'TorchKernelBackendA3', 'environment_report',
]
