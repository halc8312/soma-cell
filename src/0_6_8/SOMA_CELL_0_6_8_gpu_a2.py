# coding: utf-8
"""SOMA-CELL 0.6.8-GPU A2 — correctness-first surface/physics migration.

A2 extends the frozen A1 tensor foundation without simplifying the detailed
SOMA-CELL 0.6.6 particle world.  The authoritative event order remains the
frozen CPU implementation.  NumPy and Torch fp64 transcriptions are provided
for:

* toroidal cell-particle candidate indexing,
* membrane surface exchange for fuel/mineral/waste/alternative substrate,
* ATP-paid waste export,
* membrane-gap leakage and material emission planning,
* osmotic radius relaxation, and
* surface-flux/Brownian motion.

The hybrid runner integrates these kernels one cell at a time so mutation,
genome expression, division, death, corpse/eDNA/HGT and neural/causal systems
remain CPU-authoritative.  A2 is therefore still not a full-GPU world.
"""
from __future__ import division

import copy
import math
import os
import sys
import time
import types
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import SOMA_CELL_0_6_8_gpu as a1

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

s66 = a1.s66
s65 = a1.s65
s5 = a1.s5
s4 = a1.s4
g2 = a1.g2

BUILD = 'SOMA-CELL 0.6.8-GPU A2'
BUILD_LONG = BUILD + ' | surface exchange / leakage / radius-motion migration'
SCHEMA_VERSION = '0.6.8-GPU-A2.0'
SAVE_VERSION = 1

MEMBRANE_SEGMENTS = a1.MEMBRANE_SEGMENTS
CHANNEL_COUNT = a1.CHANNEL_COUNT
POOL_COUNT = a1.POOL_COUNT
PARTICLE_KIND_COUNT = a1.PARTICLE_KIND_COUNT
INTERACTION_BAND = 0.016

KERNEL_SURFACE_EXCHANGE = 'surface_exchange'
KERNEL_WASTE_EXPORT = 'waste_export'
KERNEL_LEAK_PLAN = 'leak_plan'
KERNEL_RADIUS = 'radius_relaxation'
KERNEL_MOTION = 'surface_flux_motion'
KERNEL_SPATIAL_INDEX = 'toroidal_spatial_index'

PORT_STATUS = dict(a1.PORT_STATUS)
PORT_STATUS.update({
    KERNEL_SURFACE_EXCHANGE: 'hybrid-integrated-correctness-first-sequential-scan',
    KERNEL_WASTE_EXPORT: 'hybrid-integrated',
    KERNEL_LEAK_PLAN: 'hybrid-integrated-torch-plan-cpu-material-emission',
    KERNEL_RADIUS: 'hybrid-integrated',
    KERNEL_MOTION: 'hybrid-integrated-cpu-authoritative-rng',
    KERNEL_SPATIAL_INDEX: 'hybrid-integrated-cpu-grid-with-torch-schema',
    'genome_replication': 'cpu-authoritative',
    'translation': 'cpu-authoritative',
    'division': 'cpu-authoritative',
    'death_corpse_edna_hgt': 'cpu-authoritative',
    'neural_causal_system': 'cpu-authoritative',
})


@dataclass
class GPU068A2Config(a1.GPU068Config):
    enable_surface_exchange: bool = True
    enable_waste_export: bool = True
    enable_leak_plan: bool = True
    enable_radius_motion: bool = True
    spatial_bins: int = 32
    strict_event_lockstep: bool = True

    @classmethod
    def from_state(cls, state):
        state = dict(state or {})
        allowed = set(cls().__dict__.keys())
        return cls(**{k: v for k, v in state.items() if k in allowed})


# ---------------------------------------------------------------------------
# Exact toroidal candidate infrastructure.
# ---------------------------------------------------------------------------


def torus_delta_numpy(points, centre):
    return (np.asarray(points, dtype=np.float64) - np.asarray(centre, dtype=np.float64) + 0.5) % 1.0 - 0.5


def torus_delta_torch(points, centre):
    return torch.remainder(points - centre + 0.5, 1.0) - 0.5


class ToroidalSpatialHash(object):
    """Deterministic CPU grid used by the hybrid correctness path.

    Query results are sorted by original particle index so cell-particle event
    order remains identical to ``np.where`` in the frozen CPU source.
    """

    def __init__(self, positions, bins=32):
        self.bins = max(4, int(bins))
        self.positions = np.asarray(positions, dtype=np.float64)
        self.cells = [[] for _ in range(self.bins * self.bins)]
        if len(self.positions):
            ij = np.floor((self.positions % 1.0) * self.bins).astype(np.int64) % self.bins
            for index, (ix, iy) in enumerate(ij):
                self.cells[int(iy) * self.bins + int(ix)].append(int(index))

    def query(self, centre, radius):
        if not len(self.positions):
            return np.zeros((0,), dtype=np.int64)
        centre = np.asarray(centre, dtype=np.float64) % 1.0
        base = np.floor(centre * self.bins).astype(np.int64) % self.bins
        reach = int(math.ceil(float(radius) * self.bins)) + 1
        candidates = set()
        for dy in range(-reach, reach + 1):
            iy = int((base[1] + dy) % self.bins)
            for dx in range(-reach, reach + 1):
                ix = int((base[0] + dx) % self.bins)
                candidates.update(self.cells[iy * self.bins + ix])
        if not candidates:
            return np.zeros((0,), dtype=np.int64)
        ordered = np.asarray(sorted(candidates), dtype=np.int64)
        deltas = torus_delta_numpy(self.positions[ordered], centre)
        keep = np.linalg.norm(deltas, axis=1) < float(radius)
        return ordered[keep]


def dense_candidate_mask_torch(particle_pos, particle_mask, cell_pos, radius, band=INTERACTION_BAND):
    """Batched exact candidate mask; used to validate future CUDA grid output."""
    delta = torus_delta_torch(particle_pos[:, None, :, :], cell_pos[:, :, None, :])
    distance = torch.linalg.vector_norm(delta, dim=-1)
    return particle_mask[:, None, :] & (distance < radius[:, :, None] + float(band))


def spatial_grid_keys_torch(particle_pos, particle_mask, bins=32):
    """Stable composite grid keys and permutation for later CUDA segmented scans."""
    bins = max(4, int(bins))
    b, p, _ = particle_pos.shape
    ij = torch.floor(torch.remainder(particle_pos, 1.0) * bins).to(torch.int64) % bins
    world = torch.arange(b, device=particle_pos.device, dtype=torch.int64)[:, None]
    key = world * (bins * bins) + ij[..., 1] * bins + ij[..., 0]
    sentinel = torch.full_like(key, b * bins * bins)
    key = torch.where(particle_mask, key, sentinel)
    # Stable tie-break by original index is encoded into sort key.
    index = torch.arange(p, device=particle_pos.device, dtype=torch.int64)[None, :]
    composite = key * (p + 1) + index
    order = torch.argsort(composite, dim=1, stable=True)
    sorted_key = torch.gather(key, 1, order)
    return sorted_key, order


# ---------------------------------------------------------------------------
# Shared exact helpers.
# ---------------------------------------------------------------------------


def _closure_numpy_exact(membrane, oxidation, radius):
    required = float(g2.base.MEMBRANE_DENSITY) * (2.0 * math.pi * float(radius) / MEMBRANE_SEGMENTS)
    base = np.clip(np.asarray(membrane, dtype=np.float64) / (max(required, 1e-9) * float(g2.base.GAP_CLOSURE_THRESHOLD)), 0.0, 1.0)
    impairment = np.exp(-1.65 * np.clip(np.asarray(oxidation, dtype=np.float64), 0.0, 2.5))
    return np.clip(base * impairment, 0.0, 1.0)


def _closure_torch_exact(membrane, oxidation, radius):
    required = float(g2.base.MEMBRANE_DENSITY) * (2.0 * math.pi * radius / MEMBRANE_SEGMENTS)
    base = (membrane / (required.clamp_min(1e-9) * float(g2.base.GAP_CLOSURE_THRESHOLD))).clamp(0.0, 1.0)
    impairment = torch.exp(-1.65 * oxidation.clamp(0.0, 2.5))
    return (base * impairment).clamp(0.0, 1.0)


def _osmolyte_numpy_exact(pools, genome_mass=0.0):
    pools = np.asarray(pools, dtype=np.float64)
    small = (pools[s5.POOL_FUEL] + pools[s5.POOL_MINERAL] + pools[s5.POOL_ATP] +
             pools[s5.POOL_WASTE] + pools[s5.POOL_ALT] + pools[s5.POOL_INTERMEDIATE] +
             pools[s5.POOL_NUCLEOTIDE])
    aggregate = 0.25 * (pools[s5.POOL_MEM_PRECURSOR] + pools[s5.POOL_TRANSPORTER_PRECURSOR])
    proteins = 0.12 * pools[s5.POOL_CATALYST]
    base = small + aggregate + proteins + 0.03 * float(genome_mass)
    return float(base + 0.18 * pools[s5.POOL_DAMAGED_PROTEIN] +
                 0.08 * pools[s5.POOL_AGGREGATE] + 0.70 * pools[s5.POOL_REACTIVE])


def _osmolyte_torch_exact(pools, genome_mass=None):
    if genome_mass is None:
        genome_mass = torch.zeros((), dtype=pools.dtype, device=pools.device)
    small = (pools[s5.POOL_FUEL] + pools[s5.POOL_MINERAL] + pools[s5.POOL_ATP] +
             pools[s5.POOL_WASTE] + pools[s5.POOL_ALT] + pools[s5.POOL_INTERMEDIATE] +
             pools[s5.POOL_NUCLEOTIDE])
    aggregate = 0.25 * (pools[s5.POOL_MEM_PRECURSOR] + pools[s5.POOL_TRANSPORTER_PRECURSOR])
    proteins = 0.12 * pools[s5.POOL_CATALYST]
    base = small + aggregate + proteins + 0.03 * genome_mass
    return base + 0.18 * pools[s5.POOL_DAMAGED_PROTEIN] + 0.08 * pools[s5.POOL_AGGREGATE] + 0.70 * pools[s5.POOL_REACTIVE]


def _tension_numpy_exact(pools, membrane, genome_mass=0.0):
    capacity = np.clip(np.sum(membrane) / (2.0 * math.pi * float(g2.base.MEMBRANE_DENSITY)), float(g2.MIN_RADIUS), float(g2.MAX_RADIUS))
    ratio = max(0.08, _osmolyte_numpy_exact(pools, genome_mass) / float(g2.base.TARGET_OSMOLYTE))
    osmotic = np.clip(float(s5.BASE_RADIUS) * math.sqrt(ratio), float(g2.MIN_RADIUS), float(g2.MAX_RADIUS) * 1.25)
    return max(0.0, osmotic / max(float(capacity), 1e-8) - 1.0)


def _tension_torch_exact(pools, membrane, genome_mass=None):
    capacity = (membrane.sum() / (2.0 * math.pi * float(g2.base.MEMBRANE_DENSITY))).clamp(float(g2.MIN_RADIUS), float(g2.MAX_RADIUS))
    ratio = torch.clamp(_osmolyte_torch_exact(pools, genome_mass) / float(g2.base.TARGET_OSMOLYTE), min=0.08)
    osmotic = (float(s5.BASE_RADIUS) * torch.sqrt(ratio)).clamp(float(g2.MIN_RADIUS), float(g2.MAX_RADIUS) * 1.25)
    return torch.clamp(osmotic / capacity.clamp_min(1e-8) - 1.0, min=0.0)


def _segment_numpy(delta):
    angle = math.atan2(float(delta[1]), float(delta[0])) % (2.0 * math.pi)
    return int((angle / (2.0 * math.pi)) * MEMBRANE_SEGMENTS) % MEMBRANE_SEGMENTS


def _segment_torch(delta):
    angle = torch.remainder(torch.atan2(delta[1], delta[0]), 2.0 * math.pi)
    return torch.floor(angle / (2.0 * math.pi) * MEMBRANE_SEGMENTS).to(torch.int64) % MEMBRANE_SEGMENTS


# ---------------------------------------------------------------------------
# Correctness-first surface exchange.  Sequential scans intentionally preserve
# ATP and internal-pool dependencies.  A later optimisation may replace these
# scans only after event-level equivalence is proven.
# ---------------------------------------------------------------------------


def surface_exchange_numpy(particle_pos, particle_kind, particle_amount, cell_pos, radius,
                           membrane, oxidation, transporters, pools, contact_trace,
                           alt_contact_trace, damage_trace, surface_flux, dt,
                           transport=True, candidate_indices=None):
    pos = np.asarray(particle_pos, dtype=np.float64).copy()
    kind = np.asarray(particle_kind, dtype=np.int64)
    amount = np.asarray(particle_amount, dtype=np.float64).copy()
    cell_pos = np.asarray(cell_pos, dtype=np.float64)
    membrane = np.asarray(membrane, dtype=np.float64)
    transporters = np.asarray(transporters, dtype=np.float64)
    pools = np.asarray(pools, dtype=np.float64).copy()
    contact = np.asarray(contact_trace, dtype=np.float64).copy()
    alt_contact = np.asarray(alt_contact_trace, dtype=np.float64).copy()
    damage = np.asarray(damage_trace, dtype=np.float64).copy()
    flux = np.asarray(surface_flux, dtype=np.float64).copy()
    closure_values = _closure_numpy_exact(membrane, oxidation, radius)
    uptake = np.zeros(2, dtype=np.float64)
    uptake_alt = 0.0
    atp_spent = 0.0
    if candidate_indices is None:
        delta_all = torus_delta_numpy(pos, cell_pos)
        candidate_indices = np.where(np.linalg.norm(delta_all, axis=1) < float(radius) + INTERACTION_BAND)[0]
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    deltas = torus_delta_numpy(pos, cell_pos)
    distances = np.linalg.norm(deltas, axis=1)

    # Base 0.1 scan; position push applies to every particle kind.
    for index in candidate_indices.tolist():
        particle_amount_i = float(amount[index])
        if particle_amount_i <= 1e-8:
            continue
        distance = float(distances[index])
        delta = deltas[index].copy()
        if distance <= 1e-10:
            delta = np.asarray([1.0, 0.0], dtype=np.float64); distance = 1e-10
        segment = _segment_numpy(delta)
        closure = float(closure_values[segment]); gap = 1.0 - closure
        normal = delta / distance
        pkind = int(kind[index])
        if distance < float(radius) * 0.985 and closure > 0.72:
            pos[index] = (cell_pos + normal * (float(radius) + 0.0025)) % 1.0
        if abs(distance - float(radius)) > INTERACTION_BAND:
            continue
        if pkind in (s5.PARTICLE_FUEL, s5.PARTICLE_MINERAL):
            channel = s4.CHANNEL_FUEL if pkind == s5.PARTICLE_FUEL else s4.CHANNEL_MINERAL
            trace_channel = 0 if pkind == s5.PARTICLE_FUEL else 1
            contact[segment, trace_channel] += particle_amount_i * float(dt) * 3.0
            facilitated = 0.0; powered = 0.0
            if transport:
                density = transporters[segment, channel] / max(membrane[segment], 1e-8)
                saturation = particle_amount_i / (0.018 + particle_amount_i)
                pool_index = s5.POOL_FUEL if pkind == s5.PARTICLE_FUEL else s5.POOL_MINERAL
                target = 0.46 if pkind == s5.PARTICLE_FUEL else 0.40
                gradient = 1.0 / (1.0 + (pools[pool_index] / target) ** 4)
                facilitated = 0.62 * density * closure * saturation * gradient * float(dt)
                if pools[s5.POOL_ATP] > 1e-7:
                    powered = min(
                        0.42 * density * closure * saturation * gradient * float(dt),
                        max(0.0, particle_amount_i - facilitated),
                        max(0.0, pools[s5.POOL_ATP] - 0.028) / 0.045,
                    )
            transported = min(particle_amount_i, facilitated + powered)
            passive = min(max(0.0, particle_amount_i - transported), 0.075 * gap ** 2.2 * particle_amount_i * float(dt))
            transfer = max(0.0, transported + passive)
            if transfer > 0.0:
                amount[index] -= transfer
                if pkind == s5.PARTICLE_FUEL:
                    pools[s5.POOL_FUEL] += transfer; uptake[0] += transfer
                else:
                    pools[s5.POOL_MINERAL] += transfer; uptake[1] += transfer
                powered_part = min(powered, transfer)
                pools[s5.POOL_ATP] -= 0.045 * powered_part
                atp_spent += 0.045 * powered_part
                flux += normal * transfer
        elif pkind == s5.PARTICLE_WASTE and gap > 0.05:
            ingress = min(particle_amount_i, 0.032 * gap * particle_amount_i * float(dt))
            if ingress > 0.0:
                amount[index] -= ingress
                pools[s5.POOL_WASTE] += ingress
                damage[segment] += ingress * 1.6

    pools[s5.POOL_ATP] = max(0.0, pools[s5.POOL_ATP])

    # 0.2 alternative-substrate scan recomputes geometry after base push.
    alt_indices = np.where(kind == s5.PARTICLE_ALT)[0]
    alt_deltas = torus_delta_numpy(pos, cell_pos)
    alt_distances = np.linalg.norm(alt_deltas, axis=1)
    for index in alt_indices.tolist():
        distance = float(alt_distances[index])
        if abs(distance - float(radius)) > INTERACTION_BAND:
            continue
        particle_amount_i = float(amount[index])
        if particle_amount_i <= 1e-9:
            continue
        delta = alt_deltas[index].copy()
        if distance <= 1e-10:
            delta = np.asarray([1.0, 0.0], dtype=np.float64); distance = 1e-10
        segment = _segment_numpy(delta)
        closure = float(closure_values[segment]); gap = 1.0 - closure
        alt_contact[segment] += particle_amount_i * float(dt) * 3.0
        facilitated = 0.0; powered = 0.0
        if transport:
            density = transporters[segment, s4.CHANNEL_ALT] / max(membrane[segment], 1e-8)
            saturation = particle_amount_i / (0.018 + particle_amount_i)
            shortage = 1.0 / (1.0 + (pools[s5.POOL_ALT] / 0.32) ** 4)
            facilitated = 0.58 * density * closure * saturation * shortage * float(dt)
            if pools[s5.POOL_ATP] > 0.030:
                powered = min(
                    0.36 * density * closure * saturation * shortage * float(dt),
                    max(0.0, particle_amount_i - facilitated),
                    max(0.0, pools[s5.POOL_ATP] - 0.028) / 0.045,
                )
        transported = min(particle_amount_i, facilitated + powered)
        passive = min(max(0.0, particle_amount_i - transported), 0.065 * gap ** 2.2 * particle_amount_i * float(dt))
        transfer = max(0.0, transported + passive)
        if transfer > 0.0:
            amount[index] -= transfer
            pools[s5.POOL_ALT] += transfer
            pools[s5.POOL_ATP] -= 0.045 * min(powered, transfer)
            uptake_alt += transfer
            flux += (delta / distance) * transfer
    pools[s5.POOL_ATP] = max(0.0, pools[s5.POOL_ATP])
    return {
        'particle_pos': pos, 'particle_amount': amount, 'pools': pools,
        'contact_trace': contact, 'alt_contact_trace': alt_contact,
        'damage_trace': damage, 'surface_flux': flux,
        'last_uptake': uptake, 'last_uptake_alt': float(uptake_alt),
        'atp_spent': float(atp_spent),
    }


def surface_exchange_torch(particle_pos, particle_kind, particle_amount, cell_pos, radius,
                           membrane, oxidation, transporters, pools, contact_trace,
                           alt_contact_trace, damage_trace, surface_flux, dt,
                           transport=True, candidate_indices=None):
    pos = particle_pos.clone(); kind = particle_kind
    amount = particle_amount.clone(); pools = pools.clone()
    contact = contact_trace.clone(); alt_contact = alt_contact_trace.clone()
    damage = damage_trace.clone(); flux = surface_flux.clone()
    closure_values = _closure_torch_exact(membrane, oxidation, radius)
    uptake = torch.zeros((2,), dtype=pools.dtype, device=pools.device)
    uptake_alt = torch.zeros((), dtype=pools.dtype, device=pools.device)
    atp_spent = torch.zeros((), dtype=pools.dtype, device=pools.device)
    deltas = torus_delta_torch(pos, cell_pos)
    distances = torch.linalg.vector_norm(deltas, dim=-1)
    if candidate_indices is None:
        candidate_indices = torch.nonzero(distances < radius + INTERACTION_BAND, as_tuple=False).flatten().tolist()
    elif isinstance(candidate_indices, torch.Tensor):
        candidate_indices = candidate_indices.detach().cpu().tolist()
    else:
        candidate_indices = [int(x) for x in np.asarray(candidate_indices).tolist()]

    for index in candidate_indices:
        particle_amount_i = amount[index]
        if float(particle_amount_i.detach().cpu()) <= 1e-8:
            continue
        distance = distances[index]
        delta = deltas[index].clone()
        if float(distance.detach().cpu()) <= 1e-10:
            delta = torch.tensor([1.0, 0.0], dtype=pos.dtype, device=pos.device)
            distance = torch.as_tensor(1e-10, dtype=pos.dtype, device=pos.device)
        segment_t = _segment_torch(delta); segment = int(segment_t.detach().cpu())
        closure = closure_values[segment]; gap = 1.0 - closure
        normal = delta / distance
        pkind = int(kind[index].detach().cpu())
        if float(distance.detach().cpu()) < float(radius.detach().cpu()) * 0.985 and float(closure.detach().cpu()) > 0.72:
            pos[index] = torch.remainder(cell_pos + normal * (radius + 0.0025), 1.0)
        if abs(float(distance.detach().cpu()) - float(radius.detach().cpu())) > INTERACTION_BAND:
            continue
        if pkind in (s5.PARTICLE_FUEL, s5.PARTICLE_MINERAL):
            channel = s4.CHANNEL_FUEL if pkind == s5.PARTICLE_FUEL else s4.CHANNEL_MINERAL
            trace_channel = 0 if pkind == s5.PARTICLE_FUEL else 1
            contact[segment, trace_channel] += particle_amount_i * float(dt) * 3.0
            facilitated = torch.zeros((), dtype=pools.dtype, device=pools.device)
            powered = torch.zeros_like(facilitated)
            if transport:
                density = transporters[segment, channel] / membrane[segment].clamp_min(1e-8)
                saturation = particle_amount_i / (0.018 + particle_amount_i)
                pool_index = s5.POOL_FUEL if pkind == s5.PARTICLE_FUEL else s5.POOL_MINERAL
                target = 0.46 if pkind == s5.PARTICLE_FUEL else 0.40
                gradient = 1.0 / (1.0 + (pools[pool_index] / target) ** 4)
                facilitated = 0.62 * density * closure * saturation * gradient * float(dt)
                if float(pools[s5.POOL_ATP].detach().cpu()) > 1e-7:
                    powered = torch.minimum(
                        0.42 * density * closure * saturation * gradient * float(dt),
                        torch.minimum(
                            (particle_amount_i - facilitated).clamp_min(0.0),
                            (pools[s5.POOL_ATP] - 0.028).clamp_min(0.0) / 0.045,
                        ),
                    )
            transported = torch.minimum(particle_amount_i, facilitated + powered)
            passive = torch.minimum((particle_amount_i - transported).clamp_min(0.0), 0.075 * gap ** 2.2 * particle_amount_i * float(dt))
            transfer = (transported + passive).clamp_min(0.0)
            if float(transfer.detach().cpu()) > 0.0:
                amount[index] -= transfer
                if pkind == s5.PARTICLE_FUEL:
                    pools[s5.POOL_FUEL] += transfer; uptake[0] += transfer
                else:
                    pools[s5.POOL_MINERAL] += transfer; uptake[1] += transfer
                powered_part = torch.minimum(powered, transfer)
                pools[s5.POOL_ATP] -= 0.045 * powered_part
                atp_spent += 0.045 * powered_part
                flux += normal * transfer
        elif pkind == s5.PARTICLE_WASTE and float(gap.detach().cpu()) > 0.05:
            ingress = torch.minimum(particle_amount_i, 0.032 * gap * particle_amount_i * float(dt))
            if float(ingress.detach().cpu()) > 0.0:
                amount[index] -= ingress
                pools[s5.POOL_WASTE] += ingress
                damage[segment] += ingress * 1.6
    pools[s5.POOL_ATP] = pools[s5.POOL_ATP].clamp_min(0.0)

    alt_indices = torch.nonzero(kind == s5.PARTICLE_ALT, as_tuple=False).flatten().detach().cpu().tolist()
    alt_deltas = torus_delta_torch(pos, cell_pos)
    alt_distances = torch.linalg.vector_norm(alt_deltas, dim=-1)
    radius_value = float(radius.detach().cpu())
    for index in alt_indices:
        distance = alt_distances[index]
        if abs(float(distance.detach().cpu()) - radius_value) > INTERACTION_BAND:
            continue
        particle_amount_i = amount[index]
        if float(particle_amount_i.detach().cpu()) <= 1e-9:
            continue
        delta = alt_deltas[index].clone()
        if float(distance.detach().cpu()) <= 1e-10:
            delta = torch.tensor([1.0, 0.0], dtype=pos.dtype, device=pos.device)
            distance = torch.as_tensor(1e-10, dtype=pos.dtype, device=pos.device)
        segment = int(_segment_torch(delta).detach().cpu())
        closure = closure_values[segment]; gap = 1.0 - closure
        alt_contact[segment] += particle_amount_i * float(dt) * 3.0
        facilitated = torch.zeros((), dtype=pools.dtype, device=pools.device)
        powered = torch.zeros_like(facilitated)
        if transport:
            density = transporters[segment, s4.CHANNEL_ALT] / membrane[segment].clamp_min(1e-8)
            saturation = particle_amount_i / (0.018 + particle_amount_i)
            shortage = 1.0 / (1.0 + (pools[s5.POOL_ALT] / 0.32) ** 4)
            facilitated = 0.58 * density * closure * saturation * shortage * float(dt)
            if float(pools[s5.POOL_ATP].detach().cpu()) > 0.030:
                powered = torch.minimum(
                    0.36 * density * closure * saturation * shortage * float(dt),
                    torch.minimum(
                        (particle_amount_i - facilitated).clamp_min(0.0),
                        (pools[s5.POOL_ATP] - 0.028).clamp_min(0.0) / 0.045,
                    ),
                )
        transported = torch.minimum(particle_amount_i, facilitated + powered)
        passive = torch.minimum((particle_amount_i - transported).clamp_min(0.0), 0.065 * gap ** 2.2 * particle_amount_i * float(dt))
        transfer = (transported + passive).clamp_min(0.0)
        if float(transfer.detach().cpu()) > 0.0:
            amount[index] -= transfer
            pools[s5.POOL_ALT] += transfer
            pools[s5.POOL_ATP] -= 0.045 * torch.minimum(powered, transfer)
            uptake_alt += transfer
            flux += (delta / distance) * transfer
    pools[s5.POOL_ATP] = pools[s5.POOL_ATP].clamp_min(0.0)
    return {
        'particle_pos': pos, 'particle_amount': amount, 'pools': pools,
        'contact_trace': contact, 'alt_contact_trace': alt_contact,
        'damage_trace': damage, 'surface_flux': flux,
        'last_uptake': uptake, 'last_uptake_alt': uptake_alt,
        'atp_spent': atp_spent,
    }


# ---------------------------------------------------------------------------
# Waste export, leakage planning, radius and motion.
# ---------------------------------------------------------------------------


def waste_export_numpy(pools, membrane, oxidation, transporters, radius, cell_pos, surface_flux, dt, enabled=True):
    pools = np.asarray(pools, dtype=np.float64).copy(); flux = np.asarray(surface_flux, dtype=np.float64).copy()
    if (not enabled) or pools[s5.POOL_WASTE] <= 1e-8:
        return {'pools': pools, 'surface_flux': flux, 'amount': 0.0, 'powered': 0.0, 'segment': -1, 'position': np.zeros(2)}
    closure = _closure_numpy_exact(membrane, oxidation, radius)
    density = np.asarray(transporters)[:, s4.CHANNEL_WASTE] / np.maximum(np.asarray(membrane), 1e-8)
    capacity = float(np.sum(density * closure)); saturation = pools[s5.POOL_WASTE] / (0.05 + pools[s5.POOL_WASTE])
    passive = min(pools[s5.POOL_WASTE], 0.0075 * capacity * saturation * float(dt))
    remaining = max(0.0, pools[s5.POOL_WASTE] - passive)
    powered = min(remaining, 0.014 * capacity * saturation * float(dt), max(0.0, pools[s5.POOL_ATP] - 0.018) / 0.026)
    total = passive + powered
    if total <= 0.0:
        return {'pools': pools, 'surface_flux': flux, 'amount': 0.0, 'powered': 0.0, 'segment': -1, 'position': np.zeros(2)}
    weights = density * closure + 1e-6; weights /= float(np.sum(weights))
    segment = int(np.argmax(weights)); angle = 2.0 * math.pi * (segment + 0.5) / MEMBRANE_SEGMENTS
    direction = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
    position = (np.asarray(cell_pos) + direction * (float(radius) + 0.004)) % 1.0
    pools[s5.POOL_WASTE] -= total; pools[s5.POOL_ATP] -= 0.026 * powered
    flux -= direction * total * 0.55
    return {'pools': pools, 'surface_flux': flux, 'amount': float(total), 'powered': float(powered), 'segment': segment, 'position': position}


def waste_export_torch(pools, membrane, oxidation, transporters, radius, cell_pos, surface_flux, dt, enabled=True):
    pools = pools.clone(); flux = surface_flux.clone()
    if (not enabled) or float(pools[s5.POOL_WASTE].detach().cpu()) <= 1e-8:
        return {'pools': pools, 'surface_flux': flux, 'amount': torch.zeros((),dtype=pools.dtype,device=pools.device), 'powered': torch.zeros((),dtype=pools.dtype,device=pools.device), 'segment': -1, 'position': torch.zeros((2,),dtype=pools.dtype,device=pools.device)}
    closure = _closure_torch_exact(membrane, oxidation, radius)
    density = transporters[:, s4.CHANNEL_WASTE] / membrane.clamp_min(1e-8)
    capacity = torch.sum(density * closure); saturation = pools[s5.POOL_WASTE] / (0.05 + pools[s5.POOL_WASTE])
    passive = torch.minimum(pools[s5.POOL_WASTE], 0.0075 * capacity * saturation * float(dt))
    remaining = (pools[s5.POOL_WASTE] - passive).clamp_min(0.0)
    powered = torch.minimum(remaining, torch.minimum(0.014 * capacity * saturation * float(dt), (pools[s5.POOL_ATP] - 0.018).clamp_min(0.0) / 0.026))
    total = passive + powered
    if float(total.detach().cpu()) <= 0.0:
        return {'pools': pools, 'surface_flux': flux, 'amount': total, 'powered': powered, 'segment': -1, 'position': torch.zeros((2,),dtype=pools.dtype,device=pools.device)}
    weights = density * closure + 1e-6; weights = weights / weights.sum()
    segment = int(torch.argmax(weights).detach().cpu()); angle = 2.0 * math.pi * (segment + 0.5) / MEMBRANE_SEGMENTS
    direction = torch.tensor([math.cos(angle), math.sin(angle)], dtype=pools.dtype, device=pools.device)
    position = torch.remainder(cell_pos + direction * (radius + 0.004), 1.0)
    pools[s5.POOL_WASTE] -= total; pools[s5.POOL_ATP] -= 0.026 * powered
    flux -= direction * total * 0.55
    return {'pools': pools, 'surface_flux': flux, 'amount': total, 'powered': powered, 'segment': segment, 'position': position}


def leak_plan_numpy(pools, membrane, oxidation, radius, dt, genome_mass=0.0):
    pools = np.asarray(pools, dtype=np.float64)
    closure = _closure_numpy_exact(membrane, oxidation, radius)
    gap_weights = np.maximum(1.0 - closure, 0.0) ** 2.4
    gap_strength = float(np.mean(gap_weights)); pressure = _tension_numpy_exact(pools, membrane, genome_mass)
    fraction = float(np.clip((0.065 * gap_strength + 0.080 * gap_strength * pressure) * float(dt), 0.0, 0.18))
    segment = int(np.argmax(gap_weights))
    base_indices = [s5.POOL_FUEL, s5.POOL_MINERAL, s5.POOL_MEM_PRECURSOR, s5.POOL_CATALYST, s5.POOL_TRANSPORTER_PRECURSOR, s5.POOL_WASTE]
    extra_indices = [s5.POOL_ALT, s5.POOL_INTERMEDIATE, s5.POOL_NUCLEOTIDE]
    return {'fraction': fraction, 'gap_strength': gap_strength, 'segment': segment,
            'base_amounts': pools[base_indices] * fraction,
            'extra_amounts': pools[extra_indices] * fraction,
            'atp_loss': pools[s5.POOL_ATP] * fraction * 1.4}


def leak_plan_torch(pools, membrane, oxidation, radius, dt, genome_mass=None):
    closure = _closure_torch_exact(membrane, oxidation, radius)
    gap_weights = (1.0 - closure).clamp_min(0.0) ** 2.4
    gap_strength = gap_weights.mean(); pressure = _tension_torch_exact(pools, membrane, genome_mass)
    fraction = ((0.065 * gap_strength + 0.080 * gap_strength * pressure) * float(dt)).clamp(0.0, 0.18)
    segment = int(torch.argmax(gap_weights).detach().cpu())
    base = torch.tensor([s5.POOL_FUEL,s5.POOL_MINERAL,s5.POOL_MEM_PRECURSOR,s5.POOL_CATALYST,s5.POOL_TRANSPORTER_PRECURSOR,s5.POOL_WASTE],dtype=torch.int64,device=pools.device)
    extra = torch.tensor([s5.POOL_ALT,s5.POOL_INTERMEDIATE,s5.POOL_NUCLEOTIDE],dtype=torch.int64,device=pools.device)
    return {'fraction': fraction, 'gap_strength': gap_strength, 'segment': segment,
            'base_amounts': pools[base] * fraction,
            'extra_amounts': pools[extra] * fraction,
            'atp_loss': pools[s5.POOL_ATP] * fraction * 1.4}


def radius_relax_numpy(pools, membrane, radius, dt, genome_mass=0.0):
    capacity = np.clip(np.sum(membrane) / (2.0 * math.pi * float(g2.base.MEMBRANE_DENSITY)), float(g2.MIN_RADIUS), float(g2.MAX_RADIUS))
    ratio = max(0.08, _osmolyte_numpy_exact(pools, genome_mass) / float(g2.base.TARGET_OSMOLYTE))
    osmotic = np.clip(float(s5.BASE_RADIUS) * math.sqrt(ratio), float(g2.MIN_RADIUS), float(g2.MAX_RADIUS) * 1.25)
    target = float(capacity) * np.clip(float(osmotic) / max(float(capacity),1e-8),0.66,float(g2.base.MEMBRANE_STRETCH_LIMIT))
    target = np.clip(target,float(g2.MIN_RADIUS),float(g2.MAX_RADIUS))
    result = float(radius) + (float(target)-float(radius))*(1.0-math.exp(-2.8*float(dt)))
    return float(np.clip(result,float(g2.MIN_RADIUS),float(g2.MAX_RADIUS)))


def radius_relax_torch(pools, membrane, radius, dt, genome_mass=None):
    capacity = (membrane.sum()/(2.0*math.pi*float(g2.base.MEMBRANE_DENSITY))).clamp(float(g2.MIN_RADIUS),float(g2.MAX_RADIUS))
    ratio = torch.clamp(_osmolyte_torch_exact(pools, genome_mass)/float(g2.base.TARGET_OSMOLYTE),min=0.08)
    osmotic = (float(s5.BASE_RADIUS)*torch.sqrt(ratio)).clamp(float(g2.MIN_RADIUS),float(g2.MAX_RADIUS)*1.25)
    target = capacity * torch.clamp(osmotic/capacity.clamp_min(1e-8),0.66,float(g2.base.MEMBRANE_STRETCH_LIMIT))
    target = target.clamp(float(g2.MIN_RADIUS),float(g2.MAX_RADIUS))
    return (radius + (target-radius)*(1.0-math.exp(-2.8*float(dt)))).clamp(float(g2.MIN_RADIUS),float(g2.MAX_RADIUS))


def motion_numpy(pos, vel, surface_flux, radius, brownian, dt):
    pos=np.asarray(pos,dtype=np.float64).copy(); vel=np.asarray(vel,dtype=np.float64).copy(); flux=np.asarray(surface_flux,dtype=np.float64).copy()
    thrust=flux*0.11; brown=np.asarray(brownian,dtype=np.float64)/max(0.65,float(radius)/float(s5.BASE_RADIUS))
    vel += thrust+brown; flux*=math.exp(-2.4*float(dt))
    speed=float(np.linalg.norm(vel)); max_speed=0.020/max(0.7,float(radius)/float(s5.BASE_RADIUS))
    if speed>max_speed: vel*=max_speed/speed
    vel*=math.exp(-1.8*float(dt)); pos=(pos+vel*float(dt))%1.0
    return {'pos':pos,'vel':vel,'surface_flux':flux}


def motion_torch(pos, vel, surface_flux, radius, brownian, dt):
    pos=pos.clone();vel=vel.clone();flux=surface_flux.clone()
    scale=torch.clamp(radius/float(s5.BASE_RADIUS),min=0.65)
    vel += flux*0.11 + brownian/scale; flux*=math.exp(-2.4*float(dt))
    speed=torch.linalg.vector_norm(vel); max_speed=0.020/torch.clamp(radius/float(s5.BASE_RADIUS),min=0.7)
    if float(speed.detach().cpu())>float(max_speed.detach().cpu()): vel*=max_speed/speed
    vel*=math.exp(-1.8*float(dt)); pos=torch.remainder(pos+vel*float(dt),1.0)
    return {'pos':pos,'vel':vel,'surface_flux':flux}


# ---------------------------------------------------------------------------
# Hybrid integration.
# ---------------------------------------------------------------------------


def _sensorimotor_surface_post(cell, field, before_fuel, before_waste, dt):
    world = getattr(field, '_world_ref', None)
    if world is None:
        return
    acquired = max(0.0, float(cell.pools[s5.POOL_FUEL]) - float(before_fuel))
    uptake = np.zeros(s4.LIGAND_COUNT, dtype=float)
    uptake[s4.LIGAND_FUEL] = max(0.0, float(cell.last_uptake[0]))
    uptake[s4.LIGAND_MINERAL] = max(0.0, float(cell.last_uptake[1]))
    uptake[s4.LIGAND_ALT] = max(0.0, float(cell.last_uptake_alt))
    uptake[s4.LIGAND_WASTE] = max(0.0, float(cell.pools[s5.POOL_WASTE]) - float(before_waste))
    cell.last_uptake_by_ligand = uptake
    cell.cumulative_uptake_by_ligand += uptake
    cell.uptake_trace *= math.exp(-0.13 * float(dt))
    cell.uptake_trace += np.clip(uptake * 70.0, 0.0, 1.5)
    for fingerprint, spec in cell.sensor_specs():
        ligand = int(spec['parameter']) % s4.LIGAND_COUNT
        perturbation = float(cell.last_gene_perturbation.get(fingerprint, 0.0))
        if perturbation != 0.0 and uptake[ligand] > 0.0:
            eligibility = float(cell.controller_eligibility.get(fingerprint, 0.0))
            eligibility += perturbation * min(1.2, uptake[ligand] * 95.0)
            cell.controller_eligibility[fingerprint] = float(np.clip(eligibility, -3.0, 3.0))
    if acquired > 0.0 and world.config.environment_mode == 'reversal' and world.age >= world.config.switch_age:
        contaminated = min(cell.pools[s5.POOL_FUEL], acquired * float(np.clip(world.config.contaminated_fraction,0.0,1.0)))
        cell.pools[s5.POOL_FUEL] -= contaminated
        cell.pools[s5.POOL_REACTIVE] += contaminated
        cell.damage_trace += contaminated / MEMBRANE_SEGMENTS * 1.8
        cell.fuel_contamination_received += contaminated
        world.fuel_contamination_total += contaminated


class TorchKernelBackendA2(a1.TorchKernelBackend):
    def __init__(self, config=None):
        super(TorchKernelBackendA2,self).__init__(config if isinstance(config,GPU068A2Config) else GPU068A2Config.from_state(config or {}))
        self.config = config if isinstance(config,GPU068A2Config) else GPU068A2Config.from_state(config or {})
        self.surface_calls=0;self.export_calls=0;self.leak_calls=0;self.radius_calls=0;self.motion_calls=0;self.spatial_queries=0

    def _tensor(self, value, integer=False):
        dtype=torch.int64 if integer else self.dtype
        return torch.as_tensor(value,dtype=dtype,device=self.device)

    def surface_exchange_inplace(self, field, cell, dt, config):
        if not cell.alive or len(field.amount)==0:
            cell.last_uptake[:]=0.0;cell.last_uptake_alt=0.0;return
        before_fuel=float(cell.pools[s5.POOL_FUEL]);before_waste=float(cell.pools[s5.POOL_WASTE])
        index=ToroidalSpatialHash(field.pos,bins=self.config.spatial_bins)
        candidates=index.query(cell.pos,cell.radius+INTERACTION_BAND);self.spatial_queries+=1
        args=(self._tensor(field.pos),self._tensor(field.kind,integer=True),self._tensor(field.amount),self._tensor(cell.pos),self._tensor(cell.radius),self._tensor(cell.membrane),self._tensor(cell.membrane_oxidation),self._tensor(cell.transporters),self._tensor(cell.pools),self._tensor(cell.contact_trace),self._tensor(cell.alt_contact_trace),self._tensor(cell.damage_trace),self._tensor(cell.surface_flux),float(dt))
        self._sync_if_cuda();start=time.perf_counter()
        out=surface_exchange_torch(*args,transport=bool(config.transport),candidate_indices=candidates)
        self._sync_if_cuda();self.kernel_seconds+=time.perf_counter()-start
        field.pos=out['particle_pos'].detach().cpu().numpy().astype(float,copy=True)
        field.amount=out['particle_amount'].detach().cpu().numpy().astype(float,copy=True)
        cell.pools=out['pools'].detach().cpu().numpy().astype(float,copy=True)
        cell.contact_trace=out['contact_trace'].detach().cpu().numpy().astype(float,copy=True)
        cell.alt_contact_trace=out['alt_contact_trace'].detach().cpu().numpy().astype(float,copy=True)
        cell.damage_trace=out['damage_trace'].detach().cpu().numpy().astype(float,copy=True)
        cell.surface_flux=out['surface_flux'].detach().cpu().numpy().astype(float,copy=True)
        cell.last_uptake[:]=out['last_uptake'].detach().cpu().numpy().astype(float,copy=True)
        cell.last_uptake_alt=float(out['last_uptake_alt'].detach().cpu())
        if float(out['atp_spent'].detach().cpu())>0.0:cell.reaction_events+=1
        _sensorimotor_surface_post(cell,field,before_fuel,before_waste,dt)
        self.surface_calls+=1

    def waste_export_inplace(self, field, cell, dt, config):
        out=waste_export_torch(self._tensor(cell.pools),self._tensor(cell.membrane),self._tensor(cell.membrane_oxidation),self._tensor(cell.transporters),self._tensor(cell.radius),self._tensor(cell.pos),self._tensor(cell.surface_flux),dt,enabled=bool(config.waste_export))
        amount=float(out['amount'].detach().cpu());cell.last_export=amount
        if amount<=0.0:return
        field.add_particle(s5.PARTICLE_WASTE,out['position'].detach().cpu().numpy().astype(float),amount,count_as_injection=False)
        cell.pools=out['pools'].detach().cpu().numpy().astype(float,copy=True)
        cell.surface_flux=out['surface_flux'].detach().cpu().numpy().astype(float,copy=True)
        self.export_calls+=1

    def leak_inplace(self, world, cell, dt):
        plan=leak_plan_torch(self._tensor(cell.pools),self._tensor(cell.membrane),self._tensor(cell.membrane_oxidation),self._tensor(cell.radius),dt,self._tensor(cell.genome_mass()))
        fraction=float(plan['fraction'].detach().cpu());gap_strength=float(plan['gap_strength'].detach().cpu())
        cell.last_leak=0.0
        if fraction<=0.0:return
        segment=int(plan['segment']);angle=2.0*math.pi*(segment+0.5)/MEMBRANE_SEGMENTS
        direction=np.asarray([math.cos(angle),math.sin(angle)],dtype=float)
        position=(cell.pos+direction*(cell.radius+0.004))%1.0
        mapping=((s5.POOL_FUEL,s5.PARTICLE_FUEL),(s5.POOL_MINERAL,s5.PARTICLE_MINERAL),(s5.POOL_MEM_PRECURSOR,s5.PARTICLE_WASTE),(s5.POOL_CATALYST,s5.PARTICLE_WASTE),(s5.POOL_TRANSPORTER_PRECURSOR,s5.PARTICLE_WASTE),(s5.POOL_WASTE,s5.PARTICLE_WASTE))
        catalyst_before=float(cell.pools[s5.POOL_CATALYST]);total=0.0
        for pool_index,particle_kind in mapping:
            amount=float(cell.pools[pool_index])*fraction
            if amount<=0.0:continue
            cell.pools[pool_index]-=amount
            world.field.add_particle(particle_kind,(position+world.rng.normal(0.0,0.003,2))%1.0,amount,count_as_injection=False)
            total+=amount
        atp_loss=float(cell.pools[s5.POOL_ATP])*fraction*1.4
        cell.pools[s5.POOL_ATP]-=atp_loss;world.dissipated_energy+=atp_loss
        catalyst_after=float(cell.pools[s5.POOL_CATALYST])
        if catalyst_before>1e-12 and catalyst_after<catalyst_before:
            scale=float(np.clip(catalyst_after/catalyst_before,0.0,1.0))
            for fingerprint in list(cell.proteins.keys()):cell.proteins[fingerprint]*=scale
            cell._sync_protein_pool()
        for pool_index,particle_kind in ((s5.POOL_ALT,s5.PARTICLE_ALT),(s5.POOL_INTERMEDIATE,s5.PARTICLE_WASTE),(s5.POOL_NUCLEOTIDE,s5.PARTICLE_WASTE)):
            amount=float(cell.pools[pool_index])*fraction
            if amount<=0.0:continue
            cell.pools[pool_index]-=amount
            world.field.add_particle(particle_kind,(position+world.rng.normal(0.0,0.003,2))%1.0,amount,count_as_injection=False)
            total+=amount
        if cell.replication_copy and gap_strength>0.08:
            count=min(len(cell.replication_copy),max(1,int(len(cell.replication_copy)*fraction)))
            del cell.replication_copy[-count:]
            amount=count*float(s5.MONOMER_MASS)
            world.field.add_particle(s5.PARTICLE_WASTE,position,amount,count_as_injection=False)
            total+=amount
        cell.last_leak=total;self.leak_calls+=1

    def radius_inplace(self, cell, dt):
        result=radius_relax_torch(self._tensor(cell.pools),self._tensor(cell.membrane),self._tensor(cell.radius),dt,self._tensor(cell.genome_mass()))
        cell.radius=float(result.detach().cpu());self.radius_calls+=1

    def motion_inplace(self, cell, rng, dt):
        brownian=rng.normal(0.0,0.00075,2)
        out=motion_torch(self._tensor(cell.pos),self._tensor(cell.vel),self._tensor(cell.surface_flux),self._tensor(cell.radius),self._tensor(brownian),dt)
        cell.pos=out['pos'].detach().cpu().numpy().astype(float,copy=True)
        cell.vel=out['vel'].detach().cpu().numpy().astype(float,copy=True)
        cell.surface_flux=out['surface_flux'].detach().cpu().numpy().astype(float,copy=True)
        self.motion_calls+=1

    def stats(self):
        out=super(TorchKernelBackendA2,self).stats();out.update({'surface_calls':self.surface_calls,'waste_export_calls':self.export_calls,'leak_calls':self.leak_calls,'radius_calls':self.radius_calls,'motion_calls':self.motion_calls,'spatial_queries':self.spatial_queries})
        return out


def _attach_backend_to_world_a2(world,backend):
    a1._attach_backend_to_world(world,backend)
    for cell in world.cells:
        if not hasattr(cell,'_soma068a2_original_surface_exchange'):
            cell._soma068a2_original_surface_exchange=cell.surface_exchange
            def _surface(this,field,dt,config):return backend.surface_exchange_inplace(field,this,dt,config)
            cell.surface_exchange=types.MethodType(_surface,cell)
        if not hasattr(cell,'_soma068a2_original_export_waste'):
            cell._soma068a2_original_export_waste=cell.export_waste
            def _export(this,field,dt,config):return backend.waste_export_inplace(field,this,dt,config)
            cell.export_waste=types.MethodType(_export,cell)
        if not hasattr(cell,'_soma068a2_original_leak'):
            cell._soma068a2_original_leak=cell.leak
            def _leak(this,world_,dt):return backend.leak_inplace(world_,this,dt)
            cell.leak=types.MethodType(_leak,cell)
        if not hasattr(cell,'_soma068a2_original_update_radius'):
            cell._soma068a2_original_update_radius=cell.update_radius
            def _radius(this,dt):return backend.radius_inplace(this,dt)
            cell.update_radius=types.MethodType(_radius,cell)
        if not hasattr(cell,'_soma068a2_original_update_motion'):
            cell._soma068a2_original_update_motion=cell.update_motion
            def _motion(this,rng,dt):return backend.motion_inplace(this,rng,dt)
            cell.update_motion=types.MethodType(_motion,cell)
        cell._soma068a2_backend=backend
    return world


class Hybrid066WorldA2(a1.Hybrid066World):
    def __init__(self,cpu_world,gpu_config=None):
        if not isinstance(cpu_world,s66.Formal066World):raise TypeError('Hybrid066WorldA2 requires Formal066World')
        self.backend=TorchKernelBackendA2(gpu_config)
        self.world=cpu_world
        _attach_backend_to_world_a2(self.world,self.backend)

    @classmethod
    def new(cls,seed=101,initial_cells=3,world_config=None,gpu_config=None):
        config=world_config if isinstance(world_config,s66.Formal066Config) else s66.Formal066Config.from_state(world_config or {})
        return cls(s66.Formal066World(seed=seed,initial_cells=initial_cells,config=config),gpu_config=gpu_config)

    def step(self,dt):
        _attach_backend_to_world_a2(self.world,self.backend)
        self.world.step(dt)
        _attach_backend_to_world_a2(self.world,self.backend)

    def summary(self):
        out=self.world.summary();out.update({'gpu_build':BUILD,'gpu_schema':SCHEMA_VERSION,'gpu_port_status':dict(PORT_STATUS),'gpu_backend':self.backend.stats(),'gpu_full_world_step':False})
        return out

    def state_dict(self):
        return {'save_version':SAVE_VERSION,'build':BUILD,'gpu_config':self.backend.config.state_dict(),'cpu_world':self.world.state_dict()}

    @classmethod
    def from_state(cls,state):
        return cls(s66.Formal066World.from_state(state['cpu_world']),gpu_config=GPU068A2Config.from_state(state.get('gpu_config',{})))

    def clone(self):return Hybrid066WorldA2.from_state(self.state_dict())


def environment_report():
    report=a1.environment_report();report.update({'build':BUILD,'schema':SCHEMA_VERSION,'port_status':dict(PORT_STATUS),'full_gpu_world_step':False,'a2_note':'surface/physics kernels are integrated in a correctness-first one-cell-at-a-time hybrid; genome/division/death/HGT/neural remain CPU-authoritative'})
    return report
