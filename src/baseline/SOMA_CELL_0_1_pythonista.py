# coding: utf-8
"""
SOMA-CELL 0.1 — Material Autopoiesis Kernel / 物質的自己生産核

Standalone Pythonista 3 / CPython + NumPy artificial-life prototype.

This branch deliberately steps away from the increasingly cognitive SOMA-7R
line and asks a more basic question: can a simulated individual continuously
manufacture the material boundary and catalytic machinery that make it an
individual at all?

Core properties
---------------
* The membrane is a ring of material-bearing segments, not a fixed circle or a
  scalar ``integrity`` variable.
* Fuel and mineral molecules remain outside unless membrane transporters or a
  physical gap let them cross.
* Internal catalysts turn fuel into usable chemical energy and catalyse their
  own replacement, transporter synthesis, and membrane-precursor synthesis.
* Membrane precursor self-assembles preferentially at thin / damaged segments.
* Membrane loss leaks internal material.  Loss of the catalytic–membrane loop,
  rather than ``health <= 0``, causes lysis and death.
* Surplus membrane, catalysts, and cytoplasm can build a septum and produce two
  closed daughters.  Division has an explicit material ledger.
* Metabolic waste damages the boundary and must be exported.  Dead cells return
  their material to the external chemistry.

This is a falsifiable numerical model of a self-producing boundary.  It is not
claimed to be biological life, consciousness, or demonstrated open-ended
 evolution.
"""

from __future__ import division

import csv
import gc
import math
import os
import pickle
import time

import numpy as np


BUILD = 'SOMA-CELL 0.1.0'
SAVE_VERSION = 1
BASE_DIR = os.path.dirname(__file__)
SAVE_FILE = os.path.join(BASE_DIR, 'soma_cell_0_1.pkl')
LOG_FILE = os.path.join(BASE_DIR, 'soma_cell_0_1_longrun.csv')
REPORT_FILE = os.path.join(BASE_DIR, 'soma_cell_0_1_report.txt')
SESSION_FILE = os.path.join(BASE_DIR, 'soma_cell_0_1_sessions.csv')

SIM_HZ = 20.0
AUTO_SAVE_INTERVAL = 30.0
LOG_INTERVAL = 10.0
MAX_CELLS = 12
MEMBRANE_SEGMENTS = 36
PARTICLE_LIMIT = 360

# External particle kinds.
PARTICLE_FUEL = 0
PARTICLE_MINERAL = 1
PARTICLE_WASTE = 2
PARTICLE_NAMES = ('fuel', 'mineral', 'waste')

# Cytoplasmic pools.  ATP is energy and is intentionally excluded from the
# material ledger; all other pools are material species.
POOL_FUEL = 0
POOL_MINERAL = 1
POOL_ATP = 2
POOL_MEM_PRECURSOR = 3
POOL_CATALYST = 4
POOL_TRANSPORTER_PRECURSOR = 5
POOL_WASTE = 6
POOL_COUNT = 7

# Membrane transporter channels.
CHANNEL_FUEL = 0
CHANNEL_MINERAL = 1
CHANNEL_WASTE = 2
CHANNEL_COUNT = 3

BASE_RADIUS = 0.047
MIN_RADIUS = 0.022
MAX_RADIUS = 0.105
MEMBRANE_DENSITY = 4.34  # mass per unit circumference
MEMBRANE_STRETCH_LIMIT = 1.22
TARGET_OSMOLYTE = 1.65
INITIAL_MEMBRANE_MASS = 2.0 * math.pi * BASE_RADIUS * MEMBRANE_DENSITY
INITIAL_SEGMENT_MASS = INITIAL_MEMBRANE_MASS / MEMBRANE_SEGMENTS
GAP_CLOSURE_THRESHOLD = 0.92


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def clamp(value, low, high):
    return float(min(max(float(value), float(low)), float(high)))


def wrapped_delta(source, target):
    delta = np.asarray(target, dtype=float) - np.asarray(source, dtype=float)
    return (delta + 0.5) % 1.0 - 0.5


def torus_distance(source, target):
    return float(np.linalg.norm(wrapped_delta(source, target)))


def unit_vector(angle):
    return np.array([math.cos(angle), math.sin(angle)], dtype=float)


def safe_div(numerator, denominator, default=0.0):
    denominator = float(denominator)
    if abs(denominator) < 1e-12:
        return float(default)
    value = float(numerator) / denominator
    return value if np.isfinite(value) else float(default)


def finite_array(array):
    return bool(np.all(np.isfinite(np.asarray(array, dtype=float))))


def circular_smooth(values, amount):
    values = np.asarray(values, dtype=float)
    amount = clamp(amount, 0.0, 0.49)
    return (
        (1.0 - 2.0 * amount) * values
        + amount * np.roll(values, 1)
        + amount * np.roll(values, -1)
    )


def _atomic_pickle(path, payload):
    temporary = path + '.tmp'
    with open(temporary, 'wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        os.replace(temporary, path)
    except AttributeError:
        if os.path.exists(path):
            os.remove(path)
        os.rename(temporary, path)


def _memory_peak_mb_estimate():
    try:
        import resource
        raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if raw <= 0.0 or not np.isfinite(raw):
            return float('nan')
        return raw / (1024.0 * 1024.0) if raw > 5.0e6 else raw / 1024.0
    except Exception:
        return float('nan')


# ---------------------------------------------------------------------------
# Configuration / ablation switches
# ---------------------------------------------------------------------------


class CellConfig(object):
    """Runtime switches used by validation and ablation experiments."""

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
    ):
        self.membrane_synthesis = bool(membrane_synthesis)
        self.catalyst_synthesis = bool(catalyst_synthesis)
        self.transport = bool(transport)
        self.targeted_repair = bool(targeted_repair)
        self.division = bool(division)
        self.waste_export = bool(waste_export)
        self.external_inflow = bool(external_inflow)
        self.environmental_damage = bool(environmental_damage)

    def state_dict(self):
        return dict(self.__dict__)

    @classmethod
    def from_state(cls, state):
        return cls(**dict(state))


# ---------------------------------------------------------------------------
# Particle field
# ---------------------------------------------------------------------------


class ParticleField(object):
    """Diffusing external matter shared by every cell."""

    def __init__(self, rng, initial=True):
        self.rng = rng
        self.pos = np.empty((0, 2), dtype=float)
        self.kind = np.empty((0,), dtype=np.int16)
        self.amount = np.empty((0,), dtype=float)
        self.recycled_fuel_buffer = 0.0
        self.recycled_mineral_buffer = 0.0
        self.injected_material = 0.0
        self.dissipated_material = 0.0
        if initial:
            self.seed_initial()

    def seed_initial(self):
        for kind, count, mean_amount in (
            (PARTICLE_FUEL, 82, 0.030),
            (PARTICLE_MINERAL, 72, 0.028),
            (PARTICLE_WASTE, 12, 0.018),
        ):
            positions = self.rng.random((count, 2))
            amounts = np.clip(
                self.rng.normal(mean_amount, mean_amount * 0.18, count),
                mean_amount * 0.45,
                mean_amount * 1.55,
            )
            self.add_many(kind, positions, amounts, count_as_injection=False)

    def add_many(self, kind, positions, amounts, count_as_injection=False):
        positions = np.asarray(positions, dtype=float).reshape(-1, 2) % 1.0
        amounts = np.asarray(amounts, dtype=float).reshape(-1)
        if positions.shape[0] != amounts.shape[0]:
            raise ValueError('positions and amounts must have equal length')
        positive = amounts > 1e-9
        if not np.any(positive):
            return
        positions = positions[positive]
        amounts = amounts[positive]
        kinds = np.full((len(amounts),), int(kind), dtype=np.int16)
        self.pos = np.concatenate([self.pos, positions], axis=0)
        self.kind = np.concatenate([self.kind, kinds], axis=0)
        self.amount = np.concatenate([self.amount, amounts], axis=0)
        if count_as_injection:
            self.injected_material += float(np.sum(amounts))
        self._enforce_limit()

    def add_particle(self, kind, position, amount, count_as_injection=False):
        self.add_many(
            kind,
            np.asarray(position, dtype=float).reshape(1, 2),
            np.asarray([amount], dtype=float),
            count_as_injection=count_as_injection,
        )

    def add_cloud(self, position, fuel=0.24, mineral=0.20, spread=0.025):
        position = np.asarray(position, dtype=float)
        for kind, total, count in (
            (PARTICLE_FUEL, float(fuel), 8),
            (PARTICLE_MINERAL, float(mineral), 7),
        ):
            positions = (position[None, :] + self.rng.normal(0.0, spread, (count, 2))) % 1.0
            amounts = np.full((count,), total / count, dtype=float)
            self.add_many(kind, positions, amounts, count_as_injection=True)

    def _enforce_limit(self):
        if len(self.amount) <= PARTICLE_LIMIT:
            return
        # Merge the smallest particles into a recycled buffer rather than
        # silently deleting matter.
        order = np.argsort(self.amount)
        remove = order[: len(self.amount) - PARTICLE_LIMIT]
        keep = np.ones(len(self.amount), dtype=bool)
        keep[remove] = False
        for index in remove:
            amount = float(self.amount[index])
            kind = int(self.kind[index])
            if kind == PARTICLE_MINERAL:
                self.recycled_mineral_buffer += amount
            elif kind == PARTICLE_FUEL:
                self.recycled_fuel_buffer += amount
            else:
                self.recycled_fuel_buffer += 0.55 * amount
                self.recycled_mineral_buffer += 0.35 * amount
                self.dissipated_material += 0.10 * amount
        self.pos = self.pos[keep]
        self.kind = self.kind[keep]
        self.amount = self.amount[keep]

    def compact(self):
        keep = np.isfinite(self.amount) & (self.amount > 1e-7)
        self.pos = self.pos[keep] % 1.0
        self.kind = self.kind[keep]
        self.amount = self.amount[keep]

    def material_total(self):
        return float(
            np.sum(self.amount)
            + self.recycled_fuel_buffer
            + self.recycled_mineral_buffer
        )

    def totals_by_kind(self):
        totals = []
        for kind in range(3):
            totals.append(float(np.sum(self.amount[self.kind == kind])))
        return totals

    def step_diffusion(self, dt):
        if len(self.amount) == 0:
            return
        kind_diffusion = np.choose(
            self.kind,
            np.asarray([0.010, 0.008, 0.006], dtype=float),
        )
        noise = self.rng.normal(0.0, 1.0, self.pos.shape)
        self.pos = (self.pos + noise * np.sqrt(2.0 * kind_diffusion[:, None] * dt)) % 1.0

    def recycle_waste(self, dt):
        waste_indices = np.where(self.kind == PARTICLE_WASTE)[0]
        if waste_indices.size:
            conversion = self.amount[waste_indices] * (1.0 - math.exp(-0.025 * dt))
            self.amount[waste_indices] -= conversion
            total = float(np.sum(conversion))
            self.recycled_fuel_buffer += 0.65 * total
            self.recycled_mineral_buffer += 0.25 * total
            self.dissipated_material += 0.10 * total

        # Recycled matter re-enters as small, spatially distributed substrate.
        for kind, attribute in (
            (PARTICLE_FUEL, 'recycled_fuel_buffer'),
            (PARTICLE_MINERAL, 'recycled_mineral_buffer'),
        ):
            buffer_value = float(getattr(self, attribute))
            while buffer_value >= 0.025 and len(self.amount) < PARTICLE_LIMIT:
                self.add_particle(
                    kind,
                    self.rng.random(2),
                    0.025,
                    count_as_injection=False,
                )
                buffer_value -= 0.025
            setattr(self, attribute, buffer_value)

    def external_inflow(self, dt, enabled=True):
        if not enabled:
            return
        # Slow geochemical inflow.  This is explicitly logged as external
        # matter; it is not created inside a cell.
        for kind, rate in (
            (PARTICLE_FUEL, 0.0160),
            (PARTICLE_MINERAL, 0.0060),
        ):
            expected = rate * dt
            if self.rng.random() < expected / 0.020:
                self.add_particle(
                    kind,
                    self.rng.random(2),
                    0.020,
                    count_as_injection=True,
                )

    def state_dict(self):
        return {
            'pos': self.pos.copy(),
            'kind': self.kind.copy(),
            'amount': self.amount.copy(),
            'recycled_fuel_buffer': float(self.recycled_fuel_buffer),
            'recycled_mineral_buffer': float(self.recycled_mineral_buffer),
            'injected_material': float(self.injected_material),
            'dissipated_material': float(self.dissipated_material),
        }

    @classmethod
    def from_state(cls, rng, state):
        field = cls(rng, initial=False)
        field.pos = np.asarray(state['pos'], dtype=float).copy()
        field.kind = np.asarray(state['kind'], dtype=np.int16).copy()
        field.amount = np.asarray(state['amount'], dtype=float).copy()
        field.recycled_fuel_buffer = float(state.get('recycled_fuel_buffer', 0.0))
        field.recycled_mineral_buffer = float(state.get('recycled_mineral_buffer', 0.0))
        field.injected_material = float(state.get('injected_material', 0.0))
        field.dissipated_material = float(state.get('dissipated_material', 0.0))
        return field


# ---------------------------------------------------------------------------
# Self-producing cell
# ---------------------------------------------------------------------------


class ProtoCell(object):
    """A membrane-bounded catalytic reaction loop with no scalar health."""

    def __init__(self, cell_id, rng, position=None, generation=0, lineage=0):
        self.cell_id = int(cell_id)
        self.generation = int(generation)
        self.lineage = int(lineage)
        self.pos = np.asarray(
            rng.random(2) if position is None else position,
            dtype=float,
        ) % 1.0
        self.vel = rng.normal(0.0, 0.0015, 2)
        self.age = 0.0
        self.alive = True
        self.death_reason = ''

        base = INITIAL_SEGMENT_MASS
        self.membrane = np.clip(
            rng.normal(base, base * 0.035, MEMBRANE_SEGMENTS),
            base * 0.75,
            base * 1.25,
        )
        self.transporters = np.zeros((MEMBRANE_SEGMENTS, CHANNEL_COUNT), dtype=float)
        self.transporters[:, CHANNEL_FUEL] = base * 0.060
        self.transporters[:, CHANNEL_MINERAL] = base * 0.052
        self.transporters[:, CHANNEL_WASTE] = base * 0.038
        self.transporters *= rng.uniform(0.90, 1.10, self.transporters.shape)

        self.pools = np.zeros((POOL_COUNT,), dtype=float)
        self.pools[POOL_FUEL] = 0.46
        self.pools[POOL_MINERAL] = 0.39
        self.pools[POOL_ATP] = 0.32
        self.pools[POOL_MEM_PRECURSOR] = 0.15
        self.pools[POOL_CATALYST] = 0.34
        self.pools[POOL_TRANSPORTER_PRECURSOR] = 0.065
        self.pools[POOL_WASTE] = 0.035

        self.radius = BASE_RADIUS
        self.contact_trace = np.zeros((MEMBRANE_SEGMENTS, 2), dtype=float)
        self.damage_trace = np.zeros((MEMBRANE_SEGMENTS,), dtype=float)
        self.surface_flux = np.zeros(2, dtype=float)
        self.division_progress = 0.0
        self.septum_mass = 0.0
        self.division_axis = float(rng.uniform(0.0, 2.0 * math.pi))
        self.lysis_timer = 0.0
        self.network_silence_timer = 0.0
        self.low_energy_timer = 0.0
        self.last_catalysis = 0.0
        self.last_assembly = 0.0
        self.last_uptake = np.zeros(2, dtype=float)
        self.last_export = 0.0
        self.last_leak = 0.0
        self.reaction_events = 0
        self.punctures_survived = 0
        self.repair_reference = None

    # ----- Geometry and viability -------------------------------------------------

    def material_mass(self):
        return float(
            self.pools[POOL_FUEL]
            + self.pools[POOL_MINERAL]
            + self.pools[POOL_MEM_PRECURSOR]
            + self.pools[POOL_CATALYST]
            + self.pools[POOL_TRANSPORTER_PRECURSOR]
            + self.pools[POOL_WASTE]
            + np.sum(self.membrane)
            + np.sum(self.transporters)
            + self.septum_mass
        )

    def osmolyte(self):
        # Membrane material is excluded; all soluble pools contribute to
        # internal pressure.  ATP is an energetic molecule and contributes here.
        return float(np.sum(self.pools))

    def membrane_capacity_radius(self):
        return clamp(
            float(np.sum(self.membrane)) / (2.0 * math.pi * MEMBRANE_DENSITY),
            MIN_RADIUS,
            MAX_RADIUS,
        )

    def osmotic_target_radius(self):
        ratio = max(0.08, self.osmolyte() / TARGET_OSMOLYTE)
        return clamp(BASE_RADIUS * math.sqrt(ratio), MIN_RADIUS, MAX_RADIUS * 1.25)

    def update_radius(self, dt):
        capacity = self.membrane_capacity_radius()
        osmotic = self.osmotic_target_radius()
        target = capacity * clamp(osmotic / max(capacity, 1e-8), 0.66, MEMBRANE_STRETCH_LIMIT)
        target = clamp(target, MIN_RADIUS, MAX_RADIUS)
        self.radius += (target - self.radius) * (1.0 - math.exp(-2.8 * dt))
        self.radius = clamp(self.radius, MIN_RADIUS, MAX_RADIUS)

    def tension(self):
        capacity = self.membrane_capacity_radius()
        return max(0.0, self.osmotic_target_radius() / max(capacity, 1e-8) - 1.0)

    def required_segment_mass(self):
        return MEMBRANE_DENSITY * (2.0 * math.pi * self.radius / MEMBRANE_SEGMENTS)

    def closure_array(self):
        required = max(self.required_segment_mass(), 1e-9)
        # Smooth, continuous closure.  A segment at the required mass is nearly
        # sealed, while a genuinely missing segment remains a physical gap.
        return np.clip(self.membrane / (required * GAP_CLOSURE_THRESHOLD), 0.0, 1.0)

    def closure(self):
        return float(np.mean(self.closure_array()))

    def worst_gap(self):
        closure = self.closure_array()
        return float(1.0 - np.min(closure))

    def reaction_loop_strength(self):
        closure = self.closure()
        catalyst = float(self.pools[POOL_CATALYST])
        fuel = float(self.pools[POOL_FUEL])
        mineral = float(self.pools[POOL_MINERAL])
        atp = float(self.pools[POOL_ATP])
        transport = float(np.sum(self.transporters))
        energy_substrate = (fuel + atp) / (0.08 + fuel + atp)
        building_substrate = (
            mineral + self.pools[POOL_MEM_PRECURSOR]
        ) / (0.08 + mineral + self.pools[POOL_MEM_PRECURSOR])
        return float(
            closure
            * (catalyst / (0.08 + catalyst))
            * energy_substrate
            * building_substrate
            * (transport / (0.025 + transport))
        )

    def boundary_points(self):
        closure = self.closure_array()
        relative = self.membrane / max(float(np.mean(self.membrane)), 1e-9)
        relative = circular_smooth(relative, 0.24)
        angles = np.linspace(0.0, 2.0 * math.pi, MEMBRANE_SEGMENTS, endpoint=False)
        local_radius = self.radius * np.clip(0.86 + 0.15 * relative + 0.04 * closure, 0.72, 1.18)
        if self.division_progress > 0.0:
            # Two-lobed geometry appears gradually while the septum is being
            # produced.  The material ring itself remains the source of closure.
            projection = np.cos(angles - self.division_axis)
            local_radius *= 1.0 + 0.28 * self.division_progress * np.abs(projection)
            local_radius *= 1.0 - 0.16 * self.division_progress * (1.0 - np.abs(projection))
        points = self.pos[None, :] + np.stack(
            [np.cos(angles) * local_radius, np.sin(angles) * local_radius], axis=1
        )
        return points % 1.0

    # ----- Surface exchange -------------------------------------------------------

    def segment_for_delta(self, delta):
        angle = math.atan2(float(delta[1]), float(delta[0])) % (2.0 * math.pi)
        return int((angle / (2.0 * math.pi)) * MEMBRANE_SEGMENTS) % MEMBRANE_SEGMENTS

    def surface_exchange(self, field, dt, config):
        if not self.alive or len(field.amount) == 0:
            self.last_uptake[:] = 0.0
            return
        deltas = (field.pos - self.pos[None, :] + 0.5) % 1.0 - 0.5
        distances = np.linalg.norm(deltas, axis=1)
        interaction_band = 0.016
        candidates = np.where(distances < self.radius + interaction_band)[0]
        closure_values = self.closure_array()
        uptake = np.zeros(2, dtype=float)
        flux_vector = np.zeros(2, dtype=float)
        atp_spent = 0.0

        for index in candidates:
            amount = float(field.amount[index])
            if amount <= 1e-8:
                continue
            distance = float(distances[index])
            delta = deltas[index]
            if distance <= 1e-10:
                delta = np.array([1.0, 0.0], dtype=float)
                distance = 1e-10
            segment = self.segment_for_delta(delta)
            closure = float(closure_values[segment])
            gap = 1.0 - closure
            normal = delta / distance
            kind = int(field.kind[index])

            # A closed boundary excludes untransported particles.  The push is
            # geometric rather than a label saying "inside".
            if distance < self.radius * 0.985 and closure > 0.72:
                field.pos[index] = (self.pos + normal * (self.radius + 0.0025)) % 1.0

            near_surface = abs(distance - self.radius) <= interaction_band
            if not near_surface:
                continue

            if kind in (PARTICLE_FUEL, PARTICLE_MINERAL):
                channel = CHANNEL_FUEL if kind == PARTICLE_FUEL else CHANNEL_MINERAL
                trace_channel = 0 if kind == PARTICLE_FUEL else 1
                self.contact_trace[segment, trace_channel] += amount * dt * 3.0
                facilitated = 0.0
                powered = 0.0
                if config.transport:
                    protein_density = self.transporters[segment, channel] / max(
                        self.membrane[segment], 1e-8
                    )
                    saturation = amount / (0.018 + amount)
                    internal_pool = (
                        self.pools[POOL_FUEL]
                        if kind == PARTICLE_FUEL
                        else self.pools[POOL_MINERAL]
                    )
                    target_pool = 0.46 if kind == PARTICLE_FUEL else 0.40
                    chemical_gradient = 1.0 / (
                        1.0 + (internal_pool / target_pool) ** 4
                    )
                    # A transporter provides facilitated diffusion even when the
                    # ATP pool is temporarily empty.  ATP adds directional pump
                    # capacity, avoiding the unphysical deadlock in which a cell
                    # needs fuel to import the fuel needed to make ATP.
                    facilitated = (
                        0.62 * protein_density * closure * saturation * chemical_gradient * dt
                    )
                    if self.pools[POOL_ATP] > 1e-7:
                        powered = (
                            0.42 * protein_density * closure * saturation * chemical_gradient * dt
                        )
                        powered = min(
                            powered,
                            max(0.0, amount - facilitated),
                            max(0.0, self.pools[POOL_ATP] - 0.028) / 0.045,
                        )
                transported = min(amount, facilitated + powered)
                passive = min(
                    max(0.0, amount - transported),
                    0.075 * (gap ** 2.2) * amount * dt,
                )
                transfer = max(0.0, transported + passive)
                if transfer > 0.0:
                    field.amount[index] -= transfer
                    if kind == PARTICLE_FUEL:
                        self.pools[POOL_FUEL] += transfer
                        uptake[0] += transfer
                    else:
                        self.pools[POOL_MINERAL] += transfer
                        uptake[1] += transfer
                    powered_part = min(powered, transfer)
                    self.pools[POOL_ATP] -= 0.045 * powered_part
                    atp_spent += 0.045 * powered_part
                    flux_vector += normal * transfer

            elif kind == PARTICLE_WASTE and gap > 0.05:
                ingress = min(amount, 0.032 * gap * amount * dt)
                if ingress > 0.0:
                    field.amount[index] -= ingress
                    self.pools[POOL_WASTE] += ingress
                    self.damage_trace[segment] += ingress * 1.6

        self.pools[POOL_ATP] = max(0.0, self.pools[POOL_ATP])
        self.last_uptake[:] = uptake
        self.surface_flux += flux_vector
        if atp_spent > 0.0:
            self.reaction_events += 1

    def export_waste(self, field, dt, config):
        self.last_export = 0.0
        if not config.waste_export or self.pools[POOL_WASTE] <= 1e-8:
            return
        closure = self.closure_array()
        exporter_density = self.transporters[:, CHANNEL_WASTE] / np.maximum(self.membrane, 1e-8)
        capacity = float(np.sum(exporter_density * closure))
        saturation = self.pools[POOL_WASTE] / (0.05 + self.pools[POOL_WASTE])
        # Waste channels provide a downhill facilitated flux without ATP.  ATP
        # powers an additional pump component.  This prevents a chemically
        # absurd deadlock in which low ATP blocks waste removal, waste suppresses
        # catalysis, and the cell can never recover.
        passive = min(
            self.pools[POOL_WASTE],
            0.0075 * capacity * saturation * dt,
        )
        remaining = max(0.0, self.pools[POOL_WASTE] - passive)
        powered = min(
            remaining,
            0.014 * capacity * saturation * dt,
            max(0.0, self.pools[POOL_ATP] - 0.018) / 0.026,
        )
        amount = passive + powered
        if amount <= 0.0:
            return
        weights = exporter_density * closure + 1e-6
        weights /= float(np.sum(weights))
        # Use the strongest local exporter.  This is deterministic and avoids
        # consuming an unrelated global random stream.
        segment = int(np.argmax(weights))
        angle = 2.0 * math.pi * (segment + 0.5) / MEMBRANE_SEGMENTS
        position = (self.pos + unit_vector(angle) * (self.radius + 0.004)) % 1.0
        field.add_particle(PARTICLE_WASTE, position, amount, count_as_injection=False)
        self.pools[POOL_WASTE] -= amount
        self.pools[POOL_ATP] -= 0.026 * powered
        self.last_export = amount
        self.surface_flux -= unit_vector(angle) * amount * 0.55

    # ----- Chemistry --------------------------------------------------------------

    def _reaction_limit(self, desired, requirements):
        amount = max(0.0, float(desired))
        for pool_index, coefficient in requirements:
            if coefficient > 0.0:
                available = self.pools[pool_index]
                if pool_index == POOL_ATP:
                    # Biosynthesis cannot consume the entire energetic pool in
                    # one local reaction.  A small reserve keeps transport,
                    # export, and membrane asymmetry chemically possible.
                    available = max(0.0, available - 0.045)
                amount = min(amount, available / coefficient)
        return max(0.0, amount)

    def metabolism(self, world, dt, config):
        if not self.alive:
            return
        volume = max(0.20, (self.radius / BASE_RADIUS) ** 2)
        fuel = self.pools[POOL_FUEL] / volume
        mineral = self.pools[POOL_MINERAL] / volume
        catalyst = self.pools[POOL_CATALYST] / volume
        waste = self.pools[POOL_WASTE] / volume
        atp = self.pools[POOL_ATP]
        inhibition = 1.0 / (1.0 + 2.8 * waste)

        # 1) Catalytic energy production: fuel material becomes waste material;
        # ATP is an energy carrier and therefore not part of the matter ledger.
        cat_rate = (
            0.12
            * catalyst
            * fuel / (0.16 + fuel)
            * inhibition
        )
        cat_amount = min(self.pools[POOL_FUEL], cat_rate * dt)
        self.pools[POOL_FUEL] -= cat_amount
        self.pools[POOL_WASTE] += cat_amount
        self.pools[POOL_ATP] += 2.10 * cat_amount
        self.last_catalysis = cat_amount / max(dt, 1e-9)
        if cat_amount > 1e-8:
            self.reaction_events += 1

        # ATP maintenance reflects the cost of keeping catalysts folded,
        # transporters cycling, and a membrane chemically asymmetric.
        maintenance = dt * (
            0.0045
            + 0.010 * self.pools[POOL_CATALYST]
            + 0.010 * float(np.sum(self.transporters))
            + 0.0025 * float(np.sum(self.membrane))
            + 0.008 * self.tension()
        )
        paid = min(self.pools[POOL_ATP], maintenance)
        self.pools[POOL_ATP] -= paid
        maintenance_shortfall = max(0.0, maintenance - paid)

        # 2) Membrane precursor.  This reaction consumes both major external
        # matter classes and preserves their total mass.
        need_membrane = clamp(
            (1.03 - self.closure()) * 2.5
            + self.tension() * 2.2
            + max(0.0, 0.17 - self.pools[POOL_MEM_PRECURSOR]) * 1.5,
            0.0,
            2.5,
        )
        desired = dt * 0.095 * self.pools[POOL_CATALYST] * need_membrane
        amount = self._reaction_limit(
            desired,
            ((POOL_FUEL, 0.62), (POOL_MINERAL, 0.38), (POOL_ATP, 0.42)),
        )
        self.pools[POOL_FUEL] -= 0.62 * amount
        self.pools[POOL_MINERAL] -= 0.38 * amount
        self.pools[POOL_ATP] -= 0.42 * amount
        self.pools[POOL_MEM_PRECURSOR] += amount

        # 3) Catalysts catalyse their own replacement.  This is the catalytic
        # arm of the closure: without existing catalyst, new catalyst is not
        # produced at an appreciable rate.
        if config.catalyst_synthesis:
            size_ratio = max(1.0, (self.radius / BASE_RADIUS) ** 2)
            target_cat = min(0.68, 0.34 + 0.30 * (size_ratio - 1.0))
            cat_need = clamp((target_cat - self.pools[POOL_CATALYST]) / 0.20, 0.0, 1.5)
            desired = dt * 0.050 * self.pools[POOL_CATALYST] * cat_need
            amount = self._reaction_limit(
                desired,
                ((POOL_FUEL, 0.70), (POOL_MINERAL, 0.30), (POOL_ATP, 0.80)),
            )
            self.pools[POOL_FUEL] -= 0.70 * amount
            self.pools[POOL_MINERAL] -= 0.30 * amount
            self.pools[POOL_ATP] -= 0.80 * amount
            self.pools[POOL_CATALYST] += amount

        # 4) Transporter precursor.  Which channel receives it is decided by
        # current embodied shortages and local contact traces below.
        transport_need = clamp(
            0.7 * max(0.0, 0.22 - self.pools[POOL_FUEL])
            + 0.8 * max(0.0, 0.19 - self.pools[POOL_MINERAL])
            + 0.9 * max(0.0, self.pools[POOL_WASTE] - 0.06)
            + 0.3,
            0.0,
            1.5,
        )
        desired = dt * 0.035 * self.pools[POOL_CATALYST] * transport_need
        amount = self._reaction_limit(
            desired,
            ((POOL_FUEL, 0.65), (POOL_MINERAL, 0.35), (POOL_ATP, 0.70)),
        )
        self.pools[POOL_FUEL] -= 0.65 * amount
        self.pools[POOL_MINERAL] -= 0.35 * amount
        self.pools[POOL_ATP] -= 0.70 * amount
        self.pools[POOL_TRANSPORTER_PRECURSOR] += amount

        # 5) Membrane precursor self-assembles into the ring.  No scalar repair
        # command is issued: local thinness, tension, and damage traces determine
        # where matter is incorporated.
        self.last_assembly = 0.0
        if config.membrane_synthesis and self.pools[POOL_MEM_PRECURSOR] > 1e-9:
            required = self.required_segment_mass()
            deficit = np.maximum(required * 1.06 - self.membrane, 0.0)
            if config.targeted_repair:
                weights = deficit * 8.0 + self.damage_trace * 2.5 + 0.02
            else:
                weights = np.ones(MEMBRANE_SEGMENTS, dtype=float)
            # Once the ring is closed, surplus precursor supports growth.
            if float(np.sum(deficit)) < 1e-5:
                weights += 0.18
            weights = np.maximum(weights, 1e-8)
            weights /= float(np.sum(weights))
            assembly_rate = 0.080 * self.pools[POOL_CATALYST] * (
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

        # 6) Transporter proteins are inserted into local membrane.  Contact
        # traces embody where useful external molecules were actually found.
        if self.pools[POOL_TRANSPORTER_PRECURSOR] > 1e-9:
            channel_need = np.asarray([
                max(0.08, 0.34 - self.pools[POOL_FUEL]),
                max(0.08, 0.29 - self.pools[POOL_MINERAL]),
                max(0.05, self.pools[POOL_WASTE] * 1.4),
            ])
            channel_need /= float(np.sum(channel_need))
            insert = min(
                self.pools[POOL_TRANSPORTER_PRECURSOR],
                0.038 * self.pools[POOL_CATALYST] * dt,
                max(0.0, self.pools[POOL_ATP] - 0.045) / 0.40,
            )
            if insert > 0.0:
                for channel in range(CHANNEL_COUNT):
                    if channel == CHANNEL_FUEL:
                        local = self.contact_trace[:, 0] + 0.03
                    elif channel == CHANNEL_MINERAL:
                        local = self.contact_trace[:, 1] + 0.03
                    else:
                        local = self.closure_array() + self.damage_trace + 0.03
                    local = np.maximum(local, 1e-8)
                    local /= float(np.sum(local))
                    self.transporters[:, channel] += local * insert * channel_need[channel]
                self.pools[POOL_TRANSPORTER_PRECURSOR] -= insert
                self.pools[POOL_ATP] -= 0.40 * insert

        # Lateral mobility makes the membrane a material surface rather than a
        # set of permanently labelled slots, while remaining slow enough for a
        # puncture to be a real local event.
        self.membrane = circular_smooth(self.membrane, min(0.49, 0.075 * dt))
        for channel in range(CHANNEL_COUNT):
            self.transporters[:, channel] = circular_smooth(
                self.transporters[:, channel], min(0.49, 0.10 * dt)
            )

        # Chemical wear.  Material is not deleted: degraded components become
        # waste that must be exported or will damage the boundary further.
        waste_stress = self.pools[POOL_WASTE] / max(0.20, volume)
        tension = self.tension()
        membrane_decay_rate = 0.00032 + 0.0010 * waste_stress + 0.0012 * tension
        if maintenance_shortfall > 0.0:
            membrane_decay_rate += 0.015 * maintenance_shortfall / max(dt, 1e-9)
        membrane_loss = np.minimum(
            self.membrane,
            self.membrane * membrane_decay_rate * dt
            + self.damage_trace * 0.0008 * dt,
        )
        self.membrane -= membrane_loss
        self.pools[POOL_WASTE] += float(np.sum(membrane_loss))

        cat_decay = min(
            self.pools[POOL_CATALYST],
            self.pools[POOL_CATALYST]
            * (0.00065 + 0.0017 * waste_stress + 0.002 * maintenance_shortfall)
            * dt,
        )
        self.pools[POOL_CATALYST] -= cat_decay
        self.pools[POOL_WASTE] += cat_decay

        transporter_decay = np.minimum(
            self.transporters,
            self.transporters
            * (0.00075 + 0.0010 * waste_stress + 0.0015 * maintenance_shortfall)
            * dt,
        )
        self.transporters -= transporter_decay
        self.pools[POOL_WASTE] += float(np.sum(transporter_decay))

        # ATP spontaneously dissipates as heat; this is energy loss, not matter
        # loss.  A finite cap prevents an unphysical energy reservoir.
        self.pools[POOL_ATP] *= math.exp(-0.018 * dt)
        self.pools[POOL_ATP] = clamp(self.pools[POOL_ATP], 0.0, 1.6)

        self.contact_trace *= math.exp(-0.80 * dt)
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
        leak_fraction = (0.065 * gap_strength + 0.080 * gap_strength * pressure) * dt
        leak_fraction = clamp(leak_fraction, 0.0, 0.18)
        self.last_leak = 0.0
        if leak_fraction <= 0.0:
            return
        segment = int(np.argmax(gap_weights))
        angle = 2.0 * math.pi * (segment + 0.5) / MEMBRANE_SEGMENTS
        position = (self.pos + unit_vector(angle) * (self.radius + 0.004)) % 1.0

        # Soluble material leaks according to the same physical opening.  ATP
        # dissipates; complex internal material becomes environmental waste.
        mapping = (
            (POOL_FUEL, PARTICLE_FUEL),
            (POOL_MINERAL, PARTICLE_MINERAL),
            (POOL_MEM_PRECURSOR, PARTICLE_WASTE),
            (POOL_CATALYST, PARTICLE_WASTE),
            (POOL_TRANSPORTER_PRECURSOR, PARTICLE_WASTE),
            (POOL_WASTE, PARTICLE_WASTE),
        )
        total = 0.0
        for pool_index, particle_kind in mapping:
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
            total += amount
        atp_loss = self.pools[POOL_ATP] * leak_fraction * 1.4
        self.pools[POOL_ATP] -= atp_loss
        world.dissipated_energy += atp_loss
        self.last_leak = total

    def update_motion(self, rng, dt):
        # No neural motor exists in 0.1.  Motion emerges only from surface flux,
        # Brownian forcing, and collisions handled by the world.
        thrust = self.surface_flux * 0.11
        brownian = rng.normal(0.0, 0.00075, 2) / max(0.65, self.radius / BASE_RADIUS)
        self.vel += thrust + brownian
        self.surface_flux *= math.exp(-2.4 * dt)
        speed = float(np.linalg.norm(self.vel))
        max_speed = 0.020 / max(0.7, self.radius / BASE_RADIUS)
        if speed > max_speed:
            self.vel *= max_speed / speed
        self.vel *= math.exp(-1.8 * dt)
        self.pos = (self.pos + self.vel * dt) % 1.0

    # ----- Division, damage, and death -------------------------------------------

    def ready_for_division(self):
        if not self.alive:
            return False
        return bool(
            self.closure() > 0.965
            and self.worst_gap() < 0.22
            and float(np.sum(self.membrane)) > INITIAL_MEMBRANE_MASS * 1.68
            and self.pools[POOL_CATALYST] > 0.50
            and self.pools[POOL_ATP] > 0.043
            and self.pools[POOL_MEM_PRECURSOR] > 0.20
            and self.material_mass() > 5.50
            and self.age > 60.0
        )

    def update_division(self, dt, config):
        if not config.division:
            return_amount = min(self.septum_mass, 0.008 * dt)
            self.septum_mass -= return_amount
            self.pools[POOL_MEM_PRECURSOR] += return_amount
            self.division_progress = clamp(self.septum_mass / 0.12, 0.0, 1.0)
            return
        if self.ready_for_division() or self.division_progress > 0.0:
            # Building a septum consumes precursor and ATP.  If chemistry can no
            # longer pay, division stalls and can regress instead of occurring by
            # a timer alone.
            precursor_use = min(
                self.pools[POOL_MEM_PRECURSOR],
                0.020 * dt,
                self.pools[POOL_ATP] / 0.65,
            )
            if precursor_use > 0.0 and self.closure() > 0.90:
                self.pools[POOL_MEM_PRECURSOR] -= precursor_use
                self.pools[POOL_ATP] -= 0.65 * precursor_use
                # Septum is explicit material, not a progress-only counter.
                self.septum_mass += precursor_use
            else:
                # A stalled septum slowly disassembles back into reusable
                # precursor; no matter disappears merely because division stops.
                return_amount = min(self.septum_mass, 0.006 * dt)
                self.septum_mass -= return_amount
                self.pools[POOL_MEM_PRECURSOR] += return_amount
            self.division_progress = clamp(self.septum_mass / 0.12, 0.0, 1.0)
        else:
            return_amount = min(self.septum_mass, 0.004 * dt)
            self.septum_mass -= return_amount
            self.pools[POOL_MEM_PRECURSOR] += return_amount
            self.division_progress = clamp(self.septum_mass / 0.12, 0.0, 1.0)

    def can_split(self):
        return bool(self.alive and self.division_progress >= 0.999)

    def split(self, world):
        """Return two daughters while conserving all material to roundoff."""
        if not self.can_split():
            return None
        before = self.material_mass()
        axis = unit_vector(self.division_axis)
        offset = axis * max(0.024, self.radius * 0.46)
        daughters = []
        asymmetry = clamp(float(world.rng.normal(0.5, 0.025)), 0.44, 0.56)
        fractions = (asymmetry, 1.0 - asymmetry)

        # A small amount of cytoplasmic matter is shed as environmental waste;
        # this is explicit and included in the division ledger.
        shed_fraction = 0.012
        shed_mass = before * shed_fraction
        remaining_factor = 1.0 - shed_fraction

        parent_atp = float(self.pools[POOL_ATP])
        world.dissipated_energy += parent_atp * shed_fraction
        for daughter_index, fraction in enumerate(fractions):
            daughter = ProtoCell(
                world.next_cell_id + daughter_index,
                world.rng,
                position=(self.pos + (1.0 if daughter_index == 0 else -1.0) * offset) % 1.0,
                generation=self.generation + 1,
                lineage=self.lineage,
            )
            # Parent surface is shared; septum progress contributes extra ring
            # material equally to the two daughters.
            phase = 0 if daughter_index == 0 else MEMBRANE_SEGMENTS // 2
            parent_membrane = np.roll(self.membrane, phase)
            parent_transporters = np.roll(self.transporters, phase, axis=0)
            daughter.membrane = parent_membrane * fraction * remaining_factor
            daughter.membrane += (
                self.septum_mass * fraction * remaining_factor
                / MEMBRANE_SEGMENTS
            )
            daughter.transporters = parent_transporters * fraction * remaining_factor
            daughter.pools = self.pools * fraction * remaining_factor
            daughter.pools[POOL_ATP] *= 0.94
            world.dissipated_energy += float(
                self.pools[POOL_ATP] * fraction * remaining_factor * 0.06
            )
            # Septum material, already paid for by precursor consumption, is
            # represented by a closed-ring bonus.  To conserve matter, it is not
            # created here; the parent precursor was converted to division
            # progress and excluded from material mass at payment time only as
            # precursor remained material.  We therefore do not add a bonus.
            daughter.radius = max(MIN_RADIUS, self.radius * math.sqrt(fraction))
            daughter.vel = self.vel + world.rng.normal(0.0, 0.002, 2)
            daughter.division_progress = 0.0
            daughter.septum_mass = 0.0
            daughter.division_axis = float(world.rng.uniform(0.0, 2.0 * math.pi))
            daughter.age = 0.0
            daughters.append(daughter)

        # Shed matter becomes external waste, preserving the material ledger.
        world.field.add_particle(
            PARTICLE_WASTE,
            self.pos,
            shed_mass,
            count_as_injection=False,
        )
        after = sum(d.material_mass() for d in daughters) + shed_mass
        residual = before - after
        world.division_parent_material += before
        world.division_daughter_material += sum(d.material_mass() for d in daughters)
        world.division_shed_material += shed_mass
        world.division_residual += residual
        world.next_cell_id += 2
        return daughters

    def puncture(self, angle, severity=0.62):
        segment = int((angle % (2.0 * math.pi)) / (2.0 * math.pi) * MEMBRANE_SEGMENTS)
        width = 2
        removed = 0.0
        for offset in range(-width, width + 1):
            index = (segment + offset) % MEMBRANE_SEGMENTS
            local = severity * math.exp(-0.65 * abs(offset))
            amount = self.membrane[index] * clamp(local, 0.0, 0.90)
            self.membrane[index] -= amount
            removed += amount
            self.damage_trace[index] += local
            # Membrane-bound transporters are torn away with the local material.
            transporter_loss = self.transporters[index] * clamp(local * 0.75, 0.0, 0.80)
            self.transporters[index] -= transporter_loss
            removed += float(np.sum(transporter_loss))
        self.repair_reference = {
            'segment': int(segment),
            'closure_before': float(self.closure()),
            'removed': float(removed),
            'age': float(self.age),
        }
        return removed

    def environmental_damage(self, rng, dt, enabled=True):
        if not enabled or not self.alive:
            return 0.0
        # Rare, local shear/oxidative events.  Their hazard is independent of a
        # cell's stored "health" because no such variable exists.
        probability = dt * (0.0015 + 0.0020 * self.pools[POOL_WASTE])
        if rng.random() >= probability:
            return 0.0
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        return self.puncture(angle, severity=float(rng.uniform(0.18, 0.38)))

    def update_viability(self, dt):
        closure = self.closure()
        loop = self.reaction_loop_strength()
        if closure < 0.33:
            self.lysis_timer += dt
        else:
            self.lysis_timer = max(0.0, self.lysis_timer - 0.65 * dt)

        if self.last_catalysis < 1e-5 and loop < 0.045:
            self.network_silence_timer += dt
        else:
            self.network_silence_timer = max(0.0, self.network_silence_timer - 0.35 * dt)

        if self.pools[POOL_ATP] < 0.006:
            self.low_energy_timer += dt
        else:
            self.low_energy_timer = max(0.0, self.low_energy_timer - 0.4 * dt)

        if self.lysis_timer > 3.5 and self.material_mass() < 1.65:
            self.alive = False
            self.death_reason = 'boundary dissolution'
        elif self.network_silence_timer > 22.0 and self.pools[POOL_CATALYST] < 0.025:
            self.alive = False
            self.death_reason = 'catalytic closure lost'
        elif self.low_energy_timer > 35.0 and self.closure() < 0.58:
            self.alive = False
            self.death_reason = 'energy-boundary collapse'

    def state_dict(self):
        return {
            'cell_id': self.cell_id,
            'generation': self.generation,
            'lineage': self.lineage,
            'pos': self.pos.copy(),
            'vel': self.vel.copy(),
            'age': float(self.age),
            'alive': bool(self.alive),
            'death_reason': self.death_reason,
            'membrane': self.membrane.copy(),
            'transporters': self.transporters.copy(),
            'pools': self.pools.copy(),
            'radius': float(self.radius),
            'contact_trace': self.contact_trace.copy(),
            'damage_trace': self.damage_trace.copy(),
            'surface_flux': self.surface_flux.copy(),
            'division_progress': float(self.division_progress),
            'septum_mass': float(self.septum_mass),
            'division_axis': float(self.division_axis),
            'lysis_timer': float(self.lysis_timer),
            'network_silence_timer': float(self.network_silence_timer),
            'low_energy_timer': float(self.low_energy_timer),
            'last_catalysis': float(self.last_catalysis),
            'last_assembly': float(self.last_assembly),
            'last_uptake': self.last_uptake.copy(),
            'last_export': float(self.last_export),
            'last_leak': float(self.last_leak),
            'reaction_events': int(self.reaction_events),
            'punctures_survived': int(self.punctures_survived),
            'repair_reference': self.repair_reference,
        }

    @classmethod
    def from_state(cls, rng, state):
        cell = cls(
            int(state['cell_id']),
            rng,
            position=state['pos'],
            generation=int(state.get('generation', 0)),
            lineage=int(state.get('lineage', 0)),
        )
        for name in (
            'pos', 'vel', 'membrane', 'transporters', 'pools',
            'contact_trace', 'damage_trace', 'surface_flux', 'last_uptake',
        ):
            setattr(cell, name, np.asarray(state[name], dtype=float).copy())
        for name in (
            'age', 'radius', 'division_progress', 'septum_mass', 'division_axis',
            'lysis_timer', 'network_silence_timer', 'low_energy_timer',
            'last_catalysis', 'last_assembly', 'last_export', 'last_leak',
        ):
            setattr(cell, name, float(state.get(name, getattr(cell, name))))
        cell.alive = bool(state.get('alive', True))
        cell.death_reason = str(state.get('death_reason', ''))
        cell.reaction_events = int(state.get('reaction_events', 0))
        cell.punctures_survived = int(state.get('punctures_survived', 0))
        cell.repair_reference = state.get('repair_reference')
        return cell


# ---------------------------------------------------------------------------
# World / laboratory
# ---------------------------------------------------------------------------


class SomaCellWorld(object):
    def __init__(self, seed=101, initial_cells=1, config=None):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.config = config if config is not None else CellConfig()
        self.age = 0.0
        self.field = ParticleField(self.rng, initial=True)
        self.cells = []
        self.next_cell_id = 0
        for index in range(int(initial_cells)):
            angle = 2.0 * math.pi * index / max(1, int(initial_cells))
            position = np.array([
                0.5 + 0.12 * math.cos(angle),
                0.5 + 0.12 * math.sin(angle),
            ]) % 1.0
            self.cells.append(
                ProtoCell(
                    self.next_cell_id,
                    self.rng,
                    position=position,
                    generation=0,
                    lineage=index,
                )
            )
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
        self.initial_total_material = self.total_material()
        self.last_step_material_residual = 0.0

    def living_cells(self):
        return [cell for cell in self.cells if cell.alive]

    def total_cell_material(self):
        return float(sum(cell.material_mass() for cell in self.cells if cell.alive))

    def total_material(self):
        return float(self.field.material_total() + self.total_cell_material())

    def matter_ledger_expected(self):
        return float(
            self.initial_total_material
            + self.field.injected_material
            - self.field.dissipated_material
        )

    def matter_ledger_residual(self):
        return float(self.total_material() - self.matter_ledger_expected())

    def _particle_interactions(self, dt):
        for cell in self.living_cells():
            cell.surface_exchange(self.field, dt, self.config)

    def _resolve_collisions(self):
        alive = self.living_cells()
        for index, first in enumerate(alive):
            for second in alive[index + 1:]:
                delta = wrapped_delta(first.pos, second.pos)
                distance = float(np.linalg.norm(delta))
                minimum = first.radius + second.radius + 0.004
                if distance <= 1e-9 or distance >= minimum:
                    continue
                normal = delta / distance
                overlap = minimum - distance
                first.pos = (first.pos - normal * overlap * 0.5) % 1.0
                second.pos = (second.pos + normal * overlap * 0.5) % 1.0
                first.vel -= normal * overlap * 0.04
                second.vel += normal * overlap * 0.04

    def _release_dead_cell(self, cell):
        total = cell.material_mass()
        # Cytoplasmic fuel and minerals retain their kinds; complex matter and
        # the membrane return as waste/debris.
        releases = (
            (PARTICLE_FUEL, cell.pools[POOL_FUEL]),
            (PARTICLE_MINERAL, cell.pools[POOL_MINERAL]),
            (
                PARTICLE_WASTE,
                cell.pools[POOL_MEM_PRECURSOR]
                + cell.pools[POOL_CATALYST]
                + cell.pools[POOL_TRANSPORTER_PRECURSOR]
                + cell.pools[POOL_WASTE]
                + float(np.sum(cell.membrane))
                + float(np.sum(cell.transporters))
                + float(cell.septum_mass),
            ),
        )
        for kind, amount in releases:
            if amount <= 1e-9:
                continue
            count = max(1, min(12, int(math.ceil(amount / 0.035))))
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
        dt = clamp(dt, 1e-5, 0.10)
        before = self.total_material()
        self.field.step_diffusion(dt)
        self.field.recycle_waste(dt)
        self.field.external_inflow(dt, enabled=self.config.external_inflow)
        self._particle_interactions(dt)

        for cell in self.living_cells():
            removed = cell.environmental_damage(
                self.rng, dt, enabled=self.config.environmental_damage
            )
            if removed > 0.0:
                self.field.add_particle(PARTICLE_WASTE, cell.pos, removed, count_as_injection=False)
                self.damage_events += 1
            cell.metabolism(self, dt, self.config)

        self._resolve_collisions()
        self._handle_divisions_and_deaths()
        self.field.compact()
        self.age += dt
        after = self.total_material()
        external_delta = self.field.injected_material
        # The authoritative residual uses the cumulative ledger.  The step
        # residual is useful for detecting a sudden implementation mistake.
        self.last_step_material_residual = after - before
        if not self.finite():
            raise FloatingPointError('non-finite SOMA-CELL state')

    def puncture_nearest(self, position, severity=0.62):
        alive = self.living_cells()
        if not alive:
            return False
        position = np.asarray(position, dtype=float) % 1.0
        cell = min(alive, key=lambda candidate: torus_distance(candidate.pos, position))
        delta = wrapped_delta(cell.pos, position)
        if float(np.linalg.norm(delta)) > cell.radius * 1.65:
            return False
        angle = math.atan2(float(delta[1]), float(delta[0]))
        removed = cell.puncture(angle, severity=severity)
        self.field.add_particle(PARTICLE_WASTE, position, removed, count_as_injection=False)
        self.manual_punctures += 1
        return True

    def inject_cloud(self, position):
        self.field.add_cloud(position)
        self.manual_injections += 1

    def finite(self):
        if not finite_array(self.field.pos) or not finite_array(self.field.amount):
            return False
        for cell in self.cells:
            for value in (
                cell.pos, cell.vel, cell.membrane, cell.transporters,
                cell.pools, cell.contact_trace, cell.damage_trace,
            ):
                if not finite_array(value):
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
        fuel, mineral, waste = self.field.totals_by_kind()
        return {
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
            'cell_material': self.total_cell_material(),
            'external_material': self.field.material_total(),
            'total_material': self.total_material(),
            'matter_residual': self.matter_ledger_residual(),
            'division_residual': float(self.division_residual),
            'dissipated_energy': float(self.dissipated_energy),
            'damage_events': int(self.damage_events),
            'manual_punctures': int(self.manual_punctures),
            'manual_injections': int(self.manual_injections),
        }

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
            'initial_total_material': float(self.initial_total_material),
            'last_step_material_residual': float(self.last_step_material_residual),
        }

    @classmethod
    def from_state(cls, state):
        world = cls(
            seed=int(state['seed']),
            initial_cells=0,
            config=CellConfig.from_state(state.get('config', {})),
        )
        world.rng.bit_generator.state = state['rng_state']
        world.age = float(state['age'])
        world.field = ParticleField.from_state(world.rng, state['field'])
        world.cells = [ProtoCell.from_state(world.rng, item) for item in state['cells']]
        # ProtoCell constructors allocate arrays using the RNG.  Reset to the
        # saved state so loading is a deterministic continuation, not a hidden
        # random intervention.
        world.rng.bit_generator.state = state['rng_state']
        for name in (
            'next_cell_id', 'births', 'divisions', 'deaths', 'manual_injections',
            'manual_punctures', 'damage_events',
        ):
            setattr(world, name, int(state.get(name, getattr(world, name))))
        for name in (
            'dissipated_energy', 'division_parent_material',
            'division_daughter_material', 'division_shed_material',
            'division_residual', 'released_dead_material',
            'initial_total_material', 'last_step_material_residual',
        ):
            setattr(world, name, float(state.get(name, getattr(world, name))))
        world.last_deaths = list(state.get('last_deaths', []))
        world.last_births = list(state.get('last_births', []))
        return world

    def save(self, path=SAVE_FILE):
        _atomic_pickle(path, self.state_dict())

    @classmethod
    def load(cls, path=SAVE_FILE):
        with open(path, 'rb') as handle:
            state = pickle.load(handle)
        return cls.from_state(state)

    def clone(self):
        return SomaCellWorld.from_state(self.state_dict())


# ---------------------------------------------------------------------------
# Long-run logging and report
# ---------------------------------------------------------------------------


LOG_FIELDS = (
    'session_id', 'build', 'wall_utc', 'reason', 'seed', 'age', 'fps',
    'sim_rate', 'memory_peak_mb', 'cells', 'births', 'divisions', 'deaths',
    'max_generation', 'mean_closure', 'min_closure', 'mean_atp',
    'mean_catalyst', 'mean_loop', 'mean_membrane', 'division_progress',
    'particles', 'external_fuel', 'external_mineral', 'external_waste',
    'cell_material', 'external_material', 'total_material',
    'matter_residual', 'division_residual', 'dissipated_energy',
    'damage_events', 'manual_punctures', 'manual_injections',
)


class LongRunLogger(object):
    def __init__(self, world, path=LOG_FILE):
        self.path = path
        nonce = int((time.time_ns() // 1000) % 9000) + 1000
        self.session_id = '{}-{}-{}'.format(
            int(time.time()), world.seed, nonce
        )
        self.last_age = -1e9
        self.rows = 0
        self.status = 'READY'

    def log(self, world, fps=0.0, sim_rate=0.0, reason='interval', force=False):
        if not force and world.age - self.last_age < LOG_INTERVAL:
            return False
        summary = world.summary()
        row = {
            'session_id': self.session_id,
            'build': BUILD,
            'wall_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'reason': reason,
            'seed': world.seed,
            'age': summary['age'],
            'fps': float(fps),
            'sim_rate': float(sim_rate),
            'memory_peak_mb': _memory_peak_mb_estimate(),
        }
        for field in LOG_FIELDS:
            if field not in row:
                row[field] = summary.get(field, 0.0)
        exists = os.path.exists(self.path) and os.path.getsize(self.path) > 0
        try:
            with open(self.path, 'a', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
            self.last_age = world.age
            self.rows += 1
            self.status = 'OK'
            return True
        except Exception:
            self.status = 'ERR'
            return False


def generate_report(log_path=LOG_FILE, report_path=REPORT_FILE, session_path=SESSION_FILE):
    if not os.path.exists(log_path):
        return 'NO LOG'
    rows = []
    try:
        with open(log_path, 'r', newline='') as handle:
            for row in csv.DictReader(handle):
                converted = dict(row)
                for key in LOG_FIELDS:
                    if key in ('session_id', 'build', 'wall_utc', 'reason'):
                        continue
                    try:
                        converted[key] = float(converted[key])
                    except Exception:
                        converted[key] = float('nan')
                rows.append(converted)
    except Exception:
        return 'READ ERR'
    if not rows:
        return 'EMPTY'

    sessions = {}
    for row in rows:
        sessions.setdefault(row['session_id'], []).append(row)
    session_rows = []
    lines = [
        'SOMA-CELL 0.1 long-run report',
        'generated: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())),
        'sessions: {}'.format(len(sessions)),
        '',
    ]
    for session_id, group in sessions.items():
        group.sort(key=lambda item: item['age'])
        first, last = group[0], group[-1]
        max_cells = int(max(item['cells'] for item in group))
        min_closure = min(item['min_closure'] for item in group)
        max_generation = int(max(item['max_generation'] for item in group))
        extinct = int(last['cells'] <= 0.0)
        max_residual = max(abs(item['matter_residual']) for item in group)
        session_row = {
            'session_id': session_id,
            'duration_s': last['age'] - first['age'],
            'final_cells': int(last['cells']),
            'max_cells': max_cells,
            'births': int(last['births']),
            'divisions': int(last['divisions']),
            'deaths': int(last['deaths']),
            'max_generation': max_generation,
            'min_closure': min_closure,
            'final_mean_loop': last['mean_loop'],
            'max_abs_matter_residual': max_residual,
            'extinct': extinct,
        }
        session_rows.append(session_row)
        lines.extend([
            '[{}]'.format(session_id),
            ' duration: {:.1f}s'.format(session_row['duration_s']),
            ' cells: final {} / max {}'.format(session_row['final_cells'], max_cells),
            ' births {} / divisions {} / deaths {} / max generation G{}'.format(
                session_row['births'], session_row['divisions'],
                session_row['deaths'], max_generation
            ),
            ' minimum mean-segment closure: {:.4f}'.format(min_closure),
            ' final reaction-loop strength: {:.4f}'.format(last['mean_loop']),
            ' max |matter ledger residual|: {:.3e}'.format(max_residual),
            ' extinct: {}'.format(bool(extinct)),
            '',
        ])

    try:
        with open(report_path, 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(lines))
        fields = list(session_rows[0].keys())
        with open(session_path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(session_rows)
    except Exception:
        return 'WRITE ERR'
    return 'OK'


# ---------------------------------------------------------------------------
# Headless entry points used by validation and desktop experiments
# ---------------------------------------------------------------------------


def run_headless_trial(seed=101, seconds=120.0, initial_cells=1, config=None):
    world = SomaCellWorld(seed=seed, initial_cells=initial_cells, config=config)
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

    class SomaCellScene(Scene):
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
                    self.world = SomaCellWorld.load(SAVE_FILE)
                    self.save_status = 'LOAD'
                else:
                    self.world = SomaCellWorld(seed=101, initial_cells=1)
            except Exception:
                self.world = SomaCellWorld(seed=101, initial_cells=1)
                self.save_status = 'RECOVER'
            self.logger = LongRunLogger(self.world)
            self.logger.log(self.world, reason='start', force=True)

        def _world_rect(self):
            width, height = float(self.size.w), float(self.size.h)
            left, right = 34.0, width - 34.0
            bottom, top = 70.0, height - 82.0
            return left, bottom, right - left, top - bottom

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
                max_steps = 7
                steps = 0
                while self.accumulator >= fixed and steps < max_steps:
                    self.world.step(fixed)
                    self.accumulator -= fixed
                    steps += 1
                if steps >= max_steps:
                    self.accumulator = min(self.accumulator, fixed)
            self.telemetry_frames += 1
            telemetry_elapsed = now - self.telemetry_wall
            if telemetry_elapsed >= 1.0:
                self.fps = self.telemetry_frames / telemetry_elapsed
                self.sim_rate = (self.world.age - self.telemetry_age) / telemetry_elapsed
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
            )
            for index in range(len(self.world.field.amount)):
                x, y = self._screen(self.world.field.pos[index])
                amount = float(self.world.field.amount[index])
                radius = 1.6 + 9.0 * math.sqrt(clamp(amount / 0.045, 0.0, 1.7))
                colour = colours[int(self.world.field.kind[index])]
                fill(colour[0], colour[1], colour[2], 0.78)
                ellipse(x - radius, y - radius, radius * 2.0, radius * 2.0)

        def _draw_cell(self, cell):
            left, bottom, width, height = self._world_rect()
            centre_x, centre_y = self._screen(cell.pos)
            scale_x, scale_y = width, height
            closure = cell.closure_array()
            points = cell.boundary_points()

            # Cytoplasm colour is derived from chemical state, not stored health.
            atp = clamp(cell.pools[POOL_ATP] / 0.60, 0.0, 1.0)
            waste = clamp(cell.pools[POOL_WASTE] / 0.24, 0.0, 1.0)
            screen_radius = cell.radius * min(scale_x, scale_y)
            fill(0.10 + 0.10 * waste, 0.28 + 0.32 * atp, 0.36 + 0.30 * atp, 0.28)
            ellipse(
                centre_x - screen_radius,
                centre_y - screen_radius,
                screen_radius * 2.0,
                screen_radius * 2.0,
            )

            # Catalyst and waste pools are shown as internal material clouds.
            catalyst_dots = max(1, min(14, int(round(cell.pools[POOL_CATALYST] * 24))))
            for index in range(catalyst_dots):
                angle = 2.399963 * index + cell.cell_id
                radial = screen_radius * 0.18 * math.sqrt((index + 1.0) / catalyst_dots)
                x = centre_x + math.cos(angle) * radial
                y = centre_y + math.sin(angle) * radial
                fill(0.80, 0.55, 1.0, 0.78)
                ellipse(x - 1.8, y - 1.8, 3.6, 3.6)
            if waste > 0.02:
                fill(1.0, 0.25, 0.25, 0.30 + 0.35 * waste)
                waste_radius = 2.0 + 8.0 * waste
                ellipse(
                    centre_x - waste_radius,
                    centre_y - waste_radius,
                    waste_radius * 2.0,
                    waste_radius * 2.0,
                )

            # Draw each material segment independently so physical gaps are
            # visible.  Wrapped cells near a world edge are displayed in their
            # principal copy; boundary crossings are rare at this cell scale.
            for index in range(MEMBRANE_SEGMENTS):
                next_index = (index + 1) % MEMBRANE_SEGMENTS
                if closure[index] < 0.055 and closure[next_index] < 0.055:
                    continue
                p0 = points[index]
                p1 = points[next_index]
                # Avoid drawing a torus-spanning chord when a cell straddles an edge.
                delta = wrapped_delta(p0, p1)
                p1_unwrapped = p0 + delta
                x0, y0 = self._screen(p0)
                x1 = left + p1_unwrapped[0] * width
                y1 = bottom + p1_unwrapped[1] * height
                local_closure = 0.5 * (closure[index] + closure[next_index])
                stroke(
                    0.38 + 0.45 * local_closure,
                    0.82 + 0.16 * local_closure,
                    0.94,
                    0.30 + 0.70 * local_closure,
                )
                stroke_weight(0.6 + 3.2 * local_closure)
                line(x0, y0, x1, y1)

                # Transporter proteins are actual membrane mass.  Their colours
                # encode fuel, mineral, and waste channels.
                if index % 2 == 0 and local_closure > 0.25:
                    transporter = cell.transporters[index]
                    channel = int(np.argmax(transporter))
                    density = float(np.sum(transporter)) / max(cell.membrane[index], 1e-8)
                    if density > 0.035:
                        colours = (
                            (0.28, 0.95, 0.45),
                            (1.00, 0.70, 0.24),
                            (1.00, 0.30, 0.55),
                        )
                        colour = colours[channel]
                        fill(colour[0], colour[1], colour[2], 0.86)
                        ellipse(x0 - 1.4, y0 - 1.4, 2.8, 2.8)

            if cell.division_progress > 0.02:
                normal = unit_vector(cell.division_axis + math.pi * 0.5)
                length = screen_radius * 0.65 * cell.division_progress
                stroke(0.98, 0.92, 0.55, 0.72)
                stroke_weight(1.0 + 2.0 * cell.division_progress)
                line(
                    centre_x - normal[0] * length,
                    centre_y - normal[1] * length,
                    centre_x + normal[0] * length,
                    centre_y + normal[1] * length,
                )

            fill(0.91, 0.97, 1.0)
            text(
                '#{} G{}  seal {:.2f}  ATP {:.2f}'.format(
                    cell.cell_id, cell.generation, cell.closure(), cell.pools[POOL_ATP]
                ),
                x=centre_x,
                y=centre_y - screen_radius - 10,
                font_size=8,
                alignment=5,
            )

        def draw(self):
            background(0.015, 0.025, 0.038)
            left, bottom, world_width, world_height = self._world_rect()
            fill(0.025, 0.048, 0.060)
            rect(left, bottom, world_width, world_height)
            self._draw_particles()
            for cell in self.world.living_cells():
                self._draw_cell(cell)

            summary = self.world.summary()
            fill(0.92, 0.98, 1.0)
            text(BUILD, x=24, y=self.size.h - 25, font_size=18, alignment=4)
            fill(0.64, 0.78, 0.86)
            text(
                'SCENE ACTIVE | SAVE {} | LOG {} | REPORT {} | {:.1f} fps | x{:.2f}'.format(
                    self.save_status, self.logger.status, self.report_status,
                    self.fps, self.sim_rate,
                ),
                x=self.size.w - 72,
                y=self.size.h - 25,
                font_size=9,
                alignment=6,
            )
            fill(0.84, 0.92, 0.97)
            text(
                'age {:.1f}s  cells {}  births {}  divisions {}  deaths {}  G{}  particles {}'.format(
                    summary['age'], summary['cells'], summary['births'],
                    summary['divisions'], summary['deaths'],
                    summary['max_generation'], summary['particles']
                ),
                x=24, y=49, font_size=11, alignment=4,
            )
            text(
                'seal {:.3f}/{:.3f}  ATP {:.3f}  catalyst {:.3f}  loop {:.3f}  division {:.2f}'.format(
                    summary['mean_closure'], summary['min_closure'],
                    summary['mean_atp'], summary['mean_catalyst'],
                    summary['mean_loop'], summary['division_progress']
                ),
                x=24, y=31, font_size=10, alignment=4,
            )
            text(
                'matter {:.3f}  ledger {:+.2e}  division {:+.2e} | tap cell puncture | empty inject | top pause | double reset'.format(
                    summary['total_material'], summary['matter_residual'],
                    summary['division_residual']
                ),
                x=24, y=14, font_size=8, alignment=4,
            )
            if summary['cells'] == 0:
                fill(1.0, 0.38, 0.32)
                text(
                    'REACTION–BOUNDARY LOOP EXTINCT — double tap to reseed',
                    x=self.size.w * 0.5,
                    y=self.size.h * 0.52,
                    font_size=17,
                    alignment=5,
                )
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
                self.world = SomaCellWorld(seed=101, initial_cells=1)
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
        print(run_headless_trial(seed=101, seconds=120.0, initial_cells=1))
    else:
        run(SomaCellScene(), LANDSCAPE, show_fps=False)
