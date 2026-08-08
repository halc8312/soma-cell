# coding: utf-8
"""SOMA-CELL 0.6.8-GPU — full-fidelity migration foundation.

This module does *not* replace SOMA-CELL 0.6.6 particle microphysics with a
coarse model.  It introduces a deterministic tensor backend, a lossless adapter
for the frozen 0.6.6 world state, and exact/offloadable kernels for the first
high-cost parts of that world:

* toroidal particle diffusion plus geochemical patch drift,
* membrane-local particle ligand profiles,
* circular membrane smoothing,
* a vectorised reference of the 0.1 core metabolism block,
* material-ledger reductions, and
* batched independent-world execution utilities.

The authoritative world remains the frozen CPU implementation.  A hybrid world
can offload the two kernels whose integration order is already proven safe
(diffusion and ligand profiling).  Unported chemistry, genome expression,
division, death, corpse chemistry, eDNA, HGT, neural tissue and causal audit
continue to execute in the frozen CPU code.  The module therefore fails closed:
it never calls a partially ported world a full GPU simulation.

CUDA is optional at import time.  Development/CI can validate on Torch CPU; an
RTX 4060 Ti uses the same code with a CUDA-enabled PyTorch build.
"""
from __future__ import division

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import pickle
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - Pythonista companion does not import this file.
    torch = None

HERE = os.path.dirname(os.path.abspath(__file__))
for rel in ('.', '../0_6_7', '../0_6_6', '../0_6_5', '../0_6_4', '../0_6_3',
            '../0_6_2', '../0_6_1', '../0_6', '../0_6_p2', '../0_6_p1',
            '../0_6_p0', '../baseline'):
    path = os.path.abspath(os.path.join(HERE, rel))
    if path not in sys.path:
        sys.path.insert(0, path)

import SOMA_CELL_0_6_6_pythonista as s66

s65 = s66.s65
s5 = s66.s5
s4 = s66.s4
g2 = s66.g2

BUILD = 'SOMA-CELL 0.6.8-GPU'
BUILD_LONG = BUILD + ' | deterministic tensor migration foundation'
SCHEMA_VERSION = '0.6.8-GPU-A1.0'
SAVE_VERSION = 1

MEMBRANE_SEGMENTS = int(s5.MEMBRANE_SEGMENTS)
CHANNEL_COUNT = int(s4.CHANNEL_COUNT)
POOL_COUNT = 13
PARTICLE_KIND_COUNT = 4

# Exact coefficients inherited from SensorimotorParticleField.step_diffusion.
PATCH_DIFFUSION = np.asarray([0.0026, 0.0023, 0.0018, 0.0025], dtype=np.float64)
PATCH_DRIFT_RATE = 0.020

KERNEL_DIFFUSION = 'particle_diffusion'
KERNEL_LIGAND_PROFILE = 'particle_ligand_profile'
KERNEL_CIRCULAR_SMOOTH = 'circular_smooth'
KERNEL_METABOLISM_CORE = 'metabolism_core_reference'
KERNEL_LEDGER = 'material_ledger'

PORT_STATUS = {
    KERNEL_DIFFUSION: 'hybrid-integrated',
    KERNEL_LIGAND_PROFILE: 'hybrid-integrated',
    KERNEL_CIRCULAR_SMOOTH: 'validated-primitive',
    KERNEL_METABOLISM_CORE: 'validated-standalone-not-integrated',
    KERNEL_LEDGER: 'validated-standalone',
    'surface_exchange': 'planned',
    'genome_replication': 'cpu-authoritative',
    'translation': 'cpu-authoritative',
    'division': 'cpu-authoritative',
    'death_corpse_edna_hgt': 'cpu-authoritative',
    'neural_causal_system': 'cpu-authoritative',
}


def sha256_bytes(data):
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def canonical_pickle_hash(value):
    return sha256_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def semantic_state_hash(value):
    """Order-stable hash for nested state dictionaries and NumPy arrays.

    Pickle byte streams may differ when equivalent dictionaries were rebuilt in
    a different insertion order.  This hash is intended for deterministic state
    comparison, not for cryptographic authentication of a release archive.
    """
    h = hashlib.sha256()
    def visit(item):
        if isinstance(item, dict):
            h.update(b'D')
            for key in sorted(item.keys(), key=lambda x: repr(x)):
                visit(key); visit(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(b'L' if isinstance(item, list) else b'T')
            h.update(str(len(item)).encode('ascii'))
            for child in item: visit(child)
        elif isinstance(item, set):
            h.update(b'S')
            for child in sorted(item, key=lambda x: repr(x)): visit(child)
        elif isinstance(item, np.ndarray):
            arr = np.ascontiguousarray(item)
            h.update(b'A'); h.update(str(arr.dtype).encode('ascii'))
            h.update(repr(tuple(arr.shape)).encode('ascii')); h.update(arr.tobytes())
        elif isinstance(item, np.generic):
            visit(item.item())
        elif isinstance(item, float):
            h.update(b'F'); h.update(np.asarray([item], dtype=np.float64).tobytes())
        elif isinstance(item, (int, bool)):
            h.update(b'I'); h.update(str(int(item)).encode('ascii'))
        elif item is None:
            h.update(b'N')
        elif isinstance(item, bytes):
            h.update(b'B'); h.update(item)
        else:
            h.update(b'R'); h.update(repr(item).encode('utf-8'))
    visit(value)
    return h.hexdigest()


def _torch_required():
    if torch is None:
        raise RuntimeError(
            'PyTorch is required for the GPU backend. Install a CUDA-enabled '
            'PyTorch build on the RTX workstation, or use --device cpu for CI.'
        )


def resolve_device(requested='auto'):
    _torch_required()
    requested = str(requested).lower()
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but this PyTorch build has no CUDA device')
    return device


def resolve_dtype(name='float64'):
    _torch_required()
    name = str(name).lower()
    table = {
        'float64': torch.float64,
        'fp64': torch.float64,
        'double': torch.float64,
        'float32': torch.float32,
        'fp32': torch.float32,
        'single': torch.float32,
    }
    if name not in table:
        raise ValueError('precision must be float64/fp64 or float32/fp32')
    return table[name]


@dataclass
class GPU068Config:
    device: str = 'auto'
    precision: str = 'float64'
    deterministic: bool = True
    compile_kernels: bool = False
    enable_diffusion: bool = True
    enable_ligand_profiles: bool = True
    strict_roundtrip: bool = True
    max_particles: int = 4096
    max_cells: int = 128
    max_genome_symbols: int = 2048
    batch_worlds: int = 1

    def state_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_state(cls, state):
        return cls(**dict(state or {}))


@dataclass
class TensorWorldBatch:
    """Padded tensor mirror of one or more frozen 0.6.6 worlds.

    The mirror intentionally contains more state than the currently offloaded
    kernels need.  This makes memory usage measurable now and avoids redesigning
    the schema when later kernels move to CUDA.  Variable Python dictionaries,
    corpses and detailed neural objects remain preserved in opaque_state until
    their dedicated kernels are ported.
    """

    config: GPU068Config
    device: object
    dtype: object
    batch_size: int
    particle_capacity: int
    cell_capacity: int
    genome_capacity: int
    particle_pos: object
    particle_kind: object
    particle_amount: object
    particle_mask: object
    patch_centres: object
    cell_mask: object
    cell_alive: object
    cell_id: object
    lineage: object
    generation: object
    cell_pos: object
    cell_vel: object
    radius: object
    membrane: object
    membrane_oxidation: object
    transporters: object
    pools: object
    contact_trace: object
    damage_trace: object
    surface_flux: object
    last_motor_command: object
    genome_data: object
    genome_length: object
    genome_count: object
    world_age: object
    dissipated_energy: object
    injected_material: object
    opaque_states: list = field(default_factory=list)
    source_hashes: list = field(default_factory=list)

    def tensors(self):
        for item in dataclasses.fields(self):
            value = getattr(self, item.name)
            if torch is not None and isinstance(value, torch.Tensor):
                yield item.name, value

    def clone(self):
        kwargs = {}
        for item in dataclasses.fields(self):
            value = getattr(self, item.name)
            if torch is not None and isinstance(value, torch.Tensor):
                kwargs[item.name] = value.clone()
            else:
                kwargs[item.name] = copy.deepcopy(value)
        return TensorWorldBatch(**kwargs)

    def to(self, device=None, precision=None):
        device = resolve_device(device or str(self.device))
        dtype = resolve_dtype(precision or self.config.precision)
        out = self.clone()
        out.device = device
        out.dtype = dtype
        for name, tensor in list(out.tensors()):
            if tensor.is_floating_point():
                setattr(out, name, tensor.to(device=device, dtype=dtype))
            else:
                setattr(out, name, tensor.to(device=device))
        return out

    def finite(self):
        for _, tensor in self.tensors():
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
                return False
        return True

    def estimated_bytes(self):
        total = 0
        for _, tensor in self.tensors():
            total += int(tensor.numel()) * int(tensor.element_size())
        return total

    def summary(self):
        return {
            'build': BUILD,
            'schema': SCHEMA_VERSION,
            'device': str(self.device),
            'precision': str(self.dtype),
            'batch_size': self.batch_size,
            'particle_capacity': self.particle_capacity,
            'cell_capacity': self.cell_capacity,
            'genome_capacity': self.genome_capacity,
            'active_particles': int(self.particle_mask.sum().item()),
            'active_cells': int(self.cell_mask.sum().item()),
            'estimated_bytes': self.estimated_bytes(),
            'finite': self.finite(),
        }


class FullFidelity066Adapter(object):
    """Losslessly mirrors frozen Formal066World objects into padded tensors."""

    def __init__(self, config=None):
        self.config = config if isinstance(config, GPU068Config) else GPU068Config.from_state(config)
        self.device = resolve_device(self.config.device)
        self.dtype = resolve_dtype(self.config.precision)

    @staticmethod
    def _world_state(world):
        state = world.state_dict()
        # Make source hashing independent of an attached hybrid backend.
        return state

    def pack(self, worlds):
        _torch_required()
        if isinstance(worlds, s66.Formal066World):
            worlds = [worlds]
        worlds = list(worlds)
        if not worlds:
            raise ValueError('at least one Formal066World is required')
        for world in worlds:
            if not isinstance(world, s66.Formal066World):
                raise TypeError('GPU adapter accepts frozen Formal066World instances only')

        batch = len(worlds)
        max_particles = max(len(w.field.amount) for w in worlds)
        max_cells = max(len(w.cells) for w in worlds)
        max_genome = max(
            [len(g) for w in worlds for c in w.cells for g in getattr(c, 'genomes', [])] or [1]
        )
        if max_particles > self.config.max_particles:
            raise ValueError('particle capacity exceeded: {} > {}'.format(max_particles, self.config.max_particles))
        if max_cells > self.config.max_cells:
            raise ValueError('cell capacity exceeded: {} > {}'.format(max_cells, self.config.max_cells))
        if max_genome > self.config.max_genome_symbols:
            raise ValueError('genome capacity exceeded: {} > {}'.format(max_genome, self.config.max_genome_symbols))

        P = max(1, max_particles)
        C = max(1, max_cells)
        G = max(1, max_genome)
        dev = self.device
        fd = self.dtype
        zeros = lambda *shape: torch.zeros(shape, dtype=fd, device=dev)
        izeros = lambda *shape: torch.zeros(shape, dtype=torch.int64, device=dev)
        bzeros = lambda *shape: torch.zeros(shape, dtype=torch.bool, device=dev)

        particle_pos = zeros(batch, P, 2)
        particle_kind = izeros(batch, P)
        particle_amount = zeros(batch, P)
        particle_mask = bzeros(batch, P)
        patch_centres = zeros(batch, PARTICLE_KIND_COUNT, 2)

        cell_mask = bzeros(batch, C)
        cell_alive = bzeros(batch, C)
        cell_id = izeros(batch, C)
        lineage = izeros(batch, C)
        generation = izeros(batch, C)
        cell_pos = zeros(batch, C, 2)
        cell_vel = zeros(batch, C, 2)
        radius = zeros(batch, C)
        membrane = zeros(batch, C, MEMBRANE_SEGMENTS)
        membrane_oxidation = zeros(batch, C, MEMBRANE_SEGMENTS)
        transporters = zeros(batch, C, MEMBRANE_SEGMENTS, CHANNEL_COUNT)
        pools = zeros(batch, C, POOL_COUNT)
        contact_trace = zeros(batch, C, MEMBRANE_SEGMENTS, 2)
        damage_trace = zeros(batch, C, MEMBRANE_SEGMENTS)
        surface_flux = zeros(batch, C, 2)
        last_motor_command = zeros(batch, C, 2)
        genome_data = torch.zeros((batch, C, 2, G), dtype=torch.uint8, device=dev)
        genome_length = izeros(batch, C, 2)
        genome_count = izeros(batch, C)
        world_age = zeros(batch)
        dissipated_energy = zeros(batch)
        injected_material = zeros(batch)

        opaque_states = []
        source_hashes = []
        for bi, world in enumerate(worlds):
            state = self._world_state(world)
            opaque_states.append(state)
            source_hashes.append(canonical_pickle_hash(state))
            n = len(world.field.amount)
            if n:
                particle_pos[bi, :n] = torch.as_tensor(world.field.pos, dtype=fd, device=dev)
                particle_kind[bi, :n] = torch.as_tensor(world.field.kind, dtype=torch.int64, device=dev)
                particle_amount[bi, :n] = torch.as_tensor(world.field.amount, dtype=fd, device=dev)
                particle_mask[bi, :n] = True
            patch_centres[bi] = torch.as_tensor(world.field.patch_centres, dtype=fd, device=dev)
            world_age[bi] = float(world.age)
            dissipated_energy[bi] = float(world.dissipated_energy)
            injected_material[bi] = float(getattr(world.field, 'injected_material', 0.0))
            for ci, cell in enumerate(world.cells):
                cell_mask[bi, ci] = True
                cell_alive[bi, ci] = bool(cell.alive)
                cell_id[bi, ci] = int(cell.cell_id)
                lineage[bi, ci] = int(cell.lineage)
                generation[bi, ci] = int(cell.generation)
                cell_pos[bi, ci] = torch.as_tensor(cell.pos, dtype=fd, device=dev)
                cell_vel[bi, ci] = torch.as_tensor(cell.vel, dtype=fd, device=dev)
                radius[bi, ci] = float(cell.radius)
                membrane[bi, ci] = torch.as_tensor(cell.membrane, dtype=fd, device=dev)
                membrane_oxidation[bi, ci] = torch.as_tensor(cell.membrane_oxidation, dtype=fd, device=dev)
                transporters[bi, ci] = torch.as_tensor(cell.transporters, dtype=fd, device=dev)
                pools[bi, ci] = torch.as_tensor(cell.pools, dtype=fd, device=dev)
                contact_trace[bi, ci] = torch.as_tensor(cell.contact_trace, dtype=fd, device=dev)
                damage_trace[bi, ci] = torch.as_tensor(cell.damage_trace, dtype=fd, device=dev)
                surface_flux[bi, ci] = torch.as_tensor(cell.surface_flux, dtype=fd, device=dev)
                last_motor_command[bi, ci] = torch.as_tensor(cell.last_motor_command, dtype=fd, device=dev)
                genomes = list(getattr(cell, 'genomes', []))[:2]
                genome_count[bi, ci] = len(genomes)
                for gi, genome in enumerate(genomes):
                    glen = min(len(genome), G)
                    genome_length[bi, ci, gi] = glen
                    if glen:
                        genome_data[bi, ci, gi, :glen] = torch.as_tensor(
                            np.asarray(genome[:glen], dtype=np.uint8), dtype=torch.uint8, device=dev
                        )

        return TensorWorldBatch(
            config=copy.deepcopy(self.config), device=dev, dtype=fd,
            batch_size=batch, particle_capacity=P, cell_capacity=C, genome_capacity=G,
            particle_pos=particle_pos, particle_kind=particle_kind,
            particle_amount=particle_amount, particle_mask=particle_mask,
            patch_centres=patch_centres, cell_mask=cell_mask, cell_alive=cell_alive,
            cell_id=cell_id, lineage=lineage, generation=generation,
            cell_pos=cell_pos, cell_vel=cell_vel, radius=radius,
            membrane=membrane, membrane_oxidation=membrane_oxidation,
            transporters=transporters, pools=pools, contact_trace=contact_trace,
            damage_trace=damage_trace, surface_flux=surface_flux,
            last_motor_command=last_motor_command, genome_data=genome_data,
            genome_length=genome_length, genome_count=genome_count,
            world_age=world_age, dissipated_energy=dissipated_energy,
            injected_material=injected_material, opaque_states=opaque_states,
            source_hashes=source_hashes,
        )

    def restore_cpu_worlds(self, batch):
        worlds = []
        for bi, state in enumerate(batch.opaque_states):
            world = s66.Formal066World.from_state(copy.deepcopy(state))
            # Some inherited loaders normalise genome_lesions to the number of
            # genome copies.  The adapter is a lossless checkpoint mirror, so
            # restore the serialized lesion vectors exactly rather than silently
            # changing an otherwise opaque CPU-owned field.
            for ci, cell_state in enumerate(state.get('cells', [])):
                if ci < len(world.cells) and 'genome_lesions' in cell_state:
                    world.cells[ci].genome_lesions = copy.deepcopy(cell_state['genome_lesions'])
            self.apply_owned_arrays(world, batch, bi)
            worlds.append(world)
        return worlds

    @staticmethod
    def apply_owned_arrays(world, batch, bi=0):
        """Write back only arrays whose semantics are already tensor-covered."""
        with torch.no_grad():
            n = int(batch.particle_mask[bi].sum().item())
            world.field.pos = batch.particle_pos[bi, :n].detach().cpu().numpy().astype(float, copy=True)
            world.field.kind = batch.particle_kind[bi, :n].detach().cpu().numpy().astype(np.int16, copy=True)
            world.field.amount = batch.particle_amount[bi, :n].detach().cpu().numpy().astype(float, copy=True)
            # Cell arrays are mirrored for checkpoint/inspection, but only
            # particle position changes are hybrid-integrated in A1.  A caller
            # must explicitly opt in before writing other tensors back.
        return world


# ---------------------------------------------------------------------------
# Deterministic device-neutral counter random stream
# ---------------------------------------------------------------------------

MASK32 = (1 << 32) - 1


def _xorshift32_numpy(x):
    x = np.asarray(x, dtype=np.uint64) & np.uint64(MASK32)
    x ^= (x << np.uint64(13)) & np.uint64(MASK32)
    x ^= x >> np.uint64(17)
    x ^= (x << np.uint64(5)) & np.uint64(MASK32)
    return x & np.uint64(MASK32)


def counter_uniform_numpy(seed, counters):
    counters = np.asarray(counters, dtype=np.uint64)
    x = (counters ^ np.uint64(int(seed) & MASK32) ^ np.uint64(0xA341316C)) & np.uint64(MASK32)
    x = _xorshift32_numpy(_xorshift32_numpy(x))
    return (x.astype(np.float64) + 0.5) / float(1 << 32)


def _xorshift32_torch(x):
    x = x.to(torch.int64) & MASK32
    x = x ^ ((x << 13) & MASK32)
    x = x ^ (x >> 17)
    x = x ^ ((x << 5) & MASK32)
    return x & MASK32


def counter_uniform_torch(seed, counters, dtype=None):
    _torch_required()
    dtype = dtype or torch.float64
    x = counters.to(torch.int64)
    x = (x ^ (int(seed) & MASK32) ^ 0xA341316C) & MASK32
    x = _xorshift32_torch(_xorshift32_torch(x))
    return (x.to(dtype) + 0.5) / float(1 << 32)


def counter_normal_torch(seed, counters, dtype=None):
    """Box-Muller normal stream; deterministic for a fixed device/dtype."""
    dtype = dtype or torch.float64
    counters = counters.to(torch.int64)
    u1 = counter_uniform_torch(seed, counters * 2, dtype=dtype).clamp_min(1e-12)
    u2 = counter_uniform_torch(seed ^ 0x9E3779B9, counters * 2 + 1, dtype=dtype)
    return torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)


# ---------------------------------------------------------------------------
# Exact/reference primitives
# ---------------------------------------------------------------------------


def circular_smooth_numpy(values, amount):
    values = np.asarray(values, dtype=np.float64)
    amount = float(np.clip(amount, 0.0, 0.49))
    return (1.0 - 2.0 * amount) * values + amount * np.roll(values, 1, axis=-1) + amount * np.roll(values, -1, axis=-1)


def circular_smooth_torch(values, amount):
    amount = max(0.0, min(0.49, float(amount)))
    return (1.0 - 2.0 * amount) * values + amount * torch.roll(values, 1, dims=-1) + amount * torch.roll(values, -1, dims=-1)


def diffuse_particles_numpy(pos, kind, mask, patch_centres, dt, noise):
    pos = np.asarray(pos, dtype=np.float64).copy()
    kind = np.asarray(kind, dtype=np.int64)
    mask = np.asarray(mask, dtype=bool)
    noise = np.asarray(noise, dtype=np.float64)
    diffusion = PATCH_DIFFUSION[np.clip(kind, 0, PARTICLE_KIND_COUNT - 1)]
    step = noise * np.sqrt(2.0 * diffusion[..., None] * float(dt))
    pos = np.mod(pos + np.where(mask[..., None], step, 0.0), 1.0)
    centres = np.take_along_axis(
        np.asarray(patch_centres, dtype=np.float64)[:, None, :, :],
        np.clip(kind, 0, PARTICLE_KIND_COUNT - 1)[..., None, None],
        axis=2,
    )[:, :, 0, :]
    drift = np.mod(centres - pos + 0.5, 1.0) - 0.5
    pos = np.mod(pos + np.where(mask[..., None], drift * (PATCH_DRIFT_RATE * float(dt)), 0.0), 1.0)
    return pos


def diffuse_particles_torch(pos, kind, mask, patch_centres, dt, noise):
    diff_table = torch.as_tensor(PATCH_DIFFUSION, dtype=pos.dtype, device=pos.device)
    safe_kind = kind.clamp(0, PARTICLE_KIND_COUNT - 1)
    diffusion = diff_table[safe_kind]
    step = noise.to(dtype=pos.dtype, device=pos.device) * torch.sqrt(2.0 * diffusion.unsqueeze(-1) * float(dt))
    out = torch.remainder(pos + torch.where(mask.unsqueeze(-1), step, torch.zeros_like(step)), 1.0)
    batch_index = torch.arange(out.shape[0], device=out.device).unsqueeze(1).expand_as(safe_kind)
    centres = patch_centres[batch_index, safe_kind]
    drift = torch.remainder(centres - out + 0.5, 1.0) - 0.5
    out = torch.remainder(out + torch.where(mask.unsqueeze(-1), drift * (PATCH_DRIFT_RATE * float(dt)), torch.zeros_like(drift)), 1.0)
    return out


def _ligand_profiles_numpy(particle_pos, particle_kind, particle_amount, particle_mask, cell_pos, radius, cell_mask):
    B, C, _ = cell_pos.shape
    P = particle_pos.shape[1]
    profiles = np.zeros((B, C, PARTICLE_KIND_COUNT, MEMBRANE_SEGMENTS), dtype=np.float64)
    for b in range(B):
        for c in range(C):
            if not cell_mask[b, c]:
                continue
            deltas = np.mod(particle_pos[b] - cell_pos[b, c][None, :] + 0.5, 1.0) - 0.5
            distances = np.linalg.norm(deltas, axis=1)
            valid = particle_mask[b] & (distances < float(radius[b, c]) + 0.20)
            for index in np.where(valid)[0]:
                distance = max(1e-7, float(distances[index]))
                kind = int(particle_kind[b, index])
                if kind < 0 or kind >= PARTICLE_KIND_COUNT:
                    continue
                angle = math.atan2(float(deltas[index, 1]), float(deltas[index, 0])) % (2.0 * math.pi)
                segment = int((angle / (2.0 * math.pi)) * MEMBRANE_SEGMENTS) % MEMBRANE_SEGMENTS
                amount = float(particle_amount[b, index])
                weight = amount * math.exp(-distance / 0.070) / (0.020 + distance)
                profiles[b, c, kind, segment] += weight
    profiles = circular_smooth_numpy(profiles, 0.34)
    profiles = circular_smooth_numpy(profiles, 0.22)
    return profiles


def ligand_profiles_torch(particle_pos, particle_kind, particle_amount, particle_mask, cell_pos, radius, cell_mask):
    """Dense exact vectorisation of SensorimotorProtoCell particle profiles."""
    B, C, _ = cell_pos.shape
    P = particle_pos.shape[1]
    deltas = torch.remainder(
        particle_pos[:, None, :, :] - cell_pos[:, :, None, :] + 0.5,
        1.0,
    ) - 0.5
    distances = torch.linalg.vector_norm(deltas, dim=-1)
    valid = particle_mask[:, None, :].expand(B, C, P)
    valid = valid & cell_mask[:, :, None] & (distances < radius[:, :, None] + 0.20)
    valid = valid & (particle_kind[:, None, :] >= 0) & (particle_kind[:, None, :] < PARTICLE_KIND_COUNT)
    safe_distance = distances.clamp_min(1e-7)
    angles = torch.remainder(torch.atan2(deltas[..., 1], deltas[..., 0]), 2.0 * math.pi)
    segments = torch.floor(angles / (2.0 * math.pi) * MEMBRANE_SEGMENTS).to(torch.int64) % MEMBRANE_SEGMENTS
    kinds = particle_kind[:, None, :].expand(B, C, P).clamp(0, PARTICLE_KIND_COUNT - 1)
    weights = particle_amount[:, None, :].expand(B, C, P)
    weights = weights * torch.exp(-safe_distance / 0.070) / (0.020 + safe_distance)
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    flat_index = kinds * MEMBRANE_SEGMENTS + segments
    flat = torch.zeros((B, C, PARTICLE_KIND_COUNT * MEMBRANE_SEGMENTS), dtype=particle_pos.dtype, device=particle_pos.device)
    flat.scatter_add_(2, flat_index, weights)
    profiles = flat.reshape(B, C, PARTICLE_KIND_COUNT, MEMBRANE_SEGMENTS)
    profiles = circular_smooth_torch(profiles, 0.34)
    profiles = circular_smooth_torch(profiles, 0.22)
    return profiles


# ---------------------------------------------------------------------------
# Standalone metabolism-core port.  This is validated but deliberately not
# wired into the hybrid world until every inherited 0.3-0.6 hook is ported.
# ---------------------------------------------------------------------------


def _closure_numpy(membrane, radius):
    required = float(g2.base.MEMBRANE_DENSITY) * (2.0 * math.pi * radius / MEMBRANE_SEGMENTS)
    required = np.maximum(required, 1e-9)
    return np.clip(membrane / (required[..., None] * float(g2.base.GAP_CLOSURE_THRESHOLD)), 0.0, 1.0)


def _tension_numpy(pools, membrane):
    membrane_mass = np.sum(membrane, axis=-1)
    capacity = np.clip(membrane_mass / (2.0 * math.pi * float(g2.base.MEMBRANE_DENSITY)), float(g2.MIN_RADIUS), float(g2.MAX_RADIUS))
    osmolyte = np.sum(pools, axis=-1)
    ratio = np.maximum(0.08, osmolyte / float(g2.base.TARGET_OSMOLYTE))
    osmotic = np.clip(float(s5.BASE_RADIUS) * np.sqrt(ratio), float(g2.MIN_RADIUS), float(g2.MAX_RADIUS) * 1.25)
    return np.maximum(0.0, osmotic / np.maximum(capacity, 1e-8) - 1.0)


def metabolism_core_numpy(pools, membrane, transporters, contact_trace, damage_trace, radius, alive, dt,
                          catalyst_synthesis=True, membrane_synthesis=True, targeted_repair=True):
    pools=np.asarray(pools,dtype=np.float64).copy(); membrane=np.asarray(membrane,dtype=np.float64).copy()
    transporters=np.asarray(transporters,dtype=np.float64).copy(); contact_trace=np.asarray(contact_trace,dtype=np.float64).copy()
    damage_trace=np.asarray(damage_trace,dtype=np.float64).copy(); radius=np.asarray(radius,dtype=np.float64)
    alive_f=np.asarray(alive,dtype=np.float64); dt=float(dt)
    tension=_tension_numpy(pools,membrane); closure=_closure_numpy(membrane,radius)
    volume=np.maximum(0.20,(radius/float(s5.BASE_RADIUS))**2)
    fuel=pools[...,s5.POOL_FUEL]/volume; catalyst=pools[...,s5.POOL_CATALYST]/volume; waste=pools[...,s5.POOL_WASTE]/volume
    inhibition=1.0/(1.0+2.8*waste)
    cat_rate=0.12*catalyst*fuel/(0.16+fuel)*inhibition
    cat_amount=np.minimum(pools[...,s5.POOL_FUEL],cat_rate*dt)*alive_f
    pools[...,s5.POOL_FUEL]-=cat_amount; pools[...,s5.POOL_WASTE]+=cat_amount; pools[...,s5.POOL_ATP]+=2.10*cat_amount
    maintenance=dt*(0.0045+0.010*pools[...,s5.POOL_CATALYST]+0.010*np.sum(transporters,axis=(-1,-2))+0.0025*np.sum(membrane,axis=-1)+0.008*tension)*alive_f
    paid=np.minimum(pools[...,s5.POOL_ATP],maintenance); pools[...,s5.POOL_ATP]-=paid; dissipated=paid.copy(); shortfall=np.maximum(0.0,maintenance-paid)
    def limit(desired,reqs):
        amount=np.maximum(0.0,desired)
        for index,coeff in reqs:
            available=pools[...,index]
            if index==s5.POOL_ATP: available=np.maximum(0.0,available-0.045)
            amount=np.minimum(amount,available/float(coeff))
        return np.maximum(0.0,amount)*alive_f
    need=np.clip((1.03-np.mean(closure,axis=-1))*2.5+tension*2.2+np.maximum(0.0,0.17-pools[...,s5.POOL_MEM_PRECURSOR])*1.5,0.0,2.5)
    amount=limit(dt*0.095*pools[...,s5.POOL_CATALYST]*need,((s5.POOL_FUEL,0.62),(s5.POOL_MINERAL,0.38),(s5.POOL_ATP,0.42)))
    pools[...,s5.POOL_FUEL]-=0.62*amount; pools[...,s5.POOL_MINERAL]-=0.38*amount; pools[...,s5.POOL_ATP]-=0.42*amount; pools[...,s5.POOL_MEM_PRECURSOR]+=amount; dissipated+=0.42*amount
    if catalyst_synthesis:
        size_ratio=np.maximum(1.0,(radius/float(s5.BASE_RADIUS))**2); target=np.minimum(0.68,0.34+0.30*(size_ratio-1.0)); need=np.clip((target-pools[...,s5.POOL_CATALYST])/0.20,0.0,1.5)
        amount=limit(dt*0.050*pools[...,s5.POOL_CATALYST]*need,((s5.POOL_FUEL,0.70),(s5.POOL_MINERAL,0.30),(s5.POOL_ATP,0.80)))
        pools[...,s5.POOL_FUEL]-=0.70*amount; pools[...,s5.POOL_MINERAL]-=0.30*amount; pools[...,s5.POOL_ATP]-=0.80*amount; pools[...,s5.POOL_CATALYST]+=amount; dissipated+=0.80*amount
    tneed=np.clip(0.7*np.maximum(0.0,0.22-pools[...,s5.POOL_FUEL])+0.8*np.maximum(0.0,0.19-pools[...,s5.POOL_MINERAL])+0.9*np.maximum(0.0,pools[...,s5.POOL_WASTE]-0.06)+0.3,0.0,1.5)
    amount=limit(dt*0.035*pools[...,s5.POOL_CATALYST]*tneed,((s5.POOL_FUEL,0.65),(s5.POOL_MINERAL,0.35),(s5.POOL_ATP,0.70)))
    pools[...,s5.POOL_FUEL]-=0.65*amount; pools[...,s5.POOL_MINERAL]-=0.35*amount; pools[...,s5.POOL_ATP]-=0.70*amount; pools[...,s5.POOL_TRANSPORTER_PRECURSOR]+=amount; dissipated+=0.70*amount
    if membrane_synthesis:
        required=float(g2.base.MEMBRANE_DENSITY)*(2.0*math.pi*radius/MEMBRANE_SEGMENTS); deficit=np.maximum(required[...,None]*1.06-membrane,0.0)
        weights=deficit*8.0+damage_trace*2.5+0.02 if targeted_repair else np.ones_like(membrane)
        weights+= (np.sum(deficit,axis=-1)<1e-5)[...,None]*0.18; weights=np.maximum(weights,1e-8); weights/=np.sum(weights,axis=-1,keepdims=True)
        rate=0.080*pools[...,s5.POOL_CATALYST]*(0.45+1.4*(1.0-np.mean(closure,axis=-1))+0.6*tension)
        assembly=np.minimum(pools[...,s5.POOL_MEM_PRECURSOR],rate*dt); assembly=np.minimum(assembly,np.maximum(0.0,pools[...,s5.POOL_ATP]-0.045)/0.34)*alive_f
        membrane+=weights*assembly[...,None]; pools[...,s5.POOL_MEM_PRECURSOR]-=assembly; pools[...,s5.POOL_ATP]-=0.34*assembly; dissipated+=0.34*assembly
    precursor=pools[...,s5.POOL_TRANSPORTER_PRECURSOR]
    cneed=np.stack([np.maximum(0.08,0.34-pools[...,s5.POOL_FUEL]),np.maximum(0.08,0.29-pools[...,s5.POOL_MINERAL]),np.maximum(0.05,pools[...,s5.POOL_WASTE]*1.4)],axis=-1); cneed/=np.sum(cneed,axis=-1,keepdims=True)
    insert=np.minimum(precursor,0.038*pools[...,s5.POOL_CATALYST]*dt); insert=np.minimum(insert,np.maximum(0.0,pools[...,s5.POOL_ATP]-0.045)/0.40)*alive_f
    locals_=[contact_trace[...,0]+0.03,contact_trace[...,1]+0.03,closure+damage_trace+0.03]
    for channel in range(min(3,CHANNEL_COUNT)):
        local=np.maximum(locals_[channel],1e-8); local/=np.sum(local,axis=-1,keepdims=True); transporters[...,channel]+=local*(insert*cneed[...,channel])[...,None]
    pools[...,s5.POOL_TRANSPORTER_PRECURSOR]-=insert; pools[...,s5.POOL_ATP]-=0.40*insert; dissipated+=0.40*insert
    membrane=circular_smooth_numpy(membrane,min(0.49,0.075*dt)); transporters=np.swapaxes(circular_smooth_numpy(np.swapaxes(transporters,-1,-2),min(0.49,0.10*dt)),-1,-2)
    waste_stress=pools[...,s5.POOL_WASTE]/np.maximum(0.20,volume); mrate=0.00032+0.0010*waste_stress+0.0012*tension+0.015*shortfall/max(dt,1e-9)
    mloss=np.minimum(membrane,membrane*mrate[...,None]*dt+damage_trace*0.0008*dt)*alive_f[...,None]; membrane-=mloss; pools[...,s5.POOL_WASTE]+=np.sum(mloss,axis=-1)
    cdec=np.minimum(pools[...,s5.POOL_CATALYST],pools[...,s5.POOL_CATALYST]*(0.00065+0.0017*waste_stress+0.002*shortfall)*dt)*alive_f; pools[...,s5.POOL_CATALYST]-=cdec; pools[...,s5.POOL_WASTE]+=cdec
    tdec=np.minimum(transporters,transporters*(0.00075+0.0010*waste_stress+0.0015*shortfall)[...,None,None]*dt)*alive_f[...,None,None]; transporters-=tdec; pools[...,s5.POOL_WASTE]+=np.sum(tdec,axis=(-1,-2))
    pools[...,s5.POOL_ATP]*=math.exp(-0.018*dt); pools[...,s5.POOL_ATP]=np.clip(pools[...,s5.POOL_ATP],0.0,1.6); contact_trace*=math.exp(-0.80*dt); damage_trace=np.clip(damage_trace*math.exp(-0.55*dt),0.0,1.5); pools=np.maximum(pools,0.0)
    return {'pools':pools,'membrane':membrane,'transporters':transporters,'contact_trace':contact_trace,'damage_trace':damage_trace,'dissipated_delta':dissipated,'catalysis':cat_amount}


def _closure_torch(membrane, radius):
    required = float(g2.base.MEMBRANE_DENSITY) * (2.0 * math.pi * radius / MEMBRANE_SEGMENTS)
    required = required.clamp_min(1e-9)
    return (membrane / (required.unsqueeze(-1) * float(g2.base.GAP_CLOSURE_THRESHOLD))).clamp(0.0, 1.0)


def _tension_torch(pools, membrane):
    membrane_mass = membrane.sum(dim=-1)
    capacity = (membrane_mass / (2.0 * math.pi * float(g2.base.MEMBRANE_DENSITY))).clamp(float(g2.MIN_RADIUS), float(g2.MAX_RADIUS))
    osmolyte = pools.sum(dim=-1)
    ratio = (osmolyte / float(g2.base.TARGET_OSMOLYTE)).clamp_min(0.08)
    osmotic = (float(s5.BASE_RADIUS) * torch.sqrt(ratio)).clamp(float(g2.MIN_RADIUS), float(g2.MAX_RADIUS) * 1.25)
    return (osmotic / capacity.clamp_min(1e-8) - 1.0).clamp_min(0.0)


def metabolism_core_torch(pools, membrane, transporters, contact_trace, damage_trace, radius, alive, dt,
                          catalyst_synthesis=True, membrane_synthesis=True, targeted_repair=True):
    """Vectorised transcription of the frozen 0.1 chemistry core.

    Returns new arrays and dissipated ATP.  Export, leakage, motion, division,
    viability, later damage/repair layers and neural hooks are intentionally not
    included, hence this function is not yet used by Hybrid066World.
    """
    pools = pools.clone(); membrane = membrane.clone(); transporters = transporters.clone()
    contact_trace = contact_trace.clone(); damage_trace = damage_trace.clone()
    alive_f = alive.to(pools.dtype)
    dt = float(dt)
    tension = _tension_torch(pools, membrane)
    closure = _closure_torch(membrane, radius)
    volume = torch.maximum(torch.full_like(radius, 0.20), (radius / float(s5.BASE_RADIUS)) ** 2)
    fuel = pools[..., s5.POOL_FUEL] / volume
    catalyst = pools[..., s5.POOL_CATALYST] / volume
    waste = pools[..., s5.POOL_WASTE] / volume
    inhibition = 1.0 / (1.0 + 2.8 * waste)
    cat_rate = 0.12 * catalyst * fuel / (0.16 + fuel) * inhibition
    cat_amount = torch.minimum(pools[..., s5.POOL_FUEL], cat_rate * dt) * alive_f
    pools[..., s5.POOL_FUEL] -= cat_amount
    pools[..., s5.POOL_WASTE] += cat_amount
    pools[..., s5.POOL_ATP] += 2.10 * cat_amount

    maintenance = dt * (
        0.0045 + 0.010 * pools[..., s5.POOL_CATALYST]
        + 0.010 * transporters.sum(dim=(-1, -2))
        + 0.0025 * membrane.sum(dim=-1) + 0.008 * tension
    ) * alive_f
    paid = torch.minimum(pools[..., s5.POOL_ATP], maintenance)
    pools[..., s5.POOL_ATP] -= paid
    dissipated = paid.clone()
    shortfall = (maintenance - paid).clamp_min(0.0)

    def reaction_limit(desired, reqs):
        amount = desired.clamp_min(0.0)
        for index, coeff in reqs:
            available = pools[..., index]
            if index == s5.POOL_ATP:
                available = (available - 0.045).clamp_min(0.0)
            amount = torch.minimum(amount, available / float(coeff))
        return amount.clamp_min(0.0) * alive_f

    need_membrane = ((1.03 - closure.mean(dim=-1)) * 2.5 + tension * 2.2
                     + (0.17 - pools[..., s5.POOL_MEM_PRECURSOR]).clamp_min(0.0) * 1.5).clamp(0.0, 2.5)
    desired = dt * 0.095 * pools[..., s5.POOL_CATALYST] * need_membrane
    amount = reaction_limit(desired, ((s5.POOL_FUEL, 0.62), (s5.POOL_MINERAL, 0.38), (s5.POOL_ATP, 0.42)))
    pools[..., s5.POOL_FUEL] -= 0.62 * amount
    pools[..., s5.POOL_MINERAL] -= 0.38 * amount
    pools[..., s5.POOL_ATP] -= 0.42 * amount
    pools[..., s5.POOL_MEM_PRECURSOR] += amount
    dissipated += 0.42 * amount

    if catalyst_synthesis:
        size_ratio = torch.maximum(torch.ones_like(radius), (radius / float(s5.BASE_RADIUS)) ** 2)
        target_cat = torch.minimum(torch.full_like(radius, 0.68), 0.34 + 0.30 * (size_ratio - 1.0))
        cat_need = ((target_cat - pools[..., s5.POOL_CATALYST]) / 0.20).clamp(0.0, 1.5)
        desired = dt * 0.050 * pools[..., s5.POOL_CATALYST] * cat_need
        amount = reaction_limit(desired, ((s5.POOL_FUEL, 0.70), (s5.POOL_MINERAL, 0.30), (s5.POOL_ATP, 0.80)))
        pools[..., s5.POOL_FUEL] -= 0.70 * amount
        pools[..., s5.POOL_MINERAL] -= 0.30 * amount
        pools[..., s5.POOL_ATP] -= 0.80 * amount
        pools[..., s5.POOL_CATALYST] += amount
        dissipated += 0.80 * amount

    transport_need = (
        0.7 * (0.22 - pools[..., s5.POOL_FUEL]).clamp_min(0.0)
        + 0.8 * (0.19 - pools[..., s5.POOL_MINERAL]).clamp_min(0.0)
        + 0.9 * (pools[..., s5.POOL_WASTE] - 0.06).clamp_min(0.0) + 0.3
    ).clamp(0.0, 1.5)
    desired = dt * 0.035 * pools[..., s5.POOL_CATALYST] * transport_need
    amount = reaction_limit(desired, ((s5.POOL_FUEL, 0.65), (s5.POOL_MINERAL, 0.35), (s5.POOL_ATP, 0.70)))
    pools[..., s5.POOL_FUEL] -= 0.65 * amount
    pools[..., s5.POOL_MINERAL] -= 0.35 * amount
    pools[..., s5.POOL_ATP] -= 0.70 * amount
    pools[..., s5.POOL_TRANSPORTER_PRECURSOR] += amount
    dissipated += 0.70 * amount

    if membrane_synthesis:
        required = float(g2.base.MEMBRANE_DENSITY) * (2.0 * math.pi * radius / MEMBRANE_SEGMENTS)
        deficit = (required.unsqueeze(-1) * 1.06 - membrane).clamp_min(0.0)
        if targeted_repair:
            weights = deficit * 8.0 + damage_trace * 2.5 + 0.02
        else:
            weights = torch.ones_like(membrane)
        closed = deficit.sum(dim=-1) < 1e-5
        weights = weights + closed.to(weights.dtype).unsqueeze(-1) * 0.18
        weights = weights.clamp_min(1e-8)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        assembly_rate = 0.080 * pools[..., s5.POOL_CATALYST] * (0.45 + 1.4 * (1.0 - closure.mean(dim=-1)) + 0.6 * tension)
        assembly = torch.minimum(pools[..., s5.POOL_MEM_PRECURSOR], assembly_rate * dt)
        assembly = torch.minimum(assembly, (pools[..., s5.POOL_ATP] - 0.045).clamp_min(0.0) / 0.34) * alive_f
        membrane += weights * assembly.unsqueeze(-1)
        pools[..., s5.POOL_MEM_PRECURSOR] -= assembly
        pools[..., s5.POOL_ATP] -= 0.34 * assembly
        dissipated += 0.34 * assembly

    precursor = pools[..., s5.POOL_TRANSPORTER_PRECURSOR]
    channel_need = torch.stack([
        torch.maximum(torch.full_like(radius, 0.08), 0.34 - pools[..., s5.POOL_FUEL]),
        torch.maximum(torch.full_like(radius, 0.08), 0.29 - pools[..., s5.POOL_MINERAL]),
        torch.maximum(torch.full_like(radius, 0.05), pools[..., s5.POOL_WASTE] * 1.4),
    ], dim=-1)
    channel_need = channel_need / channel_need.sum(dim=-1, keepdim=True)
    insert = torch.minimum(precursor, 0.038 * pools[..., s5.POOL_CATALYST] * dt)
    insert = torch.minimum(insert, (pools[..., s5.POOL_ATP] - 0.045).clamp_min(0.0) / 0.40) * alive_f
    locals_ = [contact_trace[..., 0] + 0.03, contact_trace[..., 1] + 0.03, closure + damage_trace + 0.03]
    for channel in range(min(3, CHANNEL_COUNT)):
        local = locals_[channel].clamp_min(1e-8)
        local = local / local.sum(dim=-1, keepdim=True)
        transporters[..., channel] += local * (insert * channel_need[..., channel]).unsqueeze(-1)
    pools[..., s5.POOL_TRANSPORTER_PRECURSOR] -= insert
    pools[..., s5.POOL_ATP] -= 0.40 * insert
    dissipated += 0.40 * insert

    membrane = circular_smooth_torch(membrane, min(0.49, 0.075 * dt))
    transporters = circular_smooth_torch(transporters.transpose(-1, -2), min(0.49, 0.10 * dt)).transpose(-1, -2)

    waste_stress = pools[..., s5.POOL_WASTE] / torch.maximum(torch.full_like(volume, 0.20), volume)
    membrane_decay_rate = 0.00032 + 0.0010 * waste_stress + 0.0012 * tension
    membrane_decay_rate = membrane_decay_rate + 0.015 * shortfall / max(dt, 1e-9)
    membrane_loss = torch.minimum(membrane, membrane * membrane_decay_rate.unsqueeze(-1) * dt + damage_trace * 0.0008 * dt) * alive_f.unsqueeze(-1)
    membrane -= membrane_loss
    pools[..., s5.POOL_WASTE] += membrane_loss.sum(dim=-1)

    cat_decay = torch.minimum(
        pools[..., s5.POOL_CATALYST],
        pools[..., s5.POOL_CATALYST] * (0.00065 + 0.0017 * waste_stress + 0.002 * shortfall) * dt,
    ) * alive_f
    pools[..., s5.POOL_CATALYST] -= cat_decay
    pools[..., s5.POOL_WASTE] += cat_decay

    transporter_decay = torch.minimum(
        transporters,
        transporters * (0.00075 + 0.0010 * waste_stress + 0.0015 * shortfall).unsqueeze(-1).unsqueeze(-1) * dt,
    ) * alive_f.unsqueeze(-1).unsqueeze(-1)
    transporters -= transporter_decay
    pools[..., s5.POOL_WASTE] += transporter_decay.sum(dim=(-1, -2))

    pools[..., s5.POOL_ATP] *= math.exp(-0.018 * dt)
    pools[..., s5.POOL_ATP] = pools[..., s5.POOL_ATP].clamp(0.0, 1.6)
    contact_trace *= math.exp(-0.80 * dt)
    damage_trace *= math.exp(-0.55 * dt)
    damage_trace = damage_trace.clamp(0.0, 1.5)
    pools = pools.clamp_min(0.0)
    return {
        'pools': pools,
        'membrane': membrane,
        'transporters': transporters,
        'contact_trace': contact_trace,
        'damage_trace': damage_trace,
        'dissipated_delta': dissipated,
        'catalysis': cat_amount,
    }


# ---------------------------------------------------------------------------
# Hybrid integration: full frozen CPU world, safe kernels offloaded only.
# ---------------------------------------------------------------------------


class TorchKernelBackend(object):
    def __init__(self, config=None):
        self.config = config if isinstance(config, GPU068Config) else GPU068Config.from_state(config)
        self.device = resolve_device(self.config.device)
        self.dtype = resolve_dtype(self.config.precision)
        if self.config.deterministic:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        self.diffusion_calls = 0
        self.profile_calls = 0
        self.transfer_bytes = 0
        self.kernel_seconds = 0.0

    def _sync_if_cuda(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def diffuse_field_inplace(self, field, dt):
        if len(field.amount) == 0 or getattr(field, 'environment_mode', 'patchy') == 'uniform':
            return field._soma068_original_step_diffusion(dt)
        # RNG remains authoritative on CPU for exact stream continuity.  Later
        # A2 can move it to the counter stream after event-level equivalence is
        # locked down.
        noise = field.rng.normal(0.0, 1.0, field.pos.shape)
        pos = torch.as_tensor(field.pos[None, ...], dtype=self.dtype, device=self.device)
        kind = torch.as_tensor(field.kind[None, ...], dtype=torch.int64, device=self.device)
        mask = torch.ones(kind.shape, dtype=torch.bool, device=self.device)
        centres = torch.as_tensor(field.patch_centres[None, ...], dtype=self.dtype, device=self.device)
        noise_t = torch.as_tensor(noise[None, ...], dtype=self.dtype, device=self.device)
        self._sync_if_cuda(); start = time.perf_counter()
        out = diffuse_particles_torch(pos, kind, mask, centres, dt, noise_t)
        self._sync_if_cuda(); self.kernel_seconds += time.perf_counter() - start
        result = out[0].detach().cpu().numpy().astype(float, copy=True)
        self.transfer_bytes += int(pos.numel() + out.numel() + noise_t.numel()) * int(pos.element_size())
        field.pos = result
        self.diffusion_calls += 1

    def ligand_profiles(self, field, cell):
        if len(field.amount) == 0:
            return np.zeros((PARTICLE_KIND_COUNT, MEMBRANE_SEGMENTS), dtype=float)
        particle_pos = torch.as_tensor(field.pos[None, ...], dtype=self.dtype, device=self.device)
        particle_kind = torch.as_tensor(field.kind[None, ...], dtype=torch.int64, device=self.device)
        particle_amount = torch.as_tensor(field.amount[None, ...], dtype=self.dtype, device=self.device)
        particle_mask = torch.ones(particle_kind.shape, dtype=torch.bool, device=self.device)
        cell_pos = torch.as_tensor(cell.pos.reshape(1, 1, 2), dtype=self.dtype, device=self.device)
        radius = torch.as_tensor([[cell.radius]], dtype=self.dtype, device=self.device)
        cell_mask = torch.ones((1, 1), dtype=torch.bool, device=self.device)
        self._sync_if_cuda(); start = time.perf_counter()
        result = ligand_profiles_torch(
            particle_pos, particle_kind, particle_amount, particle_mask,
            cell_pos, radius, cell_mask,
        )
        self._sync_if_cuda(); self.kernel_seconds += time.perf_counter() - start
        out = result[0, 0].detach().cpu().numpy().astype(float, copy=True)
        self.transfer_bytes += int(particle_pos.numel() + particle_amount.numel() + out.size) * int(particle_pos.element_size())
        self.profile_calls += 1
        return out

    def stats(self):
        return {
            'device': str(self.device), 'dtype': str(self.dtype),
            'diffusion_calls': self.diffusion_calls, 'profile_calls': self.profile_calls,
            'transfer_bytes': self.transfer_bytes, 'kernel_seconds': self.kernel_seconds,
        }


def _attach_backend_to_world(world, backend):
    field = world.field
    if not hasattr(field, '_soma068_original_step_diffusion'):
        field._soma068_original_step_diffusion = field.step_diffusion

        def _gpu_step_diffusion(this, dt):
            return backend.diffuse_field_inplace(this, dt)

        field.step_diffusion = types.MethodType(_gpu_step_diffusion, field)
    field._soma068_backend = backend
    for cell in world.cells:
        if not hasattr(cell, '_soma068_original_particle_profiles'):
            cell._soma068_original_particle_profiles = cell._particle_ligand_profiles

            def _gpu_profiles(this, particle_field):
                return backend.ligand_profiles(particle_field, this)

            cell._particle_ligand_profiles = types.MethodType(_gpu_profiles, cell)
        cell._soma068_backend = backend
    return world


class Hybrid066World(object):
    """Frozen 0.6.6 world with only proven kernels offloaded to Torch."""

    def __init__(self, cpu_world, gpu_config=None):
        if not isinstance(cpu_world, s66.Formal066World):
            raise TypeError('Hybrid066World requires a Formal066World')
        self.backend = TorchKernelBackend(gpu_config)
        self.world = cpu_world
        _attach_backend_to_world(self.world, self.backend)

    @classmethod
    def new(cls, seed=101, initial_cells=3, world_config=None, gpu_config=None):
        config = world_config if isinstance(world_config, s66.Formal066Config) else s66.Formal066Config.from_state(world_config or {})
        return cls(s66.Formal066World(seed=seed, initial_cells=initial_cells, config=config), gpu_config=gpu_config)

    def step(self, dt):
        _attach_backend_to_world(self.world, self.backend)
        self.world.step(dt)
        _attach_backend_to_world(self.world, self.backend)

    def summary(self):
        out = self.world.summary()
        out.update({
            'gpu_build': BUILD, 'gpu_schema': SCHEMA_VERSION,
            'gpu_port_status': dict(PORT_STATUS), 'gpu_backend': self.backend.stats(),
            'gpu_full_world_step': False,
        })
        return out

    def state_dict(self):
        return {
            'save_version': SAVE_VERSION, 'build': BUILD,
            'gpu_config': self.backend.config.state_dict(),
            'cpu_world': self.world.state_dict(),
        }

    @classmethod
    def from_state(cls, state):
        return cls(
            s66.Formal066World.from_state(state['cpu_world']),
            gpu_config=GPU068Config.from_state(state.get('gpu_config', {})),
        )

    def clone(self):
        return Hybrid066World.from_state(self.state_dict())

    def save(self, path):
        temporary = str(path) + '.tmp'
        with open(temporary, 'wb') as handle:
            pickle.dump(self.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, str(path))

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as handle:
            return cls.from_state(pickle.load(handle))


# ---------------------------------------------------------------------------
# Batched synthetic benchmark / capacity estimator
# ---------------------------------------------------------------------------


def synthetic_batch(worlds=32, particles=1024, cells=16, seed=101, device='auto', precision='float32'):
    _torch_required()
    dev = resolve_device(device); dtype = resolve_dtype(precision)
    counters = torch.arange(worlds * particles * 2, device=dev, dtype=torch.int64).reshape(worlds, particles, 2)
    pos = counter_uniform_torch(seed, counters, dtype=dtype)
    kind = (torch.arange(particles, device=dev, dtype=torch.int64)[None, :].expand(worlds, particles) % 4).contiguous()
    amount = torch.full((worlds, particles), 0.025, dtype=dtype, device=dev)
    pmask = torch.ones((worlds, particles), dtype=torch.bool, device=dev)
    centres = torch.as_tensor(s4.SensorimotorParticleField.DEFAULT_CENTRES, dtype=dtype, device=dev)[None, ...].expand(worlds, -1, -1).contiguous()
    ccounters = torch.arange(worlds * cells * 2, device=dev, dtype=torch.int64).reshape(worlds, cells, 2) + 1000003
    cpos = counter_uniform_torch(seed ^ 0x13579BDF, ccounters, dtype=dtype)
    radius = torch.full((worlds, cells), 0.04, dtype=dtype, device=dev)
    cmask = torch.ones((worlds, cells), dtype=torch.bool, device=dev)
    return dict(pos=pos, kind=kind, amount=amount, pmask=pmask, centres=centres, cpos=cpos, radius=radius, cmask=cmask)


def benchmark_kernels(worlds=32, particles=1024, cells=16, steps=50, device='auto', precision='float32'):
    data = synthetic_batch(worlds, particles, cells, device=device, precision=precision)
    dev = data['pos'].device; dtype = data['pos'].dtype
    noise_counter = torch.arange(worlds * particles * 2, device=dev, dtype=torch.int64).reshape(worlds, particles, 2)
    if dev.type == 'cuda': torch.cuda.synchronize(dev)
    start = time.perf_counter()
    pos = data['pos']
    checksum = torch.zeros((), dtype=dtype, device=dev)
    for step in range(int(steps)):
        noise = counter_normal_torch(101 + step, noise_counter + step * 10000019, dtype=dtype)
        pos = diffuse_particles_torch(pos, data['kind'], data['pmask'], data['centres'], 0.1, noise)
        profiles = ligand_profiles_torch(pos, data['kind'], data['amount'], data['pmask'], data['cpos'], data['radius'], data['cmask'])
        checksum = checksum + profiles.mean()
    if dev.type == 'cuda': torch.cuda.synchronize(dev)
    elapsed = time.perf_counter() - start
    bytes_used = sum(int(v.numel()) * int(v.element_size()) for v in data.values() if isinstance(v, torch.Tensor))
    return {
        'build': BUILD, 'schema': SCHEMA_VERSION, 'device': str(dev), 'dtype': str(dtype),
        'worlds': int(worlds), 'particles_per_world': int(particles), 'cells_per_world': int(cells),
        'steps': int(steps), 'elapsed_seconds': elapsed,
        'world_steps_per_second': float(worlds * steps / max(elapsed, 1e-12)),
        'kernel_checksum': float(checksum.detach().cpu().item()),
        'working_tensor_bytes': bytes_used,
        'cuda_available': bool(torch.cuda.is_available()),
        'full_world_step_ported': False,
    }


def environment_report():
    result = {
        'build': BUILD, 'schema': SCHEMA_VERSION,
        'python': sys.version, 'numpy': np.__version__, 'torch_installed': torch is not None,
        'port_status': dict(PORT_STATUS),
    }
    if torch is not None:
        result.update({
            'torch': torch.__version__, 'cuda_available': bool(torch.cuda.is_available()),
            'torch_cuda_version': torch.version.cuda, 'device_count': torch.cuda.device_count(),
        })
        if torch.cuda.is_available():
            result['devices'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=BUILD_LONG)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--precision', default='float32')
    parser.add_argument('--worlds', type=int, default=16)
    parser.add_argument('--particles', type=int, default=512)
    parser.add_argument('--cells', type=int, default=8)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--json', default='')
    args = parser.parse_args(argv)
    report = {'environment': environment_report(), 'benchmark': benchmark_kernels(
        worlds=args.worlds, particles=args.particles, cells=args.cells,
        steps=args.steps, device=args.device, precision=args.precision,
    )}
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json:
        Path(args.json).write_text(text + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    main()
