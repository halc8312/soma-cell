# coding: utf-8
"""A4.9a immutable world-wide resident shadow and coherence boundary.

This module deliberately owns no durable biology.  It uploads one exact,
ordered CPU world snapshot into fresh Torch storage, derives the gene cache
from that same resident ragged genome, and then permits only private,
read-only, epoch-bound leases.  CPU drift never updates old storage.  A
replacement is built separately and becomes active only through a guarded
owner swap.

There is no device-dirty state, D2H publish, scheduler hook, event claim, RNG
operation, or live biological commit in this slice.  The private validation
snapshot performs diagnostic readback only.
"""
from __future__ import division

import copy
import hashlib
import itertools
import math
import os
import pickle
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass, fields

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import SOMA_CELL_0_6_8_gpu_a4 as a4

try:
    import torch
except Exception:  # pragma: no cover - import remains diagnostic without Torch
    torch = None


BUILD = 'SOMA-CELL 0.6.8-GPU A4.9a'
BUILD_ID = BUILD
BUILD_LONG = BUILD + ' | immutable world-wide H2D shadow/cache coherence foundation'
SCHEMA_VERSION = '0.6.8-GPU-A4.9a-resident-arena-shadow'
FULL_GPU_WORLD_STEP = False

COHERENT = 'COHERENT'
CPU_NEWER = 'CPU_NEWER'
INVALID = 'INVALID'
_LIFECYCLES = frozenset((COHERENT, CPU_NEWER, INVALID))

PORT_STATUS = dict(a4.PORT_STATUS)
PORT_STATUS.update({
    'resident_world_arena': 'a4.9a-immutable-h2d-shadow-only',
    'resident_gene_cache': 'a4.9a-derived-from-same-resident-ragged',
    'resident_biology_authority': 'cpu-authoritative-shadow-only',
    'resident_device_writes': 'not-implemented-scope-error',
    'resident_d2h_publish': 'not-implemented-scope-error',
    'full_gpu_world_step': False,
})

__all__ = (
    'BUILD', 'BUILD_ID', 'BUILD_LONG', 'SCHEMA_VERSION',
    'FULL_GPU_WORLD_STEP', 'COHERENT', 'CPU_NEWER', 'INVALID',
    'PORT_STATUS', 'A4ResidentArenaError', 'A4ResidentArenaScopeError',
    'A4ResidentArenaEpochs', 'A4ResidentArenaOwner',
)


class A4ResidentArenaError(a4.A4Error):
    """The immutable A4.9a arena contract failed closed."""


class A4ResidentArenaScopeError(A4ResidentArenaError):
    """A caller requested resident authority outside the A4.9a scope."""


def _strict_epoch(value, label):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))):
        raise A4ResidentArenaError('%s must be an integer' % label)
    result = int(value)
    if result <= 0:
        raise A4ResidentArenaError('%s must be positive' % label)
    return result


@dataclass(frozen=True)
class A4ResidentArenaEpochs:
    """Logical CPU-domain epochs plus the resident cache source epoch."""

    membership_epoch: int
    ragged_epoch: int
    state_epoch: int
    cache_epoch: int
    cache_source_ragged_epoch: int

    def __post_init__(self):
        for name in (
                'membership_epoch', 'ragged_epoch', 'state_epoch',
                'cache_epoch', 'cache_source_ragged_epoch'):
            object.__setattr__(self, name, _strict_epoch(getattr(self, name), name))
        if self.cache_source_ragged_epoch > self.ragged_epoch:
            raise A4ResidentArenaError(
                'cache source ragged epoch cannot exceed ragged epoch'
            )


@dataclass(frozen=True)
class _IdentityAnchor:
    """Strong identity references prevent Python ``id`` reuse from passing."""

    labels: tuple
    objects: tuple


@dataclass(frozen=True)
class _MembershipAttestation:
    values: tuple
    anchor: _IdentityAnchor


@dataclass(frozen=True)
class _CpuAttestation:
    membership: _MembershipAttestation
    ragged_digest: str
    state_digest: str
    derived_cache_digest: str
    live_cache_digest: str
    cache_exact: bool
    ragged_anchor: _IdentityAnchor
    state_anchor: _IdentityAnchor
    cache_anchor: _IdentityAnchor
    model_config_digest: str
    model_config_flags: tuple
    model_config_anchor: _IdentityAnchor


@dataclass(frozen=True)
class _PackedCpuSnapshot:
    ragged: object
    state: object
    derived_cache: object
    attestation: _CpuAttestation


@dataclass(frozen=True)
class _ResidentSignature:
    binding_object_ids: tuple
    tensor_records: tuple


def _epoch_values(value):
    return (
        int(value.membership_epoch), int(value.ragged_epoch),
        int(value.state_epoch), int(value.cache_epoch),
        int(value.cache_source_ragged_epoch),
    )


def _copy_epochs(value):
    return A4ResidentArenaEpochs(*_epoch_values(value))


def _anchor_identity_values(value):
    return value.labels, tuple(id(item) for item in value.objects)


def _source_identity_values(value):
    return (
        value.membership.values,
        _anchor_identity_values(value.membership.anchor),
        str(value.ragged_digest), str(value.state_digest),
        str(value.derived_cache_digest), str(value.live_cache_digest),
        bool(value.cache_exact),
        _anchor_identity_values(value.ragged_anchor),
        _anchor_identity_values(value.state_anchor),
        _anchor_identity_values(value.cache_anchor),
        str(value.model_config_digest), tuple(value.model_config_flags),
        _anchor_identity_values(value.model_config_anchor),
    )


@dataclass(frozen=True)
class _ArenaCreationSeal:
    """Detached scalar seal plus strong creation-time identity anchors."""

    arena_ref: object
    source_ref: object
    binding_ref: object
    epochs_ref: object
    config_state_ref: object
    resident_signature_ref: object
    arena_id: str
    storage_generation: int
    device: str
    config_state: tuple
    epochs_values: tuple
    source_values: tuple
    resident_signature_values: tuple
    expected_digests: tuple


@dataclass(frozen=True)
class _OwnerCanonicalGuard:
    owner_token_ref: object
    arena_ref: object
    arena_seal: _ArenaCreationSeal
    arena_seal_snapshot: _ArenaCreationSeal
    config_ref: object
    epochs_ref: object
    last_observed_ref: object
    last_observed_values: tuple
    lifecycle: str
    epochs_values: tuple
    device: str
    config_state: tuple
    storage_generation: int
    lease_serial: int
    issued_lease_token: object
    active_lease_token: object
    active_lease_consumed: bool


@dataclass(frozen=True)
class _CandidateCreationSeal:
    candidate_ref: object
    owner_token: object
    expected_arena_ref: object
    expected_arena_id: str
    expected_lifecycle: str
    expected_epochs_ref: object
    expected_epochs_values: tuple
    expected_storage_generation: int
    source_ref: object
    source_values: tuple
    arena_ref: object
    arena_seal: _ArenaCreationSeal
    arena_seal_snapshot: _ArenaCreationSeal


@dataclass(frozen=True)
class _A4ResidentValidationSnapshot:
    """Detached diagnostic D2H copies; never a CPU publish authority."""

    arena_id: str
    storage_generation: int
    lifecycle: str
    epochs: A4ResidentArenaEpochs
    device: str
    resident_signature: _ResidentSignature
    ragged: object
    state: object
    cache: object


@dataclass
class _A4ResidentArena:
    arena_id: str
    storage_generation: int
    device: str
    config_state: tuple
    epochs: A4ResidentArenaEpochs
    source: _CpuAttestation
    binding: object
    resident_signature: _ResidentSignature
    expected_ragged_digest: str
    expected_state_digest: str
    expected_cache_digest: str
    lifecycle: str = COHERENT
    dirty_domains: frozenset = frozenset()
    invalid_reason: str = ''
    active_leases: int = 0
    lease_serial: int = 0
    retired: bool = False

    def __getstate__(self):
        raise A4ResidentArenaScopeError(
            'resident arenas are not serializable biology authority'
        )


_REBUILD_FACTORY_TOKEN = object()
_LEASE_FACTORY_TOKEN = object()


@dataclass
class _A4PreparedRebuild:
    _factory_token: object
    _owner_token: object
    _expected_arena_id: str
    _expected_lifecycle: str
    _expected_epochs: A4ResidentArenaEpochs
    _expected_storage_generation: int
    _source: _CpuAttestation
    _arena: _A4ResidentArena
    _seal: object = None
    _consumed: bool = False

    def __getstate__(self):
        raise A4ResidentArenaScopeError(
            'resident rebuild candidates are not serializable'
        )


_ARENA_ID_LOCK = threading.Lock()
_ARENA_ID_COUNTER = itertools.count(1)


def _next_arena_id():
    with _ARENA_ID_LOCK:
        serial = next(_ARENA_ID_COUNTER)
    return '%s:%d' % (SCHEMA_VERSION, int(serial))


def _canonical_device(value):
    if torch is None:
        raise A4ResidentArenaError(
            'PyTorch is required; A4.9a has no NumPy residency fallback'
        )
    if value is None or str(value) == 'auto':
        raise A4ResidentArenaError(
            'A4.9a requires an explicit cpu or cuda device'
        )
    try:
        device = torch.device(str(value))
    except Exception as exc:
        raise A4ResidentArenaError(
            'invalid A4.9a device: %s' % value
        ) from exc
    if device.type not in ('cpu', 'cuda'):
        raise A4ResidentArenaError(
            'A4.9a supports explicit cpu or cuda devices only'
        )
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise A4ResidentArenaError(
            'CUDA explicitly requested but unavailable'
        )
    return str(device)


def _canonical_config(value):
    if value is None:
        raise A4ResidentArenaError(
            'A4.9a requires an explicit fixed-capacity config'
        )
    try:
        config = a4.GPU068A4Config.from_state(value)
    except Exception as exc:
        raise A4ResidentArenaError(
            'invalid A4.9a fixed-capacity config'
        ) from exc
    return config


def _config_state(config):
    return tuple(
        (item.name, int(getattr(config, item.name)))
        for item in fields(a4.GPU068A4Config)
    )


def _anchor(entries):
    labels = []
    objects = []
    for label, value in entries:
        labels.append(str(label))
        objects.append(value)
    return _IdentityAnchor(tuple(labels), tuple(objects))


def _same_anchor(left, right):
    return (
        left.labels == right.labels
        and len(left.objects) == len(right.objects)
        and all(a is b for a, b in zip(left.objects, right.objects))
    )


def _collect_mutable_objects(value, label, output, seen):
    if isinstance(value, np.ndarray):
        if id(value) not in seen:
            seen.add(id(value))
            output.append((label, value))
        return
    if isinstance(value, Mapping):
        if id(value) in seen:
            return
        seen.add(id(value))
        output.append((label, value))
        for index, (key, item) in enumerate(value.items()):
            _collect_mutable_objects(
                key, '%s.key[%d]' % (label, index), output, seen,
            )
            _collect_mutable_objects(
                item, '%s.value[%d]' % (label, index), output, seen,
            )
        return
    if isinstance(value, (list, tuple)):
        if id(value) in seen:
            return
        seen.add(id(value))
        output.append((label, value))
        for index, item in enumerate(value):
            _collect_mutable_objects(
                item, '%s[%d]' % (label, index), output, seen,
            )
        return
    # Immutable scalar identities also matter at this trust boundary.  The
    # strong reference prevents id reuse and detects an equal-value rebind
    # (notably the common A3 template-lesion unpack assignment).
    output.append((label, value))


def _field_anchor(cells, names, prefix):
    entries = []
    seen = set()
    for ci, cell in enumerate(cells):
        for name in names:
            if hasattr(cell, name):
                _collect_mutable_objects(
                    getattr(cell, name),
                    '%s.cell[%d].%s' % (prefix, ci, name),
                    entries, seen,
                )
    return _anchor(entries)


_RAGGED_SOURCE_FIELDS = (
    'genomes', 'genome_lesions', 'replication_template',
    'replication_copy', 'replication_template_lesion',
    'replication_fractional',
)

_STATE_SOURCE_FIELDS = (
    'pools', 'membrane', 'membrane_oxidation', 'damage_trace',
    'proteins', 'damaged_proteins', 'last_receptor_activity',
    'last_control_vectors', 'last_control_scalars', 'neural_attachments',
    'radius', 'current_stress', 'division_progress', 'last_translation',
    'last_quiescence', 'cumulative_proofreading_atp',
    'last_edna_signal', 'last_corpse_signal', 'last_necrotoxin_signal',
    'behavioural_quiescence',
)


def _digest_update(digest, value, seen):
    if value is None:
        digest.update(b'N;')
    elif isinstance(value, (bool, np.bool_)):
        digest.update(b'B1;' if bool(value) else b'B0;')
    elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        digest.update(b'I')
        digest.update(str(int(value)).encode('ascii'))
        digest.update(b';')
    elif isinstance(value, (float, np.floating)):
        digest.update(b'F')
        raw = np.asarray(value)
        digest.update(raw.dtype.str.encode('ascii'))
        digest.update(np.ascontiguousarray(raw).tobytes())
        digest.update(b';')
    elif isinstance(value, str):
        raw = value.encode('utf-8')
        digest.update(b'S%d:' % len(raw))
        digest.update(raw)
    elif isinstance(value, bytes):
        digest.update(b'Y%d:' % len(value))
        digest.update(value)
    elif isinstance(value, np.ndarray):
        digest.update(b'A')
        digest.update(value.dtype.str.encode('ascii'))
        digest.update(repr(tuple(value.shape)).encode('ascii'))
        digest.update(np.ascontiguousarray(value).tobytes(order='C'))
    elif isinstance(value, Mapping):
        if id(value) in seen:
            raise A4ResidentArenaError('cyclic mapping in CPU gene cache')
        seen.add(id(value))
        digest.update(b'M%d{' % len(value))
        for key, item in value.items():
            _digest_update(digest, key, seen)
            _digest_update(digest, item, seen)
        digest.update(b'}')
        seen.remove(id(value))
    elif isinstance(value, (list, tuple)):
        if id(value) in seen:
            raise A4ResidentArenaError('cyclic sequence in CPU gene cache')
        seen.add(id(value))
        digest.update(b'L%d[' % len(value))
        for item in value:
            _digest_update(digest, item, seen)
        digest.update(b']')
        seen.remove(id(value))
    else:
        raise A4ResidentArenaError(
            'unsupported CPU gene-cache value type: %s' %
            type(value).__name__
        )


def _value_digest(value):
    digest = hashlib.sha256()
    _digest_update(digest, value, set())
    return digest.hexdigest()


def _strict_membership(world):
    if world is None or not hasattr(world, 'cells') or not hasattr(world, 'config'):
        raise A4ResidentArenaError(
            'A4.9a source must be a world exposing cells and config'
        )
    container = world.cells
    if not isinstance(container, (list, tuple)):
        raise A4ResidentArenaError(
            'world.cells must be one concrete ordered list or tuple'
        )
    cells = tuple(container)
    values = []
    entries = [('world', world), ('world.cells', container)]
    for index, cell in enumerate(cells):
        raw_id = getattr(cell, 'cell_id', None)
        raw_generation = getattr(cell, 'generation', None)
        raw_alive = getattr(cell, 'alive', None)
        if (isinstance(raw_id, (bool, np.bool_))
                or not isinstance(raw_id, (int, np.integer))
                or int(raw_id) < 0):
            raise A4ResidentArenaError(
                'world cell[%d] has invalid cell_id' % index
            )
        if (isinstance(raw_generation, (bool, np.bool_))
                or not isinstance(raw_generation, (int, np.integer))
                or int(raw_generation) < 0):
            raise A4ResidentArenaError(
                'world cell[%d] has invalid generation' % index
            )
        if not isinstance(raw_alive, (bool, np.bool_)):
            raise A4ResidentArenaError(
                'world cell[%d] has invalid alive participation flag' % index
            )
        values.append((int(raw_id), int(raw_generation), bool(raw_alive)))
        entries.append(('world.cells[%d]' % index, cell))
    if len({item[0] for item in values}) != len(values):
        raise A4ResidentArenaError('duplicate world cell IDs are forbidden')
    return cells, _MembershipAttestation(tuple(values), _anchor(entries))


def _model_config_attestation(model_config):
    if model_config is None:
        raise A4ResidentArenaError('world model config is absent')
    flags = []
    for name in (
            'gene_expression', 'external_translator', 'protein_repair',
            'quiescence', 'quiescence_effector'):
        if not hasattr(model_config, name):
            raise A4ResidentArenaError(
                'world model config lacks %s' % name
            )
        value = getattr(model_config, name)
        if not isinstance(value, (bool, np.bool_)):
            raise A4ResidentArenaError(
                'world model config %s must be boolean' % name
            )
        flags.append((name, bool(value)))
    state_method = getattr(model_config, 'state_dict', None)
    if not callable(state_method):
        raise A4ResidentArenaError(
            'world model config must expose state_dict()'
        )
    try:
        state = copy.deepcopy(state_method())
        payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        raise A4ResidentArenaError(
            'world model config state is not canonicalizable'
        ) from exc
    digest = hashlib.sha256(payload).hexdigest()
    return digest, tuple(flags), _anchor((('world.config', model_config),))


def _gene_specs_exact(cells, expected_specs):
    if len(cells) != len(expected_specs):
        return False
    for cell, expected in zip(cells, expected_specs):
        actual = getattr(cell, 'gene_specs', None)
        try:
            if not a4._gene_specs_exact(actual, expected):
                return False
        except Exception:
            return False
    return True


def _pack_cpu_snapshot(world, config, require_cache_exact):
    cells, membership = _strict_membership(world)
    model_digest, model_flags, model_anchor = _model_config_attestation(
        world.config,
    )
    try:
        ragged = a4.FullFidelityA4GenomeAdapter(config).pack_cells(cells)
        derived_cache = a4.decode_a4_gene_cache_numpy(ragged)
        expected_specs = derived_cache.materialize_gene_specs_host()
    except Exception as exc:
        raise A4ResidentArenaError(
            'CPU ragged/cache preflight failed'
        ) from exc

    cache_exact = _gene_specs_exact(cells, expected_specs)
    if require_cache_exact and not cache_exact:
        raise A4ResidentArenaError(
            'CPU gene_specs differs from the complete-genome decode'
        )

    state_cells = cells
    if not cache_exact:
        # Audit S independently from a stale CPU cache without touching the
        # source objects.  A build never takes this path.
        isolated = []
        for cell, expected in zip(cells, expected_specs):
            item = copy.copy(cell)
            item.gene_specs = copy.deepcopy(expected)
            isolated.append(item)
        state_cells = tuple(isolated)
    try:
        state = a4.pack_a4_translation_state(
            state_cells, ragged, world.config, config,
        )
        ragged_digest = a4._ragged_translation_provenance(ragged)
        state_digest = a4._translation_state_provenance(state)
        derived_cache_digest = a4._gene_cache_provenance(derived_cache)
    except Exception as exc:
        raise A4ResidentArenaError(
            'CPU translation-state preflight failed'
        ) from exc
    if state.source_provenance != ragged_digest:
        raise A4ResidentArenaError(
            'CPU translation state is not bound to the exact ragged source'
        )

    live_specs = [getattr(cell, 'gene_specs', None) for cell in cells]
    try:
        live_cache_digest = _value_digest(live_specs)
    except A4ResidentArenaError:
        live_cache_digest = 'malformed:' + hashlib.sha256(
            repr(tuple(type(value).__name__ for value in live_specs)).encode('utf-8')
        ).hexdigest()

    ragged_anchor = _field_anchor(cells, _RAGGED_SOURCE_FIELDS, 'R')
    state_anchor = _field_anchor(cells, _STATE_SOURCE_FIELDS, 'S')
    cache_anchor = _field_anchor(cells, ('gene_specs',), 'C')
    attestation = _CpuAttestation(
        membership=membership,
        ragged_digest=ragged_digest,
        state_digest=state_digest,
        derived_cache_digest=derived_cache_digest,
        live_cache_digest=live_cache_digest,
        cache_exact=bool(cache_exact),
        ragged_anchor=ragged_anchor,
        state_anchor=state_anchor,
        cache_anchor=cache_anchor,
        model_config_digest=model_digest,
        model_config_flags=model_flags,
        model_config_anchor=model_anchor,
    )
    return _PackedCpuSnapshot(ragged, state, derived_cache, attestation)


def _same_membership(left, right):
    return (
        left.values == right.values
        and _same_anchor(left.anchor, right.anchor)
    )


def _same_source(left, right):
    return (
        _same_membership(left.membership, right.membership)
        and left.ragged_digest == right.ragged_digest
        and left.state_digest == right.state_digest
        and left.derived_cache_digest == right.derived_cache_digest
        and left.live_cache_digest == right.live_cache_digest
        and left.cache_exact == right.cache_exact
        and _same_anchor(left.ragged_anchor, right.ragged_anchor)
        and _same_anchor(left.state_anchor, right.state_anchor)
        and _same_anchor(left.cache_anchor, right.cache_anchor)
        and left.model_config_digest == right.model_config_digest
        and left.model_config_flags == right.model_config_flags
        and _same_anchor(left.model_config_anchor, right.model_config_anchor)
    )


def _stable_cpu_snapshot(world, config, require_cache_exact=True):
    first = _pack_cpu_snapshot(world, config, require_cache_exact)
    second = _pack_cpu_snapshot(world, config, require_cache_exact)
    if not _same_source(first.attestation, second.attestation):
        raise A4ResidentArenaError(
            'CPU world changed during A4.9a preflight attestation'
        )
    return second


def _source_domain_changes(previous, current):
    changes = set()
    if not _same_membership(previous.membership, current.membership):
        changes.add('M')
    if (previous.ragged_digest != current.ragged_digest
            or not _same_anchor(previous.ragged_anchor, current.ragged_anchor)):
        changes.add('R')
    if (previous.state_digest != current.state_digest
            or not _same_anchor(previous.state_anchor, current.state_anchor)
            or previous.model_config_digest != current.model_config_digest
            or previous.model_config_flags != current.model_config_flags
            or not _same_anchor(
                previous.model_config_anchor, current.model_config_anchor,
            )):
        changes.add('S')
    if (previous.derived_cache_digest != current.derived_cache_digest
            or previous.live_cache_digest != current.live_cache_digest
            or previous.cache_exact != current.cache_exact
            or not _same_anchor(previous.cache_anchor, current.cache_anchor)):
        changes.add('C')
    if 'R' in changes:
        changes.update(('S', 'C'))
    if 'M' in changes:
        # A membership/order/object change invalidates every packed row and
        # every provenance relation, even when replacement bytes happen to
        # compare equal.
        changes.update(('R', 'S', 'C'))
    return frozenset(changes)


def _advance_epochs(epochs, changes, cache_built=False):
    membership = epochs.membership_epoch + int('M' in changes)
    ragged = epochs.ragged_epoch + int('R' in changes)
    state = epochs.state_epoch + int('S' in changes)
    cache = epochs.cache_epoch + int('C' in changes)
    cache_source = (
        ragged if cache_built else epochs.cache_source_ragged_epoch
    )
    return A4ResidentArenaEpochs(
        membership, ragged, state, cache, cache_source,
    )


def _resident_signature(binding):
    try:
        a4._require_translation_binding(binding)
    except Exception as exc:
        raise A4ResidentArenaError(
            'resident translation binding failed its trust boundary'
        ) from exc
    records = []
    for group, batch, names in (
            ('ragged', binding.ragged, a4._ARRAY_FIELDS),
            ('state', binding.state, a4._TRANSLATION_ARRAY_FIELDS),
            ('cache', binding.cache, a4._GENE_ARRAY_FIELDS)):
        for name in names:
            value = getattr(batch, name)
            if torch is None or not isinstance(value, torch.Tensor):
                raise A4ResidentArenaError(
                    'resident %s.%s is not a Torch tensor' % (group, name)
                )
            records.append((
                group, name, id(value), int(value.data_ptr()),
                int(value._version), str(value.device), str(value.dtype),
                tuple(int(item) for item in value.shape),
            ))
    if len(records) != 50:
        raise A4ResidentArenaError(
            'resident arena must own exactly 50 tensor fields, observed=%d' %
            len(records)
        )
    return _ResidentSignature(
        binding_object_ids=(
            id(binding), id(binding.ragged), id(binding.state), id(binding.cache),
        ),
        tensor_records=tuple(records),
    )


def _resident_readback(binding):
    try:
        ragged = binding.ragged.to_numpy()
        state = binding.state.to_numpy()
        cache = binding.cache.to_numpy()
        digests = (
            a4._ragged_translation_provenance(ragged),
            a4._translation_state_provenance(state),
            a4._gene_cache_provenance(cache),
        )
    except Exception as exc:
        raise A4ResidentArenaError(
            'resident diagnostic readback failed validation'
        ) from exc
    return ragged, state, cache, digests


def _make_arena_creation_seal(arena):
    signature_values = (
        tuple(arena.resident_signature.binding_object_ids),
        tuple(arena.resident_signature.tensor_records),
    )
    return _ArenaCreationSeal(
        arena_ref=arena,
        source_ref=arena.source,
        binding_ref=arena.binding,
        epochs_ref=arena.epochs,
        config_state_ref=arena.config_state,
        resident_signature_ref=arena.resident_signature,
        arena_id=str(arena.arena_id),
        storage_generation=int(arena.storage_generation),
        device=str(arena.device),
        config_state=tuple(arena.config_state),
        epochs_values=_epoch_values(arena.epochs),
        source_values=_source_identity_values(arena.source),
        resident_signature_values=signature_values,
        expected_digests=(
            str(arena.expected_ragged_digest),
            str(arena.expected_state_digest),
            str(arena.expected_cache_digest),
        ),
    )


def _copy_arena_creation_seal(seal):
    return _ArenaCreationSeal(
        arena_ref=seal.arena_ref,
        source_ref=seal.source_ref,
        binding_ref=seal.binding_ref,
        epochs_ref=seal.epochs_ref,
        config_state_ref=seal.config_state_ref,
        resident_signature_ref=seal.resident_signature_ref,
        arena_id=str(seal.arena_id),
        storage_generation=int(seal.storage_generation),
        device=str(seal.device),
        config_state=tuple(seal.config_state),
        epochs_values=tuple(seal.epochs_values),
        source_values=tuple(seal.source_values),
        resident_signature_values=tuple(seal.resident_signature_values),
        expected_digests=tuple(seal.expected_digests),
    )


def _arena_creation_seal_values(seal):
    if not isinstance(seal, _ArenaCreationSeal):
        raise A4ResidentArenaError('resident arena creation seal is malformed')
    return (
        id(seal.arena_ref), id(seal.source_ref), id(seal.binding_ref),
        id(seal.epochs_ref), id(seal.config_state_ref),
        id(seal.resident_signature_ref), str(seal.arena_id),
        int(seal.storage_generation), str(seal.device),
        tuple(seal.config_state), tuple(seal.epochs_values),
        tuple(seal.source_values), tuple(seal.resident_signature_values),
        tuple(seal.expected_digests),
    )


def _require_arena_creation_seal(arena, seal):
    if not isinstance(seal, _ArenaCreationSeal) or arena is not seal.arena_ref:
        raise A4ResidentArenaError('resident arena creation identity differs')
    if (arena.source is not seal.source_ref
            or arena.binding is not seal.binding_ref
            or arena.epochs is not seal.epochs_ref
            or arena.config_state is not seal.config_state_ref
            or arena.resident_signature is not seal.resident_signature_ref
            or str(arena.arena_id) != seal.arena_id
            or int(arena.storage_generation) != seal.storage_generation
            or str(arena.device) != seal.device
            or tuple(arena.config_state) != seal.config_state
            or _epoch_values(arena.epochs) != seal.epochs_values
            or _source_identity_values(arena.source) != seal.source_values
            or (
                tuple(arena.resident_signature.binding_object_ids),
                tuple(arena.resident_signature.tensor_records),
            ) != seal.resident_signature_values
            or (
                str(arena.expected_ragged_digest),
                str(arena.expected_state_digest),
                str(arena.expected_cache_digest),
            ) != seal.expected_digests
            or seal.expected_digests != (
                str(arena.source.ragged_digest),
                str(arena.source.state_digest),
                str(arena.source.derived_cache_digest),
            )):
        raise A4ResidentArenaError(
            'resident arena immutable metadata/source/digest seal differs'
        )
    return arena


def _attest_resident(arena, seal, readback=False):
    _require_arena_creation_seal(arena, seal)
    if arena.retired:
        raise A4ResidentArenaError('resident arena has been retired')
    if arena.lifecycle not in _LIFECYCLES:
        raise A4ResidentArenaError('resident lifecycle is malformed')
    if arena.epochs.cache_source_ragged_epoch != arena.epochs.ragged_epoch:
        raise A4ResidentArenaError(
            'resident cache source ragged epoch differs'
        )
    actual = _resident_signature(arena.binding)
    if actual != arena.resident_signature:
        raise A4ResidentArenaError(
            'resident tensor object/pointer/version/metadata changed'
        )
    if not readback:
        return None
    ragged, state, cache, digests = _resident_readback(arena.binding)
    expected = (
        arena.expected_ragged_digest,
        arena.expected_state_digest,
        arena.expected_cache_digest,
    )
    if digests != expected:
        raise A4ResidentArenaError(
            'resident tensor content differs from the immutable upload'
        )
    return ragged, state, cache


def _build_resident_arena(
        packed, config, device, epochs, storage_generation):
    if epochs.cache_source_ragged_epoch != epochs.ragged_epoch:
        raise A4ResidentArenaError(
            'fresh resident cache must source the fresh ragged epoch'
        )
    try:
        resident_ragged = packed.ragged.to_torch(device)
        resident_state = packed.state.to_torch(device)
        binding = a4.bind_a4_translation(resident_ragged, resident_state)
    except Exception as exc:
        raise A4ResidentArenaError(
            'fresh A4.9a H2D/decode construction failed'
        ) from exc
    signature = _resident_signature(binding)
    arena = _A4ResidentArena(
        arena_id=_next_arena_id(),
        storage_generation=int(storage_generation),
        device=str(device),
        config_state=_config_state(config),
        epochs=epochs,
        source=packed.attestation,
        binding=binding,
        resident_signature=signature,
        expected_ragged_digest=packed.attestation.ragged_digest,
        expected_state_digest=packed.attestation.state_digest,
        expected_cache_digest=packed.attestation.derived_cache_digest,
    )
    # This exact diagnostic readback seals content, including mutations made
    # through Tensor.data which do not advance Torch's public version counter.
    seal = _make_arena_creation_seal(arena)
    _attest_resident(arena, seal, readback=True)
    return arena, seal


class _A4ResidentReadLease:
    """Private, one-use, context-managed view of one immutable binding."""

    __slots__ = (
        '_factory_token', '_owner', '_owner_token', '_world', '_arena_id',
        '_epochs', '_cache_source_ragged_epoch', '_device', '_signature',
        '_serial', '_lease_token', '_entered', '_consumed', '_closed',
    )

    def __init__(
            self, factory_token, owner, world, arena, serial, lease_token):
        if factory_token is not _LEASE_FACTORY_TOKEN:
            raise A4ResidentArenaScopeError(
                'read leases are issued only by their arena owner'
            )
        self._factory_token = factory_token
        self._owner = owner
        self._owner_token = owner._owner_token
        self._world = world
        self._arena_id = arena.arena_id
        self._epochs = _copy_epochs(arena.epochs)
        self._cache_source_ragged_epoch = (
            arena.epochs.cache_source_ragged_epoch
        )
        self._device = arena.device
        self._signature = _ResidentSignature(
            tuple(arena.resident_signature.binding_object_ids),
            tuple(arena.resident_signature.tensor_records),
        )
        self._serial = int(serial)
        self._lease_token = lease_token
        self._entered = False
        self._consumed = False
        self._closed = False

    def __enter__(self):
        if self._entered or self._closed:
            raise A4ResidentArenaError('resident read lease cannot be reopened')
        self._owner._activate_lease(self)
        self._entered = True
        return self

    def _binding_once(self):
        if not self._entered or self._closed:
            raise A4ResidentArenaError('resident read lease is not open')
        if self._consumed:
            raise A4ResidentArenaError('resident read lease was already consumed')
        binding = self._owner._consume_lease(self)
        self._consumed = True
        return binding

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self._entered and not self._closed:
                self._owner._close_lease(self)
        finally:
            self._closed = True
        return False

    def __getstate__(self):
        raise A4ResidentArenaScopeError('resident read leases are not serializable')


class A4ResidentArenaOwner:
    """Mutable coherence owner for one immutable active resident generation."""

    __slots__ = (
        '_owner_token', '_lock', '_config', '_device', '_arena', '_epochs',
        '_arena_seal', '_last_observed', '_storage_generation', '_lifecycle',
        '_lease_serial', '_issued_lease_token', '_active_lease_token',
        '_active_lease_consumed', '_guard',
    )

    def __init__(self, *args, **kwargs):
        raise A4ResidentArenaScopeError(
            'A4ResidentArenaOwner is created only by from_cpu()'
        )

    @classmethod
    def from_cpu(cls, world, a4_config, device):
        """Build and seal one fresh world-wide immutable resident shadow."""
        config = _canonical_config(a4_config)
        resolved = _canonical_device(device)
        packed = _stable_cpu_snapshot(world, config, require_cache_exact=True)
        epochs = A4ResidentArenaEpochs(1, 1, 1, 1, 1)
        arena, arena_seal = _build_resident_arena(
            packed, config, resolved, epochs, storage_generation=1,
        )
        after = _stable_cpu_snapshot(world, config, require_cache_exact=True)
        if not _same_source(packed.attestation, after.attestation):
            raise A4ResidentArenaError(
                'CPU world changed during fresh resident construction'
            )
        owner = object.__new__(cls)
        owner._owner_token = object()
        owner._lock = threading.RLock()
        owner._config = config
        owner._device = resolved
        owner._arena = arena
        owner._arena_seal = arena_seal
        owner._epochs = _copy_epochs(epochs)
        owner._last_observed = after.attestation
        owner._storage_generation = 1
        owner._lifecycle = COHERENT
        owner._lease_serial = 0
        owner._issued_lease_token = None
        owner._active_lease_token = None
        owner._active_lease_consumed = False
        owner._guard = None
        owner._refresh_guard()
        return owner

    def _refresh_guard(self):
        self._guard = _OwnerCanonicalGuard(
            owner_token_ref=self._owner_token,
            arena_ref=self._arena,
            arena_seal=self._arena_seal,
            arena_seal_snapshot=_copy_arena_creation_seal(self._arena_seal),
            config_ref=self._config,
            epochs_ref=self._epochs,
            last_observed_ref=self._last_observed,
            last_observed_values=_source_identity_values(self._last_observed),
            lifecycle=str(self._lifecycle),
            epochs_values=_epoch_values(self._epochs),
            device=str(self._device),
            config_state=_config_state(self._config),
            storage_generation=int(self._storage_generation),
            lease_serial=int(self._lease_serial),
            issued_lease_token=self._issued_lease_token,
            active_lease_token=self._active_lease_token,
            active_lease_consumed=self._active_lease_consumed,
        )

    def _require_guard(self):
        guard = self._guard
        if not isinstance(guard, _OwnerCanonicalGuard):
            raise A4ResidentArenaError('owner canonical guard is absent')
        try:
            config_state = _config_state(self._config)
            owner_epochs = _epoch_values(self._epochs)
            arena_seal_values = _arena_creation_seal_values(self._arena_seal)
        except Exception as exc:
            raise A4ResidentArenaError(
                'owner canonical metadata is malformed'
            ) from exc
        active_count = 1 if self._active_lease_token is not None else 0
        if (self._arena is not guard.arena_ref
                or self._owner_token is not guard.owner_token_ref
                or arena_seal_values != _arena_creation_seal_values(
                    guard.arena_seal_snapshot,
                )
                or self._config is not guard.config_ref
                or self._epochs is not guard.epochs_ref
                or self._last_observed is not guard.last_observed_ref
                or _source_identity_values(self._last_observed)
                != guard.last_observed_values
                or str(self._lifecycle) != guard.lifecycle
                or str(self._arena.lifecycle) != guard.lifecycle
                or owner_epochs != guard.epochs_values
                or str(self._device) != guard.device
                or config_state != guard.config_state
                or int(self._storage_generation) != guard.storage_generation
                or int(self._lease_serial) != guard.lease_serial
                or int(self._arena.lease_serial) != guard.lease_serial
                or self._issued_lease_token is not guard.issued_lease_token
                or self._active_lease_token is not guard.active_lease_token
                or type(self._active_lease_consumed) is not bool
                or self._active_lease_consumed
                is not guard.active_lease_consumed
                or (self._active_lease_token is None
                    and self._active_lease_consumed)
                or int(self._arena.active_leases) != active_count
                or self._arena.retired):
            raise A4ResidentArenaError(
                'owner/arena canonical metadata relation differs'
            )
        if guard.lifecycle != INVALID:
            _require_arena_creation_seal(self._arena, self._arena_seal)
        return guard

    def _force_metadata_invalid(self, reason):
        """Fail closed from detached guard values; never trust modified fields."""
        guard = self._guard
        if not isinstance(guard, _OwnerCanonicalGuard):
            raise A4ResidentArenaError(reason)
        was_invalid = guard.lifecycle == INVALID
        self._arena = guard.arena_ref
        self._arena_seal = _copy_arena_creation_seal(
            guard.arena_seal_snapshot,
        )
        self._owner_token = guard.owner_token_ref
        self._config = a4.GPU068A4Config.from_state(dict(guard.config_state))
        self._device = guard.device
        self._storage_generation = guard.storage_generation
        canonical_epochs = A4ResidentArenaEpochs(*guard.epochs_values)
        self._epochs = (
            canonical_epochs if was_invalid else _advance_epochs(
                canonical_epochs, frozenset(('M', 'R', 'S', 'C')),
                cache_built=False,
            )
        )
        self._last_observed = guard.last_observed_ref
        self._lifecycle = INVALID
        self._arena.lifecycle = INVALID
        self._arena.dirty_domains = frozenset(('M', 'R', 'S', 'C'))
        self._arena.invalid_reason = str(reason)
        self._lease_serial = guard.lease_serial + (0 if was_invalid else 1)
        self._arena.lease_serial = self._lease_serial
        self._issued_lease_token = None
        self._active_lease_token = guard.active_lease_token
        self._active_lease_consumed = guard.active_lease_consumed
        self._arena.active_leases = (
            1 if self._active_lease_token is not None else 0
        )
        self._refresh_guard()

    def _guard_or_invalidate(self):
        try:
            return self._require_guard()
        except Exception as exc:
            self._force_metadata_invalid(
                'owner or immutable arena metadata was tampered'
            )
            raise A4ResidentArenaError(
                'owner or immutable arena metadata was tampered'
            ) from exc

    @property
    def arena_id(self):
        with self._lock:
            try:
                guard = self._require_guard()
            except Exception:
                self._force_metadata_invalid(
                    'owner or immutable arena metadata was tampered'
                )
                guard = self._guard
            return str(guard.arena_seal.arena_id)

    @property
    def lifecycle(self):
        with self._lock:
            try:
                self._require_guard()
            except Exception:
                self._force_metadata_invalid(
                    'owner or immutable arena metadata was tampered'
                )
            return str(self._lifecycle)

    @property
    def epochs(self):
        with self._lock:
            try:
                self._require_guard()
            except Exception:
                self._force_metadata_invalid(
                    'owner or immutable arena metadata was tampered'
                )
            return _copy_epochs(self._epochs)

    def _transition_invalid(
            self, reason, domains=('M', 'R', 'S', 'C'),
            advance_unobserved=False):
        arena = self._arena
        was_invalid = self._lifecycle == INVALID
        if advance_unobserved and not was_invalid:
            self._epochs = _advance_epochs(
                self._epochs, frozenset(('M', 'R', 'S', 'C')),
                cache_built=False,
            )
        self._lifecycle = INVALID
        arena.lifecycle = INVALID
        arena.dirty_domains = frozenset(
            set(arena.dirty_domains).union(domains)
        )
        arena.invalid_reason = str(reason)
        if not was_invalid:
            self._lease_serial += 1
            arena.lease_serial = self._lease_serial
            self._issued_lease_token = None
        if self._active_lease_token is None:
            self._active_lease_consumed = False
        self._refresh_guard()

    def _record_observation(self, source):
        changes = _source_domain_changes(self._last_observed, source)
        if changes:
            self._epochs = _advance_epochs(
                self._epochs, changes, cache_built=False,
            )
            self._last_observed = source
        return changes

    def audit_cpu(self, world):
        """Repack exact CPU authority and update one-way dirty diagnostics."""
        with self._lock:
            self._guard_or_invalidate()
            if self._lifecycle != INVALID:
                try:
                    _attest_resident(
                        self._arena, self._arena_seal, readback=True,
                    )
                except Exception as exc:
                    self._transition_invalid(
                        'resident integrity attestation failed', ('R', 'S', 'C'),
                        advance_unobserved=True,
                    )
                    raise A4ResidentArenaError(
                        'resident integrity attestation failed'
                    ) from exc
            try:
                packed = _stable_cpu_snapshot(
                    world, self._config, require_cache_exact=False,
                )
            except Exception as exc:
                self._transition_invalid(
                    'CPU source is malformed', advance_unobserved=True,
                )
                raise A4ResidentArenaError(
                    'CPU source audit failed closed'
                ) from exc
            source = packed.attestation
            observed_changes = self._record_observation(source)
            active_changes = _source_domain_changes(self._arena.source, source)
            if 'M' in active_changes:
                self._transition_invalid(
                    'world membership/order/identity differs', active_changes,
                )
            elif active_changes:
                first_or_new_drift = (
                    self._lifecycle == COHERENT or bool(observed_changes)
                )
                if self._lifecycle == COHERENT:
                    self._lifecycle = CPU_NEWER
                    self._arena.lifecycle = CPU_NEWER
                self._arena.dirty_domains = frozenset(
                    set(self._arena.dirty_domains).union(active_changes)
                )
                self._arena.invalid_reason = ''
                if first_or_new_drift:
                    self._lease_serial += 1
                    self._arena.lease_serial = self._lease_serial
                    self._issued_lease_token = None
            elif (not source.cache_exact
                  or source.live_cache_digest
                  != self._arena.source.live_cache_digest):
                first_or_new_drift = (
                    self._lifecycle == COHERENT or bool(observed_changes)
                )
                if self._lifecycle == COHERENT:
                    self._lifecycle = CPU_NEWER
                    self._arena.lifecycle = CPU_NEWER
                self._arena.dirty_domains = frozenset(
                    set(self._arena.dirty_domains).union(('C',))
                )
                if first_or_new_drift:
                    self._lease_serial += 1
                    self._arena.lease_serial = self._lease_serial
                    self._issued_lease_token = None
            # A stale arena never returns to COHERENT, even if CPU values are
            # manually restored to a previous byte pattern.
            if observed_changes and self._lifecycle == INVALID:
                self._arena.dirty_domains = frozenset(
                    set(self._arena.dirty_domains).union(observed_changes)
                )
            self._refresh_guard()
            return self._lifecycle

    def prepare_rebuild(self, world):
        """Build a full candidate in separate storage without touching owner."""
        with self._lock:
            self._guard_or_invalidate()
            if self._lifecycle == COHERENT:
                raise A4ResidentArenaError(
                    'prepare_rebuild requires a prior stale audit or invalidation'
                )
            expected_arena = self._arena
            expected_arena_seal = self._arena_seal
            expected_id = self._arena_seal.arena_id
            expected_lifecycle = self._lifecycle
            expected_epochs = _copy_epochs(self._epochs)
            expected_generation = self._storage_generation
            previous_observed = self._last_observed
            config = self._config
            device = self._device
        packed = _stable_cpu_snapshot(
            world, config, require_cache_exact=True,
        )
        if not _same_source(previous_observed, packed.attestation):
            raise A4ResidentArenaError(
                'CPU source changed after its last audit; audit_cpu is required again'
            )
        changes = frozenset()
        candidate_epochs = _advance_epochs(
            expected_epochs, changes, cache_built=True,
        )
        candidate_arena, candidate_arena_seal = _build_resident_arena(
            packed, config, device, candidate_epochs,
            storage_generation=expected_generation + 1,
        )
        after = _stable_cpu_snapshot(
            world, config, require_cache_exact=True,
        )
        if not _same_source(packed.attestation, after.attestation):
            raise A4ResidentArenaError(
                'CPU world changed during separate rebuild construction'
            )
        candidate = _A4PreparedRebuild(
            _factory_token=_REBUILD_FACTORY_TOKEN,
            _owner_token=self._owner_token,
            _expected_arena_id=expected_id,
            _expected_lifecycle=expected_lifecycle,
            _expected_epochs=expected_epochs,
            _expected_storage_generation=expected_generation,
            _source=after.attestation,
            _arena=candidate_arena,
        )
        candidate._seal = _CandidateCreationSeal(
            candidate_ref=candidate,
            owner_token=self._owner_token,
            expected_arena_ref=expected_arena,
            expected_arena_id=expected_id,
            expected_lifecycle=expected_lifecycle,
            expected_epochs_ref=candidate._expected_epochs,
            expected_epochs_values=_epoch_values(expected_epochs),
            expected_storage_generation=expected_generation,
            source_ref=after.attestation,
            source_values=_source_identity_values(after.attestation),
            arena_ref=candidate_arena,
            arena_seal=candidate_arena_seal,
            arena_seal_snapshot=_copy_arena_creation_seal(
                candidate_arena_seal,
            ),
        )
        self._require_candidate_seal(
            candidate, expected_arena, expected_arena_seal,
        )
        return candidate

    def _require_candidate_seal(
            self, candidate, expected_arena, expected_arena_seal):
        seal = getattr(candidate, '_seal', None)
        if (not isinstance(candidate, _A4PreparedRebuild)
                or candidate._factory_token is not _REBUILD_FACTORY_TOKEN
                or candidate._owner_token is not self._owner_token
                or not isinstance(seal, _CandidateCreationSeal)
                or candidate is not seal.candidate_ref
                or seal.owner_token is not self._owner_token
                or seal.expected_arena_ref is not expected_arena
                or candidate._expected_arena_id != seal.expected_arena_id
                or candidate._expected_lifecycle != seal.expected_lifecycle
                or candidate._expected_epochs is not seal.expected_epochs_ref
                or _epoch_values(candidate._expected_epochs)
                != seal.expected_epochs_values
                or candidate._expected_storage_generation
                != seal.expected_storage_generation
                or candidate._source is not seal.source_ref
                or _source_identity_values(candidate._source)
                != seal.source_values
                or candidate._arena is not seal.arena_ref
                or _arena_creation_seal_values(seal.arena_seal)
                != _arena_creation_seal_values(seal.arena_seal_snapshot)
                or candidate._arena is expected_arena
                or candidate._arena.arena_id == expected_arena_seal.arena_id
                or candidate._arena.storage_generation
                != expected_arena_seal.storage_generation + 1):
            raise A4ResidentArenaError(
                'rebuild candidate creation seal or generation differs'
            )
        _require_arena_creation_seal(candidate._arena, seal.arena_seal)
        old_records = expected_arena_seal.resident_signature_values[1]
        new_records = seal.arena_seal.resident_signature_values[1]
        old_objects = {item[2] for item in old_records}
        new_objects = {item[2] for item in new_records}
        old_storage = {
            (item[5], item[3]) for item in old_records if int(item[3]) != 0
        }
        new_storage = {
            (item[5], item[3]) for item in new_records if int(item[3]) != 0
        }
        if old_objects.intersection(new_objects) or old_storage.intersection(new_storage):
            raise A4ResidentArenaError(
                'rebuild candidate aliases old resident tensor storage'
            )
        if (seal.arena_seal.device != expected_arena_seal.device
                or seal.arena_seal.config_state != expected_arena_seal.config_state
                or _epoch_values(candidate._arena.epochs)
                != _epoch_values(_advance_epochs(
                    candidate._expected_epochs, frozenset(), cache_built=True,
                ))
                or seal.arena_seal.epochs_values
                != _epoch_values(candidate._arena.epochs)):
            raise A4ResidentArenaError(
                'rebuild candidate device/config/epoch relation differs'
            )
        return seal

    def swap_rebuild(
            self, candidate, world, expected_arena_id,
            expected_lifecycle, expected_epochs):
        """CAS-swap one fully attested private candidate and retire the old."""
        with self._lock:
            self._guard_or_invalidate()
            candidate_seal = self._require_candidate_seal(
                candidate, self._arena, self._arena_seal,
            )
            if candidate._consumed:
                raise A4ResidentArenaError('rebuild candidate was already consumed')
            if expected_lifecycle not in _LIFECYCLES:
                raise A4ResidentArenaError('expected lifecycle is invalid')
            if not isinstance(expected_epochs, A4ResidentArenaEpochs):
                raise A4ResidentArenaError('expected epochs type differs')
            if (str(expected_arena_id) != self._arena_seal.arena_id
                    or expected_lifecycle != self._lifecycle
                    or _epoch_values(expected_epochs) != _epoch_values(self._epochs)
                    or candidate._expected_arena_id != self._arena_seal.arena_id
                    or candidate._expected_lifecycle != self._lifecycle
                    or _epoch_values(candidate._expected_epochs)
                    != _epoch_values(self._epochs)
                    or candidate._expected_storage_generation
                    != self._storage_generation):
                raise A4ResidentArenaError(
                    'rebuild compare-and-swap guard differs'
                )
            if self._active_lease_token is not None:
                raise A4ResidentArenaError(
                    'cannot swap while a resident read lease is active'
                )
            # Validate both generations and the CPU source before the sole
            # pointer publication below.  A failure changes neither owner nor
            # old arena.
            _attest_resident(
                candidate._arena, candidate_seal.arena_seal, readback=True,
            )
            if self._lifecycle != INVALID:
                try:
                    _attest_resident(
                        self._arena, self._arena_seal, readback=True,
                    )
                except Exception as exc:
                    self._transition_invalid(
                        'resident integrity changed before rebuild swap',
                        ('R', 'S', 'C'), advance_unobserved=True,
                    )
                    raise A4ResidentArenaError(
                        'old resident integrity changed before rebuild swap'
                    ) from exc
            fresh = _stable_cpu_snapshot(
                world, self._config, require_cache_exact=True,
            )
            if not _same_source(candidate._source, fresh.attestation):
                raise A4ResidentArenaError(
                    'CPU source differs from prepared rebuild candidate'
                )
            if (candidate._arena.epochs.cache_source_ragged_epoch
                    != candidate._arena.epochs.ragged_epoch):
                raise A4ResidentArenaError(
                    'candidate cache/ragged epoch relation differs'
                )
            old = self._arena
            old_seal = self._arena_seal
            self._arena = candidate._arena
            self._arena_seal = candidate_seal.arena_seal
            self._epochs = _copy_epochs(candidate._arena.epochs)
            self._last_observed = fresh.attestation
            self._storage_generation = candidate._arena.storage_generation
            self._lifecycle = COHERENT
            self._arena.lifecycle = COHERENT
            self._arena.lease_serial = self._lease_serial
            self._arena.active_leases = 0
            self._issued_lease_token = None
            self._active_lease_token = None
            self._active_lease_consumed = False
            candidate._consumed = True
            old.retired = True
            old.lifecycle = INVALID
            old.invalid_reason = 'retired after guarded rebuild swap'
            old.lease_serial += 1
            self._refresh_guard()
            return self

    def invalidate(self, reason='explicit caller invalidation'):
        """Irreversibly invalidate the active generation without touching CPU."""
        with self._lock:
            self._guard_or_invalidate()
            if self._active_lease_token is not None:
                raise A4ResidentArenaError(
                    'cannot invalidate while a resident read lease is active'
                )
            self._transition_invalid(
                str(reason), advance_unobserved=True,
            )
            return INVALID

    def _validation_snapshot(self, world):
        """Return detached NumPy readbacks only after a fresh coherent audit."""
        with self._lock:
            self._guard_or_invalidate()
            if self.audit_cpu(world) != COHERENT:
                raise A4ResidentArenaError(
                    'validation snapshot requires a coherent CPU shadow'
                )
            ragged, state, cache = _attest_resident(
                self._arena, self._arena_seal, readback=True,
            )
            return _A4ResidentValidationSnapshot(
                arena_id=self._arena.arena_id,
                storage_generation=self._arena.storage_generation,
                lifecycle=self._arena.lifecycle,
                epochs=_copy_epochs(self._arena.epochs),
                device=self._arena.device,
                resident_signature=_ResidentSignature(
                    tuple(self._arena.resident_signature.binding_object_ids),
                    tuple(self._arena.resident_signature.tensor_records),
                ),
                ragged=ragged,
                state=state,
                cache=cache,
            )

    def _open_read_lease(self, world):
        """Issue a private lease; entering and consuming each occur once."""
        with self._lock:
            self._guard_or_invalidate()
            if self._active_lease_token is not None:
                raise A4ResidentArenaError(
                    'cannot issue another lease while one is active'
                )
            if self.audit_cpu(world) != COHERENT:
                raise A4ResidentArenaError(
                    'resident read lease requires a coherent CPU shadow'
                )
            self._lease_serial += 1
            self._arena.lease_serial = self._lease_serial
            serial = self._lease_serial
            lease_token = object()
            self._issued_lease_token = lease_token
            self._active_lease_consumed = False
            self._refresh_guard()
            return _A4ResidentReadLease(
                _LEASE_FACTORY_TOKEN, self, world, self._arena, serial,
                lease_token,
            )

    def _require_lease_identity(self, lease, active=False):
        arena = self._arena
        expected_token = (
            self._active_lease_token if active else self._issued_lease_token
        )
        if (not isinstance(lease, _A4ResidentReadLease)
                or lease._factory_token is not _LEASE_FACTORY_TOKEN
                or lease._owner is not self
                or lease._owner_token is not self._owner_token
                or lease._arena_id != self._arena_seal.arena_id
                or _epoch_values(lease._epochs)
                != self._arena_seal.epochs_values
                or lease._cache_source_ragged_epoch
                != arena.epochs.cache_source_ragged_epoch
                or lease._device != arena.device
                or lease._signature != arena.resident_signature
                or lease._serial != self._lease_serial
                or lease._lease_token is not expected_token):
            raise A4ResidentArenaError(
                'stale, forged, or cross-owner resident read lease'
            )

    def _activate_lease(self, lease):
        with self._lock:
            self._guard_or_invalidate()
            self._require_lease_identity(lease)
            if self.audit_cpu(lease._world) != COHERENT:
                raise A4ResidentArenaError(
                    'CPU source changed before lease open'
                )
            if self._active_lease_token is not None:
                raise A4ResidentArenaError(
                    'A4.9a permits only one event-free read lease at a time'
                )
            _attest_resident(
                self._arena, self._arena_seal, readback=True,
            )
            self._active_lease_token = lease._lease_token
            self._active_lease_consumed = False
            self._arena.active_leases = 1
            self._refresh_guard()

    def _consume_lease(self, lease):
        with self._lock:
            self._guard_or_invalidate()
            self._require_lease_identity(lease, active=True)
            if (self._active_lease_token is None
                    or self._active_lease_consumed):
                raise A4ResidentArenaError(
                    'resident read lease is inactive or already consumed'
                )
            if self.audit_cpu(lease._world) != COHERENT:
                raise A4ResidentArenaError(
                    'CPU source changed before lease consumption'
                )
            _attest_resident(
                self._arena, self._arena_seal, readback=True,
            )
            self._active_lease_consumed = True
            self._refresh_guard()
            return self._arena.binding

    def _close_lease(self, lease):
        with self._lock:
            error = None
            try:
                try:
                    self._require_guard()
                except Exception as exc:
                    self._force_metadata_invalid(
                        'owner or immutable arena metadata changed in lease'
                    )
                    raise A4ResidentArenaError(
                        'owner or immutable arena metadata changed in lease'
                    ) from exc
                if (not isinstance(lease, _A4ResidentReadLease)
                        or lease._owner is not self
                        or lease._owner_token is not self._owner_token
                        or lease._lease_token is not self._active_lease_token):
                    raise A4ResidentArenaError(
                        'resident read lease close identity differs'
                    )
                if self.audit_cpu(lease._world) != COHERENT:
                    raise A4ResidentArenaError(
                        'CPU source changed before resident lease close'
                    )
                _attest_resident(
                    self._arena, self._arena_seal, readback=True,
                )
            except Exception as exc:
                error = exc
                if self._lifecycle == COHERENT:
                    self._transition_invalid(
                        'lease-close CPU or resident integrity failed',
                        ('R', 'S', 'C'), advance_unobserved=True,
                    )
            finally:
                self._active_lease_token = None
                self._active_lease_consumed = False
                self._issued_lease_token = None
                self._arena.active_leases = 0
                self._lease_serial += 1
                self._arena.lease_serial = self._lease_serial
                self._refresh_guard()
            if error is not None:
                raise A4ResidentArenaError(
                    'resident read lease close failed closed'
                ) from error

    def __getstate__(self):
        raise A4ResidentArenaScopeError(
            'arena owners are runtime shadows and are not serializable'
        )

    def __copy__(self):
        raise A4ResidentArenaScopeError(
            'arena owners are rebuilt from a fresh CPU world, not copied'
        )

    def __deepcopy__(self, memo):
        raise A4ResidentArenaScopeError(
            'arena owners are rebuilt from a fresh CPU world, not cloned'
        )
