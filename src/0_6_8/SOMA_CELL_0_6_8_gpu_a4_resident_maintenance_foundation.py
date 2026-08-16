# coding: utf-8
"""A4.9c immutable maintenance supplement and disposable pure plan.

The promoted CPU world remains the only durable biology.  This module adds
two immutable resident shadow tensors needed by the future maintenance event
and a pure selected-row oracle.  It deliberately does not claim, publish, or
commit the rank-4 maintenance event.
"""
from __future__ import division

import copy
import hashlib
import math
import os
import sys
import weakref
from dataclasses import dataclass, fields
from numbers import Real

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import SOMA_CELL_0_6_8_gpu_a2 as a2
import SOMA_CELL_0_6_8_gpu_a3 as a3
import SOMA_CELL_0_6_8_gpu_a4 as a4
import SOMA_CELL_0_6_8_gpu_a4_resident_arena as a49a
import SOMA_CELL_0_6_8_gpu_a4_resident_translation_integration as a49b

try:
    import torch
except Exception:  # pragma: no cover - NumPy descriptors remain importable
    torch = None


BUILD = 'SOMA-CELL 0.6.8-GPU A4.9c'
BUILD_ID = BUILD
BUILD_LONG = BUILD + ' | resident maintenance composite-shadow foundation'
SCHEMA_VERSION = '0.6.8-GPU-A4.9c-resident-maintenance-foundation'
SUPPLEMENT_SCHEMA_VERSION = SCHEMA_VERSION + '-supplement'
SELECTED_PLAN_SCHEMA_VERSION = SCHEMA_VERSION + '-selected-plan'
SAVE_VERSION = 1
FULL_GPU_WORLD_STEP = False
MAINTENANCE_ORACLE_ATOL = 2.0e-12

COHERENT = a49a.COHERENT
CPU_NEWER = a49a.CPU_NEWER
INVALID = a49a.INVALID

PORT_STATUS = dict(a49b.PORT_STATUS)
PORT_STATUS.update({
    'resident_device_writes': True,
    'resident_device_writes_scope': (
        'inherited-a4.9b-selected-translation-only'
    ),
    'resident_d2h_publish': True,
    'resident_d2h_publish_scope': (
        'inherited-a4.9b-selected-translation-only'
    ),
    'selected_maintenance': 'pure-plan-only',
    'maintenance_resident_device_writes': False,
    'maintenance_resident_d2h_publish': False,
    'maintenance_live_commit': False,
    'maintenance_resident_biology_authority': False,
    'full_gpu_world_step': False,
})

_SUPPLEMENT_FIELDS = ('transporters', 'maintenance_shortfall')
_SUPPLEMENT_TOKEN = object()
_RESIDENT_SUPPLEMENT_TOKEN = object()
_BINDING_TOKEN = object()
_SELECTED_PLAN_TOKEN = object()
_SUPPLEMENT_REGISTRY = {}
_BINDING_REGISTRY = {}
_PLAN_REGISTRY = {}


def _register_creation(registry, artifact, values):
    """Keep a detached seal without extending an artifact's lifetime."""
    key = id(artifact)

    def _discard(reference, registry=registry, key=key):
        current = registry.get(key)
        if current is not None and current[0] is reference:
            registry.pop(key, None)

    reference = weakref.ref(artifact, _discard)
    registry[key] = (reference, values)


def _creation_values(registry, artifact):
    seal = registry.get(id(artifact))
    if (not isinstance(seal, tuple) or len(seal) != 2
            or not isinstance(seal[0], weakref.ReferenceType)
            or seal[0]() is not artifact):
        return None
    return seal[1]


class A4ResidentMaintenanceFoundationError(a49b.A4ResidentTranslationCommitError):
    """The bounded A4.9c shadow or pure-plan contract failed closed."""


def _host_array(value):
    if a4._is_tensor(value):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def _array_digest(value):
    array = np.ascontiguousarray(_host_array(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode('ascii'))
    digest.update(repr(tuple(array.shape)).encode('ascii'))
    digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


def _arrays_bit_exact(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    if left.dtype == np.dtype(np.float64):
        return np.array_equal(left.view(np.uint64), right.view(np.uint64))
    return np.array_equal(left, right)


def _state_host(state):
    if a4._is_tensor(state.pools):
        return state.to_numpy()
    a4.validate_a4_translation_state(state)
    return state


def _state_provenance(state):
    return a4._translation_state_provenance(_state_host(state))


def _cell_layout_digest(state):
    host = _state_host(state)
    digest = hashlib.sha256()
    digest.update(_array_digest(host.cell_ids).encode('ascii'))
    digest.update(_array_digest(host.cell_mask).encode('ascii'))
    digest.update(str(int(host.cell_capacity)).encode('ascii'))
    digest.update(str(int(host.cell_count)).encode('ascii'))
    return digest.hexdigest()


def _supplement_provenance(
        base_state_provenance, layout_digest, presence,
        transporters, maintenance_shortfall, identity_token=''):
    digest = hashlib.sha256()
    digest.update(SUPPLEMENT_SCHEMA_VERSION.encode('utf-8'))
    digest.update(str(base_state_provenance).encode('ascii'))
    digest.update(str(layout_digest).encode('ascii'))
    digest.update(str(identity_token).encode('ascii'))
    digest.update(repr(tuple(bool(value) for value in presence)).encode('ascii'))
    digest.update(_array_digest(transporters).encode('ascii'))
    digest.update(_array_digest(maintenance_shortfall).encode('ascii'))
    return digest.hexdigest()


def _supplement_metadata(batch):
    return (
        str(batch.schema_version), int(batch.cell_capacity),
        int(batch.cell_count), str(batch.source_provenance),
    )


def _strict_cells(cells, state):
    if isinstance(cells, (str, bytes)):
        raise A4ResidentMaintenanceFoundationError('cells must be an ordered container')
    try:
        snapshot = tuple(cells)
    except Exception as exc:
        raise A4ResidentMaintenanceFoundationError(
            'cells must be a stable ordered container'
        ) from exc
    if len(snapshot) != int(state.cell_count):
        raise A4ResidentMaintenanceFoundationError(
            'cell count differs from translation state'
        )
    host = _state_host(state)
    for index, cell in enumerate(snapshot):
        raw_id = getattr(cell, 'cell_id', None)
        if (isinstance(raw_id, (bool, np.bool_))
                or not isinstance(raw_id, (int, np.integer))
                or int(raw_id) < 0
                or int(raw_id) != int(host.cell_ids[index])
                or not bool(host.cell_mask[index])):
            raise A4ResidentMaintenanceFoundationError(
                'ordered cell identity differs from translation state'
            )
    return snapshot


def _capture_cpu_supplement(cells, state):
    snapshot = _strict_cells(cells, state)
    C = int(state.cell_capacity)
    N = int(state.cell_count)
    transporters = np.zeros(
        (C, int(a3.MEMBRANE_SEGMENTS), int(a3.CHANNEL_COUNT)),
        dtype=np.float64,
    )
    shortfall = np.zeros((C,), dtype=np.float64)
    presence = []
    transporter_refs = []
    shortfall_refs = []
    source_values = []
    for index, cell in enumerate(snapshot):
        raw_transporters = getattr(cell, 'transporters', None)
        if (not isinstance(raw_transporters, np.ndarray)
                or raw_transporters.dtype != np.dtype(np.float64)
                or raw_transporters.shape != (
                    int(a3.MEMBRANE_SEGMENTS), int(a3.CHANNEL_COUNT))
                or not np.isfinite(raw_transporters).all()
                or np.any(raw_transporters < 0.0)):
            raise A4ResidentMaintenanceFoundationError(
                'CPU transporters violate the fixed float64 schema'
            )
        transporters[index] = raw_transporters
        has_shortfall = hasattr(cell, 'maintenance_shortfall')
        raw_shortfall = (
            getattr(cell, 'maintenance_shortfall') if has_shortfall else None
        )
        if has_shortfall and (
                isinstance(raw_shortfall, (bool, np.bool_))
                or not isinstance(raw_shortfall, Real)):
            raise A4ResidentMaintenanceFoundationError(
                'CPU maintenance_shortfall must be one real scalar'
            )
        value = 0.0 if not has_shortfall else float(raw_shortfall)
        if not math.isfinite(value) or value < 0.0:
            raise A4ResidentMaintenanceFoundationError(
                'CPU maintenance_shortfall must be finite and nonnegative'
            )
        shortfall[index] = value
        presence.append(bool(has_shortfall))
        transporter_refs.append(raw_transporters)
        shortfall_refs.append(raw_shortfall)
        source_values.append((
            int(cell.cell_id), _array_digest(raw_transporters),
            bool(has_shortfall), float(value).hex(),
        ))
    return {
        'cells_ref': cells,
        'cells': snapshot,
        'transporter_refs': tuple(transporter_refs),
        'shortfall_refs': tuple(shortfall_refs),
        'source_values': tuple(source_values),
        'presence': tuple(presence),
        'transporters': transporters,
        'maintenance_shortfall': shortfall,
        'state_anchor': a49a._field_anchor(
            snapshot, a49a._STATE_SOURCE_FIELDS, 'S'
        ),
        'ragged_anchor': a49a._field_anchor(
            snapshot, a49a._RAGGED_SOURCE_FIELDS, 'R'
        ),
    }


def _require_state_cpu_association(cells, state):
    """Require every one of the 30 packed state arrays to match CPU cells."""
    host = _state_host(state)
    snapshot = _strict_cells(cells, host)
    direct_arrays = (
        'pools', 'membrane', 'membrane_oxidation', 'damage_trace',
        'last_receptor_activity', 'last_control_vectors',
        'last_control_scalars',
    )
    direct_scalars = (
        'radius', 'current_stress', 'division_progress',
        'last_translation', 'last_quiescence',
        'cumulative_proofreading_atp', 'last_edna_signal',
        'last_corpse_signal', 'last_necrotoxin_signal',
        'behavioural_quiescence',
    )
    for index, cell in enumerate(snapshot):
        for name in direct_arrays:
            expected = np.asarray(getattr(cell, name))
            actual = np.asarray(getattr(host, name)[index])
            if not _arrays_bit_exact(expected, actual):
                raise A4ResidentMaintenanceFoundationError(
                    'CPU cell differs from translation state array: ' + name
                )
        for name in direct_scalars:
            raw = getattr(cell, name)
            if (isinstance(raw, (bool, np.bool_))
                    or not isinstance(raw, Real)
                    or not math.isfinite(float(raw))
                    or np.float64(raw).tobytes()
                    != np.float64(getattr(host, name)[index]).tobytes()):
                raise A4ResidentMaintenanceFoundationError(
                    'CPU cell differs from translation state scalar: ' + name
                )
        for prefix, mapping_name in (
                ('active', 'proteins'), ('damaged', 'damaged_proteins')):
            mapping = getattr(cell, mapping_name)
            if not isinstance(mapping, dict):
                raise A4ResidentMaintenanceFoundationError(
                    'CPU protein source must be an ordered dict'
                )
            items = tuple(mapping.items())
            count = int(getattr(host, prefix + '_count')[index])
            if count != len(items):
                raise A4ResidentMaintenanceFoundationError(
                    'CPU protein row count differs from translation state'
                )
            expected_keys = np.asarray(
                [int(key) for key, _ in items], dtype=np.int64,
            )
            expected_mass = np.asarray(
                [float(value) for _, value in items], dtype=np.float64,
            )
            if (not _arrays_bit_exact(
                    expected_keys,
                    np.asarray(getattr(host, prefix + '_fingerprints'))[
                        index, :count])
                    or not _arrays_bit_exact(
                        expected_mass,
                        np.asarray(getattr(host, prefix + '_mass'))[
                            index, :count])):
                raise A4ResidentMaintenanceFoundationError(
                    'CPU protein order/content differs from translation state'
                )
        attachments = getattr(cell, 'neural_attachments')
        if not isinstance(attachments, dict):
            raise A4ResidentMaintenanceFoundationError(
                'CPU neural attachments must be an ordered dict'
            )
        osmolyte = np.float64(0.0)
        for attachment in attachments.values():
            osmolyte = osmolyte + np.float64(attachment.osmolyte())
        if osmolyte.tobytes() != np.float64(
                host.neural_attachment_osmolyte[index]).tobytes():
            raise A4ResidentMaintenanceFoundationError(
                'CPU neural osmolyte differs from translation state'
            )
        genomes = tuple(getattr(cell, 'genomes'))
        lesions = np.asarray(getattr(cell, 'genome_lesions'), dtype=np.float64)
        active = getattr(cell, 'replication_template') is not None
        material = sum(len(genome) for genome in genomes)
        if active:
            material += len(getattr(cell, 'replication_copy'))
        lesion_mean = float(np.mean(lesions)) if lesions.size else 1.0
        if (int(host.genome_count[index]) != len(genomes)
                or bool(host.replication_active[index]) != bool(active)
                or int(host.genome_material_symbols[index]) != int(material)
                or np.float64(host.genome_lesion_mean[index]).tobytes()
                != np.float64(lesion_mean).tobytes()):
            raise A4ResidentMaintenanceFoundationError(
                'CPU genome summary differs from translation state'
            )
    return snapshot


def _require_cpu_anchor(batch):
    anchor = getattr(batch, '_cpu_anchor', None)
    if anchor is None:
        return batch
    (world_ref, config_ref, attestation_ref, cells_ref, cells,
     transporter_refs, shortfall_refs, state_anchor, ragged_anchor) = anchor
    try:
        fresh = a49a._stable_cpu_snapshot(
            world_ref, config_ref, require_cache_exact=True,
        )
    except Exception as exc:
        raise A4ResidentMaintenanceFoundationError(
            'packed CPU world cannot be reattested'
        ) from exc
    if not a49a._same_source(attestation_ref, fresh.attestation):
        raise A4ResidentMaintenanceFoundationError(
            'packed CPU world changed after supplement creation'
        )
    observed_cells = tuple(cells_ref)
    if (len(observed_cells) != len(cells)
            or any(observed is not expected
                   for observed, expected in zip(observed_cells, cells))):
        raise A4ResidentMaintenanceFoundationError(
            'CPU cell membership/identity changed after supplement pack'
        )
    current = _capture_cpu_supplement(cells_ref, getattr(batch, '_base_state_ref'))
    if (any(current['transporter_refs'][i] is not transporter_refs[i]
            for i in range(len(cells)))
            or any(current['shortfall_refs'][i] is not shortfall_refs[i]
                   for i in range(len(cells)))
            or current['source_values'] != getattr(batch, '_cpu_source_values')
            or current['presence'] != getattr(batch, '_presence')
            or not a49a._same_anchor(state_anchor, current['state_anchor'])
            or not a49a._same_anchor(ragged_anchor, current['ragged_anchor'])):
        raise A4ResidentMaintenanceFoundationError(
            'CPU supplement identity/presence/content changed'
        )
    return batch


@dataclass(frozen=True, init=False)
class A4MaintenanceSupplementBatch:
    """Exactly two fixed-capacity maintenance shadow arrays."""

    schema_version: str
    cell_capacity: int
    cell_count: int
    source_provenance: str
    transporters: object
    maintenance_shortfall: object

    def __init__(self, *args, **kwargs):
        raise A4ResidentMaintenanceFoundationError(
            'A4MaintenanceSupplementBatch is factory-only'
        )

    def validate(self):
        return _require_supplement(self, diagnostic=True)

    def to_numpy(self):
        raise A4ResidentMaintenanceFoundationError(
            'standalone supplement conversion is forbidden; convert its binding'
        )

    def to_torch(self, device='cpu'):
        raise A4ResidentMaintenanceFoundationError(
            'standalone supplement conversion is forbidden; convert its binding'
        )

    def data_ptrs(self):
        _require_supplement(self, diagnostic=False)
        if not a4._is_tensor(self.transporters):
            raise TypeError('data_ptrs requires a Torch supplement')
        return _supplement_data_ptrs_raw(self)

    def __getstate__(self):
        raise A4ResidentMaintenanceFoundationError(
            'maintenance supplements are not serializable authority'
        )


def _make_supplement(
        cell_capacity, cell_count, source_provenance, transporters,
        maintenance_shortfall, base_state_provenance, layout_digest,
        presence, base_state_ref=None, cpu_capture=None,
        base_binding_ref=None, bindable=False, origin_supplement=None,
        identity_token='', resident_authority=False):
    batch = object.__new__(A4MaintenanceSupplementBatch)
    object.__setattr__(batch, 'schema_version', SUPPLEMENT_SCHEMA_VERSION)
    object.__setattr__(batch, 'cell_capacity', int(cell_capacity))
    object.__setattr__(batch, 'cell_count', int(cell_count))
    object.__setattr__(batch, 'source_provenance', str(source_provenance))
    object.__setattr__(batch, 'transporters', transporters)
    object.__setattr__(batch, 'maintenance_shortfall', maintenance_shortfall)
    object.__setattr__(batch, '_factory_token', _SUPPLEMENT_TOKEN)
    object.__setattr__(batch, '_metadata', _supplement_metadata(batch))
    object.__setattr__(batch, '_base_state_provenance', str(base_state_provenance))
    object.__setattr__(batch, '_layout_digest', str(layout_digest))
    object.__setattr__(batch, '_identity_token', str(identity_token))
    canonical_presence = tuple(bool(x) for x in presence)
    object.__setattr__(batch, '_presence', canonical_presence)
    object.__setattr__(batch, '_presence_ref', canonical_presence)
    object.__setattr__(batch, '_presence_values', canonical_presence)
    object.__setattr__(batch, '_base_state_ref', base_state_ref)
    object.__setattr__(batch, '_base_binding_ref', base_binding_ref)
    object.__setattr__(batch, '_bindable', bool(bindable))
    object.__setattr__(batch, '_origin_supplement', origin_supplement)
    object.__setattr__(
        batch, '_resident_authority',
        _RESIDENT_SUPPLEMENT_TOKEN if resident_authority else None,
    )
    object.__setattr__(batch, '_object_ids', (
        id(transporters), id(maintenance_shortfall),
    ))
    object.__setattr__(batch, '_content_digests', (
        _array_digest(transporters), _array_digest(maintenance_shortfall),
    ))
    if a4._is_tensor(transporters):
        object.__setattr__(batch, '_data_ptrs', _supplement_data_ptrs_raw(batch))
        object.__setattr__(batch, '_versions', (
            int(transporters._version), int(maintenance_shortfall._version),
        ))
    else:
        object.__setattr__(batch, '_data_ptrs', None)
        object.__setattr__(batch, '_versions', None)
    if cpu_capture is None:
        object.__setattr__(batch, '_cpu_anchor', None)
        object.__setattr__(batch, '_cpu_source_values', None)
    else:
        object.__setattr__(batch, '_cpu_anchor', (
            cpu_capture['world_ref'], cpu_capture['config_ref'],
            cpu_capture['source_attestation'],
            cpu_capture['cells_ref'], cpu_capture['cells'],
            cpu_capture['transporter_refs'], cpu_capture['shortfall_refs'],
            cpu_capture['state_anchor'], cpu_capture['ragged_anchor'],
        ))
        object.__setattr__(
            batch, '_cpu_source_values', cpu_capture['source_values'],
        )
    _register_creation(
        _SUPPLEMENT_REGISTRY, batch, _supplement_creation_values(batch),
    )
    return _require_supplement(batch, diagnostic=True)


def _supplement_data_ptrs_raw(batch):
    return {
        'transporters': int(batch.transporters.data_ptr()),
        'maintenance_shortfall': int(batch.maintenance_shortfall.data_ptr()),
    }


def _supplement_numpy_copy(batch, base_state_ref=None):
    _require_supplement(batch, diagnostic=True)
    state_ref = (
        getattr(batch, '_base_state_ref', None)
        if base_state_ref is None else base_state_ref
    )
    return _make_supplement(
        int(batch.cell_capacity), int(batch.cell_count),
        batch.source_provenance, _host_array(batch.transporters),
        _host_array(batch.maintenance_shortfall),
        getattr(batch, '_base_state_provenance'),
        getattr(batch, '_layout_digest'), getattr(batch, '_presence'),
        base_state_ref=state_ref,
        identity_token=getattr(batch, '_identity_token'),
    )


def _supplement_creation_values(batch):
    data_ptrs = getattr(batch, '_data_ptrs', None)
    return (
        str(batch.schema_version), type(batch.cell_capacity),
        int(batch.cell_capacity), type(batch.cell_count), int(batch.cell_count),
        str(batch.source_provenance), id(batch.transporters),
        id(batch.maintenance_shortfall),
        tuple(getattr(batch, '_metadata', ())),
        str(getattr(batch, '_base_state_provenance', '')),
        str(getattr(batch, '_layout_digest', '')),
        str(getattr(batch, '_identity_token', '')),
        id(getattr(batch, '_presence', None)),
        tuple(getattr(batch, '_presence_values', ())),
        id(getattr(batch, '_base_state_ref', None)),
        id(getattr(batch, '_base_binding_ref', None)),
        bool(getattr(batch, '_bindable', False)),
        id(getattr(batch, '_origin_supplement', None)),
        id(getattr(batch, '_resident_authority', None)),
        tuple(getattr(batch, '_object_ids', ())),
        tuple(getattr(batch, '_content_digests', ())),
        None if data_ptrs is None else tuple(sorted(data_ptrs.items())),
        getattr(batch, '_versions', None),
        id(getattr(batch, '_cpu_anchor', None)),
        getattr(batch, '_cpu_source_values', None),
    )


def _require_supplement(batch, diagnostic=False):
    if (not isinstance(batch, A4MaintenanceSupplementBatch)
            or getattr(batch, '_factory_token', None) is not _SUPPLEMENT_TOKEN):
        raise A4ResidentMaintenanceFoundationError(
            'untrusted maintenance supplement'
        )
    if (_creation_values(_SUPPLEMENT_REGISTRY, batch)
            != _supplement_creation_values(batch)):
        raise A4ResidentMaintenanceFoundationError(
            'maintenance supplement creation seal changed'
        )
    if (type(batch.cell_capacity) is not int
            or type(batch.cell_count) is not int):
        raise A4ResidentMaintenanceFoundationError(
            'maintenance capacity/count must be exact integer scalars'
        )
    C = batch.cell_capacity
    N = batch.cell_count
    presence = getattr(batch, '_presence', None)
    if (batch.schema_version != SUPPLEMENT_SCHEMA_VERSION
            or C <= 0 or N < 0 or N > C
            or _supplement_metadata(batch) != getattr(batch, '_metadata', None)
            or not isinstance(batch.source_provenance, str)
            or len(batch.source_provenance) != 64
            or any(ch not in '0123456789abcdef'
                   for ch in batch.source_provenance)
            or not isinstance(presence, tuple)
            or presence is not getattr(batch, '_presence_ref', None)
            or presence != getattr(batch, '_presence_values', None)
            or any(type(value) is not bool for value in presence)
            or len(presence) != N
            or (id(batch.transporters), id(batch.maintenance_shortfall))
            != getattr(batch, '_object_ids', None)):
        raise A4ResidentMaintenanceFoundationError(
            'maintenance supplement metadata changed'
        )
    tensor_backed = a4._is_tensor(batch.transporters)
    if tensor_backed != a4._is_tensor(batch.maintenance_shortfall):
        raise A4ResidentMaintenanceFoundationError(
            'maintenance supplement mixes NumPy and Torch arrays'
        )
    expected_shapes = (
        (C, int(a3.MEMBRANE_SEGMENTS), int(a3.CHANNEL_COUNT)), (C,),
    )
    for value, shape in zip(
            (batch.transporters, batch.maintenance_shortfall), expected_shapes):
        expected_dtype = torch.float64 if tensor_backed else np.dtype(np.float64)
        if value.dtype != expected_dtype or tuple(value.shape) != shape:
            raise A4ResidentMaintenanceFoundationError(
                'maintenance supplement dtype/shape differs'
            )
    if tensor_backed:
        if (str(batch.transporters.device)
                != str(batch.maintenance_shortfall.device)
                or _supplement_data_ptrs_raw(batch)
                != getattr(batch, '_data_ptrs', None)
                or (int(batch.transporters._version),
                    int(batch.maintenance_shortfall._version))
                != getattr(batch, '_versions', None)
                or any(int(value._version) != 0 for value in (
                    batch.transporters, batch.maintenance_shortfall))):
            raise A4ResidentMaintenanceFoundationError(
                'maintenance resident pointer/version seal changed'
            )
        if not diagnostic:
            return batch
    host_transporters = _host_array(batch.transporters)
    host_shortfall = _host_array(batch.maintenance_shortfall)
    if (not np.isfinite(host_transporters).all()
            or not np.isfinite(host_shortfall).all()
            or np.any(host_transporters < 0.0)
            or np.any(host_shortfall < 0.0)
            or np.any(host_transporters[N:].view(np.uint64) != 0)
            or np.any(host_shortfall[N:].view(np.uint64) != 0)
            or (_array_digest(host_transporters), _array_digest(host_shortfall))
            != getattr(batch, '_content_digests', None)
            or _supplement_provenance(
                getattr(batch, '_base_state_provenance'),
                getattr(batch, '_layout_digest'),
                getattr(batch, '_presence'), host_transporters, host_shortfall,
                getattr(batch, '_identity_token'),
            ) != batch.source_provenance):
        raise A4ResidentMaintenanceFoundationError(
            'maintenance supplement content/provenance changed'
        )
    _require_cpu_anchor(batch)
    if bool(getattr(batch, '_bindable', False)):
        base_binding = getattr(batch, '_base_binding_ref', None)
        try:
            a4._require_translation_binding(base_binding)
        except Exception as exc:
            raise A4ResidentMaintenanceFoundationError(
                'bindable supplement lost its exact base binding'
            ) from exc
        if (base_binding.state is not getattr(batch, '_base_state_ref', None)
                or _state_provenance(base_binding.state)
                != getattr(batch, '_base_state_provenance', None)
                or _cell_layout_digest(base_binding.state)
                != getattr(batch, '_layout_digest', None)):
            raise A4ResidentMaintenanceFoundationError(
                'bindable supplement/base generation relation changed'
            )
        origin = getattr(batch, '_origin_supplement', None)
        if getattr(batch, '_cpu_anchor', None) is None:
            resident = (
                getattr(batch, '_resident_authority', None)
                is _RESIDENT_SUPPLEMENT_TOKEN
            )
            if (not resident and (
                    not isinstance(origin, A4MaintenanceSupplementBatch)
                    or origin is batch
                    or not bool(getattr(origin, '_bindable', False)))):
                raise A4ResidentMaintenanceFoundationError(
                    'joint conversion lacks its packed source supplement'
                )
            if not resident:
                _require_supplement(origin, diagnostic=True)
    elif getattr(batch, '_base_binding_ref', None) is not None:
        raise A4ResidentMaintenanceFoundationError(
            'diagnostic supplement carries unexpected binding authority'
        )
    return batch


def pack_a4_maintenance_supplement(world, config=None):
    """Create a supplement coupled to one private stable host base binding."""
    try:
        canonical_config = a4.GPU068A4Config.from_state(config)
        stable = a49a._stable_cpu_snapshot(
            world, config, require_cache_exact=True,
        )
        translation = a4.bind_a4_translation(stable.ragged, stable.state)
    except Exception as exc:
        raise A4ResidentMaintenanceFoundationError(
            'CPU world cannot produce a trusted maintenance base'
        ) from exc
    translation_state = translation.state
    before_state = _state_provenance(translation_state)
    capture = _capture_cpu_supplement(world.cells, translation_state)
    capture.update({
        'world_ref': world, 'config_ref': config,
        'source_attestation': stable.attestation,
    })
    replay_stable = a49a._stable_cpu_snapshot(
        world, config, require_cache_exact=True,
    )
    replay = _capture_cpu_supplement(world.cells, replay_stable.state)
    if (before_state != _state_provenance(replay_stable.state)
            or not a49a._same_source(
                stable.attestation, replay_stable.attestation)
            or capture['source_values'] != replay['source_values']
            or capture['presence'] != replay['presence']
            or any(capture['cells'][i] is not replay['cells'][i]
                   for i in range(len(capture['cells'])))):
        raise A4ResidentMaintenanceFoundationError(
            'CPU source changed during maintenance supplement pack'
        )
    layout = _cell_layout_digest(translation_state)
    config_identity = (
        ('implicit-default', tuple(a49a._config_state(canonical_config)))
        if config is None else (
            'explicit', id(config), type(config).__module__,
            type(config).__qualname__,
            tuple(a49a._config_state(canonical_config)),
        )
    )
    identity_values = (
        id(world), id(world.cells), id(world.config),
        tuple(id(cell) for cell in tuple(world.cells)),
        id(translation), id(translation.state), config_identity,
    )
    identity_token = hashlib.sha256(
        repr(identity_values).encode('ascii')
    ).hexdigest()
    provenance = _supplement_provenance(
        before_state, layout, capture['presence'],
        capture['transporters'], capture['maintenance_shortfall'],
        identity_token,
    )
    return _make_supplement(
        int(translation_state.cell_capacity),
        int(translation_state.cell_count), provenance,
        capture['transporters'], capture['maintenance_shortfall'],
        before_state, layout, capture['presence'],
        base_state_ref=translation_state, cpu_capture=capture,
        base_binding_ref=translation, bindable=True,
        identity_token=identity_token,
    )


@dataclass(frozen=True, init=False)
class A4MaintenanceBinding:
    """One exact translation binding plus its two-array supplement."""

    translation: a4.A4TranslationBinding
    supplement: A4MaintenanceSupplementBatch

    def __init__(self, *args, **kwargs):
        raise A4ResidentMaintenanceFoundationError(
            'A4MaintenanceBinding is created only by bind_a4_maintenance'
        )

    def validate(self):
        return _require_maintenance_binding(self, diagnostic=True)

    def to_numpy(self):
        _require_maintenance_binding(self, diagnostic=True)
        translation = self.translation
        ragged = (
            translation.ragged.to_numpy()
            if a4._is_tensor(translation.ragged.symbols)
            else translation.ragged.clone()
        )
        state = (
            translation.state.to_numpy()
            if a4._is_tensor(translation.state.pools)
            else translation.state.clone()
        )
        translated = a4.bind_a4_translation(ragged, state)
        supplement = _joint_supplement(
            self.supplement, translated,
            _host_array(self.supplement.transporters),
            _host_array(self.supplement.maintenance_shortfall),
        )
        return _make_maintenance_binding(
            translated, supplement, conversion_source=self,
        )

    def to_torch(self, device='cpu'):
        if torch is None:
            raise RuntimeError('PyTorch is unavailable')
        _require_maintenance_binding(self, diagnostic=True)
        host = self.to_numpy()
        ragged = host.translation.ragged.to_torch(device)
        state = host.translation.state.to_torch(device)
        translated = a4.bind_a4_translation(ragged, state)
        transporters = torch.as_tensor(
            np.ascontiguousarray(_host_array(host.supplement.transporters)),
            dtype=torch.float64, device=device,
        ).clone()
        shortfall = torch.as_tensor(
            np.ascontiguousarray(
                _host_array(host.supplement.maintenance_shortfall)
            ), dtype=torch.float64, device=device,
        ).clone()
        supplement = _joint_supplement(
            self.supplement, translated, transporters, shortfall,
        )
        return _make_maintenance_binding(
            translated, supplement, conversion_source=self,
        )

    def data_ptrs(self):
        _require_maintenance_binding(self, diagnostic=False)
        if not a4._is_tensor(self.translation.state.pools):
            raise TypeError('data_ptrs requires a Torch maintenance binding')
        return _binding_data_ptrs_raw(self)

    def __getstate__(self):
        raise A4ResidentMaintenanceFoundationError(
            'maintenance bindings are not serializable authority'
        )


def _binding_metadata(binding):
    return (
        id(binding.translation), id(binding.supplement),
        int(binding.translation.state.cell_capacity),
        int(binding.translation.state.cell_count),
        str(binding.supplement.source_provenance),
    )


def _make_maintenance_binding(
        translation, supplement, conversion_source=None):
    binding = object.__new__(A4MaintenanceBinding)
    object.__setattr__(binding, 'translation', translation)
    object.__setattr__(binding, 'supplement', supplement)
    object.__setattr__(binding, '_factory_token', _BINDING_TOKEN)
    object.__setattr__(binding, '_metadata', _binding_metadata(binding))
    object.__setattr__(binding, '_source_provenance', hashlib.sha256(
        (str(translation.state.source_provenance)
         + str(supplement.source_provenance)).encode('ascii')
    ).hexdigest())
    object.__setattr__(
        binding, '_conversion_source_id', id(conversion_source),
    )
    object.__setattr__(binding, '_conversion_source_storage', (
        tuple(_array_storage_descriptor(value)
              for value in _binding_arrays(conversion_source))
        if conversion_source is not None else tuple()
    ))
    object.__setattr__(binding, '_array_object_ids', tuple(
        id(value) for value in _binding_arrays(binding)
    ))
    object.__setattr__(binding, '_array_content_digests', tuple(
        _array_digest(value) for value in _binding_arrays(binding)
    ))
    if a4._is_tensor(translation.state.pools):
        object.__setattr__(binding, '_data_ptrs', _binding_data_ptrs_raw(binding))
    else:
        object.__setattr__(binding, '_data_ptrs', None)
    _register_creation(
        _BINDING_REGISTRY, binding, _binding_creation_values(binding),
    )
    return _require_maintenance_binding(binding, diagnostic=True)


def _binding_creation_values(binding):
    pointers = getattr(binding, '_data_ptrs', None)
    return (
        id(binding.translation), id(binding.supplement),
        tuple(getattr(binding, '_metadata', ())),
        str(getattr(binding, '_source_provenance', '')),
        int(getattr(binding, '_conversion_source_id', 0)),
        tuple(getattr(binding, '_conversion_source_storage', ())),
        tuple(getattr(binding, '_array_object_ids', ())),
        tuple(getattr(binding, '_array_content_digests', ())),
        None if pointers is None else tuple(sorted(pointers.items())),
    )


def _packed_origin(supplement):
    if (getattr(supplement, '_cpu_anchor', None) is not None
            or getattr(supplement, '_resident_authority', None)
            is _RESIDENT_SUPPLEMENT_TOKEN):
        return supplement
    return getattr(supplement, '_origin_supplement', None)


def _joint_supplement(
        source, translation, transporters, maintenance_shortfall):
    origin = _packed_origin(source)
    if not isinstance(origin, A4MaintenanceSupplementBatch):
        raise A4ResidentMaintenanceFoundationError(
            'joint conversion lacks a packed supplement origin'
        )
    base_provenance = _state_provenance(translation.state)
    layout = _cell_layout_digest(translation.state)
    presence = getattr(source, '_presence')
    provenance = _supplement_provenance(
        base_provenance, layout, presence,
        transporters, maintenance_shortfall,
        getattr(source, '_identity_token'),
    )
    return _make_supplement(
        int(source.cell_capacity), int(source.cell_count), provenance,
        transporters, maintenance_shortfall, base_provenance, layout,
        presence, base_state_ref=translation.state,
        base_binding_ref=translation, bindable=True,
        origin_supplement=origin,
        identity_token=getattr(source, '_identity_token'),
    )


def _binding_data_ptrs_raw(binding):
    output = {
        'ragged.' + key: value
        for key, value in binding.translation.ragged.data_ptrs().items()
    }
    output.update({
        'state.' + key: value
        for key, value in binding.translation.state.data_ptrs().items()
    })
    output.update({
        'cache.' + key: value
        for key, value in binding.translation.cache.data_ptrs().items()
    })
    output.update({
        'supplement.' + key: value
        for key, value in binding.supplement.data_ptrs().items()
    })
    return output


def _require_maintenance_binding(binding, diagnostic=False):
    if (not isinstance(binding, A4MaintenanceBinding)
            or getattr(binding, '_factory_token', None) is not _BINDING_TOKEN):
        raise A4ResidentMaintenanceFoundationError(
            'untrusted maintenance binding'
        )
    if (_creation_values(_BINDING_REGISTRY, binding)
            != _binding_creation_values(binding)):
        raise A4ResidentMaintenanceFoundationError(
            'maintenance binding creation seal changed'
        )
    try:
        a4._require_translation_binding(binding.translation)
    except Exception as exc:
        raise A4ResidentMaintenanceFoundationError(
            'translation source binding changed'
        ) from exc
    _require_supplement(binding.supplement, diagnostic=diagnostic)
    state = binding.translation.state
    supplement = binding.supplement
    if (_binding_metadata(binding) != getattr(binding, '_metadata', None)
            or not bool(getattr(supplement, '_bindable', False))
            or getattr(supplement, '_base_binding_ref', None)
            is not binding.translation
            or int(state.cell_capacity) != int(supplement.cell_capacity)
            or int(state.cell_count) != int(supplement.cell_count)
            or getattr(supplement, '_base_state_ref', None) is not state
            or _state_provenance(state)
            != getattr(supplement, '_base_state_provenance')
            or _cell_layout_digest(state) != getattr(supplement, '_layout_digest')
            or (a4._is_tensor(state.pools)
                != a4._is_tensor(supplement.transporters))):
        raise A4ResidentMaintenanceFoundationError(
            'base/supplement source association differs'
        )
    _require_binding_fresh_arrays(binding)
    if tuple(_array_digest(value) for value in _binding_arrays(binding)) != tuple(
            getattr(binding, '_array_content_digests', ())):
        raise A4ResidentMaintenanceFoundationError(
            'maintenance binding content seal changed'
        )
    if a4._is_tensor(state.pools):
        if (str(state.pools.device) != str(supplement.transporters.device)
                or _binding_data_ptrs_raw(binding)
                != getattr(binding, '_data_ptrs', None)):
            raise A4ResidentMaintenanceFoundationError(
                'composite resident device/pointer association changed'
            )
        pointers = tuple(_binding_data_ptrs_raw(binding).values())
        if len(pointers) != 52 or len(set(pointers)) != 52:
            raise A4ResidentMaintenanceFoundationError(
                'composite source must contain 52 nonaliasing tensors'
            )
    return binding


def bind_a4_maintenance(supplement):
    """Expose only the exact private base stored by the coupled pack."""
    _require_supplement(supplement, diagnostic=True)
    if (getattr(supplement, '_cpu_anchor', None) is None
            or getattr(supplement, '_origin_supplement', None) is not None
            or not bool(getattr(supplement, '_bindable', False))):
        raise A4ResidentMaintenanceFoundationError(
            'only an original coupled host supplement can be bound publicly'
        )
    translation = getattr(supplement, '_base_binding_ref', None)
    try:
        a4._require_translation_binding(translation)
    except Exception as exc:
        raise A4ResidentMaintenanceFoundationError(
            'packed supplement lost its private base binding'
        ) from exc
    return _make_maintenance_binding(translation, supplement)


def _strict_target_index(value, state):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))):
        raise A4ResidentMaintenanceFoundationError(
            'selected maintenance target_index must be an integer'
        )
    index = int(value)
    if index < 0 or index >= int(state.cell_count):
        raise A4ResidentMaintenanceFoundationError(
            'selected maintenance target_index is not one used row'
        )
    return index


def _strict_target_cell_id(value):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))):
        raise A4ResidentMaintenanceFoundationError(
            'selected maintenance target_cell_id must be an integer'
        )
    cell_id = int(value)
    if cell_id < 0:
        raise A4ResidentMaintenanceFoundationError(
            'selected maintenance target_cell_id must be nonnegative'
        )
    return cell_id


def _strict_dt(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise A4ResidentMaintenanceFoundationError(
            'selected maintenance dt must be one real scalar'
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise A4ResidentMaintenanceFoundationError(
            'selected maintenance dt must be finite and nonnegative'
        )
    return result


def _selected_preflight(binding, target_index, target_cell_id):
    _require_maintenance_binding(binding, diagnostic=True)
    state = binding.translation.state
    index = _strict_target_index(target_index, state)
    cell_id = _strict_target_cell_id(target_cell_id)
    if a4._is_tensor(state.cell_ids):
        observed = int(state.cell_ids[index].detach().cpu().item())
        used = bool(state.cell_mask[index].detach().cpu().item())
    else:
        observed = int(state.cell_ids[index])
        used = bool(state.cell_mask[index])
    if observed != cell_id or not used:
        raise A4ResidentMaintenanceFoundationError(
            'selected maintenance index/cell ID differs from binding'
        )
    return index, cell_id


def _complete_symbols_numpy(ragged, index):
    host = ragged.to_numpy() if a4._is_tensor(ragged.symbols) else ragged
    first = int(host.cell_sequence_offsets[index])
    count = int(host.genome_counts[index])
    total = np.int64(0)
    for position in range(count):
        sequence = first + position
        total = total + np.int64(
            host.sequence_offsets[sequence + 1]
            - host.sequence_offsets[sequence]
        )
    return total


def _complete_symbols_torch(ragged, index):
    if torch is None or not a4._is_tensor(ragged.symbols):
        raise A4ResidentMaintenanceFoundationError(
            'resident complete-symbol derivation requires Torch ragged state'
        )
    device = ragged.sequence_offsets.device
    positions = torch.arange(
        int(ragged.sequence_capacity), dtype=torch.int64, device=device,
    )
    lengths = ragged.sequence_offsets[1:] - ragged.sequence_offsets[:-1]
    first = ragged.cell_sequence_offsets[int(index)]
    count = ragged.genome_counts[int(index)]
    selected = (positions >= first) & (positions < first + count)
    selected = selected & (positions < int(ragged.sequence_count))
    return torch.where(
        selected, lengths, torch.zeros_like(lengths),
    ).sum(dtype=torch.int64)


def _fixed_sum_numpy(values):
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    total = np.float64(0.0)
    for index in range(int(flat.shape[0])):
        total = total + flat[index]
    return total


def _fixed_sum_torch(values):
    flat = values.reshape(-1)
    total = flat.new_zeros(())
    for index in range(int(flat.shape[0])):
        total = total + flat[index]
    return total


def _maintenance_numpy_values(binding, dt, index):
    state = binding.translation.state
    supplement = binding.supplement
    pools = state.pools[index]
    membrane = state.membrane[index]
    material_symbols = int(state.genome_material_symbols[index])
    complete_symbols = int(_complete_symbols_numpy(
        binding.translation.ragged, index,
    ))
    tension = np.float64(a2._tension_numpy_exact(
        pools, membrane, material_symbols * float(a4.g2.MONOMER_MASS),
    ))
    maintenance = float(dt) * (
        0.0042 + 0.010 * pools[int(a3.POOL_CATALYST)]
        + 0.009 * _fixed_sum_numpy(supplement.transporters[index])
        + 0.0025 * _fixed_sum_numpy(membrane)
        + 0.008 * tension
        + 0.000030 * complete_symbols
    )
    paid = np.minimum(
        np.float64(pools[int(a3.POOL_ATP)]), np.float64(maintenance),
    )
    atp_after = np.float64(pools[int(a3.POOL_ATP)]) - paid
    shortfall_after = np.maximum(
        np.float64(0.0), np.float64(maintenance) - paid,
    )
    return atp_after, shortfall_after, complete_symbols


def _maintenance_torch_values(binding, dt, index):
    state = binding.translation.state
    supplement = binding.supplement
    pools = state.pools[index]
    membrane = state.membrane[index]
    complete_symbols = _complete_symbols_torch(
        binding.translation.ragged, index,
    )
    genome_mass = state.genome_material_symbols[index].to(
        dtype=pools.dtype,
    ) * float(a4.g2.MONOMER_MASS)
    tension = a2._tension_torch_exact(pools, membrane, genome_mass)
    maintenance = float(dt) * (
        0.0042 + 0.010 * pools[int(a3.POOL_CATALYST)]
        + 0.009 * _fixed_sum_torch(supplement.transporters[index])
        + 0.0025 * _fixed_sum_torch(membrane)
        + 0.008 * tension
        + 0.000030 * complete_symbols.to(dtype=pools.dtype)
    )
    paid = torch.minimum(pools[int(a3.POOL_ATP)], maintenance)
    atp_after = pools[int(a3.POOL_ATP)] - paid
    shortfall_after = torch.maximum(
        pools.new_zeros(()), maintenance - paid,
    )
    return atp_after, shortfall_after, complete_symbols


def _seal_torch_state(state):
    a4._validate_translation_resident_metadata(state)
    object.__setattr__(
        state, '_a4_translation_metadata',
        a4._translation_scalar_metadata(state),
    )
    object.__setattr__(
        state, '_a4_translation_data_ptrs', state.data_ptrs(),
    )
    object.__setattr__(
        state, '_a4_translation_versions', {
            name: int(getattr(state, name)._version)
            for name in a4._TRANSLATION_ARRAY_FIELDS
        },
    )
    return state


def _fresh_maintenance_outputs(binding, index, atp_after, shortfall_after):
    source = binding.translation.state
    tensor_backed = a4._is_tensor(source.pools)
    C = int(source.cell_capacity)
    values = {}
    for item in fields(a4.A4TranslationStateBatch):
        name = item.name
        value = getattr(source, name)
        if name not in a4._TRANSLATION_ARRAY_FIELDS:
            values[name] = copy.deepcopy(value)
        elif tensor_backed:
            if name == 'pools':
                row = torch.arange(
                    C, dtype=torch.int64, device=value.device,
                ).reshape(C, 1) == int(index)
                column = torch.arange(
                    int(a3.POOL_COUNT), dtype=torch.int64,
                    device=value.device,
                ).reshape(1, int(a3.POOL_COUNT)) == int(a3.POOL_ATP)
                output = torch.where(row & column, atp_after, value)
            else:
                output = value.clone()
            if int(output._version) != 0:
                raise A4ResidentMaintenanceFoundationError(
                    'fresh selected state tensor version is nonzero'
                )
            values[name] = output
        else:
            output = np.asarray(value).copy()
            if name == 'pools':
                output[int(index), int(a3.POOL_ATP)] = np.float64(atp_after)
            values[name] = output
    state_after = a4.A4TranslationStateBatch(**values)
    if tensor_backed:
        _seal_torch_state(state_after)
        target_mask = torch.arange(
            C, dtype=torch.int64, device=source.pools.device,
        ) == int(index)
        transporters = binding.supplement.transporters.clone()
        row_mask = target_mask
        shortfall = torch.where(
            row_mask, shortfall_after,
            binding.supplement.maintenance_shortfall,
        )
        if any(int(value._version) != 0 for value in (
                target_mask, transporters, shortfall)):
            raise A4ResidentMaintenanceFoundationError(
                'fresh selected supplement tensor version is nonzero'
            )
    else:
        a4.validate_a4_translation_state(state_after)
        target_mask = np.arange(C, dtype=np.int64) == int(index)
        transporters = np.asarray(binding.supplement.transporters).copy()
        shortfall = np.asarray(
            binding.supplement.maintenance_shortfall,
        ).copy()
        shortfall[int(index)] = np.float64(shortfall_after)
    base_provenance = _state_provenance(state_after)
    layout = _cell_layout_digest(state_after)
    presence = getattr(binding.supplement, '_presence')
    supplement_provenance = _supplement_provenance(
        base_provenance, layout, presence, transporters, shortfall,
    )
    supplement_after = _make_supplement(
        C, int(source.cell_count), supplement_provenance,
        transporters, shortfall, base_provenance, layout, presence,
        base_state_ref=state_after,
    )
    return state_after, supplement_after, target_mask


def _plan_metadata(plan):
    return (
        str(plan.schema_version), int(plan.target_index),
        int(plan.target_cell_id), str(plan.dt_hex),
        str(plan.source_provenance), id(plan.target_mask),
        id(plan.state_after), id(plan.supplement_after),
    )


@dataclass(frozen=True, init=False)
class A4SelectedMaintenancePlan:
    """Disposable 33-array result for one selected maintenance row."""

    schema_version: str
    target_index: int
    target_cell_id: int
    dt_hex: str
    target_mask: object
    source_provenance: str
    state_after: a4.A4TranslationStateBatch
    supplement_after: A4MaintenanceSupplementBatch

    def __init__(self, *args, **kwargs):
        raise A4ResidentMaintenanceFoundationError(
            'A4SelectedMaintenancePlan is created only by pure factories'
        )

    def validate(self):
        return _require_selected_plan(self, diagnostic=True)

    def to_numpy(self):
        _require_selected_plan(self, diagnostic=True)
        state = (
            self.state_after.to_numpy()
            if a4._is_tensor(self.state_after.pools)
            else self.state_after.clone()
        )
        supplement = _supplement_numpy_copy(
            self.supplement_after, base_state_ref=state,
        )
        mask = _host_array(self.target_mask).astype(bool, copy=False)
        return _make_selected_plan(
            int(self.target_index), int(self.target_cell_id),
            float.fromhex(self.dt_hex), mask, state, supplement,
            getattr(self, '_source_binding'), diagnostic=True,
        )

    def data_ptrs(self):
        _require_selected_plan(self, diagnostic=False)
        if not a4._is_tensor(self.target_mask):
            raise TypeError('data_ptrs requires a Torch selected plan')
        output = {'target_mask': int(self.target_mask.data_ptr())}
        output.update({
            'state_after.' + key: value
            for key, value in self.state_after.data_ptrs().items()
        })
        output.update({
            'supplement_after.' + key: value
            for key, value in self.supplement_after.data_ptrs().items()
        })
        return output

    def __getstate__(self):
        raise A4ResidentMaintenanceFoundationError(
            'selected maintenance plans are not serializable authority'
        )


def _plan_data_ptrs_raw(plan):
    output = {'target_mask': int(plan.target_mask.data_ptr())}
    output.update({
        'state_after.' + key: value
        for key, value in plan.state_after.data_ptrs().items()
    })
    output.update({
        'supplement_after.' + key: value
        for key, value in _supplement_data_ptrs_raw(
            plan.supplement_after,
        ).items()
    })
    return output


def _plan_arrays(plan):
    return ((plan.target_mask,)
            + tuple(getattr(plan.state_after, name)
                    for name in a4._TRANSLATION_ARRAY_FIELDS)
            + (plan.supplement_after.transporters,
               plan.supplement_after.maintenance_shortfall))


def _binding_arrays(binding):
    translation = binding.translation
    return (tuple(getattr(translation.ragged, name) for name in a4._ARRAY_FIELDS)
            + tuple(getattr(translation.state, name)
                    for name in a4._TRANSLATION_ARRAY_FIELDS)
            + tuple(getattr(translation.cache, name)
                    for name in a4._GENE_ARRAY_FIELDS)
            + (binding.supplement.transporters,
               binding.supplement.maintenance_shortfall))


def _array_storage_descriptor(value):
    if a4._is_tensor(value):
        return (
            'torch', str(value.device), int(value.data_ptr()),
            int(value.numel()) * int(value.element_size()),
        )
    array = np.asarray(value)
    return (
        'numpy', '', int(array.__array_interface__['data'][0]),
        int(array.nbytes),
    )


def _storage_overlaps(left, right):
    if left[0] != right[0] or left[1] != right[1]:
        return False
    left_start, left_size = int(left[2]), int(left[3])
    right_start, right_size = int(right[2]), int(right[3])
    if left_size == 0 or right_size == 0:
        return left_start != 0 and left_start == right_start
    return max(left_start, right_start) < min(
        left_start + left_size, right_start + right_size,
    )


def _require_binding_fresh_arrays(binding):
    arrays = _binding_arrays(binding)
    if (len(arrays) != 52
            or tuple(id(value) for value in arrays)
            != getattr(binding, '_array_object_ids', None)
            or len({id(value) for value in arrays}) != 52):
        raise A4ResidentMaintenanceFoundationError(
            'maintenance binding does not contain 52 exact fresh objects'
        )
    tensor_backed = a4._is_tensor(arrays[0])
    if tensor_backed:
        storage = {
            (str(value.device), int(value.data_ptr())) for value in arrays
        }
        if len(storage) != 52:
            raise A4ResidentMaintenanceFoundationError(
                'maintenance binding tensors alias storage'
            )
    else:
        for index, left in enumerate(arrays):
            for right in arrays[index + 1:]:
                if np.shares_memory(np.asarray(left), np.asarray(right)):
                    raise A4ResidentMaintenanceFoundationError(
                        'maintenance binding NumPy arrays alias storage'
                    )
    source_storage = tuple(
        getattr(binding, '_conversion_source_storage', ()),
    )
    if source_storage:
        current_storage = tuple(
            _array_storage_descriptor(value) for value in arrays
        )
        if any(_storage_overlaps(left, right)
               for left in current_storage for right in source_storage):
            raise A4ResidentMaintenanceFoundationError(
                'joint conversion aliases its source generation'
            )
    return arrays


def _require_plan_fresh_arrays(plan):
    outputs = _plan_arrays(plan)
    source = _binding_arrays(getattr(plan, '_source_binding'))
    if len(outputs) != 33 or len({id(value) for value in outputs}) != 33:
        raise A4ResidentMaintenanceFoundationError(
            'selected maintenance outputs are not 33 fresh objects'
        )
    if a4._is_tensor(outputs[0]):
        output_storage = {
            (str(value.device), int(value.data_ptr())) for value in outputs
        }
        source_storage = {
            (str(value.device), int(value.data_ptr())) for value in source
        }
        if len(output_storage) != 33 or output_storage.intersection(source_storage):
            raise A4ResidentMaintenanceFoundationError(
                'selected maintenance Torch output aliases source storage'
            )
    else:
        for left_index, left in enumerate(outputs):
            for right in outputs[left_index + 1:]:
                if np.shares_memory(np.asarray(left), np.asarray(right)):
                    raise A4ResidentMaintenanceFoundationError(
                        'selected maintenance NumPy outputs alias each other'
                    )
            for right in source:
                # A NumPy diagnostic copy cannot share storage with a Torch
                # source tensor.  In particular, do not ask NumPy to coerce a
                # CUDA tensor merely to prove this cross-backend fact.
                if a4._is_tensor(right):
                    continue
                if np.shares_memory(np.asarray(left), np.asarray(right)):
                    raise A4ResidentMaintenanceFoundationError(
                        'selected maintenance NumPy output aliases source'
                    )
    return outputs


def _plan_creation_values(plan):
    return (
        tuple(_plan_metadata(plan)), id(getattr(plan, '_source_binding', None)),
        tuple(id(value) for value in _plan_arrays(plan)),
        str(getattr(plan, '_mask_digest', '')),
        str(getattr(plan, '_state_digest', '')),
        str(getattr(plan, '_supplement_digest', '')),
        None if getattr(plan, '_data_ptrs', None) is None else tuple(
            sorted(getattr(plan, '_data_ptrs').items())
        ),
        getattr(plan, '_versions', None),
    )


def _make_selected_plan(
        target_index, target_cell_id, dt, target_mask, state_after,
        supplement_after, source_binding, diagnostic=False):
    plan = object.__new__(A4SelectedMaintenancePlan)
    object.__setattr__(plan, 'schema_version', SELECTED_PLAN_SCHEMA_VERSION)
    object.__setattr__(plan, 'target_index', int(target_index))
    object.__setattr__(plan, 'target_cell_id', int(target_cell_id))
    object.__setattr__(plan, 'dt_hex', float(dt).hex())
    object.__setattr__(plan, 'target_mask', target_mask)
    object.__setattr__(
        plan, 'source_provenance',
        str(getattr(source_binding, '_source_provenance')),
    )
    object.__setattr__(plan, 'state_after', state_after)
    object.__setattr__(plan, 'supplement_after', supplement_after)
    object.__setattr__(plan, '_factory_token', _SELECTED_PLAN_TOKEN)
    object.__setattr__(plan, '_source_binding', source_binding)
    object.__setattr__(plan, '_metadata', _plan_metadata(plan))
    object.__setattr__(plan, '_mask_digest', _array_digest(target_mask))
    object.__setattr__(
        plan, '_state_digest', _state_provenance(state_after),
    )
    object.__setattr__(
        plan, '_supplement_digest', supplement_after.source_provenance,
    )
    if a4._is_tensor(target_mask):
        object.__setattr__(plan, '_data_ptrs', _plan_data_ptrs_raw(plan))
        object.__setattr__(plan, '_versions', (
            int(target_mask._version),
            tuple(int(getattr(state_after, name)._version)
                  for name in a4._TRANSLATION_ARRAY_FIELDS),
            int(supplement_after.transporters._version),
            int(supplement_after.maintenance_shortfall._version),
        ))
    else:
        object.__setattr__(plan, '_data_ptrs', None)
        object.__setattr__(plan, '_versions', None)
    _require_plan_fresh_arrays(plan)
    _register_creation(_PLAN_REGISTRY, plan, _plan_creation_values(plan))
    return _require_selected_plan(plan, diagnostic=diagnostic)


def _require_selected_plan(plan, diagnostic=False):
    if (not isinstance(plan, A4SelectedMaintenancePlan)
            or getattr(plan, '_factory_token', None) is not _SELECTED_PLAN_TOKEN):
        raise A4ResidentMaintenanceFoundationError(
            'untrusted selected maintenance plan'
        )
    if (_creation_values(_PLAN_REGISTRY, plan)
            != _plan_creation_values(plan)):
        raise A4ResidentMaintenanceFoundationError(
            'selected maintenance plan creation seal changed'
        )
    state = plan.state_after
    index = _strict_target_index(plan.target_index, state)
    cell_id = _strict_target_cell_id(plan.target_cell_id)
    try:
        dt = float.fromhex(str(plan.dt_hex))
    except Exception as exc:
        raise A4ResidentMaintenanceFoundationError(
            'selected maintenance dt_hex is invalid'
        ) from exc
    source = getattr(plan, '_source_binding', None)
    if (not math.isfinite(dt) or dt < 0.0 or dt.hex() != plan.dt_hex
            or plan.schema_version != SELECTED_PLAN_SCHEMA_VERSION
            or _plan_metadata(plan) != getattr(plan, '_metadata', None)
            or not isinstance(source, A4MaintenanceBinding)
            or plan.source_provenance
            != getattr(source, '_source_provenance', None)):
        raise A4ResidentMaintenanceFoundationError(
            'selected maintenance plan metadata changed'
        )
    _require_maintenance_binding(source, diagnostic=diagnostic)
    _require_plan_fresh_arrays(plan)
    _require_supplement(plan.supplement_after, diagnostic=diagnostic)
    if (getattr(plan.supplement_after, '_base_state_ref', None) is not state
            or _state_provenance(state)
            != getattr(plan.supplement_after, '_base_state_provenance', None)):
        raise A4ResidentMaintenanceFoundationError(
            'selected state/supplement result association changed'
        )
    tensor_backed = a4._is_tensor(state.pools)
    if tensor_backed != a4._is_tensor(plan.target_mask):
        raise A4ResidentMaintenanceFoundationError(
            'selected maintenance plan mixes NumPy and Torch arrays'
        )
    if tensor_backed:
        pointers = _plan_data_ptrs_raw(plan)
        versions = (
            int(plan.target_mask._version),
            tuple(int(getattr(state, name)._version)
                  for name in a4._TRANSLATION_ARRAY_FIELDS),
            int(plan.supplement_after.transporters._version),
            int(plan.supplement_after.maintenance_shortfall._version),
        )
        if (pointers != getattr(plan, '_data_ptrs', None)
                or versions != getattr(plan, '_versions', None)
                or any(value != 0 for value in (
                    versions[0], versions[2], versions[3], *versions[1]))
                or len(pointers) != 33 or len(set(pointers.values())) != 33
                or set(pointers.values()).intersection(
                    _binding_data_ptrs_raw(source).values())):
            raise A4ResidentMaintenanceFoundationError(
                'selected maintenance pointer/version/freshness seal changed'
            )
        if not diagnostic:
            return plan
    host_mask = _host_array(plan.target_mask)
    host_state = _state_host(state)
    host_supplement = _supplement_numpy_copy(plan.supplement_after)
    expected_mask = np.zeros((int(state.cell_capacity),), dtype=bool)
    expected_mask[index] = True
    if (host_mask.dtype != np.dtype(bool)
            or not np.array_equal(host_mask, expected_mask)
            or int(host_state.cell_ids[index]) != cell_id
            or _array_digest(host_mask) != getattr(plan, '_mask_digest', None)
            or _state_provenance(host_state)
            != getattr(plan, '_state_digest', None)
            or host_supplement.source_provenance
            != getattr(plan, '_supplement_digest', None)):
        raise A4ResidentMaintenanceFoundationError(
            'selected maintenance result content seal changed'
        )
    _require_selected_scope(source, host_state, host_supplement, index)
    return plan


def _require_selected_scope(source, state_after, supplement_after, index):
    source_state = _state_host(source.translation.state)
    source_supplement = _supplement_numpy_copy(source.supplement)
    for name in a4._TRANSLATION_ARRAY_FIELDS:
        before = np.asarray(getattr(source_state, name))
        after = np.asarray(getattr(state_after, name))
        if name != 'pools':
            if not _arrays_bit_exact(before, after):
                raise A4ResidentMaintenanceFoundationError(
                    'selected maintenance changed excluded state array: ' + name
                )
            continue
        keep = np.ones(before.shape, dtype=bool)
        keep[index, int(a3.POOL_ATP)] = False
        if not _arrays_bit_exact(before[keep], after[keep]):
            raise A4ResidentMaintenanceFoundationError(
                'selected maintenance changed an excluded pool entry'
            )
    if not _arrays_bit_exact(
            source_supplement.transporters,
            supplement_after.transporters):
        raise A4ResidentMaintenanceFoundationError(
            'selected maintenance changed transporters'
        )
    keep_rows = np.arange(int(source_state.cell_capacity)) != int(index)
    if not _arrays_bit_exact(
            source_supplement.maintenance_shortfall[keep_rows],
            supplement_after.maintenance_shortfall[keep_rows]):
        raise A4ResidentMaintenanceFoundationError(
            'selected maintenance changed non-target shortfall rows'
        )
    return state_after


def paid_maintenance_selected_numpy(
        binding, dt, target_index, target_cell_id):
    """Return a pure NumPy selected-maintenance descriptor."""
    _require_maintenance_binding(binding, diagnostic=True)
    if a4._is_tensor(binding.translation.state.pools):
        raise A4ResidentMaintenanceFoundationError(
            'NumPy selected maintenance requires a NumPy binding'
        )
    index, cell_id = _selected_preflight(
        binding, target_index, target_cell_id,
    )
    dt = _strict_dt(dt)
    atp_after, shortfall_after, _ = _maintenance_numpy_values(
        binding, dt, index,
    )
    state, supplement, mask = _fresh_maintenance_outputs(
        binding, index, atp_after, shortfall_after,
    )
    _require_maintenance_binding(binding, diagnostic=True)
    return _make_selected_plan(
        index, cell_id, dt, mask, state, supplement, binding,
        diagnostic=True,
    )


def paid_maintenance_selected_torch(
        binding, dt, target_index, target_cell_id):
    """Return a pure resident selected plan with 33 version-zero tensors."""
    if torch is None:
        raise RuntimeError('PyTorch is unavailable')
    _require_maintenance_binding(binding, diagnostic=True)
    if not a4._is_tensor(binding.translation.state.pools):
        raise A4ResidentMaintenanceFoundationError(
            'Torch selected maintenance requires a Torch binding'
        )
    index, cell_id = _selected_preflight(
        binding, target_index, target_cell_id,
    )
    dt = _strict_dt(dt)
    atp_after, shortfall_after, _ = _maintenance_torch_values(
        binding, dt, index,
    )
    state, supplement, mask = _fresh_maintenance_outputs(
        binding, index, atp_after, shortfall_after,
    )
    _require_maintenance_binding(binding, diagnostic=True)
    return _make_selected_plan(
        index, cell_id, dt, mask, state, supplement, binding,
        diagnostic=True,
    )


def paid_maintenance_selected(binding, dt, target_index, target_cell_id):
    """Dispatch without accepting caller-owned output or commit authority."""
    _require_maintenance_binding(binding, diagnostic=True)
    if a4._is_tensor(binding.translation.state.pools):
        return paid_maintenance_selected_torch(
            binding, dt, target_index, target_cell_id,
        )
    return paid_maintenance_selected_numpy(
        binding, dt, target_index, target_cell_id,
    )


@dataclass(frozen=True)
class _MaintenanceCpuObservation:
    cells_ref: object
    cells: tuple
    transporter_refs: tuple
    shortfall_refs: tuple
    presence: tuple
    source_values: tuple
    transporters: object
    maintenance_shortfall: object
    transporter_digest: str
    shortfall_digest: str


def _observe_cpu_maintenance(world, state):
    captured = _capture_cpu_supplement(world.cells, state)
    return _MaintenanceCpuObservation(
        cells_ref=world.cells,
        cells=tuple(captured['cells']),
        transporter_refs=tuple(captured['transporter_refs']),
        shortfall_refs=tuple(captured['shortfall_refs']),
        presence=tuple(captured['presence']),
        source_values=tuple(captured['source_values']),
        transporters=np.ascontiguousarray(captured['transporters']).copy(),
        maintenance_shortfall=np.ascontiguousarray(
            captured['maintenance_shortfall'],
        ).copy(),
        transporter_digest=_array_digest(captured['transporters']),
        shortfall_digest=_array_digest(captured['maintenance_shortfall']),
    )


def _maintenance_observation_values(observation):
    if not isinstance(observation, _MaintenanceCpuObservation):
        raise A4ResidentMaintenanceFoundationError(
            'maintenance CPU observation is malformed'
        )
    transporters = np.asarray(observation.transporters)
    shortfall = np.asarray(observation.maintenance_shortfall)
    N = len(observation.cells)
    C = int(transporters.shape[0]) if transporters.ndim == 3 else -1
    if (transporters.dtype != np.dtype(np.float64)
            or shortfall.dtype != np.dtype(np.float64)
            or tuple(transporters.shape) != (
                C, int(a3.MEMBRANE_SEGMENTS), int(a3.CHANNEL_COUNT))
            or tuple(shortfall.shape) != (C,)
            or C <= 0 or N < 0 or N > C
            or len(observation.transporter_refs) != N
            or len(observation.shortfall_refs) != N
            or len(observation.presence) != N
            or len(observation.source_values) != N
            or any(type(value) is not bool
                   for value in observation.presence)
            or not np.isfinite(transporters).all()
            or not np.isfinite(shortfall).all()
            or np.any(transporters < 0.0)
            or np.any(shortfall < 0.0)
            or np.any(transporters[N:].view(np.uint64) != 0)
            or np.any(shortfall[N:].view(np.uint64) != 0)
            or _array_digest(transporters)
            != observation.transporter_digest
            or _array_digest(shortfall) != observation.shortfall_digest):
        raise A4ResidentMaintenanceFoundationError(
            'maintenance CPU observation content seal changed'
        )
    return (
        id(observation.cells_ref), tuple(id(cell) for cell in observation.cells),
        tuple(id(value) for value in observation.transporter_refs),
        tuple(id(value) for value in observation.shortfall_refs),
        tuple(observation.presence), tuple(observation.source_values),
        id(observation.transporters), id(observation.maintenance_shortfall),
        str(observation.transporter_digest),
        str(observation.shortfall_digest),
    )


def _same_maintenance_observation(left, right):
    left_values = _maintenance_observation_values(left)
    right_values = _maintenance_observation_values(right)
    # Detached diagnostic arrays are fresh per observation. Their identities
    # belong to each observation seal, not to CPU source equivalence.
    return (
        left_values[:6] + left_values[8:]
        == right_values[:6] + right_values[8:]
    )


def _copy_maintenance_observation(observation):
    _maintenance_observation_values(observation)
    return _MaintenanceCpuObservation(
        cells_ref=observation.cells_ref,
        cells=tuple(observation.cells),
        transporter_refs=tuple(observation.transporter_refs),
        shortfall_refs=tuple(observation.shortfall_refs),
        presence=tuple(observation.presence),
        source_values=tuple(observation.source_values),
        transporters=np.ascontiguousarray(
            observation.transporters,
        ).copy(),
        maintenance_shortfall=np.ascontiguousarray(
            observation.maintenance_shortfall,
        ).copy(),
        transporter_digest=str(observation.transporter_digest),
        shortfall_digest=str(observation.shortfall_digest),
    )


def _resident_identity_token(owner, arena, observation):
    return hashlib.sha256(repr((
        id(owner._owner_token), str(arena.arena_id),
        int(arena.storage_generation), id(arena.binding),
        _maintenance_observation_values(observation),
    )).encode('utf-8')).hexdigest()


def _make_resident_maintenance_binding(
        owner, arena, observation, previous_binding=None,
        transporter_tensor=None, shortfall_tensor=None):
    if torch is None:
        raise RuntimeError('PyTorch is unavailable')
    translation = arena.binding
    if transporter_tensor is None:
        transporter_tensor = torch.as_tensor(
            np.ascontiguousarray(observation.transporters),
            dtype=torch.float64, device=arena.device,
        ).clone()
    if shortfall_tensor is None:
        shortfall_tensor = torch.as_tensor(
            np.ascontiguousarray(observation.maintenance_shortfall),
            dtype=torch.float64, device=arena.device,
        ).clone()
    if (int(transporter_tensor._version) != 0
            or int(shortfall_tensor._version) != 0):
        raise A4ResidentMaintenanceFoundationError(
            'fresh resident supplement tensors must have version zero'
        )
    base_provenance = _state_provenance(translation.state)
    layout = _cell_layout_digest(translation.state)
    identity_token = _resident_identity_token(owner, arena, observation)
    provenance = _supplement_provenance(
        base_provenance, layout, observation.presence,
        transporter_tensor, shortfall_tensor, identity_token,
    )
    supplement = _make_supplement(
        int(translation.state.cell_capacity),
        int(translation.state.cell_count), provenance,
        transporter_tensor, shortfall_tensor, base_provenance, layout,
        observation.presence, base_state_ref=translation.state,
        base_binding_ref=translation, bindable=True,
        identity_token=identity_token, resident_authority=True,
    )
    binding = _make_maintenance_binding(
        translation, supplement, conversion_source=previous_binding,
    )
    return _require_maintenance_binding(binding, diagnostic=True)


def _resident_binding_values(binding):
    return (
        id(binding), tuple(_binding_creation_values(binding)),
        tuple(_supplement_creation_values(binding.supplement)),
        tuple(_array_storage_descriptor(value)
              for value in _binding_arrays(binding)),
    )


def _require_resident_source_relation(binding, arena, observation):
    _require_maintenance_binding(binding, diagnostic=True)
    supplement = binding.supplement
    if (binding.translation is not arena.binding
            or getattr(supplement, '_resident_authority', None)
            is not _RESIDENT_SUPPLEMENT_TOKEN
            or tuple(getattr(supplement, '_presence', ()))
            != tuple(observation.presence)
            or _array_digest(supplement.transporters)
            != observation.transporter_digest
            or _array_digest(supplement.maintenance_shortfall)
            != observation.shortfall_digest
            or getattr(supplement, '_base_state_ref', None)
            is not arena.binding.state
            or getattr(supplement, '_base_binding_ref', None)
            is not arena.binding
            or getattr(supplement, '_base_state_provenance', None)
            != _state_provenance(arena.binding.state)
            or getattr(supplement, '_layout_digest', None)
            != _cell_layout_digest(arena.binding.state)):
        raise A4ResidentMaintenanceFoundationError(
            'resident supplement/base/source relation differs'
        )
    return binding


@dataclass(frozen=True)
class _MaintenanceOwnerGuard:
    owner_token_ref: object
    transaction_guard_ref: object
    transaction_guard_values: tuple
    binding_ref: object
    binding_values: tuple
    source_ref: object
    source_values: tuple
    source_snapshot_ref: object
    source_snapshot_values: tuple
    last_observed_ref: object
    last_observed_values: tuple
    last_observed_snapshot_ref: object
    last_observed_snapshot_values: tuple
    prepared_ref: object
    prepared_seal_ref: object
    prepared_values: object
    rebuild_candidate_ref: object
    rebuild_seal_ref: object
    rebuild_values: object
    phase: str


def _maintenance_guard_values(guard):
    if not isinstance(guard, _MaintenanceOwnerGuard):
        raise A4ResidentMaintenanceFoundationError(
            'maintenance owner guard is malformed'
        )
    return (
        id(guard.owner_token_ref), id(guard.transaction_guard_ref),
        tuple(guard.transaction_guard_values), id(guard.binding_ref),
        tuple(guard.binding_values), id(guard.source_ref),
        tuple(guard.source_values), id(guard.source_snapshot_ref),
        tuple(guard.source_snapshot_values), id(guard.last_observed_ref),
        tuple(guard.last_observed_values),
        id(guard.last_observed_snapshot_ref),
        tuple(guard.last_observed_snapshot_values), id(guard.prepared_ref),
        id(guard.prepared_seal_ref),
        None if guard.prepared_values is None
        else tuple(guard.prepared_values), id(guard.rebuild_candidate_ref),
        id(guard.rebuild_seal_ref),
        None if guard.rebuild_values is None
        else tuple(guard.rebuild_values), str(guard.phase),
    )


@dataclass(frozen=True)
class _PreparedArenaDescriptor:
    arena_id: str
    storage_generation: int
    device: str
    config_state: tuple
    epochs_values: tuple
    binding: object


@dataclass(frozen=True)
class _PreparedCompositeSeal:
    prepared_ref: object
    values: tuple


@dataclass(frozen=True)
class _FinalizedCompositeSeal:
    finalized_ref: object
    values: tuple


def _prepared_composite_values(prepared):
    descriptor = getattr(prepared, '_a49c_arena_descriptor', None)
    return (
        id(prepared), tuple(prepared.seal.values),
        id(getattr(prepared, '_a49c_old_binding', None)),
        id(getattr(prepared, '_a49c_candidate_binding', None)),
        _resident_binding_values(
            getattr(prepared, '_a49c_candidate_binding')
        ),
        id(getattr(prepared, '_a49c_source', None)),
        _maintenance_observation_values(
            getattr(prepared, '_a49c_source')
        ),
        str(descriptor.arena_id), int(descriptor.storage_generation),
        str(descriptor.device), tuple(descriptor.config_state),
        tuple(descriptor.epochs_values), id(descriptor.binding),
    )


def _finalized_composite_values(finalized):
    return (
        id(finalized), tuple(finalized.seal.values),
        id(getattr(finalized, '_a49c_old_binding', None)),
        id(getattr(finalized, '_a49c_new_binding', None)),
        _resident_binding_values(
            getattr(finalized, '_a49c_new_binding')
        ),
        id(getattr(finalized, '_a49c_source', None)),
        _maintenance_observation_values(
            getattr(finalized, '_a49c_source')
        ),
        id(getattr(finalized, '_a49c_guard', None)),
        tuple(getattr(finalized, '_a49c_guard_snapshot')),
    )


@dataclass(frozen=True)
class _CompositeRebuildSeal:
    candidate_ref: object
    values: tuple


@dataclass
class _CompositeRebuildCandidate:
    owner_ref: object
    base_candidate: object
    old_binding_ref: object
    new_binding_ref: object
    source_ref: object
    expected_lease_serial: int
    owner_epochs_ref: object
    base_guard_ref: object
    transaction_guard_ref: object
    transaction_guard_snapshot: tuple
    maintenance_guard_ref: object
    maintenance_guard_snapshot: tuple
    seal: object = None
    consumed: bool = False

    def __getstate__(self):
        raise A4ResidentMaintenanceFoundationError(
            'composite rebuild candidates are not serializable'
        )


def _composite_rebuild_values(candidate):
    base = candidate.base_candidate
    base_seal = getattr(base, '_seal', None)
    return (
        id(candidate), id(candidate.owner_ref), id(base),
        id(getattr(base_seal, 'candidate_ref', None)),
        id(getattr(base, '_arena', None)),
        a49a._arena_creation_seal_values(base_seal.arena_seal),
        id(candidate.old_binding_ref), id(candidate.new_binding_ref),
        _resident_binding_values(candidate.new_binding_ref),
        id(candidate.source_ref),
        _maintenance_observation_values(candidate.source_ref),
        int(candidate.expected_lease_serial),
        id(candidate.owner_epochs_ref),
        a49a._epoch_values(candidate.owner_epochs_ref),
        id(candidate.base_guard_ref),
        a49b._base_guard_values(candidate.base_guard_ref),
        id(candidate.transaction_guard_ref),
        tuple(candidate.transaction_guard_snapshot),
        id(candidate.maintenance_guard_ref),
        tuple(candidate.maintenance_guard_snapshot),
        bool(candidate.consumed),
    )


class _A4ResidentMaintenanceFoundationOwner(
        a49b._A4ResidentTranslationOwner):
    """One base arena and two sidecar tensors under a single owner lock."""

    __slots__ = (
        '_maintenance_initialized', '_maintenance_binding',
        '_maintenance_source', '_maintenance_last_observed',
        '_maintenance_guard', '_maintenance_guard_snapshot',
        '_maintenance_phase', '_maintenance_rebuild_candidate',
    )

    @classmethod
    def from_cpu(cls, world, a4_config, device):
        owner = a49b._A4ResidentTranslationOwner.from_cpu.__func__(
            cls, world, a4_config, device,
        )
        with owner._lock:
            stable = a49a._stable_cpu_snapshot(
                world, owner._config, require_cache_exact=True,
            )
            if not a49a._same_source(
                    owner._arena.source, stable.attestation):
                raise A4ResidentMaintenanceFoundationError(
                    'CPU source changed before composite construction'
                )
            observation = _observe_cpu_maintenance(world, stable.state)
            binding = _make_resident_maintenance_binding(
                owner, owner._arena, observation,
            )
            after = a49a._stable_cpu_snapshot(
                world, owner._config, require_cache_exact=True,
            )
            replay = _observe_cpu_maintenance(world, after.state)
            if (not a49a._same_source(stable.attestation, after.attestation)
                    or not _same_maintenance_observation(
                        observation, replay)):
                raise A4ResidentMaintenanceFoundationError(
                    'CPU source changed during composite construction'
                )
            owner._maintenance_initialized = True
            owner._maintenance_binding = binding
            owner._maintenance_source = observation
            owner._maintenance_last_observed = replay
            owner._maintenance_guard = None
            owner._maintenance_guard_snapshot = None
            owner._maintenance_phase = 'idle'
            owner._maintenance_rebuild_candidate = None
            owner._refresh_guard()
        return owner

    def _make_maintenance_guard(self):
        prepared = self._tx_prepared
        prepared_seal = (
            getattr(prepared, '_a49c_seal', None)
            if prepared is not None else None
        )
        prepared_values = (
            tuple(prepared_seal.values)
            if isinstance(prepared_seal, _PreparedCompositeSeal) else None
        )
        rebuild = self._maintenance_rebuild_candidate
        rebuild_seal = (
            getattr(rebuild, 'seal', None) if rebuild is not None else None
        )
        rebuild_values = (
            tuple(rebuild_seal.values)
            if isinstance(rebuild_seal, _CompositeRebuildSeal) else None
        )
        source_snapshot = _copy_maintenance_observation(
            self._maintenance_source,
        )
        last_snapshot = _copy_maintenance_observation(
            self._maintenance_last_observed,
        )
        return _MaintenanceOwnerGuard(
            owner_token_ref=self._owner_token,
            transaction_guard_ref=self._tx_guard,
            transaction_guard_values=a49b._transaction_guard_values(
                self._tx_guard,
            ),
            binding_ref=self._maintenance_binding,
            binding_values=_resident_binding_values(
                self._maintenance_binding,
            ),
            source_ref=self._maintenance_source,
            source_values=_maintenance_observation_values(
                self._maintenance_source,
            ),
            source_snapshot_ref=source_snapshot,
            source_snapshot_values=_maintenance_observation_values(
                source_snapshot,
            ),
            last_observed_ref=self._maintenance_last_observed,
            last_observed_values=_maintenance_observation_values(
                self._maintenance_last_observed,
            ),
            last_observed_snapshot_ref=last_snapshot,
            last_observed_snapshot_values=_maintenance_observation_values(
                last_snapshot,
            ),
            prepared_ref=prepared,
            prepared_seal_ref=prepared_seal,
            prepared_values=prepared_values,
            rebuild_candidate_ref=rebuild,
            rebuild_seal_ref=rebuild_seal,
            rebuild_values=rebuild_values,
            phase=str(self._maintenance_phase),
        )

    def _refresh_guard(self):
        a49b._A4ResidentTranslationOwner._refresh_guard(self)
        if getattr(self, '_maintenance_initialized', False):
            self._maintenance_guard = self._make_maintenance_guard()
            self._maintenance_guard_snapshot = _maintenance_guard_values(
                self._maintenance_guard,
            )

    def _require_guard(self):
        base = a49b._A4ResidentTranslationOwner._require_guard(self)
        if not getattr(self, '_maintenance_initialized', False):
            return base
        guard = self._maintenance_guard
        if (not isinstance(guard, _MaintenanceOwnerGuard)
                or _maintenance_guard_values(guard)
                != tuple(self._maintenance_guard_snapshot)
                or guard.owner_token_ref is not self._owner_token
                or guard.transaction_guard_ref is not self._tx_guard
                or a49b._transaction_guard_values(self._tx_guard)
                != guard.transaction_guard_values
                or self._maintenance_binding is not guard.binding_ref
                or _resident_binding_values(self._maintenance_binding)
                != guard.binding_values
                or self._maintenance_source is not guard.source_ref
                or _maintenance_observation_values(self._maintenance_source)
                != guard.source_values
                or _maintenance_observation_values(
                    guard.source_snapshot_ref,
                ) != guard.source_snapshot_values
                or self._maintenance_last_observed
                is not guard.last_observed_ref
                or _maintenance_observation_values(
                    self._maintenance_last_observed,
                ) != guard.last_observed_values
                or _maintenance_observation_values(
                    guard.last_observed_snapshot_ref,
                ) != guard.last_observed_snapshot_values
                or self._tx_prepared is not guard.prepared_ref
                or getattr(self._tx_prepared, '_a49c_seal', None)
                is not guard.prepared_seal_ref
                or (
                    None if guard.prepared_seal_ref is None else tuple(
                        guard.prepared_seal_ref.values
                    )
                ) != guard.prepared_values
                or self._maintenance_rebuild_candidate
                is not guard.rebuild_candidate_ref
                or getattr(
                    self._maintenance_rebuild_candidate, 'seal', None,
                ) is not guard.rebuild_seal_ref
                or (
                    None if guard.rebuild_seal_ref is None else tuple(
                        guard.rebuild_seal_ref.values
                    )
                ) != guard.rebuild_values
                or str(self._maintenance_phase) != guard.phase):
            raise A4ResidentMaintenanceFoundationError(
                'composite maintenance owner guard differs'
            )
        if self._lifecycle != INVALID:
            _require_resident_source_relation(
                self._maintenance_binding, self._arena,
                self._maintenance_source,
            )
        return base

    def _force_metadata_invalid(self, reason):
        guard = getattr(self, '_maintenance_guard', None)
        if getattr(self, '_maintenance_initialized', False):
            if not isinstance(guard, _MaintenanceOwnerGuard):
                raise A4ResidentMaintenanceFoundationError(str(reason))
            self._maintenance_binding = guard.binding_ref
            self._maintenance_source = guard.source_snapshot_ref
            self._maintenance_last_observed = (
                guard.last_observed_snapshot_ref
            )
            self._maintenance_rebuild_candidate = None
            self._maintenance_phase = 'idle'
        return a49b._A4ResidentTranslationOwner._force_metadata_invalid(
            self, reason,
        )

    def _transition_invalid(
            self, reason, domains=('M', 'R', 'S', 'C'),
            advance_unobserved=False):
        if getattr(self, '_maintenance_initialized', False):
            self._maintenance_rebuild_candidate = None
            self._maintenance_phase = 'idle'
        return a49b._A4ResidentTranslationOwner._transition_invalid(
            self, reason, domains, advance_unobserved,
        )

    def audit_cpu(self, world):
        with self._lock:
            self._reject_external_during_transaction('audit_cpu')
            if self._maintenance_rebuild_candidate is not None:
                raise A4ResidentMaintenanceFoundationError(
                    'audit_cpu is forbidden with a pending composite rebuild'
                )
            before_state_epoch = int(self._epochs.state_epoch)
            before_lease_serial = int(self._lease_serial)
            lifecycle = a49a.A4ResidentArenaOwner.audit_cpu(self, world)
            try:
                stable = a49a._stable_cpu_snapshot(
                    world, self._config, require_cache_exact=False,
                )
                observation = _observe_cpu_maintenance(world, stable.state)
            except Exception as exc:
                self._transition_invalid(
                    'maintenance CPU source is malformed', ('S',),
                    advance_unobserved=True,
                )
                raise A4ResidentMaintenanceFoundationError(
                    'maintenance CPU source audit failed closed'
                ) from exc
            observed_change = not _same_maintenance_observation(
                self._maintenance_last_observed, observation,
            )
            active_change = not _same_maintenance_observation(
                self._maintenance_source, observation,
            )
            if observed_change:
                if (lifecycle != INVALID
                        and int(self._epochs.state_epoch)
                        == before_state_epoch):
                    self._epochs = a49a._advance_epochs(
                        self._epochs, frozenset(('S',)), cache_built=False,
                    )
                self._maintenance_last_observed = observation
            if active_change and lifecycle != INVALID:
                first_or_new = lifecycle == COHERENT or observed_change
                self._lifecycle = CPU_NEWER
                self._arena.lifecycle = CPU_NEWER
                self._arena.dirty_domains = frozenset(
                    set(self._arena.dirty_domains).union(('S',))
                )
                self._arena.invalid_reason = ''
                if first_or_new and self._lease_serial == before_lease_serial:
                    self._lease_serial += 1
                    self._arena.lease_serial = self._lease_serial
                    self._issued_lease_token = None
                lifecycle = CPU_NEWER
            self._refresh_guard()
            return self._lifecycle

    def _prebuild_composite_guards(
            self, arena, arena_seal, owner_epochs, base_source,
            binding, maintenance_source, lease_serial, transaction_serial):
        base_guard = a49a._OwnerCanonicalGuard(
            owner_token_ref=self._owner_token,
            arena_ref=arena,
            arena_seal=arena_seal,
            arena_seal_snapshot=a49a._copy_arena_creation_seal(arena_seal),
            config_ref=self._config,
            epochs_ref=owner_epochs,
            last_observed_ref=base_source,
            last_observed_values=a49a._source_identity_values(base_source),
            lifecycle=COHERENT,
            epochs_values=a49a._epoch_values(owner_epochs),
            device=str(self._device),
            config_state=a49a._config_state(self._config),
            storage_generation=int(arena.storage_generation),
            lease_serial=int(lease_serial),
            issued_lease_token=None,
            active_lease_token=None,
            active_lease_consumed=False,
        )
        transaction_guard = a49b._TranslationOwnerGuard(
            owner_token_ref=self._owner_token,
            base_guard_ref=base_guard,
            base_guard_values=a49b._base_guard_values(base_guard),
            transaction_active=False,
            transaction_serial=int(transaction_serial),
            transaction_token_ref=None,
            transaction_stage=a49b._TX_IDLE,
            transaction_world_ref=None,
            transaction_consumed=False,
            prepared_ref=None,
            prepared_seal_ref=None,
            prepared_values=None,
        )
        transaction_snapshot = a49b._transaction_guard_values(
            transaction_guard,
        )
        source_snapshot = _copy_maintenance_observation(
            maintenance_source,
        )
        last_snapshot = _copy_maintenance_observation(
            maintenance_source,
        )
        maintenance_guard = _MaintenanceOwnerGuard(
            owner_token_ref=self._owner_token,
            transaction_guard_ref=transaction_guard,
            transaction_guard_values=transaction_snapshot,
            binding_ref=binding,
            binding_values=_resident_binding_values(binding),
            source_ref=maintenance_source,
            source_values=_maintenance_observation_values(
                maintenance_source,
            ),
            source_snapshot_ref=source_snapshot,
            source_snapshot_values=_maintenance_observation_values(
                source_snapshot,
            ),
            last_observed_ref=maintenance_source,
            last_observed_values=_maintenance_observation_values(
                maintenance_source,
            ),
            last_observed_snapshot_ref=last_snapshot,
            last_observed_snapshot_values=_maintenance_observation_values(
                last_snapshot,
            ),
            prepared_ref=None,
            prepared_seal_ref=None,
            prepared_values=None,
            rebuild_candidate_ref=None,
            rebuild_seal_ref=None,
            rebuild_values=None,
            phase='idle',
        )
        return (
            base_guard, transaction_guard, transaction_snapshot,
            maintenance_guard, _maintenance_guard_values(maintenance_guard),
        )

    def _require_composite_candidate(self, candidate):
        if (not isinstance(candidate, _CompositeRebuildCandidate)
                or candidate.owner_ref is not self
                or not isinstance(candidate.seal, _CompositeRebuildSeal)
                or candidate.seal.candidate_ref is not candidate
                or candidate.seal.values
                != _composite_rebuild_values(candidate)
                or candidate.consumed):
            raise A4ResidentMaintenanceFoundationError(
                'composite rebuild candidate seal differs'
            )
        base_seal = self._require_candidate_seal(
            candidate.base_candidate, self._arena, self._arena_seal,
        )
        if (candidate.old_binding_ref is not self._maintenance_binding
                or candidate.new_binding_ref.translation
                is not candidate.base_candidate._arena.binding
                or candidate.expected_lease_serial != self._lease_serial
                or a49b._base_guard_values(candidate.base_guard_ref)
                != a49b._base_guard_values(
                    candidate.transaction_guard_ref.base_guard_ref,
                )
                or a49b._transaction_guard_values(
                    candidate.transaction_guard_ref,
                ) != tuple(candidate.transaction_guard_snapshot)
                or _maintenance_guard_values(
                    candidate.maintenance_guard_ref,
                ) != tuple(candidate.maintenance_guard_snapshot)
                or candidate.maintenance_guard_ref.binding_ref
                is not candidate.new_binding_ref
                or candidate.maintenance_guard_ref.transaction_guard_ref
                is not candidate.transaction_guard_ref):
            raise A4ResidentMaintenanceFoundationError(
                'composite rebuild candidate relation differs'
            )
        _require_resident_source_relation(
            candidate.new_binding_ref, candidate.base_candidate._arena,
            candidate.source_ref,
        )
        return base_seal

    def prepare_rebuild(self, world):
        with self._lock:
            self._reject_external_during_transaction('prepare_rebuild')
            if self._maintenance_rebuild_candidate is not None:
                raise A4ResidentMaintenanceFoundationError(
                    'another composite rebuild candidate is pending'
                )
            base_candidate = a49a.A4ResidentArenaOwner.prepare_rebuild(
                self, world,
            )
            stable = a49a._stable_cpu_snapshot(
                world, self._config, require_cache_exact=True,
            )
            if not a49a._same_source(
                    base_candidate._source, stable.attestation):
                raise A4ResidentMaintenanceFoundationError(
                    'CPU base changed during composite rebuild'
                )
            observation = _observe_cpu_maintenance(world, stable.state)
            binding = _make_resident_maintenance_binding(
                self, base_candidate._arena, observation,
                previous_binding=self._maintenance_binding,
            )
            after = a49a._stable_cpu_snapshot(
                world, self._config, require_cache_exact=True,
            )
            replay = _observe_cpu_maintenance(world, after.state)
            if (not a49a._same_source(stable.attestation, after.attestation)
                    or not _same_maintenance_observation(
                        observation, replay)):
                raise A4ResidentMaintenanceFoundationError(
                    'CPU source changed during composite rebuild'
                )
            base_candidate._arena.lease_serial = self._lease_serial
            base_candidate._arena.active_leases = 0
            owner_epochs = a49a._copy_epochs(
                base_candidate._arena.epochs,
            )
            (base_guard, transaction_guard, transaction_snapshot,
             maintenance_guard, maintenance_snapshot) = (
                self._prebuild_composite_guards(
                    base_candidate._arena,
                    base_candidate._seal.arena_seal,
                    owner_epochs, base_candidate._source,
                    binding, replay, self._lease_serial, self._tx_serial,
                )
            )
            candidate = _CompositeRebuildCandidate(
                owner_ref=self,
                base_candidate=base_candidate,
                old_binding_ref=self._maintenance_binding,
                new_binding_ref=binding,
                source_ref=replay,
                expected_lease_serial=int(self._lease_serial),
                owner_epochs_ref=owner_epochs,
                base_guard_ref=base_guard,
                transaction_guard_ref=transaction_guard,
                transaction_guard_snapshot=transaction_snapshot,
                maintenance_guard_ref=maintenance_guard,
                maintenance_guard_snapshot=maintenance_snapshot,
            )
            candidate.seal = _CompositeRebuildSeal(
                candidate_ref=candidate,
                values=_composite_rebuild_values(candidate),
            )
            self._require_composite_candidate(candidate)
            self._maintenance_rebuild_candidate = candidate
            self._maintenance_phase = 'rebuild-prepared'
            self._refresh_guard()
            return candidate

    def _terminalize_pending_rebuild(
            self, candidate, invalid_reason=None):
        if self._maintenance_rebuild_candidate is candidate:
            self._maintenance_rebuild_candidate = None
        self._maintenance_phase = 'idle'
        try:
            candidate.consumed = True
            candidate.base_candidate._consumed = True
        except Exception:
            pass
        if invalid_reason is not None:
            self._transition_invalid(
                invalid_reason, ('R', 'S', 'C'),
                advance_unobserved=True,
            )
        else:
            self._refresh_guard()

    def _validate_composite_rebuild_sources(
            self, candidate, base_seal, world):
        base = candidate.base_candidate
        a49a._attest_resident(
            base._arena, base_seal.arena_seal, readback=True,
        )
        if self._lifecycle != INVALID:
            a49a._attest_resident(
                self._arena, self._arena_seal, readback=True,
            )
        _require_resident_source_relation(
            self._maintenance_binding, self._arena,
            self._maintenance_source,
        )
        _require_resident_source_relation(
            candidate.new_binding_ref, base._arena,
            candidate.source_ref,
        )
        fresh = a49a._stable_cpu_snapshot(
            world, self._config, require_cache_exact=True,
        )
        observation = _observe_cpu_maintenance(world, fresh.state)
        if (not a49a._same_source(base._source, fresh.attestation)
                or not _same_maintenance_observation(
                    candidate.source_ref, observation)):
            raise A4ResidentMaintenanceFoundationError(
                'CPU source differs from composite rebuild candidate'
            )
        return None

    def swap_rebuild(
            self, candidate, world, expected_arena_id,
            expected_lifecycle, expected_epochs):
        with self._lock:
            self._reject_external_during_transaction('swap_rebuild')
            self._guard_or_invalidate()
            if self._maintenance_rebuild_candidate is not candidate:
                raise A4ResidentMaintenanceFoundationError(
                    'composite rebuild candidate is not owner-pending'
                )
            try:
                base_seal = self._require_composite_candidate(candidate)
            except BaseException as exc:
                self._terminalize_pending_rebuild(
                    candidate,
                    'composite rebuild candidate trust attestation failed',
                )
                raise A4ResidentMaintenanceFoundationError(
                    'composite rebuild candidate failed closed'
                ) from exc
            base = candidate.base_candidate
            if (str(expected_arena_id) != self._arena_seal.arena_id
                    or expected_lifecycle != self._lifecycle
                    or not isinstance(
                        expected_epochs, a49a.A4ResidentArenaEpochs)
                    or a49a._epoch_values(expected_epochs)
                    != a49a._epoch_values(self._epochs)
                    or base._expected_arena_id
                    != self._arena_seal.arena_id
                    or base._expected_lifecycle != self._lifecycle
                    or a49a._epoch_values(base._expected_epochs)
                    != a49a._epoch_values(self._epochs)
                    or base._expected_storage_generation
                    != self._storage_generation
                    or self._active_lease_token is not None
                    or self._tx_active
                    or self._lease_serial
                    != candidate.expected_lease_serial):
                self._terminalize_pending_rebuild(candidate)
                raise A4ResidentMaintenanceFoundationError(
                    'composite rebuild compare-and-swap guard differs'
                )
            try:
                self._validate_composite_rebuild_sources(
                    candidate, base_seal, world,
                )
            except BaseException:
                self._terminalize_pending_rebuild(candidate)
                raise

            # Sole composite publication.  Every object assigned below was
            # created and fully attested before these field writes.
            old = self._arena
            self._arena = base._arena
            self._arena_seal = base_seal.arena_seal
            self._epochs = candidate.owner_epochs_ref
            self._last_observed = base._source
            self._storage_generation = base._arena.storage_generation
            self._lifecycle = COHERENT
            self._arena.lifecycle = COHERENT
            self._arena.lease_serial = self._lease_serial
            self._arena.active_leases = 0
            self._issued_lease_token = None
            self._active_lease_token = None
            self._active_lease_consumed = False
            self._guard = candidate.base_guard_ref
            self._tx_active = False
            self._tx_token = None
            self._tx_stage = a49b._TX_IDLE
            self._tx_world = None
            self._tx_prepared = None
            self._tx_consumed = False
            self._tx_guard = candidate.transaction_guard_ref
            self._tx_guard_snapshot = candidate.transaction_guard_snapshot
            self._maintenance_binding = candidate.new_binding_ref
            self._maintenance_source = candidate.source_ref
            self._maintenance_last_observed = candidate.source_ref
            self._maintenance_rebuild_candidate = None
            self._maintenance_phase = 'idle'
            self._maintenance_guard = candidate.maintenance_guard_ref
            self._maintenance_guard_snapshot = (
                candidate.maintenance_guard_snapshot
            )
            base._consumed = True
            candidate.consumed = True
            old.retired = True
            old.lifecycle = INVALID
            old.invalid_reason = 'retired after guarded composite rebuild swap'
            old.active_leases = 0
            old.lease_serial += 1
            return self

    def _discard_composite_rebuild(self, candidate):
        with self._lock:
            self._guard_or_invalidate()
            if self._maintenance_rebuild_candidate is not candidate:
                raise A4ResidentMaintenanceFoundationError(
                    'no matching composite rebuild candidate is pending'
                )
            try:
                self._require_composite_candidate(candidate)
            except BaseException as exc:
                self._terminalize_pending_rebuild(
                    candidate,
                    'discarded composite candidate failed trust attestation',
                )
                raise A4ResidentMaintenanceFoundationError(
                    'discarded composite candidate failed closed'
                ) from exc
            self._terminalize_pending_rebuild(candidate)
            return None

    def _cohere_rank5(self, world):
        with self._lock:
            self._reject_external_during_transaction('rank-5 coherence')
            lifecycle = self.audit_cpu(world)
            if lifecycle == INVALID:
                raise A4ResidentMaintenanceFoundationError(
                    'INVALID composite generation is never auto-rebuilt'
                )
            if lifecycle == CPU_NEWER:
                expected_id = self.arena_id
                expected_lifecycle = self.lifecycle
                expected_epochs = self.epochs
                candidate = self.prepare_rebuild(world)
                self.swap_rebuild(
                    candidate, world, expected_id,
                    expected_lifecycle, expected_epochs,
                )
            if self.lifecycle != COHERENT:
                raise A4ResidentMaintenanceFoundationError(
                    'rank-5 composite source did not become coherent'
                )
            return self

    def _require_prepared_composite(self, prepared):
        a49b._require_prepared_storage(
            prepared, self, self._tx_token,
        )
        seal = getattr(prepared, '_a49c_seal', None)
        if (not isinstance(seal, _PreparedCompositeSeal)
                or seal.prepared_ref is not prepared
                or tuple(seal.values) != _prepared_composite_values(prepared)
                or getattr(prepared, '_a49c_old_binding', None)
                is not self._maintenance_binding):
            raise A4ResidentMaintenanceFoundationError(
                'prepared composite translation seal differs'
            )
        descriptor = prepared._a49c_arena_descriptor
        binding = prepared._a49c_candidate_binding
        if (descriptor.binding is not prepared.candidate_binding
                or descriptor.binding is not binding.translation
                or descriptor.arena_id != prepared.expected_arena_id
                or descriptor.storage_generation
                != prepared.expected_storage_generation
                or descriptor.device != self._device
                or descriptor.config_state
                != a49a._config_state(self._config)
                or descriptor.epochs_values
                != a49a._epoch_values(prepared.expected_epochs)):
            raise A4ResidentMaintenanceFoundationError(
                'prepared composite generation relation differs'
            )
        _require_resident_source_relation(
            binding, descriptor, prepared._a49c_source,
        )
        old_supplement = self._maintenance_binding.supplement
        new_supplement = binding.supplement
        if (not _arrays_bit_exact(
                _host_array(old_supplement.transporters),
                _host_array(new_supplement.transporters))
                or not _arrays_bit_exact(
                    _host_array(old_supplement.maintenance_shortfall),
                    _host_array(new_supplement.maintenance_shortfall))):
            raise A4ResidentMaintenanceFoundationError(
                'translation candidate changed maintenance sidecar values'
            )
        plan_arrays = (
            (prepared.selected_plan.target_mask,)
            + tuple(getattr(prepared.selected_plan.state_after, name)
                    for name in a4._TRANSLATION_ARRAY_FIELDS)
        )
        plan_storage = tuple(
            _array_storage_descriptor(value) for value in plan_arrays
        )
        side_storage = tuple(_array_storage_descriptor(value) for value in (
            new_supplement.transporters,
            new_supplement.maintenance_shortfall,
        ))
        if any(_storage_overlaps(left, right)
               for left in side_storage for right in plan_storage):
            raise A4ResidentMaintenanceFoundationError(
                'translation sidecar candidate aliases its event plan'
            )
        return prepared

    def _build_prepared_storage(
            self, world, cell, dt, target_index, sync_performed,
            integration_state):
        with self._lock:
            self._guard_or_invalidate()
            self._maintenance_phase = 'building-translation-candidate'
            self._refresh_guard()
            prepared = None
            try:
                prepared = (
                    a49b._A4ResidentTranslationOwner._build_prepared_storage(
                        self, world, cell, dt, target_index,
                        sync_performed, integration_state,
                    )
                )
                observation = self._maintenance_source
                descriptor = _PreparedArenaDescriptor(
                    arena_id=str(prepared.expected_arena_id),
                    storage_generation=int(
                        prepared.expected_storage_generation
                    ),
                    device=str(self._device),
                    config_state=a49a._config_state(self._config),
                    epochs_values=a49a._epoch_values(
                        prepared.expected_epochs,
                    ),
                    binding=prepared.candidate_binding,
                )
                old_supplement = self._maintenance_binding.supplement
                binding = _make_resident_maintenance_binding(
                    self, descriptor, observation,
                    previous_binding=self._maintenance_binding,
                    transporter_tensor=(
                        old_supplement.transporters.clone()
                    ),
                    shortfall_tensor=(
                        old_supplement.maintenance_shortfall.clone()
                    ),
                )
                prepared._a49c_old_binding = self._maintenance_binding
                prepared._a49c_candidate_binding = binding
                prepared._a49c_source = observation
                prepared._a49c_arena_descriptor = descriptor
                prepared._a49c_seal = _PreparedCompositeSeal(
                    prepared_ref=prepared, values=tuple(),
                )
                prepared._a49c_seal = _PreparedCompositeSeal(
                    prepared_ref=prepared,
                    values=_prepared_composite_values(prepared),
                )
                self._maintenance_phase = 'translation-prepared'
                self._refresh_guard()
                return self._require_prepared_composite(prepared)
            except BaseException as original:
                try:
                    if self._tx_active:
                        registered = self._tx_prepared
                        self._maintenance_phase = 'building-abort'
                        self._refresh_guard()
                        a49b._A4ResidentTranslationOwner._abort_preclaim(
                            self, world, registered,
                        )
                    self._maintenance_phase = 'idle'
                    self._refresh_guard()
                except BaseException as abort_error:
                    if self._lifecycle == COHERENT:
                        self._transition_invalid(
                            'sidecar candidate failure could not abort',
                            ('S',), advance_unobserved=True,
                        )
                    raise A4ResidentMaintenanceFoundationError(
                        'sidecar candidate failure abort failed closed'
                    ) from original
                raise

    def _require_prepared_for_claim(
            self, world, cell, dt, prepared, integration_state):
        with self._lock:
            result = (
                a49b._A4ResidentTranslationOwner._require_prepared_for_claim(
                    self, world, cell, dt, prepared, integration_state,
                )
            )
            self._require_prepared_composite(result)
            stable = a49a._stable_cpu_snapshot(
                world, self._config, require_cache_exact=True,
            )
            observation = _observe_cpu_maintenance(world, stable.state)
            if not _same_maintenance_observation(
                    self._maintenance_source, observation):
                raise A4ResidentMaintenanceFoundationError(
                    'maintenance CPU source changed before translation claim'
                )
            return result

    def _mark_translation_trust_failure(self):
        self._maintenance_phase = 'translation-trust-failed'
        self._refresh_guard()

    def _require_finalized_composite(self, finalized, prepared):
        a49b._require_finalized_swap(finalized, self, prepared)
        seal = getattr(finalized, '_a49c_final_seal', None)
        if (not isinstance(seal, _FinalizedCompositeSeal)
                or seal.finalized_ref is not finalized
                or tuple(seal.values)
                != _finalized_composite_values(finalized)
                or finalized._a49c_old_binding
                is not self._maintenance_binding
                or finalized._a49c_new_binding.translation
                is not finalized.new_arena_ref.binding
                or finalized._a49c_guard.binding_ref
                is not finalized._a49c_new_binding
                or finalized._a49c_guard.transaction_guard_ref
                is not finalized.transaction_guard_ref
                or _maintenance_guard_values(finalized._a49c_guard)
                != tuple(finalized._a49c_guard_snapshot)):
            raise A4ResidentMaintenanceFoundationError(
                'finalized composite translation seal differs'
            )
        _require_resident_source_relation(
            finalized._a49c_new_binding, finalized.new_arena_ref,
            finalized._a49c_source,
        )
        return finalized

    def _finalize_after_cpu_publish(self, world, prepared):
        with self._lock:
            try:
                self._require_prepared_composite(prepared)
            except BaseException:
                self._mark_translation_trust_failure()
                raise
            self._maintenance_phase = 'translation-finalizing'
            self._refresh_guard()
            try:
                finalized = (
                    a49b._A4ResidentTranslationOwner.
                    _finalize_after_cpu_publish(self, world, prepared)
                )
                stable = a49a._stable_cpu_snapshot(
                    world, self._config, require_cache_exact=True,
                )
                observation = _observe_cpu_maintenance(
                    world, stable.state,
                )
                if (not a49a._same_source(
                        finalized.fresh_source_ref, stable.attestation)
                        or not _same_maintenance_observation(
                            self._maintenance_source, observation)):
                    raise A4ResidentMaintenanceFoundationError(
                        'post-translation maintenance source changed'
                    )
                candidate = prepared._a49c_candidate_binding
                supplement = candidate.supplement
                binding = _make_resident_maintenance_binding(
                    self, finalized.new_arena_ref, observation,
                    previous_binding=self._maintenance_binding,
                    transporter_tensor=supplement.transporters,
                    shortfall_tensor=supplement.maintenance_shortfall,
                )
                source_snapshot = _copy_maintenance_observation(observation)
                last_snapshot = _copy_maintenance_observation(observation)
                maintenance_guard = _MaintenanceOwnerGuard(
                    owner_token_ref=self._owner_token,
                    transaction_guard_ref=finalized.transaction_guard_ref,
                    transaction_guard_values=(
                        finalized.transaction_guard_snapshot
                    ),
                    binding_ref=binding,
                    binding_values=_resident_binding_values(binding),
                    source_ref=observation,
                    source_values=_maintenance_observation_values(
                        observation,
                    ),
                    source_snapshot_ref=source_snapshot,
                    source_snapshot_values=_maintenance_observation_values(
                        source_snapshot,
                    ),
                    last_observed_ref=observation,
                    last_observed_values=_maintenance_observation_values(
                        observation,
                    ),
                    last_observed_snapshot_ref=last_snapshot,
                    last_observed_snapshot_values=(
                        _maintenance_observation_values(last_snapshot)
                    ),
                    prepared_ref=None,
                    prepared_seal_ref=None,
                    prepared_values=None,
                    rebuild_candidate_ref=None,
                    rebuild_seal_ref=None,
                    rebuild_values=None,
                    phase='idle',
                )
                finalized._a49c_old_binding = self._maintenance_binding
                finalized._a49c_new_binding = binding
                finalized._a49c_source = observation
                finalized._a49c_guard = maintenance_guard
                finalized._a49c_guard_snapshot = (
                    _maintenance_guard_values(maintenance_guard)
                )
                finalized._a49c_final_seal = _FinalizedCompositeSeal(
                    finalized_ref=finalized, values=tuple(),
                )
                finalized._a49c_final_seal = _FinalizedCompositeSeal(
                    finalized_ref=finalized,
                    values=_finalized_composite_values(finalized),
                )
                self._maintenance_phase = 'translation-prepared'
                self._refresh_guard()
                try:
                    return self._require_finalized_composite(
                        finalized, prepared,
                    )
                except BaseException:
                    self._mark_translation_trust_failure()
                    raise
            except BaseException:
                if self._maintenance_phase != 'translation-trust-failed':
                    self._maintenance_phase = 'translation-prepared'
                    self._refresh_guard()
                raise

    def _arm_final_swap(self, world, prepared, finalized):
        with self._lock:
            try:
                self._require_prepared_composite(prepared)
                self._require_finalized_composite(finalized, prepared)
            except BaseException:
                self._mark_translation_trust_failure()
                raise
            result = a49b._A4ResidentTranslationOwner._arm_final_swap(
                self, world, prepared, finalized,
            )
            try:
                self._require_prepared_composite(prepared)
                self._require_finalized_composite(result, prepared)
            except BaseException:
                self._mark_translation_trust_failure()
                raise
            stable = a49a._stable_cpu_snapshot(
                world, self._config, require_cache_exact=True,
            )
            observation = _observe_cpu_maintenance(world, stable.state)
            if (not _same_maintenance_observation(
                    finalized._a49c_source, observation)
                    or self._maintenance_binding
                    is not finalized._a49c_old_binding):
                raise A4ResidentMaintenanceFoundationError(
                    'maintenance source changed before composite CAS arm'
                )
            return result

    def _commit_armed_swap_no_fail(self, finalized):
        # Every reference below was allocated and validated before receipt
        # annotation.  This method deliberately performs assignments only.
        with self._lock:
            old = finalized.old_arena_ref
            prepared = finalized.prepared_ref
            self._arena = finalized.new_arena_ref
            self._arena_seal = finalized.new_arena_seal_ref
            self._epochs = finalized.new_owner_epochs_ref
            self._last_observed = finalized.fresh_source_ref
            self._storage_generation = (
                finalized.new_arena_ref.storage_generation
            )
            self._lifecycle = COHERENT
            self._issued_lease_token = None
            self._active_lease_token = None
            self._active_lease_consumed = False
            self._guard = finalized.base_guard_ref
            self._tx_active = False
            self._tx_serial = finalized.final_transaction_serial
            self._tx_token = None
            self._tx_stage = a49b._TX_IDLE
            self._tx_world = None
            self._tx_prepared = None
            self._tx_consumed = False
            self._tx_guard = finalized.transaction_guard_ref
            self._tx_guard_snapshot = finalized.transaction_guard_snapshot
            self._maintenance_binding = finalized._a49c_new_binding
            self._maintenance_source = finalized._a49c_source
            self._maintenance_last_observed = finalized._a49c_source
            self._maintenance_phase = 'idle'
            self._maintenance_guard = finalized._a49c_guard
            self._maintenance_guard_snapshot = (
                finalized._a49c_guard_snapshot
            )
            prepared.consumed = True
            finalized.consumed = True
            old.retired = True
            old.lifecycle = INVALID
            old.invalid_reason = (
                'retired after A4.9c composite translation CAS'
            )
            old.active_leases = 0
            old.lease_serial = self._lease_serial + 1
        return None

    def _abort_preclaim(self, world, prepared=None):
        with self._lock:
            trust_failure = False
            if prepared is not None:
                try:
                    self._require_prepared_composite(prepared)
                except BaseException:
                    trust_failure = True
            result = a49b._A4ResidentTranslationOwner._abort_preclaim(
                self, world, prepared,
            )
            self._maintenance_phase = 'idle'
            stable = a49a._stable_cpu_snapshot(
                world, self._config, require_cache_exact=True,
            )
            observation = _observe_cpu_maintenance(world, stable.state)
            if not _same_maintenance_observation(
                    self._maintenance_source, observation):
                self._transition_invalid(
                    'maintenance source changed during preclaim abort',
                    ('S',), advance_unobserved=True,
                )
                raise A4ResidentMaintenanceFoundationError(
                    'preclaim abort could not preserve old composite'
                )
            if trust_failure:
                self._transition_invalid(
                    'prepared composite translation trust changed',
                    ('R', 'S', 'C'), advance_unobserved=True,
                )
                return result
            self._refresh_guard()
            return result

    def _rollback_after_claim(self, world, prepared):
        with self._lock:
            trust_failure = (
                self._maintenance_phase == 'translation-trust-failed'
            )
            try:
                self._require_prepared_composite(prepared)
            except BaseException:
                trust_failure = True
            result = (
                a49b._A4ResidentTranslationOwner._rollback_after_claim(
                    self, world, prepared,
                )
            )
            self._maintenance_phase = 'idle'
            stable = a49a._stable_cpu_snapshot(
                world, self._config, require_cache_exact=True,
            )
            observation = _observe_cpu_maintenance(world, stable.state)
            if not _same_maintenance_observation(
                    self._maintenance_source, observation):
                self._transition_invalid(
                    'maintenance source differs after translation rollback',
                    ('S',), advance_unobserved=True,
                )
                raise A4ResidentMaintenanceFoundationError(
                    'translation rollback did not preserve old composite'
                )
            if trust_failure:
                self._transition_invalid(
                    'finalized composite translation trust changed',
                    ('R', 'S', 'C'), advance_unobserved=True,
                )
                return result
            self._refresh_guard()
            return result

    def _open_read_lease(self, world):
        with self._lock:
            if self._maintenance_rebuild_candidate is not None:
                raise A4ResidentMaintenanceFoundationError(
                    'read lease is forbidden with a pending rebuild'
                )
            return a49b._A4ResidentTranslationOwner._open_read_lease(
                self, world,
            )

    def _validation_snapshot(self, world):
        with self._lock:
            if self._maintenance_rebuild_candidate is not None:
                raise A4ResidentMaintenanceFoundationError(
                    'validation snapshot is forbidden with a pending rebuild'
                )
            return a49b._A4ResidentTranslationOwner._validation_snapshot(
                self, world,
            )

    def invalidate(self, reason='explicit caller invalidation'):
        with self._lock:
            if self._maintenance_rebuild_candidate is not None:
                raise A4ResidentMaintenanceFoundationError(
                    'invalidation is forbidden with a pending rebuild'
                )
            return a49b._A4ResidentTranslationOwner.invalidate(
                self, reason,
            )

    def _consume_lease(self, lease):
        with self._lock:
            base = a49a.A4ResidentArenaOwner._consume_lease(self, lease)
            self._guard_or_invalidate()
            if self._maintenance_binding.translation is not base:
                raise A4ResidentMaintenanceFoundationError(
                    'read lease base/composite relation differs'
                )
            return self._maintenance_binding

    def __getstate__(self):
        raise A4ResidentMaintenanceFoundationError(
            'composite resident owners are not serializable authority'
        )


def _canonical_foundation_state(value):
    if not isinstance(value, dict):
        raise A4ResidentMaintenanceFoundationError(
            'maintenance foundation save metadata is malformed'
        )
    if set(value) != {
            'schema', 'config', 'device', 'foundation', 'port_status'}:
        raise A4ResidentMaintenanceFoundationError(
            'maintenance foundation save metadata keys differ'
        )
    if value.get('schema') != SCHEMA_VERSION:
        raise A4ResidentMaintenanceFoundationError(
            'maintenance foundation save schema differs'
        )
    config = a49a._canonical_config(value.get('config'))
    device = a49a._canonical_device(value.get('device'))
    if value.get('foundation') is not True:
        raise A4ResidentMaintenanceFoundationError(
            'maintenance foundation availability flag differs'
        )
    status = value.get('port_status')
    if (not isinstance(status, dict)
            or set(status) != set(PORT_STATUS)
            or status != PORT_STATUS):
        raise A4ResidentMaintenanceFoundationError(
            'maintenance foundation port status differs'
        )
    return {
        'schema': SCHEMA_VERSION,
        'config': dict(a49a._config_state(config)),
        'device': str(device),
        'foundation': True,
        'port_status': dict(PORT_STATUS),
    }


class A4ResidentMaintenanceFoundationScheduler(
        a49b.A4ResidentTranslationEventScheduler):
    """A4.9b scheduler whose private owner carries an immutable sidecar."""

    def resident_maintenance_foundation_state(self):
        self._require_runtime_creation_seals(
            self._a49b_world, self._a49b_owner,
        )
        return {
            'schema': SCHEMA_VERSION,
            'config': dict(self._a49b_scheduler_seal.config_values),
            'device': self._a49b_scheduler_seal.device,
            'foundation': True,
            'port_status': dict(PORT_STATUS),
        }

    def _resident_owner_for_rank5(self, world):
        self._require_runtime_creation_seals(world, self._a49b_owner)
        if self._a49b_world is not world:
            raise A4ResidentMaintenanceFoundationError(
                'active scheduler/world identity differs'
            )
        if (self._a49b_owner is not None
                and self._a49b_cells is not world.cells):
            self._a49b_owner.audit_cpu(world)
            raise A4ResidentMaintenanceFoundationError(
                'active world.cells container identity differs'
            )
        if self._a49b_owner is None and self._a49b_cells is not world.cells:
            raise A4ResidentMaintenanceFoundationError(
                'world.cells changed before first composite construction'
            )
        if self._a49b_owner is None:
            self._a49b_owner = (
                _A4ResidentMaintenanceFoundationOwner.from_cpu(
                    world, self._a49b_scheduler_seal.config_ref,
                    self._a49b_scheduler_seal.device,
                )
            )
        self._require_runtime_creation_seals(world, self._a49b_owner)
        self._a49b_owner._cohere_rank5(world)
        self._a49b_owner._preflight_translation_capacity(world)
        return self._a49b_owner

    def state_dict(self):
        if (self._a49b_owner is not None and (
                self._a49b_owner._maintenance_phase != 'idle'
                or self._a49b_owner._maintenance_rebuild_candidate
                is not None
                or self._a49b_owner._issued_lease_token is not None
                or self._a49b_owner._active_lease_token is not None)):
            raise A4ResidentMaintenanceFoundationError(
                'cannot serialize a pending composite operation'
            )
        state = super(
            A4ResidentMaintenanceFoundationScheduler, self,
        ).state_dict()
        state['a4_resident_maintenance_foundation'] = (
            self.resident_maintenance_foundation_state()
        )
        return state

    @classmethod
    def from_state(cls, state):
        state = dict(state or {})
        own = _canonical_foundation_state(
            state.get('a4_resident_maintenance_foundation'),
        )
        inherited = a49b.A4ResidentTranslationEventScheduler.from_state(
            state,
        )
        if (own['config'] != dict(a49a._config_state(inherited.a4_config))
                or own['device'] != inherited.a4_device):
            raise A4ResidentMaintenanceFoundationError(
                'saved A4.9b/A4.9c config or device differs'
            )
        scheduler = cls(
            a4_config=own['config'], device=own['device'],
            max_receipts=inherited.max_receipts,
        )
        scheduler._next_step_id = int(inherited._next_step_id)
        scheduler._backend_name = str(inherited._backend_name)
        scheduler._receipts = copy.deepcopy(inherited._receipts)
        return scheduler


class Hybrid066WorldA4ResidentMaintenanceFoundation(
        a49b.Hybrid066WorldA4ResidentTranslation):
    """A4.9b world with a shadow-only maintenance composite foundation."""

    def __init__(self, cpu_world, backend=None, scheduler=None,
                 gpu_config=None, a4_config=None, a4_device=None):
        if scheduler is None:
            scheduler = A4ResidentMaintenanceFoundationScheduler(
                a4_config=a4_config, device=a4_device,
            )
        elif not isinstance(
                scheduler, A4ResidentMaintenanceFoundationScheduler):
            raise TypeError(
                'scheduler must be A4ResidentMaintenanceFoundationScheduler'
            )
        super(Hybrid066WorldA4ResidentMaintenanceFoundation, self).__init__(
            cpu_world, backend=backend, scheduler=scheduler,
            gpu_config=gpu_config, a4_config=a4_config,
            a4_device=a4_device,
        )

    @classmethod
    def new(cls, seed=101, initial_cells=3, world_config=None,
            gpu_config=None, backend=None, a4_config=None, a4_device=None):
        if a4_config is None or a4_device is None:
            raise A4ResidentMaintenanceFoundationError(
                'new A4.9c world requires explicit A4 config and device'
            )
        config = (
            world_config
            if isinstance(world_config, a49b.a3s.a2.s66.Formal066Config)
            else a49b.a3s.a2.s66.Formal066Config.from_state(
                world_config or {},
            )
        )
        world = a49b.a3s.a2.s66.Formal066World(
            seed=seed, initial_cells=initial_cells, config=config,
        )
        return cls(
            world, backend=backend,
            gpu_config=gpu_config if backend is None else None,
            a4_config=a4_config, a4_device=a4_device,
        )

    def summary(self):
        integration = (
            self.scheduler.resident_maintenance_foundation_state()
        )
        output = super(
            Hybrid066WorldA4ResidentMaintenanceFoundation, self,
        ).summary()
        output.update({
            'gpu_build': BUILD,
            'gpu_schema': SCHEMA_VERSION,
            'gpu_port_status': dict(PORT_STATUS),
            'gpu_full_world_step': False,
            'a4_resident_maintenance_foundation': copy.deepcopy(
                integration,
            ),
        })
        return output

    def state_dict(self):
        state = super(
            Hybrid066WorldA4ResidentMaintenanceFoundation, self,
        ).state_dict()
        state.update({
            'save_version': SAVE_VERSION,
            'build': BUILD,
            'a4_resident_maintenance_foundation': (
                self.scheduler.resident_maintenance_foundation_state()
            ),
        })
        return state

    @classmethod
    def from_state(cls, state, backend=None, backend_factory=None):
        state = dict(state or {})
        if (state.get('save_version') != SAVE_VERSION
                or state.get('build') != BUILD):
            raise A4ResidentMaintenanceFoundationError(
                'A4.9c save version/build differs'
            )
        own = _canonical_foundation_state(
            state.get('a4_resident_maintenance_foundation'),
        )
        inherited_state = copy.deepcopy(state)
        inherited_state['save_version'] = a49b.SAVE_VERSION
        inherited_state['build'] = a49b.BUILD
        inherited = a49b.Hybrid066WorldA4ResidentTranslation.from_state(
            inherited_state, backend=backend,
            backend_factory=backend_factory,
        )
        scheduler = A4ResidentMaintenanceFoundationScheduler.from_state(
            state.get('scheduler', {}),
        )
        if scheduler.resident_maintenance_foundation_state() != own:
            raise A4ResidentMaintenanceFoundationError(
                'saved world/scheduler A4.9c settings differ'
            )
        return cls(
            inherited.world, backend=inherited.backend,
            scheduler=scheduler,
        )


__all__ = (
    'BUILD', 'BUILD_ID', 'BUILD_LONG', 'SCHEMA_VERSION',
    'SUPPLEMENT_SCHEMA_VERSION', 'SELECTED_PLAN_SCHEMA_VERSION',
    'SAVE_VERSION', 'FULL_GPU_WORLD_STEP', 'MAINTENANCE_ORACLE_ATOL',
    'PORT_STATUS', 'COHERENT', 'CPU_NEWER', 'INVALID',
    'A4ResidentMaintenanceFoundationError',
    'A4MaintenanceSupplementBatch', 'A4MaintenanceBinding',
    'A4SelectedMaintenancePlan', 'pack_a4_maintenance_supplement',
    'bind_a4_maintenance', 'paid_maintenance_selected_numpy',
    'paid_maintenance_selected_torch', 'paid_maintenance_selected',
    'A4ResidentMaintenanceFoundationScheduler',
    'Hybrid066WorldA4ResidentMaintenanceFoundation',
)
