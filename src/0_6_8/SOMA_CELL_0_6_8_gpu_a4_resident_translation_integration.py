# coding: utf-8
"""A4.9b selected paid-translation resident transaction.

This module deliberately leaves the promoted A3 implementation, the A4 pure
core, every A4.8 integration slice, and the immutable A4.9a arena unchanged.
The public selected plans are pure descriptors.  Only the private scheduler
path below may turn a freshly regenerated descriptor into one CPU publish and
one resident-generation swap.
"""
from __future__ import division

import copy
import hashlib
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

import SOMA_CELL_0_6_8_gpu_a3_scheduler as a3s
import SOMA_CELL_0_6_8_gpu_a4 as a4
import SOMA_CELL_0_6_8_gpu_a4_integration as a48a
import SOMA_CELL_0_6_8_gpu_a4_translation_integration as a48b
import SOMA_CELL_0_6_8_gpu_a4_replication_early_noop_integration as a48c8
import SOMA_CELL_0_6_8_gpu_a4_resident_arena as a49a

try:
    import torch
except Exception:  # pragma: no cover - NumPy descriptor remains importable
    torch = None


BUILD = 'SOMA-CELL 0.6.8-GPU A4.9b'
BUILD_ID = BUILD
BUILD_LONG = (
    BUILD + ' | persistent-arena RNG-free selected paid-translation transaction'
)
SCHEMA_VERSION = '0.6.8-GPU-A4.9b-resident-selected-translation'
SELECTED_PLAN_SCHEMA_VERSION = SCHEMA_VERSION + '-plan'
SAVE_VERSION = 1
FULL_GPU_WORLD_STEP = False
TRANSLATION_ORACLE_ATOL = a48b.TRANSLATION_ORACLE_ATOL

COHERENT = a49a.COHERENT
CPU_NEWER = a49a.CPU_NEWER
INVALID = a49a.INVALID

_SELECTED_PLAN_TOKEN = object()
_SELECTED_WRITE_ARRAYS = frozenset((
    'pools', 'last_translation', 'last_quiescence',
    'active_fingerprints', 'active_mass', 'active_count',
    'damaged_fingerprints', 'damaged_mass', 'damaged_count',
))


class A4ResidentTranslationCommitError(a49a.A4ResidentArenaError):
    """The bounded A4.9b selected transaction failed closed."""


def _strict_target_index(value, state):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))):
        raise A4ResidentTranslationCommitError(
            'selected translation target_index must be an integer'
        )
    result = int(value)
    if result < 0 or result >= int(state.cell_count):
        raise A4ResidentTranslationCommitError(
            'selected translation target_index is not one used row'
        )
    return result


def _strict_target_cell_id(value):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))):
        raise A4ResidentTranslationCommitError(
            'selected translation target_cell_id must be an integer'
        )
    result = int(value)
    if result < 0:
        raise A4ResidentTranslationCommitError(
            'selected translation target_cell_id must be nonnegative'
        )
    return result


def _selected_plan_metadata(plan):
    return (
        str(plan.schema_version), int(plan.target_index),
        int(plan.target_cell_id), str(plan.dt_hex),
        str(plan.source_provenance),
        a4._translation_scalar_metadata(plan.state_after),
    )


def _array_digest(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode('ascii'))
    digest.update(repr(tuple(array.shape)).encode('ascii'))
    digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


@dataclass(frozen=True, init=False)
class A4SelectedTranslationPlan:
    """Pure full-capacity state with exactly one paid-translation row selected."""

    schema_version: str
    target_index: int
    target_cell_id: int
    dt_hex: str
    target_mask: object
    source_provenance: str
    state_after: a4.A4TranslationStateBatch

    def __init__(self, *args, **kwargs):
        raise A4ResidentTranslationCommitError(
            'A4SelectedTranslationPlan is created only by its pure factories'
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
        mask = (
            self.target_mask.detach().cpu().numpy().astype(bool, copy=True)
            if a4._is_tensor(self.target_mask)
            else np.asarray(self.target_mask, dtype=bool).copy()
        )
        return _make_selected_plan(
            int(self.target_index), int(self.target_cell_id),
            float.fromhex(self.dt_hex), mask, state,
            source_binding=None, diagnostic=True,
        )

    def data_ptrs(self):
        _require_selected_plan(self, diagnostic=False)
        if not a4._is_tensor(self.target_mask):
            raise TypeError('data_ptrs requires a Torch selected plan')
        output = {'target_mask': int(self.target_mask.data_ptr())}
        output.update({
            'state_after.' + name: pointer
            for name, pointer in self.state_after.data_ptrs().items()
        })
        return output

    def __getstate__(self):
        raise A4ResidentTranslationCommitError(
            'selected translation plans are not serializable authority'
        )


def _make_selected_plan(
        target_index, target_cell_id, dt, target_mask, state_after,
        source_binding, diagnostic=False):
    plan = object.__new__(A4SelectedTranslationPlan)
    object.__setattr__(plan, 'schema_version', SELECTED_PLAN_SCHEMA_VERSION)
    object.__setattr__(plan, 'target_index', int(target_index))
    object.__setattr__(plan, 'target_cell_id', int(target_cell_id))
    object.__setattr__(plan, 'dt_hex', float(dt).hex())
    object.__setattr__(plan, 'target_mask', target_mask)
    object.__setattr__(
        plan, 'source_provenance', str(state_after.source_provenance),
    )
    object.__setattr__(plan, 'state_after', state_after)
    object.__setattr__(plan, '_factory_token', _SELECTED_PLAN_TOKEN)
    object.__setattr__(plan, '_metadata', _selected_plan_metadata(plan))
    object.__setattr__(plan, '_source_binding', source_binding)
    object.__setattr__(plan, '_target_mask_object_id', id(target_mask))
    object.__setattr__(plan, '_state_object_id', id(state_after))
    if a4._is_tensor(state_after.pools):
        host_mask = target_mask.detach().cpu().numpy()
        host_state = state_after.to_numpy()
        object.__setattr__(
            plan, '_target_mask_ptr', int(target_mask.data_ptr()),
        )
        object.__setattr__(
            plan, '_target_mask_version', int(target_mask._version),
        )
        object.__setattr__(
            plan, '_target_mask_digest', _array_digest(host_mask),
        )
        object.__setattr__(plan, '_state_ptrs', state_after.data_ptrs())
        object.__setattr__(
            plan, '_state_versions', {
                name: int(getattr(state_after, name)._version)
                for name in a4._TRANSLATION_ARRAY_FIELDS
            },
        )
        object.__setattr__(
            plan, '_state_digest',
            a4._translation_state_provenance(host_state),
        )
    else:
        object.__setattr__(plan, '_target_mask_ptr', None)
        object.__setattr__(plan, '_target_mask_version', None)
        object.__setattr__(
            plan, '_target_mask_digest', _array_digest(target_mask),
        )
        object.__setattr__(plan, '_state_ptrs', None)
        object.__setattr__(plan, '_state_versions', None)
        object.__setattr__(
            plan, '_state_digest', a4._translation_state_provenance(state_after),
        )
    if diagnostic:
        _require_selected_plan(plan, diagnostic=True)
    return plan


def _require_selected_plan(plan, diagnostic=False):
    if (not isinstance(plan, A4SelectedTranslationPlan)
            or getattr(plan, '_factory_token', None) is not _SELECTED_PLAN_TOKEN):
        raise A4ResidentTranslationCommitError(
            'untrusted selected translation plan'
        )
    state = plan.state_after
    index = _strict_target_index(plan.target_index, state)
    cell_id = _strict_target_cell_id(plan.target_cell_id)
    try:
        dt = float.fromhex(str(plan.dt_hex))
    except Exception as exc:
        raise A4ResidentTranslationCommitError(
            'selected translation dt_hex is invalid'
        ) from exc
    if not math.isfinite(dt) or dt < 0.0 or dt.hex() != plan.dt_hex:
        raise A4ResidentTranslationCommitError(
            'selected translation dt_hex is noncanonical'
        )
    if (plan.schema_version != SELECTED_PLAN_SCHEMA_VERSION
            or id(plan.target_mask) != getattr(
                plan, '_target_mask_object_id', None)
            or id(state) != getattr(plan, '_state_object_id', None)
            or _selected_plan_metadata(plan) != getattr(plan, '_metadata', None)
            or str(plan.source_provenance) != str(state.source_provenance)):
        raise A4ResidentTranslationCommitError(
            'selected translation immutable schema metadata changed'
        )

    tensor_backed = a4._is_tensor(state.pools)
    if tensor_backed != a4._is_tensor(plan.target_mask):
        raise A4ResidentTranslationCommitError(
            'selected translation plan mixes NumPy and Torch authority'
        )
    if tensor_backed:
        a4._validate_translation_resident_metadata(state)
        if (plan.target_mask.dtype != torch.bool
                or tuple(plan.target_mask.shape) != (int(state.cell_capacity),)
                or str(plan.target_mask.device) != str(state.pools.device)
                or int(plan.target_mask.data_ptr())
                != getattr(plan, '_target_mask_ptr', None)
                or int(plan.target_mask._version)
                != getattr(plan, '_target_mask_version', None)
                or state.data_ptrs() != getattr(plan, '_state_ptrs', None)
                or {
                    name: int(getattr(state, name)._version)
                    for name in a4._TRANSLATION_ARRAY_FIELDS
                } != getattr(plan, '_state_versions', None)):
            raise A4ResidentTranslationCommitError(
                'selected translation resident pointer/version seal changed'
            )
        if any(int(getattr(state, name)._version) != 0
               for name in a4._TRANSLATION_ARRAY_FIELDS):
            raise A4ResidentTranslationCommitError(
                'selected translation final state tensors must have version zero'
            )
        if diagnostic:
            host_mask = plan.target_mask.detach().cpu().numpy()
            host_state = state.to_numpy()
            if (_array_digest(host_mask)
                    != getattr(plan, '_target_mask_digest', None)
                    or a4._translation_state_provenance(host_state)
                    != getattr(plan, '_state_digest', None)):
                raise A4ResidentTranslationCommitError(
                    'selected translation resident content seal changed'
                )
        else:
            host_mask = None
            host_state = None
    else:
        a4.validate_a4_translation_state(state)
        host_mask = np.asarray(plan.target_mask)
        host_state = state
        if (_array_digest(host_mask)
                != getattr(plan, '_target_mask_digest', None)
                or a4._translation_state_provenance(host_state)
                != getattr(plan, '_state_digest', None)):
            raise A4ResidentTranslationCommitError(
                'selected translation host content seal changed'
            )

    if host_mask is not None:
        expected = np.zeros((int(state.cell_capacity),), dtype=bool)
        expected[index] = True
        if (host_mask.dtype != np.dtype(bool)
                or host_mask.shape != expected.shape
                or not np.array_equal(host_mask, expected)
                or int(host_state.cell_ids[index]) != cell_id
                or not bool(host_state.cell_mask[index])):
            raise A4ResidentTranslationCommitError(
                'selected translation target mask/index/cell ID differs'
            )
    source_binding = getattr(plan, '_source_binding', None)
    if source_binding is not None:
        try:
            a4._require_translation_binding(source_binding)
        except Exception as exc:
            raise A4ResidentTranslationCommitError(
                'selected translation source binding changed'
            ) from exc
    return plan


def _fresh_selected_state(source, full_plan, target_index):
    values = {}
    tensor_backed = a4._is_tensor(source.pools)
    C = int(source.cell_capacity)
    if tensor_backed:
        selector = torch.arange(
            C, dtype=torch.int64, device=source.pools.device,
        ) == int(target_index)
    else:
        selector = np.arange(C, dtype=np.int64) == int(target_index)
    for item in fields(a4.A4TranslationStateBatch):
        name = item.name
        value = getattr(source, name)
        if name not in a4._TRANSLATION_ARRAY_FIELDS:
            values[name] = copy.deepcopy(value)
            continue
        if tensor_backed:
            if name in _SELECTED_WRITE_ARRAYS:
                shape = (C,) + (1,) * (value.ndim - 1)
                mask = selector.reshape(shape)
                output = torch.where(mask, getattr(full_plan, name), value)
            else:
                output = value.clone()
            if int(output._version) != 0:
                raise A4ResidentTranslationCommitError(
                    'fresh selected Torch state did not retain version zero'
                )
            values[name] = output
        else:
            output = np.asarray(value).copy()
            if name in _SELECTED_WRITE_ARRAYS:
                output[int(target_index)] = np.asarray(
                    getattr(full_plan, name)[int(target_index)]
                )
            values[name] = output
    state = a4.A4TranslationStateBatch(**values)
    if tensor_backed:
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
    else:
        a4.validate_a4_translation_state(state)
    return state, selector


def _selected_target_preflight(binding, target_index, target_cell_id):
    a4._require_translation_binding(binding)
    state = binding.state
    index = _strict_target_index(target_index, state)
    cell_id = _strict_target_cell_id(target_cell_id)
    if a4._is_tensor(state.cell_ids):
        # Correctness boundary: public target identity fails before a
        # descriptor is made.  No performance claim is attached to this sync.
        observed = int(state.cell_ids[index].detach().cpu().item())
        used = bool(state.cell_mask[index].detach().cpu().item())
    else:
        observed = int(state.cell_ids[index])
        used = bool(state.cell_mask[index])
    if observed != cell_id or not used:
        raise A4ResidentTranslationCommitError(
            'selected translation target index/cell ID differs from binding'
        )
    return index, cell_id


def paid_translation_selected_numpy(
        binding, dt, target_index, target_cell_id):
    """Return a pure NumPy plan whose only translated row is the target."""
    a4._require_translation_binding(binding)
    if a4._is_tensor(binding.state.pools):
        raise A4ResidentTranslationCommitError(
            'NumPy selected translation requires a NumPy binding'
        )
    index, cell_id = _selected_target_preflight(
        binding, target_index, target_cell_id,
    )
    dt = a4._translation_dt(dt)
    full_plan = a4.paid_translation_plan_numpy(binding, dt)
    state_after, target_mask = _fresh_selected_state(
        binding.state, full_plan, index,
    )
    a4._require_translation_binding(binding)
    return _make_selected_plan(
        index, cell_id, dt, target_mask, state_after,
        source_binding=binding, diagnostic=True,
    )


def paid_translation_selected_torch(
        binding, dt, target_index, target_cell_id):
    """Return a resident pure plan with 30 fresh version-zero state tensors."""
    if torch is None:
        raise RuntimeError('PyTorch is unavailable')
    a4._require_translation_binding(binding)
    if not a4._is_tensor(binding.state.pools):
        raise A4ResidentTranslationCommitError(
            'Torch selected translation requires a Torch binding'
        )
    index, cell_id = _selected_target_preflight(
        binding, target_index, target_cell_id,
    )
    dt = a4._translation_dt(dt)
    full_plan = a4.paid_translation_plan_torch(binding, dt)
    state_after, target_mask = _fresh_selected_state(
        binding.state, full_plan, index,
    )
    a4._require_translation_binding(binding)
    return _make_selected_plan(
        index, cell_id, dt, target_mask, state_after,
        source_binding=binding, diagnostic=False,
    )


def paid_translation_selected(binding, dt, target_index, target_cell_id):
    """Dispatch the selected plan without accepting external commit authority."""
    a4._require_translation_binding(binding)
    if a4._is_tensor(binding.state.pools):
        return paid_translation_selected_torch(
            binding, dt, target_index, target_cell_id,
        )
    return paid_translation_selected_numpy(
        binding, dt, target_index, target_cell_id,
    )


PORT_STATUS = dict(a49a.PORT_STATUS)
PORT_STATUS.update({
    'resident_world_arena': 'a4.9b-selected-translation-fresh-generation',
    'resident_biology_authority': 'cpu-durable-after-atomic-selected-publish',
    'resident_device_writes': 'a4.9b-rng-free-paid-translation-only',
    'resident_d2h_publish': 'a4.9b-one-selected-translation-row-only',
    'full_gpu_world_step': False,
})


_TRANSACTION_LEASE_TOKEN = object()
_PREPARED_STORAGE_TOKEN = object()
_TX_IDLE = 'IDLE'
_TX_ISSUED = 'ISSUED'
_TX_OPEN = 'OPEN'
_TX_CONSUMED = 'CONSUMED'
_TX_PREPARED = 'PREPARED'


def _pickle_digest(value):
    try:
        payload = pickle.dumps(
            copy.deepcopy(value), protocol=pickle.HIGHEST_PROTOCOL,
        )
    except Exception as exc:
        raise A4ResidentTranslationCommitError(
            'A4.9b private value is not snapshot-safe'
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _signature_values(signature):
    return (
        tuple(signature.binding_object_ids),
        tuple(signature.tensor_records),
    )


def _plan_seal_values(plan):
    return (
        id(plan), id(plan.target_mask), id(plan.state_after),
        tuple(getattr(plan, '_metadata', ())),
        id(getattr(plan, '_source_binding', None)),
        getattr(plan, '_target_mask_ptr', None),
        getattr(plan, '_target_mask_version', None),
        str(getattr(plan, '_target_mask_digest', '')),
        tuple(sorted((getattr(plan, '_state_ptrs', None) or {}).items())),
        tuple(sorted((getattr(plan, '_state_versions', None) or {}).items())),
        str(getattr(plan, '_state_digest', '')),
    )


def _arrays_bit_exact(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    if left.dtype == np.dtype(np.float64):
        return np.array_equal(left.view(np.uint64), right.view(np.uint64))
    return np.array_equal(left, right)


def _selected_states_match(resident, oracle, target_index):
    a4.validate_a4_translation_state(resident)
    a4.validate_a4_translation_state(oracle)
    index = int(target_index)
    for item in fields(a4.A4TranslationStateBatch):
        name = item.name
        left = getattr(resident, name)
        right = getattr(oracle, name)
        if name not in a4._TRANSLATION_ARRAY_FIELDS:
            if left != right:
                raise A4ResidentTranslationCommitError(
                    'selected resident/NumPy scalar metadata differs: %s' % name
                )
            continue
        left = np.asarray(left)
        right = np.asarray(right)
        if left.dtype != right.dtype or left.shape != right.shape:
            raise A4ResidentTranslationCommitError(
                'selected resident/NumPy array schema differs: %s' % name
            )
        if left.dtype == np.dtype(np.float64):
            if (not np.isfinite(left).all() or not np.isfinite(right).all()
                    or np.any(np.abs(left[index] - right[index])
                              > TRANSLATION_ORACLE_ATOL)
                    or not _arrays_bit_exact(
                        np.delete(left, index, axis=0),
                        np.delete(right, index, axis=0),
                    )):
                raise A4ResidentTranslationCommitError(
                    'selected resident/NumPy fp64 result differs: %s' % name
                )
        elif not np.array_equal(left, right):
            raise A4ResidentTranslationCommitError(
                'selected resident/NumPy discrete result differs: %s' % name
            )
    return resident


def _selected_output_scope(source, result, target_index):
    a4.validate_a4_translation_state(source)
    a4.validate_a4_translation_state(result)
    index = int(target_index)
    for name in a4._TRANSLATION_ARRAY_FIELDS:
        before = np.asarray(getattr(source, name))
        after = np.asarray(getattr(result, name))
        if before.dtype != after.dtype or before.shape != after.shape:
            raise A4ResidentTranslationCommitError(
                'selected output schema differs: %s' % name
            )
        if name not in _SELECTED_WRITE_ARRAYS:
            if not _arrays_bit_exact(before, after):
                raise A4ResidentTranslationCommitError(
                    'selected output changed an excluded array: %s' % name
                )
        else:
            keep = np.arange(before.shape[0]) != index
            if not _arrays_bit_exact(before[keep], after[keep]):
                raise A4ResidentTranslationCommitError(
                    'selected output changed a non-target row: %s' % name
                )
    paid = set(int(value) for value in a4.TRANSLATION_PAID_POOL_INDICES)
    unchanged = [
        position for position in range(int(a4.a3.POOL_COUNT))
        if position not in paid
        and position not in (
            int(a4.a3.POOL_CATALYST),
            int(a4.a3.POOL_DAMAGED_PROTEIN),
        )
    ]
    if not _arrays_bit_exact(
            np.asarray(source.pools)[index, unchanged],
            np.asarray(result.pools)[index, unchanged]):
        raise A4ResidentTranslationCommitError(
            'selected translation changed an out-of-scope target pool'
        )
    return result


def _dict_from_selected_row(state, index, prefix):
    count = int(getattr(state, prefix + '_count')[index])
    fingerprints = np.asarray(
        getattr(state, prefix + '_fingerprints'), dtype=np.int64,
    )[index, :count]
    masses = np.asarray(
        getattr(state, prefix + '_mass'), dtype=np.float64,
    )[index, :count]
    return {
        int(fingerprint): float(amount)
        for fingerprint, amount in zip(fingerprints, masses)
    }


@dataclass(frozen=True)
class _PreparedStorageSeal:
    prepared_ref: object
    owner_token_ref: object
    transaction_token_ref: object
    values: tuple


@dataclass
class _PreparedTranslationStorage:
    _factory_token: object
    _owner_token: object
    _transaction_token: object
    _world_ref: object
    _cells_ref: object
    _cell_ref: object
    target_index: int
    target_cell_id: int
    target_generation: int
    dt_hex: str
    old_arena_ref: object
    old_arena_id: str
    old_storage_generation: int
    old_epochs: object
    old_source_ref: object
    old_source_values: tuple
    selected_plan: object
    numpy_oracle: object
    candidate_binding: object
    candidate_signature: object
    candidate_digests: tuple
    diagnostic_ragged: object
    diagnostic_state: object
    diagnostic_cache: object
    expected_arena_id: str
    expected_storage_generation: int
    expected_epochs: object
    sync_performed: bool
    prepared_pools: object
    prepared_proteins: object
    prepared_damaged: object
    prepared_last_translation: object
    prepared_last_quiescence: object
    source_cell_snapshot: bytes
    source_gene_specs: object
    live_rng: object
    rng_state: object
    dissipated_energy: object
    world_config_ref: object
    world_config_snapshot: object
    integration_state: object
    seal: object = None
    consumed: bool = False

    def __getstate__(self):
        raise A4ResidentTranslationCommitError(
            'prepared translation storage is not serializable authority'
        )


@dataclass(frozen=True)
class _FinalizedSwapSeal:
    finalized_ref: object
    values: tuple


@dataclass
class _FinalizedTranslationSwap:
    _factory_token: object
    owner_ref: object
    prepared_ref: object
    old_arena_ref: object
    old_arena_seal_ref: object
    new_arena_ref: object
    new_arena_seal_ref: object
    new_owner_epochs_ref: object
    fresh_source_ref: object
    base_guard_ref: object
    transaction_guard_ref: object
    transaction_guard_snapshot: tuple
    final_transaction_serial: int
    retired_reason: str
    seal: object = None
    consumed: bool = False

    def __getstate__(self):
        raise A4ResidentTranslationCommitError(
            'finalized translation swaps are not serializable'
        )


_FINALIZED_SWAP_TOKEN = object()


def _finalized_swap_values(finalized):
    return (
        id(finalized), id(finalized.owner_ref), id(finalized.prepared_ref),
        id(finalized.old_arena_ref), id(finalized.old_arena_seal_ref),
        a49a._arena_creation_seal_values(finalized.old_arena_seal_ref),
        id(finalized.new_arena_ref), id(finalized.new_arena_seal_ref),
        a49a._arena_creation_seal_values(finalized.new_arena_seal_ref),
        id(finalized.new_owner_epochs_ref),
        a49a._epoch_values(finalized.new_owner_epochs_ref),
        id(finalized.fresh_source_ref),
        a49a._source_identity_values(finalized.fresh_source_ref),
        id(finalized.base_guard_ref),
        _base_guard_values(finalized.base_guard_ref),
        id(finalized.transaction_guard_ref),
        tuple(finalized.transaction_guard_snapshot),
        int(finalized.final_transaction_serial),
        str(finalized.retired_reason), bool(finalized.consumed),
    )


def _require_finalized_swap(finalized, owner, prepared):
    seal = getattr(finalized, 'seal', None)
    if (not isinstance(finalized, _FinalizedTranslationSwap)
            or finalized._factory_token is not _FINALIZED_SWAP_TOKEN
            or finalized.owner_ref is not owner
            or finalized.prepared_ref is not prepared
            or not isinstance(seal, _FinalizedSwapSeal)
            or seal.finalized_ref is not finalized
            or _finalized_swap_values(finalized) != seal.values
            or finalized.consumed):
        raise A4ResidentTranslationCommitError(
            'finalized translation swap seal differs'
        )
    return finalized


def _prepared_storage_values(prepared):
    return (
        id(prepared), id(prepared._owner_token),
        id(prepared._transaction_token), id(prepared._world_ref),
        id(prepared._cells_ref), id(prepared._cell_ref),
        int(prepared.target_index), int(prepared.target_cell_id),
        int(prepared.target_generation), str(prepared.dt_hex),
        id(prepared.old_arena_ref), str(prepared.old_arena_id),
        int(prepared.old_storage_generation),
        id(prepared.old_epochs),
        a49a._epoch_values(prepared.old_epochs),
        id(prepared.old_source_ref), tuple(prepared.old_source_values),
        _plan_seal_values(prepared.selected_plan),
        _plan_seal_values(prepared.numpy_oracle),
        id(prepared.candidate_binding),
        id(prepared.candidate_binding.ragged),
        id(prepared.candidate_binding.state),
        id(prepared.candidate_binding.cache),
        id(prepared.candidate_signature),
        _signature_values(prepared.candidate_signature),
        tuple(prepared.candidate_digests),
        id(prepared.diagnostic_ragged),
        a4._ragged_translation_provenance(prepared.diagnostic_ragged),
        id(prepared.diagnostic_state),
        a4._translation_state_provenance(prepared.diagnostic_state),
        id(prepared.diagnostic_cache),
        a4._gene_cache_provenance(prepared.diagnostic_cache),
        str(prepared.expected_arena_id),
        int(prepared.expected_storage_generation),
        id(prepared.expected_epochs),
        a49a._epoch_values(prepared.expected_epochs),
        bool(prepared.sync_performed), id(prepared.prepared_pools),
        _array_digest(prepared.prepared_pools),
        id(prepared.prepared_proteins),
        tuple(prepared.prepared_proteins.items()),
        id(prepared.prepared_damaged),
        tuple(prepared.prepared_damaged.items()),
        id(prepared.prepared_last_translation),
        float(prepared.prepared_last_translation).hex(),
        id(prepared.prepared_last_quiescence),
        float(prepared.prepared_last_quiescence).hex(),
        id(prepared.source_cell_snapshot),
        hashlib.sha256(prepared.source_cell_snapshot).hexdigest(),
        id(prepared.source_gene_specs),
        _pickle_digest(prepared.source_gene_specs), id(prepared.live_rng),
        _pickle_digest(prepared.rng_state), id(prepared.dissipated_energy),
        float(prepared.dissipated_energy).hex(),
        id(prepared.world_config_ref), id(prepared.world_config_snapshot),
        _pickle_digest(prepared.world_config_snapshot),
        id(prepared.integration_state),
        _pickle_digest(prepared.integration_state), bool(prepared.consumed),
    )


def _require_prepared_storage(prepared, owner, transaction_token=None):
    seal = getattr(prepared, 'seal', None)
    if (not isinstance(prepared, _PreparedTranslationStorage)
            or prepared._factory_token is not _PREPARED_STORAGE_TOKEN
            or prepared._owner_token is not owner._owner_token
            or not isinstance(seal, _PreparedStorageSeal)
            or seal.prepared_ref is not prepared
            or seal.owner_token_ref is not owner._owner_token
            or (transaction_token is not None
                and seal.transaction_token_ref is not transaction_token)
            or prepared._transaction_token is not seal.transaction_token_ref
            or _prepared_storage_values(prepared) != seal.values):
        raise A4ResidentTranslationCommitError(
            'prepared translation storage seal differs'
        )
    return prepared


def _base_guard_values(guard):
    if not isinstance(guard, a49a._OwnerCanonicalGuard):
        raise A4ResidentTranslationCommitError(
            'A4.9b base owner guard is malformed'
        )
    return (
        id(guard.owner_token_ref), id(guard.arena_ref),
        a49a._arena_creation_seal_values(guard.arena_seal_snapshot),
        id(guard.config_ref), id(guard.epochs_ref),
        id(guard.last_observed_ref), tuple(guard.last_observed_values),
        str(guard.lifecycle), tuple(guard.epochs_values), str(guard.device),
        tuple(guard.config_state), int(guard.storage_generation),
        int(guard.lease_serial), id(guard.issued_lease_token),
        id(guard.active_lease_token), bool(guard.active_lease_consumed),
    )


@dataclass(frozen=True)
class _TranslationOwnerGuard:
    owner_token_ref: object
    base_guard_ref: object
    base_guard_values: tuple
    transaction_active: bool
    transaction_serial: int
    transaction_token_ref: object
    transaction_stage: str
    transaction_world_ref: object
    transaction_consumed: bool
    prepared_ref: object
    prepared_seal_ref: object
    prepared_values: object


def _transaction_guard_values(guard):
    if not isinstance(guard, _TranslationOwnerGuard):
        raise A4ResidentTranslationCommitError(
            'A4.9b transaction guard is malformed'
        )
    return (
        id(guard.owner_token_ref), id(guard.base_guard_ref),
        tuple(guard.base_guard_values), bool(guard.transaction_active),
        int(guard.transaction_serial), id(guard.transaction_token_ref),
        str(guard.transaction_stage), id(guard.transaction_world_ref),
        bool(guard.transaction_consumed), id(guard.prepared_ref),
        id(guard.prepared_seal_ref),
        None if guard.prepared_values is None
        else tuple(guard.prepared_values),
    )


class _A4ResidentTranslationLease:
    __slots__ = (
        '_factory_token', '_owner', '_owner_token', '_world', '_serial',
        '_token', '_opened', '_consumed', '_closed',
    )

    def __init__(self, owner, world, serial, token):
        self._factory_token = _TRANSACTION_LEASE_TOKEN
        self._owner = owner
        self._owner_token = owner._owner_token
        self._world = world
        self._serial = int(serial)
        self._token = token
        self._opened = False
        self._consumed = False
        self._closed = False

    def __enter__(self):
        if self._opened or self._closed:
            raise A4ResidentTranslationCommitError(
                'translation source lease is one-use'
            )
        self._owner._activate_translation_lease(self)
        self._opened = True
        return self

    def _binding_once(self):
        if not self._opened or self._closed or self._consumed:
            raise A4ResidentTranslationCommitError(
                'translation source lease is inactive or consumed'
            )
        binding = self._owner._consume_translation_lease(self)
        self._consumed = True
        return binding

    def _register(self, prepared):
        self._owner._register_translation_storage(self, prepared)

    def __exit__(self, exc_type, exc_value, traceback):
        if self._closed:
            raise A4ResidentTranslationCommitError(
                'translation source lease was closed twice'
            )
        try:
            self._owner._close_translation_lease(self, exc_type is None)
        finally:
            self._closed = True
        return False

    def __getstate__(self):
        raise A4ResidentTranslationCommitError(
            'translation source leases are not serializable'
        )


class _A4ResidentTranslationOwner(a49a.A4ResidentArenaOwner):
    """A4.9a owner with one guarded selected-translation transaction."""

    __slots__ = (
        '_tx_initialized', '_tx_active', '_tx_serial', '_tx_token',
        '_tx_stage', '_tx_world', '_tx_prepared', '_tx_consumed',
        '_tx_guard', '_tx_guard_snapshot',
    )

    @classmethod
    def from_cpu(cls, world, a4_config, device):
        owner = a49a.A4ResidentArenaOwner.from_cpu.__func__(
            cls, world, a4_config, device,
        )
        owner._tx_initialized = True
        owner._tx_active = False
        owner._tx_serial = 0
        owner._tx_token = None
        owner._tx_stage = _TX_IDLE
        owner._tx_world = None
        owner._tx_prepared = None
        owner._tx_consumed = False
        owner._tx_guard = None
        owner._tx_guard_snapshot = None
        owner._refresh_guard()
        return owner

    def _make_transaction_guard(self, base_guard=None):
        base_guard = self._guard if base_guard is None else base_guard
        prepared = self._tx_prepared
        seal = getattr(prepared, 'seal', None) if prepared is not None else None
        return _TranslationOwnerGuard(
            owner_token_ref=self._owner_token,
            base_guard_ref=base_guard,
            base_guard_values=_base_guard_values(base_guard),
            transaction_active=bool(self._tx_active),
            transaction_serial=int(self._tx_serial),
            transaction_token_ref=self._tx_token,
            transaction_stage=str(self._tx_stage),
            transaction_world_ref=self._tx_world,
            transaction_consumed=bool(self._tx_consumed),
            prepared_ref=prepared,
            prepared_seal_ref=seal,
            prepared_values=(
                None if prepared is None
                else tuple(_prepared_storage_values(prepared))
            ),
        )

    def _refresh_guard(self):
        a49a.A4ResidentArenaOwner._refresh_guard(self)
        if getattr(self, '_tx_initialized', False):
            self._tx_guard = self._make_transaction_guard()
            self._tx_guard_snapshot = _transaction_guard_values(
                self._tx_guard,
            )

    def _require_guard(self):
        base = a49a.A4ResidentArenaOwner._require_guard(self)
        if not getattr(self, '_tx_initialized', False):
            return base
        guard = self._tx_guard
        prepared = self._tx_prepared
        seal = getattr(prepared, 'seal', None) if prepared is not None else None
        # Candidate content is checked at the explicit prepared-storage
        # boundary.  The owner guard seals only its identity and the original
        # detached candidate seal, so corrupt disposable storage can be
        # discarded without needlessly invalidating a trustworthy old arena.
        values = (
            None if prepared is None or not isinstance(
                seal, _PreparedStorageSeal)
            else tuple(seal.values)
        )
        if (not isinstance(guard, _TranslationOwnerGuard)
                or guard.owner_token_ref is not self._owner_token
                or _transaction_guard_values(guard)
                != tuple(self._tx_guard_snapshot)
                or _base_guard_values(self._guard) != guard.base_guard_values
                or bool(self._tx_active) != guard.transaction_active
                or int(self._tx_serial) != guard.transaction_serial
                or self._tx_token is not guard.transaction_token_ref
                or str(self._tx_stage) != guard.transaction_stage
                or self._tx_world is not guard.transaction_world_ref
                or bool(self._tx_consumed) != guard.transaction_consumed
                or prepared is not guard.prepared_ref
                or seal is not guard.prepared_seal_ref
                or values != guard.prepared_values
                or (not self._tx_active and (
                    self._tx_token is not None
                    or self._tx_stage != _TX_IDLE
                    or self._tx_world is not None
                    or self._tx_prepared is not None
                    or self._tx_consumed))
                or (self._tx_active and self._tx_token is None)):
            raise A4ResidentTranslationCommitError(
                'A4.9b outer transaction guard differs'
            )
        return base

    def _neutralize_transaction(self):
        self._tx_active = False
        self._tx_serial = int(getattr(self, '_tx_serial', 0)) + 1
        self._tx_token = None
        self._tx_stage = _TX_IDLE
        self._tx_world = None
        self._tx_prepared = None
        self._tx_consumed = False

    def _force_metadata_invalid(self, reason):
        if getattr(self, '_tx_initialized', False):
            self._neutralize_transaction()
        return a49a.A4ResidentArenaOwner._force_metadata_invalid(self, reason)

    def _transition_invalid(
            self, reason, domains=('M', 'R', 'S', 'C'),
            advance_unobserved=False):
        if getattr(self, '_tx_initialized', False):
            self._neutralize_transaction()
        return a49a.A4ResidentArenaOwner._transition_invalid(
            self, reason, domains, advance_unobserved,
        )

    def _reject_external_during_transaction(self, operation):
        self._guard_or_invalidate()
        if self._tx_active:
            raise A4ResidentTranslationCommitError(
                '%s is forbidden during an active A4.9b transaction' % operation
            )

    def audit_cpu(self, world):
        with self._lock:
            self._reject_external_during_transaction('audit_cpu')
            return a49a.A4ResidentArenaOwner.audit_cpu(self, world)

    def prepare_rebuild(self, world):
        with self._lock:
            self._reject_external_during_transaction('prepare_rebuild')
            return a49a.A4ResidentArenaOwner.prepare_rebuild(self, world)

    def swap_rebuild(self, *args, **kwargs):
        with self._lock:
            self._reject_external_during_transaction('swap_rebuild')
            return a49a.A4ResidentArenaOwner.swap_rebuild(
                self, *args, **kwargs
            )

    def invalidate(self, reason='explicit caller invalidation'):
        with self._lock:
            self._reject_external_during_transaction('invalidate')
            return a49a.A4ResidentArenaOwner.invalidate(self, reason)

    def _open_read_lease(self, world):
        with self._lock:
            self._reject_external_during_transaction('read lease')
            return a49a.A4ResidentArenaOwner._open_read_lease(self, world)

    def _validation_snapshot(self, world):
        with self._lock:
            self._reject_external_during_transaction('validation snapshot')
            return a49a.A4ResidentArenaOwner._validation_snapshot(self, world)

    def _cohere_rank5(self, world):
        with self._lock:
            self._reject_external_during_transaction('rank-5 coherence')
            lifecycle = a49a.A4ResidentArenaOwner.audit_cpu(self, world)
            if lifecycle == INVALID:
                raise A4ResidentTranslationCommitError(
                    'INVALID resident generation is never auto-rebuilt'
                )
            if lifecycle == CPU_NEWER:
                expected_id = self.arena_id
                expected_lifecycle = self.lifecycle
                expected_epochs = self.epochs
                candidate = a49a.A4ResidentArenaOwner.prepare_rebuild(
                    self, world,
                )
                a49a.A4ResidentArenaOwner.swap_rebuild(
                    self, candidate, world, expected_id,
                    expected_lifecycle, expected_epochs,
                )
            if self.lifecycle != COHERENT:
                raise A4ResidentTranslationCommitError(
                    'rank-5 resident source did not become coherent'
                )
            return self

    def _open_translation_lease(self, world):
        with self._lock:
            self._guard_or_invalidate()
            if self._tx_active or self._active_lease_token is not None:
                raise A4ResidentTranslationCommitError(
                    'another resident lease or transaction is active'
                )
            if self._lifecycle != COHERENT:
                raise A4ResidentTranslationCommitError(
                    'translation lease requires a coherent generation'
                )
            fresh = a49a._stable_cpu_snapshot(
                world, self._config, require_cache_exact=True,
            )
            if not a49a._same_source(self._arena.source, fresh.attestation):
                raise A4ResidentTranslationCommitError(
                    'CPU source changed after rank-5 coherence'
                )
            a49a._attest_resident(
                self._arena, self._arena_seal, readback=True,
            )
            self._lease_serial += 1
            self._arena.lease_serial = self._lease_serial
            self._issued_lease_token = None
            self._active_lease_consumed = False
            self._tx_serial += 1
            token = object()
            self._tx_active = True
            self._tx_token = token
            self._tx_stage = _TX_ISSUED
            self._tx_world = world
            self._tx_prepared = None
            self._tx_consumed = False
            self._refresh_guard()
            return _A4ResidentTranslationLease(
                self, world, self._tx_serial, token,
            )

    def _preflight_translation_capacity(self, world):
        """Repeat the full C/Q/S/W/P/cache union bound before lease issue."""
        with self._lock:
            self._guard_or_invalidate()
            if self._tx_active or self._lifecycle != COHERENT:
                raise A4ResidentTranslationCommitError(
                    'capacity preflight requires an idle coherent owner'
                )
            # _stable_cpu_snapshot performs the fixed-capacity full-world
            # pack.  Its state pack checks both active/gene and damaged/gene
            # fingerprint unions for every row, not merely current counts.
            packed = a49a._stable_cpu_snapshot(
                world, self._config, require_cache_exact=True,
            )
            if not a49a._same_source(
                    self._arena.source, packed.attestation):
                raise A4ResidentTranslationCommitError(
                    'CPU source changed before capacity preflight'
                )
            return None

    def _require_translation_lease(self, lease, stage):
        if (not isinstance(lease, _A4ResidentTranslationLease)
                or lease._factory_token is not _TRANSACTION_LEASE_TOKEN
                or lease._owner is not self
                or lease._owner_token is not self._owner_token
                or lease._world is not self._tx_world
                or lease._serial != self._tx_serial
                or lease._token is not self._tx_token
                or not self._tx_active
                or self._tx_stage != stage):
            raise A4ResidentTranslationCommitError(
                'stale, forged, or cross-owner translation lease'
            )

    def _activate_translation_lease(self, lease):
        with self._lock:
            self._guard_or_invalidate()
            self._require_translation_lease(lease, _TX_ISSUED)
            fresh = a49a._stable_cpu_snapshot(
                lease._world, self._config, require_cache_exact=True,
            )
            if not a49a._same_source(self._arena.source, fresh.attestation):
                raise A4ResidentTranslationCommitError(
                    'CPU source changed before translation lease open'
                )
            a49a._attest_resident(
                self._arena, self._arena_seal, readback=True,
            )
            self._tx_stage = _TX_OPEN
            self._refresh_guard()

    def _consume_translation_lease(self, lease):
        with self._lock:
            self._guard_or_invalidate()
            self._require_translation_lease(lease, _TX_OPEN)
            if self._tx_consumed:
                raise A4ResidentTranslationCommitError(
                    'translation lease binding was already consumed'
                )
            a49a._attest_resident(
                self._arena, self._arena_seal, readback=True,
            )
            self._tx_consumed = True
            self._tx_stage = _TX_CONSUMED
            self._refresh_guard()
            return self._arena.binding

    def _register_translation_storage(self, lease, prepared):
        with self._lock:
            self._guard_or_invalidate()
            self._require_translation_lease(lease, _TX_CONSUMED)
            _require_prepared_storage(prepared, self, self._tx_token)
            if self._tx_prepared is not None:
                raise A4ResidentTranslationCommitError(
                    'translation lease already owns prepared storage'
                )
            self._tx_prepared = prepared
            self._tx_stage = _TX_PREPARED
            self._refresh_guard()

    def _close_translation_lease(self, lease, successful):
        with self._lock:
            error = None
            try:
                self._guard_or_invalidate()
                self._require_translation_lease(
                    lease, _TX_PREPARED if successful else self._tx_stage,
                )
                fresh = a49a._stable_cpu_snapshot(
                    lease._world, self._config, require_cache_exact=True,
                )
                if not a49a._same_source(
                        self._arena.source, fresh.attestation):
                    raise A4ResidentTranslationCommitError(
                        'CPU source changed before translation lease close'
                    )
                a49a._attest_resident(
                    self._arena, self._arena_seal, readback=True,
                )
                if successful:
                    _require_prepared_storage(
                        self._tx_prepared, self, self._tx_token,
                    )
            except BaseException as exc:
                error = exc
            if not successful or error is not None:
                self._neutralize_transaction()
                self._refresh_guard()
            if error is not None:
                if self._lifecycle == COHERENT:
                    self._transition_invalid(
                        'translation lease close integrity failed',
                        ('R', 'S', 'C'), advance_unobserved=True,
                    )
                raise A4ResidentTranslationCommitError(
                    'translation lease close failed closed'
                ) from error

    def _abort_preclaim(self, world, prepared=None):
        with self._lock:
            try:
                self._guard_or_invalidate()
                if not self._tx_active or self._tx_world is not world:
                    raise A4ResidentTranslationCommitError(
                        'no matching A4.9b transaction to abort'
                    )
                fresh = a49a._stable_cpu_snapshot(
                    world, self._config, require_cache_exact=True,
                )
                a49a._attest_resident(
                    self._arena, self._arena_seal, readback=True,
                )
                if not a49a._same_source(
                        self._arena.source, fresh.attestation):
                    raise A4ResidentTranslationCommitError(
                        'CPU source changed before transaction abort'
                    )
            except BaseException as exc:
                if self._lifecycle == COHERENT:
                    self._transition_invalid(
                        'preclaim abort could not prove old authority',
                        ('R', 'S', 'C'), advance_unobserved=True,
                    )
                raise A4ResidentTranslationCommitError(
                    'A4.9b preclaim abort failed closed'
                ) from exc
            if prepared is not None:
                prepared.consumed = True
            self._neutralize_transaction()
            self._refresh_guard()

    def _build_prepared_storage(
            self, world, cell, dt, target_index, sync_performed,
            integration_state):
        dt = a4._translation_dt(dt)
        index = int(target_index)
        with self._open_translation_lease(world) as lease:
            source_binding = lease._binding_once()
            if (world.cells is not self._tx_world.cells
                    or tuple(world.cells)[index] is not cell
                    or int(cell.cell_id) < 0
                    or not bool(cell.alive)):
                raise A4ResidentTranslationCommitError(
                    'rank-5 target identity changed before resident planning'
                )
            target_id = int(cell.cell_id)
            generation = a48a._strict_counter(
                getattr(cell, 'generation', None), 'cell generation',
            )
            live_rng, rng_state = a48a._live_pcg64(world)
            old_arena = self._arena
            old_epochs = a49a._copy_epochs(self._epochs)
            old_ragged, old_state, old_cache, old_digests = (
                a49a._resident_readback(source_binding)
            )
            if old_digests != (
                    old_arena.expected_ragged_digest,
                    old_arena.expected_state_digest,
                    old_arena.expected_cache_digest):
                raise A4ResidentTranslationCommitError(
                    'active resident readback differs before planning'
                )
            host_source = a4.bind_a4_translation(old_ragged, old_state)
            if (a4._gene_cache_provenance(old_cache)
                    != host_source._cache_provenance):
                raise A4ResidentTranslationCommitError(
                    'active resident cache is not derived from its ragged source'
                )
            selected = paid_translation_selected_torch(
                source_binding, dt, index, target_id,
            )
            oracle = paid_translation_selected_numpy(
                host_source, dt, index, target_id,
            )
            selected_host = selected.to_numpy()
            _selected_output_scope(old_state, selected_host.state_after, index)
            _selected_states_match(
                selected_host.state_after, oracle.state_after, index,
            )

            resident_ragged = source_binding.ragged.clone()
            # The selected plan is an event-local artifact, not candidate
            # storage.  Clone all 30 final state tensors once more so the
            # generation aliases neither the old arena nor its input plan.
            resident_state = selected.state_after.clone()
            candidate_binding = a4.bind_a4_translation(
                resident_ragged, resident_state,
            )
            candidate_signature = a49a._resident_signature(candidate_binding)
            candidate_ragged, candidate_state, candidate_cache, digests = (
                a49a._resident_readback(candidate_binding)
            )
            if (digests[0] != old_digests[0]
                    or digests[2] != old_digests[2]
                    or not a48a._state_arrays_bit_exact(
                        candidate_state.state_dict(),
                        selected_host.state_after.state_dict(),
                    )
                    or not a48a._state_arrays_bit_exact(
                        candidate_ragged.state_dict(), old_ragged.state_dict(),
                    )
                    or not a48a._state_arrays_bit_exact(
                        candidate_cache.state_dict(), old_cache.state_dict(),
                    )):
                raise A4ResidentTranslationCommitError(
                    'fresh resident candidate content differs'
                )
            old_records = tuple(old_arena.resident_signature.tensor_records)
            new_records = tuple(candidate_signature.tensor_records)
            old_objects = {item[2] for item in old_records}
            new_objects = {item[2] for item in new_records}
            old_storage = {
                (item[5], item[3]) for item in old_records
                if int(item[3]) != 0
            }
            new_storage = {
                (item[5], item[3]) for item in new_records
                if int(item[3]) != 0
            }
            new_nonzero = [
                (item[5], item[3]) for item in new_records
                if int(item[3]) != 0
            ]
            plan_tensors = ((
                id(selected.target_mask), str(selected.target_mask.device),
                int(selected.target_mask.data_ptr()),
            ),) + tuple(
                (id(getattr(selected.state_after, name)),
                 str(getattr(selected.state_after, name).device),
                 int(getattr(selected.state_after, name).data_ptr()))
                for name in a4._TRANSLATION_ARRAY_FIELDS
            )
            plan_objects = {item[0] for item in plan_tensors}
            plan_storage = {
                (item[1], item[2]) for item in plan_tensors
                if int(item[2]) != 0
            }
            if (len(new_records) != 50
                    or len(new_objects) != 50
                    or len(new_nonzero) != len(set(new_nonzero))
                    or old_objects.intersection(new_objects)
                    or old_storage.intersection(new_storage)
                    or plan_objects.intersection(new_objects)
                    or plan_storage.intersection(new_storage)):
                raise A4ResidentTranslationCommitError(
                    'fresh candidate aliases old or event-local tensor storage'
                )
            expected_epochs = a49a._advance_epochs(
                old_epochs, frozenset(('S',)), cache_built=False,
            )
            prepared_pools = np.asarray(
                candidate_state.pools[index], dtype=np.float64,
            ).copy()
            prepared = _PreparedTranslationStorage(
                _factory_token=_PREPARED_STORAGE_TOKEN,
                _owner_token=self._owner_token,
                _transaction_token=self._tx_token,
                _world_ref=world,
                _cells_ref=world.cells,
                _cell_ref=cell,
                target_index=index,
                target_cell_id=target_id,
                target_generation=generation,
                dt_hex=dt.hex(),
                old_arena_ref=old_arena,
                old_arena_id=str(old_arena.arena_id),
                old_storage_generation=int(self._storage_generation),
                old_epochs=old_epochs,
                old_source_ref=old_arena.source,
                old_source_values=a49a._source_identity_values(
                    old_arena.source,
                ),
                selected_plan=selected,
                numpy_oracle=oracle,
                candidate_binding=candidate_binding,
                candidate_signature=candidate_signature,
                candidate_digests=tuple(digests),
                diagnostic_ragged=candidate_ragged,
                diagnostic_state=candidate_state,
                diagnostic_cache=candidate_cache,
                expected_arena_id=a49a._next_arena_id(),
                expected_storage_generation=(
                    int(self._storage_generation) + 1
                ),
                expected_epochs=expected_epochs,
                sync_performed=bool(sync_performed),
                prepared_pools=prepared_pools,
                prepared_proteins=_dict_from_selected_row(
                    candidate_state, index, 'active',
                ),
                prepared_damaged=_dict_from_selected_row(
                    candidate_state, index, 'damaged',
                ),
                prepared_last_translation=float(
                    candidate_state.last_translation[index]
                ),
                prepared_last_quiescence=float(
                    candidate_state.last_quiescence[index]
                ),
                source_cell_snapshot=a48b._cell_state_snapshot(cell),
                source_gene_specs=copy.deepcopy(cell.gene_specs),
                live_rng=live_rng,
                rng_state=copy.deepcopy(rng_state),
                dissipated_energy=world.dissipated_energy,
                world_config_ref=world.config,
                world_config_snapshot=a48a._world_config_snapshot(world),
                integration_state=copy.deepcopy(integration_state),
            )
            prepared.seal = _PreparedStorageSeal(
                prepared_ref=prepared,
                owner_token_ref=self._owner_token,
                transaction_token_ref=self._tx_token,
                values=_prepared_storage_values(prepared),
            )
            _require_prepared_storage(prepared, self, self._tx_token)
            lease._register(prepared)
        return prepared

    def _require_prepared_for_claim(
            self, world, cell, dt, prepared, integration_state):
        with self._lock:
            self._guard_or_invalidate()
            if (not self._tx_active
                    or self._tx_stage != _TX_PREPARED
                    or self._tx_world is not world
                    or self._tx_prepared is not prepared):
                raise A4ResidentTranslationCommitError(
                    'prepared storage is not the active owner transaction'
                )
            _require_prepared_storage(prepared, self, self._tx_token)
            dt = a4._translation_dt(dt)
            cells = tuple(world.cells)
            if (world.cells is not prepared._cells_ref
                    or prepared.target_index >= len(cells)
                    or cells[prepared.target_index] is not cell
                    or prepared._cell_ref is not cell
                    or int(cell.cell_id) != prepared.target_cell_id
                    or a48a._strict_counter(
                        getattr(cell, 'generation', None), 'cell generation',
                    ) != prepared.target_generation
                    or not bool(cell.alive)
                    or world.config is not prepared.world_config_ref
                    or dt.hex() != prepared.dt_hex
                    or a48a._world_config_snapshot(world)
                    != prepared.world_config_snapshot
                    or integration_state != prepared.integration_state
                    or a48b._cell_state_snapshot(cell)
                    != prepared.source_cell_snapshot
                    or not a4._gene_specs_exact(
                        cell.gene_specs, prepared.source_gene_specs,
                    )):
                raise A4ResidentTranslationCommitError(
                    'live target/world/config changed before claim'
                )
            live_rng, rng_state = a48a._live_pcg64(world)
            if (live_rng is not prepared.live_rng
                    or rng_state != prepared.rng_state
                    or world.dissipated_energy is not prepared.dissipated_energy
                    or float(world.dissipated_energy).hex()
                    != float(prepared.dissipated_energy).hex()):
                raise A4ResidentTranslationCommitError(
                    'RNG or world energy changed before claim'
                )
            if (self._arena is not prepared.old_arena_ref
                    or self._arena_seal.arena_id != prepared.old_arena_id
                    or self._storage_generation
                    != prepared.old_storage_generation
                    or a49a._epoch_values(self._epochs)
                    != a49a._epoch_values(prepared.old_epochs)
                    or self._lifecycle != COHERENT
                    or self._arena.source is not prepared.old_source_ref
                    or a49a._source_identity_values(self._arena.source)
                    != prepared.old_source_values):
                raise A4ResidentTranslationCommitError(
                    'owner generation changed before claim'
                )
            fresh = a49a._stable_cpu_snapshot(
                world, self._config, require_cache_exact=True,
            )
            if not a49a._same_source(
                    self._arena.source, fresh.attestation):
                raise A4ResidentTranslationCommitError(
                    'CPU source changed before claim'
                )
            old_ragged, old_state, old_cache = a49a._attest_resident(
                self._arena, self._arena_seal, readback=True,
            )
            prepared.selected_plan.validate()
            prepared.numpy_oracle.validate()
            actual_signature = a49a._resident_signature(
                prepared.candidate_binding,
            )
            if actual_signature != prepared.candidate_signature:
                raise A4ResidentTranslationCommitError(
                    'candidate pointer/version signature changed before claim'
                )
            candidate_ragged, candidate_state, candidate_cache, digests = (
                a49a._resident_readback(prepared.candidate_binding)
            )
            if (digests != prepared.candidate_digests
                    or not a48a._state_arrays_bit_exact(
                        candidate_ragged.state_dict(),
                        prepared.diagnostic_ragged.state_dict(),
                    )
                    or not a48a._state_arrays_bit_exact(
                        candidate_state.state_dict(),
                        prepared.diagnostic_state.state_dict(),
                    )
                    or not a48a._state_arrays_bit_exact(
                        candidate_cache.state_dict(),
                        prepared.diagnostic_cache.state_dict(),
                    )):
                raise A4ResidentTranslationCommitError(
                    'candidate diagnostic content changed before claim'
                )
            old_host = a4.bind_a4_translation(old_ragged, old_state)
            replay = paid_translation_selected_numpy(
                old_host, dt, prepared.target_index,
                prepared.target_cell_id,
            )
            selected_host = prepared.selected_plan.to_numpy()
            _selected_output_scope(
                old_state, selected_host.state_after,
                prepared.target_index,
            )
            _selected_states_match(
                selected_host.state_after, replay.state_after,
                prepared.target_index,
            )
            if (a4._translation_state_provenance(replay.state_after)
                    != a4._translation_state_provenance(
                        prepared.numpy_oracle.state_after)
                    or a4._gene_cache_provenance(old_cache)
                    != prepared.candidate_digests[2]):
                raise A4ResidentTranslationCommitError(
                    'fresh oracle or resident cache changed before claim'
                )
            return prepared

    def _finalize_after_cpu_publish(self, world, prepared):
        with self._lock:
            self._guard_or_invalidate()
            _require_prepared_storage(prepared, self, self._tx_token)
            if (self._tx_prepared is not prepared
                    or self._tx_stage != _TX_PREPARED
                    or self._arena is not prepared.old_arena_ref):
                raise A4ResidentTranslationCommitError(
                    'owner changed before post-publish finalization'
                )
            fresh = a49a._stable_cpu_snapshot(
                world, self._config, require_cache_exact=True,
            )
            changes = a49a._source_domain_changes(
                self._arena.source, fresh.attestation,
            )
            if changes not in (frozenset(), frozenset(('S',))):
                raise A4ResidentTranslationCommitError(
                    'CPU publish changed a domain outside logical S authority'
                )
            packed = fresh
            if (fresh.attestation.ragged_digest
                    != prepared.candidate_digests[0]
                    or fresh.attestation.state_digest
                    != prepared.candidate_digests[1]
                    or fresh.attestation.derived_cache_digest
                    != prepared.candidate_digests[2]
                    or not fresh.attestation.cache_exact
                    or not a48a._state_arrays_bit_exact(
                        packed.ragged.state_dict(),
                        prepared.diagnostic_ragged.state_dict(),
                    )
                    or not a48a._state_arrays_bit_exact(
                        packed.state.state_dict(),
                        prepared.diagnostic_state.state_dict(),
                    )
                    or not a48a._state_arrays_bit_exact(
                        packed.derived_cache.state_dict(),
                        prepared.diagnostic_cache.state_dict(),
                    )):
                raise A4ResidentTranslationCommitError(
                    'actual CPU publish differs from resident candidate source'
                )
            if (prepared.expected_storage_generation
                    != self._storage_generation + 1
                    or a49a._epoch_values(prepared.expected_epochs)
                    != a49a._epoch_values(a49a._advance_epochs(
                        self._epochs, frozenset(('S',)), cache_built=False,
                    ))):
                raise A4ResidentTranslationCommitError(
                    'candidate generation/epoch relation differs'
                )
            arena_epochs = a49a._copy_epochs(prepared.expected_epochs)
            arena = a49a._A4ResidentArena(
                arena_id=str(prepared.expected_arena_id),
                storage_generation=int(
                    prepared.expected_storage_generation
                ),
                device=str(self._device),
                config_state=a49a._config_state(self._config),
                epochs=arena_epochs,
                source=fresh.attestation,
                binding=prepared.candidate_binding,
                resident_signature=prepared.candidate_signature,
                expected_ragged_digest=prepared.candidate_digests[0],
                expected_state_digest=prepared.candidate_digests[1],
                expected_cache_digest=prepared.candidate_digests[2],
                lifecycle=COHERENT,
                dirty_domains=frozenset(),
                invalid_reason='',
                active_leases=0,
                lease_serial=int(self._lease_serial),
                retired=False,
            )
            arena_seal = a49a._make_arena_creation_seal(arena)
            a49a._attest_resident(arena, arena_seal, readback=True)
            owner_epochs = a49a._copy_epochs(arena_epochs)
            base_guard = a49a._OwnerCanonicalGuard(
                owner_token_ref=self._owner_token,
                arena_ref=arena,
                arena_seal=arena_seal,
                arena_seal_snapshot=a49a._copy_arena_creation_seal(
                    arena_seal,
                ),
                config_ref=self._config,
                epochs_ref=owner_epochs,
                last_observed_ref=fresh.attestation,
                last_observed_values=a49a._source_identity_values(
                    fresh.attestation,
                ),
                lifecycle=COHERENT,
                epochs_values=a49a._epoch_values(owner_epochs),
                device=str(self._device),
                config_state=a49a._config_state(self._config),
                storage_generation=int(arena.storage_generation),
                lease_serial=int(self._lease_serial),
                issued_lease_token=None,
                active_lease_token=None,
                active_lease_consumed=False,
            )
            final_serial = self._tx_serial + 1
            transaction_guard = _TranslationOwnerGuard(
                owner_token_ref=self._owner_token,
                base_guard_ref=base_guard,
                base_guard_values=_base_guard_values(base_guard),
                transaction_active=False,
                transaction_serial=int(final_serial),
                transaction_token_ref=None,
                transaction_stage=_TX_IDLE,
                transaction_world_ref=None,
                transaction_consumed=False,
                prepared_ref=None,
                prepared_seal_ref=None,
                prepared_values=None,
            )
            finalized = _FinalizedTranslationSwap(
                _factory_token=_FINALIZED_SWAP_TOKEN,
                owner_ref=self,
                prepared_ref=prepared,
                old_arena_ref=self._arena,
                old_arena_seal_ref=self._arena_seal,
                new_arena_ref=arena,
                new_arena_seal_ref=arena_seal,
                new_owner_epochs_ref=owner_epochs,
                fresh_source_ref=fresh.attestation,
                base_guard_ref=base_guard,
                transaction_guard_ref=transaction_guard,
                transaction_guard_snapshot=(
                    _transaction_guard_values(transaction_guard)
                ),
                final_transaction_serial=int(final_serial),
                retired_reason=(
                    'retired after A4.9b selected translation CAS'
                ),
            )
            finalized.seal = _FinalizedSwapSeal(
                finalized_ref=finalized,
                values=_finalized_swap_values(finalized),
            )
            return _require_finalized_swap(finalized, self, prepared)

    def _arm_final_swap(self, world, prepared, finalized):
        with self._lock:
            self._guard_or_invalidate()
            _require_prepared_storage(prepared, self, self._tx_token)
            _require_finalized_swap(finalized, self, prepared)
            if (self._arena is not finalized.old_arena_ref
                    or self._arena_seal is not finalized.old_arena_seal_ref
                    or self._storage_generation
                    != prepared.old_storage_generation
                    or a49a._epoch_values(self._epochs)
                    != a49a._epoch_values(prepared.old_epochs)
                    or self._tx_prepared is not prepared
                    or self._tx_stage != _TX_PREPARED):
                raise A4ResidentTranslationCommitError(
                    'final owner CAS input changed'
                )
            a49a._attest_resident(
                self._arena, self._arena_seal, readback=True,
            )
            a49a._attest_resident(
                finalized.new_arena_ref,
                finalized.new_arena_seal_ref,
                readback=True,
            )
            fresh = a49a._stable_cpu_snapshot(
                world, self._config, require_cache_exact=True,
            )
            live_rng, rng_state = a48a._live_pcg64(world)
            if (not a49a._same_source(
                    finalized.fresh_source_ref, fresh.attestation)
                    or live_rng is not prepared.live_rng
                    or rng_state != prepared.rng_state
                    or world.dissipated_energy
                    is not prepared.dissipated_energy
                    or float(world.dissipated_energy).hex()
                    != float(prepared.dissipated_energy).hex()):
                raise A4ResidentTranslationCommitError(
                    'CPU source/RNG/energy changed before final CAS arm'
                )
            return finalized

    def _commit_armed_swap_no_fail(self, finalized):
        # All references and guards below were allocated and validated before
        # receipt annotation.  Deliberately perform no callback/readback/check.
        with self._lock:
            old = finalized.old_arena_ref
            prepared = finalized.prepared_ref
            self._arena = finalized.new_arena_ref
            self._arena_seal = finalized.new_arena_seal_ref
            self._epochs = finalized.new_owner_epochs_ref
            self._last_observed = finalized.fresh_source_ref
            self._storage_generation = finalized.new_arena_ref.storage_generation
            self._lifecycle = COHERENT
            self._issued_lease_token = None
            self._active_lease_token = None
            self._active_lease_consumed = False
            self._guard = finalized.base_guard_ref
            self._tx_active = False
            self._tx_serial = finalized.final_transaction_serial
            self._tx_token = None
            self._tx_stage = _TX_IDLE
            self._tx_world = None
            self._tx_prepared = None
            self._tx_consumed = False
            self._tx_guard = finalized.transaction_guard_ref
            self._tx_guard_snapshot = finalized.transaction_guard_snapshot
            prepared.consumed = True
            finalized.consumed = True
            old.retired = True
            old.lifecycle = INVALID
            old.invalid_reason = finalized.retired_reason
            old.active_leases = 0
            old.lease_serial = self._lease_serial + 1
        return None

    def _rollback_after_claim(self, world, prepared):
        with self._lock:
            try:
                self._guard_or_invalidate()
                if (not self._tx_active
                        or self._tx_prepared is not prepared
                        or self._arena is not prepared.old_arena_ref
                        or self._storage_generation
                        != prepared.old_storage_generation
                        or a49a._epoch_values(self._epochs)
                        != a49a._epoch_values(prepared.old_epochs)):
                    raise A4ResidentTranslationCommitError(
                        'rollback owner relation differs'
                    )
                fresh = a49a._stable_cpu_snapshot(
                    world, self._config, require_cache_exact=True,
                )
                a49a._attest_resident(
                    self._arena, self._arena_seal, readback=True,
                )
                if not a49a._same_source(
                        self._arena.source, fresh.attestation):
                    raise A4ResidentTranslationCommitError(
                        'rollback did not restore exact old CPU authority'
                    )
            except BaseException as exc:
                if self._lifecycle == COHERENT:
                    self._transition_invalid(
                        'claimed translation rollback trust failure',
                        ('R', 'S', 'C'), advance_unobserved=True,
                    )
                raise A4ResidentTranslationCommitError(
                    'claimed A4.9b rollback failed closed'
                ) from exc
            prepared.consumed = True
            self._neutralize_transaction()
            self._refresh_guard()


_INTEGRATION_STATE_KEYS = frozenset(('schema', 'config', 'device'))


@dataclass(frozen=True)
class _SchedulerCreationSeal:
    scheduler_ref: object
    config_ref: object
    device_ref: object
    config_values: tuple
    device: str


@dataclass(frozen=True)
class _BackendBindingSeal:
    world_ref: object
    backend_ref: object
    backend_config_ref: object
    backend_device_ref: object
    backend_dtype_ref: object
    backend_config_digest: str
    backend_device: str
    backend_dtype: str
    backend_name: str


def _a4_config_values(config):
    return tuple(a48a._config_state(config).items())


def _scheduler_creation_seal_values(seal):
    if not isinstance(seal, _SchedulerCreationSeal):
        raise A4ResidentTranslationCommitError(
            'A4.9b scheduler creation seal is malformed'
        )
    return (
        id(seal.scheduler_ref), id(seal.config_ref), id(seal.device_ref),
        tuple(seal.config_values), str(seal.device),
    )


def _backend_binding_seal_values(seal):
    if not isinstance(seal, _BackendBindingSeal):
        raise A4ResidentTranslationCommitError(
            'A4.9b backend binding seal is malformed'
        )
    return (
        id(seal.world_ref), id(seal.backend_ref),
        id(seal.backend_config_ref), id(seal.backend_device_ref),
        id(seal.backend_dtype_ref), str(seal.backend_config_digest),
        str(seal.backend_device), str(seal.backend_dtype),
        str(seal.backend_name),
    )


def _canonical_resident_translation_integration_state(value):
    if not isinstance(value, Mapping) or set(value) != _INTEGRATION_STATE_KEYS:
        raise A4ResidentTranslationCommitError(
            'A4.9b save integration state is absent or noncanonical'
        )
    if value.get('schema') != SCHEMA_VERSION:
        raise A4ResidentTranslationCommitError(
            'A4.9b saved integration schema differs'
        )
    raw_config = value.get('config')
    if not isinstance(raw_config, Mapping):
        raise A4ResidentTranslationCommitError(
            'A4.9b saved config is invalid'
        )
    try:
        config = a4.GPU068A4Config.from_state(dict(raw_config))
        device = a49a._canonical_device(value.get('device'))
    except Exception as exc:
        raise A4ResidentTranslationCommitError(
            'A4.9b saved config/device is invalid'
        ) from exc
    return {
        'schema': SCHEMA_VERSION,
        'config': a48a._config_state(config),
        'device': device,
    }


def _translation_publish_snapshot(world, cell):
    return {
        'cell_state': a48b._cell_state_snapshot(cell),
        'pools_object': cell.pools,
        'pools': np.asarray(cell.pools, dtype=np.float64).copy(),
        'proteins_object': cell.proteins,
        'proteins_items': tuple(cell.proteins.items()),
        'damaged_object': cell.damaged_proteins,
        'damaged_items': tuple(cell.damaged_proteins.items()),
        'gene_specs_object': cell.gene_specs,
        'gene_specs_items': tuple(
            (fingerprint, spec, copy.deepcopy(spec))
            for fingerprint, spec in cell.gene_specs.items()
        ),
        'last_translation': cell.last_translation,
        'last_quiescence': cell.last_quiescence,
        'dissipated_energy': world.dissipated_energy,
        'rng_object': world.rng,
        'rng_state': copy.deepcopy(world.rng.bit_generator.state),
    }


def _restore_mapping_identity(mapping, items):
    mapping.clear()
    for key, value in items:
        mapping[key] = value
    return mapping


class A4ResidentTranslationEventScheduler(
        a48c8.A4ReplicationEarlyNoopEventScheduler):
    """c8 scheduler with one world-resident paid-translation transaction."""

    def __init__(self, a4_config, device, max_receipts=128):
        self._a49b_commit_active = False
        self._a49b_owner = None
        self._a49b_world = None
        self._a49b_cells = None
        super(A4ResidentTranslationEventScheduler, self).__init__(
            a4_config=a4_config, device=device, max_receipts=max_receipts,
        )
        self._a49b_scheduler_seal = _SchedulerCreationSeal(
            scheduler_ref=self,
            config_ref=self.a4_config,
            device_ref=self.a4_device,
            config_values=_a4_config_values(self.a4_config),
            device=str(self.a4_device),
        )
        self._a49b_scheduler_seal_snapshot = (
            _scheduler_creation_seal_values(self._a49b_scheduler_seal)
        )
        self._a49b_backend_seal = None
        self._a49b_backend_seal_snapshot = None

    def _require_runtime_creation_seals(self, world=None, owner=None):
        seal = self._a49b_scheduler_seal
        if (_scheduler_creation_seal_values(seal)
                != tuple(self._a49b_scheduler_seal_snapshot)
                or seal.scheduler_ref is not self
                or self.a4_config is not seal.config_ref
                or self.a4_device is not seal.device_ref
                or not isinstance(self.a4_device, str)
                or str(self.a4_device) != seal.device
                or _a4_config_values(self.a4_config)
                != seal.config_values):
            raise A4ResidentTranslationCommitError(
                'scheduler config/device creation seal differs'
            )
        backend_seal = self._a49b_backend_seal
        if backend_seal is not None:
            if (_backend_binding_seal_values(backend_seal)
                    != tuple(self._a49b_backend_seal_snapshot)
                    or (world is not None
                        and backend_seal.world_ref is not world)):
                raise A4ResidentTranslationCommitError(
                    'scheduler backend binding seal differs'
                )
            backend = backend_seal.backend_ref
            if backend is None:
                if (world is not None
                        and getattr(
                            world, '_soma068a3_backend', None,
                        ) is not None):
                    raise A4ResidentTranslationCommitError(
                        'an unbound scheduler acquired a live backend'
                    )
                if self._backend_name != backend_seal.backend_name:
                    raise A4ResidentTranslationCommitError(
                        'unbound scheduler backend label changed'
                    )
            else:
                if (world is not None
                        and getattr(
                            world, '_soma068a3_backend', None,
                        ) is not backend
                        or backend.config is not backend_seal.backend_config_ref
                        or backend.device is not backend_seal.backend_device_ref
                        or backend.dtype is not backend_seal.backend_dtype_ref
                        or _pickle_digest(backend.config)
                        != backend_seal.backend_config_digest
                        or str(backend.device) != backend_seal.backend_device
                        or str(backend.dtype) != backend_seal.backend_dtype
                        or self._backend_name != backend_seal.backend_name
                        or backend_seal.backend_device != seal.device):
                    raise A4ResidentTranslationCommitError(
                        'live backend/config/device relation differs'
                    )
        if owner is not None:
            owner._require_guard()
            if (owner._config is not seal.config_ref
                    or owner._device != seal.device
                    or owner._arena.device != seal.device
                    or tuple(owner._arena.config_state)
                    != tuple(seal.config_values)):
                raise A4ResidentTranslationCommitError(
                    'active arena config/device differs from scheduler seal'
                )
        return seal

    def resident_translation_integration_state(self):
        self._require_runtime_creation_seals(
            self._a49b_world, self._a49b_owner,
        )
        return {
            'schema': SCHEMA_VERSION,
            'config': dict(self._a49b_scheduler_seal.config_values),
            'device': self._a49b_scheduler_seal.device,
        }

    def _bind_resident_world(self, world, backend=None):
        if world is None or not hasattr(world, 'cells'):
            raise A4ResidentTranslationCommitError(
                'A4.9b scheduler requires one Formal066 world'
            )
        if self._a49b_commit_active:
            raise A4ResidentTranslationCommitError(
                'cannot bind a world during an active transaction'
            )
        if self._a49b_world is not None and self._a49b_world is not world:
            raise A4ResidentTranslationCommitError(
                'A4.9b scheduler cannot be rebound to another world'
            )
        self._require_runtime_creation_seals()
        live_backend = (
            getattr(world, '_soma068a3_backend', None)
            if backend is None else backend
        )
        if self._a49b_backend_seal is None:
            if live_backend is None:
                backend_seal = _BackendBindingSeal(
                    world_ref=world, backend_ref=None,
                    backend_config_ref=None, backend_device_ref=None,
                    backend_dtype_ref=None, backend_config_digest='',
                    backend_device='', backend_dtype='',
                    backend_name=str(self._backend_name),
                )
            else:
                backend_device = str(live_backend.device)
                if backend_device != self._a49b_scheduler_seal.device:
                    raise A4ResidentTranslationCommitError(
                        'A3 backend and A4 resident devices differ'
                    )
                backend_seal = _BackendBindingSeal(
                    world_ref=world, backend_ref=live_backend,
                    backend_config_ref=live_backend.config,
                    backend_device_ref=live_backend.device,
                    backend_dtype_ref=live_backend.dtype,
                    backend_config_digest=_pickle_digest(
                        live_backend.config,
                    ),
                    backend_device=backend_device,
                    backend_dtype=str(live_backend.dtype),
                    backend_name=type(live_backend).__name__,
                )
            self._a49b_backend_seal = backend_seal
            self._a49b_backend_seal_snapshot = (
                _backend_binding_seal_values(backend_seal)
            )
        self._a49b_world = world
        self._a49b_cells = world.cells
        self._require_runtime_creation_seals(world, self._a49b_owner)
        return self

    def state_dict(self):
        if (self._a49b_commit_active
                or (self._a49b_owner is not None
                    and self._a49b_owner._tx_active)):
            raise a3s.A3SchedulerProtocolError(
                'cannot serialize an active A4.9b translation transaction'
            )
        state = super(A4ResidentTranslationEventScheduler, self).state_dict()
        state['a4_resident_translation'] = (
            self.resident_translation_integration_state()
        )
        return state

    @classmethod
    def from_state(cls, state):
        state = dict(state or {})
        own = _canonical_resident_translation_integration_state(
            state.get('a4_resident_translation'),
        )
        inherited = (
            a48c8.A4ReplicationEarlyNoopEventScheduler.from_state(state)
        )
        if (own['config'] != a48a._config_state(inherited.a4_config)
                or own['device'] != inherited.a4_device):
            raise A4ResidentTranslationCommitError(
                'saved c8/A4.9b config or device differs'
            )
        scheduler = cls(
            a4_config=own['config'], device=own['device'],
            max_receipts=inherited.max_receipts,
        )
        scheduler._next_step_id = int(inherited._next_step_id)
        scheduler._backend_name = str(inherited._backend_name)
        scheduler._receipts = copy.deepcopy(inherited._receipts)
        return scheduler

    def _resident_owner_for_rank5(self, world):
        self._require_runtime_creation_seals(world, self._a49b_owner)
        if self._a49b_world is not world:
            raise A4ResidentTranslationCommitError(
                'active scheduler/world identity differs'
            )
        if (self._a49b_owner is not None
                and self._a49b_cells is not world.cells):
            # Let the arena's exact membership audit make this irreversible;
            # do not hide a container rebind behind a scheduler-only error.
            self._a49b_owner.audit_cpu(world)
            raise A4ResidentTranslationCommitError(
                'active world.cells container identity differs'
            )
        if self._a49b_owner is None and self._a49b_cells is not world.cells:
            raise A4ResidentTranslationCommitError(
                'world.cells changed before first resident construction'
            )
        if self._a49b_owner is None:
            self._a49b_owner = _A4ResidentTranslationOwner.from_cpu(
                world, self._a49b_scheduler_seal.config_ref,
                self._a49b_scheduler_seal.device,
            )
        self._require_runtime_creation_seals(world, self._a49b_owner)
        self._a49b_owner._cohere_rank5(world)
        self._a49b_owner._preflight_translation_capacity(world)
        return self._a49b_owner

    def _resident_translation_candidate_ready(
            self, world, cell, dt, config, prepared):
        """Protected fault-injection seam before the event claim."""
        return prepared

    def _translation_candidate_ready(
            self, world, cell, dt, config, prepared):
        return self._resident_translation_candidate_ready(
            world, cell, dt, config, prepared,
        )

    def _publish_resident_translation_candidate(
            self, world, cell, prepared):
        cell.pools[...] = prepared.prepared_pools
        if prepared.sync_performed:
            cell.proteins = prepared.prepared_proteins
            cell.damaged_proteins = prepared.prepared_damaged
        cell.last_translation = prepared.prepared_last_translation
        cell.last_quiescence = prepared.prepared_last_quiescence

    def _publish_translation_candidate(self, world, cell, prepared):
        return self._publish_resident_translation_candidate(
            world, cell, prepared,
        )

    def _resident_translation_finalized_ready(
            self, world, cell, prepared, finalized):
        """Protected seam before the last full revalidation/CAS arm."""
        return finalized

    def _published_resident_translation_matches(
            self, world, cell, prepared, snapshot):
        if (cell.pools is not snapshot['pools_object']
                or not _arrays_bit_exact(
                    cell.pools, prepared.prepared_pools,
                )
                or cell.last_translation
                is not prepared.prepared_last_translation
                or cell.last_quiescence
                is not prepared.prepared_last_quiescence
                or not a48b._gene_specs_snapshot_matches(cell, snapshot)
                or world.rng is not prepared.live_rng
                or world.rng.bit_generator.state != prepared.rng_state
                or world.dissipated_energy
                is not prepared.dissipated_energy):
            raise A4ResidentTranslationCommitError(
                'CPU publish changed data outside the prepared target row'
            )
        if prepared.sync_performed:
            if (cell.proteins is not prepared.prepared_proteins
                    or cell.damaged_proteins
                    is not prepared.prepared_damaged):
                raise A4ResidentTranslationCommitError(
                    'CPU publish did not attach precreated protein mappings'
                )
        else:
            if (cell.proteins is not snapshot['proteins_object']
                    or cell.damaged_proteins
                    is not snapshot['damaged_object']):
                raise A4ResidentTranslationCommitError(
                    'early translation return rebound protein mappings'
                )
        if (tuple(cell.proteins.items())
                != tuple(prepared.prepared_proteins.items())
                or tuple(cell.damaged_proteins.items())
                != tuple(prepared.prepared_damaged.items())):
            raise A4ResidentTranslationCommitError(
                'CPU protein dictionary content/order differs from D2H row'
            )
        return None

    def _rollback_resident_translation_publish(
            self, world, cell, snapshot):
        snapshot['pools_object'][...] = snapshot['pools']
        cell.pools = snapshot['pools_object']
        cell.proteins = _restore_mapping_identity(
            snapshot['proteins_object'], snapshot['proteins_items'],
        )
        cell.damaged_proteins = _restore_mapping_identity(
            snapshot['damaged_object'], snapshot['damaged_items'],
        )
        gene_specs = snapshot['gene_specs_object']
        for _, spec_object, saved_spec in snapshot['gene_specs_items']:
            spec_object.clear()
            spec_object.update(copy.deepcopy(saved_spec))
        gene_specs.clear()
        for fingerprint, spec_object, _ in snapshot['gene_specs_items']:
            gene_specs[fingerprint] = spec_object
        cell.gene_specs = gene_specs
        cell.last_translation = snapshot['last_translation']
        cell.last_quiescence = snapshot['last_quiescence']
        world.dissipated_energy = snapshot['dissipated_energy']
        world.rng = snapshot['rng_object']
        world.rng.bit_generator.state = copy.deepcopy(snapshot['rng_state'])
        if a48b._cell_state_snapshot(cell) != snapshot['cell_state']:
            raise A4ResidentTranslationCommitError(
                'CPU rollback did not restore exact target cell state'
            )

    def _commit_resident_translation(
            self, world, cell, dt, config, owner, prepared):
        owner._require_prepared_for_claim(
            world, cell, dt, prepared,
            self.resident_translation_integration_state(),
        )
        snapshot = _translation_publish_snapshot(world, cell)
        metadata = {
            'authority': (
                'A4.9b-world-resident-selected-paid-translation-atomic-CAS'
            ),
            'enabled': bool(config.gene_expression),
            'device': self.a4_device,
            'genome_count': int(
                prepared.diagnostic_state.genome_count[
                    prepared.target_index
                ]
            ),
            'source_provenance': str(
                prepared.old_source_ref.state_digest
            ),
            'final_provenance': str(prepared.candidate_digests[1]),
            'sync_performed': bool(prepared.sync_performed),
        }
        try:
            self.claim(cell, 'translation_cpu', metadata=metadata)
        except BaseException:
            owner._abort_preclaim(world, prepared)
            raise
        try:
            self._publish_translation_candidate(world, cell, prepared)
            self._published_resident_translation_matches(
                world, cell, prepared, snapshot,
            )
            finalized = owner._finalize_after_cpu_publish(world, prepared)
            ready = self._resident_translation_finalized_ready(
                world, cell, prepared, finalized,
            )
            if ready is not finalized:
                raise A4ResidentTranslationCommitError(
                    'finalized-ready seam must retain its private object'
                )
            armed = owner._arm_final_swap(world, prepared, finalized)
            amount = float(prepared.prepared_last_translation)
            annotation = {
                'amount': amount,
                'work_performed': amount > 0.0,
                'sync_performed': bool(prepared.sync_performed),
                'rng_draw_count': 0,
            }
            self.annotate_claim(cell, 'translation_cpu', annotation)
        except BaseException as original:
            rollback_error = None
            try:
                self._rollback_resident_translation_publish(
                    world, cell, snapshot,
                )
            except BaseException as exc:
                rollback_error = exc
            try:
                owner._rollback_after_claim(world, prepared)
            except BaseException as exc:
                if rollback_error is None:
                    rollback_error = exc
            if rollback_error is not None:
                raise A4ResidentTranslationCommitError(
                    'claimed A4.9b transaction rollback failed closed'
                ) from original
            raise
        owner._commit_armed_swap_no_fail(armed)
        return None

    def cpu_translation(self, world, cell, dt, config=None):
        """Commit only rank-5 RNG-free translation through a fresh arena."""
        self._require_runtime_creation_seals(world, self._a49b_owner)
        _, record = self._context(world, cell, dt)
        config = world.config if config is None else config
        if config is not world.config:
            raise A4ResidentTranslationCommitError(
                'translation config must be the active world config object'
            )
        if ('translation_cpu' in record['claimed']
                or 'replication_cpu' in record['claimed']):
            raise a3s.A3DuplicateEventError(
                'duplicate or post-replication A4.9b translation event'
            )
        if ('maintenance' not in record['claimed']
                or int(record['last_rank'])
                != int(a3s._CELL_RANK['maintenance'])):
            raise a3s.A3EventOrderError(
                'A4.9b translation requires the exact rank-5 preclaim boundary'
            )
        if (self._a49b_commit_active
                or self._a4_translation_commit_active):
            raise a3s.A3SchedulerProtocolError(
                'nested A4.9b translation commit is forbidden'
            )
        if not bool(getattr(cell, 'alive', False)):
            raise A4ResidentTranslationCommitError(
                'A4.9b translation target must be alive'
            )
        dt = a4._translation_dt(dt)
        self._a49b_commit_active = True
        self._a4_translation_commit_active = True
        owner = None
        prepared = None
        try:
            owner = self._resident_owner_for_rank5(world)
            matches = [
                index for index, candidate in enumerate(tuple(world.cells))
                if candidate is cell
            ]
            if len(matches) != 1:
                raise A4ResidentTranslationCommitError(
                    'translation target is not one exact world member'
                )
            _, host_binding = self._pack_host_binding(world, cell)
            a48b._require_cell_gene_specs(cell, host_binding)
            sync_performed = a48b._translation_sync_performed(host_binding)
            prepared = owner._build_prepared_storage(
                world, cell, dt, matches[0], sync_performed,
                self.resident_translation_integration_state(),
            )
            ready = self._translation_candidate_ready(
                world, cell, dt, config, prepared,
            )
            if ready is not prepared:
                raise A4ResidentTranslationCommitError(
                    'candidate-ready seam must retain private storage'
                )
            owner._require_prepared_for_claim(
                world, cell, dt, prepared,
                self.resident_translation_integration_state(),
            )
            return self._commit_resident_translation(
                world, cell, dt, config, owner, prepared,
            )
        except BaseException:
            if (owner is not None and owner._tx_active
                    and prepared is not None
                    and owner._tx_prepared is prepared):
                owner._abort_preclaim(world, prepared)
            raise
        finally:
            self._a4_translation_commit_active = False
            self._a49b_commit_active = False


class Hybrid066WorldA4ResidentTranslation(
        a48c8.Hybrid066WorldA4ReplicationEarlyNoop):
    """c8 world whose rank-5 translation owns a fresh resident generation."""

    def __init__(self, cpu_world, backend=None, scheduler=None,
                 gpu_config=None, a4_config=None, a4_device=None):
        if scheduler is None:
            scheduler = A4ResidentTranslationEventScheduler(
                a4_config=a4_config, device=a4_device,
            )
        elif not isinstance(
                scheduler, A4ResidentTranslationEventScheduler):
            raise TypeError(
                'scheduler must be A4ResidentTranslationEventScheduler'
            )
        super(Hybrid066WorldA4ResidentTranslation, self).__init__(
            cpu_world, backend=backend, scheduler=scheduler,
            gpu_config=gpu_config, a4_config=a4_config,
            a4_device=a4_device,
        )
        self.scheduler._bind_resident_world(self.world, self.backend)

    @classmethod
    def new(cls, seed=101, initial_cells=3, world_config=None,
            gpu_config=None, backend=None, a4_config=None, a4_device=None):
        if a4_config is None or a4_device is None:
            raise A4ResidentTranslationCommitError(
                'new A4.9b world requires explicit A4 config and device'
            )
        config = (
            world_config
            if isinstance(world_config, a3s.a2.s66.Formal066Config)
            else a3s.a2.s66.Formal066Config.from_state(world_config or {})
        )
        world = a3s.a2.s66.Formal066World(
            seed=seed, initial_cells=initial_cells, config=config,
        )
        return cls(
            world, backend=backend,
            gpu_config=gpu_config if backend is None else None,
            a4_config=a4_config, a4_device=a4_device,
        )

    def step(self, dt):
        # Formal066 publishes a fresh survivors list after a successful step.
        # Preserve this wrapper's pre-step concrete container identity only
        # after all inherited events/receipts have completed.  Copying the
        # final ordered members back means an actual object/ID/generation/
        # alive/order change remains an M-domain difference at the next arena
        # audit, while a container-only implementation rebind is not biology.
        self.scheduler._require_runtime_creation_seals(
            self.world, self.scheduler._a49b_owner,
        )
        original = self.world.cells
        if not isinstance(original, list):
            raise A4ResidentTranslationCommitError(
                'A4.9b live world requires one concrete cells list'
            )
        receipt = super(Hybrid066WorldA4ResidentTranslation, self).step(dt)
        final_container = self.world.cells
        if not isinstance(final_container, (list, tuple)):
            raise A4ResidentTranslationCommitError(
                'successful Formal066 step produced a malformed cells container'
            )
        final_members = list(final_container)
        original[:] = final_members
        self.world.cells = original
        return receipt

    def summary(self):
        integration = self.scheduler.resident_translation_integration_state()
        output = super(Hybrid066WorldA4ResidentTranslation, self).summary()
        output.update({
            'gpu_build': BUILD,
            'gpu_schema': SCHEMA_VERSION,
            'gpu_port_status': dict(PORT_STATUS),
            'gpu_full_world_step': False,
            'a4_resident_translation_device': integration['device'],
            'a4_resident_translation_config': copy.deepcopy(
                integration['config'],
            ),
        })
        return output

    def state_dict(self):
        state = super(Hybrid066WorldA4ResidentTranslation, self).state_dict()
        state.update({
            'save_version': SAVE_VERSION,
            'build': BUILD,
            'a4_resident_translation': (
                self.scheduler.resident_translation_integration_state()
            ),
        })
        return state

    @classmethod
    def from_state(cls, state, backend=None, backend_factory=None):
        state = dict(state or {})
        if (state.get('save_version') != SAVE_VERSION
                or state.get('build') != BUILD):
            raise A4ResidentTranslationCommitError(
                'A4.9b save version/build differs'
            )
        own = _canonical_resident_translation_integration_state(
            state.get('a4_resident_translation'),
        )
        inherited_state = copy.deepcopy(state)
        inherited_state['save_version'] = a48c8.SAVE_VERSION
        inherited_state['build'] = a48c8.BUILD
        inherited = a48c8.Hybrid066WorldA4ReplicationEarlyNoop.from_state(
            inherited_state, backend=backend,
            backend_factory=backend_factory,
        )
        scheduler = A4ResidentTranslationEventScheduler.from_state(
            state.get('scheduler', {}),
        )
        if scheduler.resident_translation_integration_state() != own:
            raise A4ResidentTranslationCommitError(
                'saved world/scheduler A4.9b settings differ'
            )
        return cls(
            inherited.world, backend=inherited.backend,
            scheduler=scheduler,
        )


__all__ = (
    'BUILD', 'BUILD_ID', 'BUILD_LONG', 'SCHEMA_VERSION',
    'SELECTED_PLAN_SCHEMA_VERSION', 'SAVE_VERSION',
    'FULL_GPU_WORLD_STEP', 'TRANSLATION_ORACLE_ATOL',
    'COHERENT', 'CPU_NEWER', 'INVALID', 'PORT_STATUS',
    'A4ResidentTranslationCommitError', 'A4SelectedTranslationPlan',
    'paid_translation_selected_numpy', 'paid_translation_selected_torch',
    'paid_translation_selected', 'A4ResidentTranslationEventScheduler',
    'Hybrid066WorldA4ResidentTranslation',
)
