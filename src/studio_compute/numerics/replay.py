"""Shared-prompt group replay: differentiate one prompt once for all completions.

DGX counterpart: group_replay.py (segments, checkpoint groups, summed CISPO
objective); the layer math is mlx_group_ops (DenseGroupOps on DGX).

Architecture map:
1. Shared prompt. Complete SEGMENT_TOKENS segments before the first target row
   run once, layer by layer, each segment continuing the previous one's
   carried state (GDN conv/recurrent state, attention keys/values). Those
   carries stay differentiable.
2. Completions. Each branch continues from the shared boundary state over its
   own remaining segments. Only target rows reach the vocabulary head, in
   HEAD_ROWS tiles; full-sequence vocabulary logits never exist.
3. Backward. Each branch's head VJP seeds its segments in reverse; layer
   inputs saved every ``checkpoint_layers`` layers are recomputed one group at
   a time, and every (segment, layer) unit is differentiated explicitly. The
   branches' boundary-state adjoints seed one backward
   through the shared prompt. Legacy adjoints use FP32. BF16 adjoints and adapter
   gradients use balanced sums of BF16 tensors.

The BF16 engine consumes explicit saved-layer tapes. Optional projection and
post-mixer residuals follow measured retention costs within an admitted budget.
Checkpoint interval one keeps each layer input. Larger intervals reconstruct
only discarded boundaries and preserve original carry producer identities.
Counters record all reconstructed projections and nonlinear primals.

Invariants:
- Mathematically identical to replaying each completion as its own full
  prompt+completion sequence; BF16 addition order differs (rounding only).
- Branch prompts (embeddings and three-axis positions) must match exactly
  before any state is shared.
- An absent graph edge is distinct from a zero cotangent: only live outputs
  seed a unit's VJP, in a fixed carry-before-hidden order.
Main callers: mlx_batch_training (updates) and mlx_backend.current_logprobs.
"""
from __future__ import annotations

import time
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

LAYER_PREFIX = 'layers.'
from .cispo import frozen_head_logprobs
from .compiled_gdn import CompiledGDNUnit
from .sums import BalancedSums
HEAD_ROWS = 128
from .ops import initial_states, segment_layer

SEGMENT_TOKENS = 2048


class Replay:
    def __init__(self, model, *, segment_tokens=SEGMENT_TOKENS, checkpoint_layers=1, head_rows=HEAD_ROWS, flash=None, progress=None, profile=None, execution_profile="legacy", saved_activations=False, retained_projections=None, activation_budget_bytes=0, projection_costs=None, model_ops=None, gradient_sum_levels=None):
        if execution_profile not in ("legacy", "bf16-fast"):
            raise ValueError("Unknown replay execution profile.")
        if saved_activations and execution_profile != "bf16-fast":
            raise ValueError("Saved activation kernels require the BF16 profile.")
        self.saved_activations = saved_activations
        self.retained_projections = retained_projections
        self.saved_counters = {}
        self.activation_budget_bytes = activation_budget_bytes
        self.projection_costs = {} if projection_costs is None else projection_costs
        self.retention_choices, self.retention_report = {}, {}
        self.execution_profile = execution_profile
        self.model_ops = model_ops
        # None keeps the exact balanced order. A cap bounds adapter-gradient copies.
        self.gradient_sum_levels = gradient_sum_levels
        self.gradient_dtype = mx.bfloat16 if execution_profile == "bf16-fast" else mx.float32
        self.model = model
        self.lm = model.language_model
        self.layers = self.lm.layers
        if any(type(x) is not int or x < 1 for x in (segment_tokens, checkpoint_layers, head_rows)):
            raise ValueError('Replay segment and checkpoint intervals must be positive')
        self.segment_tokens = segment_tokens
        self.checkpoint_layers = checkpoint_layers
        self.head_rows = head_rows
        self.flash = flash
        self.progress = progress
        self.profile = profile
        self._compiled_gdn_units: dict[int, CompiledGDNUnit] = {}
        # Token-only values that a recorded forward keeps for its segment VJP.
        self._segment_memos: dict = {}
        # The arrays of the group VJP that runs while the host builds the next one.
        self._inflight = None

    def report(self, phase, segment, total, **details):
        if self.progress:
            self.progress({'phase': phase, 'segment': segment, 'segments': total,
                           'active_memory_bytes': mx.get_active_memory(),
                           'peak_memory_bytes': mx.get_peak_memory(), **details})

    def _layer(self, layer, hidden, states, positions, context=None):
        if self.model_ops is not None:
            return self.model_ops.layer(layer, hidden, states, positions, context)
        if self._can_compile_gdn(layer, hidden):
            return self._gdn_unit(layer).forward(hidden, states, positions)
        return segment_layer(layer, hidden, states, positions, flash=self.flash, execution_profile=self.execution_profile)

    def _can_compile_gdn(self, layer: Any, hidden_BLD: Any) -> bool:
        return (self.model_ops is None and self.execution_profile == "legacy" and layer.is_linear and layer.training and hidden_BLD.dtype == mx.bfloat16
                and mx.default_device() == mx.gpu)

    def _gdn_unit(self, layer: Any) -> CompiledGDNUnit:
        key = id(layer)
        if key not in self._compiled_gdn_units:
            self._compiled_gdn_units[key] = CompiledGDNUnit(mx, layer, segment_layer)
        return self._compiled_gdn_units[key]

    def head(self, hidden, tokens):
        if self.model_ops is not None:
            return self.model_ops.head(self.lm, hidden, tokens, self.gradient_dtype)
        return frozen_head_logprobs(self.lm, hidden, tokens, dtype=self.gradient_dtype)

    def _context(self, branch, start, stop, mode=None):
        """Return the model context; mode "keep" or "take" adds the segment memo."""
        if self.model_ops is None:
            return None
        if mode is None:
            return self.model_ops.context(branch, start, stop)
        return self.model_ops.context(branch, start, stop, store=self._segment_memos, mode=mode)

    def run(self, embeddings, positions, prefix_length, tokens, objective=None, *, capture_prefix_gradient=False):
        """One sequence: selected log probabilities, optionally objective and adapter VJP.

        ``embeddings`` contains the full prompt followed by tokens[:-1].
        ``objective`` is a scalar function of the selected-token log probabilities.
        """
        probabilities, loss, gradients, stats = yield from self.run_group([{
            'embeddings': embeddings, 'positions': positions, 'prefix_length': prefix_length,
            'tokens': tokens, 'objective': objective}], capture_prefix_gradient=capture_prefix_gradient)
        return probabilities[0], loss, gradients, stats

    def _forward(self, branch, first_segment, stop_segment, states, *, record, phase, branch_index=None):
        """Advance one branch over segments, recording saved inputs and head rows.

        Each caller owns its state lists. Shared tensors remain immutable, so
        advancing a continuation cannot replace another branch's carries.
        Producer identities (segment, layer, kind) name every carried tensor.
        """
        embeddings, positions = branch['embeddings'], branch['positions']
        prefix_length, target_ids = branch['prefix_length'], mx.array(branch['tokens'], dtype=mx.int32)
        target_indices = branch.get('target_indices')
        states = [list(values) for values in states]
        units, heads, probabilities = {}, [], []
        for segment in range(first_segment, stop_segment):
            start = segment * self.segment_tokens
            stop = min(embeddings.shape[1], start + self.segment_tokens)
            hidden = embeddings[:, start:stop]
            row = []
            for index, layer in enumerate(self.layers):
                incoming = list(states[index])
                checkpoint = hidden if index % self.checkpoint_layers == 0 else None
                profile_started = time.perf_counter()
                tape = None
                if record and self.saved_activations:
                    from .saved import forward_saved
                    choices = self.retention_choices.get((segment,index),frozenset())
                    retained = (choices if self.retained_projections is None
                                else self.retained_projections(segment, index))
                    hidden, *values = forward_saved(
                        layer, hidden, [value for _, value in incoming],
                        positions[:, :, start:stop], retain=retained,
                        retain_hidden=checkpoint is not None,
                        retain_middle="middle" in choices or self.retained_projections is not None,
                        counters=self.saved_counters,
                    )
                    carried, tape = values[:2], values[2]
                    mx.eval(hidden, carried, tape.retained, tape.low_rank,
                            tape.input_statistics,tape.middle_statistics,
                            [] if tape.attention is None else tape.attention,
                            [] if tape.middle_BLD is None else [tape.middle_BLD])
                else:
                    hidden, *carried = self._layer(layer, hidden, [value for _, value in incoming], positions[:, :, start:stop],
                                                   self._context(branch, start, stop, 'keep' if record else None))
                    mx.eval(hidden, carried)
                if record:
                    row.append((checkpoint, incoming, tape))
                if self.profile is not None:
                    self.profile.record('forward', index, layer.is_linear,
                                        time.perf_counter() - profile_started)
                produced = [((segment, index, kind), value) for kind, value in enumerate(carried)]
                replace = (self.model_ops.replace_state(layer) if self.model_ops is not None else layer.is_linear)
                states[index] = produced if replace else incoming + produced
                if (index + 1) % self.checkpoint_layers == 0 or index + 1 == len(self.layers):
                    yield {'phase': phase, 'segment': segment, 'layer': index, 'branch': branch_index}
            if record:
                units[segment] = row
            selected = (range(max(start, prefix_length - 1), stop) if target_indices is None else
                        [prefix_length - 1 + i for i in target_indices if start <= prefix_length - 1 + i < stop])
            for tile in range(0, len(selected), self.head_rows):
                if target_indices is None:
                    # Preserve the original contiguous head path for ordinary
                    # per-response requests and the independent replay oracle.
                    offset = selected[tile]
                    end = min(stop, offset + self.head_rows)
                    rows = slice(offset - start, end - start)
                    selected_tokens = target_ids[offset - prefix_length + 1:end - prefix_length + 1]
                else:
                    offsets = mx.array(selected[tile:tile + self.head_rows], mx.int32)
                    rows = offsets - start
                    selected_tokens = target_ids[offsets - prefix_length + 1]
                selected_hidden = hidden[0, rows]
                for operation in ("vocabulary_head", "head_norm_statistics", "head_softmax"):
                    key = "forward." + operation
                    self.saved_counters[key] = self.saved_counters.get(key, 0) + 1
                logp = self.head(selected_hidden, selected_tokens)
                mx.eval(logp)
                probabilities.append(logp)
                yield {'phase': 'head', 'segment': segment, 'rows': selected_hidden.shape[0]}
                if record:
                    heads.append((segment, rows, selected_hidden, selected_tokens))
            self.report(phase, segment - first_segment + 1, stop_segment - first_segment,
                        **({'branch': branch_index} if branch_index is not None else {}))
        return units, heads, probabilities, states

    def _head_vjp(self, branch, heads, current):
        loss, dcurrent = mx.value_and_grad(branch['objective'])(current)
        mx.eval(loss, dcurrent)
        top, cursor = {}, 0
        for segment, rows, hidden, tokens in heads:
            count = hidden.shape[0]
            for operation in ("vocabulary_head","head_norm_statistics","head_softmax"):
                key="reconstructed."+operation
                self.saved_counters[key]=self.saved_counters.get(key,0)+1
            _, (gradient,) = mx.vjp(lambda values: self.head(values, tokens), [hidden], [dcurrent[cursor:cursor + count]])
            mx.eval(gradient)
            if segment not in top:
                row_count = min(self.segment_tokens, branch['embeddings'].shape[1] - segment * self.segment_tokens)
                top[segment] = mx.zeros((1, row_count, hidden.shape[-1]), hidden.dtype)
            top[segment][:, rows] = gradient[None]
            cursor += count
            yield {'phase': 'head_backward', 'segment': segment, 'rows': count}
        if cursor != current.shape[0]:
            raise RuntimeError('Replay head gradients did not cover every target')
        mx.eval(list(top.values()))
        return loss, top

    def _layer_vjp(self, layer, hidden, sources, positions, cotangents, tape=None, context=None):
        """Differentiate one (segment, layer) unit with explicit inputs and outputs.

        Use a stable carry-before-hidden reverse schedule; keep real zero-valued
        gradients and omit only absent edges.
        """
        if tape is not None:
            from .saved import backward_saved
            return backward_saved(tape, cotangents, hidden_BLD=hidden)
        if self._can_compile_gdn(layer, hidden):
            return self._gdn_unit(layer).backward(
                hidden, [value for _, value in sources], positions, cotangents)
        order = (*range(1, len(cotangents)), 0)
        active = [index for index in order if cotangents[index] is not None]
        if not active:
            raise RuntimeError('Replay encountered a unit without a live output')
        names, values = zip(*tree_flatten(layer.trainable_parameters()))

        def unit(*arguments):
            layer.update(tree_unflatten(list(zip(names, arguments[:len(names)]))))
            outputs = self._layer(layer, arguments[len(names)], list(arguments[len(names) + 1:]), positions, context)
            return [outputs[index] for index in active]
        try:
            _, back = mx.vjp(unit, [*[value.astype(self.gradient_dtype) for value in values], hidden, *[value for _, value in sources]],
                             [cotangents[index] for index in active])
        finally:
            layer.update(tree_unflatten(list(zip(names, values))))
        return names, back

    def _backward_group(self, segment, units, indices, cotangent, positions, context,
                        gradients, gradient_sums, pending, pending_sums):
        """
        Purpose: Differentiate one checkpoint group of one segment with one VJP.
        Input: The recorded units, the group layer indices, the hidden cotangent
            at the group output, and the gradient and adjoint accumulators.
        Output: The hidden cotangent at the group input, or None for no edge.
        Note: The VJP rebuilds the group forward once from its checkpoint. So no
            separate recompute of the group inputs occurs. Each carry keeps its
            original producer identity; only carries with a pending adjoint seed it.
        """
        sources = [units[segment][index][1] for index in indices]
        carries = [
            [self._take(pending, (segment, index, kind), pending_sums)
             for kind in range(self.model_ops.carry_count(self.layers[index]))]
            for index in indices
        ]
        if cotangent is None and all(value is None for row in carries for value in row):
            for index in indices:
                units[segment][index] = None
            return None
        profile_started = time.perf_counter()
        names, back = self._group_vjp(
            indices, units[segment][indices[0]][0], sources, positions, cotangent,
            carries, context)
        cursor = 0
        for index, layer_names in zip(indices, names):
            for name, gradient in zip(layer_names, back[cursor:cursor + len(layer_names)]):
                key = f'{LAYER_PREFIX}{index}.{name}'
                self._add_gradient(gradients, key, gradient.astype(self.gradient_dtype),
                                   gradient_sums)
            cursor += len(layer_names)
        cotangent = back[cursor]
        cursor += 1
        for index, layer_sources in zip(indices, sources):
            for (source, _), gradient in zip(layer_sources, back[cursor:cursor + len(layer_sources)]):
                if source is None or not self.model_ops.differentiable_carry(self.layers[index], source[2]):
                    continue
                self._add_gradient(pending, source, gradient, pending_sums)
            cursor += len(layer_sources)
        # This graph was built while the previous group ran. Wait for that group,
        # then start this one, so only one group's buffers exist at a time.
        self._finish_inflight()
        self._inflight = [cotangent, gradients, list(pending.values()),
                          gradient_sums.arrays(), pending_sums.arrays()]
        mx.async_eval(self._inflight)
        if self.profile is not None:
            self.profile.record('vjp_total', indices[0], self.layers[indices[0]].is_linear,
                                time.perf_counter() - profile_started)
        for index in indices:
            units[segment][index] = None
        return cotangent

    def _group_vjp(self, indices, hidden, sources, positions, cotangent, carries, context):
        """
        Purpose: Run one VJP through consecutive layers from their checkpoint input.
        Input: Layer indices, the checkpoint hidden rows, each layer's sources, the
            output hidden cotangent, and each layer's carry cotangents.
        Output: Per-layer parameter names, and the derivatives of the parameters,
            the hidden input, and every source, in that order.
        Note: Only live outputs seed the VJP, carries before hidden, so an absent
            edge stays distinct from a zero cotangent.
        """
        layers = [self.layers[index] for index in indices]
        flat = [tree_flatten(layer.trainable_parameters()) for layer in layers]
        names = [[name for name, _ in values] for values in flat]
        values = [[value for _, value in layer_values] for layer_values in flat]
        source_values = [[value for _, value in layer_sources] for layer_sources in sources]
        live = [(position, kind) for position, row in enumerate(carries)
                for kind, value in enumerate(row) if value is not None]

        def unit(*arguments):
            cursor = 0
            for layer, layer_names in zip(layers, names):
                layer.update(tree_unflatten(list(zip(
                    layer_names, arguments[cursor:cursor + len(layer_names)]))))
                cursor += len(layer_names)
            hidden_BLD = arguments[cursor]
            cursor += 1
            produced = []
            for layer, layer_sources in zip(layers, source_values):
                states = list(arguments[cursor:cursor + len(layer_sources)])
                cursor += len(layer_sources)
                hidden_BLD, *carried = self._layer(layer, hidden_BLD, states, positions, context)
                produced.append(carried)
            outputs = [produced[position][kind] for position, kind in live]
            return outputs + ([hidden_BLD] if cotangent is not None else [])

        primals = [value.astype(self.gradient_dtype) for row in values for value in row]
        primals += [hidden, *[value for row in source_values for value in row]]
        seeds = [carries[position][kind] for position, kind in live]
        seeds += [cotangent] if cotangent is not None else []
        try:
            _, back = mx.vjp(unit, primals, seeds)
        finally:
            for layer, layer_names, layer_values in zip(layers, names, values):
                layer.update(tree_unflatten(list(zip(layer_names, layer_values))))
        return names, back

    def _add_gradient(self, target, key, value, sums):
        """
        Purpose: Add one contribution to a gradient or adjoint accumulator.
        Input: A dict of complete values, a key, the contribution, and the sums.
        Output: None. BF16 keeps the key only as partials in sums.
        Note: A complete value in target seeds the sums. _take and _complete
            compute the total only when a consumer needs it.
        """
        if self.execution_profile == "bf16-fast":
            if key in target:
                sums.add(key, target.pop(key))
            sums.add(key, value)
        else:
            target[key] = target[key] + value if key in target else value

    @staticmethod
    def _take(target, key, sums):
        """Remove and return the complete value of one key, or None."""
        return sums.pop(key) if key in sums else target.pop(key, None)

    @staticmethod
    def _complete(target, sums):
        """Move every total from sums into target, one key at a time."""
        for key in sums.keys():
            target[key] = sums.pop(key)
            mx.eval(target[key])
        return target

    @staticmethod
    def _key_arrays(target, key, sums):
        """Return the arrays that hold one key: its partials or its complete value."""
        return sums.partials(key) if key in sums else [target[key]]

    def _finish_inflight(self):
        """Wait for the group VJP that is still running, if any."""
        if self._inflight is not None:
            mx.eval(self._inflight)
            self._inflight = None

    def _gradient_sums(self):
        return BalancedSums(self.gradient_sum_levels)

    def _backward(self, branch, units, top, *, pending=None, capture_prefix_gradient=False, phase, branch_index=None):
        """Reverse every recorded segment; returns (adapter gradients, boundary adjoints, prefix gradients)."""
        pending = {} if pending is None else pending
        gradients, prefix_gradients = {}, []
        gradient_sums, pending_sums = self._gradient_sums(), BalancedSums()
        positions = branch['positions']
        segment_ids = sorted(units, reverse=True)
        for completed, segment in enumerate(segment_ids, 1):
            start = segment * self.segment_tokens
            stop = min(branch['embeddings'].shape[1], start + self.segment_tokens)
            cotangent = top.pop(segment, None)
            for group_start in reversed(range(0, len(self.layers), self.checkpoint_layers)):
                group_stop = min(len(self.layers), group_start + self.checkpoint_layers)
                if self.model_ops is not None and units[segment][group_start][2] is None:
                    cotangent = self._backward_group(
                        segment, units, range(group_start, group_stop), cotangent,
                        positions[:, :, start:stop], self._context(branch, start, stop, 'take'),
                        gradients, gradient_sums, pending, pending_sums)
                    yield {'phase': phase, 'segment': segment, 'layer_group': group_start, 'branch': branch_index}
                    continue
                hidden = units[segment][group_start][0]
                group_inputs = []
                # Recreate only this group's hidden inputs. Original incoming
                # states retain their producer identities; recomputed carries
                # are discarded, never substituted into that graph.
                for index in range(group_start, group_stop):
                    group_inputs.append(hidden)
                    if index + 1 < group_stop:
                        sources = units[segment][index][1]
                        tape = units[segment][index][2]
                        if tape is not None:
                            from .saved import reconstruct_output
                            hidden = reconstruct_output(tape, hidden)
                        else:
                            hidden, *carried = self._layer(self.layers[index], hidden, [value for _, value in sources],
                                                           positions[:, :, start:stop], self._context(branch, start, stop))
                            del carried
                        mx.eval(hidden)
                for index in reversed(range(group_start, group_stop)):
                    hidden = group_inputs.pop()
                    sources = units[segment][index][1]
                    carry_count = (self.model_ops.carry_count(self.layers[index]) if self.model_ops is not None else 2)
                    cotangents = [cotangent] + [self._take(pending, (segment, index, kind), pending_sums)
                                                for kind in range(carry_count)]
                    if not any(value is not None for value in cotangents):
                        units[segment][index] = None
                        continue
                    profile_started = time.perf_counter()
                    names, back = self._layer_vjp(self.layers[index], hidden, sources, positions[:, :, start:stop],
                                                  cotangents, units[segment][index][2], self._context(branch, start, stop))
                    profile_evaluate_started = time.perf_counter()
                    if self.profile is not None:
                        self.profile.record('vjp_build', index, self.layers[index].is_linear,
                                            profile_evaluate_started - profile_started)
                    for name, gradient in zip(names, back[:len(names)]):
                        key = f'{LAYER_PREFIX}{index}.{name}'
                        gradient = gradient.astype(self.gradient_dtype)
                        self._add_gradient(gradients, key, gradient, gradient_sums)
                    cotangent = back[len(names)]
                    for (source, _), gradient in zip(sources, back[len(names) + 1:]):
                        if self.model_ops is not None and source is not None and not self.model_ops.differentiable_carry(self.layers[index], source[2]):
                            continue
                        if source is not None:
                            self._add_gradient(pending, source, gradient, pending_sums)
                    mx.eval(cotangent, gradients, list(pending.values()), gradient_sums.arrays(),
                            pending_sums.arrays())
                    if self.profile is not None:
                        stopped = time.perf_counter()
                        self.profile.record('vjp_eval', index, self.layers[index].is_linear,
                                            stopped - profile_evaluate_started)
                        self.profile.record('vjp_total', index, self.layers[index].is_linear,
                                            stopped - profile_started)
                    units[segment][index] = None
                    del back, cotangents, hidden, sources
                yield {'phase': phase, 'segment': segment, 'layer_group': group_start, 'branch': branch_index}
            del units[segment]
            if capture_prefix_gradient and start < branch['prefix_length']:
                prefix_gradients.append((segment, (cotangent if cotangent is not None else
                    mx.zeros_like(branch['embeddings'][:, start:stop]))[:, :min(stop, branch['prefix_length']) - start]))
            self.report(phase, completed, len(segment_ids),
                        **({'branch': branch_index} if branch_index is not None else {}))
        self._finish_inflight()
        if top:
            raise RuntimeError('Replay left unconsumed head gradients')
        return (self._complete(gradients, gradient_sums), self._complete(pending, pending_sums),
                prefix_gradients)

    def run_tree(self, branches):
        """
        Purpose: Replay training sequences once per shared token-prefix segment.
        Input: Training branches with embeddings, positions, token_ids, tokens,
            objectives, and optional target_indices. Prompts may differ.
        Output: Per-branch logprobs, the summed loss, adapter gradients, and stats.
        Method:
            1. Build a segment trie. Branches share one complete segment when
               all earlier tokens and that segment are identical and every
               sharing branch has a later segment.
            2. Visit the trie depth first. A node runs its segments once and
               computes the head rows of every branch that targets them.
            3. A leaf completes its branch objective and assigns head cotangents.
            4. A node differentiates its segments after all of its children.
               Children leave carry adjoints for the node in one pending map.
        Note: Each branch sees the same segment inputs and carried state as an
            independent replay, so forward values are identical. Only the
            summation order of BF16 adjoints and adapter gradients changes.
        """
        started = time.monotonic()
        if not branches:
            raise ValueError('Replay requires at least one branch')
        for branch in branches:
            values, pos, tokens = branch['embeddings'], branch['positions'], branch['tokens']
            if values.ndim != 3 or values.shape[0] != 1 or pos.shape[:2] != (3, 1):
                raise ValueError('Replay requires one sequence with three-axis positions per branch')
            if (branch['prefix_length'] < 1 or not tokens
                    or values.shape[1] != branch['prefix_length'] + len(tokens) - 1
                    or pos.shape[-1] != values.shape[1]
                    or len(branch['token_ids']) != values.shape[1]):
                raise ValueError('Replay requires aligned complete prompts and shifted targets')
            if not callable(branch.get('objective')):
                raise ValueError('Tree replay requires a training objective per branch')
            indices = branch.get('target_indices')
            if indices is not None and (not indices or indices != sorted(set(indices))
                                        or any(type(i) is not int or not 0 <= i < len(tokens) for i in indices)):
                raise ValueError('Replay target indices must be nonempty, ordered, unique, and in bounds')
        tree = _segment_trie([branch['token_ids'] for branch in branches], self.segment_tokens)
        context = {
            'records': [[] for _ in branches], 'currents': [None] * len(branches),
            'losses': [None] * len(branches), 'pending': {}, 'carry_dtypes': {},
            'gradients': {}, 'gradient_sums': self._gradient_sums(),
            'forward_segments': 0, 'shared_segments': 0,
        }
        empty_states = (self.model_ops.initial_states(self.layers, branches[0]['embeddings'].dtype)
                        if self.model_ops is not None else
                        initial_states(self.layers, branches[0]['embeddings'].dtype, recurrent_dtype=self.gradient_dtype))
        for node in tree:
            yield from self._tree_visit(branches, node, empty_states, context)
        if context['pending']:
            raise RuntimeError('Tree replay left unconsumed carry adjoints')
        loss = sum(context['losses'])
        self._complete(context['gradients'], context['gradient_sums'])
        mx.eval(loss, context['gradients'])
        stats = {'branches': len(branches), 'tree_nodes': _count_nodes(tree),
                 'shared_segments': context['shared_segments'],
                 'shared_prefix_tokens': context['shared_segments'] * self.segment_tokens,
                 'branch_sequence_tokens': [branch['embeddings'].shape[1] for branch in branches],
                 'forward_segments': context['forward_segments'],
                 'backward_segments': context['forward_segments'],
                 'independent_segments': sum(_segment_count(branch['token_ids'], self.segment_tokens)
                                             for branch in branches),
                 'branch_losses': [float(value.item()) for value in context['losses']],
                 'hidden_checkpoint_layers': self.checkpoint_layers, 'flash_attention': self.flash is not None,
                 'segment_tokens': self.segment_tokens, 'head_rows': self.head_rows,
                 'replay_seconds': time.monotonic() - started,
                 'saved_activations': False,
                 'projection_counts': dict(self.saved_counters),
                 'activation_retention': {}}
        return context['currents'], loss, context['gradients'], stats

    def _tree_visit(self, branches, node, states, context):
        """Forward one trie node, visit its children, then differentiate the node."""
        first, stop, members, children = node
        shared = len(members) > 1
        branch = branches[members[0]]
        context['forward_segments'] += stop - first
        if shared:
            context['shared_segments'] += stop - first
        units, end_states, records = yield from self._tree_forward(
            branches, members, first, stop, states, context,
            phase='replay_shared_forward' if shared else 'replay_branch_forward')
        if children:
            for child in children:
                yield from self._tree_visit(branches, child, end_states, context)
        else:
            for member in members:
                self._finish_branch(branches[member], member, context)
        del end_states
        top = {}
        for record in records:
            basis, cotangent = record['basis'], record['cotangent']
            # Each logprob depends on its own row only, so the row gradient is
            # the cotangent times that row's unit-cotangent gradient.
            gradient = (basis.astype(mx.float32) * cotangent.astype(mx.float32)[:, None]).astype(basis.dtype)
            segment = record['segment']
            if segment not in top:
                row_count = min(self.segment_tokens, branch['embeddings'].shape[1] - segment * self.segment_tokens)
                top[segment] = mx.zeros((1, row_count, basis.shape[-1]), basis.dtype)
            top[segment] = top[segment].at[:, record['rows']].add(gradient[None])
            mx.eval(top[segment])
            yield {'phase': 'head_backward', 'segment': segment, 'rows': basis.shape[0]}
        del records
        pending = context['pending']
        for key in [key for key in pending if first <= key[0] < stop]:
            pending[key] = pending[key].astype(context['carry_dtypes'][key])
        gradients, pending, _ = yield from self._backward(
            branch, units, top, pending=pending,
            phase='replay_shared_backward' if shared else 'replay_branch_backward')
        # One key at a time: the node total and the merged partials never coexist in full.
        for name in list(gradients):
            self._add_gradient(context['gradients'], name, gradients.pop(name), context['gradient_sums'])
            mx.eval(self._key_arrays(context['gradients'], name, context['gradient_sums']))
        for key, gradient in list(pending.items()):
            if key[0] >= first:
                raise RuntimeError('A trie node left an adjoint for its own carry')
            pending[key] = gradient.astype(self.gradient_dtype)
        mx.eval(list(pending.values()))

    def _tree_forward(self, branches, members, first, stop, states, context, *, phase):
        """Run node segments once; record head rows for every member branch."""
        branch = branches[members[0]]
        embeddings, positions = branch['embeddings'], branch['positions']
        states = [list(values) for values in states]
        units, records = {}, []
        # The layer that runs while the host builds the next one.
        running = None
        for segment in range(first, stop):
            start = segment * self.segment_tokens
            end = min(embeddings.shape[1], start + self.segment_tokens)
            hidden = (branch['segment_embeddings'](start, end) if 'segment_embeddings' in branch
                      else embeddings[:, start:end])
            row = []
            for index, layer in enumerate(self.layers):
                incoming = list(states[index])
                checkpoint = hidden if index % self.checkpoint_layers == 0 else None
                profile_started = time.perf_counter()
                hidden, *carried = self._layer(layer, hidden, [value for _, value in incoming],
                                               positions[:, :, start:end], self._context(branch, start, end, 'keep'))
                # Wait for the previous layer, then start this one: one layer's
                # buffers at a time, and the host builds during the GPU work.
                if running is not None:
                    mx.eval(running)
                running = [hidden, carried]
                mx.async_eval(running)
                row.append((checkpoint, incoming, None))
                if self.profile is not None:
                    self.profile.record('forward', index, layer.is_linear,
                                        time.perf_counter() - profile_started)
                produced = [((segment, index, kind), value) for kind, value in enumerate(carried)]
                for key, value in produced:
                    context['carry_dtypes'][key] = value.dtype
                replace = (self.model_ops.replace_state(layer) if self.model_ops is not None else layer.is_linear)
                states[index] = produced if replace else incoming + produced
                if (index + 1) % self.checkpoint_layers == 0 or index + 1 == len(self.layers):
                    yield {'phase': phase, 'segment': segment, 'layer': index, 'branch': members[0]}
            units[segment] = row
            for member in members:
                for record in self._tree_heads(branches[member], hidden, segment, start, end):
                    context['records'][member].append(record)
                    records.append(record)
                    yield {'phase': 'head', 'segment': segment, 'rows': record['basis'].shape[0]}
            self.report(phase, segment - first + 1, stop - first)
        if running is not None:
            mx.eval(running)
        return units, states, records

    def _tree_heads(self, branch, hidden, segment, start, stop):
        """Yield the same 128-row head tiles that independent replay computes.

        Each record keeps the logprobs and the gradient of each logprob with
        respect to its own hidden row. The backward pass scales that basis by
        the loss cotangent and runs no second head pass.
        """
        prefix_length = branch['prefix_length']
        target_ids = branch.setdefault('target_array', mx.array(branch['tokens'], dtype=mx.int32))
        target_indices = branch.get('target_indices')
        selected = (range(max(start, prefix_length - 1), stop) if target_indices is None else
                    [prefix_length - 1 + i for i in target_indices if start <= prefix_length - 1 + i < stop])
        for tile in range(0, len(selected), self.head_rows):
            if target_indices is None:
                offset = selected[tile]
                end = min(stop, offset + self.head_rows)
                rows = mx.arange(offset - start, end - start, dtype=mx.int32)
                selected_hidden = hidden[0, offset - start:end - start]
                selected_tokens = target_ids[offset - prefix_length + 1:end - prefix_length + 1]
            else:
                offsets = mx.array(selected[tile:tile + self.head_rows], mx.int32)
                rows = offsets - start
                selected_hidden = hidden[0, rows]
                selected_tokens = target_ids[offsets - prefix_length + 1]
            for operation in ("vocabulary_head", "head_norm_statistics", "head_softmax"):
                key = "forward." + operation
                self.saved_counters[key] = self.saved_counters.get(key, 0) + 1
            # One head pass gives the logprobs and each row's unit-cotangent
            # gradient. The basis has the shape of the rows, so it costs no more.
            (logp,), (basis,) = mx.vjp(lambda values, tokens=selected_tokens: (self.head(values, tokens),),
                                       [selected_hidden],
                                       [mx.ones((selected_hidden.shape[0],), self.gradient_dtype)])
            mx.eval(logp, basis)
            yield {'segment': segment, 'rows': rows, 'basis': basis,
                   'logp': logp, 'cotangent': None}

    def _finish_branch(self, branch, member, context):
        """Evaluate one branch objective and assign each head tile its cotangent."""
        records = context['records'][member]
        current = mx.concatenate([record['logp'] for record in records])
        loss, dcurrent = mx.value_and_grad(branch['objective'])(current)
        mx.eval(current, loss, dcurrent)
        cursor = 0
        for record in records:
            count = record['basis'].shape[0]
            record['cotangent'] = dcurrent[cursor:cursor + count]
            cursor += count
        if cursor != current.shape[0]:
            raise RuntimeError('Replay head gradients did not cover every target')
        context['currents'][member], context['losses'][member] = current, loss
        context['records'][member] = None

    def run_group(self, branches, *, capture_prefix_gradient=False):
        """Replay completions of one prompt with shared prompt work and full-prompt gradients.

        A branch is a dict with ``embeddings``, ``positions``, ``prefix_length``,
        ``tokens`` and a scalar ``objective`` callable (None for forward-only).
        Only complete segments before the first target row are shared, so each
        branch keeps single-sequence segment boundaries.

        Returns (logprob arrays, summed loss, FP32 adapter gradients keyed by
        model path, stats). Optional
        ``stats['prefix_gradient']`` differentiates the common prompt, summed
        over branches.
        """
        started = time.monotonic()
        if not branches:
            raise ValueError('Replay requires at least one branch')
        first = branches[0]
        prefix_length = first['prefix_length']
        embeddings, positions = first['embeddings'], first['positions']
        training = first.get('objective') is not None
        for branch in branches:
            values, pos, tokens = branch['embeddings'], branch['positions'], branch['tokens']
            if values.ndim != 3 or values.shape[0] != 1 or pos.shape[:2] != (3, 1):
                raise ValueError('Replay requires one sequence with three-axis positions per branch')
            if (branch['prefix_length'] != prefix_length or prefix_length < 1 or not tokens
                    or values.shape[1] != prefix_length + len(tokens) - 1 or pos.shape[-1] != values.shape[1]):
                raise ValueError('Replay requires aligned complete prompts and shifted targets')
            indices = branch.get('target_indices')
            if indices is not None and (not indices or indices != sorted(set(indices))
                                        or any(type(i) is not int or not 0 <= i < len(tokens) for i in indices)):
                raise ValueError('Replay target indices must be nonempty, ordered, unique, and in bounds')
            if values.dtype != embeddings.dtype or values.shape[-1] != embeddings.shape[-1] or pos.dtype != positions.dtype:
                raise ValueError('Replay requires matching embedding and position layouts')
            if (branch.get('objective') is not None) != training:
                raise ValueError('Replay cannot mix training and forward-only branches')
            if training and not callable(branch['objective']):
                raise ValueError('Replay objectives must be callable')
            if branch is not first and (
                    not bool(mx.all(values[:, :prefix_length] == embeddings[:, :prefix_length]).item())
                    or not bool(mx.all(pos[:, :, :prefix_length] == positions[:, :, :prefix_length]).item())):
                raise ValueError('Replay prompts differ; sharing their state would be invalid')

        if training and self.saved_activations:
            from .retention import plan_retention
            self.retention_choices, self.retention_report = plan_retention(
                self.layers,max(branch['embeddings'].shape[1] for branch in branches),
                self.segment_tokens,self.activation_budget_bytes,self.projection_costs,
                self.checkpoint_layers,
            )
        shared_segments = (prefix_length - 1) // self.segment_tokens
        segment_counts = [(branch['embeddings'].shape[1] + self.segment_tokens - 1) // self.segment_tokens
                          for branch in branches]
        empty_states = (self.model_ops.initial_states(self.layers, embeddings.dtype) if self.model_ops is not None else initial_states(self.layers, embeddings.dtype, recurrent_dtype=self.gradient_dtype))
        shared_units, shared_heads, shared_logp, shared_states = yield from self._forward(
            first, 0, shared_segments, empty_states, record=training,
            phase='replay_shared_forward')
        if shared_heads or shared_logp:
            raise RuntimeError('The shared replay prompt unexpectedly contains supervised rows')
        # Every remaining branch dependency must point to this immutable boundary.
        boundary_dtypes = {key: value.dtype for states in shared_states for key, value in states if key is not None}
        pending, gradients, currents, losses, prefix_tail = {}, {}, [], [], None
        gradient_sums, boundary_sums, prefix_sums = self._gradient_sums(), BalancedSums(), BalancedSums()
        for branch_index, (branch, segments) in enumerate(zip(branches, segment_counts)):
            units, heads, probabilities, final_states = yield from self._forward(
                branch, shared_segments, segments, shared_states, record=training,
                phase='replay_branch_forward', branch_index=branch_index)
            del final_states
            current = mx.concatenate(probabilities)
            mx.eval(current)
            currents.append(current)
            if not training:
                continue
            loss, top = yield from self._head_vjp(branch, heads, current)
            losses.append(loss)
            del heads, probabilities
            branch_gradients, branch_pending, branch_prefix = yield from self._backward(
                branch, units, top, capture_prefix_gradient=capture_prefix_gradient,
                phase='replay_branch_backward', branch_index=branch_index)
            for name, gradient in branch_gradients.items():
                self._add_gradient(gradients, name, gradient, gradient_sums)
            for key, gradient in branch_pending.items():
                if key not in boundary_dtypes:
                    raise RuntimeError('A completion left an invalid shared-prompt gradient')
                gradient = gradient.astype(self.gradient_dtype)
                self._add_gradient(pending, key, gradient, boundary_sums)
            if capture_prefix_gradient and branch_prefix:
                tail = mx.concatenate([value for _, value in sorted(branch_prefix)], axis=1).astype(self.gradient_dtype)
                if self.execution_profile == "bf16-fast":
                    prefix_sums.add("prefix", tail)
                else:
                    prefix_tail = tail if prefix_tail is None else prefix_tail + tail
            mx.eval(gradients, list(pending.values()), [] if prefix_tail is None else [prefix_tail],
                    gradient_sums.arrays(), boundary_sums.arrays(), prefix_sums.arrays())
            del branch_gradients, branch_pending, branch_prefix
        del shared_states
        if "prefix" in prefix_sums:
            prefix_tail = prefix_sums.pop("prefix")

        prefix_gradients, loss = [], None
        if training:
            self._complete(pending, boundary_sums)
            pending = {key: gradient.astype(boundary_dtypes[key]) for key, gradient in pending.items()}
            mx.eval(list(pending.values()))
            shared_gradients, pending, prefix_gradients = yield from self._backward(
                first, shared_units, {}, pending=pending, capture_prefix_gradient=capture_prefix_gradient,
                phase='replay_shared_backward')
            if pending:
                raise RuntimeError('Replay left unconsumed shared-prompt gradients')
            for name, gradient in shared_gradients.items():
                self._add_gradient(gradients, name, gradient, gradient_sums)
            self._complete(gradients, gradient_sums)
            loss = sum(losses)
            mx.eval(loss, gradients)
        forward_segments = shared_segments + sum(count - shared_segments for count in segment_counts)
        stats = {'branches': len(branches), 'shared_segments': shared_segments,
                 'shared_prefix_tokens': shared_segments * self.segment_tokens,
                 'branch_sequence_tokens': [branch['embeddings'].shape[1] for branch in branches],
                 'forward_segments': forward_segments, 'backward_segments': forward_segments if training else 0,
                 'branch_losses': [float(value.item()) for value in losses],
                 'hidden_checkpoint_layers': self.checkpoint_layers, 'flash_attention': self.flash is not None,
                 'segment_tokens': self.segment_tokens, 'head_rows': self.head_rows,
                 'replay_seconds': time.monotonic() - started,
                 'saved_activations': self.saved_activations,
                 'projection_counts': dict(self.saved_counters),
                 'activation_retention': dict(self.retention_report)}
        if capture_prefix_gradient and training:
            parts = [value for _, value in sorted(prefix_gradients)]
            if prefix_tail is not None:
                parts.append(prefix_tail.astype(embeddings.dtype))
            stats['prefix_gradient'] = mx.concatenate(parts, axis=1)
            mx.eval(stats['prefix_gradient'])
        return currents, loss, gradients if training else None, stats


def _segment_trie(token_ids_list, segment_tokens):
    """
    Purpose: Build the segment trie that run_tree visits.
    Input: The token IDs of each branch and the segment length.
    Output: A list of root nodes. A node is (first, stop, members, children).
    Note: Members share segments [first, stop) only when the members have the same
        tokens through segment stop - 1 and each member has a later segment.
        A node without children has one member and runs to the end of that branch.
    """
    return _trie_nodes(list(range(len(token_ids_list))), token_ids_list, 0, segment_tokens)


def tree_gradient_copies(token_ids_list, segment_tokens, max_levels=None):
    """
    Purpose: Count the most full adapter-gradient copies that run_tree keeps at one time.
    Input: The token IDs of each branch, the segment length, and the sum level cap.
    Output: The peak number of gradient partials, node and tree sums together.
    Method:
        1. Visit the trie nodes in the order that run_tree differentiates them.
        2. A node backward adds one gradient for each node segment to a node sum.
        3. The node total then goes into the tree sum.
        4. Count the partials after each addition.
    Note: The sums get integer contributions, so they follow the same carry rules.
    """
    tree_sums = BalancedSums(max_levels)
    peak = 0
    for segments in _backward_segments(_segment_trie(token_ids_list, segment_tokens)):
        node_sums = BalancedSums(max_levels)
        for _ in range(segments):
            node_sums.add('gradient', 1)
            peak = max(peak, node_sums.partial_count('gradient') + tree_sums.partial_count('gradient'))
        tree_sums.add('gradient', 1)
    return max(peak, tree_sums.partial_count('gradient'))


def _backward_segments(nodes):
    """Return the segment count of each node in child-before-parent order."""
    counts = []
    for first, stop, _, children in nodes:
        counts.extend(_backward_segments(children))
        counts.append(stop - first)
    return counts


def _trie_nodes(members, token_ids_list, first, segment_tokens):
    """Group members that share segment ``first``, then extend each shared run."""
    groups = {}
    for member in members:
        groups.setdefault(_segment_key(token_ids_list[member], first, segment_tokens, member), []).append(member)
    nodes = []
    for group in groups.values():
        if len(group) == 1:
            nodes.append((first, _segment_count(token_ids_list[group[0]], segment_tokens), group, []))
            continue
        stop = first + 1
        while len({_segment_key(token_ids_list[member], stop, segment_tokens, member) for member in group}) == 1:
            stop += 1
        nodes.append((first, stop, group, _trie_nodes(group, token_ids_list, stop, segment_tokens)))
    return nodes


def _segment_key(token_ids, segment, segment_tokens, member):
    """Return the shareable tokens of one segment, or a key unique to the member."""
    if _segment_count(token_ids, segment_tokens) <= segment + 1:
        return ('own', member)
    return tuple(token_ids[segment * segment_tokens:(segment + 1) * segment_tokens])


def _segment_count(token_ids, segment_tokens):
    return -(-len(token_ids) // segment_tokens)


def _count_nodes(tree):
    return sum(1 + _count_nodes(children) for _, _, _, children in tree)
