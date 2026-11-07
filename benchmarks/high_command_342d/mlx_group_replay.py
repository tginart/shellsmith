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
   branches' boundary-state adjoints accumulate in FP32 and seed one backward
   through the shared prompt. Adapter gradients accumulate in FP32.

Saved for backward: every layer's segment input by default (checkpoint_layers
= 1; 640 KiB per token on Qwen3.8-27B). Raise it (4) only when tokens x 640 KiB
exceeds the memory budget.

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

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

LAYER_PREFIX = 'layers.'
from studio_compute.numerics.cispo import frozen_head_logprobs
HEAD_ROWS = 128
from .mlx_group_ops import initial_states, segment_layer

SEGMENT_TOKENS = 2048


class Replay:
    def __init__(self, model, *, segment_tokens=SEGMENT_TOKENS, checkpoint_layers=1, flash=None, progress=None):
        self.model = model
        self.lm = model.language_model
        self.layers = self.lm.layers
        if segment_tokens < 1 or checkpoint_layers < 1:
            raise ValueError('Replay segment and checkpoint intervals must be positive')
        self.segment_tokens = segment_tokens
        self.checkpoint_layers = checkpoint_layers
        self.flash = flash
        self.progress = progress

    def report(self, phase, segment, total, **details):
        if self.progress:
            self.progress({'phase': phase, 'segment': segment, 'segments': total,
                           'active_memory_bytes': mx.get_active_memory(),
                           'peak_memory_bytes': mx.get_peak_memory(), **details})

    def _layer(self, layer, hidden, states, positions):
        return segment_layer(layer, hidden, states, positions, flash=self.flash)

    def head(self, hidden, tokens):
        return frozen_head_logprobs(self.lm, hidden, tokens)

    def run(self, embeddings, positions, prefix_length, tokens, objective=None, *, capture_prefix_gradient=False):
        """One sequence: selected log probabilities, optionally objective and adapter VJP.

        ``embeddings`` contains the full prompt followed by tokens[:-1].
        ``objective`` is a scalar function of the selected-token log probabilities.
        """
        probabilities, loss, gradients, stats = self.run_group([{
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
        states = [list(values) for values in states]
        units, heads, probabilities = {}, [], []
        for segment in range(first_segment, stop_segment):
            start = segment * self.segment_tokens
            stop = min(embeddings.shape[1], start + self.segment_tokens)
            hidden = embeddings[:, start:stop]
            row = []
            for index, layer in enumerate(self.layers):
                incoming = list(states[index])
                if record:
                    row.append((hidden if index % self.checkpoint_layers == 0 else None, incoming))
                hidden, *carried = self._layer(layer, hidden, [value for _, value in incoming], positions[:, :, start:stop])
                mx.eval(hidden, carried)
                produced = [((segment, index, kind), value) for kind, value in enumerate(carried)]
                states[index] = produced if layer.is_linear else incoming + produced
            if record:
                units[segment] = row
            for offset in range(max(start, prefix_length - 1), stop, HEAD_ROWS):
                end = min(stop, offset + HEAD_ROWS)
                selected_hidden = hidden[0, offset - start:end - start]
                selected_tokens = target_ids[offset - prefix_length + 1:end - prefix_length + 1]
                logp = self.head(selected_hidden, selected_tokens)
                mx.eval(logp)
                probabilities.append(logp)
                if record:
                    heads.append((segment, offset - start, selected_hidden, selected_tokens))
            self.report(phase, segment - first_segment + 1, stop_segment - first_segment,
                        **({'branch': branch_index} if branch_index is not None else {}))
        return units, heads, probabilities, states

    def _head_vjp(self, branch, heads, current):
        loss, dcurrent = mx.value_and_grad(branch['objective'])(current)
        mx.eval(loss, dcurrent)
        top, cursor = {}, 0
        for segment, offset, hidden, tokens in heads:
            count = hidden.shape[0]
            _, (gradient,) = mx.vjp(lambda values: self.head(values, tokens), [hidden], [dcurrent[cursor:cursor + count]])
            mx.eval(gradient)
            if segment not in top:
                rows = min(self.segment_tokens, branch['embeddings'].shape[1] - segment * self.segment_tokens)
                top[segment] = mx.zeros((1, rows, hidden.shape[-1]), hidden.dtype)
            top[segment][:, offset:offset + count] = gradient[None]
            cursor += count
        if cursor != current.shape[0]:
            raise RuntimeError('Replay head gradients did not cover every target')
        mx.eval(list(top.values()))
        return loss, top

    def _layer_vjp(self, layer, hidden, sources, positions, cotangents):
        """Differentiate one (segment, layer) unit with explicit inputs and outputs.

        Use a stable carry-before-hidden reverse schedule; keep real zero-valued
        gradients and omit only absent edges.
        """
        active = [index for index in (1, 2, 0) if cotangents[index] is not None]
        if not active:
            raise RuntimeError('Replay encountered a unit without a live output')
        names, values = zip(*tree_flatten(layer.trainable_parameters()))

        def unit(*arguments):
            layer.update(tree_unflatten(list(zip(names, arguments[:len(names)]))))
            outputs = self._layer(layer, arguments[len(names)], list(arguments[len(names) + 1:]), positions)
            return [outputs[index] for index in active]
        try:
            _, back = mx.vjp(unit, [*[value.astype(mx.float32) for value in values], hidden, *[value for _, value in sources]],
                             [cotangents[index] for index in active])
        finally:
            layer.update(tree_unflatten(list(zip(names, values))))
        return names, back

    def _backward(self, branch, units, top, *, pending=None, capture_prefix_gradient=False, phase, branch_index=None):
        """Reverse every recorded segment; returns (adapter gradients, boundary adjoints, prefix gradients)."""
        pending = {} if pending is None else pending
        gradients, prefix_gradients = {}, []
        positions = branch['positions']
        segment_ids = sorted(units, reverse=True)
        for completed, segment in enumerate(segment_ids, 1):
            start = segment * self.segment_tokens
            stop = min(branch['embeddings'].shape[1], start + self.segment_tokens)
            cotangent = top.pop(segment, None)
            for group_start in reversed(range(0, len(self.layers), self.checkpoint_layers)):
                group_stop = min(len(self.layers), group_start + self.checkpoint_layers)
                hidden = units[segment][group_start][0]
                group_inputs = []
                # Recreate only this group's hidden inputs. Original incoming
                # states retain their producer identities; recomputed carries
                # are discarded, never substituted into that graph.
                for index in range(group_start, group_stop):
                    group_inputs.append(hidden)
                    if index + 1 < group_stop:
                        sources = units[segment][index][1]
                        hidden, *carried = self._layer(self.layers[index], hidden, [value for _, value in sources],
                                                       positions[:, :, start:stop])
                        mx.eval(hidden)
                        del carried
                for index in reversed(range(group_start, group_stop)):
                    hidden = group_inputs.pop()
                    sources = units[segment][index][1]
                    cotangents = [cotangent] + [pending.pop((segment, index, kind), None) for kind in range(2)]
                    names, back = self._layer_vjp(self.layers[index], hidden, sources, positions[:, :, start:stop],
                                                  cotangents)
                    for name, gradient in zip(names, back[:len(names)]):
                        key = f'{LAYER_PREFIX}{index}.{name}'
                        gradient = gradient.astype(mx.float32)
                        gradients[key] = gradients[key] + gradient if key in gradients else gradient
                    cotangent = back[len(names)]
                    for (source, _), gradient in zip(sources, back[len(names) + 1:]):
                        if source is not None:
                            pending[source] = pending[source] + gradient if source in pending else gradient
                    mx.eval(cotangent, gradients, list(pending.values()))
                    units[segment][index] = None
                    del back, cotangents, hidden, sources
            del units[segment]
            if capture_prefix_gradient and start < branch['prefix_length']:
                prefix_gradients.append((segment, cotangent[:, :min(stop, branch['prefix_length']) - start]))
            self.report(phase, completed, len(segment_ids),
                        **({'branch': branch_index} if branch_index is not None else {}))
        if top:
            raise RuntimeError('Replay left unconsumed head gradients')
        return gradients, pending, prefix_gradients

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

        shared_segments = (prefix_length - 1) // self.segment_tokens
        segment_counts = [(branch['embeddings'].shape[1] + self.segment_tokens - 1) // self.segment_tokens
                          for branch in branches]
        shared_units, shared_heads, shared_logp, shared_states = self._forward(
            first, 0, shared_segments, initial_states(self.layers, embeddings.dtype), record=training,
            phase='replay_shared_forward')
        if shared_heads or shared_logp:
            raise RuntimeError('The shared replay prompt unexpectedly contains supervised rows')
        # Every remaining branch dependency must point to this immutable boundary.
        boundary_dtypes = {key: value.dtype for states in shared_states for key, value in states if key is not None}
        pending, gradients, currents, losses, prefix_tail = {}, {}, [], [], None
        for branch_index, (branch, segments) in enumerate(zip(branches, segment_counts)):
            units, heads, probabilities, final_states = self._forward(
                branch, shared_segments, segments, shared_states, record=training,
                phase='replay_branch_forward', branch_index=branch_index)
            del final_states
            current = mx.concatenate(probabilities)
            mx.eval(current)
            currents.append(current)
            if not training:
                continue
            loss, top = self._head_vjp(branch, heads, current)
            losses.append(loss)
            del heads, probabilities
            branch_gradients, branch_pending, branch_prefix = self._backward(
                branch, units, top, capture_prefix_gradient=capture_prefix_gradient,
                phase='replay_branch_backward', branch_index=branch_index)
            for name, gradient in branch_gradients.items():
                gradients[name] = gradients[name] + gradient if name in gradients else gradient
            for key, gradient in branch_pending.items():
                if key not in boundary_dtypes:
                    raise RuntimeError('A completion left an invalid shared-prompt gradient')
                gradient = gradient.astype(mx.float32)
                pending[key] = pending[key] + gradient if key in pending else gradient
            if capture_prefix_gradient and branch_prefix:
                tail = mx.concatenate([value for _, value in sorted(branch_prefix)], axis=1).astype(mx.float32)
                prefix_tail = tail if prefix_tail is None else prefix_tail + tail
            mx.eval(gradients, list(pending.values()), [] if prefix_tail is None else [prefix_tail])
            del branch_gradients, branch_pending, branch_prefix
        del shared_states

        prefix_gradients, loss = [], None
        if training:
            pending = {key: gradient.astype(boundary_dtypes[key]) for key, gradient in pending.items()}
            mx.eval(list(pending.values()))
            shared_gradients, pending, prefix_gradients = self._backward(
                first, shared_units, {}, pending=pending, capture_prefix_gradient=capture_prefix_gradient,
                phase='replay_shared_backward')
            if pending:
                raise RuntimeError('Replay left unconsumed shared-prompt gradients')
            for name, gradient in shared_gradients.items():
                gradients[name] = gradients[name] + gradient if name in gradients else gradient
            loss = sum(losses)
            mx.eval(loss, gradients)
        forward_segments = shared_segments + sum(count - shared_segments for count in segment_counts)
        stats = {'branches': len(branches), 'shared_segments': shared_segments,
                 'shared_prefix_tokens': shared_segments * self.segment_tokens,
                 'branch_sequence_tokens': [branch['embeddings'].shape[1] for branch in branches],
                 'forward_segments': forward_segments, 'backward_segments': forward_segments if training else 0,
                 'branch_losses': [float(value.item()) for value in losses],
                 'hidden_checkpoint_layers': self.checkpoint_layers, 'flash_attention': self.flash is not None,
                 'replay_seconds': time.monotonic() - started}
        if capture_prefix_gradient and training:
            parts = [value for _, value in sorted(prefix_gradients)]
            if prefix_tail is not None:
                parts.append(prefix_tail.astype(embeddings.dtype))
            stats['prefix_gradient'] = mx.concatenate(parts, axis=1)
            mx.eval(stats['prefix_gradient'])
        return currents, loss, gradients if training else None, stats
