"""Sequential full-prompt replay with explicit differentiable segment carries.

Adapted from High Command 10e830's audited segment-layer VJP schedule. Every
layer boundary is BF16; attention keys/values and GDN recurrent state have
producer identities whose adjoints survive across segment boundaries. Only
selected prediction rows reach the vocabulary head, in bounded tiles.
"""

from __future__ import annotations

import time

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from .cispo import frozen_head_logprobs
from .ops import initial_states, segment_layer

SEGMENT_TOKENS, HEAD_ROWS = 2048, 128


class Replay:
    def __init__(self, language_model, *, segment_tokens=SEGMENT_TOKENS, progress=None, memory_gib=128,
                 qualification_dtype=None, checkpoint_layers=1):
        if type(segment_tokens) is not int or segment_tokens < 1:
            raise ValueError("Replay segment size must be a positive integer.")
        self.lm, self.layers = language_model, language_model.layers
        self.segment_tokens, self.progress, self.memory_gib = segment_tokens, progress, memory_gib
        if type(checkpoint_layers) is not int or not 1 <= checkpoint_layers <= len(self.layers):
            raise ValueError("Replay checkpoint interval must fit the decoder layer count.")
        self.checkpoint_layers = checkpoint_layers
        # FP32 is an explicit numerical-qualification path, never a run setting.
        self.boundary_dtype = mx.bfloat16 if qualification_dtype is None else qualification_dtype
        if self.boundary_dtype not in (mx.bfloat16, mx.float32):
            raise ValueError("Replay qualification dtype must be FP32 or production BF16.")

    def report(self, phase, completed, total):
        if self.progress:
            self.progress({"phase": phase, "segment": completed, "segments": total,
                           "active_memory_bytes": mx.get_active_memory(), "peak_memory_bytes": mx.get_peak_memory()})

    def memory_estimate(self, length, width, size, training):
        checkpoints = (len(self.layers) + self.checkpoint_layers - 1) // self.checkpoint_layers
        boundary = length * width * checkpoints * size if training else 0
        # Recomputed boundaries exist for only one segment/block at a time.
        working = min(length, self.segment_tokens) * width * self.checkpoint_layers * size if training else 0
        carries, carry_adjoints = 0, 0
        segments = (length + self.segment_tokens - 1) // self.segment_tokens
        for layer in self.layers:
            if layer.is_linear:
                mixer = layer.linear_attn
                carries += segments * (mixer.num_v_heads * mixer.head_v_dim * mixer.head_k_dim * 4 +
                                       (mixer.conv_kernel_size - 1) * mixer.conv_dim * size)
                if training:
                    # Recurrent adjoints only reach the preceding segment;
                    # attention adjoints can reach every earlier KV producer.
                    carry_adjoints += (mixer.num_v_heads * mixer.head_v_dim * mixer.head_k_dim * 4 +
                                       (mixer.conv_kernel_size - 1) * mixer.conv_dim * size)
            else:
                attn = layer.self_attn
                carries += length * attn.num_key_value_heads * attn.head_dim * 2 * size
                if training:
                    carry_adjoints += length * attn.num_key_value_heads * attn.head_dim * 2 * size
        return {"boundary_bytes": boundary, "carry_bytes": carries, "carry_adjoint_bytes": carry_adjoints,
                "recompute_working_bytes": working, "reserve_bytes": 2 << 30,
                "additional_bytes": boundary + carries + carry_adjoints + working + (2 << 30)}

    def _admit(self, embeddings, training):
        estimate = self.memory_estimate(*embeddings.shape[1:], embeddings.nbytes // embeddings.size, training)
        if self.memory_gib is None:
            return estimate
        needed = estimate["additional_bytes"]
        if mx.get_active_memory() + needed > self.memory_gib * 1024 ** 3:
            raise MemoryError(f"MLX full-context replay requires approximately {needed / 1024 ** 3:.2f} GiB additional memory; "
                              "the explicit context/profile must be reduced before collecting rollouts.")
        return estimate

    def head(self, hidden, tokens):
        return frozen_head_logprobs(self.lm, hidden, tokens)

    def run(self, prompt_ids, tokens, objective=None, *, capture_prefix_gradient=False):
        """Replay exact prompt + sampled_tokens[:-1], preserving prompt gradients."""
        prompt_ids, tokens = tuple(prompt_ids), tuple(tokens)
        if not prompt_ids or not tokens:
            raise ValueError("Replay needs a nonempty prompt and sampled completion.")
        ids = mx.array([prompt_ids + tokens[:-1]], dtype=mx.int32)
        embeddings = self.lm.model.embed_tokens(ids).astype(self.boundary_dtype)
        positions = mx.broadcast_to(mx.arange(ids.shape[1], dtype=mx.int32)[None, None], (3, 1, ids.shape[1]))
        return self.run_embeddings(embeddings, positions, len(prompt_ids), tokens, objective,
                                   capture_prefix_gradient=capture_prefix_gradient)

    def run_embeddings(self, embeddings, positions, prefix_length, tokens, objective=None, *, capture_prefix_gradient=False):
        started = time.monotonic()
        training = objective is not None
        if (embeddings.ndim != 3 or embeddings.shape[0] != 1 or positions.shape != (3, 1, embeddings.shape[1])
                or prefix_length < 1 or not tokens or embeddings.shape[1] != prefix_length + len(tokens) - 1):
            raise ValueError("Replay requires aligned full prompts, three-axis positions, and shifted completion targets.")
        if embeddings.dtype != self.boundary_dtype:
            raise ValueError("Replay boundaries differ from the selected numerical qualification dtype.")
        estimate = self._admit(embeddings, training)
        segments = [(start, min(embeddings.shape[1], start + self.segment_tokens))
                    for start in range(0, embeddings.shape[1], self.segment_tokens)]
        states = initial_states(self.layers, embeddings.dtype)
        units, heads, probabilities = {}, [], []
        targets = mx.array(tokens, dtype=mx.int32)
        boundary_bytes = 0
        for segment, (start, stop) in enumerate(segments):
            hidden = embeddings[:, start:stop]
            row = []
            for index, layer in enumerate(self.layers):
                incoming = list(states[index])
                if training:
                    saved = hidden if index % self.checkpoint_layers == 0 else None
                    row.append((saved, incoming))
                    boundary_bytes += saved.nbytes if saved is not None else 0
                hidden, *carried = segment_layer(layer, hidden, [value for _, value in incoming], positions[:, :, start:stop])
                mx.eval(hidden, carried)
                produced = [((segment, index, kind), value) for kind, value in enumerate(carried)]
                states[index] = produced if layer.is_linear else incoming + produced
            if training:
                units[segment] = row
            for offset in range(max(start, prefix_length - 1), stop, HEAD_ROWS):
                end = min(stop, offset + HEAD_ROWS)
                selected = hidden[0, offset - start:end - start]
                selected_tokens = targets[offset - prefix_length + 1:end - prefix_length + 1]
                logprob = self.head(selected, selected_tokens)
                mx.eval(logprob)
                probabilities.append(logprob)
                if training:
                    heads.append((segment, offset - start, selected, selected_tokens))
            self.report("replay_forward", segment + 1, len(segments))
        del states
        current = mx.concatenate(probabilities)
        mx.eval(current)
        if current.shape != (len(tokens),):
            raise RuntimeError("Replay did not score every sampled token exactly once.")
        loss, gradients, prefix_gradients, carry_gradients = None, None, [], {}
        if training:
            loss, dcurrent = mx.value_and_grad(objective)(current)
            mx.eval(loss, dcurrent)
            top, cursor = {}, 0
            for segment, offset, selected, selected_tokens in heads:
                count = selected.shape[0]
                _, (gradient,) = mx.vjp(lambda value: self.head(value, selected_tokens), [selected],
                                       [dcurrent[cursor:cursor + count]])
                mx.eval(gradient)
                if segment not in top:
                    start, stop = segments[segment]
                    top[segment] = mx.zeros((1, stop - start, selected.shape[-1]), selected.dtype)
                top[segment][:, offset:offset + count] = gradient[None]
                cursor += count
            del heads, probabilities, dcurrent
            pending, gradients = {}, {}
            for completed, segment in enumerate(reversed(range(len(segments))), 1):
                start, stop = segments[segment]
                cotangent = top.pop(segment, None)
                recomputed = {}
                for index in reversed(range(len(self.layers))):
                    layer = self.layers[index]
                    hidden, sources = units[segment][index]
                    if self.checkpoint_layers > 1:
                        # Rebuild each block with the identical BF16 parameters,
                        # positions and incoming carries used by its forward.
                        # Carries keep their original producer IDs: their VJPs
                        # below still flow to every earlier segment, including
                        # the entire prompt. No gradient is detached or cropped.
                        if not recomputed:
                            block = (index // self.checkpoint_layers) * self.checkpoint_layers
                            value = units[segment][block][0]
                            for previous in range(block, index + 1):
                                recomputed[previous] = value
                                if previous < index:
                                    value, *unused = segment_layer(self.layers[previous], value,
                                        [carry for _, carry in units[segment][previous][1]], positions[:, :, start:stop])
                                    mx.eval(value)
                                    del unused
                            del value
                        hidden = recomputed.pop(index)
                    cotangents = [cotangent] + [pending.pop((segment, index, kind), None) for kind in range(2)]
                    if capture_prefix_gradient:
                        for kind in range(2):
                            if cotangents[kind + 1] is not None:
                                carry_gradients[(segment, index, kind)] = cotangents[kind + 1]
                    active = [kind for kind in (1, 2, 0) if cotangents[kind] is not None]
                    if not active:
                        raise RuntimeError("Replay encountered a layer without a live output adjoint.")
                    named = tree_flatten(layer.trainable_parameters())
                    if not named:
                        raise RuntimeError("Replay requires the all-layer canonical adapter.")
                    names, values = zip(*named)

                    def unit(*arguments):
                        layer.update(tree_unflatten(list(zip(names, arguments[:len(names)]))))
                        outputs = segment_layer(layer, arguments[len(names)], list(arguments[len(names) + 1:]),
                                                positions[:, :, start:stop])
                        return [outputs[kind] for kind in active]

                    try:
                        _, backward = mx.vjp(unit, [*[value.astype(mx.float32) for value in values], hidden,
                                                  *[value for _, value in sources]],
                                             [cotangents[kind] for kind in active])
                    finally:
                        layer.update(tree_unflatten(list(zip(names, values))))
                    for name, gradient in zip(names, backward[:len(names)]):
                        key = f"layers.{index}.{name}"
                        gradient = gradient.astype(mx.float32)
                        gradients[key] = gradients[key] + gradient if key in gradients else gradient
                    cotangent = backward[len(names)]
                    for (source, _), gradient in zip(sources, backward[len(names) + 1:]):
                        if source is not None:
                            pending[source] = pending[source] + gradient if source in pending else gradient
                    mx.eval(cotangent, gradients, list(pending.values()))
                    units[segment][index] = None
                    del backward, hidden, sources, cotangents
                del units[segment]
                if capture_prefix_gradient and start < prefix_length:
                    prefix_gradients.append((start, cotangent[:, :min(stop, prefix_length) - start]))
                self.report("replay_backward", completed, len(segments))
            if top or pending:
                raise RuntimeError("Replay left unconsumed hidden/carry adjoints.")
            mx.eval(loss, gradients)
        stats = {"sequence_tokens": embeddings.shape[1], "prompt_tokens": prefix_length, "completion_tokens": len(tokens),
                 "segments": len(segments), "hidden_checkpoint_layers": self.checkpoint_layers,
                 "hidden_checkpoint_storage": "bf16" if self.boundary_dtype == mx.bfloat16 else "fp32",
                 "boundary_stored_bytes": boundary_bytes, "replay_seconds": time.monotonic() - started,
                 "peak_memory_bytes": mx.get_peak_memory(), "memory_estimate": estimate}
        if capture_prefix_gradient and training:
            stats["prefix_gradient"] = mx.concatenate([value for _, value in sorted(prefix_gradients)], axis=1)
            stats["carry_gradients"] = carry_gradients
            mx.eval(stats["prefix_gradient"], list(carry_gradients.values()))
        return current, loss, gradients, stats
