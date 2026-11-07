"""Tinker SDK calls behind Shellsmith's backend boundary."""

from __future__ import annotations
import asyncio


def datum_without_mask(*, deps, datum):
    return deps.types.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={
            key: value for key, value in datum.loss_fn_inputs.items() if key != "mask"
        },
    )


class TinkerBackend:
    def __init__(
        self,
        *,
        deps,
        training_client,
        sampler=None,
        service_client=None,
        reference_client=None,
    ):
        self.deps, self.training_client, self.sampler = deps, training_client, sampler
        self.service_client, self.reference_client = service_client, reference_client
        self.policy_ref, self.policy_identity = "", {"backend": "tinker"}
        self.operations = {}

    @classmethod
    async def for_sampling(
        cls,
        *,
        deps,
        config,
        sampler,
        policy=None,
        require_context=None,
        before_setup=None,
    ):
        if before_setup:
            before_setup()
        service = deps.tinker.ServiceClient()
        if require_context:
            await require_context(service, model_name=config.player.base_model)
        sampling = await (
            service.create_sampling_client_async(model_path=policy)
            if policy
            else service.create_sampling_client_async(
                base_model=config.player.base_model
            )
        )
        sampler.set_sampling_client(sampling)
        adapter = cls(
            deps=deps, training_client=None, sampler=sampler, service_client=service
        )
        adapter.policy_ref = policy or "tinker-base:" + config.player.base_model
        return adapter, sampling

    @classmethod
    async def create(
        cls,
        *,
        deps,
        config,
        sampler,
        checkpoint=None,
        require_context=None,
        before_training=None,
    ):
        service = deps.tinker.ServiceClient()
        if require_context:
            await require_context(service, model_name=config.player.base_model)
        if before_training:
            before_training()
        if checkpoint:
            training = (
                await service.create_training_client_from_state_with_optimizer_async(
                    checkpoint["path"]
                )
            )
            sampling = await service.create_sampling_client_async(
                model_path=checkpoint["sampler_path"]
            )
        else:
            training = await service.create_lora_training_client_async(
                base_model=config.player.base_model, rank=config.player.lora_rank
            )
            sampling = await service.create_sampling_client_async(
                base_model=config.player.base_model
            )
        reference = (
            await service.create_sampling_client_async(
                base_model=config.player.base_model
            )
            if config.train.kl_penalty_coef > 0
            else None
        )
        adapter = cls(
            deps=deps,
            training_client=training,
            sampler=sampler,
            service_client=service,
            reference_client=reference,
        )
        sampler.set_sampling_client(sampling)
        adapter.policy_ref = (
            checkpoint["sampler_path"]
            if checkpoint
            else "tinker-base:" + config.player.base_model
        )
        return adapter, sampling

    async def update_raw(self, datums, *, config, before_submit=None, sequences=None):
        train = config.train
        # Reference shaping retains its historical per-response discount
        # boundaries. Pack only after transferring the shaped sampled credit.
        if sequences is not None:
            from types import SimpleNamespace
            if train.kl_penalty_coef > 0:
                sequences = await self._shape_segment_credit(sequences, train)
            datums = [SimpleNamespace(datum=self._datum(s, include_action_mask=True)) for s in sequences]
        elif train.kl_penalty_coef > 0:
            from tinker_cookbook.rl.metrics import incorporate_kl_penalty

            await incorporate_kl_penalty(
                [item.datum for item in datums],
                self.reference_client,
                train.kl_penalty_coef,
                train.kl_discount_factor,
            )
        adam = self.deps.types.AdamParams(
            learning_rate=train.learning_rate, beta1=config.optimizer.beta1,
            beta2=config.optimizer.beta2, eps=config.optimizer.epsilon
        )
        values = [
            datum_without_mask(deps=self.deps, datum=item.datum) for item in datums
        ]
        if before_submit:
            before_submit()
        forward = await self.training_client.forward_backward_async(
            values,
            loss_fn="cispo",
            loss_fn_config={
                "clip_low_threshold": train.cispo_clip_low,
                "clip_high_threshold": train.cispo_clip_high,
            },
        )
        result = await forward.result_async()
        optimizer = await self.training_client.optim_step_async(adam)
        optim = await optimizer.result_async()
        return {
            "forward_metrics": dict(getattr(result, "metrics", {}) or {}),
            "optim_metrics": dict(getattr(optim, "metrics", {}) or {}),
        }

    async def _shape_segment_credit(self, sequences, train):
        """Keep KL discount local to each sampled response, then restore masks."""
        from dataclasses import replace
        from tinker_cookbook.rl.metrics import incorporate_kl_penalty
        from .backend import TokenSequence

        responses, locations = [], []
        for index, sequence in enumerate(sequences):
            full = sequence.prompt_token_ids + sequence.sampled_tokens
            spans = sequence.source_spans or ((sequence.request_id, len(sequence.prompt_token_ids), len(full)),)
            for name, start, stop in spans:
                offset = start - len(sequence.prompt_token_ids)
                mask = sequence.loss_mask[offset:offset + stop-start]
                if not any(mask):
                    continue
                response = TokenSequence(
                    full[:start], full[start:stop],
                    sequence.behavior_logprobs[offset:offset + stop-start],
                    sequence.advantages[offset:offset + stop-start], mask, name)
                responses.append(self._datum(response, include_action_mask=True))
                locations.append((index, offset, start-1, mask))
        if responses:
            await incorporate_kl_penalty(responses, self.reference_client,
                train.kl_penalty_coef, train.kl_discount_factor)
        advantages = [list(s.advantages) for s in sequences]
        for datum, (index, offset, prefix, mask) in zip(responses, locations):
            values = datum.loss_fn_inputs["advantages"]
            values = values.to_torch().tolist() if hasattr(values, "to_torch") else list(values)
            # Discounted KL can spill into observation rows. Only the intended
            # sampled action positions may contribute direct policy loss.
            advantages[index][offset:offset + len(mask)] = [
                value if enabled else 0.0 for value, enabled in zip(values[prefix:], mask)]
        return [replace(s, advantages=tuple(values)) for s, values in zip(sequences, advantages)]

    async def save_checkpoint(self, *, kind, name, metadata=None):
        future = await (
            self.training_client.save_weights_for_sampler_async(name)
            if kind == "sampler"
            else self.training_client.save_state_async(name)
        )
        result = await future.result_async()
        if not isinstance(getattr(result, "path", None), str) or not result.path:
            raise RuntimeError(f"Tinker {kind} checkpoint {name!r} returned no path.")
        if kind == "sampler":
            self.pending_policy = result.path
        return result.path

    async def publish_policy(self, *, revision):
        sampling = await self.service_client.create_sampling_client_async(
            model_path=self.pending_policy
        )
        self.sampler.set_sampling_client(sampling)
        self.policy_ref = self.pending_policy
        return self.policy_ref

    def _datum(self, sequence, *, include_action_mask=False):
        prefix = len(sequence.prompt_token_ids) - 1
        types, deps = self.deps.types, self.deps
        tensor = lambda values: deps.TensorData.from_torch(deps.torch.tensor(values))
        return types.Datum(
            model_input=types.ModelInput.from_ints(
                list(sequence.prompt_token_ids) + list(sequence.sampled_tokens[:-1])
            ),
            loss_fn_inputs={
                **({"mask": tensor([0.0] * prefix + list(sequence.loss_mask))} if include_action_mask else {}),
                "target_tokens": tensor(
                    list(sequence.prompt_token_ids[1:]) + list(sequence.sampled_tokens)
                ),
                "logprobs": tensor([0.0] * prefix + list(sequence.behavior_logprobs)),
                "advantages": tensor(
                    [0.0] * prefix
                    + [a * m for a, m in zip(sequence.advantages, sequence.loss_mask)]
                ),
            },
        )

    async def forward_backward(self, batch, *, request_id, **options):
        loss = options.get("loss", "cispo")
        if loss != "cispo":
            raise ValueError(
                "This Shellsmith Tinker adapter exposes the existing CISPO path."
            )
        future = await self.training_client.forward_backward_async(
            [self._datum(x) for x in batch.sequences],
            loss_fn="cispo",
            loss_fn_config={
                "clip_low_threshold": options.get("clip_low", 0.0),
                "clip_high_threshold": options.get("clip_high", 4.0),
            },
        )
        result = await future.result_async()
        value = {"forward_metrics": dict(getattr(result, "metrics", {}) or {})}
        self.operations[request_id] = {"status": "succeeded", "result": value}
        return value

    async def optimizer_step(self, *, learning_rate, backward_operation, request_id):
        params = self.deps.types.AdamParams(
            learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
        )
        future = await self.training_client.optim_step_async(params)
        result = await future.result_async()
        value = {"optim_metrics": dict(getattr(result, "metrics", {}) or {})}
        self.operations[request_id] = {"status": "succeeded", "result": value}
        return value

    async def update(self, batch, *, learning_rate, clip_low, clip_high):
        import uuid

        request_id = uuid.uuid4().hex
        forward = await self.forward_backward(
            batch,
            request_id=request_id + ":backward",
            clip_low=clip_low,
            clip_high=clip_high,
        )
        optimizer = await self.optimizer_step(
            learning_rate=learning_rate,
            backward_operation=request_id + ":backward",
            request_id=request_id + ":optimizer",
        )
        return {**forward, **optimizer}

    async def restore_state(self, reference, *, request_id):
        self.training_client = await self.service_client.create_training_client_from_state_with_optimizer_async(
            reference
        )
        result = {"checkpoint": reference}
        self.operations[request_id] = {"status": "succeeded", "result": result}
        return result

    async def sample_tokens(
        self, *, policy, prompt_token_ids, max_tokens, request_id, **options
    ):
        async def setup():
            return await self.service_client.create_sampling_client_async(
                model_path=policy
            )

        client = await setup()

        def draw():
            params = self.deps.types.SamplingParams(
                max_tokens=max_tokens,
                temperature=options.get("temperature", 1.0),
                top_p=options.get("top_p", 1.0),
            )
            result = client.sample(
                prompt=self.deps.types.ModelInput.from_ints(prompt_token_ids),
                num_samples=1,
                sampling_params=params,
            ).result()
            sequence = result.sequences[0]
            return {
                "sampled_tokens": list(sequence.tokens),
                "behavior_logprobs": list(sequence.logprobs),
                "prompt_tokens": len(prompt_token_ids),
                "completion_tokens": len(sequence.tokens),
                "policy": policy,
            }

        result = await asyncio.to_thread(draw)
        self.operations[request_id] = {"status": "succeeded", "result": result}
        return result

    async def token_logprobs(
        self, *, policy, prompt_token_ids, target_token_ids, request_id
    ):
        client = await self.service_client.create_sampling_client_async(
            model_path=policy
        )
        values = await asyncio.to_thread(
            lambda: client.compute_logprobs(
                self.deps.types.ModelInput.from_ints(
                    prompt_token_ids + target_token_ids
                )
            ).result()
        )
        result = {
            "logprobs": list(values)[-len(target_token_ids) :],
            "policy": policy,
            "prompt_tokens": len(prompt_token_ids),
            "completion_tokens": len(target_token_ids),
        }
        self.operations[request_id] = {"status": "succeeded", "result": result}
        return result

    async def operation_status(self, request_id):
        # Hosted durability remains the SDK's responsibility. These are local
        # adapter receipts and deliberately make no Studio retry guarantee.
        return self.operations[request_id]

    def load_adapter(self, path):
        raise ValueError("Tinker policies use hosted checkpoint references.")

    def close(self):
        pass
