"""BF16 fixtures for extracted grouped replay and cooperative execution."""

import os
from types import SimpleNamespace
import pytest

os.environ.setdefault("MLX_ENABLE_TF32", "0")
mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_vlm")
from mlx_vlm.models.qwen3_5.config import TextConfig
from mlx_vlm.models.qwen3_5.language import LanguageModel
from mlx.utils import tree_flatten
from studio_compute.numerics.adapter import Adapter
from studio_compute.numerics.cispo import cispo_loss, reference_loss_and_gradient
from studio_compute.numerics.optimizer import MasterAdam
from studio_compute.numerics.replay import Replay
from studio_compute.cache import PrefixCache
from studio_compute.engine import ResidentEngine
from studio_compute.runtime import RuntimeEngine
from studio_compute.model import Qwen27Model
from studio_compute.service import Service


def tiny():
    mx.random.seed(17)
    config = TextConfig(
        model_type="qwen3_5_text",
        hidden_size=64,
        intermediate_size=128,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_hidden_layers=4,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=96,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        head_dim=16,
        rope_parameters={
            "type": "default",
            "partial_rotary_factor": 0.5,
            "rope_theta": 10000,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
    )
    lm = LanguageModel(config)
    lm.set_dtype(mx.bfloat16)
    nn.quantize(lm, group_size=64, bits=8)
    adapter = Adapter(lm, ["linear_attention"] * 3 + ["full_attention"])
    adapter.load(adapter.initial(23))
    adapter.load(
        {
            name: value + 0.001 if name.endswith(".b") else value
            for name, value in adapter.state().items()
        }
    )
    lm.train()
    mx.eval(lm.parameters())
    return lm, adapter


def branch(lm, prompt, targets, weights=None):
    ids = mx.array([tuple(prompt) + tuple(targets[:-1])], mx.int32)
    return {
        "embeddings": lm.model.embed_tokens(ids).astype(mx.bfloat16),
        "positions": mx.broadcast_to(
            mx.arange(ids.shape[1], dtype=mx.int32)[None, None], (3, 1, ids.shape[1])
        ),
        "prefix_length": len(prompt),
        "tokens": targets,
        "objective": lambda current: cispo_loss(
            current, [-5.0] * len(targets), weights or [0.1] * len(targets)
        ),
    }


def drain(generator, select=lambda: None):
    while True:
        select()
        try:
            next(generator)
        except StopIteration as result:
            return result.value


def flat(values):
    return mx.concatenate(
        [values[name].astype(mx.float32).reshape(-1) for name in sorted(values)]
    )


def test_cispo_sum_mask_and_normalization_once():
    current, old, advantages, mask = (
        [-2.0, -1.0, -4.0, -2.0],
        [-3.0, -4.0, -1.0, -2.0],
        [0.1, -0.25, 0.5, 0.0],
        [1, 1, 1, 0],
    )
    loss, gradient = mx.value_and_grad(lambda x: cispo_loss(x, old, advantages, mask))(
        mx.array(current)
    )
    expected, expected_gradient = reference_loss_and_gradient(
        current, old, advantages, mask
    )
    assert loss.item() == pytest.approx(expected, abs=1e-6)
    assert gradient.tolist() == pytest.approx(expected_gradient, abs=1e-6)


@pytest.mark.parametrize("head_rows", [128, 2])
def test_full_prompt_grouped_gradient_and_native_bf16_likelihood(head_rows):
    lm, adapter = tiny()
    prompt, targets = tuple(range(1, 22)), (17, 23, 11, 8, 29)
    branches = [branch(lm, prompt, targets), branch(lm, prompt, (21, 15, 9))]
    currents, loss, gradients, stats = drain(
        Replay(
            SimpleNamespace(language_model=lm), segment_tokens=7, checkpoint_layers=2, head_rows=head_rows
        ).run_group(branches, capture_prefix_gradient=True)
    )
    assert stats["shared_prefix_tokens"] == 14
    assert float(mx.sum(stats["prefix_gradient"].astype(mx.float32) ** 2).item()) > 0
    individual = [
        drain(
            Replay(
                SimpleNamespace(language_model=lm),
                segment_tokens=7,
                checkpoint_layers=2,
                head_rows=head_rows,
            ).run_group([x])
        )
        for x in branches
    ]
    combined = {
        name: sum(result[2][name] for result in individual) for name in gradients
    }
    difference = float(mx.linalg.norm(flat(gradients) - flat(combined)).item())
    assert difference / float(mx.linalg.norm(flat(combined)).item()) < 0.08
    assert loss.item() == pytest.approx(sum(x[1].item() for x in individual), rel=0.025)
    ids = mx.array([prompt + targets[:-1]], mx.int32)
    native = (
        lm(ids, cache=lm.make_cache(), position_ids=branches[0]["positions"])
        .logits[0, len(prompt) - 1 :]
        .astype(mx.float32)
    )
    expected = mx.take_along_axis(native, mx.array(targets)[:, None], -1)[
        :, 0
    ] - mx.logsumexp(native, -1)
    assert (
        max(abs(x - y) for x, y in zip(currents[0].tolist(), expected.tolist())) < 0.15
    )


def test_recomputation_carry_adjoints_and_interleaving_are_exact():
    lm, adapter = tiny()
    first = adapter.state()
    second = adapter.initial(37)
    values = branch(lm, tuple(range(1, 22)), (17, 23, 11, 8, 29))
    holder = SimpleNamespace(language_model=lm)
    baseline = drain(
        Replay(holder, segment_tokens=7, checkpoint_layers=1).run_group(
            [values], capture_prefix_gradient=True
        )
    )
    replay = Replay(holder, segment_tokens=7, checkpoint_layers=2).run_group(
        [values], capture_prefix_gradient=True
    )
    units = 0
    while True:
        adapter.load(first)
        lm.train()
        try:
            next(replay)
        except StopIteration as result:
            interleaved = result.value
            break
        adapter.load(second)
        lm.eval()
        output = lm(
            mx.array([[1, 2, 3]], mx.int32),
            cache=lm.make_cache(),
            position_ids=mx.broadcast_to(
                mx.arange(3, dtype=mx.int32)[None, None], (3, 1, 3)
            ),
        ).logits
        mx.eval(output)
        units += 1
    assert units > 3
    assert all(
        bool(mx.array_equal(interleaved[2][name], value).item())
        for name, value in baseline[2].items()
    )
    assert bool(
        mx.array_equal(
            interleaved[3]["prefix_gradient"], baseline[3]["prefix_gradient"]
        ).item()
    )


@pytest.mark.parametrize("options", [{}, {"beta1": 0.8, "beta2": 0.98, "epsilon": 1e-7}])
def test_frozen_base_and_exact_adam_restore(options):
    lm, adapter = tiny()
    frozen = {
        name: mx.array(value)
        for name, value in tree_flatten(lm.parameters())
        if not name.endswith((".a", ".b"))
    }
    values = branch(lm, tuple(range(1, 14)), (17, 23, 11, 8, 29))
    _, _, gradients, _ = drain(
        Replay(
            SimpleNamespace(language_model=lm), segment_tokens=7, checkpoint_layers=2
        ).run_group([values])
    )
    optimizer = MasterAdam(adapter, 5e-5, **options)
    optimizer.accumulate(gradients)
    assert optimizer.step() > 0
    tensors, metadata = optimizer.state()
    assert (metadata["beta1"], metadata["beta2"], metadata["eps"]) == (
        options.get("beta1", 0.9), options.get("beta2", 0.95), options.get("epsilon", 1e-8))
    if options:
        with pytest.raises(ValueError, match="semantics differ"):
            MasterAdam(adapter, 5e-5).load(tensors, metadata)
    restored = MasterAdam(adapter, 5e-5, **options)
    restored.load(tensors, metadata)
    for instance in (optimizer, restored):
        instance.accumulate(gradients)
        instance.step()
    assert all(
        bool(mx.array_equal(value, restored.master[name]).item())
        for name, value in optimizer.master.items()
    )
    current = dict(tree_flatten(lm.parameters()))
    assert all(
        bool(mx.array_equal(value, current[name]).item())
        for name, value in frozen.items()
    )


def test_upstream_cache_snapshot_is_immutable_and_bounded():
    from mlx_vlm.models.cache import KVCache

    cache = KVCache()
    cache.update_and_fetch(
        mx.ones((1, 2, 4, 16), mx.bfloat16), mx.ones((1, 2, 4, 16), mx.bfloat16)
    )
    store = PrefixCache(mx, 32768)
    store.put("m", "p0", "bf16", [1, 2, 3, 4], [cache], mx.zeros((1, 96)))
    cloned, _, count = store.get("m", "p0", "bf16", [1, 2, 3, 4, 5])
    cloned[0].update_and_fetch(
        mx.zeros((1, 2, 1, 16), mx.bfloat16), mx.zeros((1, 2, 1, 16), mx.bfloat16)
    )
    untouched, _, count = store.get("m", "p0", "bf16", [1, 2, 3, 4])
    assert count == 4 and untouched[0].offset == 4
    assert float(untouched[0].state[0][..., :4, :].sum().item()) == 128
    assert store.get("m", "p1", "bf16", [1, 2, 3, 4]) is None
    assert store.get("m", "p0", "8bit", [1, 2, 3, 4]) is None
    assert store.bytes <= store.limit


class TinyOwner(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.language_model = lm


class TinyResident(Qwen27Model):
    def __init__(self, *args, **kwargs):
        lm, adapter = tiny()
        self.mx, self.lm, self.adapter = mx, lm, adapter
        self.model = TinyOwner(lm)
        self.budget = 240 * 1024**3
        self.selected, self.training = None, None
        self.identity = {"model": "tiny-bf16", "numerics": "resident-engine-fixture"}
        self.capabilities = {
            "model": "tiny",
            "context_window": 4096,
            "vocabulary_size": 96,
            "cache_precisions": ["bf16"],
            "precision": {"activations": "bf16", "lora": "bf16", "optimizer": "fp32"},
        }
        self.bundle = {"hash": "test", "files": {}}
        self.frozen = [
            (name, value)
            for name, value in tree_flatten(self.model.parameters())
            if not name.endswith((".a", ".b"))
        ]
        self.base_allocation = tuple(
            (name, id(value), value.nbytes) for name, value in self.frozen
        )


@pytest.mark.parametrize("beta2", [0.95, 0.98])
def test_resident_clients_rng_and_optimizer_isolation(tmp_path, beta2):
    service = Service(
        tmp_path,
        lambda: RuntimeEngine("tiny", model_factory=TinyResident, execution_profile="legacy", checkpoint_layers=2),
    )

    def call(client, kind, body, id):
        service.submit(id, client, kind, body)
        op = service.wait(id)
        assert op["status"] == "succeeded", op
        return op["result"]

    try:
        call("a", "create", {"seed": 23, "optimizer": {"beta2": beta2}}, "create-a")
        call("b", "create", {"seed": 37}, "create-b")
        first_a = call(
            "a", "inspect", {"hash_state": True, "hash_base": True}, "inspect-a"
        )
        first_b = call("b", "inspect", {"hash_state": True}, "inspect-b")
        assert first_a["base_allocation"] == first_b["base_allocation"]
        assert first_a["state_hashes"]["adapter"] != first_b["state_hashes"]["adapter"]
        p = call("a", "publish", {"revision": 0}, "pub")["policy"]
        call(
            "a",
            "sample",
            {"policy": p, "prompt_token_ids": [1, 2, 3], "max_tokens": 3},
            "sample-a",
        )
        sequence = {
            "prompt_token_ids": [1, 2, 3],
            "sampled_tokens": [4, 5, 6],
            "weights": [0.2, -0.1, 0.3],
            "loss_mask": [1.0, 0.0, 1.0],
        }
        result = call(
            "a",
            "forward_backward",
            {"loss": "weighted_cross_entropy", "sequences": [sequence]},
            "back-a",
        )
        expected = -0.2 * result["logprobs"][0][0] - 0.3 * result["logprobs"][0][2]
        assert result["forward_metrics"]["loss"] == pytest.approx(expected, abs=1e-5)
        assert result["forward_metrics"]["loss_kind"] == "weighted_cross_entropy"
        assert result["forward_metrics"]["weighted_cross_entropy_loss"] == pytest.approx(expected, abs=1e-5)
        assert "cispo_loss" not in result["forward_metrics"]
        call(
            "a",
            "optim_step",
            {"learning_rate": 5e-5, "backward_operation": "back-a"},
            "step-a",
        )
        after_a = call(
            "a", "inspect", {"hash_state": True, "hash_base": True}, "after-a"
        )
        after_b = call("b", "inspect", {"hash_state": True}, "after-b")
        assert (
            first_b["state_hashes"] == after_b["state_hashes"]
            and first_b["rng"] == after_b["rng"]
        )
        assert first_a["frozen_base_hash"] == after_a["frozen_base_hash"]
        assert first_a["rng"] != after_a["rng"]
        assert after_a["optimizer_steps"] == 1 and after_b["optimizer_steps"] == 0
    finally:
        service.close()
    service = Service(tmp_path, lambda: RuntimeEngine("tiny", model_factory=TinyResident, execution_profile="legacy", checkpoint_layers=2))
    try:
        recovered = call("a", "inspect", {"hash_state": True}, "recovered-a")
        assert recovered["state_hashes"] == after_a["state_hashes"]
        assert recovered["rng"] == after_a["rng"]
        assert service.engine.clients["a"]["optimizer"].beta2 == beta2
    finally:
        service.close()


def test_real_bf16_cache_snapshots_yield_memory_to_active_work():
    engine = RuntimeEngine("tiny", model_factory=TinyResident, execution_profile="legacy", checkpoint_layers=2, segment_tokens=2)
    try:
        engine.create("a", {"seed":23})
        engine.policies["p"] = {"factors":engine.clients["a"]["adapter"].state(),"revision":0}
        for token in (1,2,3,4):
            drain(engine._sample({"policy":"p","revision":0,"prompt_token_ids":[token]*32,
                "max_tokens":1,"seed":17}, __import__("time").monotonic()))
        retained = engine.prefixes.bytes
        assert retained > 0 and len(engine.prefixes.entries) == 4
        engine.reserved["live-replay"] = retained // 4
        engine.resident.budget = mx.get_active_memory() + retained // 5
        engine.admit(retained * 3 // 5)
        assert engine.prefixes.metrics()["evictions"] > 0
        assert engine.prefixes.bytes < retained
        assert engine.reserved == {"live-replay":retained // 4}
        assert mx.get_active_memory() + retained * 3 // 5 <= engine.resident.budget
    finally:
        engine.close()


def test_native_decode_progress_accounts_for_eos_and_batch_members():
    engine = RuntimeEngine("tiny", model_factory=TinyResident, execution_profile="legacy", checkpoint_layers=2)
    try:
        engine.create("a", {"seed": 23})
        engine.policies["p"] = {"factors": engine.clients["a"]["adapter"].state(), "revision": 0}
        observed = []
        engine.set_inference_observer(lambda request, count, finished: observed.append((request, count, finished)))
        body = {"policy": "p", "revision": 0, "prompt_token_ids": [1, 2], "max_tokens": 3, "seed": 17,
                "operation_id": "one"}
        first_stops = {**body, "stop_token_ids": list(range(96))}
        progress = []
        work = engine.sample_batch([first_stops, {**body, "seed": 18, "operation_id": "two"}])
        while True:
            try:
                progress.append(next(work))
            except StopIteration as done:
                results = done.value
                break
        decode = [row for row in progress if row["phase"] == "batched_decode"]
        assert [row["generated_tokens"] for row in decode] == [[1, 1], [1, 2], [1, 3]]
        assert decode[0]["finished_requests"] == [True, False]
        assert decode[-1]["finished_requests"] == [True, True]
        assert [len(row["sampled_tokens"]) for row in results] == [1, 3]
        assert observed == [("one", 1, True), ("two", 1, False), ("two", 2, False), ("two", 3, True)]
        single = engine._sample(first_stops, __import__("time").monotonic())
        rows = []
        while True:
            try:
                rows.append(next(single))
            except StopIteration as done:
                assert len(done.value["sampled_tokens"]) == 1
                break
        assert rows[-1]["generated_tokens"] == [1] and rows[-1]["finished_requests"] == [True]
    finally:
        engine.close()


def test_grouped_requests_preserve_input_alignment_and_peak_admission():
    engine = ResidentEngine("tiny", model_factory=TinyResident, execution_profile="legacy", checkpoint_layers=2)
    try:
        engine.create("a", {"seed": 23})
        sequences = [
            {"prompt_token_ids": [1, 2, 3], "sampled_tokens": [4, 5]},
            {"prompt_token_ids": [8, 9], "sampled_tokens": [10, 11, 12]},
            {"prompt_token_ids": [1, 2, 3], "sampled_tokens": [13]},
        ]
        for sequence in sequences:
            sequence.update(weights=[0.1] * len(sequence["sampled_tokens"]))
        estimates = []
        original = engine.resident.estimate_replay

        def estimate(group, checkpoint_layers, **options):
            estimates.append(len(group))
            return original(group, checkpoint_layers, **options)

        engine.resident.estimate_replay = estimate
        result = drain(
            engine.work(
                "a",
                "forward_backward",
                {
                    "operation_id": "grouped",
                    "loss": "weighted_cross_entropy",
                    "sequences": sequences,
                },
            )
        )
        assert sorted(estimates) == [1, 2]
        assert engine.reserved == {("optimizer", "a"): engine.optimizer_step_bytes("a")}
        engine.abort("a")
        for index, sequence in enumerate(sequences):
            individual = drain(
                engine.work(
                    "a",
                    "forward_backward",
                    {
                        "operation_id": f"individual-{index}",
                        "loss": "weighted_cross_entropy",
                        "sequences": [sequence],
                    },
                )
            )
            assert result["logprobs"][index] == individual["logprobs"][0]
            engine.abort("a")
        assert not engine.reserved
    finally:
        engine.close()


@pytest.mark.parametrize("checkpoint_layers", [1, 2, 4])
def test_packed_rollout_matches_independent_response_gradients(checkpoint_layers):
    from shellsmith.train.backend import TokenSequence
    from shellsmith.train.segments import pack_rollout_segments
    lm, adapter = tiny()
    frozen_before = [(k, v.tolist()) for k,v in tree_flatten(lm.parameters()) if "lora" not in k]
    model = SimpleNamespace(language_model=lm)
    p = [1,2,3,4,5,6,7,8,9]
    targets = ([10,11,12], [15,16,17,18], [21,22])
    responses=[]
    for i,tokens in enumerate(targets):
        mask = [1.] * len(tokens)
        if i==0: mask[1]=0. # deterministic close
        advantages = [(i+1)*(.07 if j%2 else -.11)*m for j,m in enumerate(mask)]
        responses.append(TokenSequence(tuple(p),tuple(tokens),tuple([-4.7]*len(tokens)),tuple(advantages),tuple(mask),str(i)))
        p = p + tokens + [13+i,14+i]  # intervening tool output
    packed,=pack_rollout_segments([(0,r) for r in responses]).sequences
    def run(s,sparse=False):
        selected=[i for i,m in enumerate(s.loss_mask) if m]
        b=branch(lm,s.prompt_token_ids,list(s.sampled_tokens))
        b["objective"]=lambda current: cispo_loss(current,
            [s.behavior_logprobs[i] for i in selected] if sparse else s.behavior_logprobs,
            [s.advantages[i] for i in selected] if sparse else s.advantages,
            None if sparse else s.loss_mask)
        if sparse:b["target_indices"]=selected
        return drain(Replay(model,segment_tokens=4,checkpoint_layers=checkpoint_layers,head_rows=2).run_group([b],capture_prefix_gradient=True))
    refs=[run(r) for r in responses]
    currents,loss,grads,stats=run(packed,True)
    expected=sum(float(r[1].item()) for r in refs)
    expected_grad={k:sum(r[2][k] for r in refs) for k in grads}
    g,ref=flat(grads),flat(expected_grad)
    assert abs(float(loss.item())-expected) < .02
    assert float(mx.linalg.norm(g-ref)/mx.maximum(mx.linalg.norm(ref),1e-8)) < .04
    assert float(mx.sum(g*ref)/(mx.linalg.norm(g)*mx.linalg.norm(ref))) > .999
    assert float(mx.linalg.norm(stats["prefix_gradient"])) > 0
    prefix_reference=sum(r[3]["prefix_gradient"][:, :9].astype(mx.float32) for r in refs)
    prefix_packed=stats["prefix_gradient"].astype(mx.float32)
    assert float(mx.linalg.norm(prefix_packed-prefix_reference)/mx.linalg.norm(prefix_reference)) < .05
    # Packed path has no losses on tools, injected close, or prompt rows.
    assert currents[0].shape[0]==sum(packed.loss_mask)
    assert [(k,v.tolist()) for k,v in tree_flatten(lm.parameters()) if "lora" not in k]==frozen_before


def test_training_handoff_releases_snapshots_and_preserves_active_working_cache():
    from dataclasses import asdict
    from shellsmith.train.backend import TokenSequence
    engine = RuntimeEngine("tiny", model_factory=TinyResident, execution_profile="legacy", checkpoint_layers=2, segment_tokens=2)
    try:
        engine.create("a", {"seed":23})
        engine.policies["p"]={"factors":engine.clients["a"]["adapter"].state(),"revision":0}
        body={"operation_id":"inference", "policy":"p", "revision":0,
              "prompt_token_ids":[1,2,3],"target_token_ids":[4,5,6],"seed":17}
        expected=drain(engine._sample(body,0))["logprobs"]
        assert engine.prefixes.bytes>0
        working=engine._sample({**body,"operation_id":"active"},0)
        while next(working)["phase"]!="decode":pass
        sequence=TokenSequence((1,2,3),(4,5,6,7,8),(-4.,)*5,(.1,0.,.2,0.,0.),
                               (1.,0.,1.,0.,0.),"r",source_spans=(("r",3,8),))
        work=engine.work("a","forward_backward",{"operation_id":"backward","sequences":[asdict(sequence)]})
        progress=[]
        while True:
            try:progress.append(next(work))
            except StopIteration as end:result=end.value;break
        assert engine.prefixes.bytes==0
        assert result["forward_metrics"]["released_prefix_cache_bytes"]>0
        final=progress[-1]
        assert final["completed_segments"]==final["total_segments"]==1
        assert final["completed_tokens"]==final["total_tokens"]==7
        assert final["completed_supervised_tokens"]==2 and not final["optimizer_committed"]
        assert result["logprob_target_indices"]==[[0,2]]
        assert sum(p.get("rows",0) for p in progress if p["phase"]=="head")==2
        # A live inference cache was cloned before snapshots were cleared.
        actual=drain(working)["logprobs"]
        assert actual==pytest.approx(expected,abs=1e-6)
        step=drain(engine.work("a","optim_step",{"backward_operation":"backward","learning_rate":5e-5}))
        assert step["optim_metrics"]["optimizer_steps"]==1 and step["optim_metrics"]["optimizer_committed"]
    finally:engine.close()


def test_context_only_segment_still_allows_one_zero_gradient_adam_update():
    from dataclasses import asdict
    from shellsmith.train.backend import TokenSequence
    engine=RuntimeEngine("tiny",model_factory=TinyResident, execution_profile="legacy",checkpoint_layers=2)
    try:
        engine.create("a",{"seed":23})
        optimizer = engine.clients["a"]["optimizer"]
        future_gradients = sum(value.nbytes for value in optimizer.master.values())
        engine.resident.budget = (
            max(engine.persistent_bytes(), mx.get_active_memory())
            + engine.optimizer_step_bytes("a") + future_gradients + 1024**2
        )
        s=TokenSequence((1,2),(3,4),(-4.,-4.),(0.,0.),(0.,0.),"context",source_spans=(("context",2,4),))
        result=drain(engine.work("a","forward_backward",{"operation_id":"zero","sequences":[asdict(s)]}))
        assert result["forward_metrics"]["cispo_loss"]==0.
        assert result["logprobs"] == [[]]
        assert result["logprob_target_indices"] == [[]]
        assert result["forward_metrics"]["completed_segments"] == 1
        assert result["forward_metrics"]["completed_tokens"] == 3
        assert result["forward_metrics"]["supervised_tokens"] == 0
        assert engine.reserved == {("optimizer", "a"): engine.optimizer_step_bytes("a")}
        step=drain(engine.work("a","optim_step",{"backward_operation":"zero","learning_rate":5e-5}))
        assert step["optim_metrics"]["optimizer_steps"]==1
    finally:engine.close()


class OrdinaryReplay(Replay):
    """Independent unfused reference for the same explicit replay algorithm."""

    def _can_compile_gdn(self, layer, hidden_BLD):
        return False


def test_compiled_gdn_reuse_preserves_current_policy_and_full_gradients():
    """
    Purpose: Verify current factors and state after multiple cache uses.
    Input: Two nonzero policies and the same causal inputs.
    Output: Exact likelihoods, all factor gradients, and prefix gradients.
    """
    if mx.default_device() != mx.gpu:
        pytest.skip("The compiled path requires the Metal GPU")
    lm, adapter = tiny()
    first = adapter.state()
    second = {
        name: value + 0.002 if name.endswith(".b") else value
        for name, value in adapter.initial(37).items()
    }
    holder = SimpleNamespace(language_model=lm)
    compiled = Replay(holder, segment_tokens=7, checkpoint_layers=1)
    cached = None
    for factors in (first, second, first):
        adapter.load(factors)
        lm.train()
        values = branch(lm, tuple(range(1, 22)), (17, 23, 11, 8, 29))
        actual = drain(compiled.run_group([values], capture_prefix_gradient=True))
        expected = drain(
            OrdinaryReplay(holder, segment_tokens=7, checkpoint_layers=1).run_group(
                [values], capture_prefix_gradient=True
            )
        )
        assert actual[1].item() == expected[1].item()
        assert mx.array_equal(actual[0][0], expected[0][0]).item()
        assert actual[2].keys() == expected[2].keys()
        for name, value in actual[2].items():
            assert value.dtype == mx.float32
            assert mx.array_equal(value.view(mx.uint32), expected[2][name].view(mx.uint32)).item()
        assert mx.array_equal(actual[3]["prefix_gradient"], expected[3]["prefix_gradient"]).item()
        identities = {key: id(unit) for key, unit in compiled._compiled_gdn_units.items()}
        assert len(identities) == 3
        if cached is not None:
            assert identities == cached
        cached = identities
        assert any(bool(mx.any(value != 0).item()) for name, value in actual[2].items()
                   if name.endswith(".a"))


def test_training_graph_cache_does_not_replace_native_inference():
    """
    Purpose: Keep native inference after a compiled graph exists for the layer.
    Input: A mode change on the same model. Output: Exact native inference results.
    """
    from studio_compute.numerics.ops import initial_states, segment_layer

    lm, _ = tiny()
    replay = Replay(SimpleNamespace(language_model=lm))
    layer = lm.layers[0]
    hidden_BLD = lm.model.embed_tokens(mx.array([[1, 2, 3, 4, 5]], mx.int32)).astype(mx.bfloat16)
    states = [value for _, value in initial_states([layer], mx.bfloat16)[0]]
    positions = mx.broadcast_to(mx.arange(5, dtype=mx.int32)[None, None], (3, 1, 5))
    lm.train()
    trained = replay._layer(layer, hidden_BLD, states, positions)
    mx.eval(trained)
    if mx.default_device() == mx.gpu:
        assert len(replay._compiled_gdn_units) == 1
    lm.eval()
    actual = replay._layer(layer, hidden_BLD, states, positions)
    expected = segment_layer(layer, hidden_BLD, states, positions)
    mx.eval(actual, expected)
    for a, b in zip(actual, expected):
        assert mx.array_equal(a, b).item()

def test_forward_then_zero_rate_update_and_restore():
    """A forward evaluation must leave the client ready for another backward."""
    engine = RuntimeEngine("tiny", model_factory=TinyResident, execution_profile="legacy", checkpoint_layers=1)
    engine.create("a", {"seed": 23})
    sequence = {
        "prompt_token_ids": [1, 2, 3],
        "sampled_tokens": [4, 5],
        "weights": [0.5, 0.5],
        "loss_mask": [1.0, 1.0],
    }
    drain(engine.work("a", "forward", {
        "sequences": [sequence], "operation_id": "evaluate",
        "loss": "weighted_cross_entropy",
    }))
    assert engine.clients["a"]["backward"] is None
    before = engine.clients["a"]["adapter"].state()
    drain(engine.work("a", "forward_backward", {
        "sequences": [sequence], "operation_id": "back",
        "loss": "weighted_cross_entropy",
    }))
    drain(engine.work("a", "optim_step", {
        "learning_rate": 0.0, "backward_operation": "back",
    }))
    after = engine.clients["a"]["adapter"].state()
    assert all(bool(mx.array_equal(value, after[name]).item())
               for name, value in before.items())
    optimizer = engine.clients["a"]["optimizer"]
    tensors, metadata = optimizer.state()
    optimizer.validated_state(tensors, metadata)
    assert optimizer.steps == 1
    engine.close()
