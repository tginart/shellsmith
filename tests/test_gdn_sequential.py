"""
Purpose: Check BF16 GDN against independent sequential values and derivatives.
Input: Short tails, nonzero incoming states, and all incoming cotangents.
Output: Forward, final-state, and six derivative error bounds.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
np = pytest.importorskip("numpy")
from studio_compute.numerics.ops import gated_delta_chunked

# B: batch, L: tokens, H: value heads, J: key heads, K: key width, V: value width.


def sequential(inputs: list, cotangents: list) -> tuple:
    """
    Purpose: Apply the token recurrence and reverse its elementary equations.
    Input: FP64 NumPy tensors and output cotangents. Output: Values and manual gradients.
    Note: This reference uses no chunk solve, inverse, or Metal kernel.
    """
    q, k, v, g, beta, initial = inputs
    groups = v.shape[2] // q.shape[2]
    q = np.repeat(q, groups, axis=2)
    k = np.repeat(k, groups, axis=2)
    state = initial.copy()
    output, tape = [], []
    for t in range(v.shape[1]):
        before = state
        decayed = g[:, t, :, None, None] * before
        difference = v[:, t] - np.einsum("bhvk,bhk->bhv", decayed, k[:, t])
        delta = beta[:, t, :, None] * difference
        state = decayed + delta[..., None] * k[:, t, :, None, :]
        output.append(np.einsum("bhvk,bhk->bhv", state, q[:, t]))
        tape.append((before, decayed, difference, delta, state))
    dy, dstate = cotangents
    dq, dk, dv, dg, dbeta = [np.zeros_like(x) for x in (q,k,v,g,beta)]
    for t in reversed(range(v.shape[1])):
        before, decayed, difference, delta, current = tape[t]
        dq[:, t] = np.einsum("bhvk,bhv->bhk", current, dy[:, t])
        dstate = dstate + dy[:, t, :, :, None] * q[:, t, :, None, :]
        ddelta = np.einsum("bhvk,bhk->bhv", dstate, k[:, t])
        dk[:, t] = np.einsum("bhvk,bhv->bhk", dstate, delta)
        dbeta[:, t] = np.sum(ddelta * difference, axis=-1)
        dv[:, t] = beta[:, t, :, None] * ddelta
        dprediction = -dv[:, t]
        dk[:, t] += np.einsum("bhvk,bhv->bhk", decayed, dprediction)
        ddecayed = dstate + dprediction[..., None] * k[:, t, :, None, :]
        dg[:, t] = np.sum(ddecayed * before, axis=(-1,-2))
        dstate = g[:, t, :, None, None] * ddecayed
    shape = (*inputs[0].shape[:3], groups, inputs[0].shape[-1])
    dq = dq.reshape(shape).sum(axis=3)
    dk = dk.reshape(shape).sum(axis=3)
    return [np.stack(output, axis=1), state], [dq,dk,dv,dg,dbeta,dstate]


@pytest.mark.parametrize("length", [17,70,193])
@pytest.mark.parametrize("unit_gates", [False, True])
def test_chunk_pipeline_matches_sequential_values_and_derivatives(length: int, unit_gates: bool) -> None:
    mx.random.seed(7)
    inputs = [
        (mx.random.normal((1,length,1,64))*.02).astype(mx.bfloat16),
        (mx.random.normal((1,length,1,64))*.08).astype(mx.bfloat16),
        (mx.random.normal((1,length,2,16))*.1).astype(mx.bfloat16),
        mx.random.uniform(.96,.995,(1,length,2)).astype(mx.bfloat16),
        mx.random.uniform(.1,.6,(1,length,2)).astype(mx.bfloat16),
        (mx.random.normal((1,2,16,64))*.02).astype(mx.bfloat16),
    ]
    if unit_gates:
        inputs[3] = mx.ones_like(inputs[3])
    cotangents = [mx.random.normal(inputs[2].shape).astype(mx.bfloat16),
                  mx.random.normal(inputs[-1].shape).astype(mx.bfloat16)]
    expected = sequential(
        [np.asarray(x.astype(mx.float32),dtype=np.float64) for x in inputs],
        [np.asarray(x.astype(mx.float32),dtype=np.float64) for x in cotangents],
    )
    actual = mx.vjp(lambda *values: gated_delta_chunked(
        *values,compute_dtype=mx.bfloat16,backend="metal"
    ),inputs,cotangents)
    for group, references in zip(actual,expected,strict=True):
        for value,reference in zip(group,references,strict=True):
            plain = np.asarray(value.astype(mx.float32),dtype=np.float64)
            error = np.linalg.norm(plain-reference)/max(np.linalg.norm(reference),1e-9)
            assert error<.08,error
