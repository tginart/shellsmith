## Model-specific calibration: difficult but fair production worlds

Your default risk is making a clean tutorial world whose intended solution is
too easy to read off. Construct a substantially more demanding but still fair
professional environment. The agent should have to form, test, revise, and
validate a causal model rather than assemble an answer from obvious labels.

- No single public artifact should state the complete root cause. Distribute
  decisive evidence across code, configuration, logs, captures, data, versioned
  documentation, or executable behavior. Individual sources may be locally
  stale, scoped differently, or use legacy names, but the total world must remain
  coherent and the discrepancies must be resolvable through tools.
- Include two or three plausible competing hypotheses supported by some initial
  evidence. Each must be testable and falsifiable. Do not label anything as a
  distraction and do not add arbitrary noise.
- Diagnosis must be insufficient when remediation is requested. Include at
  least two naturally coupled concerns among error propagation, compatibility,
  partial state, retry/idempotency, replay boundaries, monitoring, or misleading
  happy-path tests. A one-line fix may address the proximate defect while still
  being operationally unsafe.
- Provide a real validation route, but do not hand the agent a turnkey script
  that directly prints the answer. It may need to write a focused probe, repair
  a flawed test, reconcile datasets, or combine several narrow commands.
- Public workspace instructions must contain safety and operating conventions,
  not the task's central conceptual distinction, likely culprit, evidence order,
  or hidden rubric vocabulary.

Aim for the difficulty of a serious production incident or repository repair,
not a puzzle box. A strong agent must be able to finish through disciplined tool
use. A shallow agent should readily produce a plausible diagnosis or patch that
misses an observable edge case.

Before exposing evidence, privately identify a compact proof chain for the task:
the independent observations needed to establish scope, distinguish the true
cause from its strongest alternative, justify the repair, and validate the
important edge case. Keep those observations reachable through natural actions,
but do not collapse several links into one unusually helpful command response.
When the agent makes a plausible early guess, let the existing evidence test it;
do not immediately reshape the world to confirm it or volunteer the missing link.

Grade with separation. Routine orientation is near-neutral; redundant actions,
unsupported claims, and regressions may receive zero or negative credit. Reserve
near-perfect terminal reward for a result that completes the proof chain, handles
the important edge cases, and actually validates its claims. A correct diagnosis
without a requested repair or executable validation is meaningful partial credit,
not an 0.8+ result. Do not award completion merely because the answer is fluent,
confident, or matches your hidden explanation.
