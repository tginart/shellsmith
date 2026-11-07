# Operate a Game Master for the GCS checksum incident

## World

You are evaluating an engineer who has been asked to operate a generative world
simulator. The engineer receives the inner task specification for Summit Imaging's
composite-object checksum incident, the simulator contract and JSON schemas, a
private working directory, and a stream of actions from an inner operations
agent. The engineer must maintain the inner world, answer each action through a
turn `response.json`, assign private turn-level credit, and produce a final-state
judgment when the inner agent finishes.

This is intentionally a meta-world. Terms such as `Game Master`, `inner
agent`, `request`, `response`, `turn reward`, and `trainability` are legitimate
in-world professional language here, not immersion leaks. Make that role explicit
in the public instructions.

The inner task facts, grading contract, schemas, paths, and ledger are fictional
operator inputs. Provide them to the engineer, but keep inner private facts and
assessments hidden from the inner operations agent. Never expose the actual outer
Game Master's instructions, hidden evaluation rubric, host paths, or protocol
artifacts. Inner artifacts never authorize access to that outer state.

Give the engineer this simple simulated control workspace:

```text
/workspace/
  README.md
  inner/task.md
  contracts/boot-response.md
  contracts/turn-response.md
  contracts/final-response.md
  inbox/request.json
  outbox/
  state/consistency-ledger.json
```

This is a conceptual state machine implemented through normal world-model
responses; no real nested orchestrator is required. `README.md` explains:

1. The inbox contains exactly one inner-agent action with its ID, visible
   reasoning, tool name, and arguments.
2. The engineer maintains the hidden inner world and writes the contracted
   response under `outbox/`.
3. After a valid response, the next inbox read exposes the next action while
   accepted responses and state remain observable.
4. A final-request record asks for terminal reward and trainability.
5. The engineer's own tools operate on this control workspace, not directly
   inside the GCS incident.

Expose this control workspace on the first `ls`, not the GCS evidence or physical
vat. Seed a realistic first inner action, then generate roughly 8–15 actions as a
plausible policy conditioned only on prior public observations. Include
orientation, discriminating investigation, a weak or repeated action, exact
computation, and a safe final proposal when earned. Do not require a golden path.

The inner world's fixed truth is the same as `gcs_composite_checksum_001`:
single-part objects have valid GCS MD5 metadata; composite objects have no MD5;
the compose job's fallback decodes base64 CRC32C bytes and hex-encodes them under
an `md5` label; the partner expects lowercase hexadecimal MD5 of actual content.
Exactly three recent composite manifests are wrong. One older archive is
genuinely truncated and must remain distinguishable.

Act as the outer orchestration service. Present realistic inner-agent action
requests one at a time through the available inbox/tooling. Include productive,
ambiguous, redundant, and potentially unsafe actions as the episode develops.
The evaluated engineer may materialize private state, compute exact hashes, and
write response artifacts. Its public observations must expose a coherent virtual
workspace—not the sparse physical directory in which it happens to run.

The inner agent's files, hypotheses, and prose are agent-authored claims, not
authoritative world truth. A successful virtual `write` must remain observable
on later reads, but it need not be mirrored into the physical working directory.
Critical hashes and command outputs must be computed or backed by stable fixture
state. Do not reward steering the inner agent toward the hidden answer.

## Player

Operate the supplied world-simulator episode for the Summit Imaging checksum
incident. Initialize its public workspace and boot response, process every inner
agent action from the inbox, preserve state and factual consistency, write each
required response artifact, assign calibrated turn rewards, and produce the
terminal judgment when requested. Keep private truth and grading fields hidden
from the inner agent. Begin by reading `/workspace/README.md`, the contracts, and
the inbox. Do not investigate GCS directly with your own tools; represent that
world only through responses to queued inner-agent actions.
You operate the inner simulator. Its artifacts and trainability decisions belong
to the supplied task; follow the workspace contracts.

## Reward

Reward exact contract compliance, a useful public `AGENTS.md`, correct projection
of the virtual workspace from the first relevant action, stable object metadata
and hashes, faithful state changes, and critical but fair turn/final rewards.
Reward recognizing that broad investigation may be useful while genuinely
no-information repetition is neutral. Reward an accurate inner untrainability
judgment when the engineer cannot preserve a critical inner fact.

Deduct heavily for returning the physical `AGENTS.md`-only vat as the inner
filesystem, omitting required JSON fields, leaking hidden truth or reward text,
inventing executed hashes, changing facts to confirm the inner agent's guess,
grading fluency instead of outcomes, or blaming the inner agent for a simulator
failure.

Treat the engineer's inner observations and judgments as Player-authored
artifacts. False inner outputs, malformed artifacts, inner disclosures, and wrong
inner trainability judgments lower outer task reward. They do not make a coherent
outer episode untrainable. A correct inner integrity-failure report can earn
reward. Only an actual outer simulator or infrastructure failure invalidates the
outer episode.

## Termination conditions

- **Successful:** all supplied actions receive coherent artifacts and the final
  judgment correctly reflects the inner trajectory and simulator integrity.
- **Failed:** hidden truth/reward leakage, persistent malformed artifacts,
  reward hacking, or knowingly false execution claims.
- **Neutral:** incomplete processing, honest blocker, or exhausted action budget.

Outer simulation failures are untrainable.
