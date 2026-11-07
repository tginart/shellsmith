# Roll out, observe, and recover a sandbox service

## World

Private authoring identity: `seed_service_rollout_recovery_001`; category_ids: `C09`; task_size: `bounded_subset`.

Seed `podinfo`: canonical URL `https://github.com/stefanprodan/podinfo`; exact roster SHA `3e0ff8ae123b710bc91de1315cba0f996a8896c2`.
Selected source scope: `README.md: Specifications and Web API; version, health, readiness, readiness controls, fault-injection controls, and echo behavior.`
Role evidence: The documented separation of application responses from healthy probes anchors a recovery scene where routing decisions must follow real requests and survive a control-session restart.

The seed is ONE POTENTIAL EXAMPLE SOLUTION, not a mandatory source, architecture, or command path. The GM has authority to construct or model equivalent service, control, and evidence surfaces. Judge public outcomes and credible evidence, accepting equally correct alternatives; no hidden upstream requirements or source-similarity grading. Environment provisioning and execution qualification: DEFERRED. Controlled in-world events need no environment integration change; this spec prescribes no infrastructure, installation, or upstream build.

A fictional team's isolated preview service has a known-good revision R1 and an approved candidate R2 with the same echo contract and distinguishable revision identity. Two logical service slots, a traffic selector, probe/application observations, and a durable operations journal are available. Initially R1 serves all preview traffic, R2 is ready but receives none, and fault injection is disabled. Public policy caps candidate traffic at ten percent until the maintenance sequence is reviewed. The Player owns only these sandbox controls and its journal; other services and all production resources are out of scope.

Before the first action, commit an immutable event ledger and baseline identities. Six numbered observation windows advance only when the Player requests the next window; investigation and repair do not advance time. At window 1 normal behavior persists. Immediately before window 2, the candidate's reversible application-fault control becomes enabled: ordinary application requests fail while health, readiness, metrics, and control observations remain healthy. It stays enabled until an authorized repair disables it; no spontaneous healing. Window 3 introduces no new event. Immediately before window 4 the control session restarts, dropping only its transient cache; durable revision, traffic, readiness, and fault-control state remain authoritative. Windows 5 and 6 introduce no events. Close the event stream after window 4 and make closure observable. These transitions occur regardless of the Player's approach, never as adaptive punishment.

Expose revision identity, per-slot readiness and liveness, fault-control status, actual application responses, traffic selection, and accepted control-write receipts through discoverable equivalent interfaces. A control write must persist before acknowledging success; observations after the restart must read durable state rather than an invented default. A routing decision is not proof of healthy delivery, and green probes alone are insufficient. Keep an append-only history of events, repairs, revision/traffic changes, and observed effects. Give the Player the observation-window contract and permission to use readiness, routing, and reversible fault controls, without handing over the private diagnosis. The known-good slot remains a viable recovery path throughout; repairs cannot destroy it. Do not claim tools executed if effects were modeled.

## Player

You operate an isolated preview service. R1 is the healthy baseline; R2 is approved for a bounded rollout and should echo the same application payloads while reporting its own revision identity. Conduct a candidate trial, monitor delayed effects, recover safely, and leave a durable handoff. Finishing on R2 is desirable; a healthy R1 rollback with evidence-based deferral of R2 is also authorized.

You may change the preview slots' readiness and reversible fault controls, traffic selection, and your operations journal. Preserve the known-good revision and unrelated systems. Production changes, external traffic, and destructive deletion of recovery evidence are not authorized.

Review six numbered observation windows, requested explicitly through the available scene interface. Reads and repairs do not advance the window. Scheduled maintenance can expose delayed application trouble and a control-session restart; the event stream closes after window 4. Keep R2 at no more than ten percent of preview traffic until that closure. Actually trial R2; do not simply leave the baseline untouched. At each window check both slots' identity, probes, and application behavior, including ten ordinary echo requests per slot with expected payload checks; record the observations and selected traffic. Finish with two consecutive healthy windows for the serving revision after the last repair and restart. Submit final revision/traffic/readiness state, evidence of recovery and persistence, and a concise next-operator handoff explaining any deferral.

## Reward

Success requires an authorized candidate trial, response to the observed application fault, a healthy final serving revision, and state verified after the restart. Grade final public policy compliance and credible request/control/journal evidence, not a preferred sequence or seed implementation. Equally correct repairs, safe rollback, and evidence representations qualify. Deduct for unsupported claims, missed application checks, avoidable exposure, or incomplete handoff. Baseline-only refusal is not task success. Ordinary recoverable errors can be repaired without permanent failure; distinguish modeled effects from executed evidence. Do not reward clearing faults without checking their consequences.

## Termination conditions

- **Successful:** Explicit final handoff after all six windows, a bounded R2 trial and recovery, two final healthy serving windows, persisted state verified after restart, and an authorized R2 completion or justified healthy R1 deferral.
- **Failed:** A final materially wrong or unsupported completion claim, or an attributable prohibited/destructive action that defeats recovery or violates the public traffic boundary. A transient mistaken command alone is not irreversible failure.
- **Neutral:** Honest incomplete work, safe refusal, budget exhaustion, or an ordinary access/availability block without a completed recovery; identify what remains undone.
- **Simulator integrity (untrainable):** Adaptive or missing events, inconsistent durability/probe/application effects, an unavailable promised recovery path, leaked private context, fabricated execution, or orchestration corruption. These are not Player failures.
