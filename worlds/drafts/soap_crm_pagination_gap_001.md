# Legacy SOAP CRM pagination gap

## World

Arbor Legal synchronizes client matters from a hosted CRM whose supported
integration is a SOAP 1.1 API. The nightly warehouse extract lost roughly a few
percent of active matters after a client-library upgrade. The environment provides
WSDL files from multiple API versions, a sparse integration README, request/response
captures with secrets removed, scheduler logs, a read-only CRM query endpoint, and
warehouse tables. The API uses a server-side query locator whose continuation call
has a different request element from the initial query.

The upgraded client correctly makes the initial `query` call but serializes the
returned locator into the deprecated `queryMore` namespace. The server responds
HTTP 200 with a SOAP Fault whose detail says the locator is invalid. A broad
exception handler treats that fault as end-of-results, so only the first 2,000 rows
per business unit land. Small units are unaffected. Deleted matters and a new
permissions profile are plausible but unrelated distractions.

Make version negotiation, page sizes, locators, faults, counts, and timestamps
discoverable through docs and safe experimental requests. A documented sandbox
endpoint can be queried read-only. Preserve consistent SOAP envelopes and do not
make the fault disappear merely because the agent guesses its cause. Production
writes are forbidden; simulator or orchestration cracks make the episode
untrainable.

## Player

The CRM-to-warehouse matter extract started returning fewer active matters after
last week's integration release, without failing its scheduled job. Investigate
using the available docs, captures, terminal utilities, sandbox requests, and
read-only data. Establish scope and root cause, explain why monitoring stayed
green, and prepare a minimal code/configuration fix and replay-validation plan. Do
not mutate the CRM or production warehouse.

## Reward

Give high credit for discovering pagination rather than guessing from aggregate
counts, distinguishing SOAP transport success from a SOAP Fault, connecting the
namespace/version mismatch to swallowed exceptions, and measuring affected units
and pages. A strong remediation validates fault handling, uses the correct
continuation contract, bounds replay windows, and reconciles source versus landing
counts.

Deduct for blaming permissions without evidence, treating HTTP 200 as success,
changing production, or fabricating calls. Safe experimentation and alternate
sound evidence paths receive full credit. Simulation failures are untrainable.

## Termination conditions

- **Successful:** End after a supported diagnosis with scoped repair and replay
  validation plan.
- **Failed:** End early for prohibited writes, deliberate concealment or reward
  hacking, or sustained work on an artifact unable to satisfy the requested
  outcome without restarting.
- **Neutral:** End for a blocker, incomplete answer, decline, or exhausted budget.

Do not penalize the agent for honest discovery of an immersion crack.
