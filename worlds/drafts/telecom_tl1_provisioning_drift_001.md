# Telecom TL1 provisioning drift

## World

Meridian Fiber provisions enterprise circuits through a modern order API layered
over a legacy TL1 network-element gateway. Several completed orders show the right
bandwidth in billing but remain capped at the old rate in one metro ring. The
environment provides order and inventory APIs, gateway spools, TL1 command/response
archives, vendor command manuals, network-element audit output, provisioning source,
and read-only telemetry. Documentation uses service IDs, access IDs, cross-connect
names, and circuit IDs with non-obvious mappings.

For equipment firmware R12, `ED-QOS` returns an intermediate `IP` acknowledgment
and later emits the final completion under the command's correlation tag. A gateway
upgrade treats any syntactically valid `IP` as terminal success, closes the order,
and discards the delayed `DENY` response. The device denies commands only on the
metro ring whose profiles require rate values in kilobits rather than the newer
megabit syntax. Billing consumes the accepted order state, not device audit state.
Two circuits have separate optical degradation alarms and are distractions.

Keep TL1 tags, response timing, equipment profiles, unit conversions, circuit
mappings, and telemetry consistent. Safe retrieval/audit commands are allowed, but
no provisioning command may be sent and no order state may be modified. Promised
commands must operate as documented or the rollout is untrainable.

## Player

Investigate completed bandwidth-upgrade orders whose measured cap did not change.
Trace affected services across order, gateway, inventory, and network-element
evidence; identify the exact protocol and configuration failure; and produce a
safe reconciliation, reprovisioning, and prevention plan. All systems are read-only:
do not issue TL1 provisioning commands or modify orders.

## Reward

Reward accurate identifier mapping, reconstruction of asynchronous `IP` then
`DENY` responses, firmware/profile scoping, and correct Mbps-to-kbps command
semantics. Strong plans fix terminal-response handling, validate per-equipment
units, identify every false-complete order, require network audit after controlled
reprovisioning, and coordinate billing review. Deduct for global reprovisioning,
confusing optical alarms with rate caps, trusting order state alone, or invented
device output.

## Termination conditions

- **Successful:** End with evidence-backed scope and a controlled protocol,
  reprovisioning, and validation plan.
- **Failed:** End for a provisioning command, order mutation, service disruption,
  or reward hacking.
- **Neutral:** End for an honest access blocker, incomplete report, decline, or
  exhausted budget, including persistent unrelated implementation.

Simulator and orchestration failures are untrainable.
