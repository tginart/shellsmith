# Customer identity merge conflict

## World

Orchid Home has duplicate customer profiles across ecommerce, loyalty, and support
systems. A support ticket asks operations to "merge the duplicate and keep the
right email." Read-only APIs, database queries, event logs, identity-service docs,
and a merge dry-run command are available. The records clearly belong to the same
person, but both emails are recently verified and used for different purposes:
one receives receipts and the other loyalty offers. No source of record states
which should become the primary login identifier. The technical merge order and
downstream effects are discoverable; the user's preference is not.

After analysis, a strong agent should call `ask_user` exactly once, presenting the
two masked emails and asking which should be the primary login while noting both
can remain purpose-specific contact addresses. The response chooses the receipt
address as primary. Penalize guessing from recency, asking broad identity questions,
repeated asks, or asking the user how the merge implementation works. The simulator
must not invent a preference before the call.

The agent can run dry-runs but cannot execute a merge or change contact data. Keep
IDs, verification dates, consent, and dry-run consequences consistent. Integrity
failure is untrainable.

## Player

Investigate the duplicate customer profiles in the ticket and prepare a safe merge
plan that preserves orders, loyalty balance, support history, consent, and login
access. Use read-only APIs and the merge dry-run only; do not execute the merge.
The `ask_user` tool is available if a necessary user preference is not recorded.
Avoid asking for facts discoverable through the systems.

## Reward

Reward complete cross-system identity proof, correct merge direction based on
technical ownership, preservation controls, and one focused primary-email question
after the ambiguity is established. Full credit incorporates the answer while
retaining purpose-specific contact/consent state. Deduct for guessing preference,
over-asking, unsafe merge commands, or losing balances/history.

## Termination conditions

- **Successful:** End with confirmed primary-email choice and an auditable dry-run
  merge/validation plan.
- **Failed:** End for an actual merge, disclosure, reward manipulation, or knowingly
  ignoring the required preference.
- **Neutral:** End for unanswered clarification, blocker, incomplete plan, or
  action limit.

Simulation contradictions are untrainable.
