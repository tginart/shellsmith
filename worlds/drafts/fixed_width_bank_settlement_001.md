# Fixed-width bank settlement mismatch

## World

Morrow Travel reconciles card settlements against a bank's fixed-width mainframe
export. A month-end report differs from cash receipts for one merchant account.
The environment includes SFTP manifests, encrypted and decrypted sample files,
COBOL copybooks, a Python parser, database staging tables, settlement reports, and
deployment history. Several copybooks exist, and the bank's file header carries a
two-character layout version that the current parser ignores.

On the bank's `C7` layout, the settlement amount grew from 11 to 13 characters and
the following debit/credit indicator shifted two columns. The parser still slices
the `C6` positions, producing plausible smaller positive amounts for records whose
leading digits are spaces and reversing a subset of debits. File-level totals in
the trailer use the correct values. Only files with header version C7 after the
bank cutover are affected. A holiday delay and duplicate-looking adjustment IDs
are valid business behavior, not causes.

Make the layout choice and cutover discoverable from headers, copybooks, source,
raw lines, trailer controls, and staging data. The simulator should execute local
parsing and arithmetic when requested and keep exact positions/totals coherent.
Bank SFTP and production databases are read-only.

## Player

Month-end settlement records for one bank merchant account do not reconcile to
cash receipts, although every expected SFTP file arrived. Investigate with the
available terminal tools and documentation. Quantify the affected files and
financial difference, identify the parser defect with raw-record evidence, and
produce a safe patch/backfill/validation plan. Do not modify production data or
retrieve new bank files.

## Reward

Reward selecting the correct copybook from file evidence, demonstrating field
boundaries on representative records, reconciling decoded totals to trailer and
cash controls, and scoping the C7 cutover. The repair should explicitly dispatch on
layout version, add regression fixtures for credits and debits, preserve originals,
and bound the backfill.

Deduct for manual balancing adjustments, applying C7 positions to all history,
ignoring trailer controls, unsafe writes, or invented results. Simulator
contradictions and non-executed computational claims make the episode untrainable.

## Termination conditions

- **Successful:** End with a quantified diagnosis and controlled parser/backfill
  plan supported by reproducible evidence.
- **Failed:** End for financial mutation, evidence destruction, or reward hacking.
- **Neutral:** End for a stated blocker, incomplete report, decline, or budget
  exhaustion, including irrecoverable pursuit of an unrelated implementation;
  score only real progress.

Integrity failures are untrainable.
