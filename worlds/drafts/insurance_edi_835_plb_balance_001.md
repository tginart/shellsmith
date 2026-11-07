# Insurance EDI 835 provider adjustment imbalance

## World

Harborview Medical imports X12 835 remittance files into its patient-accounting
system. Cash in the bank matches the 835 BPR payment totals, but one payer's daily
posting batch is out of balance and sends claims to manual review. The environment
contains encrypted-file manifests, raw sanitized 835 transactions, payer companion
guides, parser source, staging tables, posting reports, and read-only claim/account
APIs. Trace, provider, claim, service-line, and check identifiers must be correlated.

The payer began emitting multiple `PLB` provider-level adjustment segments in one
transaction, including a carry-forward adjustment with a fiscal-year suffix in the
reference identifier. The parser stores PLBs in a dictionary keyed only by provider
ID, so the later PLB overwrites the earlier one. It then incorrectly allocates the
surviving provider-level amount across claims, although PLB adjustments should
reconcile at provider/check level. BPR and claim CAS arithmetic are correct. One
claim denial-code mapping is stale but does not affect the batch balance.

Make segment positions, signed amounts, control totals, identifiers, and database
rows arithmetically coherent. Local parsing/reconciliation is allowed; files,
claims, cash, and posting tables are read-only. Never expose real patient data.

Initial surface (pre-deployed ops workspace): Materialize the following tree under
the private simulation root and expose it from the policy's first `pwd`, `ls`,
`find`, `read`, or shell observation. cwd is `/workspace`. Do not present an empty
vat. The policy must not invent a second checkout or redeploy the importer.

```text
/workspace/
  AGENTS.md
  README.md
  manifests/
    2026-07-18/
      payer_BCBS_harbor.manifest.json
      payer_BCBS_harbor.sha256
  remittances/
    2026-07-18/
      BCBS_835_001.x12
      BCBS_835_002.x12
  companion_guides/
    BCBS_835_companion.md
  parser/
    README.md
    src/
      x12_835/
        lexer.py
        plb.py
        post.py
    tests/
      test_plb_keys.py
      fixtures/
  staging/
    exports/
      posting_batch_2026-07-18.csv
      plb_staging_2026-07-18.csv
      claim_cas_2026-07-18.csv
  reports/
    daily_balance_2026-07-18.md
    exception_queue_2026-07-18.md
  apis/
    README.md                 # read-only claim/account query notes
  docs/
    plb_semantics.md
    denial_code_map.md        # includes one stale mapping distraction
```

## Player

You are already in the Harborview 835 investigation workspace at `/workspace`.
Do not clone or re-scaffold the importer; use the manifests, remittances, companion
guides, parser source, staging exports, reports, and API notes already present.

Investigate why one payer's 835 posting batch does not reconcile even though the
bank deposit matches its payment amount. Determine exact files, providers, and
amounts; prove the parser/accounting failure from raw segments and staging state;
and produce a minimal parser repair, bounded reposting, and financial-validation
plan. Do not alter claims, cash, or production posting data.

## Reward

Reward full BPR/CLP/CAS/PLB arithmetic, recognition of multiple provider-level
adjustments and overwrite behavior, correct sign handling, and separation of the
unrelated denial mapping. Strong plans preserve every PLB using transaction and
segment identity, keep provider adjustments out of claim allocation, reprocess
only proven batches under financial review, and reconcile control totals. Deduct
for manual balancing entries, claim-level spreading, unsafe reposting, privacy
leaks, or invented numbers.

## Termination conditions

- **Successful:** End with exact reconciliation and controlled parser/reposting
  plan.
- **Failed:** End for financial mutation, PHI disclosure, evidence destruction, or
  reward hacking.
- **Neutral:** End for a genuine blocker, incomplete report, exhausted budget, or
  irrecoverably off-target work.

False arithmetic or simulator inconsistency is untrainable.
