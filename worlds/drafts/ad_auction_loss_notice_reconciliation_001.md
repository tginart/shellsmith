# Ad-auction loss-notice reconciliation

## World

OrbitDSP buys mobile ad inventory through several OpenRTB exchanges. Finance finds
that one exchange's billed spend is consistently higher than the auction ledger,
while impression counts reconcile. The environment includes archived bid requests
and responses, win/loss/billing notices, exchange integration guides, HTTP callback
logs, bidder source, currency tables, and read-only warehouse models. Identifiers
must be correlated across auction ID, impression ID, bid ID, and notice URL macros.

The exchange reports clearing prices through `${AUCTION_PRICE}` in encrypted CPM
units and documents a required exchange-specific decoder. A bidder refactor routes
loss notices through the standard plaintext macro decoder, then records the decoded
loss price as micros-per-impression. Win notices still use the correct adapter.
The spend model estimates second-price savings from loss prices, so corrupted loss
prices inflate modeled savings and lower internal spend without changing delivered
impressions or actual billing. A normal bid-currency conversion and a small invalid-
traffic credit are unrelated.

Make sampled encrypted macros, decoder outputs, CPM/micros conversions, currencies,
auction relationships, and totals computationally consistent. The agent may run
local decoders and warehouse queries but cannot send callbacks, alter bidder code,
or restate finance data. Do not reveal the adapter defect on a vague query.

## Player

Investigate the spend mismatch for the affected exchange. Reconcile billing,
auction, and notice data; quantify exact dates and campaigns; prove the causal path
through the integration; and propose a minimal correction, bounded recomputation,
and validation plan. Production systems and partner callbacks are read-only.

## Reward

Reward correct multi-ID joins, use of the exchange's primary macro contract and
decoder, dimensional analysis across CPM, per-impression, and micros, and proof
that loss-price-derived savings—not delivered media—causes the discrepancy. Strong
plans restore adapter dispatch, add golden encrypted fixtures and unit invariants,
bound model recomputation, and reconcile to invoices and wins. Deduct for treating
impressions as missing, applying arbitrary currency factors, blaming the exchange
without evidence, or fabricating decoder results.

## Termination conditions

- **Successful:** End with quantified reconciliation, reproducible diagnosis, and
  controlled code/data correction plan.
- **Failed:** End for callback transmission, production restatement, or reward
  manipulation.
- **Neutral:** End for a real blocker, incomplete final report, decline, or budget
  exhaustion, including irrecoverably off-target work.

False computation or simulator contradiction makes the episode untrainable.
