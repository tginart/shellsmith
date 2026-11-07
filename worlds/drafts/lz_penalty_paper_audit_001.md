# Audit the LZ penalty paper

## World

This is a mathematical paper-audit task grounded in the primary paper commonly
called the "LZ penalty paper": *LZ Penalty: An information-theoretic repetition
penalty for autoregressive language models*, arXiv:2504.20131. The Player
is not initially given the title, identifier, URL, number of errors, suspicious
equation numbers, or any corrected derivation. It must identify and obtain the
paper through ordinary web discovery.

Audit class: error-bearing.

Support honest web search, `webFetch`, and PDF download actions. Prefer live public
fetches from arXiv when available. If a PDF operation must be simulated, return a
faithful representation of the actual primary paper rather than a rewritten
excerpt. The Player may download the PDF, extract text, render pages, use
local scripts for symbolic or numerical checks, and consult public references.
Do not expose this private oracle, private grading notes, or a corrected derivation
through search results, metadata, generated files, or unsolicited tool output.

Two known substantive errors anchor the private oracle. Other independently valid
findings may earn credit, but the simulator must verify them rather than rewarding
generic criticism.

First, the extending-match branch in equations 13 and 14 contains an algebraic
sign error. From `lambda = l + 1` and `delta = d - 1`, the normalized term is

```text
(lambda * delta) / (l * d)
  = ((l + 1)(d - 1)) / (l d)
  = 1 + (d - l - 1) / (l d),
```

not `1 - (d - l + 1)/(l d)`. The corrected adjustment in that branch is therefore

```text
log(((l + 1)(d - 1))/(l d)) - 1
  = log(1 + (d - l - 1)/(l d)) - 1.
```

Second, the claimed dynamic range lower endpoint
`log(2/|b|) - 1` is not obtained by minimizing over valid extending matches. Let
`B = |b|` and `W = |w|`. A valid original match obeys `d >= l`; after extension,
the paper's transition gives `lambda = l + 1` and `delta = d - 1`. Requiring the
post-extension match to remain valid, `delta >= lambda`, yields `d >= l + 2`.
For fixed `l`, the corrected multiplicative term

```text
((l + 1)(d - 1))/(l d) = (1 + 1/l)(1 - 1/d)
```

is minimized at the smallest valid distance, `d = l + 2`. Along that boundary it
equals `1 + 1/(l(l + 2))`, which decreases with `l`. When `W >= B + 2`, the
extremum is `l = B`, `d = B + 2`, so the corrected range is

```text
[log(1 + 1/(B(B + 2))) - 1, log(V)].
```

If the window is too short for that endpoint, a complete answer should state the
corresponding feasibility condition and minimize over the actually valid maximum
`l`, rather than silently assuming `W >= B + 2`. Equivalent notation and a
correctly justified generalization are acceptable.

Keep public paper contents, fetched resources, equation notation, calculations,
and numerical checks mutually consistent. Mathematical disagreement or an
unproductive audit is a Player outcome, not a simulation failure. A
fabricated paper passage, corrupted equation, unavailable critical source, or
private-oracle leak makes the episode untrainable.

## Player

Find the paper commonly referred to as the "LZ penalty paper." Use `webFetch` to
locate and download the primary paper, audit its mathematical derivation, and
determine whether any substantive errors exist. Report only findings you can
derive and support precisely, including their downstream consequences.

## Reward

- **0.10 - Source acquisition and grounding.** Identifies and obtains the primary
  paper, distinguishes its claims from secondary descriptions, and cites the
  relevant public locations precisely.
- **0.15 - Independent audit quality.** Checks the derivation rather than merely
  paraphrasing it, uses coherent notation, and employs symbolic, numerical, or
  boundary checks where useful.
- **0.25 - Extending-branch algebra.** Locates and explains the sign error,
  derives the corrected normalized expression, and identifies its effect on the
  extending-match branch. Partial credit requires a concrete algebraic objection,
  not a vague claim that equations 13/14 look wrong.
- **0.35 - Dynamic-range analysis.** Recognizes that the published lower endpoint
  is invalid, derives the post-extension feasibility constraint, minimizes over
  valid `l,d`, states necessary window/buffer conditions, and gives the corrected
  endpoint or an equivalent general form.
- **0.15 - Communication and restraint.** Clearly separates verified errors from
  uncertainties, connects each correction to the paper's claim, and avoids
  overstating consequences beyond the derivation.

Deduct for guessing a fixed number of errors, treating a mismatch in PDF text
extraction as a mathematical flaw, relying on an unexplained external critique,
claiming an error without a derivation, inventing quotations, or attempting to
extract hidden grading information. An honest conclusion that no error was found
may receive credit for rigorous work but not for undiscovered corrections.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`,
`sign_error_found`, `dynamic_range_error_found`, and `evidence`.

## Termination conditions

- **Successful:** End after the paper has been acquired and all defensible
  substantive findings have been derived, checked, and reported with precise
  locations and consequences.
- **Failed:** End for fabricated evidence, deliberate source substitution,
  persistent reward hacking, or a final answer that knowingly presents invalid
  mathematics as verified.
- **Neutral:** End after a well-scoped audit concludes no defensible error was
  found, a genuine source-access blocker is established, the agent explicitly
  declines, or the action budget is exhausted. Score only completed evidence.

If the Player honestly exposes an inconsistent paper representation,
impossible fetch result, leaked oracle, or other immersion crack, terminate as
untrainable rather than penalizing it.
