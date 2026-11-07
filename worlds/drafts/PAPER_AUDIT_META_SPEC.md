# Recent-paper audit world file meta-spec

This is an authoring template, not a runnable world file. It defines balanced research
audits where the policy must locate a primary paper, inspect a mathematical or
computer-science claim, and decide whether a substantive error exists.

## Campaign balance

Maintain an approximately 50/50 split:

- **Error-bearing worlds:** a later primary revision, erratum, author correction,
  or corrected proof/source establishes a concrete mistake in an earlier version.
- **No-error controls:** the focused derivation is correct. The world supplies
  credible objections and edge cases, but a rigorous audit should reject them
  rather than inventing a flaw.

Put exactly one private marker in `World`: `Audit class: error-bearing.` or
`Audit class: no-error control.` Never expose this class to the Player.

## Primary provenance

Prefer recent arXiv, OpenReview, conference, or journal papers with stable public
versions. Error-bearing worlds must identify the exact earlier and corrected
versions, dates, changed equations/proof steps, and authoritative correction
evidence. A metadata comment alone is a lead, not a sufficient oracle: reconstruct
the correction and its consequence.

Controls must identify a focused nontrivial claim, independently derive why it is
correct, and specify executable symbolic/numerical checks where appropriate.
Controls should include plausible decoys such as notation ambiguity, finite-precision
differences, omitted implementation detail, a stronger unstated claim, or an
incorrect reimplementation. None may secretly make the target erroneous.

## Player prompt

Do not provide the paper identifier, exact URL, version delta, equation number,
number of errors, suspicious quote, correction, or class. Give only a natural
title, nickname, or descriptive handle that makes the primary paper discoverable.
Require ordinary web discovery followed by acquisition of the primary source, an
independent derivation or proof check, and restrained reporting of only supported
findings. The prompt may describe this capability as `webFetch`, but it must not
imply that a literal executable or dedicated tool with that name exists.

The same prompt posture must work for both classes: “determine whether any
substantive error exists.” Never imply that an error is guaranteed.

## Simulation and grading

The Game Master must support realistic search-engine queries made with
`curl`, `wget`, Python HTTP clients, or equivalent shell actions. Search results
should expose public metadata and candidate links without identifying the correct
result unsolicited. It may then fetch live public PDFs/source, render pages,
extract text, and execute local symbolic or numerical checks. It must faithfully
represent the selected version and must not leak later-version wording into an
earlier-version observation. OCR or PDF extraction artifacts are not paper errors.

Reward source grounding, independent reconstruction, adversarial checks, exact
version comparison when discovered, downstream consequences, and calibrated
conclusions. A correct no-error conclusion earns full credit in a control. Deduct
for asserting that any revision proves an error, hallucinating criticism, relying
only on secondary commentary, metadata guessing, fabricated quotations, or oracle
probing.

Every instance requests `task_outcomes.paper_audit`, including source acquisition,
the focused checks performed, whether a substantive error was found, and concise
evidence. Error-specific fields may be added.

Successful termination follows a reproducible audit and defensible conclusion.
Failed termination covers fabricated evidence, knowingly invalid mathematics,
destructive behavior, or reward hacking. Neutral termination covers genuine source
failure, careful inconclusiveness, decline, or budget exhaustion. A corrupted or
contradictory paper representation is untrainable.

## Quality checklist

- Primary URLs and all referenced versions resolve.
- The private oracle has been independently checked, not copied from metadata.
- The embedded prompt does not leak class or correction.
- The embedded prompt does not leak the paper identifier or exact URL, and a
  realistic search query can discover it.
- The focused claim is difficult enough to require real reasoning.
- A small executable check is possible when computation is relevant.
- Error consequences are neither exaggerated nor omitted.
- Controls contain credible decoys but no hidden falsehood.
- The complete paper set remains near 50/50 by private class.
