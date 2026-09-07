# CaptureFlow gate contract

This document separates runtime safety decisions from development quality and
phase-completion evidence. A rule belongs to only one gate.

## 1. Plan contract gate

The Coordinator plan is a hard runtime contract. It must:

- pass the typed schema;
- use only implemented actions;
- cite source spans present in the capture;
- remain within the configured step and model-call budgets.

An invalid plan receives at most one structured repair attempt. CaptureFlow
does not fabricate a fallback plan.

## 2. Item grounding gate

The grounding gate answers only: “Is this generated item supported by its
cited source?” It has two checks:

1. required quotes occur in the capture after whitespace normalization;
2. the Critic checks entailment, changed meaning, and added facts.

A concise paraphrase or restatement passes when the source entails it. Summary
usefulness, depth, style, brevity, and multi-sentence synthesis are not runtime
grounding criteria.

The Organizer receives at most one revision. After the second check,
specifically rejected items are omitted. Unrejected items may be retained, and
items surviving deterministic quote filtering still receive semantic review.
If no supported item remains, the run is `grounding_failed`.

## 3. Tool and review gate

Typed tools are evaluated separately from source grounding. A task retains its
original date expression separately from any deterministic resolution. An
ambiguous date or an explicit review note produces `needs_review`; it does not
invalidate otherwise grounded output. Unsupported date expressions and tool
failures remain visible, while unrelated supported items may survive.

## 4. Run statuses

- `completed`: all returned items are grounded and need no confirmation.
- `completed_with_omissions`: grounded items remain after rejected items were
  removed.
- `needs_review`: grounded output exists, but user confirmation is required.
- `grounding_failed`: no safely retainable generated item remains.
- `failed`: planning, model, schema, or orchestration failed.

## 5. Offline quality evaluation

Quality judgments belong in a labeled development evaluation, not the runtime
Critic. They include:

- summary usefulness, concision, and synthesis;
- action-routing exact-set accuracy;
- task precision and recall;
- unsupported-claim rate;
- Critic false-accept and false-reject rates.

A single live run is demonstration evidence, not a reliability measurement.

## 6. Phase 2 delivery gate

Phase 2 requires passing scripted routes plus retained runs of the frozen
article, email, and idea fixtures in
[`phase-2-live-fixtures.md`](phase-2-live-fixtures.md). Each run is compared
with its human-authored expected action and minimum acceptable output. Failed
attempts remain in history and may not be replaced with easier inputs.
