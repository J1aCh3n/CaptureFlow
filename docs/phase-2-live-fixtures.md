# Phase 2 frozen live fixtures

Frozen on 2026-09-07. The capture text already stored in SQLite is the
authoritative input. These examples are delivery checks, not a statistical
benchmark.

## Article

- Conversation: `news`
- Capture ID: `6b6b8a82-8ea2-4825-a3b5-8d706cac8efb`
- Title: `US watchdog opens probe into Tesla's Cybercab self-certification`
- Expected action set: exactly `summarize`
- Minimum acceptable output: at least one source-grounded claim that NHTSA
  opened an Audit Query into Tesla's Cybercab self-certification.
- Optional supported detail: the inquiry examines Tesla's process or technical
  data, or the dispute over claimed FMVSS inapplicability.
- Unacceptable: an extracted user task, invented legal conclusions, or runtime
  rejection solely because a supported summary is concise or close to the
  source wording.

## Email

- Conversation: `emails`
- Capture ID: `c0fb08d9-0157-4315-a2fe-ea08917a6da9`
- Expected action set: exactly `extract_task`
- Minimum acceptable output: one grounded task to pay the outstanding `$42.98`
  utility payment, retaining the stated urgency.
- Date expectation: no calendar due date. `At your earliest convenience` is
  urgency language, not a resolvable date.
- Unacceptable: an invented date or an unrelated summary.

## Idea

- Conversation: `ideas`
- Capture ID: `f80470b1-40c4-48c7-9b5e-5cb6e2d6b869`
- Expected action set: exactly `summarize`; the capture describes a product
  concept rather than assigning an implementation task.
- Minimum acceptable output: at least one grounded claim that the proposed
  assistant turns meeting transcripts or chat records into a risk-aware,
  structured timeline and task output.
- Optional supported detail: the proposed ASR, LLM/function-calling,
  SQLite/PostgreSQL, and Streamlit/React pipeline.
- Unacceptable: treating suggested technologies as committed implementation,
  inventing owners or dates, or runtime rejection solely for lack of
  higher-level synthesis.

## Evidence accounting

- Preserve every attempt in run history and identify the latest evaluated run.
- `completed_with_omissions` is acceptable only when the fixture's minimum
  output remains.
- `needs_review` is acceptable only when the review item is genuinely present
  in the source and the minimum output remains.
- `grounding_failed` and `failed` do not pass a fixture.
- Scripted tests establish control-flow behavior; these live runs establish
  only that the frozen examples work with the configured model.
