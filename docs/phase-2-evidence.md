# Phase 2 evidence status

**Status:** Phase 2 gate passed on 2026-09-07. The implementation passed 36
scripted tests and all three frozen live checks have retained SQLite traces.

## Implemented boundary

The flow is fixed as Coordinator plan → Organizer draft → deterministic date
resolution → deterministic quote validation → Grounding Critic → at most one
Organizer revision. LangGraph owns the transitions and model-call count; model
adapters only return typed plan, draft, and critic responses.

Implemented actions are deliberately limited to `summarize` and
`extract_task`. Memory inspection, long-term memory, checkpoint resume, and
multi-turn chat remain Phase 3 work.

## Scripted verification

Environment: `llm-course` with Python 3.11.15.

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -v
python -m compileall -q app.py src tests
```

Result on 2026-09-07: 36 tests passed; bytecode compilation also passed.

The tests cover valid and invalid plans, English and Chinese date ambiguity,
documented `M/D` date rollover, date-tool failure, one successful revision,
meeting-task revision with complete cross-clause evidence, retry exhaustion
with generated items omitted, persisted run traces, complete run-history
retrieval, an external-search capability boundary, preserved ambiguous source
terms, no redundant summary outside a summary plan, no exposed unimplemented
conditional fields, omitted temperature for reasoning-model compatibility, and
a visible configuration failure when no model is configured. They also cover
item-level omission after retry exhaustion, semantic Critic review after
deterministic quote filtering, and the separation of source grounding from
summary-usefulness judgments. Source grounding checks original date expressions
but not deterministic typed date resolutions.

## Implemented repairs and diagnostics

- The OpenAI-compatible adapter omits `temperature`, so the configured
  GPT-5.6 Terra endpoint can use its default-only setting.
- `M/D` dates use the reference year when the date is not earlier than the
  reference date; otherwise they roll to the next year. For example, with a
  2026-09-06 reference date, `9/8` resolves to 2026-09-08.
- A task spanning a problem statement, action, and deadline must cite one
  continuous source span containing all needed clauses. The Organizer receives
  Critic feedback for its single permitted revision.
- A task-only plan deterministically omits an unplanned model-generated
  summary and records a warning. An unplanned task remains a typed failure.
- If a Coordinator plan cites a span outside the captured source, it receives
  one structured repair pass with the validation error and the invalid plan.
  A valid repair uses one additional model call; grounding still permits only
  one Organizer revision and rejects all generated items when that revision
  fails.
- Non-calendar urgency phrases such as `at your earliest convenience` and
  `ASAP` are retained in task text but removed from `due_expression`, so they
  do not produce a misleading date-tool warning.
- The source-grounding Critic receives the original due expression but not the
  deterministic date-tool result; it must not demand the resolved calendar
  date or year in the source quote.
- The Critic now evaluates only source entailment and added or changed facts.
  Usefulness, depth, style, brevity, and multi-sentence synthesis are offline
  evaluation criteria and cannot cause a runtime grounding rejection.
- For an idea or product proposal, the Organizer is prompted to synthesize the
  objective, input-to-outcome flow, and explicit risk principle before an
  optional architecture claim. Suggested technologies remain proposals rather
  than implementation commitments, and an idea is not converted into a task
  unless the source assigns work.
- After the one permitted Organizer revision, only specifically rejected items
  are omitted. Supported survivors are returned as
  `completed_with_omissions`; ambiguous dates or explicit confirmations produce
  `needs_review`; `grounding_failed` means no safely retainable item remains.
- Latest-run and run-history views expose structured Critic feedback when a
  run is rejected.

## Saved live trace: meeting note

On 2026-09-07T04:19:32Z, a configured live run processed this meeting note:

> 背景：上线临近；问题：接口延迟阻塞联调；待办：P0，9/8修复交付。

It completed after five model calls with a passing grounding report. The
validated plan selected `extract_task`; the output proposed “修复接口延迟并交付
（P0）”, cited the combined problem-and-task span, and resolved `9/8` to
2026-09-08 using the documented reference-date rule.

The same run retained a review note asking the user to confirm the year in
`9/8`. This is visible evidence of an unresolved Phase 3 limitation: Phase 2
can display review notes, but has no reply, confirm, edit, or task-update
transition. The note is not actionable by entering another capture; doing so
creates a separate capture.

This meeting note is supplementary evidence rather than one of the three
frozen Phase 2 gate fixtures.

## Frozen live-check evidence

The exact article, email, and idea inputs and their human-authored minimum
expectations are frozen in
[`phase-2-live-fixtures.md`](phase-2-live-fixtures.md). Runtime safety and
offline quality are separated in [`gate-contract.md`](gate-contract.md).

The latest persisted run for each frozen capture was checked directly in
`data/captureflow.sqlite3`:

| Fixture | UTC time | Run ID | Result |
| --- | --- | --- | --- |
| Article (`news`) | 2026-09-07T18:49:18.236080Z | `d7b359de-425d-4b72-98ba-4d4ea2c247a8` | `completed`; 3 model calls; exact action set `summarize`; two grounded claims; Critic `pass`; no warnings or errors |
| Email (`emails`) | 2026-09-07T18:20:49.270398Z | `5a79efd5-f762-4267-8f82-744c1836e568` | `completed`; 3 model calls; exact action set `extract_task`; grounded `$42.98` payment task; no invented date; Critic `pass`; no warnings or errors |
| Idea (`ideas`) | 2026-09-07T18:48:40.672185Z | `1f3f62d2-b37c-454b-8e75-34ad79604fd0` | `completed`; 3 model calls; exact action set `summarize`; grounded product/risk and proposed-pipeline claims; Critic `pass`; no warnings or errors |

All three latest runs meet the minimum expectations frozen before the reruns.
Earlier failed attempts remain in run history rather than being removed or
replaced with easier captures.

## Gate conclusion and claim boundary

Phase 2's planned Organizer flow has passed its scripted route tests and three
frozen live acceptance examples. This establishes that the bounded Coordinator
→ Organizer → typed date handling → grounding path works for those retained
examples with the configured model.

It does not establish reliable understanding of arbitrary articles, emails, or
ideas; cross-run stability; general summary quality; or superiority over a
simpler single-prompt design. Those claims require the larger labeled offline
evaluation defined for later work.
