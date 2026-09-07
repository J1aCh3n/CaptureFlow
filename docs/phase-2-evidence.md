# Phase 2 evidence status

**Status:** implementation and scripted routes verified; Phase 2 gate is not
complete until three configured live runs are saved.

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

Result on 2026-09-06: 30 tests passed.

The tests cover valid and invalid plans, English and Chinese date ambiguity,
documented `M/D` date rollover, date-tool failure, one successful revision,
meeting-task revision with complete cross-clause evidence, retry exhaustion
with generated items omitted, persisted run traces, complete run-history
retrieval, an external-search capability boundary, preserved ambiguous source
terms, no redundant summary outside a summary plan, no exposed unimplemented
conditional fields, omitted temperature for reasoning-model compatibility, and
a visible configuration failure when no model is configured. Source grounding
checks original date expressions but not deterministic typed date resolutions.

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
- The source-grounding Critic receives the original due expression but not the
  deterministic date-tool result; it must not demand the resolved calendar
  date or year in the source quote.
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

This is one retained live trace, not evidence that the Phase 2 gate is
complete. It is a meeting note rather than one of the required article, email,
and idea examples.

## Required live evidence

Before marking the Phase 2 gate complete, run and retain one configured live
capture for each of these input styles: article, email, and idea. Each stored
trace must show its validated plan, output or typed failure, grounding result,
warnings, and model-call count. Scripted results do not establish live planning
or grounding quality.
