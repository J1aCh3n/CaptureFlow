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

Result on 2026-09-06: 20 tests passed.

The tests cover valid and invalid plans, dependency cycles, date ambiguity,
date-tool failure, one successful revision, retry exhaustion with generated
items omitted, persisted run traces, and a visible configuration failure when
no model is configured.

## Required live evidence

Before marking the Phase 2 gate complete, run and retain one configured live
capture for each of these input styles: article, email, and idea. Each stored
trace must show its validated plan, output or typed failure, grounding result,
warnings, and model-call count. Scripted results do not establish live planning
or grounding quality.
