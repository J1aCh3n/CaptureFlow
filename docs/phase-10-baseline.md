# Phase 10 Baseline

## Scope frozen on 2026-09-09

Phase 10 replaces only top-level natural-language dispatch. It must preserve
deterministic `/event`, `/idea`, and `/note` shortcuts, SQLite validation, and
the existing local-only product boundary.

The target is one contextual structured decision per ordinary natural-language
turn. Python remains the authority for candidate scope, evidence validation,
date resolution, database writes, idempotency, and Undo.

## Current verified behavior

At `b3952eb` (before Phase 10 production changes), `ui.render_chat_input()`
uses keyword/regular-expression predicates before it considers the assigned
project branch. The current router receives only the new user message and an
optional pending clarification; it does not receive normal recent conversation
history.

The two reported calendar messages therefore reproduce as `project_tracker`
when their conversation is assigned to a Project:

1. A detailed Thursday Algorithm & Code Review meeting statement becomes a
   tracker turn instead of a local calendar creation.
2. "这个会议在周四 14:00 - 15:00" becomes a tracker turn; the legacy branch
   cannot resolve "这个会议" from an earlier ordinary message.

Those observations and the desired Phase 10 outcomes are stored in
[`tests/fixtures/phase10_contextual_router_cases.json`](../tests/fixtures/phase10_contextual_router_cases.json).
`tests/test_phase_ten_baseline.py` executes the current UI decision order
against every fixture row. It is a baseline test, not evidence that the desired
contextual router exists.

## Acceptance contract

| Input category | Required intent | Permitted committed effect |
| --- | --- | --- |
| Detailed meeting statement | `calendar_create` | One `create_local_event` |
| Time correction referring to one prior meeting | `calendar_update` | One `update_local_event` |
| Work to complete, not a scheduled meeting | `tracker_change` | Project record only |
| Question about a known meeting | `local_answer` | No write |
| Correction without sufficient replacement time or target | `clarify` | No write |

`这周四` may be resolved by the existing date tool using the supplied reference
date. Bare `周四` remains ambiguous unless the contextual decision identifies
an existing candidate whose date supplies an unambiguous target; otherwise the
router must clarify.

## Deterministic baseline

The repository has no usable `python` command: it is a Microsoft Store alias.
The available interpreter is:

```text
C:\Users\frank\anaconda3\envs\llm-course\python.exe (Python 3.11.15)
```

`pytest` is not installed in that environment. The project uses `unittest`;
before adding the Phase 10 baseline files, the command below completed
successfully with 168 tests in 25.825 seconds:

```powershell
& 'C:\Users\frank\anaconda3\envs\llm-course\python.exe' -m unittest discover -s tests -v
```

After adding the two Phase 10 baseline tests, the same command completed with
170 tests in 25.513 seconds.

The environment's editable CaptureFlow install points at the main workspace.
Every worktree verification must therefore prepend this PowerShell assignment,
then confirm `captureflow.__file__` points at this worktree before running
tests:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
```

No API call is part of this phase. A live acceptance fixture may use the same
case file only after the deterministic Phase 10 gates pass and API use is
explicitly authorized for that run.
