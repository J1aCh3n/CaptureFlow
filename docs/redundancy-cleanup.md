# Redundancy cleanup

After Phase 11 was merged to `main` at `8626655`, everything the running app could
no longer reach was removed. Reachability was traced from `app.py` and the
remaining acceptance script, not guessed from file names. Migrations and existing
SQLite tables were left unchanged, so an existing database keeps its rows.

| | Before (`8626655`) | After |
| --- | --- | --- |
| `src/` | 21 files, 12,170 lines | 14 files, 6,713 lines |
| `tests/` | 23 files, 8,562 lines | 16 files, 5,224 lines |
| `scripts/` (Python) | 2 files | 1 file |
| Deterministic tests | 270 | 159 |

## Commits

Each step passed the full suite and can be reverted on its own.

| Commit | Removed |
| --- | --- |
| `95ff6cf` | The V2 Phase 3 turn router (`router_service`), its `TurnDecision` schema and prompt, `time_tool`, the recurrence phrase detector, and prepared-operation helpers only that router used |
| `dbe9381` | Services the contextual router replaced: the Phase 5 project responder, the Phase 6 tracker proposer, Phase 7 retrieval, Phase 9.1 local-answer synthesis and its query classifiers, the Phase 8 acceptance runner, and the Phase 10 baseline test |
| `8d36083` | The original Phase 1–3 capture workflow and Memory: the plan/organize/critique graph, Memory review, their schemas, storage methods, hidden UI pages, and the `langgraph` dependency |
| `3dd73b3` | Leftovers and readability, listed below |

## What was kept, and where it moved

- Tracker change validation is `project_service.apply_validated_changes`, a plain
  function; its lifecycle tests were kept.
- The credential guard is `credentials.py`, formerly part of `memory.py`.
- The reference clock is in `time_config.py`, formerly in `workflow.py`.
- Local recall is the deterministic Phase 9 lookup, which the router always used.
- `delete_conversation` still clears memory rows an older database may hold,
  because their events have no cascade.

## Readability changes in `3dd73b3`

- `date_tool` keeps the calendar table and `asked_date`, the one day a recall
  question names. The old weekday and month-day vocabulary had no caller left;
  `asked_date` returned the same day as the old lookup on all 48 queries compared.
- `ConversationRouterService.submit` reads as one dispatch. Clarifications and
  local answers share one early return, and `_execute` commits the single write.
- `CalendarEvent.start_date` replaces three copies of the same helper;
  `ProjectContextWindow` is now `ContextWindow`, since every conversation uses it.
- `InvalidStructuredOutput` no longer stores raw output, and the always-false
  Repair diagnostic is gone.
- The Chat snapshot reads the configured timezone like the rest of the app, and
  the UI tests pin the timezone so date-relative assertions do not depend on the
  machine.

## Removed documents

Documents that described removed code or superseded plans were deleted: Phase 1,
2, 7, 8, and 9 system-check evidence, the gate contract, the deferred UI plan,
the Phase 10 worktree plan, and the Phase 10.3 time-parsing specification that
Phase 11 replaced. They remain in Git history.

## Not changed

- The router prompt and the decision schema the model sees.
- `CaptureStore`, whose name predates the chat product. Renaming it would touch
  nearly every test for no behavioral gain.

## Evidence boundary

The evidence for this cleanup is the deterministic suite, which passed after every
commit, plus the old-versus-new comparison for `asked_date`. The frozen Phase 10
live acceptance was attempted at `3dd73b3` but could not run: every model call
returned `429 insufficient_quota` because the API account had no credits. It
should be re-run once credits are available, since `submit` was restructured.
