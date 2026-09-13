# CaptureFlow

CaptureFlow is a local, single-user project continuity assistant. Its interface is
persisted text chat: it keeps bounded conversation context and maintains
source-grounded local records in SQLite — Calendar events, Ideas & Notes, and
per-project tracker records such as tasks, decisions, and blockers. The default
workspace keeps chat beside a read-only snapshot of the selected project;
Calendar, Tracker, and Ideas & Notes are also available as full management views.
The sidebar scopes its conversation list to the selected project. Deleting a
project removes its tracker state and audit links but keeps its conversations,
messages, calendar events, ideas, and notes as unassigned local records.

It is deliberately not a general chatbot. On each ordinary chat turn one model
call reads the message in its bounded conversation context and returns a typed
decision. Python validates the source quote, candidate target, lifecycle,
duplicate/conflict, and credential rules before any SQLite write. The working
principle is: understanding belongs to the model; authorization stays in Python.
It has no external calendar synchronization, email, meetings, vector retrieval,
or access to external services.

## How a chat turn works

With `CAPTUREFLOW_API_KEY` and `CAPTUREFLOW_MODEL` configured, each ordinary
message goes to the contextual router
([conversation_router_service.py](src/captureflow/conversation_router_service.py)):

1. A deterministic credential check runs first and blocks the turn without a
   model call.
2. The model receives the bounded recent message window (at most eight messages
   and 12,000 characters), at most three candidate events, ideas/notes, and
   project records, the current local date and time, a calendar table, and any
   pending clarification.
3. It returns one `ConversationDecision`: `calendar_create`, `calendar_update`,
   `calendar_delete`, `save_idea`, `save_note`, `tracker_change` (for a
   conversation assigned to a project), `local_answer`, `project_answer`,
   `clarify`, or `reply_only`.
4. Python checks every cited quote and candidate ID, applies policy, and commits
   at most one local write through an audited, restart-safe operation. Every
   committed change is visible in Chat and has a guarded Undo.

Dates and times arrive already normalized by the model as `date`, `start_time`,
`end_time`, `duration_minutes`, `timezone`, and `recurring` fields. Python attaches
the configured timezone and checks structure and plausibility; it does not
interpret date or time wording.

Behavior worth knowing:

- A follow-up such as `把这场会议改到明天下午两点到三点` updates the event the
  conversation created. An update or delete that names no event the conversation
  identified asks which event is meant.
- A stated foreign timezone or a repeating schedule is clarified rather than
  stored, because neither is supported.
- A meeting reported as already held becomes a note, not a Calendar event.
- One turn commits one action. When a message states a second item, CaptureFlow
  asks about it in a follow-up question.
- A task deadline appears on the Calendar without becoming an event.
- Replacing an active decision or constraint waits for approval from its Chat card.
- A meeting question (`local_answer`) is answered by a deterministic lookup of
  active Calendar, Ideas & Notes, and same-project records, with source cards and
  times rendered from SQLite. It works in an unassigned conversation.

Known limits: a bare hour such as `明天十点` is resolved to 10:00 without a
question, and the reply states it; a message that only mentions an event by name
does not count as identifying it for an update; range questions such as
`我下周有什么安排？` are not answered.

These explicit commands need no model call:

```text
/event YYYY-MM-DD | title
/idea text
/note text
```

For example, `/event 2026-09-15 | Submit report` creates an all-day local event.
Without model configuration, ordinary text is persisted with a visible
configuration failure and creates no record.

### Optional diagnostics

Turn on **Diagnostics** in the Streamlit sidebar before submitting a message. The
matching assistant response then includes a collapsed session-only panel with
the intent, candidate counts, model-call count, model time, local
validation/write time, outcome, and total submit time. Diagnostics are off by
default and do not store prompts, raw model output, credentials, or new database
rows.

## Status and evidence

The deterministic suite passes 159 tests. It verifies the database, validation,
and model-write boundaries with scripted model decisions.

The frozen Phase 10 live acceptance (16 ordered assertions against
`gpt-5.6-terra`) exercises calendar creation and correction, task distinction,
meeting questions, replay, cross-conversation lookups, and an ambiguous
correction. Each run is demonstration evidence, not a reliability benchmark.

| Document | Covers |
| --- | --- |
| [phase-9-local-context-evidence.md](docs/phase-9-local-context-evidence.md) | Deterministic local recall |
| [phase-10-baseline.md](docs/phase-10-baseline.md) | The contextual router's acceptance contract |
| [phase-10-live-evidence.md](docs/phase-10-live-evidence.md) | First frozen live acceptance |
| [phase-10.4](docs/phase-10.4-calendar-deletion-and-noon-times.md) – [phase-10.7](docs/phase-10.7-capability-boundary-probe.md) | Feedback rounds: deletion, compound turns, deadlines, boundary probe |
| [phase-10.8-frozen-acceptance-rerun.md](docs/phase-10.8-frozen-acceptance-rerun.md) | Acceptance reruns after those rounds |
| [phase-11-model-normalized-time-design.md](docs/phase-11-model-normalized-time-design.md) | Moving date and time understanding to the model |
| [phase-11-evidence.md](docs/phase-11-evidence.md) | Phase 11 results and acceptance reruns |
| [redundancy-cleanup.md](docs/redundancy-cleanup.md) | Removal of code the app no longer reaches |

The live acceptance needs explicit API-cost approval. It creates only an isolated
local database and writes a trace without credentials or raw model output to the
ignored output directory:

```powershell
python scripts/run_phase10_live_acceptance.py --output-dir runs/phase10-live-YYYYMMDD
```

## Development environment

Use the existing Conda environment:

```powershell
conda activate llm-course
```

Install the project's declared dependencies into that environment, then run the
app:

```powershell
python -m pip install -e .
streamlit run app.py
```

When several checkouts of this repository exist, an editable install of one of
them shadows the others, and the app then runs the wrong code without any error.
`scripts/run_local_app.ps1` puts this checkout's `src` first, refuses to start if
`captureflow` still imports from elsewhere, and runs Streamlit from the
repository root regardless of the current directory:

```powershell
.\scripts\run_local_app.ps1
```

The application stores local data in `data/captureflow.sqlite3`, which is not
committed. To run the deterministic tests:

```powershell
python -m unittest discover -s tests -v
```

## Model configuration

The router uses a configured OpenAI-compatible chat-completions endpoint. Set
these environment variables in the shell that starts Streamlit:

```powershell
$env:CAPTUREFLOW_API_KEY = "..."
$env:CAPTUREFLOW_MODEL = "your-model-id"
# Optional for a compatible provider endpoint:
$env:CAPTUREFLOW_BASE_URL = "https://provider.example/v1"
# Optional; defaults to the computer's timezone:
$env:CAPTUREFLOW_TIMEZONE = "America/Vancouver"
# Optional; defaults to 30 seconds and must be positive:
$env:CAPTUREFLOW_MODEL_TIMEOUT_SECONDS = "30"
streamlit run app.py
```

The endpoint must support chat-completions JSON-object responses. Without the
required key and model variables, Chat saves and displays a typed configuration
failure; it does not fabricate a result.
