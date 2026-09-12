# CaptureFlow

CaptureFlow is a local, single-user project continuity assistant under staged
development. Its default interface is persisted text chat: it retains bounded
current-conversation context and maintains source-grounded project records in
local SQLite. Calendar events and explicit ideas/notes remain available as
deterministic local capture tools. The default workspace keeps chat beside a
read-only project snapshot that aggregates records from the project's assigned
conversations; Calendar, Tracker, and Ideas & Notes remain available as full
management views. The sidebar scopes its conversation list to the selected
project. Project deletion removes project tracker state and audit links, but
retains its conversations, messages, calendar events, ideas, and notes as
unassigned local records.

It is deliberately not a general chatbot. On each ordinary chat turn one model
call reads the message in its bounded conversation context and returns a typed
decision — a local calendar event, idea, note, tracker change, answer,
clarification, or short reply. Python validates the source quote, candidate
target, lifecycle, duplicate/conflict, and credential rules before any SQLite
write. The working principle is: understanding belongs to the model;
authorization stays in Python. It has no external calendar synchronization,
email, meetings, vector retrieval, or access to external services.

## Current status

Phases 1–3 of the original capture workflow are complete. Phase 2 passed 36
scripted tests and the frozen article, email, and idea live checks with retained
SQLite traces. Its evidence and claim boundary are in
[docs/phase-2-evidence.md](docs/phase-2-evidence.md). A locally retained Phase 3
acceptance run recorded 12/12; those local acceptance records are not committed
to the repository. Memory reviews support approval, edit, rejection, and
controlled conflict supersession. The original capture workflow and Memory
tooling remain in the codebase for regression checks, but are not exposed in
the default interface.

V2 Phase 1 adds versioned, additive SQLite migrations plus typed persistence
for messages, local calendar events, explicit ideas/notes, tool operations, and
pending date clarifications. V2 Phase 2 adds the persisted chat timeline,
conversation switching, Calendar and Ideas & Notes management pages, explicit
edit/delete/restore controls, and idempotent inline undo. V2 Phase 3 adds a
typed TurnRouter, one bounded repair, source-quote validation, supported
date/time resolution, date clarification, credential blocking, prepared local
operations, and restart-safe operation resume. Phases 4–5 add the deterministic
project tracker and a recent bounded conversation window (at most eight
messages and 12,000 message-content characters). Phase 6 adds the independent
`project_change_operations` audit log: unambiguous task completion or
question/blocker resolution can be committed, while decision/constraint
replacement or active conflicts remain reviewable before any mutation. Every
committed change is source-linked, visible in Chat, and has a guarded Undo.

Phase 7 adds bounded cross-conversation project retrieval: current-status
questions use active records plus completed milestones, while historical/why
questions may use superseded records and event history. Answer source cards are
persisted locally. The frozen scripted retrieval fixture reports Recall@6 1.0;
it is not evidence of general semantic retrieval or live-model answer quality.

The local deterministic/scripted suite passes 270 tests. This verifies the
database, retrieval, and model-write boundaries. A frozen live Phase 8 run
against `gpt-5.6-terra` passed its 12 runner assertions on 2026-09-08 and again
on 2026-09-12; its reproducibility metadata and limitations are recorded in
[phase-8-live-evidence.md](docs/phase-8-live-evidence.md). One run is
demonstration evidence, not a reliability or live-model-quality benchmark.

Runtime source safety is separated from offline output-quality evaluation in
[docs/gate-contract.md](docs/gate-contract.md). The fixed Phase 2 article,
email, and idea checks are recorded in
[docs/phase-2-live-fixtures.md](docs/phase-2-live-fixtures.md).

The completed original sequence was:

1. Phase 1: deterministic Streamlit, typed models, and SQLite persistence.
2. Phase 2: bounded Coordinator and Organizer flow with grounded output.
3. Phase 3: cross-conversation memory and a recorded 12/12 fixed acceptance run.

The optional original Phase 4 evaluation is not a prerequisite for V2. V2
Phases 1–11 are implemented; Phases 1–7 remain regression requirements, and
Phase 8 has both frozen scripted coverage and one recorded live acceptance
demonstration. Phase 9 adds deterministic, source-backed recall of
CaptureFlow's own active local Calendar and Ideas & Notes records across
conversations. Phase 9.1 optionally synthesizes a bounded answer from those
selected records when a model is configured, while preserving the Phase 9
deterministic reply when it is not. Neither path creates a local target record.

Phase 10 replaces the default chat path with a contextual router: one model call
per turn sees the bounded recent conversation and a small candidate list, so a
follow-up such as `把这场会议改到明天下午两点到三点` can update the event the
conversation created. It adds audited calendar updates and deletes with Undo,
tracker deadlines shown on the Calendar without becoming events, one action per
turn with a follow-up question for a second stated item, finished meetings kept
as notes, and a clarification when an update or delete names no event the
conversation identified. The frozen Phase 10 fixture passed 16 of 16 on
2026-09-09 and again after these changes; see
[phase-10-live-evidence.md](docs/phase-10-live-evidence.md) and
[phase-10.8-frozen-acceptance-rerun.md](docs/phase-10.8-frozen-acceptance-rerun.md).

Phase 11 moves date and time understanding to the model. Calendar actions and
tracker deadlines arrive as normalized `date`, `start_time`, `end_time`,
`duration_minutes`, `timezone`, and `recurring` fields, resolved against a
calendar table and the current local date and time supplied on every turn.
Python attaches the configured timezone, checks structure and plausibility, and
applies policy; it no longer interprets date or time wording. A stated foreign
timezone or a repeating schedule is clarified rather than stored, because
neither is supported. See
[phase-11-model-normalized-time-design.md](docs/phase-11-model-normalized-time-design.md)
and [phase-11-evidence.md](docs/phase-11-evidence.md).

## V2 local chat behavior

With `CAPTUREFLOW_API_KEY` and `CAPTUREFLOW_MODEL` configured, the default Chat
page sends each ordinary message to the contextual router. A deterministic
credential check runs first and blocks the turn without a model call. The model
then returns one `ConversationDecision`: `calendar_create`, `calendar_update`,
`calendar_delete`, `save_idea`, `save_note`, `tracker_change` (for an assigned
project conversation), `local_answer`, `project_answer`, `clarify`, or
`reply_only`. It sees only the bounded recent message window (at most eight
messages and 12,000 characters), at most three candidate events, saved items,
and project records, the reference date and time, and any pending
clarification. Python validates the decision, cites the source message, and
performs SQLite writes. The model never writes the database directly.

Known limits: a bare hour such as `明天十点` is resolved to 10:00 without a
question, and the reply states it; a message that only mentions an event by name
does not count as identifying it for an update; range questions such as
`我下周有什么安排？` are not answered.

A local-context question such as `我明天的会议要做什么？` or
`When is the calendar meeting?` searches only active Calendar and Note/Idea
records in the local workspace. Python validates cited IDs and renders exact
event date/time from SQLite. Any preparation suggestion is visibly labeled as an
inference, never a stored task. This lookup works in an unassigned conversation.
Calendar/Note recall is not external-calendar access, general chat memory, or
semantic search.

## Phase 8 acceptance and demo

The frozen acceptance contract is
[phase8_tracker_acceptance.json](tests/fixtures/phase8_tracker_acceptance.json).
Run its no-cost scripted version with:

```powershell
python -m unittest tests.test_phase_eight_acceptance -v
```

For a short manual demonstration, create a **CaptureFlow** project, assign a
conversation, then enter a task, a decision with rationale, and a blocker.
Complete the task in a follow-up, approve a decision replacement from its Chat
card, then create a second conversation under the same project and ask
`现在做到哪里了？` and `Why did we keep SQLite?`. The answer cards show the
stored records used as sources.

The frozen sequence and evidence boundaries are in
[phase-8-acceptance-protocol.md](docs/phase-8-acceptance-protocol.md). Further
live runs require explicit API-cost approval. Run against the configured
endpoint and preserve the ignored output directory:

```powershell
python scripts/run_phase8_live_acceptance.py --output-dir runs/phase8-live-YYYYMMDD
```

Each run costs model tokens. The recorded 2026-09-08 trace is local and ignored
by Git; the committed evidence summary intentionally excludes credentials and
raw model output.

The frozen Phase 10 contextual-router acceptance runs the same way:

```powershell
python scripts/run_phase10_live_acceptance.py --output-dir runs/phase10-live-YYYYMMDD
```

These explicit commands remain available without any model call:

```text
/event YYYY-MM-DD | title
/idea text
/note text
```

For example, `/event 2026-09-15 | Submit report` creates an all-day local
SQLite event. Without model configuration, ordinary text is persisted with a
visible configuration failure and creates no record.

### Optional diagnostics

Turn on **Diagnostics** in the Streamlit sidebar before submitting a message.
The matching assistant response then includes a collapsed session-only panel
with model-call count, model time, local validation/write time, repair usage,
outcome, and total submit time. Diagnostics are off by default and do not store
prompts, raw model output, credentials, or new database rows.

## Development environment

Use the existing Conda environment:

```powershell
conda activate llm-course
```

Install the project's declared dependencies into that environment, then run the
local shell:

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
committed. To run the deterministic unit tests:

```powershell
python -m unittest discover -s tests -v
```

## Model configuration

The V2 router uses a configured OpenAI-compatible chat-completions endpoint.
Set these environment variables in the shell that starts Streamlit:

```powershell
$env:CAPTUREFLOW_API_KEY = "..."
$env:CAPTUREFLOW_MODEL = "your-model-id"
# Optional for a compatible provider endpoint:
$env:CAPTUREFLOW_BASE_URL = "https://provider.example/v1"
# Optional; defaults to America/Vancouver:
$env:CAPTUREFLOW_TIMEZONE = "America/Vancouver"
# Optional; defaults to 30 seconds and must be positive:
$env:CAPTUREFLOW_MODEL_TIMEOUT_SECONDS = "30"
streamlit run app.py
```

The configured endpoint must support chat-completions JSON-object responses.
Without the required key and model variables, Chat saves and displays a typed
configuration failure; it does not fabricate a result.
