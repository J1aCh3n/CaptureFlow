# Phase 10 Worktree: Contextual Conversation Router

## Goal

Replace top-level keyword routing with one structured, context-aware model
decision. The model proposes an intent and evidence; Python validates candidates
and commits only bounded local actions.

This phase must support ordinary natural-language calendar creation and a
follow-up correction such as "this meeting moves to Thursday, 14:00-15:00".
It must not add external calendars, email, Zoom, Responses API, or unbounded
agent tool use.

## Delivery order

### 0. Freeze the contract and baseline

- Read the current V2 models, router, storage, migrations, UI, and tests.
- Record the current deterministic-test count and an isolated live failure
  fixture for the two reported calendar inputs.
- Define acceptance examples for calendar create, calendar update, task,
  retrieval answer, and clarification.

Exit gate: no production change; the failure cases are executable tests before
the implementation begins.

### 1. Add contextual decision and provenance types

- Extend `models.py` with a top-level conversation intent, structured
  multi-message `EvidenceReference`, and calendar-update fields including a
  candidate `target_event_id`.
- Keep existing `TurnDecision` and explicit `/event`, `/idea`, `/note`
  contracts compatible while introducing the new contract beside them.
- Add schema tests for invalid intent/action combinations and invalid evidence.

Exit gate: typed schema rejects an update without one validated candidate target
or evidence that names a source message.

Commit: `feat: add contextual conversation decision schema`

### 2. Expose bounded router context and candidates

- Add `ProjectContextWindow.for_router()` in `context_service.py`; it includes
  stable message IDs, roles, and text without changing existing `for_model()`.
- Add deterministic candidate selectors for active calendar events, saved items,
  and project records.
- Bound every candidate list and include only workspace-safe active records.

Exit gate: router context has at most the established message/character budget,
and each candidate can be traced to a stable local ID.

Commit: `feat: expose bounded context for conversation routing`

### 3. Implement one-call contextual model proposal

- Extend `llm.py` with a structured `propose_conversation_decision()` call.
- Provide current message, bounded context, candidate IDs, reference date,
  timezone, and pending clarification state in its prompt.
- Require one of: calendar create/update, saved item, tracker change, local or
  project answer, clarify, reply only.
- Do not permit the model to supply arbitrary database IDs or act outside the
  supplied candidates.

Exit gate: fixture/model tests distinguish an event statement from a task, and
the model needs only one proposal call per normal natural-language turn.

Commit: `feat: propose contextual conversation decisions`

### 4. Make calendar updates safe and reversible

- Add an additive migration for update-operation audit data and multi-message
  calendar provenance. Do not rebuild existing V2 tables destructively.
- Extend storage to transactionally update title, description, start, end, and
  timezone with idempotency support.
- Add deterministic validation for target existence, active state, workspace
  scope, time ordering, and ambiguous target handling.
- Extend Undo: creation soft-deletes the new event; an update restores its
  pre-update snapshot.

Exit gate: restart, retry, invalid update, and undo tests pass without duplicate
events or writes outside the intended workspace.

Commit: `feat: support audited local calendar updates`

### 5. Add the unified conversation-router service

- Create `conversation_router_service.py` as the sole normal-language entry.
- It persists the current message, builds bounded context/candidates, requests
  one proposal, validates evidence and candidates, and dispatches to local,
  project, retrieval, or clarification executors.
- Split `project_service.py` so a validated tracker decision can be applied
  without a second model call.
- Retain `local_context_service.py` only for deterministic shortcuts, candidate
  selection, and no-model fallback—not final authority.

Exit gate: a project-assigned conversation no longer defaults to tracker merely
because a regex did not recognise a meeting.

Commit: `feat: route ordinary chat through contextual decisions`

### 6. Switch the Streamlit entry point and diagnostics

- Simplify `ui.py`: slash commands stay deterministic; all other natural
  language goes through the conversation-router service.
- Show intent, bounded-context count, candidate counts, model-call count, and
  final outcome in diagnostics without storing prompts, raw outputs, or keys.
- Preserve existing chat rendering and project assignment controls.

Exit gate: the two reported inputs reach calendar create/update rather than the
project tracker from the UI path.

Commit: `feat: use contextual routing in chat UI`

### 7. Complete deterministic and live acceptance

- Cover schema, context, LLM payload, storage migration, calendar update/undo,
  unified routing, project apply-without-second-call, UI dispatch, and fallback.
- Run the full deterministic suite.
- Run an isolated API fixture using the approved model/key only after the
  deterministic gates pass. Test create, correction, task distinction, local
  meeting question, ambiguity, replay/idempotency, and cross-conversation
  retrieval.
- Record only redacted timings, decisions, IDs, and outcomes in a committed
  evidence document.

Exit gate: all deterministic tests pass, all live assertions pass, and any
remaining natural-language limitations are explicitly documented.

Commit: `test: validate contextual calendar routing`
Commit: `docs: record phase 10 live acceptance`

## Dependency tree

```text
0 baseline
└── 1 decision schema
    └── 2 bounded context and candidates
        └── 3 structured model proposal
            ├── 4 safe calendar update and undo
            └── 5 unified service and project apply split
                └── 6 UI integration
                    └── 7 deterministic and live acceptance
```

## Explicit non-goals

- External calendar sync or provider APIs.
- Email, Zoom, transcription, or meeting-minutes automation.
- Responses API migration.
- Semantic/vector retrieval overhaul.
- More than one model call for ordinary routing/extraction.

