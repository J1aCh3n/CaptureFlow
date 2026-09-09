# CaptureFlow live system check

Run date: 2026-09-09  
Checked commit: `382b042717f3da1f2300fdda9ca2710dd0503512`  
Model: `gpt-5.6-terra`  
Endpoint family: Chat Completions  
Timezone: `America/Vancouver`  
Reference date for Phase 9.1: `2026-09-09`

This check used new isolated SQLite databases, projects, and conversations. It
did not read or modify the normal CaptureFlow application database. Test
prompts described fictional meetings and project data.

## Results

| Gate | Result | Evidence |
| --- | --- | --- |
| Full deterministic suite | 168/168 passed | `python -m unittest discover -s tests` |
| Frozen Phase 8 live tracker sequence | Failed one assertion | Explicit decision rationale was not persisted in the structured `rationale` field |
| Phase 9.1 live local-context sequence | 10/10 passed | Calendar, Idea, Note, tracker, cross-conversation recall, rejection, replay, and clarification checks |

The two live sequences made 18 model calls. Recorded model time was 30.067
seconds for the Phase 8 sequence and 23.320 seconds for the Phase 9.1 sequence.

## Passed live behavior

- Created a new project and separate planning and follow-up conversations.
- Created a local Calendar event with the expected title, description,
  `2026-09-11 10:30` start, and `11:30` end.
- Stored one Idea and one Note in the assigned project conversation.
- Created a source-grounded project task.
- Answered the meeting time, subject, and preparation question from another
  conversation in the same project using the Calendar event plus the saved
  Idea and Note.
- Answered a broad cross-conversation project-status question with visible
  project-record sources.
- Rejected a zero-relevance quantum-networking project query without a model
  call.
- Replayed the Calendar source message without another model call or a
  duplicate event.
- Created a pending clarification for an ambiguous Friday and committed the
  event after an exact-date follow-up.
- Passed the frozen Phase 8 task lifecycle, supersession review, blocker
  lifecycle, casual no-write, cross-project isolation, and restart-resume
  checks.

## Findings

### Structured decision rationale is not reliable

For `Decision: keep SQLite. Rationale: this is a local single-user MVP.`, the
model proposed the decision content `Keep SQLite for the local single-user
MVP.` but returned `rationale: null`. The record was committed and later
retrieval could describe the reason from the content/source quote, but the
dedicated rationale field was empty. This violates the frozen Phase 8
acceptance assertion.

### Chinese time text can bypass the temporal-authority validator

The Phase 9.1 answer correctly rendered the stored Calendar range locally, but
the model also emitted `会议安排在上午10:30开始。` as a summary point. The validator
is intended to reject model-generated Calendar date/time text, but its numeric
time expression uses word boundaries. There is no word boundary between the
Chinese character in `上午` and the digit in `10:30`, so this form was accepted.
The observed time was correct, but the deterministic Calendar authority is not
fully enforced for this format.

## Evidence boundary

This is one live run, not a reliability benchmark. The Streamlit UI is covered
by the deterministic suite; this run exercised the same service paths directly
rather than performing a manual browser interaction. The local trace and
isolated databases are intentionally ignored by Git:

- `artifacts/system-live-2026-09-09-phase8-382b042/trace.json`
- `artifacts/system-live-2026-09-09-phase9-382b042/trace.json`
