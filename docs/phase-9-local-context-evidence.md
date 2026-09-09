# Phase 9 local-context evidence

Phase 9 adds a deterministic, read-only answer path for CaptureFlow's active
local Calendar and Ideas & Notes records. It remains available without
`CAPTUREFLOW_API_KEY` or `CAPTUREFLOW_MODEL` and then has zero model calls.
Phase 9.1 may add one bounded model synthesis after this deterministic selector
when configuration exists; model failure or invalid output falls back to the
Phase 9 answer.

The frozen case is
[`tests/fixtures/phase9_local_context_cases.json`](../tests/fixtures/phase9_local_context_cases.json).
It creates a next-day local meeting and a meeting Note in one conversation,
then asks `我明天的会议要做什么？` from a different, unassigned conversation.
The accepted response contains the saved event date and Note, renders both
source cards, and says that no explicit personal preparation task was stored.

The selector reads only active CalendarEvent and SavedItem rows in the local
workspace. Date selection uses the existing narrow deterministic date tool;
text selection uses exact English token and CJK character overlap. Deleted
items are excluded. If no local record matches, it returns an explicit unknown
without a model call. Multiple candidates are listed rather than silently
choosing one.

Migration `007` adds `local_context_answer_sources`, whose rows carry a source
type, ID, and bounded rank. This preserves Calendar/Note/Idea source cards
across UI reruns without changing the Phase 7 project-record source table.

Phase 9.1 sends at most three events and three saved items, including typed IDs,
source quotes, the submit reference date, timezone, and resolved target date.
Python validates every selected/cited source and renders authoritative Calendar
time from SQLite. Suggestions are marked as inferences and never become tasks.

This does not provide external-calendar access, arbitrary historical-chat
search, a vector index, or a semantic-retrieval quality claim.
