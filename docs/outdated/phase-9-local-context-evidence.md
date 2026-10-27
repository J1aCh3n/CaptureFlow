# Phase 9 local-context evidence

Phase 9 added a deterministic, read-only answer path for CaptureFlow's active
local Calendar and Ideas & Notes records. The contextual router now reaches it
when the model chooses `local_answer`; the lookup and the reply make no further
model call.

The frozen case is
[`tests/fixtures/phase9_local_context_cases.json`](../tests/fixtures/phase9_local_context_cases.json).
It creates a next-day local meeting and a meeting Note in one conversation,
then asks `我明天的会议要做什么？` from a different, unassigned conversation.
The accepted response contains the saved event date and Note, renders both
source cards, and says that no explicit personal preparation task was stored.

The selector reads only active CalendarEvent and SavedItem rows in the local
workspace, plus active records of the same Project for an assigned conversation.
A question that names one day — `今天`, `明天`, `后天`, `大后天`, their English
forms, or a full ISO or Chinese date — selects events on that day
(`date_tool.asked_date`). Otherwise text selection uses English token and CJK
character overlap. Deleted items are excluded. If no local record matches, it
returns an explicit unknown. Multiple candidates are listed rather than silently
choosing one.

Migration `007` adds `local_context_answer_sources`, whose rows carry a source
type, ID, and bounded rank, so Calendar/Note/Idea source cards survive UI reruns.

Phase 9.1 once added an optional model synthesis over the selected records. The
contextual router never enabled it, and it was removed in the
[redundancy cleanup](redundancy-cleanup.md).

This does not provide external-calendar access, arbitrary historical-chat
search, a vector index, a range question such as `我下周有什么安排？`, or a
semantic-retrieval quality claim.
