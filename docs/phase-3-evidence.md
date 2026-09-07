# Phase 3 evidence status

**Status:** implementation in progress. The persistent memory-policy foundation
is tested, but the Phase 3 gate is not complete.

## Implemented in this increment

- Typed memory proposals remain untrusted model output; only deterministic
  policy code can classify and persist them.
- Lexical retrieval is limited to five active memories in the default
  workspace; deleted and superseded memories are excluded.
- Explicit, first-person, high-confidence low-risk proposals may be automatic;
  decisions, ideas, inferred/low-confidence proposals, and conflicts require
  review; apparent credentials or missing source evidence are blocked.
- Auto writes record activity and support undo, delete, and restore. Reviewed
  conflicts may supersede the prior active memory only after user approval.
- Stable IDs make automatic-memory writes, pending reviews, and task commits
  idempotent across a repeated checkpointed commit attempt.

## Verification

On 2026-09-07, the suite completed 42 tests successfully. The new memory
tests cover automatic write idempotency, blocked secrets, conflict review and
supersede, undo/delete/restore, bounded retrieval, and no duplicate task after
a repeated checkpointed commit.

## Still required for the Phase 3 gate

- Memory Agent invocation from the bounded graph, including opt-in retrieval.
- A functional Memory page and stateful confirm/edit/reject controls.
- Real checkpoint resume rather than only idempotent commit primitives.
- The frozen 12 Core acceptance examples and recorded protocol.
- Cross-conversation retrieval evidence and all remaining Phase 3 gate paths.
