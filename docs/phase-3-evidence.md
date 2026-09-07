# Phase 3 evidence status

**Status:** implementation in progress. The bounded Memory Agent flow and
Memory page are implemented and tested, but the Phase 3 gate is not complete.

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
- The Coordinator may opt in with `inspect_memory`. Only then does the
  LangGraph Memory Agent subgraph retrieve up to five active memories, request
  proposals, and pass them to the policy-controlled commit service. A memory
  proposal failure records a typed omission and never writes memory.
- The Coordinator routing contract distinguishes durable user-specific
  preferences, decisions, facts, and ideas from actionable requests. Durable
  statements select `inspect_memory`; `extract_task` is reserved for requested,
  assigned, promised, or otherwise actionable next steps. Both actions are
  selected only when the source independently contains both kinds of content.
- The Streamlit Memory page supports active-memory search, edit, delete and
  restore; automatic-write undo; and pending-review approval, edit, conflict
  supersession, or rejection. These are explicit user actions rather than
  direct model writes.
- The default Memory view shows active items and pending reviews. Superseded
  and deleted versions are kept in a collapsed history view rather than mixed
  into the working set. Unchanged edits are no-ops; real edits preserve
  replacement lineage in the event log.
- Conflict approval resolves any edit lineage created while the review was
  pending before superseding the current active version. Legacy edits without
  lineage metadata fall back to matching the original source provenance.

## Verification

On 2026-09-07, the suite completed 51 tests successfully. The Phase 3 tests
cover opt-in retrieval followed by an automatic write, typed Memory Agent
failure without a write, and the Memory page display plus automatic-write
undo. A Coordinator prompt-contract test preserves the memory-versus-task
routing boundary. The policy tests cover automatic-write idempotency, blocked
secrets, conflict review and supersede, undo/delete/restore, bounded retrieval,
and no duplicate task after a repeated checkpointed commit.

Schema-invalid Coordinator responses, including an empty executable plan, now
use the same single bounded plan-repair pass as source-span validation errors.
The plan schema remains strict; no empty plan or fabricated fallback is accepted.

Memory lifecycle tests also cover unchanged-edit no-op behavior and approval of
a conflict whose original target was edited while the review remained pending.

## Still required for the Phase 3 gate

- Real checkpoint resume rather than only idempotent commit primitives.
- The frozen 12 Core acceptance examples and recorded protocol.
- Cross-conversation retrieval evidence and all remaining Phase 3 gate paths.
