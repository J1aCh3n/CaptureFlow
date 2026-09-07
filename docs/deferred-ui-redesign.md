# Deferred chat-agent UI redesign

**Status:** the presentation redesign is deferred until Phase 3 has passed its
functional gate.

The current Streamlit page is a functional Phase 1/2 shell: it exposes saved
captures, runs, grounded results, and errors for development. It is not the
intended end-user chat-agent experience and should not be treated as a final
product-quality interface.

## Trigger and boundary

Start a dedicated UI redesign only after Phase 3 is functionally complete and
its gate has passed. The redesign is separate from the Phase 3 functional
gate, so visual polish must not hide capability limits or change the approved
phase scope.

This deferral applies to presentation, not to required Phase 3 interaction.
Before the Phase 3 gate can pass, the existing shell must provide a stateful
path to reply to, confirm, edit, or reject a pending review item and apply the
result to its originating capture or task. Creating another capture must not
be treated as a reply.

## Design goals

- Present a conversation as a clear chat-like timeline, rather than as a
  collection of database records.
- Keep a compact, visible distinction between user captures, processing state,
  validated output, source evidence, and items needing user confirmation.
- Make task candidates and their next actions easy to scan without surfacing
  raw run traces as the primary interface.
- Retain run history, plan details, and model errors as collapsible diagnostic
  material for development and debugging.
- Preserve the existing capability boundary: the UI must not imply web search,
  calendar writes, or other actions that the active phase cannot perform.

## Completion criteria for the redesign

A user should be able to start or continue a conversation, enter a capture,
understand what the system extracted and why, and see the system's limits
without reading developer-oriented database or workflow terminology. Pending
review resolution itself is a functional Phase 3 prerequisite, not a feature
that waits for this post-gate redesign.
