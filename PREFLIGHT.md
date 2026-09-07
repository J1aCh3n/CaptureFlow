# CaptureFlow Preflight

> Status: Phases 1 and 2 are implemented and their gate evidence is recorded in
> [`docs/phase-1-evidence.md`](docs/phase-1-evidence.md) and
> [`docs/phase-2-evidence.md`](docs/phase-2-evidence.md). No Phase 3 or later
> application behavior, agent, evaluation result, or deployment described below
> should be treated as implemented until its corresponding phase gate has passed
> with saved evidence.

## 1. Purpose

CaptureFlow is a small, single-user AI capture inbox. A user pastes one piece of
plain text, such as an article excerpt, an email, or an idea. The system creates
a bounded execution plan and decides whether the content should be:

- summarized;
- converted into one or more actionable tasks;
- proposed as long-term memory for later retrieval;
- processed by any valid combination of the three.

The product must remain small enough for one developer to implement, evaluate,
and deliver. Its value is not the number of supported content types. Its value
is reliable routing, grounded outputs, controlled memory, and useful recall
across conversations.

## 2. First-version user story

Given one pasted text item, the user should be able to:

1. see the plan chosen by the system;
2. receive summary, task, and memory cards only when relevant;
3. inspect the source evidence for extracted claims and tasks;
4. accept, edit, reject, or undo proposed changes;
5. continue the conversation about the captured item;
6. close and restart the application without losing accepted conversations,
   tasks, or memories;
7. ask a later conversation to retrieve an earlier, relevant memory.

Example behavior:

| Input | Likely plan |
| --- | --- |
| Article excerpt | Summarize; optionally propose durable ideas |
| Email containing a request and deadline | Summarize; extract tasks; resolve relative dates |
| Personal project idea | Organize the idea; optionally extract a next action; propose an idea memory |

These are examples, not hard-coded content-type rules. The Coordinator must
produce a validated plan for the actual input.

## 3. Scope lock

### Included in the Core MVP (Phases 1-3)

- One local user and one default workspace.
- Pasted plain text only, with a default limit of 20,000 Unicode characters.
- Streamlit UI with two pages: `Inbox` and `Memory`.
- Multi-turn chat about the current captured item.
- Bounded planning over three actions: summarize, extract tasks, inspect memory.
- One Coordinator Agent, one Organizer Agent, and one Memory Agent.
- One lightweight grounding/reflection node.
- At most one Organizer revision per capture run.
- Three model-selectable, typed tools:
  `resolve_relative_date`, `search_memory`, and `list_open_tasks`.
- SQLite persistence for local delivery.
- Visible, reversible low-risk automatic memory writes.
- Human review for uncertain, conflicting, or consequential memories.
- Deterministic enforcement of state transitions, retry limits, memory policy,
  and database commits.
- Twelve fixed, labeled acceptance examples for regression checks: four
  article-style, four email-style, and four idea-style inputs.
- Documented local setup and execution.

### Explicit non-goals for v0

- PDF, image, audio, video, OCR, or speech transcription.
- Gmail, Outlook, Notion, Slack, Calendar, browser, or web-search integration.
- Sending messages, scheduling events, or modifying external systems.
- Multiple users, organizations, role-based access, billing, or subscriptions.
- Mobile applications or browser extensions.
- Vector databases or semantic embedding retrieval.
- Arbitrary code or shell execution.
- General autonomous research or open-ended task execution.
- More than the three declared processing actions.
- More than one reflection retry.
- Claims of production readiness, general planning ability, or multi-agent
  superiority without matched evidence.
- Docker packaging, public deployment, a larger evaluation set, and comparative
  ablations. These belong to optional Phase 4 portfolio hardening.

Any proposal that adds one of these items requires an explicit post-v0 scope
decision. It must not be introduced as incidental implementation work.

## 4. Architecture boundary

```text
Pasted text + current conversation
              |
              v
      deterministic input validation
              |
              v
      Coordinator / Planner Agent
              |
              v
        validated bounded plan
              |
       +------+-------+
       |              |
       v              v
 Organizer Agent   Memory Agent
 summary/tasks     retrieval/candidates/conflicts
       |              |
       v              |
 grounding checks     memory risk policy
       |              |
  pass | revise once  | auto/review/blocked
       +------+-------+
              |
              v
   deterministic result aggregation
              |
              v
     user-visible cards and chat
              |
              v
 deterministic commit / review / undo
```

The LangGraph orchestrator is not an agent. Normal Python code owns routing,
parallel dispatch, budgets, retries, checkpointing, approval interrupts,
idempotent commits, and terminal failure states.

The first version implements **bounded task planning**, not general autonomous
planning. The action vocabulary is fixed, even when the plan contains multiple
steps, dependencies, source spans, and conditions.

## 5. Component responsibilities

| Component | Responsibility | Context boundary | Capabilities |
| --- | --- | --- | --- |
| Coordinator Agent | Converse with the user and turn one capture into a bounded, conditional plan | Current input, current chat, compact worker results; not the complete memory database | Produce validated plans and final conversational responses |
| Organizer Agent | Produce grounded summaries and task candidates | Source text, relevant plan steps, date-tool results; no unrestricted memory access | Summarize, extract tasks, call date resolution when needed |
| Memory Agent | Retrieve relevant memories and propose new, merged, or conflicting memories | Source text, memory policy, bounded retrieved items; no private Organizer scratch context | Search memory, classify memory risk, return candidates |
| Grounding Critic | Check whether Organizer outputs are supported by the source | Source text and structured Organizer output only | Return `pass` or structured revision feedback |
| Orchestrator | Execute the plan and enforce all limits and transitions | Validated application state | Route, retry, checkpoint, aggregate, interrupt, commit |
| Persistence layer | Store conversations, captures, tasks, memories, and checkpoints | SQLite only | Typed read/write operations; no model judgment |

Each agent must have a distinct prompt, structured input/output contract,
context boundary, and tool allowlist. Merely assigning different role names to
the same unrestricted model loop does not qualify as multi-agent design.

The Grounding Critic is an LLM-backed reflection node, not a fourth autonomous
agent. It has no tools, no independent conversation, and no write access.

## 6. Planning contract

The Coordinator must return a schema-validated plan. An illustrative contract
is:

```python
class StepCondition(BaseModel):
    source_step_id: str
    predicate: Literal[
        "has_tasks",
        "has_memory_candidates",
        "has_conflict",
    ]
    expected: bool = True

class PlanStep(BaseModel):
    id: str
    action: Literal["summarize", "extract_task", "inspect_memory"]
    source_span: str
    depends_on: list[str] = Field(default_factory=list)
    run_if: StepCondition | None = None
    reason: str

class CapturePlan(BaseModel):
    steps: list[PlanStep]
    user_clarification_required: bool = False
    clarification_question: str | None = None
```

Planning rules:

- A simple input may produce a one-step plan.
- A multi-topic input may produce multiple steps over different source spans.
- Dependencies must refer only to step IDs in the same plan.
- Cyclic dependencies, unknown actions, and empty plans are invalid.
- `run_if` may reference only a validated boolean result exposed by a declared
  dependency. The Coordinator selects from the allowed predicates; the
  deterministic orchestrator evaluates the predicate from graph state.
- `run_if.source_step_id` must appear in the same step's `depends_on` list.
  `has_tasks` is valid only for an `extract_task` result;
  `has_memory_candidates` and `has_conflict` are valid only for an
  `inspect_memory` result. The orchestrator rejects incompatible predicates.
- Free-text execution conditions are invalid. Natural-language explanations
  belong in `reason` and never control routing.
- `depends_on` expresses dependencies between plan steps only. Tool calls are
  internal to the agent executing a step and are not plan nodes. For example,
  the Organizer may call `resolve_relative_date` while executing
  `extract_task`; the plan does not create a separate date-resolution step.
- A valid cross-step dependency is an `inspect_memory` step that runs only when
  a preceding `extract_task` step reports `has_tasks=True`.
- A memory commit is never a plan action. The plan may inspect or propose
  memory; deterministic policy controls the commit.
- If required information is genuinely missing, the plan may ask one concise
  clarification question instead of guessing.

The orchestrator validates and topologically orders the plan before executing
it. Before a conditional step runs, it deterministically reads the referenced
validated result and compares it with `expected`. It does not ask an LLM to
reinterpret the condition. Invalid model output becomes an explicit run failure
or one schema-repair attempt; it must not silently fall back to a fabricated
default plan.

## 7. State contract

The initial state model should reserve the following fields:

```python
class InboxState(TypedDict):
    run_id: str
    thread_id: str
    capture_id: str
    source_text: str
    plan: list[PlanStep]
    current_step_ids: list[str]
    tool_results: list[dict]
    organizer_drafts: dict
    grounding_report: dict
    retry_count: int
    retrieved_memories: list[dict]
    memory_candidates: list[dict]
    pending_approval: list[dict]
    committed_item_ids: list[str]
    errors: list[dict]
    final_response: str
```

This is a design contract, not an implementation claim. The final Python model
may split nested values into stricter Pydantic models, but it must preserve the
same responsibility boundaries.

State rules:

- Agent output never mutates the database directly.
- Every agent response is validated before it enters state.
- Retry count and tool budgets are deterministic.
- Commit operations use stable IDs and must be idempotent after checkpoint
  resume.
- Private model reasoning is not persisted. Only structured decisions,
  evidence, tool events, errors, and user-visible outputs are stored.

## 8. Tool contracts

### `resolve_relative_date`

Purpose: convert expressions such as "next Wednesday" into a candidate calendar
date using an explicit reference date and timezone.

Required inputs:

- original expression;
- reference date;
- configured user timezone.

Required output:

- normalized date or a typed ambiguity result;
- original expression;
- reference date and timezone used;
- confidence and explanation suitable for display.

The tool must not silently choose between genuinely ambiguous dates.

### `search_memory`

Purpose: retrieve a bounded set of prior memory items using lexical search.

Required controls:

- default workspace scope only;
- allowed memory types;
- result limit, initially capped at five;
- exclusion of deleted and superseded entries unless explicitly requested;
- source and status returned with every match.

SQLite storage alone is not evidence of Agent tool use. This counts as Agent
tool use only when the Memory Agent decides whether and how to invoke the typed
tool.

### `list_open_tasks`

Purpose: retrieve existing incomplete tasks when the current capture may repeat,
update, or conflict with an earlier commitment.

The tool is read-only. Task and memory writes are performed only by the
deterministic commit node after policy and any required user review.

## 9. Grounding and reflection

Organizer outputs must carry source evidence. An illustrative task contract is:

```python
class TaskCandidate(BaseModel):
    text: str
    source_quote: str
    due_expression: str | None = None
    resolved_due_date: date | None = None
    confidence: float
```

Summary output should be a small list of claims, each linked to one or more
source quotes. The first version should prefer omission over unsupported
completion.

Runtime grounding has two layers:

1. **Deterministic validation:** required quotes must occur in the submitted
   source after documented whitespace normalization.
2. **Semantic Critic:** the Critic checks whether each paraphrased claim or task
   is entailed by its cited source span without adding facts or changing its
   meaning.

The runtime Critic evaluates source support only. Summary usefulness, depth,
style, brevity, and whether a summary combines multiple sentences are offline
quality criteria, not reasons to reject grounded content during a run.

If the Critic rejects an output:

1. structured feedback is returned to the Organizer;
2. the Organizer may revise once;
3. the revised output is checked again;
4. specifically rejected items are omitted and surfaced as warnings;
5. unrejected grounded items may complete as `completed_with_omissions`; if no
   supported item remains, the run is `grounding_failed`.

There is no third attempt and no silent acceptance after retry exhaustion.
Grounded output that contains an ambiguous date or explicit confirmation item
is `needs_review`, distinct from both completion and grounding failure. The
full runtime and evaluation separation is in
[`docs/gate-contract.md`](docs/gate-contract.md).

The runtime Critic is not the evaluation oracle. Offline evaluation must use
independently labeled expected actions and source spans. A model must not be
treated as accurate merely because another invocation of the same model marked
its output as correct.

## 10. Memory model and autonomy policy

Conversation state and long-term memory are separate mechanisms:

- **Conversation/checkpoint state:** one thread's messages and graph state,
  persisted so the same conversation can resume.
- **Long-term memory:** selected information available to later conversations
  in the default workspace.
- **Source capture:** the submitted text from which a task, summary, or memory
  was derived.

An illustrative memory record is:

```python
class MemoryItem(BaseModel):
    id: str
    memory_type: Literal["preference", "decision", "idea", "fact"]
    content: str
    source_capture_id: str
    source_quote: str
    write_mode: Literal["auto", "reviewed"]
    status: Literal["active", "superseded", "deleted"]
    confidence: float
    created_at: datetime
    updated_at: datetime
```

Memory writes use three risk levels:

| Level | Criteria | Behavior |
| --- | --- | --- |
| `auto` | Explicit, low-risk, stable first-person fact or preference; direct source evidence; no conflict | Commit automatically, show an activity notification, and provide undo |
| `review` | Inferred preference, consequential decision, commitment, identity-related fact, low confidence, or conflict | Require explicit user confirmation or edit before commit |
| `blocked` | Credential, secret, authentication token, or prohibited sensitive inference | Do not create a derived long-term memory; warn the user |

Additional rules:

- Auto writes are visible and reversible; they are not hidden from the user.
- A conflicting candidate never overwrites an active memory automatically.
- Superseding a memory preserves the old record and provenance.
- The Memory page supports view, edit, delete, undo, and source inspection.
- Derived memories must be compact. They must not duplicate the full capture.
- Retrieval is bounded. The complete memory database is never inserted into a
  prompt.
- The UI must disclose that submitted content is sent to the configured model
  provider during live use.
- Secrets must be redacted from logs and run artifacts.

## 11. Persistence model

SQLite is sufficient for v0 local delivery. At minimum, persistence should
separate:

- conversations and messages;
- LangGraph checkpoints;
- source captures;
- task items;
- memory items;
- run and tool events.

The default workspace ID must still be stored explicitly so a future migration
does not require reinterpreting unscoped rows. Multi-user isolation is not
implemented or claimed in v0.

The application must survive a normal process restart. Streamlit session state
alone does not satisfy persistence. Database schema initialization and migration
must be deterministic and testable.

## 12. User interface

### Inbox page

- Conversation selector and new-conversation action.
- Chat history and text input.
- Visible execution status without exposing private chain-of-thought.
- Plan card showing selected actions, dependencies, and short reasons.
- Summary, task, memory, warning, and tool-result cards.
- Accept, edit, reject, and undo controls where applicable.

### Memory page

- Search and type/status filters.
- Memory content, write mode, confidence, timestamps, and source link.
- Edit, delete, restore, and supersede actions.
- Visible activity history for automatic writes and undo operations.

The first version does not need a dashboard, analytics page, settings portal,
or administration UI. Timezone may be one local configuration value.

## 13. Budgets and failure behavior

Default capture-run limits:

- maximum six model calls;
- maximum one Organizer revision;
- maximum six tool calls;
- maximum five retrieved memories;
- maximum 20,000 input characters;
- explicit per-call and total-run timeouts.

Expected failures must become typed visible outcomes, including:

- invalid plan schema;
- plan dependency cycle;
- model/API failure;
- tool ambiguity or timeout;
- grounding failure after revision;
- database or checkpoint failure;
- blocked memory candidate.

The application must never replace a configured model failure with fabricated
local content that appears model-generated.

## 14. Verification strategy

### Unit tests

Test deterministic behavior without a live API key:

- plan schema validation and dependency ordering;
- invalid or cyclic plans;
- state routing and iteration limits;
- exact source-quote grounding;
- date-resolution rules and ambiguity;
- memory risk-level policy;
- conflict, supersede, delete, restore, and undo behavior;
- bounded memory retrieval;
- checkpoint resume and idempotent commit;
- secret redaction;
- database initialization and restart persistence.

### Scripted-model integration tests

Cover complete graph routes reproducibly:

- summary only;
- task extraction only;
- memory proposal only;
- all three actions;
- multi-topic plan with dependencies;
- Organizer output accepted by the Critic;
- Critic rejection followed by successful revision;
- failure after revision exhaustion;
- automatic low-risk memory write and undo;
- review-required memory approval and rejection;
- blocked secret candidate;
- conflict with an existing memory;
- date-tool ambiguity;
- API/tool/database failure;
- pause and resume without duplicate writes.

Scripted tests establish orchestration and policy behavior, not live-model
planning, grounding, or memory quality.

### Core acceptance examples

Freeze 12 small inputs before completing Phase 3:

- 4 article-style inputs;
- 4 email-style inputs;
- 4 idea-style inputs.

The set should include simple, multi-topic, ambiguous, conflicting, and
sensitive examples. Each case should label:

- expected action set;
- acceptable plan dependencies;
- expected tasks and source spans;
- expected memory action: auto, review, blocked, or none;
- relevant prior-memory IDs for cross-session cases.

These are acceptance and regression examples, not a public benchmark dataset.
They require no fixture repositories, hidden evaluator tests, multi-attempt
protocol, or configuration leaderboard. Store each example as one input plus
its human-authored expected structured result.

Core checks:

- expected versus actual action set;
- expected tasks and source grounding;
- expected memory decision: auto, review, blocked, or none;
- relevant-memory retrieval for cross-conversation examples;
- zero tolerated blocked-to-write violations.

The Core MVP may report these 12 cases individually and as simple counts. It
must describe them as a small development set and must not use them to claim
general model or architecture performance.

### Optional portfolio evaluation

Phase 4 may expand the set from 12 to 30 inputs and generate:

- action-routing exact-set accuracy and macro F1;
- task precision, recall, and grounded-task rate;
- unsupported summary-claim rate;
- Critic false-accept and false-reject rates;
- memory-policy accuracy and retrieval hit rate;
- average model calls, latency, and estimated cost per capture.

Only Phase 4 runs the same fixed inputs with Critic disabled and enabled. This
matched comparison may measure whether reflection reduces unsupported outputs.
It does not establish general superiority over other agent architectures.

## 15. Implementation phases and gates

### Phase 0: Preflight

Deliverable: this document is reviewed and scope is frozen.

Gate:

- included and excluded scope is explicit;
- agent, node, service, tool, and persistence boundaries are distinguishable;
- no implementation or performance claim is made.

### Phase 1: Deterministic vertical slice

Deliverable: local Streamlit shell, typed domain models, SQLite storage, and one
hand-authored capture that can be saved, reopened, and deleted without an LLM.

Gate:

- restart persistence works;
- database operations are idempotent;
- deterministic unit tests pass;
- no later-phase agent behavior is simulated as real output.

### Phase 2: Planned Organizer flow

Deliverable: Coordinator plan, Organizer output, typed date tool, grounding
checks, one-retry reflection, and result cards.

Gate:

- scripted tests cover valid, invalid, revise, exhaustion, and tool-failure
  routes;
- the frozen article, email, and idea fixtures each retain their human-authored
  expected action set and minimum acceptable grounded output;
- all attempts remain in run history, so a failure cannot be hidden by swapping
  in an easier input;
- unsupported outputs remain visible rather than being silently accepted.

The frozen fixture definitions and evidence-accounting rules are in
[`docs/phase-2-live-fixtures.md`](docs/phase-2-live-fixtures.md).

### Phase 3: Cross-conversation memory

Deliverable: Memory Agent, bounded retrieval, auto/review/blocked policy,
conflict handling, Memory page, checkpoint resume, and the 12 Core acceptance
examples. Passing Phase 3 completes the Core MVP.

Gate:

- a new conversation retrieves a relevant accepted memory;
- automatic writes are visible and undoable;
- review and blocked paths are enforced;
- checkpoint resume cannot duplicate a memory or task;
- all 12 acceptance examples run under one recorded protocol and retain both
  successes and failures;
- local setup and limitations are documented.

### Phase 4: Optional portfolio hardening

Phase 4 is not required to call the Core MVP complete. Start it only after the
Phase 3 gate passes and time permits.

Deliverable: expanded 30-input development set, generated metrics, Critic
ablation, optional Docker packaging or public deployment, and polished
demonstration material.

Gate:

- evaluation output is generated from saved run records;
- failures remain in the aggregate;
- any Docker or deployment path starts from documented clean-environment steps;
- no secret or API key is committed;
- README claims match current evidence.

No later phase should be scaffolded before the current phase gate passes.

## 16. Definition of done for the Core MVP

CaptureFlow's Core MVP is complete at the end of Phase 3 only when all of the
following are true:

1. A user can paste one supported text item and receive a validated visible
   plan.
2. Relevant summary, task, and memory paths execute without always invoking all
   agents.
3. Every accepted task and summary claim has inspectable source evidence.
4. Unsupported outputs are rejected or clearly omitted after at most one retry.
5. Relative dates are resolved through a typed tool or reported as ambiguous.
6. Conversation state survives a process restart.
7. A later conversation can retrieve a relevant long-term memory.
8. Auto, review, blocked, conflict, undo, and delete memory paths work.
9. Checkpoint resume does not duplicate side effects.
10. Unit and scripted integration tests pass without a live API key.
11. All 12 fixed acceptance examples run under one recorded protocol and retain
    both successes and failures.
12. A new user can run the project locally from documented setup steps.

Phase 4 portfolio hardening is complete only when its expanded evaluation and
any chosen packaging or deployment deliverables pass their own gate. Its
incomplete status does not reopen a completed Core MVP.

## 17. Evidence and claim boundaries

- A working UI does not prove durable memory.
- SQLite storage does not by itself prove Agent tool use.
- Separate prompts do not by themselves prove meaningful multi-agent design.
- Scripted-model tests prove declared graph routes and policies, not live-model
  judgment quality.
- A live demonstration proves that one run completed, not reliability.
- The 12-input Core set is a small collection of acceptance examples, not a
  benchmark. A later 30-input set remains a development evaluation, not a
  general benchmark.
- Critic implementation does not prove improvement; only the matched ablation
  can support a scoped reflection claim.
- Memory implementation does not prove useful recall; cross-session cases must
  measure it.
- Local single-user delivery is not a multi-user or production deployment.

Until the relevant gates pass, the defensible project description is:

> A preflight design for a bounded LangGraph capture assistant with planned
> routing, source-grounded reflection, typed tools, and risk-tiered persistent
> memory.

After implementation, wording must be updated only from actual saved evidence.
