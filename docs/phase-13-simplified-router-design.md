# Phase 13: A Simpler Contextual Router

**Status:** implemented, with acceptance protocol version 3 and five runs recorded
under Evidence. The decisions are recorded in the last section.

## Why

`ambiguous_correction` (`这个会议改到周四。`) has failed in many runs, each time for
a different reason:

- a date tool resolving the weekday;
- a pending update required to carry a value;
- `recurring: null`;
- `source_ids` on a write;
- evidence citing an assistant reply;
- a `clarify` without a pending action.

None of these was the model misunderstanding the message. Every one came from a
layer that exists to make up for the router's shape: one model call per turn that
must return one strict structured decision, from a few pre-selected candidates.

A tool-using agent handles the same message without these layers. It looks the
event up; if it cannot tell which one is meant, it asks in plain language; the next
turn continues from the transcript. CaptureFlow is small, and tool use is not
wanted. This design keeps one call per turn and removes the compensating layers
instead.

## Principle

Understanding belongs to the model; authorization stays in Python.

**Finding a target.** For editing or deleting something, the user's rule applies:

> If the model cannot find the specific event or record in the conversation, it
> must ask the user, then complete the action with the one the user names.

A target counts as found when either:

- the current conversation mentioned it, or
- the current message names the item itself, such as by its title.

An item that appeared only in an older, different conversation is not found. The
model asks.

**Source of truth.** Stored state is current. The transcript is used only to
understand what the user refers to. When the two disagree, for example an earlier
reply says 14:00 but the event is now at 15:00, the stored state wins. The prompt
states this rule.

**Who authorizes a write.** Only the current user message. An earlier request in
the transcript is never carried out again.

## What stays: the authorization boundary

Every rule here is one Python can check against stored data without interpreting
wording:

- **Credential guard.** A message that discloses a credential never reaches the
  model, neither in its own turn nor later in the transcript.
- **Candidate IDs only.** A write may target only an event or record ID the router
  supplied.
- **Who authorizes a write.** It is the current user message, never an assistant
  message.
- **Write safety.** Every write is an audited, idempotent, restart-safe operation
  with Undo.
- **Visible result.** Every write's reply names what changed and its new schedule,
  so a wrong target is visible and can be undone at once.
- **Time policy.** Python attaches the timezone and rejects implausible dates.
  Foreign timezones and repeating schedules are refused. An end time and a
  duration must agree.
- **Tracker rules.** Allowed lifecycle transitions, no duplicate active records,
  and replacing a decision or constraint waits for review.
- **Project answers.** Citations must be in the snapshot, and dates are rendered
  from SQLite.
- **Answers.** Times in `local_answer` come from SQLite.

## What changes

### A. A clarification is a plain question

**Today:**
- A clarification is stored state: a `pending_clarifications` row with the
  unexecuted action, a `missing_field`, and one allowed per conversation.
- The logic around it:
  - a newer question cancels an unanswered one;
  - a write of the matching kind resolves it;
  - `_ANSWERING_INTENTS` decides which writes count;
  - the model receives the pending action on the next turn;
  - the Chat page shows a cancel banner.
- A compound message's second item is stored the same way (`deferred_action`).

**Proposed:**
- **The model's question.** `clarify` is a question in `reply`, and nothing is
  stored. The schema drops `pending_action`, `deferred_action`, and
  `clarification_question`.
- **Python's policy questions** (timezone, recurrence, implausible date, conflicting
  end and duration, an end for an all-day event) are also plain replies.
- **A second item** in a compound message is asked about with `follow_up_question`,
  appended to the reply. Nothing is stored.
- **Continuing after the answer.** The next turn has the question and the answer in
  its recent turns (see D), and the model completes the action from the transcript.
  For example:

  | Turn | Speaker | Message | Result |
  | --- | --- | --- | --- |
  | 1 | User | 明天开评审会 | `clarify` → "几点开始？" |
  | 2 | User | 下午三点 | `calendar_create` for tomorrow 15:00, citing both user messages |

- **Ignoring a question** leaves nothing hanging, so no cancel button is needed.
- **Leftover structure is noise.** A key the schema no longer has, such as
  `evidence` or `pending_action`, is ignored. A payload the intent never reads is
  dropped, not rejected: a `clarify` that also carries an action writes nothing, and
  a follow-up question without a write is not shown.

**Removed:**
- Router: `_save_pending_question`, `_save_clarification`, `_unresolved_field`,
  `_replace_unanswered_question`, `_save_follow_up_clarification`, and
  `_ANSWERING_INTENTS`.
- The `pending_clarification` payload sent to the model.
- The Chat banner.
- The storage methods, once unused.

The table itself stays; see compatibility below.

### B. The model judges whether the target was found

**Today:** after the model chooses a target, Python decides whether the conversation
"identified" it: a message in the window created or changed the event, or the
current message shares a Chinese character pair with its title
(`_target_is_ungrounded`, `_distinctive_terms`). Python second-guesses
understanding with a wording heuristic, which the principle rules out.

**Proposed:** the prompt states the rule for finding a target, as defined under
Principle. The model chooses a target only when it is found, and asks otherwise.
Python keeps only the ID check. A wrong choice is caught by the reply that names the
event, plus Undo.

**Removed:** `_target_is_ungrounded`, `_distinctive_terms`, `_GENERIC_EVENT_TERMS`,
`list_calendar_event_message_ids`.

### C. Sources are message IDs, not copied quotes

**Today:**
- Every action carries one to three evidence entries, each an exact quote copied
  from a message.
- Every tracker change carries an exact `source_quote`, which must occur in the
  current message.
- A wrong quote fails the whole turn.

Copying quotes is a frequent source of structural failure.

**Proposed for Calendar events, ideas, and notes:**
- The model lists `source_message_ids`, IDs only: the user messages the item comes
  from, such as turn 1 and turn 2 in the example above.
- Python adds the current message if it is missing. It keeps only IDs of user
  messages in the context and drops an assistant or unknown ID; that is not a
  rejection, because the current message still authorizes the write.
- A source card stores the whole message text as its quote, which already satisfies
  the stored substring rule.
- **Descriptions.** The model may write a description in its own words, like the
  title. The source card still shows what the user said.

**Proposed for tracker changes.** The quote stays, because the Tracker shows it
prominently and a whole pasted text would be unreadable there. What changes is that
a wrong quote no longer fails the turn:
- A change may name `source_message_id`, the user message in the context its quote
  comes from. It defaults to the current message.
- If the quote occurs in that message, it is kept. If not, Python uses that
  message's whole text instead.
- The record's source is that message. The operation is still authorized by the
  current message.
- A stored operation without the field uses its operation's source message, as
  today, so existing operations still commit and undo.

This is what lets long input work: a task saved in the turn "继续记剩下的" quotes the
minutes pasted a turn earlier.

**Removed:**
- `EvidenceReference` on decisions.
- The evidence quote checks in `llm.py` and the router.
- `_current_evidence` and `_quoted_description`.
- The rule that a tracker quote must occur in the current message.

### D. Context in layers, within a budget

**Today:**
- The transcript is the last eight messages, up to 12,000 characters.
- Candidates are at most three per kind:
  - events nearest the reference date;
  - ideas and notes by keyword overlap;
  - active records by keyword overlap.

Asking a model to find a meeting among three guesses is why a correct question can
still lack its target.

**Proposed.** The model sees an index of everything that matters all the time, and
full text only where it is needed. Each layer has a budget and a rule for what
happens when the budget is exceeded:

| Layer | Contents | Form | When over budget |
| --- | --- | --- | --- |
| Fixed | System prompt, rules, reference date | As is | — |
| Current conversation | Recent turns | As is. An older long message becomes a one-line reference | Older turns are dropped. Later: a rolling summary |
| Structured state | Events, active records, ideas and notes | One line per item: ID, title, time, state | Filtered, then truncated to the bounds below |
| Document index | Long texts such as meeting minutes | One line per document: title and summary | Later: only the top k relevant documents |
| Document text | Long texts in chunks | As is | Later: fetched only when an answer needs it, in a second call |

A single user's data fits inside every budget, so in practice the model sees
everything. As data grows, the same structure compresses, then filters, then
retrieves. No layer changes shape.

#### Implemented in this phase

**Current conversation**
- Budget: 12,000 characters, estimated by character count because no Qwen
  tokenizer is available locally. The eight-message limit is removed.
- The last two turns (four messages) always keep their full text, so the model can
  keep working on something pasted a moment ago.
- An older message over 800 characters becomes a one-line reference: its ID, its
  length, and its first 80 characters.
- Older turns that do not fit are dropped.
- **A message blocked by the credential guard is left out of the transcript.**
  Today such a message is stored and shown to the model in later turns.

**Structured state**
- Each item is one compact line, for example:

  ```text
  event e-41 | 算法优化与代码审查会议 | 2026-09-17 14:00–15:00
  record r-12 | task | 排查延迟 | active | due 2026-09-20
  ```

- The filters and bounds:

  | Kind | Filter | Bound | Order when truncated |
  | --- | --- | --- | --- |
  | Events | The calendar table's window, 7 days back and 27 ahead. **Not filtered by project**: the calendar is shared, and an event created in another conversation must still be found | 15 | Nearest the reference date |
  | Active records | The conversation's project | 30 | Most recently updated |
  | Ideas and notes | The conversation's project, or the workspace if it has none | 20 | Most recent |

- **Named events.** When the current message names an event that did not make the
  15, Python adds up to 5 events whose titles share at least two Latin words or
  Chinese character pairs with it. Without this, the
  ask-then-complete loop breaks: the user names the meeting, and it is still not a
  candidate. This is retrieval, not authorization; Python never decides the target.
- **A truncated list is marked** in the prompt. The model then knows a missing item
  may exist and asks, instead of concluding that nothing matches.
- `local_answer` keeps its date and keyword lookup for a question whose items are
  not candidates, such as a date outside the window.

The later layers are listed under Later.

### E. Answer intents

`local_answer` and `project_answer` stay as Phase 12 defined them. Their
`source_ids` follow D. `project_answer` keeps its second call (decision 6).

## Long input

A pasted text such as meeting minutes is turned into long-term memory, not kept in
the context:

1. In the turn it is pasted, the model saves what it can with the existing intents
   (tasks and decisions as tracker changes, a meeting as an event) and says in its
   reply what is still unsaved.
2. The user continues, for example "继续记剩下的". The pasted text is still within the
   last two turns, so the model still sees it, and each new tracker record quotes
   the pasted message (see C).
3. Once the text leaves the last two turns, it appears only as a one-line reference.
   What was extracted stays visible through the structured state.

Nothing is stored beyond what already is: the pasted text is a message in
`messages`.

## Stored data and compatibility

- One migration, for delete confirmation (see below). Nothing else changes the
  schema.
- `pending_clarifications` keeps its rows. Existing pending questions are simply no
  longer shown or used.
- `calendar_event_sources` and `project_records.source_quote` keep their columns.
  New rows store the whole source message as the quote.
- Existing operations must still commit, resume, and undo. Calendar operations store
  their before and after events. Tracker operations re-validate their stored
  `ProjectChange`, which keeps `source_quote`. This is a test requirement for the
  implementation, not an assumption.

## Acceptance (protocol version 3)

Checks are behaviors a user sees, never internal state.

`ambiguous_correction` is replaced by two cases that test the principle itself. They
run last, in this order, as three submits:

1. **Found, so act** (`target_found_acts`). In the conversation that just asked
   about the meeting, send `这个会议改到下午三点。` Expect a committed update and the
   event starting at 15:00.
2. **Not found, so ask, then complete.**
   - `target_not_found_asks`: in a new conversation that never mentioned the
     meeting, send `这个会议改到周四。`. Expect a question, no write, and the event
     unchanged.
   - `target_named_completes`: then answer `算法优化与代码审查会议，改到 9 月 17 日。`
     Expect a committed update and the event on 2026-09-17.

Other cases keep their checks. A run has ten submits and ten effect checks, 20
records. Each submit also records its input and output tokens and model time, so
the effect of the wider context is measured. The counts come from the provider's
usage report; no prompt or output is kept.

## Evidence

The deterministic suite passes 187 tests.

Protocol version 3 ran five times on `qwen3.8-flash` at `3d86638`, from a clean
working tree, with ten model calls per run. No run reached the free quota, so no
run needed another model. Runs are `runs/phase13-v3-1` to `-5`.

| Case | Passed | Input tokens | Output tokens | Model seconds |
| --- | --- | --- | --- | --- |
| `calendar_create` | 5 of 5 | 4,146 | 383–611 | 7–12 |
| `calendar_update` | 5 of 5 | 4,340 | 572–1,222 | 8–18 |
| `tracker_change` | 5 of 5 | 4,498 | 407–744 | 7–12 |
| `meeting_question` | 5 of 5 | 4,637–4,638 | 272–1,667 | 6–27 |
| `replay_idempotency` | 5 of 5 | — | — | 0 |
| `cross_conversation_local_retrieval` | 5 of 5 | 4,232–4,233 | 289–1,045 | 7–24 |
| `cross_conversation_project_retrieval` | 5 of 5 | 5,267–5,275 | 408–1,214 | 8–23 |
| `target_found_acts` | 5 of 5 | 4,484–4,493 | 1,219–4,757 | 31–223 |
| `target_not_found_asks` | 1 of 5 | 4,210–4,216 | 586–1,018 | 14–22 |
| `target_named_completes` | 5 of 5 | 4,320–4,394 | 691–2,655 | 14–75 |

Records per run were 19, 20, 19, 19, and 19 of 20. Model time per run was
141–382 seconds. The project answer's token counts cover both of its calls.

- **Every case except one passed every run**, including the former failure modes:
  `tracker_change` and both answer cases held, and a named event was always
  completed.
- **`target_not_found_asks` failed four times, in two ways:**
  - Three times (runs 3–5) the model did not ask. In a conversation whose only
    message was `这个会议改到周四。`, it chose the one candidate meeting and moved
    it to Thursday, which was already that event's date, so the turn wrote nothing
    (`calendar_unchanged`). This is the wrong-target risk below, observed: the
    prompt's rule was not followed.
  - Once (run 1) the model chose `clarify` without putting its question in
    `reply`, and the turn failed validation. The trace keeps no model output, so
    where the question went is not known.
- **Cost of the layered context.** With this small database each routing call reads
  about 4,100–4,500 input tokens, close to the estimate of about 4,000. Latency is
  dominated by output: the slowest turn wrote 4,757 tokens, mostly reasoning.

How `target_not_found_asks` was addressed is decision 7.

### After decision 7, on `qwen3.5-flash`

Five runs at `00985e7`, as an explicit cross-model comparison: `qwen3.8-flash` had
too little free quota left. Runs are `runs/phase13-v3-named-q35-1` to `-5`. There is
no baseline on this model before the change, so the table shows the result after it
only.

| Case | Passed | Input tokens | Output tokens | Model seconds |
| --- | --- | --- | --- | --- |
| `calendar_create` | 5 of 5 | 4,110 | 498–581 | 4–5 |
| `calendar_update` | 5 of 5 | 4,304 | 611–1,283 | 4–9 |
| `tracker_change` | 4 of 5 | 4,462 | 426–5,191 | 4–32 |
| `meeting_question` | 4 of 5 | 4,569–4,618 | 533–827 | 4–7 |
| `replay_idempotency` | 5 of 5 | — | — | 0 |
| `cross_conversation_local_retrieval` | 4 of 5 | 4,142–4,197 | 478–776 | 4–9 |
| `cross_conversation_project_retrieval` | 4 of 5 | 5,023–5,195 | 4,793–8,235 | 36–50 |
| `target_found_acts` | 5 of 5 | 4,424–4,510 | 825–4,355 | 6–33 |
| `target_not_found_asks` | 4 of 5 | 4,126–4,181 | 539–1,124 | 4–7 |
| `target_named_completes` | 4 of 5 | 4,271–4,358 | 683–1,940 | 5–13 |

Records per run were 20, 19, 18, 19, and 18 of 20. Model time per run was 77–122
seconds, faster than `qwen3.8-flash`.

- **The target cases.** A demonstrative in a new conversation was asked about in
  four runs of five; once (run 4) the model still bound it to the meeting, which
  wrote nothing. The named event was completed in four runs; once (run 3) the model
  chose `clarify` without a question in `reply`, the same failure seen once on
  `qwen3.8-flash`.
- **The other failures are this model's, one each:** `tracker_change` asked a
  question instead of creating the task (run 5), so that run's project answer had
  no task to cite; `meeting_question` cited an ID outside the candidates and failed
  validation (run 2); and `cross_conversation_local_retrieval` answered with
  `reply_only` (run 3).


## Risks

- **A wrong target.** Without the Python heuristic, a model could bind "这个会议" to
  an event in a new conversation. Mitigations: the prompt rule, the reply naming the
  event, and Undo. A delete additionally waits for a confirmation click.
- **Missing sources.** Python can add only the current message to an item's sources.
  An earlier message the model fails to list cannot be recovered, so a source card
  may show only the current message.
- **Unextracted long input.** If the user moves on before everything in a pasted
  text is saved, the rest is no longer visible once the text leaves the last two
  turns. The document layers are not implemented in this phase.
- **An event beyond the bounds.** An event outside the calendar window, or not among
  the 15 and not matched by name, is not a candidate. The model asks, and the user
  may need to name it more precisely.
- **A question scrolling out of the context.** A question and its answer far apart
  are no longer linked. That is acceptable for an unanswered question.
- **Weaker source cards.** A Calendar, idea, or note card shows the whole message
  instead of the exact span. A tracker record falls back to the whole message only
  when its quote is wrong.
- **Model-written descriptions.** A description may no longer be the user's words.
  The source card beside it is.
- **The size of the change.** The router loses roughly a quarter of its code. The
  schema shrinks. Most scripted tests in `test_conversation_router_service.py` build
  decisions with `evidence` and must be rewritten, as must the candidate selection
  tests in `test_context_service.py`. The prompt changes substantially, so the
  acceptance must be re-run several times.

## Implementation plan

Three commits:

1. **Router, schema, and context:** A–D, the prompt rules, tracker quotes, delete
   confirmation with its migration, rewritten tests, and this document.
2. **Acceptance protocol version 3:** the two new cases replacing
   `ambiguous_correction`, and token and time records.
3. **Evidence:** five runs on `qwen3.8-flash`, with per-case results.

## Later

Not in this phase. Each fits the layers in D without changing their shape.

- **Rolling summary.** Older turns that exceed the conversation budget are
  summarized instead of dropped.
- **Document index.** Long texts get a title and a summary line; only the top k
  relevant ones are shown.
- **Document text on demand.** When an answer needs a document's text, a second call
  fetches the relevant chunks.
- **Extraction in one turn (Phase 14).** A pasted text's tasks, decisions, and events
  are all saved in the turn it is pasted. This needs several writes of different
  kinds in one turn, per-item validation, a rule for one invalid item among many,
  and undo for the whole batch. Today a turn writes one action or up to three
  tracker changes.

## Open decisions

1. **Description text.** Today a description is kept only if it is an exact quote
   of the user's words. Decided: the model may write it in its own words (C).
2. **Tracker source quotes.** Decided: keep the quote, taken from any user message
   in the context, with the whole message as the fallback instead of a failed turn
   (C). Today the quote must occur in the current message, so a task saved in the
   turn "继续记剩下的" could not quote minutes pasted a turn earlier.
3. **Delete confirmation.** Decided: yes. See Delete confirmation below.
4. **Candidate bounds.** The first draft proposed 30 events and 30 records.
   **ANS：** 30 events might be too much for testing, reduce to 15
   Decided: 15 events, 30 records, 20 ideas and notes, and an 800-character
   threshold for replacing an older message with a one-line reference.
5. **Replacing `ambiguous_correction`.** Decided: yes, with the two cases above.
6. **The second `project_answer` call.** The router now sees the project's active
   records, so it could write a project answer itself. Decided: keep the second
   call in this phase. Merging would add closed and superseded records to every
   turn and move the citation checks into the router. Revisit with the token and
   time records from protocol version 3.
7. **A demonstrative in a new conversation.** `target_not_found_asks` passed one run
   of five: the model usually bound `这个会议` to the only candidate meeting (see
   Evidence). The prompt said a target is found when the message is specific enough
   to match exactly one candidate, which the only candidate satisfies. Decided:
   change only that sentence, to a message that names the item itself, such as by
   its title. The existing sentence that a demonstrative in a conversation that
   never mentioned the item finds nothing stays as it is. Verified with five runs on
   `qwen3.5-flash`, because `qwen3.8-flash` has almost no free quota left.

## Delete confirmation

Python no longer checks that the conversation identified an event, so a delete waits
for the user instead of running at once. It reuses the review flow tracker changes
already have:

- A `calendar_delete` saves its operation with status `review` and replies naming
  the event and its time. The turn's outcome is `calendar_delete_review`.
- The Chat page shows **Delete** and **Keep** buttons under that reply. **Delete**
  commits the operation; **Keep** marks it failed with "rejected by user".
- Creates and updates still run at once, with Undo.
- An unconfirmed delete stays in review. It is not a pending clarification: the
  model is not told about it, and a later message does not resolve it. Replaying
  the same message does not commit it either.

This needs one migration (version 17): the calendar operations table's status check
did not allow `review`. The migration rebuilds the table with that value added.
Unlike earlier rebuilds, another table references this one, so foreign keys are
off while it runs; otherwise dropping the old table would delete the stored update
sources. Foreign keys are checked before the migration commits.
