# Phase 7 retrieval evaluation

The frozen scripted set is
[`tests/fixtures/project_retrieval_cases.json`](../tests/fixtures/project_retrieval_cases.json).
It contains three queries and eight expected project-record IDs:

- a current-status query covering active tracker state plus a completed milestone;
- a historical SQLite-decision question;
- a historical superseded database-decision question.

The deterministic SQLite structured-filter plus lexical-ranking implementation
returns Recall@6 = 1.0 on this fixture, with no failed case. The corresponding
test reports the metric and failing case names.

This is a frozen, synthetic regression check for project isolation, lifecycle
filtering, deterministic ranking, and the current retrieval policy. It is not
evidence of live-model answer quality, general semantic retrieval quality, or
production-scale long-term memory.
