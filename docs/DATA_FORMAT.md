# Data format

The canonical bundle is `data/llm_natural/standard.json`.

| Top-level field | Contents |
|---|---|
| `schema_version`, `dataset` | Format and dataset identifiers |
| `sessions` | Ordered session objects with domain and observation time |
| `questions` | Public question fields plus evaluator-only supervision |
| `sequences` | Domain trajectories, question IDs and scheduling |
| `metadata` | Shared system prompt, task taxonomy, sampling and temporal conventions |

## Sessions

`session_id`, `sequence_id`, `timestamp`, `session_index`, `content`, `memory_input`, `messages`. Ingest `memory_input` or `messages`. Observation timestamps are part of the memory input because observation-time tasks need them.

## Questions

`question_id`, `sequence_id`, `task_id`, `dimension`, `question_type`, `text`, `user_prompt`, `knowledge_scope` and scheduling fields. Retrieve using `text`; compose answer prompts with `metadata.system_prompt`, retrieved memory, `user_prompt` and `text`. Use `src.qa.prompts.build_messages` for the supplied prompt protocol.

Every question runs at `evaluation_at: trajectory_end`. This scheduling choice does not remove the historical cutoffs in `knowledge_scope`.

## Evaluator-only fields

Question `evaluation` contains reference answers, evidence and audit annotations. Evidence uses session timestamps and aligned source text.

The virtual facts and `construction/` contents are construction/evaluation inputs only. They must not be ingested by the memory system.

## Evaluation protocol

Build memory separately for each sequence in increasing `session_index` order. Retrieve and answer at trajectory end using the public prompts.