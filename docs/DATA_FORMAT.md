# Data format

The canonical bundle is `data/llm_natural/06_standard.json`, with schema `6.1-full-history-interchange`.

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

Question `evaluation` contains reference answers, evidence and audit annotations. Evidence uses session timestamps and aligned source text. Character offsets refer to Python Unicode indices in session `content`, excluding metadata headers. Minimal support sets can encode alternative sufficient evidence combinations.

The virtual facts and `construction/` contents are construction/evaluation inputs only. They must not be ingested by the memory system. Gold supervision being publicly downloadable does not authorize benchmark models to consume it.

## Evaluation protocol

Build memory separately for each sequence in increasing `session_index` order. Retrieve and answer at trajectory end using the public prompts, respecting D1/D2 historical cutoffs. Save predictions before judging. Report answer accuracy, evidence accuracy and grounded accuracy (both correct), overall and per task/dimension, with denominators and judge errors counted separately. Accept alternative sufficient evidence sets within the allowed scope. Record the release hash, memory system, answer/judge models, prompts and retrieval budget. This repository does not implement a judge or supply baseline scores.
