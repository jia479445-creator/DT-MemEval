# Dataset card

## Purpose and composition

DT-MemEval evaluates the ability of conversational memory systems to distinguish factual time from observation time. The released English-language dataset contains 4 domain sequences, 519 synthetic sessions and 2,000 questions. Each of 8 tasks has 250 questions; each of 4 dimensions has 500. Domain sequence identifiers are `countries_byGDP`, `athletes_byPayment`, `companies_byRevenue`, and `organizations`. These are four trajectories, not 519 independent participants.

The release provides a fixed evaluation set. It does not define train/development splits. Reporting per-task and per-domain results is recommended. Sessions are synthetic and use fictionalized names; they are not records of real user conversations.

## Construction and supervision

Historical source intervals are fictionalized by replacing entity names while retaining interval structure and timestamp precision. Controlled scenes verbalize scheduled endpoint reports; start and end reports can occur in different sessions. Independent model extraction and mechanical checks align reports to text. QA candidates, reference answers, historical scopes and minimal evidence support sets are computed offline. Seed 42, balanced task quotas, condition strata and domain round-robin sampling determine the released questions.

## Intended uses

Research on temporal reasoning, long-term memory, retrieval, evidence attribution and knowledge-state reconstruction. The dataset is not intended to measure general conversational realism or factual knowledge of the real entities underlying the source intervals.

## Limitations

- Synthetic and controlled prose may differ from human conversation.
- Four correlated domain streams limit conclusions about generalization.
- Fictional names do not prove absence of pretraining contamination; closed-book controls are required.
- Model auditing and exact text checks do not guarantee semantic correctness.
- Coarse date precision limits which orders and states can be established.
- Ambiguous repeated records are excluded from corresponding question candidates.
- No baseline scores or human-calibrated judge reliability are released.
- Early-model history is maintainer-confirmed; retained session provenance labels later records as GPT-5.4-mini.

## Access and supervision separation

The bundle publicly contains evaluator supervision. Keep `evaluation`, virtual facts and audit records out of model inputs. For historical questions, do not use reports beyond the specified cutoff to establish the historical answer.

## Provenance and rights

Early dialogue generation used GPT-4o-mini according to project history; candidate naming used GPT-5.4-mini. Retained records label generation and auditing as GPT-5.4-mini. Original sentence-level authorship is not independently traceable. QA and reference answers are computed offline without LLM calls.

No code or dataset license is assigned. Upstream licensing and complete source attribution remain unverified; fictionalization does not resolve those obligations.
