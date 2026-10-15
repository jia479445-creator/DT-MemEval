# Dataset card

## Purpose and composition

DT-MemEval evaluates the ability of conversational memory systems to distinguish factual time from observation time. The released English-language dataset contains 4 domain sequences, 519 synthetic sessions and 2,000 questions. Each of 8 tasks has 250 questions; each of 4 dimensions has 500. Domain sequence identifiers are `countries_byGDP`, `athletes_byPayment`, `companies_byRevenue`, and `organizations`.
The release provides a fixed evaluation set. Sessions are synthetic and use fictionalized names; they are not records of real user conversations.

## Construction and supervision

Historical source intervals are fictionalized by replacing entity names while retaining interval structure and timestamp precision. Controlled scenes verbalize scheduled endpoint reports; start and end reports can occur in different sessions. QA candidates, reference answers, historical scopes and minimal evidence support sets are computed offline. Seed 42, balanced task quotas, condition strata and domain round-robin sampling determine the released questions.

## Provenance and rights

All 519 sessions were generated using GPT-4o-mini. Candidate names were generated using GPT-5.4-mini. QA and reference answers are computed offline without LLM calls.
