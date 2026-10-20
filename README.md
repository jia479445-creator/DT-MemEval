# DT-MemEval

**A benchmark for reasoning over factual time and observation time in long-term conversational memory.**

DT-MemEval separates when an event happened from when its report became observable. It tests temporal attribution, consolidation across sessions, cross-axis reasoning, and retrospective knowledge states using fictionalized entities and controlled synthetic conversations.

| Release | Sequences | Sessions | Questions | Tasks | Schema |
|---|---:|---:|---:|---:|---|
| 1.0.0 | 4 | 519 | 2,000 | 8 (250 each) | 6.1-full-history-interchange |

This repository releases the fixed dataset and its documentation. It does not include a baseline runner, a model judge, or experimental scores. No publication, venue acceptance, or DOI is claimed.

## Quick start

Download or clone this repository and load `data/llm_natural/standard.json` in your evaluation framework. The bundle contains sessions, questions, shared prompts and evaluator-only reference answers. No dataset generation or rebuild step is required.

Ingest session `memory_input` or `messages` in session order for each sequence. Retrieve using question `text`. Send `metadata.system_prompt` as the system message and combine retrieved memories, the question's `user_prompt`, and `QUESTION: {text}` in the user message. Keep question `evaluation` fields out of model inputs.

## Tasks

| Dimension | ID | Task | Questions |
|---|---|---|---:|
| Temporal attribution | A1 | Single-axis localization | 250 |
| Temporal attribution | A2 | Dual-time attribution | 250 |
| Temporal consolidation | B1 | Interval reconstruction | 250 |
| Temporal consolidation | B2 | Dual-history reconstruction | 250 |
| Cross-axis reasoning | C1 | Dual-axis ordering | 250 |
| Cross-axis reasoning | C2 | State determination and earliest sufficient observation | 250 |
| Epistemic state discrimination | D1 | Retrospective interval knowledge | 250 |
| Epistemic state discrimination | D2 | Knowledge evolution between observation cutoffs | 250 |

Fact intervals use `[start, end)`. Observation time is the session timestamp. A report about an end can arrive before a report about the corresponding start. All questions are scheduled after the trajectory; D1/D2 still impose explicit historical knowledge cutoffs. Unknown values must remain unknown when the allowed reports do not establish them.

## Repository layout

```text
data/llm_natural/        Dataset, fictional facts and release manifest
docs/                   Dataset card and data/evaluation specification
CITATION.cff            Citation metadata
```

The virtual facts are evaluator-side data. Memory systems ingest session `memory_input` or `messages`; retrieval uses question `text`. The complete bundle includes gold supervision for evaluation, so it is not safe to feed the entire JSON object to an answering model.

The release manifest records dataset counts and SHA-256 checksums for the packaged files.

## Model provenance

All 519 sessions were generated using **GPT-4o-mini**. Candidate names were generated using **GPT-5.4-mini**. Questions and reference answers are generated programmatically without an LLM.

## Documentation

- [Dataset card](docs/DATASET_CARD.md): composition, construction, provenance and limitations.
- [Data and evaluation specification](docs/DATA_FORMAT.md): fields, visibility rules and evaluation protocol.
- [Citation metadata](CITATION.cff).

## Citation

For anonymous review, cite the artifact as DT-MemEval and record the release/commit used. Do not substitute an unpublished paper title or an invented venue. Please include the QA schema version and the dataset SHA-256 in experimental reports.
