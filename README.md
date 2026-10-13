# DT-MemEval

**A benchmark for reasoning over factual time and observation time in long-term conversational memory.**

DT-MemEval separates when an event happened from when its report became observable. It tests temporal attribution, consolidation across sessions, cross-axis reasoning, and retrospective knowledge states using fictionalized entities and controlled synthetic conversations.

| Release | Sequences | Sessions | Questions | Tasks | Schema |
|---|---:|---:|---:|---:|---|
| 1.0.0 | 4 | 519 | 2,000 | 8 (250 each) | 6.1-full-history-interchange |

This repository releases the dataset, audited construction inputs, offline question-generation code, and semantic tests. It does not include a baseline runner, a model judge, or experimental scores. No publication, venue acceptance, or DOI is claimed.

## Quick start

Python 3.10 or newer is required. Offline use needs only the standard library. Download the source archive from the anonymous artifact page, extract it into a directory named `DT-MemEval`, and run:

```bash
cd DT-MemEval
python scripts/inspect_benchmark.py
python -m unittest discover -s tests -v
```

```python
import json
from pathlib import Path
from src.qa.prompts import build_messages

bundle = json.loads(Path("data/llm_natural/06_standard.json").read_text())
question = bundle["questions"][0]
# Replace this full-history example with your memory system's retrieved context.
context = "\n\n".join(s["memory_input"] for s in bundle["sessions"]
                         if s["sequence_id"] == question["sequence_id"])
messages = build_messages(question, context,
                          system_prompt=bundle["metadata"]["system_prompt"])
# Send messages to the answering model. Keep question['evaluation'] out of its inputs.
```

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
construction/           Compact audited sessions needed for offline reconstruction
src/                    Temporal semantics, dialogue construction, QA and prompts
scripts/                Construction tools, offline rebuild, release inspection
tests/                  Semantic and construction regression tests
docs/                   Dataset card and data/evaluation specification
```

The virtual facts and construction annotations are evaluator-side artifacts. Memory systems ingest session `memory_input` or `messages`; retrieval uses question `text`. The complete bundle includes gold supervision for evaluation, so it is not safe to feed the entire JSON object to an answering model.

## Reproducibility

```bash
python scripts/rebuild_full_history_qa.py \
  --input-dir data/llm_natural \
  --checkpoint construction/audited_sessions.json \
  --seed 42 --base-quota 250
python scripts/inspect_benchmark.py
```

The QA rebuild uses existing audited dialogue and makes no model calls. It checks the published session objects before rebuilding questions. Historical API generation scripts and naming inputs are excluded. This release supports dataset use and offline QA reconstruction from the frozen audited sessions. Rebuilding QA writes the benchmark and a compact QA rebuild report. The shipped release manifest describes the initial packaged artifacts; compare checksums before and after rebuilding.

## Model provenance

The maintainer confirms that early dialogue generation used **GPT-4o-mini**. Candidate-name generation used **GPT-5.4-mini**. The retained records for all 519 sessions label both generation and auditing as `gpt-5.4-mini`; these later records do not by themselves establish which model originally authored every sentence. We preserve both the maintainer's history and the stored provenance rather than retrospectively relabeling the data. Questions and reference answers are generated programmatically without an LLM.

## Documentation

- [Dataset card](docs/DATASET_CARD.md): composition, construction, provenance and limitations.
- [Data and evaluation specification](docs/DATA_FORMAT.md): fields, visibility rules and evaluation protocol.
- [Citation metadata](CITATION.cff).

## Citation

For anonymous review, cite the artifact as DT-MemEval and record the release/commit used. Do not substitute an unpublished paper title or an invented venue. Please include the QA schema version and the dataset SHA-256 in experimental reports.
