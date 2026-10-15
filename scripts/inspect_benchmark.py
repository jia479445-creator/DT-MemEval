"""Validate the canonical release and print its task/domain composition."""
import json
from pathlib import Path
from collections import Counter
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.qa.validation import validate_standard

def main():
    bundle = json.loads((ROOT / "data/llm_natural/standard.json").read_text())
    validate_standard(bundle)
    questions = bundle["questions"]
    ids = [q["question_id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question IDs")
    counts = {"sequences": len(bundle["sequences"]), "sessions": len(bundle["sessions"]),
              "questions": len(questions)}
    if counts != {"sequences": 4, "sessions": 519, "questions": 2000}:
        raise ValueError(f"Unexpected release composition: {counts}")
    tasks = Counter(q["task_id"] for q in questions)
    if len(tasks) != 8 or set(tasks.values()) != {250}:
        raise ValueError("Expected eight tasks with 250 questions each")
    print(json.dumps({"schema": bundle["schema_version"], **counts, "tasks": dict(tasks),
                      "domains": dict(Counter(q["sequence_id"] for q in questions))}, indent=2))

if __name__ == "__main__":
    main()
