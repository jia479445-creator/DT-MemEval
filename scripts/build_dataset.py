"""Build domain-level conversation streams from entity-attribute timelines."""
from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import json
import math
import random
import re
from datetime import date, timedelta
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.dialogue_scenarios import PLAIN_SCENARIO, fact_statement, scenario_pool, select_scenario
from src.core.temporal import bounds, display

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "temporal_interval_qa.json"
DEFAULT_OUT = ROOT / "work" / "constructed_sessions.json"
DATE_RE = re.compile(r"\+(\d{4})-(\d{2})-(\d{2})T")


def parse_date(text: str) -> tuple[date, str]:
    match = DATE_RE.search(text)
    if not match:
        raise ValueError(f"invalid timestamp: {text}")
    year, month, day = map(int, match.groups())
    if month == 0:
        return date(year, 1, 1), "year"
    if day == 0:
        return date(year, month, 1), "month"
    return date(year, month, day), "day"


def parse_facts(node: dict) -> list[dict]:
    facts = []
    if 'facts' in node and 'answers' in node:
        raise ValueError('Ambiguous source node: both facts and answers present')
    for source_index, raw in enumerate(node.get("facts", node.get("answers", []))):
        parts = re.split(r'\|\s*([SE])\s*:', raw)
        if len(parts) not in (3, 5) or parts[1] != 'S' or (len(parts) == 5 and parts[3] != 'E'):
            raise ValueError(f'Malformed source fact: {raw}')
        value = parts[0].strip()
        start_raw = parts[2]
        start, start_precision = parse_date(start_raw)
        end = None
        end_precision = None
        if len(parts) == 5:
            end, end_precision = parse_date(parts[4])
            if bounds(end.isoformat(), end_precision)[1] < bounds(start.isoformat(), start_precision)[0]:
                raise ValueError(f"Provably reversed source interval: {raw}")
        facts.append({
            "value": value,
            "source_index": source_index,
            "source_record": raw,
            "valid_from": start.isoformat(),
            "valid_to": end.isoformat() if end else None,
            "precision_start": start_precision,
            "precision_end": end_precision,
            "end_status": "recorded" if end else "unrecorded",
        })
    facts.sort(key=lambda x: (x["valid_from"], x["value"]))
    for index, fact in enumerate(facts):
        fact["fact_id"] = f"fact_{index:04d}"
        fact["record_ref"] = f"R{index:04d}"
        fact["chain_index"] = index
    return facts


def iter_timelines(source: dict):
    for category, entities in source.items():
        for subject, node in entities.items():
            if category in {"countries_byGDP", "organizations"}:
                for relation, relation_node in node.items():
                    yield category, subject, relation, relation_node
            else:
                relation = "team" if category == "athletes_byPayment" else "chief executive officer"
                yield category, subject, f"{relation} of {subject}", node


def endpoint_date(fact: dict, endpoint: str) -> date:
    value = fact["valid_from"] if endpoint == "start" else fact["valid_to"]
    if value is None:
        raise ValueError("an unrecorded end has no endpoint")
    return bounds(value, fact["precision_start"] if endpoint == "start" else fact["precision_end"])[1]


def make_groups(facts: list[dict], rng: random.Random) -> list[list[tuple]]:
    mentions = []
    for fact in facts:
        mentions.append((fact, "start"))
        if fact["valid_to"] is not None:
            mentions.append((fact, "end"))
    # Keep the original per-timeline grouping before mixing timelines.
    rng.shuffle(mentions)
    session_count = max(3, math.ceil(len(mentions) / 3))
    buckets = [[] for _ in range(session_count)]
    for index, mention in enumerate(mentions):
        buckets[index % session_count].append(mention)
    # Repair same-session start/end collisions without changing bucket capacities.
    # Two records of the same person remain distinct fact IDs.
    for bi, bucket in enumerate(buckets):
        for mi in range(len(bucket)):
            fid = bucket[mi][0]['fact_id']
            if sum(f['fact_id'] == fid for f, _ in bucket) < 2:
                continue
            swapped = False
            for bj, other in enumerate(buckets):
                if bj == bi or any(f['fact_id'] == fid for f, _ in other):
                    continue
                for mj, (peer, _) in enumerate(other):
                    if not any(f['fact_id'] == peer['fact_id'] for f, _ in bucket):
                        bucket[mi], other[mj] = other[mj], bucket[mi]
                        swapped = True
                        break
                if swapped:
                    break
            if not swapped:
                raise ValueError('Cannot split fact endpoints across sessions within capacity')
    # Enforce the six-endpoint capacity. This should hold for E/3, but is checked explicitly.
    if any(len(bucket) > 6 for bucket in buckets):
        raise ValueError("session endpoint capacity exceeded")

    return buckets


def seeded_rng(seed: int, identity: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{identity}".encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


def merge_domain(domain_id: str, timelines: list[dict], seed: int, dialogue_style: str = "varied") -> dict:
    rng = seeded_rng(seed, domain_id)
    queues = {}
    for timeline in timelines:
        tid = timeline["timeline_id"]
        groups = make_groups(timeline["facts"], seeded_rng(seed, tid))
        timeline["local_num_sessions"] = len(groups)
        timeline["session_indices"] = []
        queues[tid] = deque(enumerate(groups, 1))
    by_id = {t["timeline_id"]: t for t in timelines}
    observations = {tid: {} for tid in by_id}
    sessions = []
    latest_fact = max(endpoint_date(fact, endpoint)
                      for timeline in timelines
                      for fact in timeline["facts"]
                      for endpoint in (("start", "end") if fact["valid_to"] is not None else ("start",)))
    floor = latest_fact + timedelta(days=30)
    while queues:
        active = list(queues)
        rng.shuffle(active)
        # At most two groups, always from different timelines. Each group's
        # local order and membership survive the domain-level merge.
        selected = active[:2]
        entries, source_groups = [], []
        for tid in selected:
            local_index, bucket = queues[tid].popleft()
            source_groups.append({"timeline_id": tid, "local_session_index": local_index})
            entries.extend((tid, fact, endpoint) for fact, endpoint in bucket)
            if not queues[tid]:
                del queues[tid]
        if len(entries) > 6:
            raise ValueError("domain session exceeds six endpoints")
        rng.shuffle(entries)
        stage = len(sessions) + 1
        session_time = floor + timedelta(days=rng.choice((1, 7, 30)))
        floor = session_time
        # Dialogue draws never consume the endpoint/time scheduling RNG.
        cycle = (stage - 1) // len(scenario_pool(domain_id))
        scene_rng = seeded_rng(seed, f"scene:{domain_id}:{cycle}")
        scene = select_scenario(domain_id, stage, scene_rng) if dialogue_style == "varied" else PLAIN_SCENARIO
        wording_rng = seeded_rng(seed, f"wording:{domain_id}:{stage}")
        content_parts, mentions = [], []
        if scene.opening:
            content_parts.append(f"{scene.participants[0]}: {scene.opening}")
        for mention_index, (tid, fact, endpoint) in enumerate(entries, 1):
            valid_time = fact["valid_from"] if endpoint == "start" else fact["valid_to"]
            relation = by_id[tid]["relation"]
            rendered_time = display(valid_time, fact["precision_start"] if endpoint == "start" else fact["precision_end"])
            if dialogue_style == "plain":
                action = "became" if endpoint == "start" else "ceased to be"
                text = f"At {rendered_time}, {fact['value']} {action} the recorded value of {relation}."
            else:
                statement = fact_statement(domain_id, by_id[tid]["subject"], relation, fact["value"], endpoint, rendered_time)
                text = f"{wording_rng.choice(scene.lead_ins)} {statement}"
            content_parts.append(f"{scene.participants[len(content_parts) % 2]}: {text}")
            mentions.append({
                "mention_id": f"{domain_id}::s{stage}::m{mention_index}",
                "timeline_id": tid,
                "fact_id": fact["fact_id"],
                "record_ref": fact["record_ref"],
                "chain_index": fact["chain_index"],
                "endpoint": endpoint,
                "value": fact["value"],
                "valid_time": valid_time,
                "precision": fact["precision_start"] if endpoint == "start" else fact["precision_end"],
            })
            observations[tid][(fact["fact_id"], endpoint)] = stage
        if scene.closing:
            content_parts.append(f"{scene.participants[len(content_parts) % 2]}: {scene.closing}")
        if not content_parts:
            content_parts.append("The participants compare their notes and organize the material.")
        for group in source_groups:
            by_id[group["timeline_id"]]["session_indices"].append(stage)
        sessions.append({
            "session_id": f"{domain_id}::s{stage}",
            "session_index": stage,
            "session_time": session_time.isoformat(),
            "participants": list(scene.participants),
            "scenario_id": scene.scenario_id,
            "scenario": scene.description,
            "content": "\n".join(content_parts),
            "_gold": {
                "source_groups": source_groups,
                "mentions": mentions,
                "latest_valid_time_in_session": max((m["valid_time"] for m in mentions), default=None),
            },
        })
    for timeline in timelines:
        tid = timeline["timeline_id"]
        positions = observations[tid]
        timeline["observation_profile"] = {
            "session_from": {str(f["chain_index"]): positions[(f["fact_id"], "start")] for f in timeline["facts"]},
            "session_end": {str(f["chain_index"]): positions[(f["fact_id"], "end")] for f in timeline["facts"] if f["valid_to"] is not None},
        }
    mixed = sum(len({m["timeline_id"] for m in s["_gold"]["mentions"]}) > 1 for s in sessions)
    return {
        "domain_id": domain_id,
        "timelines": timelines,
        "sessions": sessions,
        "schedule": {
            "num_sessions": len(sessions),
            "num_timelines": len(timelines),
            "schedule_type": "randomized",
            "endpoint_rule": "per timeline: m=max(3,ceil(E/3)); separate start/end; merge up to two distinct timeline groups",
            "max_endpoints_per_session": 6,
            "mixed_session_count": mixed,
            "mixed_session_ratio": mixed / len(sessions),
            "scenario_counts": dict(sorted(Counter(s["scenario_id"] for s in sessions).items())),
            "seed": seed,
        },
    }


def normalize_timelines(source: dict, limit: int = 0, per_domain_limit: int = 0) -> dict:
    """Parse facts without creating sessions or questions."""
    if limit < 0 or per_domain_limit < 0:
        raise ValueError("limits must be nonnegative")
    grouped = {}
    total = 0
    for domain_id, subject, relation, node in iter_timelines(source):
        if per_domain_limit and len(grouped.get(domain_id, [])) >= per_domain_limit:
            continue
        facts = parse_facts(node)
        if len(facts) < 2:
            continue
        timeline_id = f"{subject}::{relation}".replace(" ", "_")
        grouped.setdefault(domain_id, []).append({
            "timeline_id": timeline_id,
            "domain_id": domain_id,
            "subject": subject,
            "relation": relation,
            "chain_length": len(facts),
            "facts": facts,
        })
        total += 1
        if limit and total >= limit:
            break
    return grouped


def schedule_document(grouped: dict, seed: int = 42, dialogue_style: str = 'varied',
                      limit: int = 0, per_domain_limit: int = 0) -> dict:
    """Create controlled session text and observation metadata, without QA."""
    from copy import deepcopy
    if dialogue_style not in {'varied', 'plain'}:
        raise ValueError('dialogue_style must be varied or plain')
    grouped = deepcopy(grouped)
    return {
        "benchmark": "DT-MemEval",
        "version": "3.0-precision-paired",
        "session_scope": "domain",
        "config": {"seed": seed, "limit": limit, "per_domain_limit": per_domain_limit,
                   "dialogue_style": dialogue_style, "dialogue_version": "paragraph-v1",
                   "conversation_format": "paragraph", "text_policy": "controlled_facts",
                   "interval_semantics": "start_inclusive_end_exclusive_unknown_end",
                   "date_storage": "lower_bound_with_explicit_precision"},
        "domains": [merge_domain(key, timelines, seed, dialogue_style) for key, timelines in grouped.items()],
    }


def build_document(source: dict, limit: int = 0, seed: int = 42, per_domain_limit: int = 0, dialogue_style: str = "varied") -> dict:
    return schedule_document(normalize_timelines(source, limit, per_domain_limit), seed, dialogue_style,
                             limit, per_domain_limit)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build an offline session schedule from fictional facts.")
    parser.add_argument("--source", type=Path, default=ROOT / "data/llm_natural/virtual_facts.json")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    document = build_document(json.loads(args.source.read_text()), seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
