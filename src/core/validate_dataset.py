"""Validate construction records and their exact public textual evidence; no model evaluation."""
from __future__ import annotations
from collections import Counter
from datetime import date
from .dialogue_scenarios import PLAIN_SCENARIO, scenario_pool, fact_statement
from .temporal import bounds, endpoint_text


def extract_observations(domain):
    """Bind controlled text spans to source records; fail on missing or extra statements.

    This is a constrained-template extractor, not an arbitrary-prose semantic parser.
    Internal references come from the plan only AFTER the corresponding sentence is verified.
    """
    timelines = {t['timeline_id']: t for t in domain['timelines']}
    if len(timelines) != len(domain['timelines']):
        raise ValueError('Duplicate timeline ID')
    facts = {(tid, f['fact_id']): f for tid, t in timelines.items() for f in t['facts']}
    expected = {(tid, fid, e) for (tid, fid), f in facts.items() for e in ('start', 'end')
                if e == 'start' or f['valid_to'] is not None}
    for f in facts.values():
        if f['valid_to'] is not None and bounds(f['valid_to'], f['precision_end'])[1] < bounds(f['valid_from'], f['precision_start'])[0]:
            raise ValueError('Reversed source interval')
    observations, seen, session_ids, mention_ids = [], set(), set(), set()
    scenes = {s.scenario_id: s for s in (*scenario_pool(domain['domain_id']), PLAIN_SCENARIO)}
    previous = None
    for stage, session in enumerate(domain['sessions'], 1):
        sid = session['session_id']
        if sid in session_ids or session['session_index'] != stage:
            raise ValueError('Duplicate session or incorrect session_index')
        session_ids.add(sid)
        now = date.fromisoformat(session['session_time'])
        if previous and (now - previous).days not in (1, 7, 30):
            raise ValueError('Incorrect session time gap')
        previous = now
        scene = scenes[session['scenario_id']]
        natural = session.get('_gold', {}).get('generation', {}).get('method') == 'llm_natural'
        if natural:
            from .natural_dialogue import aligned_spans
            actual = aligned_spans(session, domain['timelines'])
        else:
            actual, offset = [], 0
            for line in session['content'].splitlines(keepends=True):
                text = line.rstrip('\r\n')
                speaker, sep, body = text.partition(': ')
                if not sep or speaker not in scene.participants:
                    raise ValueError(f'{sid}: invalid controlled-text speaker/line')
                if body not in (scene.opening, scene.closing):
                    lead_length = 0
                    for lead in scene.lead_ins:
                        if body.startswith(lead + ' '):
                            lead_length = len(lead) + 1
                            body = body[lead_length:]
                            break
                    start = offset + len(speaker) + 2 + lead_length
                    actual.append({'quote': body, 'char_start': start, 'char_end': start + len(body)})
                offset += len(line)
        mentions = session['_gold']['mentions']
        if not 0 < len(mentions) <= 6 or len(mentions) != len(actual):
            raise ValueError(f'{sid}: endpoint/text capacity mismatch')
        for mention, span in zip(mentions, actual):
            tid, fid, ep = mention['timeline_id'], mention['fact_id'], mention['endpoint']
            key = (tid, fid, ep)
            if key not in expected or key in seen or mention['mention_id'] in mention_ids:
                raise ValueError(f'{sid}: unknown/duplicate endpoint')
            seen.add(key); mention_ids.add(mention['mention_id'])
            t, f = timelines[tid], facts[(tid, fid)]
            value = f['valid_from' if ep == 'start' else 'valid_to']
            precision = f['precision_start' if ep == 'start' else 'precision_end']
            if (mention['value'], mention['valid_time'], mention['precision']) != (f['value'], value, precision):
                raise ValueError(f'{sid}: endpoint metadata does not match source')
            if bounds(value, precision)[1] > now:
                raise ValueError(f'{sid}: potentially future endpoint')
            if scene.scenario_id == 'plain':
                sentence = f"At {endpoint_text(f, ep)}, {f['value']} {'became' if ep == 'start' else 'ceased to be'} the recorded value of {t['relation']}."
            else:
                sentence = fact_statement(t['domain_id'], t['subject'], t['relation'], f['value'], ep, endpoint_text(f, ep))
            if not natural and (span['quote'] != sentence or session['content'][span['char_start']:span['char_end']] != sentence):
                raise ValueError(f'{sid}: public text does not match the assigned endpoint')
            observations.append({'event_id': mention['mention_id'], 'timeline_id': tid, 'fact_id': fid,
                                 'subject': t['subject'], 'relation': t['relation'], 'domain_id': t['domain_id'],
                                 'value': f['value'], 'endpoint': ep, 'valid_time': value, 'precision': precision,
                                 'session_id': sid, 'session_index': stage, 'session_time': session['session_time'], **span})
    if seen != expected:
        raise ValueError('Missing endpoint coverage')
    if domain['schedule']['num_sessions'] != len(domain['sessions']):
        raise ValueError('Schedule/session count mismatch')
    position = {(e['timeline_id'], e['fact_id'], e['endpoint']): e['session_index'] for e in observations}
    for tid, t in timelines.items():
        for f in t['facts']:
            if f['valid_to'] is not None and position[(tid, f['fact_id'], 'start')] == position[(tid, f['fact_id'], 'end')]:
                raise ValueError('Complete fact endpoints must be in distinct sessions')
        for ep, field in [('start', 'session_from'), ('end', 'session_end')]:
            profile = {str(f['chain_index']): position[(tid, f['fact_id'], ep)] for f in t['facts']
                       if ep == 'start' or f['valid_to'] is not None}
            if t['observation_profile'][field] != profile:
                raise ValueError('Observation profile mismatch')
    return observations


def validate_document(doc, *, check_text=True):
    try:
        ids = [d['domain_id'] for d in doc['domains']]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('Empty/duplicate domains')
        for domain in doc['domains']:
            extract_observations(domain)
        return []
    except (ValueError, KeyError, TypeError) as exc:
        return [str(exc)]


if __name__ == '__main__':
    import argparse, json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    errors = validate_document(json.loads(args.data.read_text()))
    print(json.dumps({'errors': errors}))
    raise SystemExit(bool(errors))
