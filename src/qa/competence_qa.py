"""Eight dual-timeline competencies, solved from audited reports, without model calls."""
from __future__ import annotations
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date, timedelta
from itertools import combinations, product

from src.core.validate_dataset import extract_observations
from src.core.temporal import bounds, compare_bounds
from .prompts import SYSTEM_PROMPT, user_prompt
from .checkpoint_qa import stable, record_key, event_date, order, describe, evidence_rows

VERSION = '6.1-dual-timeline-competence'
SCHEMA = '6.1-full-history-interchange'
TASKS = {
    'single_axis_localization': ('A1', 'temporal_attribution'),
    'dual_time_attribution': ('A2', 'temporal_attribution'),
    'interval_reconstruction': ('B1', 'temporal_consolidation'),
    'dual_history_reconstruction': ('B2', 'temporal_consolidation'),
    'dual_axis_order': ('C1', 'cross_axis_reasoning'),
    'state_determination': ('C2', 'cross_axis_reasoning'),
    'retrospective_state': ('D1', 'epistemic_state_discrimination'),
    'knowledge_evolution': ('D2', 'epistemic_state_discrimination'),
}


def first_reports(events):
    out = {}
    for e in sorted(events, key=lambda x: (x['session_time'], x['event_id'])):
        out.setdefault((*record_key(e), e['endpoint']), e)
    return list(out.values())


def record_reports(events, event, cutoff):
    return [e for e in first_reports(events)
            if record_key(e) == record_key(event) and e['session_index'] <= cutoff]


def state(reports):
    ep = {e['endpoint']: e for e in reports}
    return {'start': event_date(ep['start']) if 'start' in ep else None,
            'end': event_date(ep['end']) if 'end' in ep else None,
            'completeness': 'Complete' if len(ep) == 2 else 'Partial' if ep else 'Unknown'}


def determination(reports, moment):
    """Conservative interval membership for [start,end), under endpoint precision."""
    t = date.fromisoformat(moment)
    ep = {e['endpoint']: e for e in reports}
    a = bounds(ep['start']['valid_time'], ep['start']['precision']) if 'start' in ep else None
    b = bounds(ep['end']['valid_time'], ep['end']['precision']) if 'end' in ep else None
    if (a and t < a[0]) or (b and t >= b[1]):
        return 'Refuted'
    if a and b and a[1] <= t < b[0]:
        return 'Confirmed'
    return 'Undetermined'


def support_options(reports, moment):
    verdict = determination(reports, moment)
    # Indeterminacy needs the actual known reports plus the exhaustive-prefix audit.
    if verdict == 'Undetermined':
        return [reports]
    for n in range(1, len(reports) + 1):
        options = [list(c) for c in combinations(reports, n) if determination(c, moment) == verdict]
        if options:
            return options
    raise ValueError('No support for determined proposition')


def merge_options(groups):
    options = []
    for parts in product(*groups):
        unique = {e['event_id']: e for part in parts for e in part}
        options.append(list(unique.values()))
    # Keep inclusion-minimal supports, preserving alternative valid evidence.
    ids = [frozenset(e['session_time'] for e in row) for row in options]
    return [row for i, row in enumerate(options)
            if ids[i] not in ids[:i] and not any(other < ids[i] for other in ids)]


def chronology(chosen, axis):
    if axis == 'factual':
        for a, b in combinations(chosen, 2):
            if compare_bounds(bounds(a['valid_time'], a['precision']), bounds(b['valid_time'], b['precision'])) is None:
                raise ValueError('Indeterminate order at reported date precision')
        keys = [bounds(e['valid_time'], e['precision'])[0] for e in chosen]
    else:
        keys = [e['session_time'] for e in chosen]
    grouped = defaultdict(list)
    for i, key in enumerate(keys):
        grouped[key].append(chr(65 + i))
    return [sorted(grouped[key]) for key in sorted(grouped)]


def solve(query, events, last):
    """Return gold and ALL minimal sufficient event sets (not just one canonical set)."""
    by_id = {e['event_id']: e for e in events}
    chosen = [by_id[i] for i in query['event_ids']]
    e = chosen[0]; task = query['operation']
    if task == 'single_axis_localization':
        field = 'endpoint_time' if query['axis'] == 'valid' else 'session_time'
        value = event_date(e) if query['axis'] == 'valid' else e['session_time']
        return {'answer_type': task, field: value}, [chosen]
    if task == 'dual_time_attribution':
        return {'answer_type': task, 'endpoint_time': event_date(e), 'session_time': e['session_time']}, [chosen]
    if task in ('interval_reconstruction', 'retrospective_state'):
        cutoff = query.get('cutoff', last)
        reports = record_reports(events, e, cutoff)
        answer = state(reports)
        if task == 'interval_reconstruction':
            if len(reports) != 2 or len({r['session_index'] for r in reports}) != 2:
                raise ValueError('B1 requires two cross-session endpoints')
            answer.pop('completeness')
        return {'answer_type': task, **answer}, [reports]
    if task == 'dual_history_reconstruction':
        if not 3 <= len(chosen) <= 5:
            raise ValueError('B2 needs 3--5 labelled events')
        return {'answer_type': task, 'factual_order': chronology(chosen, 'factual'),
                'observation_order': chronology(chosen, 'observation')}, [chosen]
    if task == 'dual_axis_order':
        a, b = chosen
        factual = compare_bounds(bounds(a['valid_time'], a['precision']), bounds(b['valid_time'], b['precision']))
        if factual is None:
            raise ValueError('Uncertain pair order')
        return {'answer_type': task, 'factual_relation': factual,
                'observation_relation': order(a['session_time'], b['session_time'])}, [chosen]
    if task == 'state_determination':
        reports = record_reports(events, e, last)
        verdict = determination(reports, query['valid_at'])
        when = None
        eligible = reports
        if verdict != 'Undetermined':
            for stage in sorted({r['session_index'] for r in reports}):
                prefix = [r for r in reports if r['session_index'] <= stage]
                if determination(prefix, query['valid_at']) == verdict:
                    eligible = prefix
                    when = next(r['session_time'] for r in reports if r['session_index'] == stage)
                    break
        return {'answer_type': task, 'determination': verdict, 'earliest_session_time': when}, support_options(eligible, query['valid_at'])
    if task == 'knowledge_evolution':
        a, b = query['from_session'], query['to_session']
        if not 1 <= a < b <= last:
            raise ValueError('Invalid observation cutoffs')
        before = record_reports(events, e, a); after = record_reports(events, e, b)
        new = [r for r in after if r['session_index'] > a]
        return {'answer_type': task,
                'earlier_determination': determination(before, query['valid_at']),
                'later_determination': determination(after, query['valid_at']),
                'new_information': {r['endpoint']: event_date(r) for r in new}}, merge_options([
                    support_options(before, query['valid_at']), support_options(after, query['valid_at']), [new]])
    raise ValueError(f'Unknown task: {task}')


def observation_date(events, stage):
    dates = {e['session_time'] for e in events if e['session_index'] == stage}
    if len(dates) != 1:
        raise ValueError('Missing or ambiguous session date')
    return next(iter(dates))


def scope_for(query, events):
    if query['operation'] == 'retrospective_state':
        return {'mode': 'as_of', 'session_index': query['cutoff'],
                'session_time': observation_date(events, query['cutoff'])}
    if query['operation'] == 'knowledge_evolution':
        return {'mode': 'between', 'from_session': query['from_session'], 'to_session': query['to_session'],
                'from_session_time': observation_date(events, query['from_session']),
                'to_session_time': observation_date(events, query['to_session'])}
    return {'mode': 'all'}


def render(query, events):
    """Render only the question; instructions belong to the prompt fields."""
    by_id = {e['event_id']: e for e in events}
    chosen = [by_id[i] for i in query['event_ids']]
    e = chosen[0]; task = query['operation']
    event = f"the {e['endpoint']} endpoint of {describe(e)}"
    if task == 'single_axis_localization':
        if query['axis'] == 'valid':
            return f'What factual date was reported for {event}?'
        return f'On what session_time date was {event} first reported?'
    if task == 'dual_time_attribution':
        return f'What is the factual date of {event}, and on what session_time date was this endpoint first reported?'
    if task == 'interval_reconstruction':
        return f'What is the validity interval of {describe(e)}?'
    if task in ('dual_history_reconstruction', 'dual_axis_order'):
        labels = '; '.join(f"{chr(65+i)}: the {x['endpoint']} endpoint of {describe(x)}" for i, x in enumerate(chosen))
        if task == 'dual_history_reconstruction':
            return f'{labels}. What is the order of these events by factual endpoint date, and what is their order by the session_time of first reports?'
        return f'{labels}. How does A relate to B in factual endpoint date and in the session_time of first reports?'
    if task == 'state_determination':
        return f'Does the query date {query["valid_at"]} lie within the validity interval of {describe(e)}, and at what earliest session_time was that determination supported?'
    if task == 'knowledge_evolution':
        a = observation_date(events, query['from_session']); b = observation_date(events, query['to_session'])
        return f'As of session_time {a} and {b}, respectively, did the query date {query["valid_at"]} lie within the validity interval of {describe(e)}, and what new endpoint information was reported between those checkpoints?'
    if task == 'retrospective_state':
        return f"As of session_time {observation_date(events, query['cutoff'])}, what interval had been reported for {describe(e)}, and how complete was the endpoint information?"
    raise ValueError(f'Unknown task: {task}')


def audit_for(query, events, last, options):
    by_id = {e['event_id']: e for e in events}
    keys = {record_key(by_id[i]) for i in query['event_ids']}
    relevant = [e for e in first_reports(events) if record_key(e) in keys]
    scope = scope_for(query, events)
    cutoff = scope.get('session_index', scope.get('to_session', last))
    # Never ship later reports to the judge for a retrospective question.
    permitted = [e for e in relevant if e['session_index'] <= cutoff]
    stages = [query['from_session'], query['to_session']] if query['operation'] == 'knowledge_evolution' else [cutoff]
    prefix_checks = []
    for stage in stages:
        for key in sorted(keys):
            known = [e for e in relevant if record_key(e) == key and e['session_index'] <= stage]
            prefix_checks.append({'record_id': stable(key), 'cutoff_session_time': observation_date(events, stage),
                                  'known_endpoints': {e['endpoint']: event_date(e) for e in known},
                                  'unreported_endpoints': [ep for ep in ('start', 'end') if ep not in {r['endpoint'] for r in known}],
                                  'verified_against_complete_prefix': True})
    return {'protocol': VERSION, 'citation_unit': 'session_time',
            'minimal_support_date_sets': sorted({tuple(sorted({e['session_time'] for e in option})) for option in options}),
            'reports': [{**row, 'endpoint': e['endpoint'], 'endpoint_time': event_date(e),
                         'precision': e['precision'], 'record_id': stable(record_key(e))}
                        for e in permitted for row in evidence_rows([e])],
            'prefix_checks': prefix_checks}


def stratified(items, limit, seed, key):
    groups = defaultdict(list)
    for item in items:
        groups[item['sequence_id']].append(item)
    for seq in groups:
        groups[seq].sort(key=lambda x: stable([seed, key, x]))
    out = []; i = 0
    while len(out) < limit:
        added = False
        for seq in sorted(groups):
            if i < len(groups[seq]):
                out.append(groups[seq][i]); added = True
            if len(out) == limit:
                break
        if not added:
            break
        i += 1
    return out


def build_questions(document, seed=42, base_quota=250):
    if type(base_quota) is not int or base_quota < 1:
        raise ValueError('base_quota must be positive')
    observed = {d['domain_id']: extract_observations(d) for d in document['domains']}
    lasts = {d['domain_id']: len(d['sessions']) for d in document['domains']}
    candidates = defaultdict(dict); exclusions = Counter()
    def add(seq, query, stratum='all'):
        key = stable(query)
        candidates[query['operation']][(seq, key)] = {'sequence_id': seq, 'query': query, 'stratum': stratum}
    for domain in document['domains']:
        seq = domain['domain_id']; events = observed[seq]; last = lasts[seq]
        counts = {t['timeline_id']: Counter(f['value'] for f in t['facts']) for t in domain['timelines']}
        safe = [e for e in first_reports(events) if counts[e['timeline_id']][e['value']] == 1]
        exclusions['ambiguous_repeated_record_endpoints'] += len(events) - len(safe)
        groups = defaultdict(list); records = defaultdict(list)
        for e in safe:
            groups[e['timeline_id']].append(e); records[record_key(e)].append(e)
            for axis in ('valid', 'observation'):
                add(seq, {'operation': 'single_axis_localization', 'event_ids': [e['event_id']], 'axis': axis}, axis)
            add(seq, {'operation': 'dual_time_attribution', 'event_ids': [e['event_id']]})
        for reports in records.values():
            reports.sort(key=lambda x: x['session_index'])
            e = reports[0]; ids = [e['event_id']]; early = reports[0]['session_index']; late = reports[-1]['session_index']
            ep = {r['endpoint']: r for r in reports}
            if len(ep) == 2 and early != late:
                add(seq, {'operation': 'interval_reconstruction', 'event_ids': ids})
            for stage in sorted({max(1, early-1), early, late}):
                if stage < last:
                    q = {'operation': 'retrospective_state', 'event_ids': ids, 'cutoff': stage}
                    a, _ = solve(q, reports, last); add(seq, q, a['completeness'])
            start = bounds(ep['start']['valid_time'], ep['start']['precision']) if 'start' in ep else None
            end = bounds(ep['end']['valid_time'], ep['end']['precision']) if 'end' in ep else None
            moments = set()
            for r in reports:
                lo, hi = bounds(r['valid_time'], r['precision'])
                moments.update([lo-timedelta(days=1), lo, hi, hi+timedelta(days=1)])
            if start and end and start[1] < end[0]:
                moments.add(start[1]+(end[0]-start[1])//2)
            for moment in sorted(moments):
                q = {'operation': 'state_determination', 'event_ids': ids, 'valid_at': moment.isoformat()}
                a, _ = solve(q, reports, last); add(seq, q, a['determination'])
                stages = sorted({1, max(1, early-1), early, max(1, late-1), late, min(last, late+1)})
                for x, y in combinations(stages, 2):
                    q2 = {'operation': 'knowledge_evolution', 'event_ids': ids, 'valid_at': moment.isoformat(), 'from_session': x, 'to_session': y}
                    a2, _ = solve(q2, reports, last)
                    transition = a2['earlier_determination'] + '->' + a2['later_determination']
                    add(seq, q2, transition)
        for tid, group in groups.items():
            for a, b in combinations(group, 2):
                if record_key(a) == record_key(b):
                    continue
                factual = compare_bounds(bounds(a['valid_time'], a['precision']), bounds(b['valid_time'], b['precision']))
                if factual is None:
                    continue
                obs = order(a['session_time'], b['session_time'])
                cond = 'inverted' if {factual, obs} == {'before', 'after'} else 'aligned' if factual == obs else 'tied_axis'
                ids = [a['event_id'], b['event_id']]
                if int(stable(ids), 16) % 2:
                    ids.reverse()
                add(seq, {'operation': 'dual_axis_order', 'event_ids': ids}, cond)
            # Bounded deterministic permutations avoid combinatorial materialization.
            for turn in range(12):
                shuffled = sorted(group, key=lambda x: stable([seed, tid, turn, x['event_id']]))
                for n in (3, 4, 5):
                    if len(shuffled) < n:
                        continue
                    q = {'operation': 'dual_history_reconstruction', 'event_ids': [x['event_id'] for x in shuffled[:n]]}
                    try:
                        solve(q, group, last)
                    except ValueError:
                        exclusions['uncertain_history_order'] += 1; continue
                    add(seq, q, str(n))
    selected = []; shortfalls = {}; strata_report = {}
    for task in TASKS:
        strata = defaultdict(list)
        for item in candidates[task].values():
            strata[item['stratum']].append(item)
        chosen = []
        if task == 'single_axis_localization':
            # Paired queries ask about the SAME endpoint on the two different axes.
            cohort = stratified(strata['valid'], (base_quota + 1)//2, seed, 'paired_axis_endpoints')
            for item in cohort:
                for axis in ('valid', 'observation'):
                    if len(chosen) >= base_quota:
                        break
                    row = deepcopy(item); row['query']['axis'] = axis; row['stratum'] = axis
                    chosen.append(row)
        else:
            for i, name in enumerate(sorted(strata)):
                target = base_quota // len(strata) + (i < base_quota % len(strata))
                chosen.extend(stratified(strata[name], target, seed, [task, name]))
        # Report shortages instead of duplicating examples; fill from unused valid cases.
        used = {(x['sequence_id'], stable(x['query'])) for x in chosen}
        rest = [x for x in candidates[task].values() if (x['sequence_id'], stable(x['query'])) not in used]
        chosen += stratified(rest, base_quota-len(chosen), seed, [task, 'remaining'])
        if len(chosen) < base_quota:
            shortfalls[task] = {'target': base_quota, 'available': len(chosen)}
        strata_report[task] = dict(Counter(x['stratum'] for x in chosen))
        selected.extend(chosen)
    questions = []
    for item in selected:
        seq = item['sequence_id']; query = item['query']; task = query['operation']; events = observed[seq]; last = lasts[seq]
        answer, options = solve(query, events, last)
        scope = scope_for(query, events); cutoff = scope.get('session_index', scope.get('to_session', last))
        task_id, dimension = TASKS[task]
        support = min(options, key=lambda xs: (len({e['session_time'] for e in xs}), stable([e['event_id'] for e in xs])))
        questions.append({'question_id': f'{seq}::v6::{task_id}::{stable(query)}', 'sequence_id': seq,
                          'task': task, 'task_id': task_id, 'dimension': dimension, 'text': render(query, events), 'user_prompt': user_prompt(query), 'query': query,
                          'evaluation_at': 'trajectory_end', 'knowledge_scope': scope,
                          'checkpoint_session': last, 'available_after_session': last, 'as_of_session': cutoff,
                          'retention_checkpoints': [], 'family_id': stable([seq, sorted(query['event_ids'])]),
                          'reference_answer': answer, 'evidence': evidence_rows(support),
                          'evidence_audit': audit_for(query, events, last, options),
                          'answerable_after_session': max((e['session_index'] for e in support), default=cutoff),
                          'difficulty': 'cross_session' if len({e['session_time'] for e in support}) > 1 else 'single_session',
                          'stratum': item['stratum']})
    questions.sort(key=lambda q: (q['sequence_id'], q['task_id'], q['question_id']))
    result = {'schema_version': VERSION, 'system_prompt': SYSTEM_PROMPT, 'task_taxonomy': {k: {'task_id': v[0], 'dimension': v[1]} for k, v in TASKS.items()},
              'questions': questions, 'question_summary': dict(Counter(q['task'] for q in questions)),
              'dimension_summary': dict(Counter(q['dimension'] for q in questions)),
              'scope_summary': dict(Counter(q['knowledge_scope']['mode'] for q in questions)),
              'sampling': {'seed': seed, 'base_quota': base_quota, 'target_questions': base_quota*8,
                           'candidate_counts': {k: len(v) for k, v in candidates.items()}, 'shortfalls': shortfalls,
                           'strata_counts': strata_report, 'policy': 'equal task quotas; condition strata; domain round robin'},
              'excluded_candidates': dict(exclusions)}
    validate_questions(document, result)
    return result


def validate_questions(document, result):
    if result.get('system_prompt') != SYSTEM_PROMPT:
        raise ValueError('Missing or mismatched shared system prompt')
    observed = {d['domain_id']: extract_observations(d) for d in document['domains']}
    lasts = {d['domain_id']: len(d['sessions']) for d in document['domains']}
    ids = set(); semantic = set()
    safe = {}
    for seq, events in observed.items():
        names = defaultdict(set)
        for e in events:
            names[(e['timeline_id'], e['value'])].add(e['fact_id'])
        safe[seq] = {e['event_id'] for e in events if len(names[(e['timeline_id'], e['value'])]) == 1}
    for q in result['questions']:
        seq = q['sequence_id']; events = observed[seq]; last = lasts[seq]; query = q['query']
        if not set(query['event_ids']) <= safe[seq]:
            raise ValueError('Ambiguous repeated-term reference')
        if q['question_id'] in ids:
            raise ValueError('Duplicate question ID')
        ids.add(q['question_id'])
        if (q['task_id'], q['dimension']) != TASKS[query['operation']] or q['task'] != query['operation']:
            raise ValueError('Incorrect task taxonomy')
        if q['evaluation_at'] != 'trajectory_end' or q['checkpoint_session'] != last or q['available_after_session'] != last or q['retention_checkpoints']:
            raise ValueError('All questions must run after full trajectory')
        scope = scope_for(query, events)
        if q['knowledge_scope'] != scope or q['as_of_session'] != scope.get('session_index', scope.get('to_session', last)):
            raise ValueError('Wrong observation scope')
        gold, options = solve(query, events, last)
        if q.get('user_prompt') != user_prompt(query):
            raise ValueError('Question user prompt mismatch')
        if q['reference_answer'] != gold or q['text'] != render(query, events):
            raise ValueError('Question/gold mismatch')
        # Normalize JSON list/tuple serialization in the audit.
        if stable(q['evidence_audit']) != stable(audit_for(query, events, last, options)):
            raise ValueError('Evidence/absence audit mismatch')
        if q['evidence'] not in [evidence_rows(xs) for xs in options]:
            raise ValueError('Unsupported evidence set')
        if any(e['session_index'] > q['as_of_session'] for e in q['evidence']):
            raise ValueError('Future evidence')
        key = stable([seq, q['text'], gold])
        if key in semantic:
            raise ValueError('Duplicate semantic question')
        semantic.add(key)
    for summary, field in [('question_summary', 'task'), ('dimension_summary', 'dimension'), ('scope_summary', None)]:
        actual = Counter(q[field] if field else q['knowledge_scope']['mode'] for q in result['questions'])
        if result[summary] != dict(actual):
            raise ValueError('Summary mismatch')
    return True


def export_bundle(document, qa, public_sessions):
    validate_questions(document, qa)
    questions = []; sequences = []
    for q in qa['questions']:
        row = {k: deepcopy(q[k]) for k in ('question_id', 'sequence_id', 'task_id', 'dimension', 'text', 'user_prompt', 'evaluation_at',
               'knowledge_scope', 'checkpoint_session', 'available_after_session', 'as_of_session', 'retention_checkpoints')}
        row['question_type'] = q['task']
        row['evaluation'] = {k: deepcopy(q[k]) for k in ('evidence', 'evidence_audit', 'family_id', 'difficulty', 'answerable_after_session', 'stratum')}
        row['evaluation']['reference_answers'] = [deepcopy(q['reference_answer'])]
        row['evaluation']['construction_query'] = deepcopy(q['query'])
        questions.append(row)
    for d in document['domains']:
        seq = d['domain_id']; n = len(d['sessions']); qids = [q['question_id'] for q in questions if q['sequence_id'] == seq]
        sequences.append({'sequence_id': seq, 'session_ids': [s['session_id'] for s in d['sessions']], 'question_ids': qids,
                          'evaluation_at': 'trajectory_end', 'checkpoints': [{'after_session': k, 'question_ids': qids if k == n else []} for k in range(1, n+1)]})
    return {'schema_version': SCHEMA, 'dataset': 'DT-MemEval', 'sessions': deepcopy(public_sessions),
            'questions': questions, 'sequences': sequences,
            'metadata': {'qa_version': VERSION, 'system_prompt': qa['system_prompt'], 'task_taxonomy': qa['task_taxonomy'],
                         'question_summary': qa['question_summary'], 'dimension_summary': qa['dimension_summary'],
                         'scope_summary': qa['scope_summary'], 'sampling': qa['sampling'], 'evaluation_at': 'trajectory_end',
                         'valid_time_semantics': '[start,end); endpoint_time is explicitly a start or end, not an entire interval',
                         'observation_answer_target': 'session_time', 'citation_unit': 'session_time',
                         'evaluation_fields_private': True}}
