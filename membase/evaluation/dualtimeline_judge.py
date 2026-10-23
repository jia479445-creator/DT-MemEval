"""DualTimeline judging: configuration, native answer/evidence scoring and reports.

The native MemBase LLMJudge remains the only model transport. Dataset loading lives
in datasets/dualtimeline.py; prompt text lives in inference_utils/prompts.py.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import date
import json
from pathlib import Path
import re
import time

DEFAULT_JUDGE = 'gpt-4o-mini'
PROTOCOL = 'dualtimeline-v6-answer-evidence'
GROUNDED_METRICS = ('llm_judge', 'evidence_judge', 'grounded_accuracy')
API_CONFIG_KEYS = ('api_keys', 'base_urls', 'api_key', 'base_url')

def normalize_model(model):
    return model.strip().casefold().split('/')[-1].split(':')[-1]


def ensure_distinct_models(qa_model, judge_model):
    if not qa_model or not judge_model: raise ValueError('Specify both qa_model and judge_model')


def read_api_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    keys, urls = config.get('api_keys'), config.get('base_urls')
    if not isinstance(keys,list) or not keys or not isinstance(urls,list) or len(keys)!=len(urls):
        raise ValueError('Judge config needs equally sized nonempty api_keys and base_urls arrays')
    if any(not isinstance(k,str) or not k.strip() or k.startswith('YOUR_') for k in keys):
        raise ValueError('Fill judge API keys in the judge configuration')
    if any(not isinstance(u,str) or not u.startswith(('https://','http://')) or 'YOUR_' in u for u in urls):
        raise ValueError('Fill valid judge base_urls')
    return {'api_keys':keys,'base_urls':urls}


def resolve_judge_settings(dataset_type, qa_model, judge_model, metrics, metric_configs,
                           judge_api_config_path, qa_interface, judge_batch_size):
    """Resolve before QA calls so invalid judge selection cannot waste an answer run."""
    semantic = dataset_type.casefold() == 'dualtimeline'
    chosen = list(metrics) if metrics is not None else (['llm_judge'] if semantic else None)
    model = judge_model or (DEFAULT_JUDGE if semantic else 'gpt-4.1-mini')
    configs = deepcopy(metric_configs or {})
    name = next((m for m in (chosen or ['llm_judge']) if m in ('llm_judge','dualtimeline_judge')),None)
    if name:
        cfg=configs.setdefault(name,{})
        cfg.setdefault('judge_model',model);cfg.setdefault('judge_batch_size',judge_batch_size)
        if judge_api_config_path:cfg.update(read_api_config(judge_api_config_path))
        else:
            for k in ('api_keys','base_urls'):
                if k in qa_interface:cfg.setdefault(k,qa_interface[k])
        if semantic:
            ensure_distinct_models(qa_model,cfg['judge_model'])
            if not cfg.get('api_keys') or not cfg.get('base_urls'):
                raise ValueError('DualTimeline LLM judging requires an API config (--judge-api-config-path)')
    return model,chosen,configs


def configure_dataset_judge(metric_config, qa_model, defaults):
    """Resolve one dataset call; explicit metric credentials override QA defaults."""
    config = deepcopy(metric_config)
    actual_qa_model = qa_model or config.pop('qa_model', None)
    config.pop('qa_model', None)
    config.setdefault('judge_model', defaults.get('judge_model', DEFAULT_JUDGE))
    config.setdefault('judge_batch_size', defaults.get('judge_batch_size', 4))
    config.setdefault('max_retries', defaults.get('max_retries', 3))
    for key in API_CONFIG_KEYS:
        if key in defaults:
            config.setdefault(key, defaults[key])
    ensure_distinct_models(actual_qa_model, config['judge_model'])
    return config


def parse_judge_response(content):
    """Parse a standalone first-line Yes/No verdict without substring matching."""
    text = re.sub(r'<think>.*?</think>', '', content or '', flags=re.S | re.I).strip()
    if not text:
        raise ValueError('Empty judge response')
    verdict = text.splitlines()[0].strip().strip('`*_ ').rstrip('.!:').casefold()
    if verdict not in ('yes', 'no'):
        raise ValueError('DualTimeline judge must return only Yes or No')
    return float(verdict == 'yes')


def prepare_legacy_pairs(qa_pairs):
    """Keep v4/v5 answer-only evaluation available without a second judge backend."""
    pairs = []
    for qa in qa_pairs:
        item = qa.model_copy(deep=True)
        meta = item.metadata
        item.update_metadata({'evidence': meta.get('evaluation', {}).get('evidence', meta.get('evidence', []))})
        item.question = meta.get('original_text', qa.question)
        if meta.get('sequence_id') is not None:
            item.question += f"\nSequence: {meta['sequence_id']}"
        if meta.get('knowledge_scope') is not None:
            item.question += '\nKnowledge scope: ' + json.dumps(meta['knowledge_scope'])
        elif meta.get('as_of_session') is not None:
            item.question += f"\nAllowed observation cutoff: conversation {meta['as_of_session']}"
        pairs.append(item)
    return pairs


def parse_citations(prediction):
    """Read the citation field only: factual dates elsewhere are never citations."""
    value = None
    text = prediction.strip()
    if text.startswith('```'):
        text = '\n'.join(text.splitlines()[1:-1])
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for key, item in obj.items():
                if re.sub(r'[_\s-]+', '', key).casefold() in ('evidencedates', 'citations'):
                    value = item
                    break
    except ValueError:
        pass
    if value is None:
        matches = re.findall(r'(?im)^\s*(?:[-*]\s*)?(?:\*\*)?Evidence[ _-]+dates(?:\*\*)?\s*:\s*(.*)$', prediction)
        if len(matches) != 1:
            raise ValueError('Missing or repeated Evidence dates field')
        value = matches[0].strip().strip('*').strip()
    if isinstance(value, str) and value.casefold().rstrip('.') in ('none', '[]'):
        return []
    if isinstance(value, list):
        entries = value
    elif isinstance(value, str):
        entries = [v.strip(' []\"\'') for v in re.split('[,;]', value.rstrip('.'))]
    else:
        raise ValueError('Evidence dates must be a list or None')
    dates = []
    for entry in entries:
        if not isinstance(entry, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', entry):
            raise ValueError('Evidence dates require YYYY-MM-DD session dates')
        date.fromisoformat(entry)
        dates.append(entry)
    return sorted(set(dates))


def check_citations(qa, prediction):
    """Deterministic provenance checks before semantic evidence adjudication."""
    audit = qa.metadata['evaluation']['evidence_audit']
    cited = set(parse_citations(prediction))
    options = [set(xs) for xs in audit['minimal_support_date_sets']]
    allowed = set().union(*options)
    if not any(option <= cited for option in options):
        return False, sorted(cited), 'missing_required_support'
    if cited - allowed:
        return False, sorted(cited), 'irrelevant_or_out_of_scope_dates'
    if not cited and not all(p['verified_against_complete_prefix'] for p in audit['prefix_checks']):
        return False, [], 'absence_not_verified'
    return True, sorted(cited), None


def prepared_pair(qa, kind, cited=None):
    item = qa.model_copy(deep=True)
    meta = qa.metadata; evaluation = meta['evaluation']; audit = evaluation['evidence_audit']
    item.question = meta.get('original_text', qa.question)
    item.question += '\nPermitted observation scope: ' + json.dumps(meta['knowledge_scope'], ensure_ascii=False)
    if kind == 'answer':
        evidence = {'reference_reports': evaluation['evidence'], 'complete_prefix_audit': audit['prefix_checks']}
    else:
        evidence = {'cited_session_times': cited,
                    'cited_reports': [r for r in audit['reports'] if r['session_time'] in cited],
                    'acceptable_minimal_support_date_sets': audit['minimal_support_date_sets'],
                    'complete_prefix_audit': audit['prefix_checks']}
    item.update_metadata({'evidence': json.dumps(evidence, ensure_ascii=False)})
    return item


def summarize_scores(scores):
    values = [score['value'] for score in scores if score['value'] is not None]
    return {'total': len(scores), 'judged': len(values), 'errors': len(scores)-len(values),
            'accuracy': sum(values)/len(values) if values else None}


def summarize(results, metric_names=GROUNDED_METRICS):
    """Shared by normal evaluation and rejudging; failed requests are not mistakes."""
    return {name: summarize_scores([row[name] for row in results if name in row])
            for name in metric_names}


def grouped_summary(results, metadata, field, metric_names=GROUNDED_METRICS):
    groups = {}
    for result, meta in zip(results, metadata):
        groups.setdefault(meta.get(field, 'unknown'), []).append(result)
    return {label: summarize(rows, metric_names) for label, rows in groups.items()}


def evaluate_grounded(qa_pairs, predictions, judge_config, parse_response, native_compute=None):
    """Keep the original LLMJudge implementation; run two isolated binary judgments."""
    from .llm_judge import LLMJudge
    from smartcomment import comment_graph
    config = deepcopy(judge_config)
    workers = config.pop('judge_batch_size', 4)
    max_retries = int(config.pop('max_retries', 3))
    generation = config.pop('generation_config', {'temperature': 0.0, 'max_tokens': 32, 'stream': False})
    generation.setdefault('max_tolerance', max_retries)
    if 'qwen3' in str(config.get('judge_model', '')).casefold():
        generation.setdefault('enable_thinking', False)
    if workers < 1:
        raise ValueError('judge_batch_size must be positive')
    if len(qa_pairs) != len(predictions):
        raise ValueError('QA/prediction length mismatch')
    judge = LLMJudge(judge_batch_size=1, **config)
    compute = native_compute or judge.compute
    def score_one(args):
        qa, prediction = args
        metrics = {}
        for kind, metric in [('answer', 'llm_judge'), ('evidence', 'evidence_judge')]:
            start = time.perf_counter()
            cited = None
            if kind == 'evidence':
                try:
                    valid, cited, reason = check_citations(qa, prediction)
                except ValueError:
                    valid, reason = False, 'invalid_citation_field'
                if not valid:
                    metrics[metric] = {'value': 0.0, 'metadata': {'status': 'judged', 'method': 'citation_validation',
                                       'reason': reason, 'cited_dates': cited, 'protocol': PROTOCOL}}
                    continue
            pair = prepared_pair(qa, kind, cited)
            try:
                result = compute([prediction], [pair.golden_answers], qa_pairs=[pair],
                                 get_judge_template_name=lambda _: f'dualtimeline-{kind}-judge-v6',
                                 parse_judge_response=parse_response, generation_config=generation)[0]
                if result['value'] not in (0.0, 1.0):
                    raise ValueError('Expected a binary judge verdict')
                result['metadata'].update(status='judged', judge_model=config.get('judge_model'),
                                          protocol=PROTOCOL, latency_s=time.perf_counter()-start)
                metrics[metric] = result
            except Exception as exc:
                # Failed requests/invalid judge output are never counted as model mistakes.
                metrics[metric] = {'value': None, 'metadata': {'status': 'judge_error', 'error_type': type(exc).__name__,
                                   'judge_model': config.get('judge_model'), 'protocol': PROTOCOL,
                                   'latency_s': time.perf_counter()-start}}
        a, e = metrics['llm_judge']['value'], metrics['evidence_judge']['value']
        metrics['grounded_accuracy'] = {'value': a*e if a is not None and e is not None else None,
                                       'metadata': {'status': 'judged' if a is not None and e is not None else 'judge_error',
                                                    'protocol': PROTOCOL}}
        return metrics
    def one(args):
        # Native tracing uses thread-local graph state; give each worker its own scope.
        # This also makes standalone rejudging work when tracing is enabled by default.
        with comment_graph():
            return score_one(args)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, zip(qa_pairs, predictions)))
