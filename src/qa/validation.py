from collections import Counter
"""Validate the released interchange bundle."""
def validate_standard(document):
    ss = {s['session_id']: s for s in document['sessions']}
    qs = {q['question_id']: q for q in document['questions']}
    if len(ss) != len(document['sessions']) or len(qs) != len(document['questions']):
        raise ValueError('Duplicate export IDs')
    for seq in document['sequences']:
        if any(ss[sid]['sequence_id'] != seq['sequence_id'] for sid in seq['session_ids']):
            raise ValueError('Session outside sequence')
        for qid in seq['question_ids']:
            q = qs[qid]
            if q['sequence_id'] != seq['sequence_id'] or not 1 <= q['checkpoint_session'] <= len(seq['session_ids']):
                raise ValueError('Question checkpoint outside sequence')
            if document.get('schema_version') in ('6.0-full-history-interchange', '6.1-full-history-interchange'):
                from src.qa.competence_qa import TASKS
                task = q['question_type']; n = len(seq['session_ids']); scope = q['knowledge_scope']
                if task not in TASKS or (q['task_id'], q['dimension']) != TASKS[task]:
                    raise ValueError('Invalid full-history taxonomy')
                if q['evaluation_at'] != 'trajectory_end' or q['checkpoint_session'] != n or q['available_after_session'] != n or q['retention_checkpoints']:
                    raise ValueError('Full-history evaluation must occur at trajectory end')
                if document['schema_version'] == '6.1-full-history-interchange':
                    from src.qa.prompts import SYSTEM_PROMPT, user_prompt
                    if document['metadata'].get('system_prompt') != SYSTEM_PROMPT:
                        raise ValueError('Missing or mismatched shared system prompt')
                    query = q['evaluation']['construction_query']
                    if q.get('user_prompt') != user_prompt(query):
                        raise ValueError('Missing or mismatched question user prompt')
                    # Full question/gold validation is performed before export.
                    if not q['text'].endswith('?') or any(marker in q['text'] for marker in (
                            'Return ', 'Return:', 'Preserve ', 'Evidence dates:',
                            'Using the complete conversation history', 'Looking back from the complete history')):
                        raise ValueError('Question text contains prompt instructions')
                if task == 'retrospective_state':
                    if set(scope) not in ({'mode','session_index'},{'mode','session_index','session_time'}) or scope['mode'] != 'as_of' or not 1 <= scope['session_index'] < n:
                        raise ValueError('Invalid retrospective scope')
                    cutoff = scope['session_index']
                elif task in ('knowledge_change', 'knowledge_evolution'):
                    if set(scope) not in ({'mode','from_session','to_session'},{'mode','from_session','to_session','from_session_time','to_session_time'}) or scope['mode'] != 'between' or not 1 <= scope['from_session'] < scope['to_session'] <= n:
                        raise ValueError('Invalid knowledge-change scope')
                    cutoff = scope['to_session']
                else:
                    if scope != {'mode':'all'}: raise ValueError('Full-history task has a prefix restriction')
                    cutoff = n
                if q['as_of_session'] != cutoff: raise ValueError('Scope/compatibility cutoff mismatch')
                for time_key,index_key in [('session_time','session_index'),('from_session_time','from_session'),('to_session_time','to_session')]:
                    if time_key in scope and scope[time_key] != ss[seq['session_ids'][scope[index_key]-1]]['timestamp']:
                        raise ValueError('Observation date and internal session index mismatch')
            for ev in q['evaluation']['evidence']:
                if ev['session_id'] not in seq['session_ids'] or ev['session_index'] > q['as_of_session']:
                    raise ValueError('Future/foreign evidence')
                session = ss[ev['session_id']]
                if session['content'][ev['char_start']:ev['char_end']] != ev['quote']:
                    raise ValueError('Evidence span mismatch in export')
            if any(k <= q['checkpoint_session'] or k > len(seq['session_ids']) for k in q['retention_checkpoints']):
                raise ValueError('Invalid retention checkpoint')
        references = [qid for cp in seq['checkpoints'] for qid in cp['question_ids']]
        if Counter(references) != Counter(seq['question_ids']):
            raise ValueError('Checkpoint question references inconsistent')
    return True
