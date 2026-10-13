"""Rebuild QA offline from completed audited dialogues; preserve every public session byte."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_dataset import normalize_timelines, schedule_document
from src.core.natural_dialogue import session_plan,sha
from src.core.validate_dataset import validate_document
from src.qa.competence_qa import build_questions,export_bundle,VERSION
from src.qa.validation import validate_standard


def rebuild(input_dir,checkpoint,seed=42,quota=250):
    directory=Path(input_dir);cp=Path(checkpoint)
    old_bytes=(directory/'06_standard.json').read_bytes();old=json.loads(old_bytes)
    state=json.loads(cp.read_text())
    if state.get('status')!='complete' or state.get('failed'):raise ValueError('Complete natural dialogue checkpoint required')
    facts=json.loads((directory/'01_virtual_facts.json').read_text())
    doc=schedule_document(normalize_timelines(facts),seed)
    plans=[{'session_id':s['session_id'],'plan':session_plan(s,d['timelines'])} for d in doc['domains'] for s in d['sessions']]
    if sha(plans)!=state['provenance']['plans_sha256']:raise ValueError('Checkpoint does not match source schedule')
    by_id={s['session_id']:s for s in old['sessions']}
    if len(by_id)!=len(plans) or set(by_id)!=set(state['completed']):raise ValueError('Session identity mismatch')
    for d in doc['domains']:
        for s in d['sessions']:
            record=state['completed'][s['session_id']]
            s['content']=record['content']
            s['_gold']['generation']=record['generation']
            s['_gold']['natural_audit']=record['audit']
            original=by_id[s['session_id']]
            if (s['content'],s['session_time'],s['session_index'])!=(original['content'],original['timestamp'],original['session_index']):
                raise ValueError('Saved natural session differs from published session')
    errors=validate_document(doc)
    if errors:raise ValueError(errors[:5])
    qa=build_questions(doc,seed,quota)
    bundle=export_bundle(doc,qa,old['sessions']);validate_standard(bundle)
    assert bundle['sessions']==old['sessions']
    report = {'schema_version': VERSION, 'counts': {'sessions': len(bundle['sessions']), 'questions': len(bundle['questions']), 'sequences': len(bundle['sequences'])}, **{k:qa[k] for k in ('question_summary','dimension_summary','scope_summary','sampling')}, 'session_objects_unchanged': True, 'llm_calls': 0}
    for name, data in [('06_standard.json', bundle), ('qa_rebuild_report.json', report)]:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=directory,
                                         delete=False, suffix='.json') as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
            temporary = handle.name
        os.replace(temporary, directory/name)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,default=ROOT/'data/llm_natural')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'construction/audited_sessions.json')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--base-quota',type=int,default=250)
    a=p.parse_args();r=rebuild(a.input_dir,a.checkpoint,a.seed,a.base_quota)
    print(json.dumps({k:r[k] for k in ('counts','question_summary','dimension_summary','scope_summary','sampling')},ensure_ascii=False,indent=2))
