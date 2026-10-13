"""Shared event identity and evidence helpers."""

from datetime import date
import hashlib
import json
from src.core.temporal import bounds, compare_bounds, display

def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]

def record_key(event):
    return event['timeline_id'], event['fact_id']

def event_date(e):
    return display(e['valid_time'], e['precision'])

def order(a, b):
    return 'before' if a < b else 'after' if a > b else 'same'

def describe(e):
    if e['domain_id'] == 'athletes_byPayment':
        return f"the recorded affiliation of {e['subject']} with {e['value']}"
    if e['domain_id'] == 'organizations':
        if e['relation'].lower() == 'headquarters location':
            return f"the recorded use of {e['value']} as the headquarters of {e['subject']}"
        return f"the recorded term of {e['value']} as {e['relation']} of {e['subject']}"
    return f"the recorded term of {e['value']} as {e['relation']}"

def evidence_rows(events):
    fields = ('event_id', 'session_id', 'session_index', 'session_time', 'char_start', 'char_end', 'quote')
    return [{k: e[k] for k in fields} for e in sorted(events, key=lambda x: (x['session_index'], x['char_start']))]
