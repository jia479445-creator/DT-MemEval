"""v6 temporal semantics and evidence oracles, independent of any model judge."""
from copy import deepcopy
from datetime import date
import unittest
from src.qa.competence_qa import (solve, determination, render, audit_for, build_questions,
                                  validate_questions, TASKS)
from scripts.build_dataset import normalize_timelines, schedule_document
from src.qa.prompts import SYSTEM_PROMPT, user_prompt, build_messages


def reports():
    common = {'timeline_id': 'country::president', 'fact_id': 'a', 'domain_id': 'countries_byGDP',
              'subject': 'Noria', 'relation': 'President of Noria', 'value': 'Arel Vesk', 'precision': 'day',
              'char_start': 0, 'char_end': 1, 'quote': 'x'}
    return [{**common, 'event_id': 'end', 'endpoint': 'end', 'valid_time': '2010-01-01',
             'session_index': 2, 'session_id': 's2', 'session_time': '2025-01-02'},
            {**common, 'event_id': 'start', 'endpoint': 'start', 'valid_time': '2001-01-01',
             'session_index': 5, 'session_id': 's5', 'session_time': '2025-01-05'},
            {**common, 'fact_id': 'b', 'value': 'Belen Daro', 'event_id': 'b-start',
             'endpoint': 'start', 'valid_time': '2015-01-01', 'session_index': 1,
             'session_id': 's1', 'session_time': '2025-01-01'}]


class CompetenceTests(unittest.TestCase):
    def test_end_exclusive_and_single_endpoint_refutation(self):
        rs = reports()
        self.assertEqual(determination(rs[:2], '2001-01-01'), 'Confirmed')
        self.assertEqual(determination(rs[:2], '2010-01-01'), 'Refuted')
        self.assertEqual(determination(rs[:1], '2005-01-01'), 'Undetermined')
        q = {'operation': 'state_determination', 'event_ids': ['start'], 'valid_at': '2011-01-01'}
        a, supports = solve(q, rs, 5)
        self.assertEqual(a['determination'], 'Refuted')
        self.assertEqual(a['earliest_session_time'], '2025-01-02')
        self.assertEqual([[r['event_id'] for r in s] for s in supports], [['end']])
        q['valid_at'] = '2005-01-01'
        self.assertEqual(solve(q, rs, 5)[0]['earliest_session_time'], '2025-01-05')

    def test_precision_uncertainty_and_unknown_end(self):
        rs = reports(); rs[1].update(valid_time='2001-01-01', precision='year')
        self.assertEqual(determination(rs[:2], '2001-06-01'), 'Undetermined')
        self.assertEqual(determination(rs[1:2], '2005-01-01'), 'Undetermined')
        a, _ = solve({'operation': 'dual_time_attribution', 'event_ids': ['start']}, rs, 5)
        self.assertEqual(a['endpoint_time'], '2001')
        self.assertNotIn('valid_time', a)

    def test_complete_partial_unknown_with_no_future_audit_leak(self):
        rs = reports()
        for cutoff, expected in [(1, 'Unknown'), (2, 'Partial'), (5, 'Complete')]:
            q = {'operation': 'retrospective_state', 'event_ids': ['start'], 'cutoff': cutoff}
            a, supports = solve(q, rs, 6)
            self.assertEqual(a['completeness'], expected)
            audit = audit_for(q, rs, 6, supports)
            self.assertTrue(all(r['session_index'] <= cutoff for r in audit['reports']))
            if cutoff == 1:
                self.assertEqual(audit['minimal_support_date_sets'], [()])
                self.assertEqual(audit['prefix_checks'][0]['unreported_endpoints'], ['start', 'end'])

    def test_evolution_may_change_determination_or_stay_refuted(self):
        rs = reports(); q = {'operation': 'knowledge_evolution', 'event_ids': ['start'],
                            'from_session': 2, 'to_session': 5, 'valid_at': '2005-01-01'}
        a, _ = solve(q, rs, 6)
        self.assertEqual((a['earlier_determination'], a['later_determination']), ('Undetermined', 'Confirmed'))
        self.assertEqual(a['new_information'], {'start': '2001-01-01'})
        q['valid_at'] = '2011-01-01'
        a, _ = solve(q, rs, 6)
        self.assertEqual((a['earlier_determination'], a['later_determination']), ('Refuted', 'Refuted'))

    def test_three_event_dual_order_and_precision_rejection(self):
        rs = reports(); q = {'operation': 'dual_history_reconstruction', 'event_ids': ['end', 'start', 'b-start']}
        a, _ = solve(q, rs, 5)
        self.assertEqual(a['factual_order'], [['B'], ['A'], ['C']])
        self.assertEqual(a['observation_order'], [['C'], ['A'], ['B']])
        rs[0].update(valid_time='2001-01-01', precision='year')
        with self.assertRaises(ValueError): solve(q, rs, 5)

    def test_endpoint_questions_dont_call_one_date_an_interval(self):
        rs = reports()
        q = {'operation': 'single_axis_localization', 'axis': 'valid', 'event_ids': ['end']}
        text = render(q, rs)
        self.assertIn('end endpoint', text)
        self.assertIn('Endpoint time', user_prompt(q))
        self.assertNotIn('Return', text)
        self.assertNotIn('2010-01-01', text)

    def test_deterministic_build_and_tampered_absence_audit(self):
        data = {'countries_byGDP': {'Noria': {'President of Noria': {'facts': [
            'Arel Vesk |S: +1990-00-00T00:00:00Z |E: +1993-00-00T00:00:00Z',
            'Belen Daro |S: +1995-01-01T00:00:00Z |E: +1999-01-01T00:00:00Z',
            'Caren Rusk |S: +2000-01-01T00:00:00Z |E: +2005-01-01T00:00:00Z']}}}}
        doc = schedule_document(normalize_timelines(data), 42)
        result = build_questions(doc, base_quota=2)
        self.assertEqual(result, build_questions(doc, base_quota=2))
        self.assertTrue(validate_questions(doc, result))
        self.assertEqual(set(result['question_summary']), set(TASKS))
        self.assertEqual(result['system_prompt'], SYSTEM_PROMPT)
        for question in result['questions']:
            self.assertEqual(question['evaluation_at'], 'trajectory_end')
            self.assertTrue(question['text'].endswith('?'))
            for instruction in ('Return', 'Preserve', 'Evidence dates:', 'Use ONLY', 'Do not'):
                self.assertNotIn(instruction, question['text'])
            self.assertTrue(question['user_prompt'].startswith('Return:'))
        broken_prompt = deepcopy(result)
        broken_prompt['questions'][0].pop('user_prompt')
        with self.assertRaises(ValueError): validate_questions(doc, broken_prompt)
        broken = deepcopy(result)
        broken['questions'][0]['evidence_audit']['prefix_checks'][0]['known_endpoints'] = {}
        with self.assertRaises(ValueError): validate_questions(doc, broken)

    def test_message_assembly_keeps_private_annotations_out(self):
        q = {'text': 'When was the start reported?', 'user_prompt': 'Return: Observation time.',
             'evaluation_at': 'trajectory_end', 'evaluation': {'reference_answers': ['PRIVATE_GOLD']},
             'reference_answer': 'PRIVATE_INTERNAL_GOLD'}
        messages = build_messages(q, 'Session time: 2025-01-02\nA report with literal {braces}.')
        self.assertEqual(messages[0], {'role': 'system', 'content': SYSTEM_PROMPT})
        self.assertIn('Evidence dates:', messages[0]['content'])
        self.assertEqual(messages[1]['content'],
                         'Session time: 2025-01-02\nA report with literal {braces}.\n\n'
                         'Return: Observation time.\n\nQUESTION: When was the start reported?')
        self.assertNotIn('PRIVATE', str(messages))
        self.assertNotIn('trajectory_end', str(messages))
        self.assertEqual(build_messages(q, '', system_prompt='custom')[0]['content'], 'custom')

if __name__ == '__main__': unittest.main()
