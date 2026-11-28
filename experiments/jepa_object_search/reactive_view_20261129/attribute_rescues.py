"""Pair previous and extended paths request by request and attribute only pre-confirmation events."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def breakdown(records):
    counts = Counter()
    increments = []
    for row in records:
        events = row['events']
        t = row['new_confirmation']
        before = row['baseline_length']
        if t is None or t <= before or not events[t]['correct']:
            continue
        increments.append(t-before)
        counts['rescued'] += 1
        previously_clear = any(e['px'] >= 369 for e in events[:before+1])
        counts['initial_route_already_had_clear_opportunity' if previously_clear else 'clear_opportunity_first_obtained_on_extension'] += 1
        decisions = [r for r in row['replans'] if r['t'] < t]
        counts['first_extension_after_' + decisions[0]['trigger']] += 1
        counts['final_view_decision_' + decisions[-1]['details']['reason']] += 1
        if any(r['trigger'] == 'contact_or_stall' for r in decisions):
            counts['at_least_one_replan_due_to_contact'] += 1
        if any(r['details'].get('local_distance_m', 0) > 1.5 for r in decisions):
            counts['selected_farther_than_1_5m'] += 1
        if any(abs(r['details'].get('relative_bearing_deg', 0)) > 90 for r in decisions):
            counts['selected_lateral_direction_over_90deg'] += 1
    return {'counts': dict(counts), 'extra_actions_until_valid_confirmation': {
        'min': int(min(increments)), 'median': float(np.median(increments)), 'max': int(max(increments))}}


def missed_views(records):
    """Describe motion only for requests with neither a clear frame nor a submission."""
    totals = Counter()
    motion = []
    for row in records:
        ev = row['events']
        if row['new_confirmation'] is not None or any(e['px'] >= 369 for e in ev):
            continue
        knocks = sum(e['bump'] for e in ev)
        used = len(ev)-1
        left = row['budget']-used
        totals['groups'] += 1
        totals['touched_or_stalled'] += int(knocks > 0)
        totals['touched_or_stalled_at_least_three_times'] += int(knocks >= 3)
        totals['ended_with_at_least_two_unused_actions'] += int(left >= 2)
        totals['peak_target_area_under_half_the_required_pixels'] += int(max(e['px'] for e in ev) < 369/2)
        if row['replans'] and not row['replans'][-1]['actions']:
            totals['terminal_' + row['replans'][-1]['details']['reason']] += 1
        elif used == row['budget']:
            totals['terminal_budget_exhausted'] += 1
        motion.append((knocks, max(e['px'] for e in ev), left))
    medians = np.median(motion,axis=0)
    return {'counts':dict(totals), 'medians':dict(zip(('contacts_per_request','best_target_pixels','unused_actions'),medians.tolist()))}


if __name__ == '__main__':
    records = [json.loads(p.read_text()) for p in sorted((HERE/'private_roll').glob('*/*.json'))]
    focus = [r for r in records if r['focus']]
    assert len(records) == 180 and len(focus) == 94
    result = {'focus': breakdown(focus), 'all': breakdown(records), 'focus_missed_views':missed_views(focus)}
    assert result['focus']['counts']['rescued'] == 10 and result['all']['counts']['rescued'] == 24
    (HERE/'rescue_attribution.json').write_text(json.dumps(result, indent=2) + '\n')
    print(result)
