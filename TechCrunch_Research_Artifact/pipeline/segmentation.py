"""Assign the lead speaker using speaker roles and speaking duration."""
from __future__ import annotations

from collections import defaultdict

from pipeline.schemas.outputs import Turn
from pipeline.utils.time import Interval, merge_intervals


def assign_lead(turns: list[Turn], roles: dict[str, str]) -> dict:
    judges = {s for s, role in roles.items() if role == "JUDGE"}
    judge_turns = [t for t in turns if t.speaker in judges]
    first_judge = min(judge_turns, key=lambda t: t.start).speaker if judge_turns else None
    durations: dict[str, float] = defaultdict(float)
    for t in turns:
        durations[t.speaker] += max(0.0, t.end - t.start)
    candidates = [(s, d) for s, d in durations.items() if s != first_judge]
    if not candidates:
        raise ValueError("No speaker available for lead assignment")
    lead = sorted(candidates, key=lambda item: item[1], reverse=True)[0][0]
    spans = merge_intervals([Interval(t.start, t.end) for t in turns if t.speaker == lead], gap=0.05)
    return {
        "lead_speaker": lead,
        "judge_speaker": first_judge,
        "first_judge_time": min((t.start for t in judge_turns), default=None),
        "lead_all_intervals": [[s.start, s.end] for s in spans],
        "speaker_roles": roles,
    }
