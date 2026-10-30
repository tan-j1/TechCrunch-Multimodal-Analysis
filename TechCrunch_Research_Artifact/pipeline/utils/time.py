from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class Interval:
    start: float
    end: float

    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def merge_intervals(intervals: Iterable[Interval], gap: float = 0.0) -> list[Interval]:
    xs = sorted(intervals, key=lambda z: z.start)
    if not xs:
        return []
    out: list[Interval] = [xs[0]]
    for it in xs[1:]:
        last = out[-1]
        if it.start <= last.end + gap:
            out[-1] = Interval(start=last.start, end=max(last.end, it.end))
        else:
            out.append(it)
    return out


def intersect_intervals(a: Iterable[Interval], b: Iterable[Interval]) -> list[Interval]:
    aa = sorted(a, key=lambda z: z.start)
    bb = sorted(b, key=lambda z: z.start)
    out: list[Interval] = []
    i = 0
    j = 0
    while i < len(aa) and j < len(bb):
        s = max(aa[i].start, bb[j].start)
        e = min(aa[i].end, bb[j].end)
        if e > s:
            out.append(Interval(s, e))
        if aa[i].end < bb[j].end:
            i += 1
        else:
            j += 1
    return out


def interval_total_seconds(intervals: Iterable[Interval]) -> float:
    return sum(iv.duration() for iv in intervals)


def contains_time(intervals: List[Interval], t: float) -> bool:
    # intervals 通常已 merge 且有序
    for iv in intervals:
        if t < iv.start:
            return False
        if iv.start <= t < iv.end:
            return True
    return False


def clip_intervals(intervals: Iterable[Interval], start: float, end: float) -> list[Interval]:
    out: list[Interval] = []
    for iv in intervals:
        s = max(start, iv.start)
        e = min(end, iv.end)
        if e > s:
            out.append(Interval(s, e))
    return out

