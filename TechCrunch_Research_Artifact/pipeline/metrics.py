"""Aggregate video-level measures from timestamped evidence."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from pipeline.config import AppConfig
from pipeline.modules.acoustics import compute_pause_count, f0_semitone_std, mask_f0_by_intervals
from pipeline.utils.time import Interval, intersect_intervals, interval_total_seconds, merge_intervals


def _spans(pairs: list) -> list[Interval]:
    return [Interval(float(a), float(b)) for a, b in pairs if float(b) > float(a)]


def _in(t: float, spans: list[Interval]) -> bool:
    return any(s.start <= t < s.end for s in spans)


def _ratio(n: float, d: float) -> float | None:
    return float(n / d) if d > 0 else None


def count_humor(jokes: list[dict], lead: list[Interval]) -> int:
    count = 0
    for joke in jokes:
        if not joke.get("joke_attempt", False):
            continue
        try:
            end = float(joke.get("end_s", joke.get("end", 0.0)))
        except (ValueError, TypeError):
            continue
        if any(s.start <= end <= s.end + 1e-6 for s in lead):
            count += 1
    return count


def summarize(evidence: dict[str, Any], cfg: AppConfig) -> dict[str, Any]:
    seg = evidence["segmentation"]
    lead = _spans(seg.get("lead_all_intervals", []))
    speaker = seg.get("lead_speaker")
    turns = [t for t in evidence.get("turns", []) if t.get("speaker") == speaker
             and any(float(t["end"]) > s.start and float(t["start"]) < s.end for s in lead)]
    faces = [f for f in evidence.get("face_frames", [])
             if f.get("usable_face") and _in(float(f["t"]), lead)]
    poses = [f for f in evidence.get("pose_frames", [])
             if f.get("usable_pose") and _in(float(f["t"]), lead)]
    events = [e for e in evidence.get("gesture_events", []) if _in(float(e["t"]), lead)]
    openness = [float(p["openness"]) for p in poses
                if p.get("openness") is not None and math.isfinite(float(p["openness"]))]
    out: dict[str, Any] = {
        "video_id": (evidence.get("meta") or {}).get("video_id", evidence.get("video_id")),
        "lead_speaker": speaker,
        "usable_face_frames": len(faces),
        "smile_frames": sum(bool(f.get("smile")) for f in faces),
        "smile_duration_ratio": _ratio(sum(bool(f.get("smile")) for f in faces), len(faces)),
        "eye_contact_ratio": _ratio(sum(bool(f.get("eye_contact")) for f in faces), len(faces)),
        "usable_pose_frames": len(poses),
        "pocket_hands_ratio": _ratio(sum(bool(p.get("pocket_any")) for p in poses), len(poses)),
        "body_openness_mean": float(np.mean(openness)) if openness else None,
        "hand_gesture_events": len(events),
        "words": sum(int(t.get("word_count", 0)) for t in turns),
    }
    # Audio evidence carries original-time VAD, F0 and detected laughter spans.
    audio = evidence.get("audio_evidence")
    speaking_s = None
    if audio is not None:
        speech = intersect_intervals(lead, _spans(audio.get("vad_intervals", [])))
        speaking_s = interval_total_seconds(speech)
        out["lead_speaking_time_s"] = speaking_s
        out["speech_rate_wpm"] = _ratio(out["words"] * 60, speaking_s)
        out["hand_gesture_frequency_per_min"] = _ratio(len(events) * 60, speaking_s)
        pauses = compute_pause_count(speech, cfg.audio.pause.min_silence_s)
        out["pause_count"] = pauses
        out["pause_per_100_words"] = _ratio(pauses * 100, out["words"])
        f0 = np.asarray([np.nan if x is None else x for x in audio.get("f0_hz", [])], dtype=float)
        times = np.asarray(audio.get("f0_times_s", []), dtype=float)
        if len(f0) != len(times):
            raise ValueError("F0 values and times must have equal length")
        out["vocal_pitch_var_semitone_std"] = f0_semitone_std(mask_f0_by_intervals(times, f0, speech))
        extended = merge_intervals([Interval(s.start, s.end + cfg.laughter.post_speech_window_s)
                                    for s in speech], gap=0.05)
        laugh_s = interval_total_seconds(intersect_intervals(_spans(audio.get("laughter_intervals", [])), extended))
        out["laughter_duration_ratio"] = _ratio(laugh_s, speaking_s)
    coding = evidence.get("humor_coding") or {}
    for version, jokes in coding.items():
        n = count_humor(jokes, lead)
        out[f"humor_{version}_attempts"] = n
        if speaking_s is not None:
            out[f"humor_{version}_per_min"] = _ratio(n * 60, speaking_s)
    if all(out.get(f"humor_{v}_per_min") is not None for v in ("A", "B")):
        out["humor_mean_per_min"] = (out["humor_A_per_min"] + out["humor_B_per_min"]) / 2
    return out
