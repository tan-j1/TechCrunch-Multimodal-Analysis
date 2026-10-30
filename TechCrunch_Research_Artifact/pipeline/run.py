"""Extract multimodal measures from TechCrunch pitch videos."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

import numpy as np

from pipeline.config import RunPaths, load_config, resolve_run_paths
from pipeline.metrics import summarize
from pipeline.schemas.outputs import Turn
from pipeline.segmentation import assign_lead
from pipeline.utils.io import safe_write_json


def extract_one(video: Path, cfg, paths: RunPaths, recorded_dir: Path | None,
                coding_a_dir: Path | None, coding_b_dir: Path | None) -> dict:
    # Model and video packages are imported only by raw-video extraction.
    from pipeline.modules.asr_cloud import _asr_deepgram
    from pipeline.modules.acoustics import run_vad_silero, run_f0_pyin
    from pipeline.modules.llm_coding import classify_speaker_roles, detect_jokes
    from pipeline.modules.openface import run_openface
    from pipeline.modules.pose_mediapipe import run_pose_mediapipe
    from pipeline.modules.laughter import _audioset_ast_laughter
    from pipeline.utils.audio import extract_audio_wav, load_wav_mono
    from pipeline.utils.video import VideoSampler, get_video_id

    video_id = get_video_id(video)
    recorded = recorded_dir / f"{video_id}.json" if recorded_dir else None
    if recorded is not None and recorded.is_file():
        original = json.loads(recorded.read_text(encoding="utf-8"))
        turns = [Turn(**{k: v for k, v in t.items() if k in Turn.__dataclass_fields__})
                 for t in original["turns"]]
        seg = original["segmentation"]
    else:
        transcript = _asr_deepgram(video, video_id, cfg, paths)
        if not transcript.ok:
            raise RuntimeError(transcript.error)
        turns = transcript.turns
        roles = classify_speaker_roles(turns, video_id, cfg, paths)
        if not roles.ok:
            raise RuntimeError(roles.error)
        seg = assign_lead(turns, roles.speaker_roles)
    lead_turns = [t for t in turns if t.speaker == seg["lead_speaker"]]
    if not lead_turns or not seg.get("lead_all_intervals"):
        raise ValueError("Lead speaker must have timestamped speech intervals")

    wav = extract_audio_wav(video, paths.cache_dir / "audio" / f"{video_id}.wav", sr=cfg.audio.f0.sr)
    signal, sr = load_wav_mono(wav)
    vad = run_vad_silero(signal, sr, cfg)
    if not vad.ok:
        raise RuntimeError(vad.error)
    f0 = run_f0_pyin(signal, sr, cfg)
    if not f0.ok:
        raise RuntimeError(f0.error)
    laughter = _audioset_ast_laughter(signal, sr, cfg)
    if not laughter.ok:
        raise RuntimeError(laughter.error)

    face = run_openface(video, video_id, cfg, paths)
    if not face.ok:
        raise RuntimeError(face.error)
    with VideoSampler(video, cfg.video.sample_fps, cfg.video.resize_long_side) as sampler:
        frames = [(fr.t, fr.image_bgr) for fr in sampler.iter_frames(max_frames=cfg.video.max_frames)]
    pose = run_pose_mediapipe(frames, cfg)
    if not pose.ok:
        raise RuntimeError(pose.error)

    humor = {}
    for label, source in (("A", coding_a_dir), ("B", coding_b_dir)):
        if source:
            raw = json.loads((source / f"{video_id}.json").read_text(encoding="utf-8"))
            humor[label] = raw.get("jokes", raw) if isinstance(raw, dict) else raw
            if not isinstance(humor[label], list):
                raise ValueError(f"Humor coding {label} must contain a jokes list")
        else:
            # Separate caches preserve each saved coding output.
            coding_paths = replace(paths, cache_dir=paths.cache_dir / "humor" / label)
            result = detect_jokes(lead_turns, video_id, cfg, coding_paths)
            if not result.ok:
                raise RuntimeError(result.error)
            humor[label] = result.jokes

    evidence = {
        "meta": {"video_id": video_id, "video_filename": video.name},
        "segmentation": seg,
        "turns": [asdict(t) for t in turns],
        "face_frames": [asdict(f) for f in face.frames],
        "pose_frames": [asdict(f) for f in pose.frames],
        "gesture_events": pose.gesture_events or [],
        "audio_evidence": {
            "vad_intervals": [[s.start, s.end] for s in vad.speech_intervals],
            "f0_times_s": f0.times_s.tolist(),
            "f0_hz": [float(x) if np.isfinite(x) else None for x in f0.f0_hz],
            "laughter_intervals": [[s.start, s.end] for s in laughter.laughter_intervals],
            "vad_method": vad.method,
            "laughter_method": laughter.method,
        },
        "humor_coding": humor,
    }
    row = summarize(evidence, cfg)
    evidence["metrics"] = row
    safe_write_json(paths.per_video_dir / f"{video_id}.json", evidence)
    text_dir = paths.output_dir / "lead_transcripts"
    text_dir.mkdir(parents=True, exist_ok=True)
    (text_dir / f"{video_id}.txt").write_text("\n".join(t.text for t in sorted(lead_turns, key=lambda t: t.start)), encoding="utf-8")
    safe_write_json(text_dir / f"{video_id}.json", {
        "video_id": video_id,
        "lead_speaker": seg["lead_speaker"],
        "turns": [asdict(t) for t in sorted(lead_turns, key=lambda t: t.start)],
    })
    return row


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing .mp4 videos")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/multimodal")
    parser.add_argument("--config", type=Path, default=root / "configs/multimodal.yaml")
    parser.add_argument("--recorded-dir", type=Path, help="Per-video JSON folder with saved turns and segmentation")
    parser.add_argument("--coding-a-dir", type=Path, help="Saved humor JSON folder, each file named <video_id>.json")
    parser.add_argument("--coding-b-dir", type=Path, help="Saved alternative humor JSON folder")
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(args.input_dir, args.output_dir, cfg)
    shutil.copy2(args.config, paths.output_dir / "config_used.yaml")
    videos = sorted(args.input_dir.glob("*.mp4"))
    if not videos:
        parser.error(f"No .mp4 videos found in {args.input_dir}")
    rows = []
    errors = []
    for video in videos:
        try:
            row = extract_one(video, cfg, paths, args.recorded_dir, args.coding_a_dir, args.coding_b_dir)
            rows.append(row)
            print(f"Completed {video.name}")
        except Exception as exc:
            errors.append({"video": video.name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"Failed {video.name}: {exc}")
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (paths.output_dir / "metrics_per_video.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    safe_write_json(paths.output_dir / "run_status.json", {
        "videos": len(videos), "completed": len(rows), "errors": errors,
    })
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
