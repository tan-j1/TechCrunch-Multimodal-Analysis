from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def extract_audio_wav(
    video_path: Path,
    wav_path: Path,
    sr: int = 16000,
    overwrite: bool = False,
    clip_seconds: Optional[float] = None,
    clip_start_s: Optional[float] = None,
) -> Path:
    """
    使用 ffmpeg 将视频音频抽取为单声道 wav。
    """
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    if wav_path.exists() and not overwrite:
        return wav_path
    cmd = ["ffmpeg", "-y"]
    # 精确起见：把 -ss 放在 -i 之后（慢一点但更准）；这里只取短片段，开销可接受
    cmd += ["-i", str(video_path)]
    if clip_start_s is not None and float(clip_start_s) >= 0:
        cmd += ["-ss", str(float(clip_start_s))]
    if clip_seconds is not None and float(clip_seconds) > 0:
        cmd += ["-t", str(float(clip_seconds))]
    cmd += [
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "wav",
        str(wav_path),
    ]
    # 不抛出 stdout/stderr，便于日志
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 抽音频失败: {proc.stderr[-500:]}")
    return wav_path


def load_wav_mono(wav_path: Path) -> tuple[np.ndarray, int]:
    """
    读取单声道 wav。
    - 优先 soundfile（若已安装）
    - 否则 fallback 到 librosa（保证不因 soundfile 缺失而在 import 阶段崩溃）
    """
    try:
        import soundfile as sf  # type: ignore

        y, sr = sf.read(str(wav_path))
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(axis=1)
        y = np.asarray(y, dtype=np.float32)
        return y, int(sr)
    except Exception:
        # fallback
        import librosa  # type: ignore

        y, sr = librosa.load(str(wav_path), sr=None, mono=True)
        y = np.asarray(y, dtype=np.float32)
        return y, int(sr)


def slice_audio(y: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    s = max(0, int(round(start_s * sr)))
    e = min(len(y), int(round(end_s * sr)))
    if e <= s:
        return np.zeros((0,), dtype=np.float32)
    return y[s:e]

