from __future__ import annotations

from dataclasses import dataclass

from typing import Optional

import numpy as np

from pipeline.config import AppConfig

from pipeline.utils.time import Interval, intersect_intervals, interval_total_seconds, merge_intervals

@dataclass
class VadOutput:
    ok: bool
    method: str  # "webrtcvad" | "silero" | "energy" | "missing"
    speech_intervals: list[Interval]
    error: Optional[str] = None

def _vad_flags_to_intervals(flags: list[bool], frame_s: float) -> list[Interval]:
    intervals: list[Interval] = []
    in_speech = False
    s0 = 0.0
    for i, f in enumerate(flags):
        t0 = i * frame_s
        t1 = (i + 1) * frame_s
        if f and not in_speech:
            in_speech = True
            s0 = t0
        if in_speech and (not f):
            in_speech = False
            intervals.append(Interval(s0, t1))
    if in_speech:
        intervals.append(Interval(s0, len(flags) * frame_s))
    return intervals

def run_vad_silero_pysilero(y: np.ndarray, sr: int, cfg: AppConfig) -> VadOutput:
    """
    Silero VAD（pysilero-vad 打包版）：
    - 优点：无需 torch hub 拉取 GitHub repo；Windows 环境更稳定
    - 依赖：`pip install pysilero-vad`
    """
    try:
        import librosa
        from pysilero_vad import SileroVoiceActivityDetector
    except Exception as e:
        return VadOutput(ok=False, method="silero", speech_intervals=[], error=f"pysilero-vad 不可用: {e}")

    target_sr = 16000
    if sr != target_sr:
        try:
            y_rs = librosa.resample(y.astype(float), orig_sr=int(sr), target_sr=target_sr)
        except Exception as e:
            return VadOutput(ok=False, method="silero", speech_intervals=[], error=f"pysilero 重采样失败: {e}")
    else:
        y_rs = y.astype(float)

    # 16-bit PCM bytes（单声道）
    y16 = np.clip(y_rs, -1.0, 1.0)
    pcm = (y16 * 32767.0).astype(np.int16).tobytes()

    vad = SileroVoiceActivityDetector()
    chunk_bytes = int(vad.chunk_bytes())
    chunk_samples = int(vad.chunk_samples())
    if chunk_bytes <= 0 or chunk_samples <= 0:
        return VadOutput(ok=False, method="silero", speech_intervals=[], error="pysilero chunk size 非法")

    n_chunks = len(pcm) // chunk_bytes
    if n_chunks <= 1:
        return VadOutput(ok=True, method="silero", speech_intervals=[])

    thr = float(cfg.audio.vad.silero_threshold)
    flags: list[bool] = []
    for i in range(n_chunks):
        b = pcm[i * chunk_bytes : (i + 1) * chunk_bytes]
        if len(b) != chunk_bytes:
            break
        try:
            p = float(vad(b))
        except Exception:
            p = 0.0
        flags.append(p >= thr)

    intervals = _vad_flags_to_intervals(flags, frame_s=float(chunk_samples) / float(target_sr))
    pad = cfg.audio.vad.pad_ms / 1000.0
    padded = [Interval(max(0.0, iv.start - pad), iv.end + pad) for iv in intervals]
    merged = merge_intervals(padded, gap=0.05)
    return VadOutput(ok=True, method="silero", speech_intervals=merged)

@dataclass
class F0Output:
    ok: bool
    f0_hz: np.ndarray
    times_s: np.ndarray
    error: Optional[str] = None

def run_f0_pyin(y: np.ndarray, sr: int, cfg: AppConfig) -> F0Output:
    try:
        import librosa
    except Exception as e:
        return F0Output(ok=False, f0_hz=np.array([]), times_s=np.array([]), error=f"librosa 不可用: {e}")

    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=cfg.audio.f0.fmin,
            fmax=cfg.audio.f0.fmax,
            sr=sr,
            hop_length=int(cfg.audio.f0.hop_length),
        )
        # times for frames
        times = librosa.times_like(f0, sr=sr, hop_length=int(cfg.audio.f0.hop_length))
        f0 = np.asarray(f0, dtype=float)
        return F0Output(ok=True, f0_hz=f0, times_s=np.asarray(times, dtype=float))
    except Exception as e:
        return F0Output(ok=False, f0_hz=np.array([]), times_s=np.array([]), error=f"pyin 失败: {e}")

def f0_semitone_std(f0_hz: np.ndarray) -> Optional[float]:
    f0 = f0_hz[np.isfinite(f0_hz) & (f0_hz > 0)]
    if f0.size < 5:
        return None
    # semitone relative to 55Hz (A1) - 相对基准不影响 std
    st = 12.0 * np.log2(f0 / 55.0)
    return float(np.std(st))

def compute_pause_count(lead_speech_intervals: list[Interval], min_silence_s: float) -> int:
    if not lead_speech_intervals:
        return 0
    xs = merge_intervals(lead_speech_intervals, gap=0.05)
    xs = sorted(xs, key=lambda z: z.start)
    pauses = 0
    for i in range(1, len(xs)):
        gap = xs[i].start - xs[i - 1].end
        if gap >= min_silence_s:
            pauses += 1
    return pauses

def mask_f0_by_intervals(times_s: np.ndarray, f0_hz: np.ndarray, intervals: list[Interval]) -> np.ndarray:
    if times_s.size == 0 or f0_hz.size == 0:
        return np.array([])
    mask = np.zeros_like(times_s, dtype=bool)
    for iv in intervals:
        mask |= (times_s >= iv.start) & (times_s < iv.end)
    return f0_hz[mask]


_SILERO_MODEL = None

_SILERO_UTILS = None

def _load_silero_vad():
    global _SILERO_MODEL, _SILERO_UTILS
    if _SILERO_MODEL is not None and _SILERO_UTILS is not None:
        return _SILERO_MODEL, _SILERO_UTILS
    import torch

    # 使用 torch hub（会自动缓存到用户目录）
    model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False)
    _SILERO_MODEL, _SILERO_UTILS = model, utils
    return model, utils

def run_vad_silero(y: np.ndarray, sr: int, cfg: AppConfig) -> VadOutput:
    """
    模型 VAD（Silero）：一般比能量阈值更稳，且不需要额外 token。
    依赖：torch（自动下载并缓存权重）。
    """
    # 先尝试 pysilero-vad（更适合 Windows/离线/不方便访问 GitHub 的环境）
    out_pkg = run_vad_silero_pysilero(y, sr, cfg)
    if out_pkg.ok:
        return out_pkg

    try:
        import librosa
    except Exception as e:
        return VadOutput(ok=False, method="silero", speech_intervals=[], error=f"silero 需要 librosa 做重采样: {e}")

    try:
        model, utils = _load_silero_vad()
        (get_speech_timestamps, _, _, _, _) = utils
    except Exception as e:
        msg = f"silero_vad(torch hub) 不可用: {e}"
        if out_pkg.error:
            msg = f"{out_pkg.error}; {msg}"
        return VadOutput(ok=False, method="silero", speech_intervals=[], error=msg)

    # Silero 推荐 16000Hz 或 8000Hz
    target_sr = 16000
    if sr != target_sr:
        try:
            y_rs = librosa.resample(y.astype(float), orig_sr=int(sr), target_sr=target_sr)
        except Exception as e:
            return VadOutput(ok=False, method="missing", speech_intervals=[], error=f"silero 重采样失败: {e}")
    else:
        y_rs = y.astype(float)

    try:
        ts = get_speech_timestamps(
            y_rs,
            model,
            sampling_rate=target_sr,
            threshold=float(cfg.audio.vad.silero_threshold),
            min_speech_duration_ms=int(cfg.audio.vad.silero_min_speech_duration_ms),
            min_silence_duration_ms=int(cfg.audio.vad.silero_min_silence_duration_ms),
        )
    except Exception as e:
        msg = f"silero 推理失败: {e}"
        if out_pkg.error:
            msg = f"{out_pkg.error}; {msg}"
        return VadOutput(ok=False, method="silero", speech_intervals=[], error=msg)

    intervals: list[Interval] = []
    for x in ts or []:
        try:
            s = float(x.get("start", 0)) / float(target_sr)
            e = float(x.get("end", 0)) / float(target_sr)
            if e > s:
                intervals.append(Interval(s, e))
        except Exception:
            continue

    pad = cfg.audio.vad.pad_ms / 1000.0
    padded = [Interval(max(0.0, iv.start - pad), iv.end + pad) for iv in intervals]
    merged = merge_intervals(padded, gap=0.05)
    return VadOutput(ok=True, method="silero", speech_intervals=merged)
