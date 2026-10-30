from __future__ import annotations

from dataclasses import dataclass

from typing import Optional

import numpy as np

from pipeline.config import AppConfig

from pipeline.utils.time import Interval, intersect_intervals, interval_total_seconds, merge_intervals

@dataclass
class LaughterOutput:
    ok: bool
    method: str  # "model" | "heuristic" | "missing"
    laughter_intervals: list[Interval]
    error: Optional[str] = None

_HF_AST_PIPES: dict[tuple[str, int], object] = {}

def _resolve_device(device_cfg: str) -> int:
    d = str(device_cfg).strip().lower()
    if d == "auto":
        try:
            import torch

            return 0 if torch.cuda.is_available() else -1
        except Exception:
            return -1
    try:
        return int(d)
    except Exception:
        return -1

def _get_ast_pipeline(model_id: str, device_cfg: str):
    device = _resolve_device(device_cfg)
    key = (str(model_id), int(device))
    if key in _HF_AST_PIPES:
        return _HF_AST_PIPES[key]
    try:
        from transformers import pipeline
    except Exception as e:
        msg = str(e)
        # 常见：transformers 安装不完整/缺依赖
        if "mpmath" in msg.lower():
            raise RuntimeError("缺少依赖 mpmath（请先运行：pip install -U mpmath transformers）") from e
        raise

    # CPU 推理；HuggingFace 会自动缓存权重
    pipe = pipeline(task="audio-classification", model=model_id, device=device)
    _HF_AST_PIPES[key] = pipe
    return pipe

def _audioset_ast_laughter(y: np.ndarray, sr: int, cfg: AppConfig) -> LaughterOutput:
    """Detect laughter with windowed AudioSet AST scores and thresholded segments."""
    try:
        import librosa
    except Exception as e:
        return LaughterOutput(ok=False, method="missing", laughter_intervals=[], error=f"audioset_ast 需要 librosa: {e}")

    try:
        pipe = _get_ast_pipeline(cfg.laughter.audioset_ast_model_id, getattr(cfg.laughter, "audioset_ast_device", "auto"))
    except Exception as e:
        msg = str(e)
        if "mpmath" in msg.lower():
            msg = "缺少依赖 mpmath（请运行：pip install -U mpmath transformers）"
        return LaughterOutput(ok=False, method="missing", laughter_intervals=[], error=f"transformers/AST 加载失败: {msg}")

    # AST 通常以 16k 输入最稳
    target_sr = 16000
    if sr != target_sr:
        try:
            y_rs = librosa.resample(y.astype(float), orig_sr=int(sr), target_sr=target_sr)
        except Exception as e:
            return LaughterOutput(ok=False, method="missing", laughter_intervals=[], error=f"audioset_ast 重采样失败: {e}")
    else:
        y_rs = y.astype(float)

    win_s = float(cfg.laughter.audioset_ast_window_s)
    hop_s = float(cfg.laughter.audioset_ast_hop_s)
    if win_s <= 0 or hop_s <= 0:
        return LaughterOutput(ok=False, method="missing", laughter_intervals=[], error="audioset_ast window/hop 非法")

    win_n = int(win_s * target_sr)
    hop_n = int(hop_s * target_sr)
    if win_n <= 0 or hop_n <= 0:
        return LaughterOutput(ok=False, method="missing", laughter_intervals=[], error="audioset_ast window/hop 样本数非法")

    n = int(len(y_rs))
    if n < max(1, win_n // 2):
        return LaughterOutput(ok=True, method="audioset_ast", laughter_intervals=[])

    # 逐 hop 滑窗打分
    keywords = [k.lower() for k in (cfg.laughter.audioset_ast_label_keywords or ["laughter"])]
    thr = float(cfg.laughter.audioset_ast_threshold)
    agg = str(getattr(cfg.laughter, "audioset_ast_score_agg", "max") or "max").strip().lower()
    scores: list[float] = []
    times: list[float] = []

    inputs = []
    starts = []
    for start in range(0, max(1, n - win_n + 1), hop_n):
        seg = y_rs[start : start + win_n]
        if seg.size < win_n:
            break
        inputs.append({"array": seg.astype(np.float32), "sampling_rate": target_sr})
        starts.append(int(start))

    if not inputs:
        return LaughterOutput(ok=True, method="audioset_ast", laughter_intervals=[])

    # 批量推理：显著减少 pipeline 调用开销
    bs = int(getattr(cfg.laughter, "audioset_ast_batch_size", 8) or 8)
    bs = max(1, min(64, bs))
    top_k = int(getattr(cfg.laughter, "audioset_ast_top_k", 20) or 20)
    top_k = max(5, min(100, top_k))
    for i in range(0, len(inputs), bs):
        batch = inputs[i : i + bs]
        batch_starts = starts[i : i + bs]
        try:
            preds_batch = pipe(batch, top_k=top_k)
        except TypeError:
            preds_batch = pipe(batch)

        # transformers 在 batch 模式下返回 list[list[dict]]
        if isinstance(preds_batch, dict):
            preds_batch = [preds_batch]

        for j, preds in enumerate(preds_batch or []):
            t0 = float(batch_starts[j]) / float(target_sr)
            hit_scores: list[float] = []
            for it in preds or []:
                lab = str(it.get("label", "")).lower()
                sc = float(it.get("score", 0.0))
                if any(k in lab for k in keywords):
                    hit_scores.append(sc)

            p_max = max(hit_scores) if hit_scores else 0.0
            p_sum = float(min(1.0, sum(hit_scores))) if hit_scores else 0.0
            p = p_sum if agg == "sum" else p_max

            scores.append(float(p))
            times.append(float(t0))

    if not scores:
        return LaughterOutput(ok=True, method="audioset_ast", laughter_intervals=[])

    above = np.array(scores, dtype=float) >= thr

    intervals: list[Interval] = []
    in_seg = False
    s0 = 0.0
    for i, a in enumerate(above):
        t0 = times[i]
        t1 = t0 + hop_s
        if a and (not in_seg):
            in_seg = True
            s0 = t0
        if in_seg and (not a):
            in_seg = False
            intervals.append(Interval(s0, t1))
    if in_seg:
        intervals.append(Interval(s0, times[-1] + hop_s))

    merged = merge_intervals(intervals, gap=float(cfg.laughter.audioset_ast_min_gap_s))
    min_len = float(cfg.laughter.audioset_ast_min_segment_s)
    merged = [iv for iv in merged if (iv.end - iv.start) >= min_len]
    return LaughterOutput(ok=True, method="audioset_ast", laughter_intervals=merged)

def laughter_time_in(intervals: list[Interval], within: list[Interval]) -> float:
    return interval_total_seconds(intersect_intervals(intervals, within))
