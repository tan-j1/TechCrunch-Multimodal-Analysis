from __future__ import annotations

import hashlib

import os

import time

from dataclasses import dataclass

from pathlib import Path

from typing import Any, Optional

import requests

from requests.exceptions import ProxyError, SSLError, Timeout

from pipeline.config import AppConfig, RunPaths

from pipeline.schemas.outputs import Turn

from pipeline.utils.audio import extract_audio_wav

from pipeline.utils.io import safe_read_json, safe_write_json

from pipeline.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class ASROutput:
    ok: bool
    provider: Optional[str]
    turns: list[Turn]
    raw: dict[str, Any] | None = None
    error: Optional[str] = None

def _cache_key(video_path: Path, provider: str, cfg: AppConfig) -> str:
    h = hashlib.sha1()
    h.update(str(video_path.resolve()).encode("utf-8"))
    h.update(str(video_path.stat().st_mtime_ns).encode("utf-8"))
    h.update(provider.encode("utf-8"))
    h.update(cfg.asr.language.encode("utf-8"))
    h.update(str(cfg.asr.enable_diarize_in_cloud).encode("utf-8"))
    return h.hexdigest()[:16]

def _cache_path(paths: RunPaths, video_id: str, provider: str, key: str) -> Path:
    return paths.cache_dir / "asr" / f"{video_id}__{provider}__{key}.json"

def _asr_deepgram(video_path: Path, video_id: str, cfg: AppConfig, paths: RunPaths) -> ASROutput:
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        return ASROutput(ok=False, provider="deepgram", turns=[], error="DEEPGRAM_API_KEY 未设置")

    key = _cache_key(video_path, "deepgram", cfg)
    cpath = _cache_path(paths, video_id, "deepgram", key)
    if cpath.exists():
        raw = safe_read_json(cpath)
        turns = _parse_deepgram_to_turns(raw)
        return ASROutput(ok=True, provider="deepgram", turns=turns, raw=raw)

    # Deepgram listen API
    url = "https://api.deepgram.com/v1/listen"
    params = {
        "model": "nova-2",
        "smart_format": "true",
        "punctuate": "true",
        "diarize": "true" if cfg.asr.enable_diarize_in_cloud else "false",
        "language": cfg.asr.language,
    }
    headers = {"Authorization": f"Token {api_key}"}
    # 大文件直接 read_bytes 容易导致“write operation timed out”，改为流式上传；
    # 同时显式设置 connect/read timeout，避免上传阶段写入超时。
    timeout = (30, 900)  # connect=30s, read=900s

    # Windows 上 requests 可能会从系统代理设置读取代理，导致 ProxyError。
    # Use a direct connection for uploads.
    session = requests.Session()
    session.trust_env = False

    # 优先上传 wav（更小更稳）；若 ffmpeg 不可用则回退上传 mp4。
    upload_path = video_path
    content_type = "video/mp4"
    try:
        # 只要系统能调用 ffmpeg，就抽取音频用于 ASR
        wav_path = paths.cache_dir / "audio" / f"{video_id}__deepgram.wav"
        extract_audio_wav(video_path, wav_path, sr=int(cfg.audio.f0.sr), overwrite=False)
        if wav_path.exists() and wav_path.stat().st_size > 0:
            upload_path = wav_path
            content_type = "audio/wav"
    except Exception as e:
        logger.info("Deepgram: 抽音频失败，回退上传 mp4: %s", e)

    for attempt in range(cfg.asr.max_retries):
        try:
            with upload_path.open("rb") as f:
                r = session.post(
                    url,
                    params=params,
                    headers={**headers, "Content-Type": content_type, "Accept": "application/json"},
                    data=f,
                    timeout=timeout,
                )
            if r.status_code >= 400:
                raise RuntimeError(f"Deepgram HTTP {r.status_code}: {r.text[:300]}")
            raw = r.json()
            safe_write_json(cpath, raw)
            turns = _parse_deepgram_to_turns(raw)
            return ASROutput(ok=True, provider="deepgram", turns=turns, raw=raw)
        except (Timeout, ProxyError, SSLError) as e:
            logger.warning("Deepgram ASR 网络/代理/SSL 失败 attempt=%d: %s", attempt + 1, e)
            time.sleep(min(10, 2 ** attempt))
        except Exception as e:
            logger.warning("Deepgram ASR 失败 attempt=%d: %s", attempt + 1, e)
            time.sleep(min(10, 2 ** attempt))
    return ASROutput(ok=False, provider="deepgram", turns=[], error="Deepgram ASR 多次失败")

def _parse_deepgram_to_turns(raw: dict[str, Any]) -> list[Turn]:
    """
    Deepgram JSON -> Turn 列表（speaker turns）。
    若 diarize 未开启或无 speaker 信息，则 speaker=spk0。
    """
    turns: list[Turn] = []
    try:
        alt = raw["results"]["channels"][0]["alternatives"][0]
        words = alt.get("words", [])
        # words: [{word,start,end,speaker,...}]
        if not words:
            # fallback：用 transcript 作为单一 turn
            txt = alt.get("transcript", "") or ""
            turns.append(Turn(speaker="spk0", start=0.0, end=0.0, text=txt, word_count=len(txt.split())))
            return turns

        # 按 speaker + 连续性聚合
        cur_spk = None
        cur_start = None
        cur_end = None
        cur_words: list[str] = []
        last_end = None
        for w in words:
            spk = str(w.get("speaker", "spk0"))
            s = float(w.get("start", 0.0))
            e = float(w.get("end", s))
            token = str(w.get("punctuated_word") or w.get("word") or "").strip()
            # 新 turn 条件：speaker 变或时间间隔过大
            if cur_spk is None:
                cur_spk, cur_start, cur_end, cur_words, last_end = spk, s, e, [token], e
                continue
            gap = (s - (last_end or s))
            if spk != cur_spk or gap > 1.0:
                txt = " ".join([x for x in cur_words if x])
                turns.append(Turn(speaker=f"spk{cur_spk}", start=float(cur_start), end=float(cur_end), text=txt, word_count=len(txt.split())))
                cur_spk, cur_start, cur_end, cur_words = spk, s, e, [token]
            else:
                cur_end = max(float(cur_end), e)
                cur_words.append(token)
            last_end = e
        if cur_spk is not None:
            txt = " ".join([x for x in cur_words if x])
            turns.append(Turn(speaker=f"spk{cur_spk}", start=float(cur_start), end=float(cur_end), text=txt, word_count=len(txt.split())))
    except Exception:
        # 极端兜底：不让 pipeline 崩
        try:
            transcript = raw["results"]["channels"][0]["alternatives"][0].get("transcript", "") or ""
        except Exception:
            transcript = ""
        turns = [Turn(speaker="spk0", start=0.0, end=0.0, text=transcript, word_count=len(transcript.split()))]
    return turns
