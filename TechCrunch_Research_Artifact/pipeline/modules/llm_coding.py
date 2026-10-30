from __future__ import annotations

import hashlib

import json

import os

from dataclasses import dataclass

from pathlib import Path

from typing import Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from pipeline.config import AppConfig, RunPaths

from pipeline.schemas.outputs import Turn

from pipeline.utils.io import safe_read_json, safe_write_json

from pipeline.utils.logging import get_logger

logger = get_logger(__name__)

def _hash_payload(payload: Any) -> str:
    b = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(b).hexdigest()[:16]

def _cache_path(paths: RunPaths, kind: str, video_id: str, key: str) -> Path:
    return paths.cache_dir / "llm" / kind / f"{video_id}__{key}.json"

class LLMError(RuntimeError):
    pass

def _get_openai_client():
    from openai import OpenAI

    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def _ensure_json(x: Any) -> Any:
    if isinstance(x, str):
        # 尝试严格 JSON
        return json.loads(x)
    return x

JOKE_PROMPT_TEMPLATE = """You are a rigorous research assistant. You will receive transcribed turns from a [lead speaker] in a pitch video, with start/end times and text per turn.

Task: Identify "joke attempts" (humor attempts, jokes, witty remarks) and output strict JSON (no extra text).

Criteria (follow strictly):
- joke_attempt=true: text contains obvious humor / wit / pun / self-deprecation / exaggeration / deliberate comedy (even if unsuccessful).
- Do NOT treat ordinary "nice/thanks" as jokes.

Output JSON schema (strict field names/types):
{
  "version": "v1",
  "jokes": [
    {
      "turn_index": 0,
      "start_s": 12.34,
      "end_s": 15.67,
      "joke_attempt": true,
      "joke_type": "pun|self_deprecating|sarcasm|exaggeration|callback|other",
      "confidence": 0.0,
      "evidence": "Brief quote from the text"
    }
  ],
  "notes": ["..."]
}

Input turns (JSON):
{turns_json}
"""

ROLE_CLASSIFY_PROMPT_TEMPLATE = """You are a rigorous research assistant. You will see excerpts from multiple speakers in the same pitch / road-show / startup-battlefield session.

Task: For each speaker, identify their role. Choose EXACTLY one of:
- TEAM    : the startup team / presenters who pitch the company (e.g., founder, CEO, CTO, co-founder, product lead).
- JUDGE   : investors / VCs / sharks who evaluate the deal and decide whether to invest. They typically probe the business with substantive questions about market, traction, financials, defensibility, team, valuation, etc.
- OTHER   : EVERYONE else, including:
            * host / MC / announcer / moderator / event organizer (someone who introduces the segment or runs the show but does NOT invest);
            * voice-over narrator;
            * audience members;
            * event staff;
            * unclear / unidentifiable speakers.

CRITICAL DISAMBIGUATION (READ CAREFULLY):
- A "host", "MC", or "moderator" who only welcomes the team, frames the segment, asks the team to begin, or gives a brief biographical introduction is OTHER, NOT JUDGE — even if they speak first or ask a single warm-up question.
- Only speakers whose questions look like an INVESTOR evaluating the deal (probing the business model, financials, traction, defensibility, valuation, exit, etc.) should be JUDGE.
- A voice-over / narrator that summarizes the segment is OTHER, never JUDGE.

Output strict JSON (no other text):
{
  "version": "v2",
  "speakers": [
    {"speaker": "spk0", "role": "TEAM", "confidence": 0.0, "rationale": "Brief reason in English"}
  ],
  "notes": ["..."]
}

Input speakers (JSON):
{speakers_json}
"""

@dataclass
class LLMJokeResult:
    ok: bool
    jokes: list[dict]
    raw: dict[str, Any] | None = None
    error: Optional[str] = None
    notes: list[str] | None = None

@dataclass
class LLMRoleResult:
    ok: bool
    speaker_roles: dict[str, str]
    raw: dict[str, Any] | None = None
    error: Optional[str] = None
    notes: list[str] | None = None

def _turns_to_jsonable(turns: list[Turn], max_chars_per_turn: int = 500) -> list[dict[str, Any]]:
    xs = []
    for i, t in enumerate(turns):
        txt = (t.text or "").strip()
        if len(txt) > max_chars_per_turn:
            txt = txt[:max_chars_per_turn] + "…"
        xs.append({"turn_index": i, "speaker": t.speaker, "start_s": float(t.start), "end_s": float(t.end), "text": txt})
    return xs

def _speakers_to_examples(turns: list[Turn], max_examples_per_speaker: int = 4) -> list[dict[str, Any]]:
    by = {}
    for t in turns:
        by.setdefault(t.speaker, []).append(t)
    out = []
    for spk, ts in by.items():
        # 取最长的若干条作为示例
        ts2 = sorted(ts, key=lambda x: (x.end - x.start, len((x.text or ""))), reverse=True)[:max_examples_per_speaker]
        out.append(
            {
                "speaker": spk,
                "examples": [{"start_s": float(x.start), "end_s": float(x.end), "text": (x.text or "")[:300]} for x in ts2],
            }
        )
    return out

def _llm_enabled(cfg: AppConfig) -> bool:
    return bool(cfg.llm.enabled) and bool(os.environ.get("OPENAI_API_KEY"))

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=20))
def _openai_chat_json(model: str, temperature: float, system: str, user: str) -> dict[str, Any]:
    client = _get_openai_client()
    # 强制 JSON 输出：使用 response_format={"type":"json_object"}（新 SDK 支持）
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    if not content:
        raise LLMError("OpenAI 返回空内容")
    try:
        return _ensure_json(content)
    except Exception as e:
        raise LLMError(f"JSON 解析失败: {e}")

def detect_jokes(turns: list[Turn], video_id: str, cfg: AppConfig, paths: RunPaths) -> LLMJokeResult:
    if not _llm_enabled(cfg):
        return LLMJokeResult(ok=False, jokes=[], error="LLM 不可用（OPENAI_API_KEY 未设置或 llm.enabled=false）")

    # Cache keys include the prompt, input, model, and temperature.
    payload = {"kind": "joke", "prompt_version": "v2", "turns": _turns_to_jsonable(turns),
               "model": cfg.llm.model, "temperature": cfg.llm.temperature,
               "prompt_sha256": hashlib.sha256(JOKE_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()}
    key = _hash_payload(payload)
    cpath = _cache_path(paths, "joke", video_id, key)
    if cpath.exists():
        raw = safe_read_json(cpath)
        return LLMJokeResult(ok=True, jokes=raw.get("jokes", []) or [], raw=raw, notes=raw.get("notes") or [])

    # 注意：模板中包含大量 JSON 花括号，不能用 str.format（会触发 KeyError）。
    user = JOKE_PROMPT_TEMPLATE.replace("{turns_json}", json.dumps(payload["turns"], ensure_ascii=False))
    system = "You output strict JSON only. All output must be in English."
    try:
        raw = _openai_chat_json(cfg.llm.model, cfg.llm.temperature, system, user)
        safe_write_json(cpath, raw)
        jokes = raw.get("jokes", []) or []
        return LLMJokeResult(ok=True, jokes=jokes, raw=raw, notes=raw.get("notes") or [])
    except Exception as e:
        return LLMJokeResult(ok=False, jokes=[], error=f"LLM joke 失败: {e}")

def classify_speaker_roles(turns: list[Turn], video_id: str, cfg: AppConfig, paths: RunPaths) -> LLMRoleResult:
    if not (_llm_enabled(cfg) and cfg.llm.role_classification_enabled):
        return LLMRoleResult(ok=False, speaker_roles={}, error="LLM 角色识别未启用或不可用")

    speakers = _speakers_to_examples(turns)
    payload = {"kind": "role", "prompt_version": "v2", "speakers": speakers,
               "model": cfg.llm.model, "temperature": cfg.llm.temperature,
               "prompt_sha256": hashlib.sha256(ROLE_CLASSIFY_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()}
    key = _hash_payload(payload)
    cpath = _cache_path(paths, "roles", video_id, key)
    if cpath.exists():
        raw = safe_read_json(cpath)
        roles = {x["speaker"]: x["role"] for x in (raw.get("speakers") or []) if "speaker" in x and "role" in x}
        return LLMRoleResult(ok=True, speaker_roles=roles, raw=raw, notes=raw.get("notes") or [])

    user = ROLE_CLASSIFY_PROMPT_TEMPLATE.replace("{speakers_json}", json.dumps(speakers, ensure_ascii=False))
    system = "You output strict JSON only. All output must be in English."
    try:
        raw = _openai_chat_json(cfg.llm.model, cfg.llm.temperature, system, user)
        safe_write_json(cpath, raw)
        roles = {x["speaker"]: x["role"] for x in (raw.get("speakers") or []) if "speaker" in x and "role" in x}
        return LLMRoleResult(ok=True, speaker_roles=roles, raw=raw, notes=raw.get("notes") or [])
    except Exception as e:
        return LLMRoleResult(ok=False, speaker_roles={}, error=f"LLM role 失败: {e}")
