"""Parameters used by the TechCrunch multimodal extraction workflow."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    cache_dir: str = "cache"
    per_video_dirname: str = "per_video"
    log_level: str = "INFO"


class VideoConfig(BaseModel):
    sample_fps: float = 5.0
    max_frames: Optional[int] = None
    resize_long_side: Optional[int] = 640


class OpenFaceConfig(BaseModel):
    enabled: bool = True
    binary_path: Optional[str] = None
    process_fps: float = 5.0
    process_resize_long_side: int = 640
    tau_au12: float = 1.0
    min_face_area: float = 2500.0
    max_yaw_deg: float = 25.0


class EyeContactConfig(BaseModel):
    judge_yaw_center_deg: float = -35.0
    judge_yaw_width_deg: float = 25.0


class PosePocketROIConfig(BaseModel):
    width_scale: float = 0.6
    height_scale: float = 0.6
    y_offset_scale: float = 0.15


class GestureConfig(BaseModel):
    speed_thresh: float = 2.0
    min_event_gap_s: float = 0.4


class OpennessConfig(BaseModel):
    closed_cross_x_thresh: float = 0.10
    chest_y_band: tuple[float, float] = (0.25, 0.65)


class PoseConfig(BaseModel):
    min_kpt_vis: float = 0.6
    pocket_roi: PosePocketROIConfig = Field(default_factory=PosePocketROIConfig)
    gesture: GestureConfig = Field(default_factory=GestureConfig)
    openness: OpennessConfig = Field(default_factory=OpennessConfig)


class VadConfig(BaseModel):
    silero_threshold: float = 0.5
    silero_min_speech_duration_ms: int = 250
    silero_min_silence_duration_ms: int = 100
    pad_ms: int = 300


class F0Config(BaseModel):
    fmin: float = 50
    fmax: float = 400
    hop_length: int = 256
    sr: int = 16000


class PauseConfig(BaseModel):
    min_silence_s: float = 0.5


class AudioConfig(BaseModel):
    vad: VadConfig = Field(default_factory=VadConfig)
    f0: F0Config = Field(default_factory=F0Config)
    pause: PauseConfig = Field(default_factory=PauseConfig)


class ASRConfig(BaseModel):
    language: str = "en"
    enable_diarize_in_cloud: bool = True
    max_retries: int = 3


class LLMConfig(BaseModel):
    enabled: bool = True
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    role_classification_enabled: bool = True


class LaughterConfig(BaseModel):
    audioset_ast_model_id: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    audioset_ast_device: str = "auto"
    audioset_ast_window_s: float = 1.0
    audioset_ast_hop_s: float = 0.5
    audioset_ast_threshold: float = 0.1
    audioset_ast_score_agg: str = "sum"
    audioset_ast_top_k: int = 20
    audioset_ast_min_segment_s: float = 0.2
    audioset_ast_min_gap_s: float = 0.15
    audioset_ast_batch_size: int = 8
    post_speech_window_s: float = 3.0
    audioset_ast_label_keywords: list[str] = Field(
        default_factory=lambda: ["laugh", "laughter", "giggle", "chuckle", "chortle", "snicker"]
    )


class AppConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    openface: OpenFaceConfig = Field(default_factory=OpenFaceConfig)
    eye_contact: EyeContactConfig = Field(default_factory=EyeContactConfig)
    pose: PoseConfig = Field(default_factory=PoseConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    laughter: LaughterConfig = Field(default_factory=LaughterConfig)


def load_config(path: str | Path) -> AppConfig:
    p = Path(path).resolve()
    cfg = AppConfig.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    # A relative cache location belongs to this config's parent artifact folder.
    cache = Path(os.path.expandvars(cfg.project.cache_dir)).expanduser()
    if not cache.is_absolute():
        cache = p.parent.parent / cache
    cfg.project.cache_dir = str(cache)
    binary = os.environ.get("OPENFACE_BINARY") or cfg.openface.binary_path
    if binary:
        b = Path(os.path.expandvars(binary)).expanduser()
        cfg.openface.binary_path = str(b if b.is_absolute() else p.parent / b)
    return cfg


@dataclass(frozen=True)
class RunPaths:
    input_dir: Path
    output_dir: Path
    cache_dir: Path
    per_video_dir: Path


def resolve_run_paths(input_dir: str | Path, output_dir: str | Path, cfg: AppConfig) -> RunPaths:
    output = Path(output_dir).resolve()
    cache = Path(cfg.project.cache_dir).resolve()
    per_video = output / cfg.project.per_video_dirname
    for p in (output, cache, per_video):
        p.mkdir(parents=True, exist_ok=True)
    return RunPaths(Path(input_dir).resolve(), output, cache, per_video)
