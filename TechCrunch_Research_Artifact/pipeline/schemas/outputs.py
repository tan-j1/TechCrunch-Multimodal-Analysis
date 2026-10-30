from __future__ import annotations

from dataclasses import dataclass, field

from typing import Any, Literal, Optional

@dataclass
class Turn:
    speaker: str
    start: float
    end: float
    text: str = ""
    word_count: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

@dataclass
class SegmentationResult:
    judge_speaker: Optional[str]
    lead_speaker: Optional[str]
    first_judge_time: Optional[float]
    pitch_intervals: list[tuple[float, float]]
    qa_intervals: list[tuple[float, float]]
    lead_all_intervals: list[tuple[float, float]]
    notes: list[str] = field(default_factory=list)

@dataclass
class FaceFrameResult:
    t: float
    source: Literal["OPENFACE"]
    face_detected: bool
    usable_face: bool
    face_area: Optional[float] = None
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    au12: Optional[float] = None
    smile: Optional[bool] = None
    eye_contact: Optional[bool] = None

@dataclass
class PoseFrameResult:
    t: float
    usable_pose: bool
    pose_detected: bool = False
    shoulder_width: Optional[float] = None
    pocket_left: Optional[bool] = None
    pocket_right: Optional[bool] = None
    pocket_any: Optional[bool] = None
    arm_spread: Optional[float] = None
    lean: Optional[float] = None
    closed_penalty: Optional[float] = None
    openness: Optional[float] = None
    wrist_speed: Optional[float] = None
