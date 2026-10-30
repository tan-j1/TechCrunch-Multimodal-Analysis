from __future__ import annotations

import math

import hashlib

import os

import shutil

import subprocess

from dataclasses import dataclass

from pathlib import Path

from typing import Optional

import numpy as np

import pandas as pd

from pipeline.config import AppConfig, RunPaths

from pipeline.schemas.outputs import FaceFrameResult

from pipeline.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class OpenFaceOutput:
    ok: bool
    frames: list[FaceFrameResult]
    source: str
    face_detected_frames: int = 0
    usable_face_frames: int = 0
    face_coverage: Optional[float] = None
    error: Optional[str] = None

def _find_openface_binary(cfg: AppConfig) -> Optional[str]:
    if cfg.openface.binary_path:
        p = Path(cfg.openface.binary_path)
        if p.exists():
            # 如果用户误配置为 OpenFaceOffline.exe，但同目录存在 FeatureExtraction.exe，则自动纠正（避免弹窗）
            if p.name.lower() == "openfaceoffline.exe":
                cand = p.parent / "FeatureExtraction.exe"
                if cand.exists():
                    return str(cand)
            return str(p)
    # 常见名称
    for name in ["FeatureExtraction", "FeatureExtraction.exe"]:
        x = shutil.which(name)
        if x:
            return x
    return None

def _windows_creationflags_no_window() -> int:
    # 防止弹出任何控制台窗口（GUI 弹窗由 OpenFaceOffline.exe 触发，本项目会禁止调用它）
    try:
        return int(getattr(subprocess, "CREATE_NO_WINDOW"))
    except Exception:
        return 0

def _deg(x: float) -> float:
    # OpenFace pose 可能为弧度；若绝对值小于 ~3.2，认为是弧度
    if abs(x) <= 3.2:
        return float(x * 180.0 / math.pi)
    return float(x)

def run_openface(video_path: Path, video_id: str, cfg: AppConfig, paths: RunPaths) -> OpenFaceOutput:
    """
    运行 OpenFace FeatureExtraction 并解析 AU/pose/bbox。
    若 OpenFace 不可用，返回 ok=False，由上层决定 fallback 或 missing。
    """
    if not cfg.openface.enabled:
        return OpenFaceOutput(ok=False, frames=[], source="OPENFACE", error="OpenFace 未启用")

    bin_path = _find_openface_binary(cfg)
    if not bin_path:
        return OpenFaceOutput(ok=False, frames=[], source="OPENFACE", error="未找到 OpenFace FeatureExtraction 可执行文件")

    # 强制只允许 CLI 的 FeatureExtraction，避免任何 OpenFaceOffline GUI 弹窗
    exe_name = Path(bin_path).name.lower()
    if "openfaceoffline" in exe_name or "openfacedemo" in exe_name or "recording" in exe_name:
        return OpenFaceOutput(
            ok=False,
            frames=[],
            source="OPENFACE",
            error=f"配置的 openface.binary_path 不是 FeatureExtraction.exe（检测到: {Path(bin_path).name}），为避免弹窗已拒绝调用。",
        )

    out_dir = paths.cache_dir / "openface" / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Windows 下 OpenFace 对包含中文/特殊字符的路径兼容性较差（输入视频路径/输出目录都可能失败且不产出 CSV）。
    # 为最大化鲁棒性：在系统临时目录（通常为 ASCII 路径）复制一份视频并将输出写到临时目录，再回拷结果。
    fingerprint = hashlib.sha256(
        f"{video_path.resolve()}|{video_path.stat().st_mtime_ns}|{video_path.stat().st_size}|{cfg.openface.process_fps}|{cfg.openface.process_resize_long_side}".encode("utf-8")
    ).hexdigest()[:16]
    temp_root = Path(os.environ.get("TEMP") or os.environ.get("TMP") or str(paths.cache_dir)) / "techcrunch_openface" / fingerprint
    temp_in = temp_root / "input.mp4"
    temp_in_proc = temp_root / "input_proc.mp4"
    temp_out = temp_root / "out"
    try:
        temp_out.mkdir(parents=True, exist_ok=True)
        if (not temp_in.exists()) or (temp_in.stat().st_size != video_path.stat().st_size):
            temp_in.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video_path, temp_in)
        run_video_path = temp_in
        run_out_dir = temp_out
    except Exception:
        # 若临时目录不可用，则回退使用原路径
        run_video_path = video_path
        run_out_dir = out_dir

    # 提速：对输入视频先做下采样/缩放再跑 OpenFace（best-effort）
    try:
        fps = cfg.openface.process_fps
        rls = cfg.openface.process_resize_long_side
        if (fps is not None and float(fps) > 0) or (rls is not None and int(rls) > 0):
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise RuntimeError("FFmpeg is required for configured OpenFace frame sampling")
            if ffmpeg and Path(run_video_path).exists():
                vf = []
                if fps is not None and float(fps) > 0:
                    vf.append(f"fps={float(fps)}")
                if rls is not None and int(rls) > 0:
                    # 保持长边为 rls，短边等比；并确保偶数（视频编码更稳）
                    s = int(rls)
                    vf.append(f"scale='if(gte(iw,ih),{s},-2)':'if(gte(iw,ih),-2,{s})'")
                cmd_pre = [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(run_video_path),
                    "-an",
                    "-vf",
                    ",".join(vf),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    str(temp_in_proc),
                ]
                p = subprocess.run(
                    cmd_pre,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=str(temp_root),
                    creationflags=_windows_creationflags_no_window() if os.name == "nt" else 0,
                )
                if p.returncode == 0 and temp_in_proc.exists() and temp_in_proc.stat().st_size > 0:
                    run_video_path = temp_in_proc
                else:
                    raise RuntimeError(f"OpenFace video resampling failed: {p.stderr[-400:]}")
    except Exception as e:
        return OpenFaceOutput(ok=False, frames=[], source="OPENFACE", error=str(e))

    # OpenFace 输出 CSV 通常放在 out_dir 里，文件名为 <video_stem>.csv
    cmd = [
        bin_path,
        "-f",
        str(run_video_path),
        "-out_dir",
        str(run_out_dir),
        "-aus",
        "-pose",
        "-2Dfp",
    ]
    try:
        # OpenFace 常使用相对路径加载 model/ 等资源；设置 cwd 为可执行文件所在目录更稳。
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(Path(bin_path).parent),
            creationflags=_windows_creationflags_no_window() if os.name == "nt" else 0,
        )
        if proc.returncode != 0:
            return OpenFaceOutput(ok=False, frames=[], source="OPENFACE", error=f"OpenFace 运行失败: {proc.stderr[-400:]}")
    except Exception as e:
        return OpenFaceOutput(ok=False, frames=[], source="OPENFACE", error=f"OpenFace 调用异常: {e}")

    # 解析 CSV
    # 不同 OpenFace 版本可能会把输出写到 out_dir 的子目录，或 CSV 文件名不完全等于 stem；
    # 因此使用递归搜索并优先选择包含关键列的 CSV。
    csv_candidates = []
    root_out = Path(run_out_dir)
    try:
        csv_candidates = list(root_out.rglob("*.csv"))
    except Exception:
        csv_candidates = []

    if not csv_candidates:
        tail = ""
        try:
            tail = (proc.stderr or "")[-1200:] if "proc" in locals() else ""
        except Exception:
            tail = ""
        msg = "OpenFace 未生成 CSV 输出"
        if tail:
            msg += f"；stderr_tail={tail}"
        return OpenFaceOutput(ok=False, frames=[], source="OPENFACE", error=msg)

    # 先尝试挑选“像 OpenFace 主 CSV”的文件：包含 timestamp/success/pose_Ry/AU12_r 等列
    def score_csv(p: Path) -> int:
        try:
            # 只读表头，避免大文件开销
            df0 = pd.read_csv(p, nrows=1)
            # OpenFace CSV 表头常带空格，统一 strip
            cols = set(str(c).strip() for c in df0.columns)
        except Exception:
            return -1
        score = 0
        for c in ["timestamp", "success", "pose_Ry", "AU12_r", "bounding_box_w", "bounding_box_h"]:
            if c in cols:
                score += 1
        return score

    scored = [(score_csv(p), p) for p in csv_candidates]
    scored = [x for x in scored if x[0] >= 0]
    if not scored:
        # 都读不出来，选第一个让后续报错更明确
        csv_path = csv_candidates[0]
    else:
        scored.sort(key=lambda x: x[0], reverse=True)
        csv_path = scored[0][1]

    # 若使用了临时目录运行，则把 CSV 复制回 cache（便于审计）
    try:
        if Path(run_out_dir) != out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(csv_path, out_dir / csv_path.name)
    except Exception:
        pass

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return OpenFaceOutput(ok=False, frames=[], source="OPENFACE", error=f"OpenFace CSV 读取失败: {e}")

    # OpenFace CSV 表头常带空格（例如 " timestamp"），统一 strip，保证后续列名匹配
    try:
        df.columns = [str(c).strip() for c in df.columns]
    except Exception:
        pass

    # 关键列兜底
    success_col = "success" if "success" in df.columns else None
    ts_col = "timestamp" if "timestamp" in df.columns else None
    au12_col = "AU12_r" if "AU12_r" in df.columns else ("AU12" if "AU12" in df.columns else None)
    yaw_col = "pose_Ry" if "pose_Ry" in df.columns else None
    pitch_col = "pose_Rx" if "pose_Rx" in df.columns else None
    roll_col = "pose_Rz" if "pose_Rz" in df.columns else None
    bb_w = "bounding_box_w" if "bounding_box_w" in df.columns else None
    bb_h = "bounding_box_h" if "bounding_box_h" in df.columns else None

    frames: list[FaceFrameResult] = []
    face_detected = 0
    usable = 0

    for i, row in df.iterrows():
        success = bool(int(row[success_col])) if success_col else True
        t = float(row[ts_col]) if ts_col else float(i) / 30.0
        area = None
        if bb_w and bb_h:
            try:
                area = float(row[bb_w]) * float(row[bb_h])
            except Exception:
                area = None

        yaw = _deg(float(row[yaw_col])) if yaw_col else None
        pitch = _deg(float(row[pitch_col])) if pitch_col else None
        roll = _deg(float(row[roll_col])) if roll_col else None

        au12 = float(row[au12_col]) if au12_col else None

        face_detected_flag = bool(success)
        if face_detected_flag:
            face_detected += 1

        usable_face = (
            face_detected_flag
            and (area is None or area > cfg.openface.min_face_area)
            and (yaw is None or abs(yaw) < cfg.openface.max_yaw_deg)
        )
        if usable_face:
            usable += 1

        smile = (au12 is not None) and (au12 > cfg.openface.tau_au12) and usable_face

        center = cfg.eye_contact.judge_yaw_center_deg
        width = cfg.eye_contact.judge_yaw_width_deg
        eye_contact = usable_face and (yaw is not None) and (abs(yaw - center) < width)

        frames.append(
            FaceFrameResult(
                t=t,
                source="OPENFACE",
                face_detected=face_detected_flag,
                usable_face=usable_face,
                face_area=area,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                au12=au12,
                smile=bool(smile) if usable_face and au12 is not None else None,
                eye_contact=bool(eye_contact) if usable_face and yaw is not None else None,
            )
        )

    face_coverage = (usable / face_detected) if face_detected > 0 else None
    return OpenFaceOutput(
        ok=True,
        frames=frames,
        source="OPENFACE",
        face_detected_frames=face_detected,
        usable_face_frames=usable,
        face_coverage=face_coverage,
        error=None,
    )
