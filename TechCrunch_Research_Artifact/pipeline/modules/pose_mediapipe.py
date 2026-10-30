from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pipeline.config import AppConfig
from pipeline.schemas.outputs import PoseFrameResult


@dataclass
class PoseOutput:
    ok: bool
    frames: list[PoseFrameResult]
    usable_pose_frames: int = 0
    pose_detected_frames: int = 0
    pose_coverage: Optional[float] = None
    gesture_events: list[dict] | None = None  # [{t, amplitude}]
    error: Optional[str] = None


def run_pose_mediapipe(video_frames_bgr: list[tuple[float, np.ndarray]], cfg: AppConfig) -> PoseOutput:
    """
    MediaPipe Pose：
    - usable_pose_frame：肩/肘/腕/髋关键点存在且 visibility > min_kpt_vis
    - Pocket：wrist 落入以髋点为中心的 ROI
    - Gesture：基于 wrist 速度阈值的事件识别（输出 events/speaking_min 在 metrics 里算）
    - Openness：arm_spread + lean - closed_penalty（z-score 在视频内部标准化）
    """
    try:
        import mediapipe as mp
    except Exception as e:
        return PoseOutput(ok=False, frames=[], error=f"mediapipe 不可用: {e}")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)

    # MediaPipe Pose landmark indices
    L_SHO, R_SHO = 11, 12
    L_ELB, R_ELB = 13, 14
    L_WRI, R_WRI = 15, 16
    L_HIP, R_HIP = 23, 24

    frames: list[PoseFrameResult] = []
    detected = 0
    usable = 0

    # 为 gesture 计算保存 wrist 轨迹
    ts: list[float] = []
    lw: list[tuple[float, float] | None] = []
    rw: list[tuple[float, float] | None] = []
    sws: list[float | None] = []  # shoulder width per frame

    for t, img_bgr in video_frames_bgr:
        h, w = img_bgr.shape[:2]
        img_rgb = img_bgr[:, :, ::-1]
        res = pose.process(img_rgb)
        if not res.pose_landmarks:
            frames.append(PoseFrameResult(t=float(t), pose_detected=False, usable_pose=False))
            ts.append(float(t))
            lw.append(None)
            rw.append(None)
            sws.append(None)
            continue

        detected += 1
        lm = res.pose_landmarks.landmark

        def ok_vis(i: int) -> bool:
            return (lm[i].visibility is not None) and (float(lm[i].visibility) >= cfg.pose.min_kpt_vis)

        needed = [L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI, L_HIP, R_HIP]
        usable_pose = all(ok_vis(i) for i in needed)

        def pt(i: int) -> tuple[float, float]:
            return (float(lm[i].x) * w, float(lm[i].y) * h)

        # shoulder width
        sho_l = pt(L_SHO)
        sho_r = pt(R_SHO)
        shoulder_width = float(np.hypot(sho_l[0] - sho_r[0], sho_l[1] - sho_r[1]))

        hip_l = pt(L_HIP)
        hip_r = pt(R_HIP)
        hip_width = float(np.hypot(hip_l[0] - hip_r[0], hip_l[1] - hip_r[1]))
        hip_width = max(hip_width, 1.0)

        wl = pt(L_WRI)
        wr = pt(R_WRI)

        # pocket ROI around each hip
        roi_w = cfg.pose.pocket_roi.width_scale * hip_width
        roi_h = cfg.pose.pocket_roi.height_scale * hip_width
        y_off = cfg.pose.pocket_roi.y_offset_scale * hip_width

        def in_roi(wrist, hip_center) -> bool:
            cx, cy = hip_center[0], hip_center[1] + y_off
            return (abs(wrist[0] - cx) <= roi_w / 2.0) and (abs(wrist[1] - cy) <= roi_h / 2.0)

        pocket_left = usable_pose and in_roi(wl, hip_l)
        pocket_right = usable_pose and in_roi(wr, hip_r)
        pocket_any = bool(pocket_left or pocket_right) if usable_pose else None

        # arm spread: wrists x-distance / shoulder_width
        arm_spread = abs(wr[0] - wl[0]) / max(1.0, shoulder_width) if usable_pose else None

        # lean: torso vector angle vs vertical (abs), normalized later
        mid_sho = ((sho_l[0] + sho_r[0]) / 2.0, (sho_l[1] + sho_r[1]) / 2.0)
        mid_hip = ((hip_l[0] + hip_r[0]) / 2.0, (hip_l[1] + hip_r[1]) / 2.0)
        dx = mid_sho[0] - mid_hip[0]
        dy = mid_hip[1] - mid_sho[1]
        lean = float(np.arctan2(abs(dx), max(1e-6, abs(dy)))) if usable_pose else None  # radians

        # closed_penalty: wrists close & in chest band
        # chest band defined relative to torso height
        torso_h = max(1.0, abs(mid_hip[1] - mid_sho[1]))
        wy = (wl[1] + wr[1]) / 2.0
        rel_y = (wy - mid_sho[1]) / torso_h  # 0 at shoulders, 1 at hips
        x_close = (abs(wr[0] - wl[0]) / max(1.0, shoulder_width)) < cfg.pose.openness.closed_cross_x_thresh
        in_chest = cfg.pose.openness.chest_y_band[0] <= rel_y <= cfg.pose.openness.chest_y_band[1]
        closed_penalty = 1.0 if (usable_pose and x_close and in_chest) else 0.0 if usable_pose else None

        if usable_pose:
            usable += 1

        frames.append(
            PoseFrameResult(
                t=float(t),
                pose_detected=True,
                usable_pose=bool(usable_pose),
                shoulder_width=float(shoulder_width) if shoulder_width else None,
                pocket_left=bool(pocket_left) if usable_pose else None,
                pocket_right=bool(pocket_right) if usable_pose else None,
                pocket_any=bool(pocket_any) if usable_pose else None,
                arm_spread=float(arm_spread) if arm_spread is not None else None,
                lean=float(lean) if lean is not None else None,
                closed_penalty=float(closed_penalty) if closed_penalty is not None else None,
                openness=None,  # 第二遍标准化后填充
                wrist_speed=None,
            )
        )

        ts.append(float(t))
        lw.append(wl if usable_pose else None)
        rw.append(wr if usable_pose else None)
        sws.append(shoulder_width if usable_pose else None)

    pose.close()

    # 第二遍：wrist speed + openness（视频内标准化）
    # wrist speed：使用 left/right 的最大速度（归一化到 shoulder_width）
    speeds: list[float | None] = [None] * len(ts)
    for i in range(1, len(ts)):
        if lw[i] is None or rw[i] is None or lw[i - 1] is None or rw[i - 1] is None or sws[i] is None:
            continue
        dt = max(1e-6, ts[i] - ts[i - 1])
        sw = max(1.0, float(sws[i]))
        dl = float(np.hypot(lw[i][0] - lw[i - 1][0], lw[i][1] - lw[i - 1][1])) / sw
        dr = float(np.hypot(rw[i][0] - rw[i - 1][0], rw[i][1] - rw[i - 1][1])) / sw
        speeds[i] = max(dl, dr) / dt

    # openness z-score components
    arm = np.array([f.arm_spread for f in frames if f.usable_pose and f.arm_spread is not None], dtype=float)
    lean = np.array([f.lean for f in frames if f.usable_pose and f.lean is not None], dtype=float)
    pen = np.array([f.closed_penalty for f in frames if f.usable_pose and f.closed_penalty is not None], dtype=float)

    def z(x: np.ndarray) -> np.ndarray:
        if x.size < 2:
            return np.zeros_like(x)
        mu = float(np.mean(x))
        sd = float(np.std(x) + 1e-8)
        return (x - mu) / sd

    z_arm = z(arm)
    z_lean = z(lean)
    z_pen = z(pen)

    # 填回 openness 与 speed
    ai = 0
    li = 0
    pi = 0
    for i, f in enumerate(frames):
        if f.usable_pose:
            arm_v = arm[ai] if arm.size else 0.0
            lean_v = lean[li] if lean.size else 0.0
            pen_v = pen[pi] if pen.size else 0.0
            openness = float(z_arm[ai] + z_lean[li] - z_pen[pi]) if arm.size and lean.size and pen.size else 0.0
            f.openness = openness
            f.wrist_speed = float(speeds[i]) if speeds[i] is not None else None
            ai += 1
            li += 1
            pi += 1
        else:
            f.openness = None
            f.wrist_speed = None

    # gesture events：threshold crossing segments -> amplitude
    events: list[dict] = []
    in_evt = False
    evt_start_idx = 0
    last_evt_t = -1e9
    for i in range(1, len(ts)):
        sp = speeds[i]
        if sp is None:
            continue
        above = sp >= cfg.pose.gesture.speed_thresh
        if above and (not in_evt) and (ts[i] - last_evt_t >= cfg.pose.gesture.min_event_gap_s):
            in_evt = True
            evt_start_idx = i
        if in_evt and (not above):
            # close event
            evt_end_idx = i
            in_evt = False
            # amplitude = max displacement in [start,end]
            if lw[evt_start_idx] is None or rw[evt_start_idx] is None or sws[evt_start_idx] is None:
                continue
            sw = max(1.0, float(sws[evt_start_idx]))
            amp = 0.0
            for j in range(evt_start_idx, evt_end_idx + 1):
                if lw[j] is None or rw[j] is None:
                    continue
                # 使用更强的一侧位移
                dl = float(np.hypot(lw[j][0] - lw[evt_start_idx][0], lw[j][1] - lw[evt_start_idx][1])) / sw
                dr = float(np.hypot(rw[j][0] - rw[evt_start_idx][0], rw[j][1] - rw[evt_start_idx][1])) / sw
                amp = max(amp, dl, dr)
            t_evt = float(ts[evt_start_idx])
            events.append({"t": t_evt, "amplitude": float(amp)})
            last_evt_t = t_evt

    pose_coverage = (usable / detected) if detected > 0 else None
    return PoseOutput(
        ok=True,
        frames=frames,
        usable_pose_frames=usable,
        pose_detected_frames=detected,
        pose_coverage=pose_coverage,
        gesture_events=events,
        error=None,
    )

