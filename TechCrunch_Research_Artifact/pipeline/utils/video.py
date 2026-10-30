from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2


@dataclass
class Frame:
    t: float  # seconds
    image_bgr: "cv2.Mat"
    frame_idx: int


def _resize_long_side(img_bgr, long_side: int) -> "cv2.Mat":
    h, w = img_bgr.shape[:2]
    if max(h, w) <= long_side:
        return img_bgr
    if h >= w:
        new_h = long_side
        new_w = int(w * (long_side / h))
    else:
        new_w = long_side
        new_h = int(h * (long_side / w))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


class VideoSampler:
    """
    顺序采样（推荐）：按 sample_fps 从头到尾遍历视频，返回 (t, frame)。
    优点：不依赖随机 seek，鲁棒且对 mp4 兼容性更好。
    """

    def __init__(self, video_path: Path, sample_fps: float, resize_long_side: Optional[int] = 640):
        self.video_path = Path(video_path)
        self.sample_fps = float(sample_fps)
        self.resize_long_side = resize_long_side
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频: {self.video_path}")
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration_s = self.frame_count / self.src_fps if self.frame_count > 0 else None

        self._next_sample_t = 0.0

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass

    def __enter__(self) -> "VideoSampler":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def iter_frames(self, max_frames: Optional[int] = None) -> Iterator[Frame]:
        step = 1.0 / max(1e-6, self.sample_fps)
        idx = 0
        yielded = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            t = idx / self.src_fps
            if t + 1e-6 >= self._next_sample_t:
                if self.resize_long_side is not None:
                    frame = _resize_long_side(frame, self.resize_long_side)
                yield Frame(t=t, image_bgr=frame, frame_idx=idx)
                yielded += 1
                self._next_sample_t += step
                if max_frames is not None and yielded >= max_frames:
                    break
            idx += 1


def get_video_id(video_path: Path) -> str:
    # 保留空格但更安全：替换为下划线
    return video_path.stem.replace(" ", "_")

