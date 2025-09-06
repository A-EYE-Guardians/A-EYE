from __future__ import annotations
import cv2
from typing import Optional, Tuple

class BaseGaze:
    def attach(self, window_name: str): ...
    def get(self, frame_w: int, frame_h: int) -> Tuple[int, int]: ...

class MouseGaze(BaseGaze):
    def __init__(self):
        self._x: Optional[int] = None
        self._y: Optional[int] = None

    def _cb(self, event, x, y, flags, param):
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            self._x, self._y = x, y

    def attach(self, window_name: str):
        cv2.setMouseCallback(window_name, self._cb)

    def get(self, w: int, h: int) -> Tuple[int, int]:
        if self._x is None or self._y is None:
            self._x, self._y = w//2, h//2
        return int(self._x), int(self._y)

class EyeGaze(BaseGaze):
    """
    이후 팀에서 eye_tracking/api.py에 get_gaze(w,h)->(x,y)만 구현하면 바로 대체됩니다.
    """
    def __init__(self):
        try:
            from eye_tracking.api import get_gaze
            self._fn = get_gaze
        except Exception:
            self._fn = None

    def attach(self, window_name: str): pass

    def get(self, w: int, h: int) -> Tuple[int, int]:
        if self._fn:
            try:
                x, y = self._fn(w, h)
                return int(x), int(y)
            except Exception:
                pass
        return w//2, h//2  # 폴백

def build_gaze(source: str) -> BaseGaze:
    s = source.lower()
    if s in ("mouse", "cursor"): return MouseGaze()
    if s in ("eye", "eyetracking", "eye-tracking"): return EyeGaze()
    raise ValueError(f"Unknown gaze source: {source}")
