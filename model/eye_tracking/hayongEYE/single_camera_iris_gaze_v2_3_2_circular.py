#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
single_camera_iris_gaze_v2_3_circular.py

v2.3(원형-헤드블렌드):
- 홍채 특징(vx,vy)을 눈-정렬 좌표계에서 극좌표(r,θ)로 변환하여 '원형'으로 제한/강조
  * r-클리핑(soft/hard), 라디얼 감도(radial_gain), 감마(radial_gamma)로 대각선 반응을 강화
- r이 커질수록 헤드 포즈(yaw/pitch) 기반 성분을 자동 혼합(smoothstep(r0->r1))하여 실제 사용 행태(시선 끝단=고개 돌림)를 반영
- 시각화: 원형 가이드 링, 토글 키(o=원형, k=링, p=헤드 블렌드)
- 나머지(v2.2)의 캘리브/EMA/호모그래피/ArUco/수동ROI/깜박임 게이팅 유지

실행 예(Windows PowerShell):
  python .\single_camera_iris_gaze_v2_3_2_circular.py `
    --single_cam --eye_cam 0 --flip_eye `
    --feat_gain 1.2 --blink_gate `
    --circ_on `
    --circ_r_max 0.38 --circ_clip soft --radial_gain 1.6 --radial_gamma 0.85 `
    --head_on --head_mix_auto --head_w0 0.25 --head_w1 0.70 `
    --draw_ring
"""

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np

# -------- MediaPipe 임포트 --------
try:
    import mediapipe as mp
except ImportError as e:
    raise SystemExit(
        "mediapipe 임포트 실패. 설치 예: pip install mediapipe\n"
        f"원본 오류: {e}"
    )

# -------- ArUco(선택) 임포트 --------
try:
    import cv2.aruco as aruco
    HAVE_ARUCO = True
except Exception:
    HAVE_ARUCO = False

# ----------------------------
# 보조 상수 / 랜드마크 인덱스
# ----------------------------
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# ----------------------------
# 보조 클래스/함수
# ----------------------------
@dataclass
class EMA2D:
    alpha: float = 0.25
    value: Optional[np.ndarray] = None
    clamp: Optional[Tuple[float, float]] = None

    def update(self, pt: np.ndarray) -> np.ndarray:
        # 한글 주석: 지수이동평균(EMA)로 (x,y) 노이즈를 줄임
        if self.value is None or not np.isfinite(self.value).all():
            self.value = pt.astype(np.float32)
        else:
            self.value = self.alpha * pt.astype(np.float32) + (1.0 - self.alpha) * self.value
        if self.clamp is not None:
            lo, hi = self.clamp
            self.value = np.clip(self.value, lo, hi)
        return self.value

    def reset(self):
        self.value = None


def parse_args():
    ap = argparse.ArgumentParser(description="Single-Camera Iris Gaze v2.3 (circular + head blend)")
    # 카메라
    ap.add_argument("--single_cam", action="store_true", help="단일 카메라(eye)만 사용하여 테스트")
    ap.add_argument("--world_cam", type=int, default=0, help="월드 카메라 인덱스 (single_cam이 아니면 사용)")
    ap.add_argument("--eye_cam", type=int, default=1, help="눈 카메라 인덱스")
    ap.add_argument("--world_w", type=int, default=1280, help="월드 카메라 가로 해상도(요청값)")
    ap.add_argument("--world_h", type=int, default=720, help="월드 카메라 세로 해상도(요청값)")
    ap.add_argument("--eye_w", type=int, default=640, help="눈 카메라 가로 해상도(요청값)")
    ap.add_argument("--eye_h", type=int, default=480, help="눈 카메라 세로 해상도(요청값)")
    ap.add_argument("--flip_eye", action="store_true", help="눈 카메라 좌우 반전")
    ap.add_argument("--draw_scale", type=float, default=1.0, help="도형/텍스트 스케일")

    # 추적 신뢰도
    ap.add_argument("--min_det_conf", type=float, default=0.5, help="FaceMesh min_detection_confidence")
    ap.add_argument("--min_trk_conf", type=float, default=0.5, help="FaceMesh min_tracking_confidence")

    # EMA
    ap.add_argument("--ema_alpha", type=float, default=0.25, help="(레거시) 기본 EMA 알파")
    ap.add_argument("--feat_ema_alpha", type=float, default=None, help="특징(vx,vy) EMA 알파 (미지정시 ema_alpha)")
    ap.add_argument("--coord_ema_alpha", type=float, default=None, help="화면좌표(sx,sy) EMA 알파 (미지정시 ema_alpha)")

    # 특징 스케일 부스트
    ap.add_argument("--feat_gain", type=float, default=1.0, help="(vx,vy) 감도 스케일")

    # 한쪽 눈만 사용
    ap.add_argument("--use_left_eye_only", action="store_true", help="좌안만 사용")
    ap.add_argument("--use_right_eye_only", action="store_true", help="우안만 사용")

    # 캘리브 파일
    ap.add_argument("--calib_path", type=str, default="gaze_calib.npz", help="캘리브 계수 저장/로드 경로")

    # 깜박임 게이팅
    ap.add_argument("--blink_gate", action="store_true", help="깜박임/부분가림 게이팅 기본 on")
    ap.add_argument("--blink_v_ratio", type=float, default=0.6, help="hv < ratio * hv_ema → 프레임 무시")

    # === [신규] 원형-라디얼 파라미터 ===
    ap.add_argument("--circ_on", action="store_true", help="원형 라디얼 모드 기본 on")
    ap.add_argument("--circ_r_max", type=float, default=0.38, help="원형 반경 상한(정규화 좌표계)")
    ap.add_argument("--circ_clip", type=str, default="soft", choices=["soft", "hard"], help="반경 클리핑 방식")
    ap.add_argument("--radial_gain", type=float, default=1.4, help="라디얼 감도(>1이면 멀리, 대각선 반응 커짐)")
    ap.add_argument("--radial_gamma", type=float, default=0.90, help="라디얼 감마(1보다 작으면 빠르게 커짐)")

    # === [신규] 헤드 포즈 블렌드 ===
    ap.add_argument("--head_on", action="store_true", help="헤드 포즈 성분 사용")
    ap.add_argument("--head_mix_auto", action="store_true", help="r에 따른 자동 혼합(smoothstep)")
    ap.add_argument("--head_w0", type=float, default=0.2, help="auto-mix 하한 r0 (w=0 시작점)")
    ap.add_argument("--head_w1", type=float, default=0.65, help="auto-mix 상한 r1 (w=1 도달점)")
    ap.add_argument("--head_fixed_w", type=float, default=0.35, help="auto-mix 미사용 시 고정 가중치")
    ap.add_argument("--head_gain_yaw", type=float, default=0.8, help="yaw → 화면 x 변환 감도")
    ap.add_argument("--head_gain_pitch", type=float, default=0.8, help="pitch → 화면 y 변환 감도")

    # === [신규] 원형 링 표시 ===
    ap.add_argument("--draw_ring", action="store_true", help="원형 가이드 링 표시")

    return ap.parse_args()


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"카메라 열기 실패 (index={index})")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def draw_hud(frame_bgr: np.ndarray, text_lines: List[str], scale: float = 1.0, org=(10,20)):
    x, y = org
    for t in text_lines:
        cv2.putText(frame_bgr, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += int(20 * scale)


def _indices_from_connections(conns):
    idxs = set()
    for a, b in conns:
        idxs.add(a); idxs.add(b)
    return sorted(list(idxs))


def _fit_circle_ls(pts: np.ndarray):
    # 한글 주석: 최소제곱으로 원 맞춤(중심, 반지름)
    if pts is None or len(pts) < 3:
        if pts is None or len(pts) == 0:
            return np.array([np.nan, np.nan], np.float32), np.float32('nan')
        return pts.mean(axis=0).astype(np.float32), np.float32('nan')
    x = pts[:,0:1]; y = pts[:,1:2]
    A = np.hstack([2*x, 2*y, np.ones((len(pts),1), np.float32)])
    b = (x*x + y*y)
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        a, b2, c = float(sol[0,0]), float(sol[1,0]), float(sol[2,0])
        center = np.array([a, b2], dtype=np.float32)
        r = np.sqrt(max(center.dot(center) + c, 0.0)).astype(np.float32)
        return center, r
    except Exception:
        return pts.mean(axis=0).astype(np.float32), np.float32('nan')


def eye_feature_from_facemesh_ex(face_landmarks, mp_face_mesh, which: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
    """
    - 눈-정렬 좌표계(u: 좌->우, v: 위->아래 직교)에서 iris 중심의 오프셋을 (vx,vy)로 정규화
    - vx,vy는 각각 눈 가로폭(hw), 세로폭(hv)로 나눠 스케일링되어 '타원' → 이후 원형 보정에서 r 계산
    """
    if which == 'left':
        eye_conns = mp_face_mesh.FACEMESH_LEFT_EYE
        iris_idxs = _indices_from_connections(mp_face_mesh.FACEMESH_LEFT_IRIS)
    else:
        eye_conns = mp_face_mesh.FACEMESH_RIGHT_EYE
        iris_idxs = _indices_from_connections(mp_face_mesh.FACEMESH_RIGHT_IRIS)

    eye_idxs = _indices_from_connections(eye_conns)
    try:
        lm = face_landmarks
        eye_pts = np.array([(lm[i].x, lm[i].y) for i in eye_idxs if i < len(lm)], dtype=np.float32)
        iris_pts = np.array([(lm[i].x, lm[i].y) for i in iris_idxs if i < len(lm)], dtype=np.float32)
    except Exception:
        return None, None
    if eye_pts.size == 0 or iris_pts.size == 0:
        return None, None

    iL = int(np.argmin(eye_pts[:,0])); iR = int(np.argmax(eye_pts[:,0]))
    pL, pR = eye_pts[iL], eye_pts[iR]
    u = pR - pL
    hw = float(np.linalg.norm(u))
    if hw < 1e-6:
        return None, None
    u = u / hw
    v = np.array([-u[1], u[0]], dtype=np.float32)

    proj_v = eye_pts @ v
    hv = float(proj_v.max() - proj_v.min())
    if hv < 1e-6:
        hv = 1e-6

    iris_c, _ = _fit_circle_ls(iris_pts)
    eye_ctr = 0.5 * (pL + pR)

    d = iris_c - eye_ctr
    vx = float(d @ u) / hw
    vy = float(d @ v) / hv

    feat = np.array([vx, vy], dtype=np.float32)
    dbg = {
        'pL': pL, 'pR': pR, 'u': u, 'v': v, 'eye_ctr': eye_ctr,
        'iris_c': iris_c, 'hw': np.array([hw], np.float32), 'hv': np.array([hv], np.float32)
    }
    return feat, dbg


# === 2차 다항 회귀(캘리브) ===
@dataclass
class Poly2Calibrator:
    cx: Optional[np.ndarray] = None
    cy: Optional[np.ndarray] = None
    samples_f: List[np.ndarray] = None
    samples_s: List[np.ndarray] = None
    active: bool = False
    use: bool = False

    def __post_init__(self):
        self.samples_f = []
        self.samples_s = []

    def _phi(self, v):
        x, y = float(v[0]), float(v[1])
        return np.array([x, y, x*y, x*x, y*y, 1.0], dtype=np.float32)

    def add(self, feat_vxy, scr_sxy):
        self.samples_f.append(np.array(feat_vxy, dtype=np.float32))
        self.samples_s.append(np.array(scr_sxy, dtype=np.float32))

    def fit(self):
        if len(self.samples_f) < 6:
            return False
        F = np.stack([self._phi(v) for v in self.samples_f], axis=0)
        S = np.stack(self.samples_s, axis=0)
        self.cx, *_ = np.linalg.lstsq(F, S[:,0], rcond=None)
        self.cy, *_ = np.linalg.lstsq(F, S[:,1], rcond=None)
        return True

    def map(self, vxy):
        if self.cx is None or self.cy is None:
            return None
        phi = self._phi(vxy)
        sx = float(phi @ self.cx)
        sy = float(phi @ self.cy)
        return np.clip(np.array([sx, sy], dtype=np.float32), 0.0, 1.0)

    def save(self, path="gaze_calib.npz"):
        if self.cx is not None and self.cy is not None:
            np.savez(path, cx=self.cx, cy=self.cy)

    def load(self, path="gaze_calib.npz"):
        z = np.load(path)
        self.cx = z["cx"]; self.cy = z["cy"]; self.use = True


# === 캘리브 타깃 포인트 ===
CAL_POINTS = [
    (0.1,0.1),(0.5,0.1),(0.9,0.1),
    (0.1,0.5),(0.5,0.5),(0.9,0.5),
    (0.1,0.9),(0.5,0.9),(0.9,0.9)
]


# === ArUco 준비 ===
_ARUCO_IDS_WANT = (0, 1, 2, 3)  # TL,TR,BR,BL
if HAVE_ARUCO:
    try:
        _ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        _ARUCO_PARAMS = aruco.DetectorParameters()
        _HAS_NEW_API = hasattr(aruco, "ArucoDetector")
    except Exception:
        HAVE_ARUCO = False
        _HAS_NEW_API = False


def _detect_aruco(gray: np.ndarray):
    if not HAVE_ARUCO:
        return None, None, None
    try:
        if _HAS_NEW_API:
            detector = aruco.ArucoDetector(_ARUCO_DICT, _ARUCO_PARAMS)
            corners, ids, rejected = detector.detectMarkers(gray)
        else:
            corners, ids, rejected = aruco.detectMarkers(gray, _ARUCO_DICT, parameters=_ARUCO_PARAMS)
        return corners, ids, rejected
    except Exception:
        return None, None, None


def estimate_homography_from_aruco(world_bgr: np.ndarray) -> Optional[np.ndarray]:
    if not HAVE_ARUCO:
        return None
    gray = cv2.cvtColor(world_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _detect_aruco(gray)
    if ids is None or len(ids) < 4 or corners is None:
        return None
    ids_flat = ids.flatten().tolist()
    if not all(k in ids_flat for k in _ARUCO_IDS_WANT):
        return None
    id_to_center = {}
    for c, i in zip(corners, ids.flatten()):
        center = c[0].mean(axis=0).astype(np.float32)
        id_to_center[int(i)] = center
    try:
        pts_dst = np.float32([
            id_to_center[0], id_to_center[1], id_to_center[2], id_to_center[3]
        ])
    except KeyError:
        return None
    pts_src = np.float32([[0,0],[1,0],[1,1],[0,1]])
    H, _ = cv2.findHomography(pts_src, pts_dst, method=cv2.RANSAC)
    return H


def overlay_virtual_aruco(frame: np.ndarray, margin_ratio: float = 0.06, marker_ratio: float = 0.12):
    """Eye 프레임 위에 가상 ArUco 4개를 오버레이(TL=0,TR=1,BR=2,BL=3)."""
    if not HAVE_ARUCO:
        return False
    h, w = frame.shape[:2]
    ms = int(min(h, w) * marker_ratio)
    mg = int(min(h, w) * margin_ratio)
    ids = [0,1,2,3]
    # 마커 생성
    markers = []
    for i in ids:
        img = aruco.drawMarker(_ARUCO_DICT, i, ms)
        markers.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    # 위치(TL,TR,BR,BL)
    pos = [
        (mg, mg),                # TL
        (w - mg - ms, mg),       # TR
        (w - mg - ms, h - mg - ms), # BR
        (mg, h - mg - ms)        # BL
    ]
    for p, m in zip(pos, markers):
        x, y = p
        frame[y:y+ms, x:x+ms] = m
    return True


# === 수동 ROI 마우스 콜백 ===
class ManualHomogPicker:
    def __init__(self, window_name: str):
        self.window = window_name
        self.active = False
        self.pts: List[Tuple[int,int]] = []  # TL, TR, BR, BL

    def start(self):
        self.active = True
        self.pts.clear()
        cv2.setMouseCallback(self.window, self._on_mouse)

    def stop(self):
        self.active = False
        cv2.setMouseCallback(self.window, lambda *args: None)

    def _on_mouse(self, event, x, y, flags, param):
        if not self.active:
            return
        if event == cv2.EVENT_LBUTTONDOWN and len(self.pts) < 4:
            self.pts.append((x,y))

    def draw(self, frame: np.ndarray):
        # 진행상황 오버레이
        for i, (x,y) in enumerate(self.pts):
            cv2.circle(frame, (x,y), 5, (0,255,0), -1)
            cv2.putText(frame, f"{i}", (x+6,y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"{i}", (x+6,y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    def compute_H(self) -> Optional[np.ndarray]:
        if len(self.pts) != 4:
            return None
        pts_dst = np.float32(self.pts)  # TL,TR,BR,BL
        pts_src = np.float32([[0,0],[1,0],[1,1],[0,1]])
        H, _ = cv2.findHomography(pts_src, pts_dst, method=0)
        return H


# === [신규] 헤드 포즈 근사: yaw/pitch 추정 (정규화) ===
def estimate_head_yaw_pitch(face_landmarks) -> Optional[np.ndarray]:
    """
    간단 근사치(정규화):
    - yaw ≈ 코끝(x) - 얼굴중심(x) / 얼굴폭
    - pitch ≈ (눈중심(y) - 코끝(y)) / 얼굴높이
    ※ FaceMesh 카메라 정규좌표([0,1]) 기반 대략값이며, 캘리브레이션으로 보정되는 전제
    """
    lm = face_landmarks
    n = len(lm)
    if n < 468:
        return None

    # 기준 포인트 추출 (양 눈가/귀 부근 등으로 폭·높이 추정)
    # 눈가 대략 인덱스(33, 263), 코끝(1), 턱끝(152), 이마 부근(10) 등
    try:
        p_left = np.array([lm[33].x, lm[33].y], np.float32)
        p_right = np.array([lm[263].x, lm[263].y], np.float32)
        p_nose = np.array([lm[1].x, lm[1].y], np.float32)
        p_chin = np.array([lm[152].x, lm[152].y], np.float32)
        p_fore = np.array([lm[10].x, lm[10].y], np.float32)
    except Exception:
        return None

    face_c = 0.5 * (p_left + p_right)
    face_w = float(np.linalg.norm(p_right - p_left))
    face_h = float(abs(p_fore[1] - p_chin[1]))
    if face_w < 1e-6 or face_h < 1e-6:
        return None

    yaw = (p_nose[0] - face_c[0]) / face_w      # 좌우 회전(+우측)
    eye_mid = np.array([(p_left[0]+p_right[0])/2, (p_left[1]+p_right[1])/2], np.float32)
    pitch = (eye_mid[1] - p_nose[1]) / face_h    # 위/아래(+아래)

    return np.array([yaw, pitch], np.float32)


# === [신규] 원형-라디얼 변환 & 헤드 블렌드 ===
def softclip_r(r, r_max):
    # 한글: 부드러운 포화. r_max를 넘으면 부드럽게 완만해짐
    # 여기서는 tanh 기반 간단 소프트클립
    if r <= r_max:
        return r
    s = r / (r_max + 1e-6)
    return r_max * np.tanh(s)

def radial_nonlin(r, gain=1.4, gamma=0.9):
    # 한글: 라디얼 감도와 감마로 대각선 방향까지 반응을 키움
    # r' = (gain * r) ** gamma
    return (max(gain, 1e-6) * max(r, 0.0)) ** max(gamma, 1e-6)

def smoothstep(x, a, b):
    # 한글: [a,b] 구간에서 0→1로 부드럽게
    if b <= a:
        return 1.0 if x >= b else 0.0
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)

def apply_circular_and_head(vxy: np.ndarray,
                            head_xy: Optional[np.ndarray],
                            circ_on: bool,
                            r_max: float,
                            clip_mode: str,
                            radial_gain: float,
                            radial_gamma: float,
                            head_on: bool,
                            auto_mix: bool,
                            w0: float, w1: float,
                            w_fixed: float,
                            head_gain_yaw: float,
                            head_gain_pitch: float) -> np.ndarray:
    """
    1) 원형 처리: (vx,vy) → (r,θ) → r를 감도/감마/클리핑 적용 → (x',y')
    2) 헤드 포즈 혼합: r이 커질수록 head_xy(= [yaw*gx, pitch*gy])를 더 많이 섞음
    """
    v = np.array(vxy, np.float32)
    # 라디얼 변환
    if circ_on:
        r = float(np.linalg.norm(v))
        theta = float(np.arctan2(v[1], v[0]))
        # 소프트/하드 클립
        if clip_mode == "hard":
            r_eff = min(r, r_max)
        else:
            r_eff = softclip_r(r, r_max)

        # 감도/감마
        r_eff = radial_nonlin(r_eff, gain=radial_gain, gamma=radial_gamma)

        v_eye = np.array([r_eff * np.cos(theta), r_eff * np.sin(theta)], np.float32)
    else:
        v_eye = v.copy()

    # 헤드 포즈 성분
    if head_on and head_xy is not None and np.isfinite(head_xy).all():
        h = np.array([head_gain_yaw * head_xy[0], head_gain_pitch * head_xy[1]], np.float32)
        # r 기반 자동 혼합
        if circ_on:
            r_now = float(np.linalg.norm(v))
            w = smoothstep(r_now, w0, w1) if auto_mix else np.clip(w_fixed, 0.0, 1.0)
        else:
            w = np.clip(w_fixed, 0.0, 1.0)
        out = (1.0 - w) * v_eye + w * h
    else:
        out = v_eye

    return out


def main():
    args = parse_args()
    feat_alpha = args.feat_ema_alpha if args.feat_ema_alpha is not None else args.ema_alpha
    coord_alpha = args.coord_ema_alpha if args.coord_ema_alpha is not None else args.ema_alpha

    # 카메라 열기
    if args.single_cam:
        world_cap = None
        eye_cap = open_camera(args.eye_cam, args.eye_w, args.eye_h)
    else:
        world_cap = open_camera(args.world_cam, args.world_w, args.world_h)
        eye_cap = open_camera(args.eye_cam, args.eye_w, args.eye_h)

    # MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=args.min_det_conf, min_tracking_confidence=args.min_trk_conf,
    )

    # 필터
    ema_feat = EMA2D(alpha=feat_alpha, clamp=None)
    ema_norm = EMA2D(alpha=coord_alpha, clamp=(0.0, 1.0))

    # 캘리브
    cal = Poly2Calibrator()
    cal_idx = 0

    # 호모그래피
    use_homog = False
    H_world = None
    use_virtual_aruco = False
    manual_picker = ManualHomogPicker("World (Gaze Overlay)")

    # 깜박임 게이팅
    blink_gate = args.blink_gate
    hv_ema = None

    # UI
    show_help = True
    show_debug = False

    # 원형/헤드 토글 상태
    circ_on = args.circ_on
    draw_ring = args.draw_ring
    head_on = args.head_on

    while True:
        # 프레임 읽기
        if args.single_cam:
            ok_eye, eye_bgr = eye_cap.read()
            if not ok_eye:
                print("경고: eye 카메라 프레임 읽기 실패.")
                break
            world_bgr = eye_bgr.copy()  # single-cam: world=eye
        else:
            ok_world, world_bgr = world_cap.read()
            ok_eye, eye_bgr = eye_cap.read()
            if not ok_world or not ok_eye:
                print("경고: 카메라 프레임 읽기 실패.")
                break

        if args.flip_eye:
            eye_bgr = cv2.flip(eye_bgr, 1)
            if args.single_cam:
                world_bgr = cv2.flip(world_bgr, 1)

        # 가상 ArUco 오버레이(단일/이중 모드 모두에서 동작)
        if use_virtual_aruco:
            overlay_virtual_aruco(world_bgr)

        # Eye → FaceMesh
        eye_rgb = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2RGB)
        eye_rgb.flags.writeable = False
        res = face_mesh.process(eye_rgb)
        eye_rgb.flags.writeable = True

        gaze_pt01 = None
        dbg_left = dbg_right = None
        feat_v = None
        hv_curr_list = []
        head_xy = None

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            # 눈 특징
            left_v, dbg_left = eye_feature_from_facemesh_ex(lm, mp_face_mesh, 'left')
            right_v, dbg_right = eye_feature_from_facemesh_ex(lm, mp_face_mesh, 'right')

            if args.use_left_eye_only and left_v is not None:
                base_v = left_v
                if dbg_left is not None: hv_curr_list.append(float(dbg_left['hv'][0]))
            elif args.use_right_eye_only and right_v is not None:
                base_v = right_v
                if dbg_right is not None: hv_curr_list.append(float(dbg_right['hv'][0]))
            else:
                vs = []
                if left_v is not None:
                    vs.append(left_v)
                    if dbg_left is not None: hv_curr_list.append(float(dbg_left['hv'][0]))
                if right_v is not None:
                    vs.append(right_v)
                    if dbg_right is not None: hv_curr_list.append(float(dbg_right['hv'][0]))
                if len(vs) == 1:
                    base_v = vs[0]
                elif len(vs) == 2:
                    base_v = 0.5 * (vs[0] + vs[1])
                else:
                    base_v = None

            # 헤드 포즈 근사
            hp = estimate_head_yaw_pitch(lm)
            if hp is not None and np.isfinite(hp).all():
                head_xy = hp  # yaw→x, pitch→y (후속 gain 적용)

            if base_v is not None and np.isfinite(base_v).all():
                # 감도 스케일 → EMA
                base_v = base_v * float(args.feat_gain)
                base_v = ema_feat.update(base_v)

                # 원형/헤드 혼합 적용
                feat_v = apply_circular_and_head(
                    vxy=base_v,
                    head_xy=head_xy,
                    circ_on=circ_on,
                    r_max=float(args.circ_r_max),
                    clip_mode=args.circ_clip,
                    radial_gain=float(args.radial_gain),
                    radial_gamma=float(args.radial_gamma),
                    head_on=head_on,
                    auto_mix=args.head_mix_auto,
                    w0=float(args.head_w0),
                    w1=float(args.head_w1),
                    w_fixed=float(args.head_fixed_w),
                    head_gain_yaw=float(args.head_gain_yaw),
                    head_gain_pitch=float(args.head_gain_pitch),
                )

            # 캘리브 사용
            if cal.use and feat_v is not None:
                gated_out = False
                if blink_gate and len(hv_curr_list) > 0:
                    hv_curr = float(np.mean(hv_curr_list))
                    if hv_ema is None or not np.isfinite(hv_ema):
                        hv_ema = hv_curr
                    else:
                        hv_ema = 0.9 * hv_ema + 0.1 * hv_curr
                    if hv_curr < args.blink_v_ratio * hv_ema:
                        gated_out = True
                if not gated_out:
                    mapped = cal.map(feat_v)
                    if mapped is not None and np.isfinite(mapped).all():
                        gaze_pt01 = ema_norm.update(mapped.astype(np.float32))
            else:
                # 보정 미사용시: iris 중심 평균 폴백 (EMA로 약간만 안정화)
                def _center_from_dbg(dbg):
                    return dbg['iris_c'] if (dbg is not None and np.isfinite(dbg['iris_c']).all()) else None
                l_c = _center_from_dbg(dbg_left)
                r_c = _center_from_dbg(dbg_right)
                base = None
                if args.use_left_eye_only and l_c is not None:
                    base = l_c
                elif args.use_right_eye_only and r_c is not None:
                    base = r_c
                elif l_c is not None and r_c is not None:
                    base = 0.5 * (l_c + r_c)
                if base is not None and np.isfinite(base).all():
                    gaze_pt01 = ema_norm.update(base.astype(np.float32))

            # Eye 디버그(좌표축/홍채중심)
            if show_debug:
                h_eye, w_eye = eye_bgr.shape[:2]
                def DN(p): return (int(p[0]*w_eye), int(p[1]*h_eye))
                if dbg_left is not None:
                    ctr = dbg_left['eye_ctr']; u = dbg_left['u']; v = dbg_left['v']; ic = dbg_left['iris_c']
                    cv2.circle(eye_bgr, DN(ctr), 3, (255, 0, 0), -1)
                    cv2.line(eye_bgr, DN(ctr - 0.2*u), DN(ctr + 0.2*u), (255, 0, 0), 2)
                    cv2.line(eye_bgr, DN(ctr - 0.2*v), DN(ctr + 0.2*v), (0, 255, 0), 2)
                    cv2.circle(eye_bgr, DN(ic), 4, (0, 0, 255), -1)
                if dbg_right is not None:
                    ctr = dbg_right['eye_ctr']; u = dbg_right['u']; v = dbg_right['v']; ic = dbg_right['iris_c']
                    cv2.circle(eye_bgr, DN(ctr), 3, (255, 0, 255), -1)
                    cv2.line(eye_bgr, DN(ctr - 0.2*u), DN(ctr + 0.2*u), (255, 0, 255), 2)
                    cv2.line(eye_bgr, DN(ctr - 0.2*v), DN(ctr + 0.2*v), (0, 255, 255), 2)
                    cv2.circle(eye_bgr, DN(ic), 4, (0, 165, 255), -1)

        # 호모그래피 갱신(가상 ArUco 또는 수동 ROI)
        if use_virtual_aruco:
            H = estimate_homography_from_aruco(world_bgr)
            if H is not None:
                H_world = H
        if manual_picker.active and len(manual_picker.pts) == 4:
            H = manual_picker.compute_H()
            if H is not None:
                H_world = H
                manual_picker.stop()

        # 월드 프레임에 시선 마커
        if gaze_pt01 is not None:
            if use_homog and H_world is not None:
                src = np.array([[[gaze_pt01[0], gaze_pt01[1]]]], dtype=np.float32)
                pt = cv2.perspectiveTransform(src, H_world)[0, 0]
                gx, gy = int(pt[0]), int(pt[1])
            else:
                h_w, w_w = world_bgr.shape[:2]
                gx, gy = int(gaze_pt01[0] * w_w), int(gaze_pt01[1] * h_w)
            r = int(10 * args.draw_scale)
            cv2.circle(world_bgr, (gx, gy), r, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(world_bgr, (gx, gy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        # HUD
        if show_help:
            mode = "SINGLE" if args.single_cam else "DUAL"
            help_lines = [
                f"Iris Gaze v2.3 [{mode}]  (Circular+HeadBlend + Calib + Homog)",
                "[q] 종료  [h] 도움말  [d] 디버그  [r] EMA리셋",
                "[c] 캘리브 모드  [SPACE] 샘플추가  [g] 보정 on/off  [S/L] 저장/로드",
                "[a] 호모그래피 on/off  [m] 수동ROI(4클릭)  [v] 가상ArUco on/off",
                f"(o) 원형모드={circ_on}  (k) 링표시={draw_ring}  (p) 헤드블렌드={head_on}",
                f"circ_r_max={args.circ_r_max:.2f} clip={args.circ_clip} gain={args.radial_gain:.2f} gamma={args.radial_gamma:.2f}",
                f"head_auto={args.head_mix_auto} r0={args.head_w0:.2f} r1={args.head_w1:.2f} fixed_w={args.head_fixed_w:.2f}",
                f"EMA(feat,coord)=({feat_alpha:.2f},{coord_alpha:.2f}), feat_gain={args.feat_gain:.2f}, blink_gate={blink_gate}",
            ]
            if cal.active and 0 <= cal_idx < len(CAL_POINTS):
                sx, sy = CAL_POINTS[cal_idx]
                help_lines.append(f"Target #{cal_idx+1}/{len(CAL_POINTS)} at ({sx:.2f},{sy:.2f}) → SPACE로 채집")
            if manual_picker.active:
                help_lines.append("수동 ROI: TL→TR→BR→BL 순서로 4번 클릭하세요")
            draw_hud(world_bgr, help_lines, scale=args.draw_scale)

        # 캘리 타깃 오버레이
        if cal.active and 0 <= cal_idx < len(CAL_POINTS):
            h_w, w_w = world_bgr.shape[:2]
            sx, sy = CAL_POINTS[cal_idx]
            cx, cy = int(sx*w_w), int(sy*h_w)
            cv2.circle(world_bgr, (cx,cy), int(14*args.draw_scale), (0,255,0), 2, cv2.LINE_AA)

        # 수동 ROI 포인트 시각화
        if manual_picker.active:
            manual_picker.draw(world_bgr)

        # === [신규] 원형 링 그리기 (시각 가이드) ===
        if draw_ring:
            h_w, w_w = world_bgr.shape[:2]
            # 화면 중앙 기준 링(시선 정규좌표 0~1 → 중앙(0.5,0.5))
            cx, cy = int(0.5*w_w), int(0.5*h_w)
            # 시선 정규 반경 → 픽셀 반경
            r_pix = int(args.circ_r_max * min(w_w, h_w))
            cv2.circle(world_bgr, (cx, cy), r_pix, (120, 120, 120), 1, cv2.LINE_AA)

        # 창 표시
        cv2.imshow("World (Gaze Overlay)", world_bgr)
        if show_debug:
            cv2.imshow("Eye (Debug)", eye_bgr)
        else:
            w = min(480, eye_bgr.shape[1])
            h = int(w * eye_bgr.shape[0] / max(1, eye_bgr.shape[1]))
            cv2.imshow("Eye (Debug)", cv2.resize(eye_bgr, (w, h)))

        # 키 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'):
            show_help = not show_help
        elif key == ord('d'):
            show_debug = not show_debug
        elif key == ord('r'):
            ema_feat.reset(); ema_norm.reset(); hv_ema = None
        elif key == ord('c'):
            cal.active = not cal.active
            cal_idx = 0
            print(f"[Calib] active={cal.active}")
        elif key == 32:  # SPACE
            if cal.active:
                if feat_v is not None and np.isfinite(feat_v).all():
                    sx, sy = CAL_POINTS[cal_idx]
                    cal.add(feat_v, (sx, sy))
                    print(f"[Calib] sample {cal_idx+1}/{len(CAL_POINTS)} added")
                    cal_idx += 1
                    if cal_idx >= len(CAL_POINTS):
                        ok = cal.fit()
                        cal.active = False
                        cal.use = ok
                        print(f"[Calib] done. use={cal.use}")
                else:
                    print("[Calib] 현재 프레임에서 유효한 특징(vx,vy)을 얻지 못했습니다.")
        elif key == ord('g'):
            cal.use = not cal.use
            print(f"[Calib] use={cal.use}")
        elif key == ord('S'):
            try:
                cal.save(args.calib_path)
                print(f"[Calib] saved to {args.calib_path}")
            except Exception as e:
                print("[Calib] save failed:", e)
        elif key == ord('L'):
            try:
                cal.load(args.calib_path)
                print(f"[Calib] loaded from {args.calib_path} (enabled)")
            except Exception as e:
                print("[Calib] load failed:", e)
        elif key == ord('a'):
            use_homog = not use_homog
            print(f"[H] use_homog={use_homog}")
        elif key == ord('m'):
            if manual_picker.active:
                manual_picker.stop()
                print("[H] manual ROI canceled")
            else:
                manual_picker.start()
                print("[H] manual ROI: TL→TR→BR→BL 순서로 4점 클릭")
        elif key == ord('v'):
            if not HAVE_ARUCO:
                print("[H] opencv-contrib-python(ArUco)이 설치되어 있지 않습니다.")
            use_virtual_aruco = not use_virtual_aruco
            print(f"[H] virtual ArUco overlay = {use_virtual_aruco}")
        # === [신규] 원형/링/헤드 토글 & 반경 미세 조절 ===
        elif key == ord('o'):
            circ_on = not circ_on
            print(f"[CIRC] circular mode = {circ_on}")
        elif key == ord('k'):
            draw_ring = not draw_ring
            print(f"[CIRC] draw ring = {draw_ring}")
        elif key == ord('p'):
            head_on = not head_on
            print(f"[HEAD] head blend = {head_on}")
        elif key == ord('['):
            args.circ_r_max = max(0.05, args.circ_r_max - 0.01)
            print(f"[CIRC] r_max = {args.circ_r_max:.2f}")
        elif key == ord(']'):
            args.circ_r_max = min(0.80, args.circ_r_max + 0.01)
            print(f"[CIRC] r_max = {args.circ_r_max:.2f}")

    if world_cap is not None:
        world_cap.release()
    eye_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
