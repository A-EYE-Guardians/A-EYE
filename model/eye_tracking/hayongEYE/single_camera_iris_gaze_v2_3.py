#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
single_camera_iris_gaze_v2_3.py

v2.3: "조금만 눈을 움직여도 월드 화면은 확확" 목표로 감도 증폭 파이프라인 추가
- (A) 눈-정렬 좌표계 특징 (vx, vy)  ← v2.1/2.2와 동일
- (B) **감도 증폭(Sensitivity Amplifier)** : deadzone / 감마 비선형 / 축별 게인 / 포화
- (C) 2차 다항 캘리브레이션  : (증폭된 특징) → (스크린 정규좌표)
- (D) 호모그래피(H)            : (스크린) → (월드 픽셀), 단일 카메라에서는 Eye 프레임을 월드 캔버스로 사용
- (E) 깜박임 게이팅, 중립 오프셋(Neutral) EMA, 분리된 EMA(특징/좌표)

핵심 변경점
- 이전 버전의 --feat_gain 은 단순 선형 스케일이었음. v2.3은 **deadzone + 감마(γ<1) + 게인 + 포화(tanh)** 로
  중심 근처에서도 민감하게 '확확' 움직이되 과도한 끝단 과민은 제한.
- **캘리브레이션은 "증폭된 특징(v_amp)" 기준**으로 수행/적용됩니다. (= 증폭 파이프라인도 교정 대상)
- 캘리브레이션 OFF일 땐, 간이 맵핑: sx=0.5+0.5*tanh(kx*v_amp.x), sy=… 로 즉시 큰 움직임 제공.

실행 예(단일 카메라):
  python single_camera_iris_gaze_v2_3.py --single_cam --eye_cam 0 --flip_eye \
    --sens_gain_x 4.0 --sens_gain_y 4.0 --sens_gamma_x 0.8 --sens_gamma_y 0.8 --blink_gate

키보드:
  q 종료 | h HUD | d 디버그 | r EMA리셋
  c 캘리브 on/off | SPACE 샘플 | g 보정 on/off | S/L 저장/로드
  a 호모그래피 on/off | m 수동ROI | v 가상ArUco
  n 중립 오프셋(Neutral) 추적 on/off | N Neutral 즉시 리셋
  X/x 감도 X축 +10%/-10% | Y/y 감도 Y축 +10%/-10%

주의: opencv-contrib-python을 설치해야 ArUco 사용 가능.
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
# 상수 / 랜드마크 인덱스
# ----------------------------
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# ----------------------------
# 보조 클래스/함수
# ----------------------------
@dataclass
class EMA2D:
    """2차원 점 EMA. clamp=(lo,hi)일 때 각 축 동일 클램프."""
    alpha: float = 0.25
    value: Optional[np.ndarray] = None
    clamp: Optional[Tuple[float, float]] = None

    def update(self, pt: np.ndarray) -> np.ndarray:
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
    ap = argparse.ArgumentParser(description="Single-Cam Iris Gaze v2.3 (aggressive sensitivity + calibration + homography)")
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

    # v2.3 감도 증폭 파라미터
    ap.add_argument("--sens_gain_x", type=float, default=4.0, help="X축 감도 게인(증폭)")
    ap.add_argument("--sens_gain_y", type=float, default=4.0, help="Y축 감도 게인(증폭)")
    ap.add_argument("--sens_gamma_x", type=float, default=0.8, help="X축 감마(0<γ<=1). 작을수록 중심 민감↑")
    ap.add_argument("--sens_gamma_y", type=float, default=0.8, help="Y축 감마(0<γ<=1). 작을수록 중심 민감↑")
    ap.add_argument("--sens_deadzone", type=float, default=0.02, help="데드존 절대값(threshold). 이내는 0으로 취급")
    ap.add_argument("--sens_sat", type=float, default=0.95, help="증폭 결과 포화 한계(|v_amp|<=sens_sat)")

    # 캘리브 파일
    ap.add_argument("--calib_path", type=str, default="gaze_calib.npz", help="캘리브 계수 저장/로드 경로")

    # 중립 오프셋(고개/개인 편차 보정) 추적
    ap.add_argument("--neutral_track", action="store_true", default=True, help="중립 오프셋 EMA 추적 on")
    ap.add_argument("--no_neutral_track", dest="neutral_track", action="store_false", help="중립 오프셋 EMA 추적 off")
    ap.add_argument("--neutral_alpha", type=float, default=0.02, help="중립 오프셋 EMA 계수(느리게 적응 권장)")

    # 깜박임 게이팅
    ap.add_argument("--blink_gate", action="store_true", help="깜박임/부분가림 게이팅 기본 on")
    ap.add_argument("--blink_v_ratio", type=float, default=0.6, help="hv < ratio * hv_ema → 프레임 무시")

    # Fallback 맵핑(보정 OFF 시)
    ap.add_argument("--fallback_slope_x", type=float, default=2.5, help="보정 OFF일 때 tanh 경사 X")
    ap.add_argument("--fallback_slope_y", type=float, default=2.5, help="보정 OFF일 때 tanh 경사 Y")

    # 한쪽 눈만 사용 옵션(원 요청의 인터페이스 유지)
    ap.add_argument("--use_left_eye_only", action="store_true", help="좌안만 사용")
    ap.add_argument("--use_right_eye_only", action="store_true", help="우안만 사용")

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
    """2D 원 최소자승. 실패 시 평균점 반환."""
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
    눈-정렬 좌표계 특징 (vx,vy) + 디버그 패키지 반환.
    - u축: 내/외측 눈꼬리를 잇는 단위 벡터
    - v축: u에 수직
    - iris 중심: 링 점들 원맞춤으로 추정
    - 정규화: u방향 폭(hw), v방향 높이(hv)로 나눠 무차원화
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


# === 2차 다항 회귀 ===
@dataclass
class Poly2Calibrator:
    cx: Optional[np.ndarray] = None
    cy: Optional[np.ndarray] = None
    samples_f: List[np.ndarray] = None  # (증폭된) 특징
    samples_s: List[np.ndarray] = None  # 스크린 타깃
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


# === v2.3: 감도 증폭기 ===
def apply_sensitivity(v: np.ndarray,
                      gain_xy: Tuple[float, float],
                      gamma_xy: Tuple[float, float],
                      deadzone: float,
                      sat: float) -> np.ndarray:
    """
    입력: v = (vx, vy)
    1) 데드존 제거: |v|<=deadzone → 0
    2) 감마 보정: sign(v)*(|v|^γ), 0<γ<=1 → 중심부 민감도 ↑
    3) 축별 게인: (gx, gy) 곱
    4) 포화: |v_amp|<=sat 로 클램프
    """
    x, y = float(v[0]), float(v[1])

    def proc(val, g, gm):
        s = 1.0 if val >= 0.0 else -1.0
        a = abs(val)
        if a <= deadzone:
            a = 0.0
        else:
            a = (a - deadzone) / max(1e-6, (1.0 - deadzone))  # 0~1 재정규화
        a = a ** gm
        out = s * a * g
        out = np.clip(out, -sat, sat)
        return out

    outx = proc(x, gain_xy[0], gamma_xy[0])
    outy = proc(y, gain_xy[1], gamma_xy[1])
    return np.array([outx, outy], dtype=np.float32)


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
    ema_feat = EMA2D(alpha=feat_alpha, clamp=None)          # 특징(v) 스무딩
    ema_norm = EMA2D(alpha=coord_alpha, clamp=(0.0, 1.0))   # (sx,sy) 스무딩
    neutral_ema = EMA2D(alpha=args.neutral_alpha, clamp=None) if args.neutral_track else None

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
    neutral_track_on = args.neutral_track

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

        # 가상 ArUco 오버레이
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
        feat_v_amp = None
        hv_curr_list = []

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            left_v, dbg_left = eye_feature_from_facemesh_ex(lm, mp_face_mesh, 'left')
            right_v, dbg_right = eye_feature_from_facemesh_ex(lm, mp_face_mesh, 'right')

            # 양안 통합
            vs = []
            if args.use_left_eye_only and left_v is not None:
                vs = [left_v]
                if dbg_left is not None: hv_curr_list.append(float(dbg_left['hv'][0]))
            elif args.use_right_eye_only and right_v is not None:
                vs = [right_v]
                if dbg_right is not None: hv_curr_list.append(float(dbg_right['hv'][0]))
            else:
                if left_v is not None:
                    vs.append(left_v); 
                    if dbg_left is not None: hv_curr_list.append(float(dbg_left['hv'][0]))
                if right_v is not None:
                    vs.append(right_v); 
                    if dbg_right is not None: hv_curr_list.append(float(dbg_right['hv'][0]))
            if len(vs) == 1:
                feat_v = vs[0]
            elif len(vs) == 2:
                feat_v = 0.5 * (vs[0] + vs[1])

            # 특징 EMA
            if feat_v is not None and np.isfinite(feat_v).all():
                feat_v = ema_feat.update(feat_v)

            # 깜박임 게이팅 신호 업데이트
            if blink_gate and len(hv_curr_list) > 0:
                hv_curr = float(np.mean(hv_curr_list))
                if hv_ema is None or not np.isfinite(hv_ema):
                    hv_ema = hv_curr
                else:
                    hv_ema = 0.9 * hv_ema + 0.1 * hv_curr

            # 중립 오프셋 EMA(고개/셋업 편차 제거)
            if neutral_track_on and (not cal.active) and feat_v is not None and np.isfinite(feat_v).all():
                neutral_ema.update(feat_v)

            # 감도 증폭
            if feat_v is not None and np.isfinite(feat_v).all():
                v_centered = feat_v.copy()
                if neutral_track_on and (neutral_ema is not None) and (neutral_ema.value is not None):
                    v_centered = v_centered - neutral_ema.value  # 중립 기준 제거
                feat_v_amp = apply_sensitivity(
                    v_centered,
                    (args.sens_gain_x, args.sens_gain_y),
                    (args.sens_gamma_x, args.sens_gamma_y),
                    args.sens_deadzone,
                    args.sens_sat
                )

            # 보정 적용 or Fallback
            gated_out = False
            if blink_gate and len(hv_curr_list) > 0:
                if hv_ema is not None and np.isfinite(hv_ema):
                    if hv_curr_list[-1] < args.blink_v_ratio * hv_ema:
                        gated_out = True

            if not gated_out:
                if cal.use and (feat_v_amp is not None):
                    mapped = cal.map(feat_v_amp)
                    if mapped is not None and np.isfinite(mapped).all():
                        gaze_pt01 = ema_norm.update(mapped)
                elif feat_v_amp is not None:
                    # 간이(무보정) 대범위 맵핑: tanh
                    sx = 0.5 + 0.5 * np.tanh(args.fallback_slope_x * float(feat_v_amp[0]))
                    sy = 0.5 + 0.5 * np.tanh(args.fallback_slope_y * float(feat_v_amp[1]))
                    gaze_pt01 = ema_norm.update(np.array([sx, sy], dtype=np.float32))

            # Eye 디버그
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
                f"Iris Gaze v2.3 [{mode}]  (Eye-aligned + SensAmp + Calib + Homog)",
                "[q] 종료  [h] 도움말  [d] 디버그  [r] EMA리셋",
                "[c] 캘리브 on/off  [SPACE] 샘플  [g] 보정 on/off  [S/L] 저장/로드",
                "[a] 호모그래피 on/off  [m] 수동ROI  [v] 가상ArUco",
                "[n] Neutral on/off  [N] Neutral reset  [X/x] X gain ±10%  [Y/y] Y gain ±10%",
                f"Calib: active={cal.active}, use={cal.use}, samples={len(cal.samples_f)}",
                f"H: use={use_homog}, ready={H_world is not None}, virtual={'ON' if use_virtual_aruco else 'OFF'}, manual={'ON' if manual_picker.active else 'OFF'}",
                f"Sens: gain=({args.sens_gain_x:.2f},{args.sens_gain_y:.2f}), gamma=({args.sens_gamma_x:.2f},{args.sens_gamma_y:.2f}), dz={args.sens_deadzone:.3f}, sat={args.sens_sat:.2f}",
                f"EMA(feat,coord)=({feat_alpha:.2f},{coord_alpha:.2f}), Neutral={'ON' if neutral_track_on else 'OFF'}, BlinkGate={'ON' if blink_gate else 'OFF'}",
            ]
            if cal.active and 0 <= cal_idx < len(CAL_POINTS):
                sx, sy = CAL_POINTS[cal_idx]
                help_lines.append(f"Target #{cal_idx+1}/{len(CAL_POINTS)} at ({sx:.2f},{sy:.2f}) → SPACE로 채집")
            if manual_picker.active:
                help_lines.append("수동 ROI: TL→TR→BR→BL 순서로 4번 클릭")
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
            if neutral_ema: neutral_ema.reset()
        elif key == ord('c'):
            cal.active = not cal.active
            cal_idx = 0
            print(f"[Calib] active={cal.active}")
        elif key == 32:  # SPACE
            if cal.active:
                if feat_v_amp is not None and np.isfinite(feat_v_amp).all():
                    sx, sy = CAL_POINTS[cal_idx]
                    cal.add(feat_v_amp, (sx, sy))  # ★ 증폭된 특징으로 학습
                    print(f"[Calib] sample {cal_idx+1}/{len(CAL_POINTS)} added")
                    cal_idx += 1
                    if cal_idx >= len(CAL_POINTS):
                        ok = cal.fit()
                        cal.active = False
                        cal.use = ok
                        print(f"[Calib] done. use={cal.use}")
                else:
                    print("[Calib] 현재 프레임에서 유효한 증폭 특징(v_amp)을 얻지 못했습니다.")
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
        elif key == ord('n'):
            neutral_track_on = not neutral_track_on
            print(f"[Neutral] track={neutral_track_on}")
        elif key == ord('N'):
            if neutral_ema: neutral_ema.reset()
            print("[Neutral] reset")
        elif key == ord('X'):
            args.sens_gain_x *= 1.1; print(f"[Sens] gain_x={args.sens_gain_x:.2f}")
        elif key == ord('x'):
            args.sens_gain_x *= 0.9; print(f"[Sens] gain_x={args.sens_gain_x:.2f}")
        elif key == ord('Y'):
            args.sens_gain_y *= 1.1; print(f"[Sens] gain_y={args.sens_gain_y:.2f}")
        elif key == ord('y'):
            args.sens_gain_y *= 0.9; print(f"[Sens] gain_y={args.sens_gain_y:.2f}")

    if world_cap is not None:
        world_cap.release()
    eye_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
