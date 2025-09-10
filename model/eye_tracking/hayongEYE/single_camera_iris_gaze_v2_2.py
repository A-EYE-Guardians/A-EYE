#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
single_camera_iris_gaze_v2_2.py

v2.2: 단일 카메라(eye)만으로도 캘리브레이션/월드정합을 테스트할 수 있는 모드 추가
- Single-Cam 모드(--single_cam): eye 카메라 프레임을 '월드' 캔버스로 사용
  * 9점 캘리 타깃을 Eye 창 위에 표시 → 그 점을 응시하며 SPACE로 샘플 수집
  * 월드 정합(호모그래피) 테스트 3가지:
    (1) Identity: 화면 정규좌표 → Eye 프레임 픽셀 1:1 매핑
    (2) Manual ROI: 'm'로 네 꼭짓점 클릭(TL,TR,BR,BL) → 그 사각형으로 정규좌표를 투영
    (3) Virtual ArUco: 'v'로 가상 ArUco 4개(TL=0, TR=1, BR=2, BL=3)를 Eye 프레임에 오버레이 후
        감지하여 H를 추정(opencv-contrib-python 필요)

기존 v2.1의 개선점(눈-정렬 좌표계 특징, 원맞춤 iris 중심, 깜박임 게이팅, EMA 분리)은 그대로 유지.

키보드:
  q : 종료
  h : 도움말 HUD 토글
  d : Eye 창에 축/홍채 표시 토글
  r : EMA 리셋
  c : 9-포인트 캘리브 시작/종료
  SPACE : (캘리 모드) 현재 프레임 샘플 채집
  g : 보정(캘리브 매핑) on/off
  S/L : 보정 계수 저장/로드 (npz)
  a : 호모그래피 사용 on/off (H가 준비되어야 적용)
  m : 수동 ROI 호모그래피 모드 토글(4클릭으로 TL,TR,BR,BL 찍기)
  v : 가상 ArUco 오버레이 on/off (contrib 필요)

실행 예:
  python dual_camera_iris_gaze_v2_2.py --single_cam --eye_cam 0 --flip_eye --feat_gain 1.2 --blink_gate
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
    ap = argparse.ArgumentParser(description="Dual-Camera Iris Gaze v2.2 (single-cam ready)")
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

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            left_v, dbg_left = eye_feature_from_facemesh_ex(lm, mp_face_mesh, 'left')
            right_v, dbg_right = eye_feature_from_facemesh_ex(lm, mp_face_mesh, 'right')

            if args.use_left_eye_only and left_v is not None:
                feat_v = left_v
                if dbg_left is not None: hv_curr_list.append(float(dbg_left['hv'][0]))
            elif args.use_right_eye_only and right_v is not None:
                feat_v = right_v
                if dbg_right is not None: hv_curr_list.append(float(dbg_right['hv'][0]))
            else:
                vs = []
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

            if feat_v is not None and np.isfinite(feat_v).all():
                feat_v = feat_v * float(args.feat_gain)
                feat_v = ema_feat.update(feat_v)

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
                        gaze_pt01 = ema_norm.update(mapped)
            else:
                # 보정 미사용시: iris 중심 평균 폴백
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
                f"Dual-Camera Iris Gaze v2.2 [{mode}]  (Eye-aligned + Calib + Homog)",
                "[q] 종료  [h] 도움말  [d] 디버그  [r] EMA리셋",
                "[c] 캘리브 모드  [SPACE] 샘플추가  [g] 보정 on/off  [S/L] 저장/로드",
                "[a] 호모그래피 on/off  [m] 수동ROI(4클릭)  [v] 가상ArUco on/off",
                f"옵션: left-only={args.use_left_eye_only}, right-only={args.use_right_eye_only}, flip_eye={args.flip_eye}",
                f"Calib: active={cal.active}, use={cal.use}, samples={len(cal.samples_f)}",
                f"H: use={use_homog}, ready={H_world is not None}, virtual={'ON' if use_virtual_aruco else 'OFF'}, manual={'ON' if manual_picker.active else 'OFF'}",
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

    if world_cap is not None:
        world_cap.release()
    eye_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
