#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dual_camera_iris_gaze_v2_1.py

업그레이드 v2.1: 2대 카메라 + MediaPipe FaceMesh(iris) + OpenCV
- (A) 눈-정렬 좌표계(eye-aligned frame)에서의 홍채 중심 상대좌표 (vx, vy)
- (B) 2차 다항 캘리브레이션(sx, sy) ← (vx, vy)
- (C) ArUco 기반 월드 평면 정합(호모그래피) (선택)
- (D) 깜박임/부분가림 프레임 간이 게이팅 + 분리된 EMA(특징/좌표)

핵심 변화(v2.1)
- "눈 컨투어 bbox" 기준 대신, **눈꼬리(좌/우)로 정의한 로컬 축(u,v)** 에서 홍채 중심을 투영하여
  회전(roll) 변화에 강하고, **순수 안구 이동**에 민감한 특징 (vx, vy)를 사용
- 홍채 중심은 iris 링(4~5점)에 대해 **원 맞춤(Least-Squares)** 으로 계산(평균 대비 안정)

키보드:
  q : 종료
  h : 도움말 HUD 토글
  d : 디버그(눈 창에 축/홍채 표시) 토글
  r : EMA 리셋
  c : 9-포인트 캘리브 시작/종료
  SPACE : (캘리 모드) 현재 프레임 샘플 채집
  g : 보정(캘리브 매핑) on/off
  S/L : 보정 계수 저장/로드 (npz)
  a : ArUco 호모그래피 on/off
  b : 깜박임 게이팅 on/off

실행 예:
  python dual_camera_iris_gaze_v2_1.py --world_cam 0 --eye_cam 1 --flip_eye --feat_gain 1.2
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
# - opencv-contrib-python 가 설치되어 있어야 함
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
    """2차원 점에 대한 지수이동평균(EMA) 필터.
    - clamp: (min, max) 범위로 클램프. None이면 클램프 없음.
    """
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
    ap = argparse.ArgumentParser(description="Dual-Camera Iris Gaze v2.1 (eye-aligned features + calibration + homography)")
    ap.add_argument("--world_cam", type=int, default=0, help="월드(배경) 카메라 OpenCV 인덱스")
    ap.add_argument("--eye_cam", type=int, default=1, help="눈/얼굴 카메라 OpenCV 인덱스")
    ap.add_argument("--world_w", type=int, default=1280, help="월드 카메라 캡처 가로 해상도(요청값)")
    ap.add_argument("--world_h", type=int, default=720, help="월드 카메라 캡처 세로 해상도(요청값)")
    ap.add_argument("--eye_w", type=int, default=640, help="눈 카메라 캡처 가로 해상도(요청값)")
    ap.add_argument("--eye_h", type=int, default=480, help="눈 카메라 캡처 세로 해상도(요청값)")
    ap.add_argument("--flip_eye", action="store_true", help="눈 카메라 영상을 좌우 반전(셀피 느낌)")
    ap.add_argument("--draw_scale", type=float, default=1.0, help="도형/텍스트 스케일 팩터")

    # 추적 신뢰도
    ap.add_argument("--min_det_conf", type=float, default=0.5, help="FaceMesh min_detection_confidence")
    ap.add_argument("--min_trk_conf", type=float, default=0.5, help="FaceMesh min_tracking_confidence")

    # EMA 설정 (기본값은 예전 --ema_alpha를 폴백으로 사용)
    ap.add_argument("--ema_alpha", type=float, default=0.25, help="(레거시) EMA 기본 알파. 아래 두 옵션이 None이면 이 값을 사용")
    ap.add_argument("--feat_ema_alpha", type=float, default=None, help="특징(vx,vy) EMA 알파(미지정 시 --ema_alpha 사용)")
    ap.add_argument("--coord_ema_alpha", type=float, default=None, help="화면좌표(sx,sy) EMA 알파(미지정 시 --ema_alpha 사용)")

    # 특징 스케일 부스트
    ap.add_argument("--feat_gain", type=float, default=1.0, help="특징 (vx,vy)에 곱하는 감도 스케일(캘리브가 흡수하지만 디버깅/체감용)")

    # 한쪽 눈만 사용
    ap.add_argument("--use_left_eye_only", action="store_true", help="좌안(Left)만 사용")
    ap.add_argument("--use_right_eye_only", action="store_true", help="우안(Right)만 사용")

    # 캘리브 파일
    ap.add_argument("--calib_path", type=str, default="gaze_calib.npz", help="캘리브레이션 계수 저장/로드 경로")

    # 깜박임/부분가림 게이팅
    ap.add_argument("--blink_gate", action="store_true", help="깜박임/부분가림 게이팅 기본 on")
    ap.add_argument("--blink_v_ratio", type=float, default=0.6, help="hv < ratio * hv_ema 이면 프레임 무시 (0.5~0.7 권장)")

    return ap.parse_args()


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    """OpenCV 카메라 열기 유틸 (Windows: CAP_DSHOW 우선 시도)."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"카메라 열기 실패 (index={index})")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def draw_hud(frame_bgr: np.ndarray, text_lines: List[str], scale: float = 1.0, org=(10,20)):
    """세계(월드) 프레임 위 왼쪽 상단에 도움말/상태 텍스트 표시."""
    x, y = org
    for t in text_lines:
        cv2.putText(frame_bgr, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += int(20 * scale)


def _indices_from_connections(conns):
    """MediaPipe connection set에서 인덱스 집합 추출."""
    idxs = set()
    for a, b in conns:
        idxs.add(a); idxs.add(b)
    return sorted(list(idxs))


def _fit_circle_ls(pts: np.ndarray):
    """
    2D 원 Least-Squares.
    pts: (N,2), N>=3 권장. 반환: (center(x,y), radius). 실패 시 (mean, NaN)
    좌표계는 입력 그대로(정규화 좌표).
    """
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
    눈 특징 (vx,vy) + 디버그 정보 반환.
    - 로컬 축 u: 내/외측 눈꼬리(eye contour에서 x min/max)
    - 로컬 축 v: u에 수직 (시계/반시계 상관없음)
    - 정규화: 가로 hw=|pR-pL|, 세로 hv=eye_pts를 v축으로 투영한 높이
    - 홍채 중심: iris 링(4~5점)에 대해 원 맞춤
    반환:
      feat: np.array([vx,vy])  (대략 -0.5~+0.5 범위, 사람마다 다름)
      dbg: {'pL','pR','u','v','eye_ctr','iris_c','hw','hv'}  (모두 정규화 좌표계 기준)
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

    # 1) 눈꼬리(좌/우) 및 로컬 축
    iL = int(np.argmin(eye_pts[:,0])); iR = int(np.argmax(eye_pts[:,0]))
    pL, pR = eye_pts[iL], eye_pts[iR]
    u = pR - pL
    hw = float(np.linalg.norm(u))
    if hw < 1e-6:
        return None, None
    u = u / hw
    v = np.array([-u[1], u[0]], dtype=np.float32)

    # 2) 세로 높이 hv
    proj_v = eye_pts @ v
    hv = float(proj_v.max() - proj_v.min())
    if hv < 1e-6:
        hv = 1e-6

    # 3) 홍채 중심(원 맞춤)
    iris_c, _ = _fit_circle_ls(iris_pts)

    # 4) 기준점: 눈꼬리 중점
    eye_ctr = 0.5 * (pL + pR)

    # 5) 로컬 좌표로 투영 후 정규화
    d = iris_c - eye_ctr
    vx = float(d @ u) / hw
    vy = float(d @ v) / hv

    feat = np.array([vx, vy], dtype=np.float32)
    dbg = {
        'pL': pL, 'pR': pR, 'u': u, 'v': v, 'eye_ctr': eye_ctr,
        'iris_c': iris_c, 'hw': np.array([hw], np.float32), 'hv': np.array([hv], np.float32)
    }
    return feat, dbg


# === 2차 다항 회귀 기반 캘리브레이터 ===
@dataclass
class Poly2Calibrator:
    cx: Optional[np.ndarray] = None  # (6,)
    cy: Optional[np.ndarray] = None  # (6,)
    samples_f: List[np.ndarray] = None  # (vx,vy)
    samples_s: List[np.ndarray] = None  # (sx,sy)
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


# === ArUco 기반 호모그래피 추정 ===
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
        center = c[0].mean(axis=0)  # (x,y) 픽셀
        id_to_center[int(i)] = center.astype(np.float32)
    try:
        pts_dst = np.float32([
            id_to_center[_ARUCO_IDS_WANT[0]],
            id_to_center[_ARUCO_IDS_WANT[1]],
            id_to_center[_ARUCO_IDS_WANT[2]],
            id_to_center[_ARUCO_IDS_WANT[3]],
        ])
    except KeyError:
        return None
    pts_src = np.float32([[0,0],[1,0],[1,1],[0,1]])  # 정규 스크린
    H, _ = cv2.findHomography(pts_src, pts_dst, method=cv2.RANSAC)
    return H


def main():
    args = parse_args()
    feat_alpha = args.feat_ema_alpha if args.feat_ema_alpha is not None else args.ema_alpha
    coord_alpha = args.coord_ema_alpha if args.coord_ema_alpha is not None else args.ema_alpha

    world_cap = open_camera(args.world_cam, args.world_w, args.world_h)
    eye_cap = open_camera(args.eye_cam, args.eye_w, args.eye_h)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=args.min_det_conf,
        min_tracking_confidence=args.min_trk_conf,
    )

    ema_feat = EMA2D(alpha=feat_alpha, clamp=None)
    ema_norm = EMA2D(alpha=coord_alpha, clamp=(0.0, 1.0))

    cal = Poly2Calibrator()
    cal_idx = 0

    use_homog = False
    H_world = None

    blink_gate = args.blink_gate
    hv_ema = None

    show_help = True
    show_debug = False

    while True:
        ok_world, world_bgr = world_cap.read()
        ok_eye, eye_bgr = eye_cap.read()
        if not ok_world or not ok_eye:
            print("경고: 카메라 프레임 읽기 실패(월드/눈 중 하나).")
            break

        if args.flip_eye:
            eye_bgr = cv2.flip(eye_bgr, 1)

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
                    vs.append(left_v)
                    if dbg_left is not None: hv_curr_list.append(float(dbg_left['hv'][0]))
                if right_v is not None:
                    vs.append(right_v)
                    if dbg_right is not None: hv_curr_list.append(float(dbg_right['hv'][0]))
                if len(vs) == 1:
                    feat_v = vs[0]
                elif len(vs) == 2:
                    feat_v = 0.5 * (vs[0] + vs[1])

            if feat_v is not None and np.isfinite(feat_v).all():
                feat_v = feat_v * float(args.feat_gain)
                feat_v = ema_feat.update(feat_v)

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

        if use_homog:
            H = estimate_homography_from_aruco(world_bgr)
            if H is not None:
                H_world = H

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

        if show_help:
            help_lines = [
                "Dual-Camera Iris Gaze v2.1 (Eye-aligned + Calib + Homog)",
                "[q] 종료  [h] 도움말  [d] 디버그  [r] EMA리셋",
                "[c] 캘리브 모드  [SPACE] 샘플추가  [g] 보정 on/off  [S/L] 저장/로드",
                "[a] ArUco 호모그래피 on/off  [b] 깜박임 게이팅 on/off",
                f"옵션: left-only={args.use_left_eye_only}, right-only={args.use_right_eye_only}, flip_eye={args.flip_eye}",
                f"Calib: active={cal.active}, use={cal.use}, samples={len(cal.samples_f)}",
                f"Homog: enabled={use_homog}, H ready={H_world is not None}, ArUco={'OK' if HAVE_ARUCO else 'N/A'}",
                f"EMA(feat,coord)=({feat_alpha:.2f},{coord_alpha:.2f}), feat_gain={args.feat_gain:.2f}, blink_gate={blink_gate}",
            ]
            if cal.active and 0 <= cal_idx < len(CAL_POINTS):
                sx, sy = CAL_POINTS[cal_idx]
                help_lines.append(f"Target #{cal_idx+1}/{len(CAL_POINTS)} at ({sx:.2f},{sy:.2f}) → SPACE로 채집")
            draw_hud(world_bgr, help_lines, scale=args.draw_scale)

        if cal.active and 0 <= cal_idx < len(CAL_POINTS):
            h_w, w_w = world_bgr.shape[:2]
            sx, sy = CAL_POINTS[cal_idx]
            cx, cy = int(sx*w_w), int(sy*h_w)
            cv2.circle(world_bgr, (cx,cy), int(14*args.draw_scale), (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow("World (Gaze Overlay)", world_bgr)
        if show_debug:
            cv2.imshow("Eye (Debug)", eye_bgr)
        else:
            w = min(480, eye_bgr.shape[1])
            h = int(w * eye_bgr.shape[0] / max(1, eye_bgr.shape[1]))
            cv2.imshow("Eye (Debug)", cv2.resize(eye_bgr, (w, h)))

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
            if not HAVE_ARUCO:
                print("[H] opencv-contrib-python(ArUco)이 설치되어 있지 않습니다.")
            use_homog = not use_homog
            print(f"[H] use_homog={use_homog}")
        elif key == ord('b'):
            blink_gate = not blink_gate
            print(f"[Gate] blink_gate={blink_gate}")

    world_cap.release()
    eye_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
