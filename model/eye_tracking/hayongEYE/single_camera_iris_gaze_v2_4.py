#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
single_camera_iris_gaze_v2_4.py

v2.4 주요 변경 사항 (v2.3 → v2.4)
- [세로 감도/대각선 정확도] 가장 큰 개선점!
  1) "로컬 세로 정규화" 도입: vy를 '현재 가로 위치(u)에서의 위/아래 눈꺼풀 경계까지 세로 여백'으로 정규화
     → 중앙/가장자리에서 세로 여백이 달라도 균일한 감도 확보, 대각선 응답 자연스러움 ↑
  2) 캘리브레이터 선형 프리보정(화이트닝): (증폭 특성)의 공분산을 이용해 선형 화이트닝 후 2차 다항 회귀
     → u,v 상관에 의한 대각 왜곡 완화, 적은 샘플로도 안정적 캘리브
  3) 깜박임 게이팅 기준 개선: 기존 hv 기반 → '로컬 세로 반폭(gap_half)' 기반(옵션)
     → 실제 눈꺼풀 닫힘/가림에 더 민감하고 견고
  4) 디버그 정보 강화: (u,v) 프레임의 상/하 경계 곡선과 iris 위치를 Eye 디버그에 오버레이(스몰 인셋)

- 기존 파이프라인 및 키 바인딩은 동일:
  Eye 정렬 좌표 → 감도 증폭(데드존/감마/게인/포화) → 화이트닝+2차다항 캘리브 → 호모그래피 → 시각화
  q 종료 | h HUD | d 디버그 | r EMA리셋
  c 캘리브 on/off | SPACE 샘플 | g 보정 on/off | S/L 저장/로드
  a 호모그래피 on/off | m 수동ROI | v 가상ArUco
  n 중립 오프셋 on/off | N 즉시 리셋
  X/x Y/y 감도 ±10%

※ opencv-contrib-python을 설치해야 ArUco 사용 가능.
※ mediapipe, numpy, opencv-python 설치 필요.
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
    ap = argparse.ArgumentParser(description="Single-Cam Iris Gaze v2.4 (local-vertical norm + whitening + sensitivity + calibration + homography)")
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
    ap.add_argument("--calib_path", type=str, default="gaze_calib_v24.npz", help="캘리브 계수 저장/로드 경로")

    # 중립 오프셋(고개/개인 편차 보정) 추적
    ap.add_argument("--neutral_track", action="store_true", default=True, help="중립 오프셋 EMA 추적 on")
    ap.add_argument("--no_neutral_track", dest="neutral_track", action="store_false", help="중립 오프셋 EMA 추적 off")
    ap.add_argument("--neutral_alpha", type=float, default=0.02, help="중립 오프셋 EMA 계수(느리게 적응 권장)")

    # 깜박임 게이팅
    ap.add_argument("--blink_gate", action="store_true", help="깜박임/부분가림 게이팅 on")
    ap.add_argument("--blink_ratio", type=float, default=0.6, help="gap_half < ratio * gap_ema → 프레임 무시 (v2.4: gap 기반)")

    # Fallback 맵핑(보정 OFF 시)
    ap.add_argument("--fallback_slope_x", type=float, default=2.5, help="보정 OFF일 때 tanh 경사 X")
    ap.add_argument("--fallback_slope_y", type=float, default=2.5, help="보정 OFF일 때 tanh 경사 Y")

    # 한쪽 눈만 사용 옵션
    ap.add_argument("--use_left_eye_only", action="store_true", help="좌안만 사용")
    ap.add_argument("--use_right_eye_only", action="store_true", help="우안만 사용")

    return ap.parse_args()


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    """Windows 환경에서 CAP_DSHOW 우선 시도, 실패 시 기본 백엔드 재시도."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"카메라 열기 실패 (index={index})")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def draw_hud(frame_bgr: np.ndarray, text_lines: List[str], scale: float = 1.0, org=(10, 20)):
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


# === 2차 다항 회귀 (v2.4: 선형 화이트닝 프리보정 추가) ===
@dataclass
class Poly2Calibrator:
    cx: Optional[np.ndarray] = None  # (6,)
    cy: Optional[np.ndarray] = None  # (6,)
    samples_f: List[np.ndarray] = None  # (증폭된) 특징 (N,2)
    samples_s: List[np.ndarray] = None  # 스크린 타깃 (N,2)
    active: bool = False
    use: bool = False
    # v2.4 추가: 화이트닝 파라미터
    mu: Optional[np.ndarray] = None  # (1,2)
    W: Optional[np.ndarray] = None   # (2,2)

    def __post_init__(self):
        self.samples_f = []
        self.samples_s = []

    @staticmethod
    def _phi(v: np.ndarray) -> np.ndarray:
        """2차 다항 특징 벡터: [x, y, xy, x^2, y^2, 1]"""
        x, y = float(v[0]), float(v[1])
        return np.array([x, y, x*y, x*x, y*y, 1.0], dtype=np.float32)

    def add(self, feat_vxy, scr_sxy):
        self.samples_f.append(np.array(feat_vxy, dtype=np.float32))
        self.samples_s.append(np.array(scr_sxy, dtype=np.float32))

    def fit(self) -> bool:
        """v_amp 샘플들을 화이트닝 후 2차 다항 선형회귀."""
        if len(self.samples_f) < 6:
            return False
        F = np.stack(self.samples_f, axis=0)  # (N,2)
        S = np.stack(self.samples_s, axis=0)  # (N,2)

        # v2.4: 선형 화이트닝 (대각 왜곡/상관 완화)
        self.mu = F.mean(axis=0, keepdims=True)  # (1,2)
        C = np.cov((F - self.mu).T) + 1e-6*np.eye(2, dtype=np.float32)
        try:
            L = np.linalg.cholesky(C)
            self.W = np.linalg.inv(L).astype(np.float32)  # (2,2)
        except np.linalg.LinAlgError:
            # 드물게 수치 문제가 있으면 단위 행렬 사용
            self.W = np.eye(2, dtype=np.float32)

        Fw = (F - self.mu) @ self.W.T  # (N,2)

        Phi = np.stack([self._phi(v) for v in Fw], axis=0)  # (N,6)
        # x, y 각각 최소자승
        self.cx, *_ = np.linalg.lstsq(Phi, S[:,0], rcond=None)
        self.cy, *_ = np.linalg.lstsq(Phi, S[:,1], rcond=None)
        return True

    def map(self, vxy: np.ndarray) -> Optional[np.ndarray]:
        """입력 vxy(증폭 특징)에 화이트닝 적용 후 2차 다항을 거쳐 (sx,sy) 예측."""
        if self.cx is None or self.cy is None:
            return None
        vv = np.array(vxy, dtype=np.float32)
        if self.mu is not None and self.W is not None:
            vv = (vv - self.mu[0]) @ self.W.T
        phi = self._phi(vv)
        sx = float(phi @ self.cx)
        sy = float(phi @ self.cy)
        return np.clip(np.array([sx, sy], dtype=np.float32), 0.0, 1.0)

    def save(self, path="gaze_calib_v24.npz"):
        if self.cx is not None and self.cy is not None:
            np.savez(path, cx=self.cx, cy=self.cy, mu=(self.mu if self.mu is not None else np.zeros((1,2), np.float32)), W=(self.W if self.W is not None else np.eye(2, dtype=np.float32)))

    def load(self, path="gaze_calib_v24.npz"):
        z = np.load(path)
        self.cx = z["cx"]; self.cy = z["cy"]
        self.mu = z["mu"]; self.W = z["W"]
        self.use = True


# === 캘리브 타깃 포인트 (9점) ===
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
        pts_dst = np.float32([id_to_center[0], id_to_center[1], id_to_center[2], id_to_center[3]])
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
    markers = []
    for i in ids:
        img = aruco.drawMarker(_ARUCO_DICT, i, ms)
        markers.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    pos = [
        (mg, mg),                       # TL
        (w - mg - ms, mg),              # TR
        (w - mg - ms, h - mg - ms),     # BR
        (mg, h - mg - ms)               # BL
    ]
    for p, m in zip(pos, markers):
        x, y = p
        frame[y:y+ms, x:x+ms] = m
    return True


# === v2.4 핵심: (u,v) 로컬 세로 정규화가 적용된 특징 추출 ===
def eye_feature_from_facemesh_ex(face_landmarks, mp_face_mesh, which: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
    """
    반환: (vx, vy), dbg
    - (u,v) 축: u = 외/내측 눈꼬리 방향 단위벡터, v = u에 수직
    - iris 중심을 (u,v) 평면으로 사영
    - vx: 가로는 기존처럼 눈 가로폭(hw)로 정규화
    - vy: v2.4 '로컬 세로 정규화' → 현재 u에서의 위/아래 눈꺼풀 경계까지의 반폭(gap_half)로 정규화
      => 중앙/가장자리 등 위치에 따른 세로 여백 차이를 흡수
    - dbg에 곡선 계수/갭/중앙선 등 진단정보 포함
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

    # u,v 축 생성
    iL = int(np.argmin(eye_pts[:,0])); iR = int(np.argmax(eye_pts[:,0]))
    pL, pR = eye_pts[iL], eye_pts[iR]
    u = pR - pL
    hw = float(np.linalg.norm(u))
    if hw < 1e-6:
        return None, None
    u = u / hw
    v = np.array([-u[1], u[0]], dtype=np.float32)

    # iris/eye center 추정
    iris_c, _ = _fit_circle_ls(iris_pts)
    eye_ctr = 0.5 * (pL + pR)

    # 모든 눈 포인트를 (u,v)평면으로 투영
    proj_u = eye_pts @ u    # (M,)
    proj_v = eye_pts @ v    # (M,)

    # 위/아래 경계 근사: v가 큰 쪽/작은 쪽으로 1차 분할 후 각각 2차 다항 피팅
    median_v = np.median(proj_v)
    upper_mask = proj_v >= median_v
    u_upper, v_upper = proj_u[upper_mask], proj_v[upper_mask]
    u_lower, v_lower = proj_u[~upper_mask], proj_v[~upper_mask]

    def fit_quad(u_arr, v_arr):
        A = np.stack([u_arr*u_arr, u_arr, np.ones_like(u_arr)], axis=1)
        coef, *_ = np.linalg.lstsq(A, v_arr, rcond=None)
        return coef  # a,b,c

    have_local_gap = False
    a1=b1=c1=a2=b2=c2=0.0
    if len(u_upper) >= 6 and len(u_lower) >= 6:
        a1,b1,c1 = fit_quad(u_upper, v_upper)
        a2,b2,c2 = fit_quad(u_lower, v_lower)
        have_local_gap = True

    # iris/center를 (u,v)로
    iris_uv = np.array([iris_c @ u, iris_c @ v], np.float32)
    eye_ctr_uv = np.array([eye_ctr @ u, eye_ctr @ v], np.float32)

    # vx: 기존과 동일(가로 폭으로 정규화)
    vx = (iris_uv[0] - eye_ctr_uv[0]) / max(1e-6, hw)

    # vy: v2.4 로컬 세로 정규화(가장 큰 변화)
    if have_local_gap:
        uu = iris_uv[0]  # 현재 가로 위치
        # 위/아래 경계 v값
        vu_up = a1*uu*uu + b1*uu + c1
        vu_lo = a2*uu*uu + b2*uu + c2
        gap_half = max(1e-6, 0.5*(vu_up - vu_lo))
        v_mid = 0.5*(vu_up + vu_lo)
        vy = (iris_uv[1] - v_mid) / gap_half
    else:
        # 폴백: 전체 높이 정규화 (이전 방식)
        hv = float(proj_v.max() - proj_v.min()) or 1e-6
        gap_half = 0.5 * hv
        v_mid = eye_ctr_uv[1]
        vy = (iris_uv[1] - v_mid) / gap_half

    feat = np.array([vx, vy], dtype=np.float32)

    # 디버그 패키지: 튜닝/진단용 정보 전달
    dbg = {
        'pL': pL, 'pR': pR, 'u': u, 'v': v,
        'eye_ctr': eye_ctr, 'iris_c': iris_c,
        'hw': np.array([hw], np.float32),
        'proj_u': proj_u, 'proj_v': proj_v,
        'upper_coef': np.array([a1,b1,c1], np.float32),
        'lower_coef': np.array([a2,b2,c2], np.float32),
        'have_local_gap': np.array([1.0 if have_local_gap else 0.0], np.float32),
        'gap_half': np.array([gap_half], np.float32),
        'v_mid': np.array([v_mid], np.float32),
        'iris_uv': iris_uv, 'eye_ctr_uv': eye_ctr_uv
    }
    return feat, dbg


# === v2.3 감도 증폭기 (그대로 사용) ===
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

    # MediaPipe 초기화
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=args.min_det_conf, min_tracking_confidence=args.min_trk_conf,
    )

    # 필터/상태
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

    # 깜박임 게이팅: v2.4에서는 gap_half 기반으로 업데이트
    blink_gate = args.blink_gate
    gap_half_ema = None

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

        # 가상 ArUco 오버레이(선택)
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
        last_gap_half = None

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            left_v, dbg_left = eye_feature_from_facemesh_ex(lm, mp_face_mesh, 'left')
            right_v, dbg_right = eye_feature_from_facemesh_ex(lm, mp_face_mesh, 'right')

            # 양안 통합: 품질 가중치(간단 버전) → gap_half 큰 쪽 가중치↑
            candidates = []
            if args.use_left_eye_only and left_v is not None and dbg_left is not None:
                candidates = [(left_v, float(dbg_left['gap_half'][0]))]
            elif args.use_right_eye_only and right_v is not None and dbg_right is not None:
                candidates = [(right_v, float(dbg_right['gap_half'][0]))]
            else:
                if left_v is not None and dbg_left is not None:
                    candidates.append((left_v, float(dbg_left['gap_half'][0])))
                if right_v is not None and dbg_right is not None:
                    candidates.append((right_v, float(dbg_right['gap_half'][0])))

            if len(candidates) == 1:
                feat_v, last_gap_half = candidates[0]
            elif len(candidates) == 2:
                (v1, g1), (v2, g2) = candidates
                w1 = g1; w2 = g2
                if not np.isfinite(w1): w1 = 0.0
                if not np.isfinite(w2): w2 = 0.0
                if w1 + w2 < 1e-6:
                    feat_v = 0.5*(v1+v2); last_gap_half = 0.5*(g1+g2)
                else:
                    feat_v = (w1*v1 + w2*v2) / (w1 + w2)
                    last_gap_half = (w1*g1 + w2*g2) / (w1 + w2)

            # 특징 EMA
            if feat_v is not None and np.isfinite(feat_v).all():
                feat_v = ema_feat.update(feat_v)

            # 깜박임 게이팅 신호 업데이트: gap_half 기반
            if blink_gate and last_gap_half is not None and np.isfinite(last_gap_half):
                if gap_half_ema is None or not np.isfinite(gap_half_ema):
                    gap_half_ema = last_gap_half
                else:
                    gap_half_ema = 0.9 * gap_half_ema + 0.1 * last_gap_half

            # 중립 오프셋 EMA(고개/셋업 편차 제거)
            if neutral_track_on and (not cal.active) and feat_v is not None and np.isfinite(feat_v).all():
                neutral_ema.update(feat_v)

            # 감도 증폭 (중립 오프셋 제거 후)
            if feat_v is not None and np.isfinite(feat_v).all():
                v_centered = feat_v.copy()
                if neutral_track_on and (neutral_ema is not None) and (neutral_ema.value is not None):
                    v_centered = v_centered - neutral_ema.value
                feat_v_amp = apply_sensitivity(
                    v_centered,
                    (args.sens_gain_x, args.sens_gain_y),
                    (args.sens_gamma_x, args.sens_gamma_y),
                    args.sens_deadzone,
                    args.sens_sat
                )

            # 보정 적용 or Fallback (깜박임 게이팅 반영)
            gated_out = False
            if blink_gate and last_gap_half is not None and gap_half_ema is not None and np.isfinite(gap_half_ema):
                if last_gap_half < args.blink_ratio * gap_half_ema:
                    gated_out = True

            if not gated_out:
                if cal.use and (feat_v_amp is not None):
                    mapped = cal.map(feat_v_amp)
                    if mapped is not None and np.isfinite(mapped).all():
                        gaze_pt01 = ema_norm.update(mapped)
                elif feat_v_amp is not None:
                    # 간이(무보정) tanh 맵핑
                    sx = 0.5 + 0.5 * np.tanh(args.fallback_slope_x * float(feat_v_amp[0]))
                    sy = 0.5 + 0.5 * np.tanh(args.fallback_slope_y * float(feat_v_amp[1]))
                    gaze_pt01 = ema_norm.update(np.array([sx, sy], dtype=np.float32))

            # Eye 디버그: (u,v) 경계 곡선, iris 표시(인셋)
            if show_debug:
                def draw_uv_inset(canvas, dbg, at=(8,8), size=(200,140)):
                    if dbg is None:
                        return
                    x0,y0 = at; W,H = size
                    # 배경
                    cv2.rectangle(canvas, (x0,y0), (x0+W, y0+H), (20,20,20), -1)
                    cv2.rectangle(canvas, (x0,y0), (x0+W, y0+H), (80,80,80), 1)
                    proj_u = dbg['proj_u']; proj_v = dbg['proj_v']
                    a1,b1,c1 = dbg['upper_coef']; a2,b2,c2 = dbg['lower_coef']
                    have_local = bool(int(dbg['have_local_gap'][0]))
                    # 좌표 정규화용 범위
                    umin, umax = float(np.min(proj_u)), float(np.max(proj_u))
                    vmin, vmax = float(np.min(proj_v)), float(np.max(proj_v))
                    if umax - umin < 1e-6: umax = umin + 1e-6
                    if vmax - vmin < 1e-6: vmax = vmin + 1e-6
                    def to_px(uu, vv):
                        xx = int(x0 + (uu-umin)/(umax-umin) * (W-1))
                        yy = int(y0 + (1.0 - (vv-vmin)/(vmax-vmin)) * (H-1))
                        return xx, yy
                    # 점들 간략 도트
                    for uu, vv in zip(proj_u[::2], proj_v[::2]):
                        px, py = to_px(float(uu), float(vv))
                        cv2.circle(canvas, (px,py), 1, (160,160,160), -1)
                    # 상/하 곡선
                    if have_local:
                        for t in np.linspace(0,1,80):
                            uu = umin + t*(umax-umin)
                            vu_up = a1*uu*uu + b1*uu + c1
                            vu_lo = a2*uu*uu + b2*uu + c2
                            px1, py1 = to_px(uu, vu_up)
                            px2, py2 = to_px(uu, vu_lo)
                            canvas[py1:py1+1, max(px1-1, x0):min(px1+2, x0+W)] = (0,255,0)
                            canvas[py2:py2+1, max(px2-1, x0):min(px2+2, x0+W)] = (0,200,200)
                    # iris 현재 위치
                    uu = float(dbg['iris_uv'][0]); vv = float(dbg['iris_uv'][1])
                    px, py = to_px(uu, vv)
                    cv2.circle(canvas, (px,py), 3, (0,0,255), -1)
                    cv2.putText(canvas, "UV inset", (x0+6,y0+16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1, cv2.LINE_AA)

                # 원래 eye 디버그 + UV 인셋
                h_eye, w_eye = eye_bgr.shape[:2]
                w_small = min(480, w_eye)
                h_small = int(w_small * h_eye / max(1, w_eye))
                eye_dbg = cv2.resize(eye_bgr, (w_small, h_small))
                draw_uv_inset(eye_dbg, dbg_left, at=(8,8))
                draw_uv_inset(eye_dbg, dbg_right, at=(8,8+150))
                cv2.imshow("Eye (Debug)", eye_dbg)
            else:
                # 축소 표시
                h_eye, w_eye = eye_bgr.shape[:2]
                w_small = min(480, w_eye)
                h_small = int(w_small * h_eye / max(1, w_eye))
                cv2.imshow("Eye (Debug)", cv2.resize(eye_bgr, (w_small, h_small)))

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
                f"Iris Gaze v2.4 [{mode}]  (Local-Vert Norm + Whitening + SensAmp + Calib + Homog)",
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

        # 키 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'):
            show_help = not show_help
        elif key == ord('d'):
            show_debug = not show_debug
        elif key == ord('r'):
            ema_feat.reset(); ema_norm.reset(); gap_half_ema = None
            if neutral_ema: neutral_ema.reset()
        elif key == ord('c'):
            cal.active = not cal.active
            cal_idx = 0
            print(f"[Calib] active={cal.active}")
        elif key == 32:  # SPACE
            if cal.active:
                if feat_v_amp is not None and np.isfinite(feat_v_amp).all():
                    sx, sy = CAL_POINTS[cal_idx]
                    cal.add(feat_v_amp, (sx, sy))  # ★ v2.4: (화이트닝은 fit 내부에서)
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
