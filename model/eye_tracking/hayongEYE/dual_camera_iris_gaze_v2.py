#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dual_camera_iris_gaze_v2.py

업그레이드 프로토타입: 2대의 카메라 + MediaPipe Face Mesh(iris) + OpenCV
- (1) 눈-내부 정규화 특징(vx, vy) 기반 2차 다항 캘리브레이션
- (2) ArUco(또는 체커보드)로 월드 평면(모니터/보드) 정합(호모그래피)
- (3) EMA 스무딩, 디버그 HUD, 키보드 토글

핵심 아이디어
- 얼굴/눈 카메라(eye)에서 얻은 홍채 중심을 "눈 컨투어 bbox"에 대해 정규화한 상대 벡터 (vx, vy)로 표현 →
  비선형 왜곡을 2차 다항식으로 회귀해 화면 정규좌표(sx, sy)로 보정 매핑
- 월드 카메라(world)에는 ArUco 마커 4개(코너)를 붙여, 화면 정규좌표→월드 픽셀로 변환하는 호모그래피 H를 추정

준비물
- Python 3.9 이상(3.11 테스트됨)
- mediapipe >= 0.10.x
- opencv-python
- (선택) opencv-contrib-python  (ArUco 사용 시 필요)
- numpy

실행 예시(장치 인덱스/해상도는 환경에 맞게 조정):
    python dual_camera_iris_gaze_v2.py --world_cam 0 --eye_cam 1 --world_w 1280 --world_h 720 --eye_w 640 --eye_h 480 --flip_eye

키 조작:
    q : 종료
    h : 도움말 HUD 토글
    d : 눈 프레임에 홍채 디버그 원 표시 토글
    r : EMA(지수이동평균) 스무딩 상태 리셋

    c : 9-포인트 캘리브레이션 시작/종료 토글
        - 캘리 모드에서 화면에 초록 점(정규좌표)이 표시됨 → 해당 점을 바라본 뒤 SPACE로 샘플 채집
        - 9점 모두 채집되면 자동으로 회귀가 수행되어 보정이 활성화됨
    SPACE : (캘리 모드일 때) 현재 프레임의 특징(vx,vy)과 목표점(sx,sy)을 한 샘플로 채집
    g : 보정(캘리브레이션 매핑) on/off
    S : 보정 계수 저장 (npz)
    L : 보정 계수 로드 (npz)

    a : ArUco 기반 월드 호모그래피 사용 on/off (opencv-contrib-python 필요)
        - 화면 4개 코너에 ArUco 마커를 TL=ID0, TR=ID1, BR=ID2, BL=ID3 순으로 배치했다고 가정

라이선스: MIT
작성: ChatGPT (사용자 요청에 따른 프로토타입 확장판)
"""

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List

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
# refine_landmarks=True로 FaceMesh를 실행할 때 홍채(iris) 랜드마크가 포함됩니다.
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
# 참고: 일부 구현은 468(우), 473(좌)을 iris 중심 proxy로 사용

# ----------------------------
# 보조 클래스/함수
# ----------------------------
@dataclass
class EMA2D:
    """2차원 점에 대한 지수이동평균(EMA) 필터.
    - clamp: (min, max) 범위로 클램프하고 싶을 때 지정. None이면 클램프 없음.
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
    ap = argparse.ArgumentParser(description="MediaPipe + OpenCV 2카메라 시선 오버레이(캘리브/호모그래피 확장판)")
    ap.add_argument("--world_cam", type=int, default=0, help="월드(배경) 카메라 OpenCV 인덱스")
    ap.add_argument("--eye_cam", type=int, default=1, help="눈/얼굴 카메라 OpenCV 인덱스")
    ap.add_argument("--world_w", type=int, default=1280, help="월드 카메라 캡처 가로 해상도(요청값)")
    ap.add_argument("--world_h", type=int, default=720, help="월드 카메라 캡처 세로 해상도(요청값)")
    ap.add_argument("--eye_w", type=int, default=640, help="눈 카메라 캡처 가로 해상도(요청값)")
    ap.add_argument("--eye_h", type=int, default=480, help="눈 카메라 캡처 세로 해상도(요청값)")
    ap.add_argument("--flip_eye", action="store_true", help="눈 카메라 영상을 좌우 반전(셀피 느낌)")
    ap.add_argument("--draw_scale", type=float, default=1.0, help="도형/텍스트 스케일 팩터")
    ap.add_argument("--min_det_conf", type=float, default=0.5, help="FaceMesh min_detection_confidence")
    ap.add_argument("--min_trk_conf", type=float, default=0.5, help="FaceMesh min_tracking_confidence")
    ap.add_argument("--ema_alpha", type=float, default=0.25, help="EMA 알파(0<alpha<=1). 특징/좌표 모두에 사용")
    ap.add_argument("--use_left_eye_only", action="store_true", help="좌안(Left)만 사용")
    ap.add_argument("--use_right_eye_only", action="store_true", help="우안(Right)만 사용")
    ap.add_argument("--calib_path", type=str, default="gaze_calib.npz", help="캘리브레이션 계수 저장/로드 경로")
    return ap.parse_args()


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    """
    OpenCV 카메라 열기 유틸.
    - 일부 Windows 환경에서 CAP_DSHOW 백엔드가 더 안정적이라 먼저 시도
    - 해상도 설정은 장치에 따라 무시될 수 있음
    """
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)  # 백엔드 힌트 없이 재시도
    if not cap.isOpened():
        raise RuntimeError(f"카메라 열기 실패 (index={index})")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def draw_hud(frame_bgr: np.ndarray, text_lines: List[str], scale: float = 1.0, org=(10,20)):
    """세계(월드) 프레임 위 왼쪽 상단에 간단한 도움말/상태 텍스트 표시."""
    x, y = org
    for t in text_lines:
        # 그림자(검정) + 본문(흰색) 두 번 그리기
        cv2.putText(frame_bgr, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += int(20 * scale)


def _indices_from_connections(conns):
    """MediaPipe connection set에서 인덱스 집합 추출."""
    idxs = set()
    for a, b in conns:
        idxs.add(a); idxs.add(b)
    return sorted(list(idxs))


def iris_center_from_landmarks(landmarks, frame_shape: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    주어진 얼굴 랜드마크들(정규화 좌표)을 사용해 좌/우 홍채 중심 근사값을 구함.
    - landmarks: MediaPipe가 반환한 얼굴 랜드마크 리스트(정규화 [0,1] 좌표)
    - frame_shape: (H, W) (참고용; 본 함수에서는 정규화 좌표만 사용)
    반환: (left_center01, right_center01) — 둘 중 없을 수도 있으므로 Optional
    """
    def gather_xy(idxs):
        pts = []
        for i in idxs:
            if i < len(landmarks):
                lm = landmarks[i]
                pts.append((lm.x, lm.y))
        if len(pts) != len(idxs):
            return None
        return np.array(pts, dtype=np.float32)

    left_pts01 = gather_xy(LEFT_IRIS)
    right_pts01 = gather_xy(RIGHT_IRIS)

    def center01(pts01):
        if pts01 is None:
            return None
        return pts01.mean(axis=0)  # (2,)

    return center01(left_pts01), center01(right_pts01)


def eye_feature_from_facemesh(face_landmarks, mp_face_mesh, which: str) -> Optional[np.ndarray]:
    """
    눈 특징 벡터: iris 중심 대비 '눈 컨투어 bbox 중심'의 상대좌표를
    눈 크기(폭/높이)로 정규화한 (vx, vy) 반환. 대략 -1~+1 범위.
    which: 'left' or 'right'
    """
    lm = face_landmarks  # list of NormalizedLandmark
    if which == 'left':
        eye_conns = mp_face_mesh.FACEMESH_LEFT_EYE
        iris_idxs = _indices_from_connections(mp_face_mesh.FACEMESH_LEFT_IRIS)
    else:
        eye_conns = mp_face_mesh.FACEMESH_RIGHT_EYE
        iris_idxs = _indices_from_connections(mp_face_mesh.FACEMESH_RIGHT_IRIS)

    eye_idxs = _indices_from_connections(eye_conns)

    # 좌표 수집
    try:
        eye_pts = np.array([(lm[i].x, lm[i].y) for i in eye_idxs if i < len(lm)], dtype=np.float32)
        iris_pts = np.array([(lm[i].x, lm[i].y) for i in iris_idxs if i < len(lm)], dtype=np.float32)
    except Exception:
        return None

    if eye_pts.size == 0 or iris_pts.size == 0:
        return None

    # 눈 컨투어 bbox와 중심
    ex_min, ey_min = eye_pts.min(axis=0)
    ex_max, ey_max = eye_pts.max(axis=0)
    ew = max(ex_max - ex_min, 1e-6)
    eh = max(ey_max - ey_min, 1e-6)
    eye_ctr = np.array([(ex_min + ex_max) * 0.5, (ey_min + ey_max) * 0.5], dtype=np.float32)

    # 홍채 중심
    iris_ctr = iris_pts.mean(axis=0)  # (x,y) normalized

    # 상대 좌표(눈 크기로 정규화): iris가 오른쪽/아래로 가면 +가 커짐
    vx = (iris_ctr[0] - eye_ctr[0]) / ew
    vy = (iris_ctr[1] - eye_ctr[1]) / eh
    return np.array([vx, vy], dtype=np.float32)


# === 2차 다항 회귀 기반 캘리브레이터 ===
@dataclass
class Poly2Calibrator:
    # x, y 각각 계수 6개: [x, y, xy, x^2, y^2, 1]
    cx: Optional[np.ndarray] = None  # (6,)
    cy: Optional[np.ndarray] = None  # (6,)
    samples_f: List[np.ndarray] = None  # 특징 (vx,vy)
    samples_s: List[np.ndarray] = None  # 스크린 정규좌표 (sx,sy)
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
        F = np.stack([self._phi(v) for v in self.samples_f], axis=0)  # (N,6)
        S = np.stack(self.samples_s, axis=0)  # (N,2)
        # 최소자승 해
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


# === 캘리브레이션 타깃 포인트 (정규좌표) ===
CAL_POINTS = [
    (0.1,0.1),(0.5,0.1),(0.9,0.1),
    (0.1,0.5),(0.5,0.5),(0.9,0.5),
    (0.1,0.9),(0.5,0.9),(0.9,0.9)
]


# === ArUco 기반 호모그래피 추정 ===
# 마커 배치 가정: TL=ID0, TR=ID1, BR=ID2, BL=ID3
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
    """버전 호환을 고려한 ArUco 검출."""
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
    """
    4개 코너에 서로 다른 ID의 마커를 고정(예: TL=0, TR=1, BR=2, BL=3)했다고 가정.
    각 마커의 '중심' 픽셀을 네 점으로 삼아, (0,0)-(1,1) 정규 스크린 좌표 → 월드 픽셀로의 H를 추정.
    """
    if not HAVE_ARUCO:
        return None

    gray = cv2.cvtColor(world_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _detect_aruco(gray)
    if ids is None or len(ids) < 4 or corners is None:
        return None

    # 필요한 ID 모두 있는지 확인
    ids_flat = ids.flatten().tolist()
    if not all(k in ids_flat for k in _ARUCO_IDS_WANT):
        return None

    # id -> 마커 '중심' 픽셀 좌표 (더 정확히는 각 꼭짓점 평균)
    id_to_center = {}
    for c, i in zip(corners, ids.flatten()):
        center = c[0].mean(axis=0)  # (x,y)
        id_to_center[int(i)] = center.astype(np.float32)

    try:
        pts_dst = np.float32([
            id_to_center[_ARUCO_IDS_WANT[0]],  # TL
            id_to_center[_ARUCO_IDS_WANT[1]],  # TR
            id_to_center[_ARUCO_IDS_WANT[2]],  # BR
            id_to_center[_ARUCO_IDS_WANT[3]],  # BL
        ])  # world pixels
    except KeyError:
        return None

    # 정규 좌표의 네 모서리
    pts_src = np.float32([[0,0],[1,0],[1,1],[0,1]])  # screen norm

    H, _ = cv2.findHomography(pts_src, pts_dst, method=cv2.RANSAC)
    return H


def main():
    args = parse_args()

    # ---------- 카메라 열기 ----------
    world_cap = open_camera(args.world_cam, args.world_w, args.world_h)  # 월드(배경)
    eye_cap = open_camera(args.eye_cam, args.eye_w, args.eye_h)          # 눈(얼굴)

    # ---------- MediaPipe Face Mesh 초기화 ----------
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,          # 동영상 스트림 모드
        max_num_faces=1,                  # 최대 1명
        refine_landmarks=True,            # 홍채 랜드마크 포함(필수)
        min_detection_confidence=args.min_det_conf,
        min_tracking_confidence=args.min_trk_conf,
    )

    # ---------- 스무딩 ----------
    ema_feat = EMA2D(alpha=args.ema_alpha, clamp=None)        # 특징(vx,vy): 클램프 없음
    ema_norm = EMA2D(alpha=args.ema_alpha, clamp=(0.0, 1.0))  # 정규좌표(sx,sy)
    # 주의: 둘은 별개로 운용

    # ---------- 캘리브레이션 ----------
    cal = Poly2Calibrator()
    cal_idx = 0  # 현재 캘리 포인트 인덱스

    # ---------- 월드 호모그래피 ----------
    use_homog = False
    H_world = None

    # ---------- UI 토글 상태 ----------
    show_help = True     # 도움말 표시
    show_debug = False   # 눈 프레임에 디버그 원 표시

    while True:
        # 1) 월드/눈 프레임 읽기
        ok_world, world_bgr = world_cap.read()
        ok_eye, eye_bgr = eye_cap.read()
        if not ok_world or not ok_eye:
            print("경고: 카메라 프레임 읽기 실패(월드/눈 중 하나).")
            break

        # 2) 눈 프레임 전처리(좌우 반전 옵션)
        if args.flip_eye:
            eye_bgr = cv2.flip(eye_bgr, 1)

        # 3) MediaPipe 처리용 RGB로 변환(BGR→RGB)
        eye_rgb = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2RGB)
        eye_rgb.flags.writeable = False
        res = face_mesh.process(eye_rgb)
        eye_rgb.flags.writeable = True

        # 4) 눈 특징 및 홍채 중심 계산
        gaze_pt01 = None           # 최종 화면 정규좌표 (매핑 결과)
        left01 = right01 = None    # 디버그용 홍채 중심
        feat_v = None              # (vx, vy)

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            # (A) 눈 특징: 눈 컨투어 bbox 기준 정규화된 (vx, vy)
            left_v  = eye_feature_from_facemesh(lm, mp_face_mesh, 'left')
            right_v = eye_feature_from_facemesh(lm, mp_face_mesh, 'right')

            if args.use_left_eye_only and left_v is not None:
                feat_v = left_v
            elif args.use_right_eye_only and right_v is not None:
                feat_v = right_v
            elif left_v is not None and right_v is not None:
                feat_v = 0.5 * (left_v + right_v)

            if feat_v is not None and np.isfinite(feat_v).all():
                feat_v = ema_feat.update(feat_v)  # 특징 EMA

            # (B) 디버그용: 기존 홍채 중심(정규좌표)도 계산
            left01, right01 = iris_center_from_landmarks(lm, eye_bgr.shape[:2])

            # (C) 보정(캘리브레이션) 사용 여부에 따라 화면 정규좌표 산출
            if cal.use and feat_v is not None:
                mapped = cal.map(feat_v)  # 보정된 화면 정규좌표
                if mapped is not None and np.isfinite(mapped).all():
                    gaze_pt01 = ema_norm.update(mapped)  # 화면 좌표 EMA
            else:
                # 후방호환: 보정 미사용 시 좌/우 홍채 평균을 그대로 사용
                base = None
                if args.use_left_eye_only and left01 is not None:
                    base = left01
                elif args.use_right_eye_only and right01 is not None:
                    base = right01
                elif left01 is not None and right01 is not None:
                    base = 0.5 * (left01 + right01)

                if base is not None and np.isfinite(base).all():
                    gaze_pt01 = ema_norm.update(base.astype(np.float32))

            # (옵션) 디버그: 눈 프레임에 좌/우 중심 근사 그리기
            if show_debug and (left01 is not None or right01 is not None):
                h_eye, w_eye = eye_bgr.shape[:2]
                if left01 is not None:
                    lx, ly = int(left01[0] * w_eye), int(left01[1] * h_eye)
                    cv2.circle(eye_bgr, (lx, ly), int(6 * args.draw_scale), (0, 255, 0), 2, cv2.LINE_AA)
                if right01 is not None:
                    rx, ry = int(right01[0] * w_eye), int(right01[1] * h_eye)
                    cv2.circle(eye_bgr, (rx, ry), int(6 * args.draw_scale), (0, 255, 255), 2, cv2.LINE_AA)

        # (선택) 4.5) 월드 호모그래피 갱신
        if use_homog:
            H = estimate_homography_from_aruco(world_bgr)
            if H is not None:
                H_world = H

        # 5) 월드 프레임에 시선 마커 오버레이
        if gaze_pt01 is not None:
            if use_homog and H_world is not None:
                # H는 (정규 스크린 좌표) → (월드 픽셀) 매핑
                src = np.array([[[gaze_pt01[0], gaze_pt01[1]]]], dtype=np.float32)  # (1,1,2)
                pt = cv2.perspectiveTransform(src, H_world)[0, 0]
                gx, gy = int(pt[0]), int(pt[1])
            else:
                h_w, w_w = world_bgr.shape[:2]
                gx, gy = int(gaze_pt01[0] * w_w), int(gaze_pt01[1] * h_w)

            r = int(10 * args.draw_scale)
            cv2.circle(world_bgr, (gx, gy), r, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(world_bgr, (gx, gy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        # 6) HUD(도움말/상태) 그리기
        if show_help:
            help_lines = [
                "Dual-Camera Iris Gaze v2 (Calib + Homography)",
                "[q] 종료  [h] 도움말  [d] 디버그  [r] EMA리셋",
                "[c] 캘리브 모드  [SPACE] 샘플추가  [g] 보정 on/off  [S/L] 저장/로드",
                "[a] ArUco 호모그래피 on/off",
                f"옵션: left-only={args.use_left_eye_only}, right-only={args.use_right_eye_only}, flip_eye={args.flip_eye}",
                f"Calib: active={cal.active}, use={cal.use}, samples={len(cal.samples_f)}",
                f"Homog: enabled={use_homog}, H ready={H_world is not None}, ArUco={'OK' if HAVE_ARUCO else 'N/A'}",
            ]
            if cal.active and 0 <= cal_idx < len(CAL_POINTS):
                sx, sy = CAL_POINTS[cal_idx]
                help_lines.append(f"Target #{cal_idx+1}/{len(CAL_POINTS)} at ({sx:.2f},{sy:.2f}) → SPACE로 채집")
            draw_hud(world_bgr, help_lines, scale=args.draw_scale)

        # (캘리 타깃 포인트 오버레이)
        if cal.active and 0 <= cal_idx < len(CAL_POINTS):
            h_w, w_w = world_bgr.shape[:2]
            sx, sy = CAL_POINTS[cal_idx]
            cx, cy = int(sx*w_w), int(sy*h_w)
            cv2.circle(world_bgr, (cx,cy), int(14*args.draw_scale), (0,255,0), 2, cv2.LINE_AA)

        # 7) 결과 표시(월드 / 눈)
        cv2.imshow("World (Gaze Overlay)", world_bgr)
        # 눈 창: 디버그 off일 때는 축소해서 표시
        if show_debug:
            cv2.imshow("Eye (Debug)", eye_bgr)
        else:
            w = min(480, eye_bgr.shape[1])
            h = int(w * eye_bgr.shape[0] / max(1, eye_bgr.shape[1]))
            cv2.imshow("Eye (Debug)", cv2.resize(eye_bgr, (w, h)))

        # 8) 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'):
            show_help = not show_help
        elif key == ord('d'):
            show_debug = not show_debug
        elif key == ord('r'):
            ema_feat.reset(); ema_norm.reset()
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

    # 자원 해제
    world_cap.release()
    eye_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
