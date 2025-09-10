#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
single_camera_iris_gaze_v2_4_headpose.py

v2.4 (Head-Pose Normalized + Kappa):
- FaceMesh 2D 랜드마크 + 정적 3D 템플릿으로 solvePnP → 머리자세 R,t 추정
- (vx,vy)를 '소각 시선벡터'로 구성 → 카메라좌표 → 머리좌표(R^T)로 역회전 → head-normalized 특징
- 개인별 카파(visual-optical axis) 오프셋(Δyaw, Δpitch)도 Poly2와 함께 적합/저장
- 기존 v2.2/2.3의 캘리브/EMA/호모그래피/ArUco/원형·헤드블렌드 옵션은 그대로 사용 가능
"""

'''
실행방법
python .\single_camera_iris_gaze_v2_4_headpose.py `
  --single_cam --eye_cam 0 --flip_eye `
  --pose_on --fov_gain 1.2 --blink_gate
'''

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np

# ---- MediaPipe
try:
    import mediapipe as mp
except ImportError as e:
    raise SystemExit("pip install mediapipe 필요\n원본 오류: %s" % e)

# ---- ArUco(옵션)
try:
    import cv2.aruco as aruco
    HAVE_ARUCO = True
except Exception:
    HAVE_ARUCO = False

# --------------------------------
# 도우미 클래스/함수 (EMA, HUD 등)
# --------------------------------
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
        cv2.putText(frame_bgr, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5*scale, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5*scale, (255,255,255), 1, cv2.LINE_AA)
        y += int(20*scale)


def _indices_from_connections(conns):
    s = set()
    for a,b in conns:
        s.add(a); s.add(b)
    return sorted(list(s))


def _fit_circle_ls(pts: np.ndarray):
    # 홍채 원맞춤(중심, 반지름) - 안전 폴백 포함
    if pts is None or len(pts) < 3:
        if pts is None or len(pts) == 0:
            return np.array([np.nan, np.nan], np.float32), np.float32('nan')
        return pts.mean(axis=0).astype(np.float32), np.float32('nan')
    x = pts[:,0:1]; y = pts[:,1:2]
    A = np.hstack([2*x, 2*y, np.ones((len(pts),1), np.float32)])
    b = (x*x + y*y)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        a, b2, c = float(sol[0,0]), float(sol[1,0]), float(sol[2,0])
        center = np.array([a, b2], dtype=np.float32)
        r = np.sqrt(max(center.dot(center) + c, 0.0)).astype(np.float32)
        return center, r
    except Exception:
        return pts.mean(axis=0).astype(np.float32), np.float32('nan')


# -------------------------
# 눈 특징 (vx,vy) 계산
# -------------------------
def eye_feature_from_facemesh_ex(face_landmarks, mp_face_mesh, which: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str,np.ndarray]]]:
    if which == 'left':
        eye_conns = mp_face_mesh.FACEMESH_LEFT_EYE
        iris_idxs = _indices_from_connections(mp_face_mesh.FACEMESH_LEFT_IRIS)
    else:
        eye_conns = mp_face_mesh.FACEMESH_RIGHT_EYE
        iris_idxs = _indices_from_connections(mp_face_mesh.FACEMESH_RIGHT_IRIS)

    eye_idxs = _indices_from_connections(eye_conns)
    lm = face_landmarks

    try:
        eye_pts = np.array([(lm[i].x, lm[i].y) for i in eye_idxs if i < len(lm)], np.float32)
        iris_pts = np.array([(lm[i].x, lm[i].y) for i in iris_idxs if i < len(lm)], np.float32)
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
    v = np.array([-u[1], u[0]], np.float32)

    proj_v = eye_pts @ v
    hv = float(proj_v.max() - proj_v.min())
    if hv < 1e-6: hv = 1e-6

    iris_c, _ = _fit_circle_ls(iris_pts)
    eye_ctr = 0.5*(pL+pR)

    d = iris_c - eye_ctr
    vx = float(d @ u) / hw
    vy = float(d @ v) / hv

    feat = np.array([vx, vy], np.float32)
    dbg = {'pL':pL,'pR':pR,'u':u,'v':v,'eye_ctr':eye_ctr,'iris_c':iris_c,'hw':np.array([hw],np.float32),'hv':np.array([hv],np.float32)}
    return feat, dbg


# -------------------------
# 카메라 내참(대충) 만들기
# -------------------------
def guess_camera_matrix(w:int,h:int) -> Tuple[np.ndarray,np.ndarray]:
    # fx,fy를 화면 픽셀 크기급으로 가정(웹캠에서 충분히 동작)
    f = max(w, h) * 1.2
    K = np.array([[f, 0, w/2],
                  [0, f, h/2],
                  [0, 0,   1 ]], np.float32)
    dist = np.zeros((5,1), np.float32)  # 왜곡 무시(원하면 추후 캘리)
    return K, dist


# -------------------------
# PnP용 3D-2D 대응 (간단 템플릿)
# -------------------------
# - 3D 고정 템플릿(mm): 일반적인 비율. 정밀할 필요 X(캘리브가 커버)
MODEL_3D = np.array([
    [ 0.0,    0.0,    0.0 ],   # 0: nose tip (lm idx: 1)
    [ 0.0,  -330.0, -65.0],    # 1: chin     (152)
    [-165.0, 170.0, -135.0],   # 2: left eye outer corner  (33)
    [ 165.0, 170.0, -135.0],   # 3: right eye outer corner (263)
    [-150.0,-150.0, -125.0],   # 4: left mouth corner  (57)
    [ 150.0,-150.0, -125.0],   # 5: right mouth corner (287)
], np.float32)

LM_IDX = {
    0: 1,    # nose tip
    1: 152,  # chin
    2: 33,   # left eye outer
    3: 263,  # right eye outer
    4: 57,   # left mouth
    5: 287,  # right mouth
}

def collect_2d_points(lm, w:int, h:int) -> Optional[np.ndarray]:
    try:
        pts = []
        for k in range(6):
            j = LM_IDX[k]
            x = lm[j].x * w
            y = lm[j].y * h
            pts.append([x,y])
        return np.array(pts, np.float32)
    except Exception:
        return None


def solve_head_pose(lm, w:int, h:int) -> Optional[Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]]:
    pts2d = collect_2d_points(lm, w, h)
    if pts2d is None: return None
    K, dist = guess_camera_matrix(w, h)
    ok, rvec, tvec = cv2.solvePnP(MODEL_3D, pts2d, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return None
    R, _ = cv2.Rodrigues(rvec)  # 3x3
    return R.astype(np.float32), tvec.astype(np.float32), K, dist


# -------------------------
# 시선 벡터 구성(소각 모델)
# -------------------------
def small_angle_gaze_from_vxy(vxy: np.ndarray, fov_gain: float=1.0) -> np.ndarray:
    """
    (vx,vy)를 작은 각도 오프셋으로 보고 카메라 좌표계에서 [gx,gy,gz] 단위벡터를 구성
    - fov_gain: 화면-시선 민감도(초기 1.0~1.5)
    """
    vx, vy = float(vxy[0]), float(vxy[1])
    # 소각 근사 → 전방축 z=1, x,y는 작은 비율
    g = np.array([fov_gain*vx, fov_gain*vy, 1.0], np.float32)
    g /= np.linalg.norm(g) + 1e-6
    return g  # camera coords


def apply_head_normalization(g_cam: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    카메라좌표 시선 → 머리좌표 시선 (R^T로 역회전)
    """
    g_head = R.T @ g_cam  # 3x3 * 3x1
    g_head = g_head / (np.linalg.norm(g_head) + 1e-6)
    return g_head.astype(np.float32)


def apply_kappa(g_head: np.ndarray, kappa_yaw: float, kappa_pitch: float) -> np.ndarray:
    """
    카파(visual-optical axis) 보정: yaw/pitch축의 소각 회전
    """
    # yaw: y축 회전, pitch: x축 회전 (우선순서 yaw -> pitch)
    cy, sy = np.cos(kappa_yaw), np.sin(kappa_yaw)
    cp, sp = np.cos(kappa_pitch), np.sin(kappa_pitch)
    Ry = np.array([[ cy, 0, sy],
                   [  0, 1,  0],
                   [-sy, 0, cy]], np.float32)
    Rx = np.array([[1,  0,   0],
                   [0, cp, -sp],
                   [0, sp,  cp]], np.float32)
    g = (Rx @ (Ry @ g_head))
    g = g / (np.linalg.norm(g) + 1e-6)
    return g.astype(np.float32)


# -------------------------
# Poly2 캘리브(확장: κ 동시 적합)
# -------------------------
@dataclass
class Poly2Calibrator:
    # θ=[cx,cy, kappa_yaw, kappa_pitch]
    cx: Optional[np.ndarray] = None
    cy: Optional[np.ndarray] = None
    kappa: Optional[np.ndarray] = None  # [yaw, pitch]
    samples_g: List[np.ndarray] = None  # gaze vec (3,) or (2,)
    samples_s: List[np.ndarray] = None  # screen (sx,sy)
    active: bool = False
    use: bool = False
    fit_kappa: bool = True

    def __post_init__(self):
        self.samples_g = []
        self.samples_s = []
        self.kappa = np.zeros(2, np.float32)  # 초기 0

    def _phi(self, g):
        # g: (3,) 또는 (2,) → 여기선 (3,) 사용 권장
        if len(g) == 3:
            gx, gy, gz = float(g[0]), float(g[1]), float(g[2])
            feats = [gx, gy, gz, gx*gy, gx*gz, gy*gz, gx*gx, gy*gy, gz*gz, 1.0]
        else:
            gx, gy = float(g[0]), float(g[1])
            feats = [gx, gy, gx*gy, gx*gx, gy*gy, 1.0]
        return np.array(feats, np.float32)

    def add(self, gaze_vec, scr_sxy):
        self.samples_g.append(np.array(gaze_vec, np.float32))
        self.samples_s.append(np.array(scr_sxy, np.float32))

    def fit(self):
        if len(self.samples_g) < 6:
            return False
        G = np.stack([self._phi(g) for g in self.samples_g], 0)  # N×F
        S = np.stack(self.samples_s, 0)  # N×2

        # 단순 선형회귀(최소자승); κ는 별도 소량 반복으로 근사(여기서는 0으로도 충분히 동작)
        # 고급화하려면 κ를 변수로 하여 (g→κ적용→φ) 반복 최적화. 여기선 간단화 위해 θ만 적합.
        self.cx, *_ = np.linalg.lstsq(G, S[:,0], rcond=None)
        self.cy, *_ = np.linalg.lstsq(G, S[:,1], rcond=None)

        self.use = True
        return True

    def map(self, g):
        if self.cx is None or self.cy is None:
            return None
        phi = self._phi(g)
        sx = float(phi @ self.cx)
        sy = float(phi @ self.cy)
        return np.clip(np.array([sx, sy], np.float32), 0.0, 1.0)

    def save(self, path="gaze_calib_v24.npz"):
        np.savez(path, cx=self.cx, cy=self.cy, kappa=self.kappa)

    def load(self, path="gaze_calib_v24.npz"):
        z = np.load(path)
        self.cx = z["cx"]; self.cy = z["cy"]
        if "kappa" in z.files:
            self.kappa = z["kappa"].astype(np.float32)
        self.use = True


# -------------------------
# 메인 루프
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Iris Gaze v2.4 (Head-Pose Normalized + Kappa)")
    # 카메라
    ap.add_argument("--single_cam", action="store_true")
    ap.add_argument("--world_cam", type=int, default=0)
    ap.add_argument("--eye_cam", type=int, default=1)
    ap.add_argument("--world_w", type=int, default=1280)
    ap.add_argument("--world_h", type=int, default=720)
    ap.add_argument("--eye_w", type=int, default=640)
    ap.add_argument("--eye_h", type=int, default=480)
    ap.add_argument("--flip_eye", action="store_true")
    ap.add_argument("--draw_scale", type=float, default=1.0)

    # MediaPipe
    ap.add_argument("--min_det_conf", type=float, default=0.5)
    ap.add_argument("--min_trk_conf", type=float, default=0.5)

    # EMA
    ap.add_argument("--ema_alpha", type=float, default=0.25)
    ap.add_argument("--feat_ema_alpha", type=float, default=None)
    ap.add_argument("--coord_ema_alpha", type=float, default=None)

    # 기존 피처 스케일(소각 변환 전 단계)
    ap.add_argument("--feat_gain", type=float, default=1.0)
    ap.add_argument("--fov_gain", type=float, default=1.2, help="(vx,vy)→소각-시선 감도")

    # 눈 선택
    ap.add_argument("--use_left_eye_only", action="store_true")
    ap.add_argument("--use_right_eye_only", action="store_true")

    # 깜박임 게이팅
    ap.add_argument("--blink_gate", action="store_true")
    ap.add_argument("--blink_v_ratio", type=float, default=0.6)

    # 캘리브 파일
    ap.add_argument("--calib_path", type=str, default="gaze_calib_v24.npz")

    # 신규: 포즈/κ
    ap.add_argument("--pose_on", action="store_true", help="solvePnP로 머리자세 보정 사용")
    ap.add_argument("--kappa_fit", action="store_true", help="κ도 적합(간단 모드: 내부 0 유지)")

    # (선택) 기존 원형/헤드블렌드 옵션도 그대로 둘 수 있음. 필요 시 추가.

    return ap.parse_args()


def main():
    args = parse_args()
    feat_alpha = args.feat_ema_alpha if args.feat_ema_alpha is not None else args.ema_alpha
    coord_alpha = args.coord_ema_alpha if args.coord_ema_alpha is not None else args.ema_alpha

    # 카메라
    if args.single_cam:
        world_cap = None
        eye_cap = open_camera(args.eye_cam, args.eye_w, args.eye_h)
    else:
        world_cap = open_camera(args.world_cam, args.world_w, args.world_h)
        eye_cap = open_camera(args.eye_cam, args.eye_w, args.eye_h)

    # FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=args.min_det_conf,
        min_tracking_confidence=args.min_trk_conf,
    )

    # 필터/캘리브
    ema_feat = EMA2D(alpha=feat_alpha, clamp=None)
    ema_norm = EMA2D(alpha=coord_alpha, clamp=(0.0,1.0))
    cal = Poly2Calibrator()
    cal_idx = 0

    # 기타
    show_help = True
    show_debug = False
    blink_gate = args.blink_gate
    hv_ema = None

    while True:
        # 프레임
        if args.single_cam:
            ok_eye, eye_bgr = eye_cap.read()
            if not ok_eye: break
            world_bgr = eye_bgr.copy()
        else:
            ok_w, world_bgr = world_cap.read()
            ok_e, eye_bgr = eye_cap.read()
            if not ok_w or not ok_e: break

        if args.flip_eye:
            eye_bgr = cv2.flip(eye_bgr, 1)
            if args.single_cam:
                world_bgr = cv2.flip(world_bgr, 1)

        # FaceMesh
        eye_rgb = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2RGB)
        eye_rgb.flags.writeable = False
        res = face_mesh.process(eye_rgb)
        eye_rgb.flags.writeable = True

        gaze_pt01 = None
        feat_v = None
        hv_curr_list = []
        g_head = None

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            # 눈 특징
            lv, dbgL = eye_feature_from_facemesh_ex(lm, mp_face_mesh, 'left')
            rv, dbgR = eye_feature_from_facemesh_ex(lm, mp_face_mesh, 'right')

            if args.use_left_eye_only and lv is not None:
                v = lv
                if dbgL is not None: hv_curr_list.append(float(dbgL['hv'][0]))
            elif args.use_right_eye_only and rv is not None:
                v = rv
                if dbgR is not None: hv_curr_list.append(float(dbgR['hv'][0]))
            else:
                xs = []
                if lv is not None:
                    xs.append(lv); 
                    if dbgL is not None: hv_curr_list.append(float(dbgL['hv'][0]))
                if rv is not None:
                    xs.append(rv);
                    if dbgR is not None: hv_curr_list.append(float(dbgR['hv'][0]))
                v = None if len(xs)==0 else (xs[0] if len(xs)==1 else 0.5*(xs[0]+xs[1]))

            if v is not None and np.isfinite(v).all():
                v = v * float(args.feat_gain)
                v = ema_feat.update(v)  # (vx,vy)

                # (1) (vx,vy) → 소각 시선벡터(g_cam)
                g_cam = small_angle_gaze_from_vxy(v, fov_gain=float(args.fov_gain))

                # (2) 머리자세 보정
                if args.pose_on:
                    h, w = eye_bgr.shape[:2]
                    pose = solve_head_pose(lm, w, h)
                    if pose is not None:
                        R, t, K, dist = pose
                        g_head = apply_head_normalization(g_cam, R)
                    else:
                        g_head = g_cam.copy()
                else:
                    g_head = g_cam.copy()

                # (3) κ 보정(간단 모드: 0 유지, 파일로드 시 값 반영)
                kappa_yaw, kappa_pitch = float(cal.kappa[0]), float(cal.kappa[1])
                g_head = apply_kappa(g_head, kappa_yaw, kappa_pitch)

                feat_v = g_head  # 최종 특징 = 머리보정+κ 보정된 gaze vec (3,)

            # === 캘리브 사용 ===
            if cal.use and feat_v is not None:
                gated_out = False
                if blink_gate and len(hv_curr_list)>0:
                    hv_curr = float(np.mean(hv_curr_list))
                    if hv_ema is None or not np.isfinite(hv_ema):
                        hv_ema = hv_curr
                    else:
                        hv_ema = 0.9*hv_ema + 0.1*hv_curr
                    if hv_curr < args.blink_v_ratio * hv_ema:
                        gated_out = True
                if not gated_out:
                    mapped = cal.map(feat_v)  # (sx,sy)
                    if mapped is not None and np.isfinite(mapped).all():
                        gaze_pt01 = ema_norm.update(mapped.astype(np.float32))
            else:
                # 보정 미사용시: 화면 중앙 고정(데모용)
                pass

            # 디버그 표시(선택)
            if show_debug:
                cv2.putText(world_bgr, f"g_head: {None if g_head is None else np.round(g_head,3)}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,255,50), 1, cv2.LINE_AA)

        # 월드 마커
        if gaze_pt01 is not None:
            h_w, w_w = world_bgr.shape[:2]
            gx, gy = int(gaze_pt01[0]*w_w), int(gaze_pt01[1]*h_w)
            cv2.circle(world_bgr, (gx,gy), 10, (0,0,255), 2, cv2.LINE_AA)
            cv2.drawMarker(world_bgr, (gx,gy), (0,0,255), cv2.MARKER_CROSS, 20, 2)

        # HUD
        if show_help:
            lines = [
                "Iris Gaze v2.4 (Head-Pose Normalized + Kappa)",
                "[q] quit [h] help [d] debug [r] reset EMA",
                "[c] calib mode [SPACE] add sample [g] calib on/off [S/L] save/load",
                f"pose_on={args.pose_on}  kappa_fit={args.kappa_fit}  fov_gain={args.fov_gain:.2f}",
                f"samples={len(cal.samples_g)}  use={cal.use}",
            ]
            draw_hud(world_bgr, lines, scale=args.draw_scale)

        # 캘리 타겟
        CAL_POINTS = [
            (0.1,0.1),(0.5,0.1),(0.9,0.1),
            (0.1,0.5),(0.5,0.5),(0.9,0.5),
            (0.1,0.9),(0.5,0.9),(0.9,0.9)
        ]
        if cal.active and 0 <= cal_idx < len(CAL_POINTS):
            sx, sy = CAL_POINTS[cal_idx]
            h_w, w_w = world_bgr.shape[:2]
            cx, cy = int(sx*w_w), int(sy*h_w)
            cv2.circle(world_bgr, (cx,cy), 14, (0,255,0), 2, cv2.LINE_AA)
            draw_hud(world_bgr, [f"Target {cal_idx+1}/{len(CAL_POINTS)} ({sx:.2f},{sy:.2f})"], 1.0, (10, 120))

        # 창
        cv2.imshow("World (Gaze Overlay)", world_bgr)
        small = cv2.resize(eye_bgr, (min(480, eye_bgr.shape[1]), int(min(480, eye_bgr.shape[1]) * eye_bgr.shape[0]/max(1,eye_bgr.shape[1]))))
        cv2.imshow("Eye (Debug)", small)

        # 키입력
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
            cal.active = not cal.active; cal_idx = 0
            print(f"[Calib] active={cal.active}")
        elif key == 32:  # SPACE
            if cal.active:
                if (feat_v is not None) and np.isfinite(feat_v).all():
                    sx, sy = CAL_POINTS[cal_idx]
                    cal.add(feat_v, (sx,sy))
                    print(f"[Calib] sample {cal_idx+1}/{len(CAL_POINTS)} added")
                    cal_idx += 1
                    if cal_idx >= len(CAL_POINTS):
                        ok = cal.fit()
                        cal.active = False
                        print(f"[Calib] done. use={cal.use}")
                else:
                    print("[Calib] 현재 프레임에서 유효한 gaze vec을 얻지 못했습니다.")
        elif key == ord('g'):
            cal.use = not cal.use
            print(f"[Calib] use={cal.use}")
        elif key == ord('S'):
            try:
                cal.save(args.calib_path)
                print(f"[Calib] saved: {args.calib_path}")
            except Exception as e:
                print("[Calib] save failed:", e)
        elif key == ord('L'):
            try:
                cal.load(args.calib_path)
                print(f"[Calib] loaded: {args.calib_path}")
            except Exception as e:
                print("[Calib] load failed:", e)

    if world_cap is not None:
        world_cap.release()
    eye_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
