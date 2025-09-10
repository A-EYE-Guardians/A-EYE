# l2cs_level2_calibrated_gazepoint.py
# -*- coding: utf-8 -*-
"""
L2CS-Net + 2D 캘리브레이션(2차 다항 회귀)
- (yaw, pitch) -> (sx, sy) in [0,1] 로 매핑 학습
- 3x3 타깃을 순회하며 SPACE로 샘플 수집(프레임 평균)
- 추론 시 화면 정규좌표를 영상 창 크기(W,H)에 매핑해 픽셀로 시점 점 찍기
조작:
  C : 캘리브레이션 시작/재시작
  SPACE : 현재 타깃에서 샘플 수집(기본 15프레임 평균)
  U : 캘리브레이션 사용 토글
  S : 보정값 저장(calibrator.json)
  L : 보정값 불러오기(calibrator.json)
  Q : 종료
"""

import cv2
import time
import json
import math
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
from l2cs import Pipeline, render

# -----------------------------
# 설정
# -----------------------------
WEIGHTS = r"C:\Gukbi\hayongEYE2\models\L2CSNet_gaze360.pkl"
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480
SAMPLE_FRAMES = 15      # SPACE 누르면 이만큼 프레임 평균으로 (yaw,pitch) 수집
TARGET_R = 10           # 타깃 원 반지름(px)
ARROW_SCALE = 0.5       # (디버깅용) render와 비슷한 화살표 길이 비율
CALIB_FILE = "calibrator.json"


# -----------------------------
# 얼굴 선택: 가장 큰 bbox
# -----------------------------
def select_largest_face(bboxes) -> Optional[int]:
    if bboxes is None or len(bboxes) == 0:
        return None
    areas = []
    for i, b in enumerate(bboxes):
        try:
            x1, y1, x2, y2 = map(float, b)
        except Exception:
            continue
        areas.append((i, max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))))
    if not areas:
        return None
    areas.sort(key=lambda x: x[1], reverse=True)
    return areas[0][0]


# -----------------------------
# 안전 bbox
# -----------------------------
def safe_bbox_xyxy(bbox, W, H):
    x1, y1, x2, y2 = map(int, bbox)
    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    x2 = min(W - 1, x2)
    y2 = min(H - 1, y2)
    return x1, y1, x2, y2


# -----------------------------
# 레벨1 투영 식(디버깅용)
# -----------------------------
def arrow_endpoint_like_render(bbox_xyxy, pitch, yaw, scale=ARROW_SCALE):
    x1, y1, x2, y2 = bbox_xyxy
    w = x2 - x1
    h = y2 - y1
    cx = int(round(x1 + w * 0.5))
    cy = int(round(y1 + h * 0.5))
    L = int(round(max(w, h) * float(scale)))
    ex = int(round(cx - L * math.sin(float(yaw))))
    ey = int(round(cy - L * math.sin(float(pitch))))
    return cx, cy, ex, ey


# -----------------------------
# 2차 다항 보정기
# phi = [x, y, x*y, x^2, y^2, 1]
# -----------------------------
def phi_2d(x: float, y: float) -> np.ndarray:
    return np.array([x, y, x * y, x * x, y * y, 1.0], dtype=np.float64)

@dataclass
class Poly2Calibrator:
    cx: Optional[np.ndarray] = None   # (6,)
    cy: Optional[np.ndarray] = None   # (6,)
    use: bool = False                 # 추론 시 사용 여부

    def fit(self, xs: List[float], ys: List[float],
            sxs: List[float], sys: List[float]) -> None:
        """
        xs, ys : 수집한 yaw, pitch (단위: L2CS 결과 그대로. 도/라디안 변환 없이 일관 유지)
        sxs, sys : 해당 시점의 정규 좌표 [0,1] (캘리브 대상)
        """
        assert len(xs) == len(ys) == len(sxs) == len(sys) and len(xs) >= 6
        X = np.stack([phi_2d(x, y) for x, y in zip(xs, ys)], axis=0)  # [N,6]
        sx = np.asarray(sxs, dtype=np.float64)  # [N]
        sy = np.asarray(sys, dtype=np.float64)  # [N]
        self.cx, *_ = np.linalg.lstsq(X, sx, rcond=None)
        self.cy, *_ = np.linalg.lstsq(X, sy, rcond=None)

    def predict(self, x: float, y: float) -> Tuple[float, float]:
        assert self.cx is not None and self.cy is not None, "Calibrator not fitted"
        u = phi_2d(x, y)  # (6,)
        sx = float(u @ self.cx)
        sy = float(u @ self.cy)
        # 범위 보장
        sx = max(0.0, min(1.0, sx))
        sy = max(0.0, min(1.0, sy))
        return sx, sy

    def save(self, path: str) -> None:
        data = {
            "cx": self.cx.tolist() if self.cx is not None else None,
            "cy": self.cy.tolist() if self.cy is not None else None,
            "use": self.use,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.cx = np.array(data.get("cx"), dtype=np.float64) if data.get("cx") else None
        self.cy = np.array(data.get("cy"), dtype=np.float64) if data.get("cy") else None
        self.use = bool(data.get("use", False))
        return self.cx is not None and self.cy is not None


# -----------------------------
# 3x3 타깃 그리드 (정규 좌표)
# -----------------------------
def gen_3x3_targets() -> List[Tuple[float, float]]:
    xs = [0.15, 0.50, 0.85]
    ys = [0.15, 0.50, 0.85]
    targets = []
    for ty in ys:
        for tx in xs:
            targets.append((tx, ty))  # (sx, sy)
    return targets  # 총 9개


def draw_target_overlay(img, sx: float, sy: float, color=(0, 255, 255), r=TARGET_R):
    H, W = img.shape[:2]
    x = int(round(sx * (W - 1)))
    y = int(round(sy * (H - 1)))
    cv2.circle(img, (x, y), r, color, 2, cv2.LINE_AA)
    cv2.line(img, (x - r * 2, y), (x + r * 2, y), color, 1, cv2.LINE_AA)
    cv2.line(img, (x, y - r * 2), (x, y + r * 2), color, 1, cv2.LINE_AA)
    return (x, y)


# -----------------------------
# yaw/pitch/face 추출 (L2CS 결과 형식 가정: results.yaw, results.pitch, results.bboxes)
# -----------------------------
def extract_main_yaw_pitch_bbox(results, frame_W, frame_H):
    bboxes = getattr(results, "bboxes", None)
    pitch = getattr(results, "pitch", None)
    yaw = getattr(results, "yaw", None)
    if pitch is None or yaw is None:
        return None, None, None
    if hasattr(pitch, "shape") and pitch.shape[0] == 0:
        return None, None, None
    # 가장 큰 얼굴 인덱스
    idx = select_largest_face(bboxes) if bboxes is not None else 0
    if idx is None:
        return None, None, None
    try:
        y = float(pitch[idx])
        x = float(yaw[idx])
    except Exception:
        return None, None, None

    bbox = bboxes[idx]
    x1, y1, x2, y2 = safe_bbox_xyxy(bbox, frame_W, frame_H)
    return x, y, (x1, y1, x2, y2)  # (yaw, pitch, bbox)


# -----------------------------
# 메인
# -----------------------------
def main():
    # 장치/파이프라인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] device: {device}')
    torch.backends.cudnn.benchmark = True

    weights_path = Path(WEIGHTS)
    if not weights_path.exists():
        raise FileNotFoundError(f'Weights not found: {weights_path}')

    gaze_pipeline = Pipeline(weights=weights_path, arch='ResNet50', device=device)

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f'Camera not opened (index={CAM_INDEX}).')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 캘리브레이터
    calib = Poly2Calibrator()
    # 시작 시 로딩 시도
    calib.load(CALIB_FILE)

    targets = gen_3x3_targets()
    t_idx = None            # 현재 타깃 인덱스 (None = 캘리브레이션 중 아님)
    xs, ys = [], []         # 수집된 yaw/pitch
    sxs, sys = [], []       # 수집된 정답 정규좌표

    print('[INFO] Q=quit, C=calibrate, SPACE=capture, U=use on/off, S=save, L=load')

    ema_fps = None
    t_prev = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            now = time.time()
            dt = max(now - t_prev, 1e-6)
            t_prev = now
            inst_fps = 1.0 / dt
            ema_fps = inst_fps if ema_fps is None else (0.9 * ema_fps + 0.1 * inst_fps)

            # 추론
            try:
                with torch.inference_mode():
                    results = gaze_pipeline.step(frame)
                vis = render(frame.copy(), results)
            except Exception as e:
                vis = frame.copy()
                cv2.putText(vis, f'Error: {str(e)}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('L2CS (calib)', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            H, W = vis.shape[:2]
            yaw_val, pitch_val, bbox = extract_main_yaw_pitch_bbox(results, W, H)

            # (옵션) 디버깅용 화살표-끝점
            if bbox is not None and yaw_val is not None and pitch_val is not None:
                cx, cy, ex, ey = arrow_endpoint_like_render(bbox, pitch_val, yaw_val)
                # cv2.arrowedLine(vis, (cx, cy), (ex, ey), (0, 255, 255), 2, tipLength=0.25)

            # --- 캘리브레이션 진행 중이면 타깃/가이드 표시 ---
            if t_idx is not None:
                sx_t, sy_t = targets[t_idx]
                tx, ty = draw_target_overlay(vis, sx_t, sy_t, color=(0, 255, 255), r=TARGET_R)
                cv2.putText(vis, f'CALIB [{t_idx+1}/{len(targets)}]  SPACE=capture',
                            (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            # --- 캘리브레이션 결과 사용 시, 화면 시점 표시 ---
            if calib.use and (yaw_val is not None) and (pitch_val is not None) and (calib.cx is not None):
                sx, sy = calib.predict(yaw_val, pitch_val)
                gx = int(round(sx * (W - 1)))
                gy = int(round(sy * (H - 1)))
                cv2.circle(vis, (gx, gy), 6, (0, 0, 255), -1)  # 빨간 점 = 화면 시점(보정 후)
                cv2.putText(vis, f'GazePt({gx},{gy})  sx={sx:.3f} sy={sy:.3f}',
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

            # 보조 정보
            fps_text = f'FPS: {ema_fps:.1f}' if ema_fps else 'FPS: ...'
            cv2.putText(vis, fps_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

            if calib.use:
                cv2.putText(vis, 'CALIB: ON', (W - 140, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(vis, 'CALIB: OFF', (W - 150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2, cv2.LINE_AA)

            cv2.imshow('L2CS (calib)', vis)
            key = cv2.waitKey(1) & 0xFF

            # 키 입력 처리
            if key == ord('q') or key == ord('Q'):
                break

            elif key == ord('c') or key == ord('C'):
                # 캘리브레이션 시작/재시작
                t_idx = 0
                xs.clear(); ys.clear(); sxs.clear(); sys.clear()
                print('[CALIB] Start 3x3. Look at the target and press SPACE to capture.')

            elif key == ord(' ') and t_idx is not None:
                # 현재 타깃에서 샘플 수집: SPACE
                if yaw_val is None or pitch_val is None:
                    print('[CALIB] No face/gaze. Try again.')
                    continue

                # SAMPLE_FRAMES 만큼 평균 내기
                yaw_buf, pitch_buf = [], []
                for _ in range(SAMPLE_FRAMES):
                    ret2, frame2 = cap.read()
                    if not ret2 or frame2 is None:
                        continue
                    try:
                        with torch.inference_mode():
                            res2 = gaze_pipeline.step(frame2)
                        yv, pv, _bb = extract_main_yaw_pitch_bbox(res2, W, H)
                        if (yv is not None) and (pv is not None):
                            yaw_buf.append(yv)
                            pitch_buf.append(pv)
                    except Exception:
                        pass
                if len(yaw_buf) == 0:
                    print('[CALIB] No sample captured. Try again.')
                    continue

                yavg = float(np.mean(yaw_buf))
                pavg = float(np.mean(pitch_buf))
                sx_t, sy_t = targets[t_idx]

                xs.append(yavg); ys.append(pavg)
                sxs.append(sx_t); sys.append(sy_t)

                print(f'[CALIB] {t_idx+1}/{len(targets)} captured: '
                      f'yaw={yavg:.3f}, pitch={pavg:.3f} -> (sx,sy)=({sx_t:.2f},{sy_t:.2f})')

                t_idx += 1
                if t_idx >= len(targets):
                    # 모델 피팅
                    try:
                        calib.fit(xs, ys, sxs, sys)
                        calib.use = True
                        t_idx = None
                        print('[CALIB] Done. Calibration enabled (U to toggle, S to save).')
                    except Exception as e:
                        print(f'[CALIB] Fit failed: {e}')
                        calib.cx = calib.cy = None
                        calib.use = False
                        t_idx = None

            elif key == ord('u') or key == ord('U'):
                calib.use = not calib.use
                print(f'[CALIB] use = {calib.use}')

            elif key == ord('s') or key == ord('S'):
                try:
                    calib.save(CALIB_FILE)
                    print(f'[CALIB] Saved to {CALIB_FILE}')
                except Exception as e:
                    print(f'[CALIB] Save failed: {e}')

            elif key == ord('l') or key == ord('L'):
                ok = calib.load(CALIB_FILE)
                print(f'[CALIB] Load {CALIB_FILE}: {ok}')
                if ok:
                    calib.use = True

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

# 정확도 팁

# 타깃을 볼 때 머리를 최대한 고정하고 눈으로만 추적하면 결과가 좋습니다.

# 조명 균일, 얼굴이 프레임 중앙/적절한 크기(얼굴 bbox가 너무 작지 않게).

# 더 고정밀이 필요하면 5×3, 5×5 등 타깃 개수를 늘리고, SAMPLE_FRAMES를 20~30으로 올리세요.

# 특정 모니터 해상도(다른 창)로 직접 찍고 싶다면, vis 대신 별도 풀스크린 캘리브 창에서 타깃을 띄우는 구조로 바꿔줄 수도 있어요.