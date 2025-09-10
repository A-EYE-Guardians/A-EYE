# l2cs_level1_gazepoint_aligned.py
# -*- coding: utf-8 -*-
"""
L2CS vis.py(render)와 '완전히 같은' 원점/부호/스케일로
화살표 끝점 == '시점 점' 을 표시하는 데모

- 원점: 얼굴 bbox 중심
- 부호: ex = cx - L * sin(yaw), ey = cy - L * sin(pitch)
- 길이: L = max(w, h) * scale  (scale=0.5 기본값, 필요시 조절)
- pitch, yaw: L2CS 결과값을 '그대로' 사용(라디안/도 변환 없음) → render와 동일 가정
"""

import cv2
import time
import math
import torch
import numpy as np
from pathlib import Path
from l2cs import Pipeline, render


def safe_bbox_xyxy(bbox, W, H):
    """vis.py가 하는 것처럼 x_min,y_min은 0 아래로 내려가지 않게 안전 처리"""
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = int(bbox[2])
    y_max = int(bbox[3])
    if x_min < 0: x_min = 0
    if y_min < 0: y_min = 0
    # (x_max, y_max는 vis.py에서 min 클램프를 안 하지만, 화면 그리기 위해선 안전 클램프가 유용)
    x_max = min(int(W - 1), x_max)
    y_max = min(int(H - 1), y_max)
    return x_min, y_min, x_max, y_max


def arrow_endpoint_like_render(bbox_xyxy, pitch, yaw, scale=0.5):
    """
    vis.py의 draw_gaze와 동일한 수학적 형태:
      - 원점: bbox 중심
      - 끝점: (cx - L*sin(yaw), cy - L*sin(pitch))
      - L = max(w, h) * scale
    ※ pitch, yaw 단위(라디안/도)는 L2CS 결과값과 '동일'하게 취급 (변환 X)
    """
    x_min, y_min, x_max, y_max = bbox_xyxy
    w  = x_max - x_min
    h  = y_max - y_min
    cx = int(round(x_min + w * 0.5))
    cy = int(round(y_min + h * 0.5))

    L = int(round(max(w, h) * float(scale)))

    # draw_gaze의 추정 투영식과 동일한 부호 사용
    ex = int(round(cx - L * math.sin(float(yaw))))
    ey = int(round(cy - L * math.sin(float(pitch))))
    return cx, cy, ex, ey


def main():
    # --- 0) 장치/백엔드 설정 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] device: {device}')
    torch.backends.cudnn.benchmark = True  # 고정 해상도 추론 최적화

    # --- 1) 가중치 경로 검증 ---
    weights_path = Path(r"C:\Gukbi\hayongEYE2\models\L2CSNet_gaze360.pkl")
    if not weights_path.exists():
        raise FileNotFoundError(f'Weights not found: {weights_path}')

    # --- 2) L2CS 파이프라인 생성 ---
    gaze_pipeline = Pipeline(weights=weights_path, arch='ResNet50', device=device)

    # --- 3) 카메라 열기 ---
    cam_index = 0
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)  # Windows에서 안정적
    if not cap.isOpened():
        raise RuntimeError(f'Camera not opened (index={cam_index}). 다른 인덱스를 시도하세요.')

    # 해상도/코덱/FPS/버퍼 최적화(장치 호환되는 항목만 적용됨)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    # 워밍업
    for _ in range(5):
        cap.read()

    print('[INFO] Press Q to quit.')
    ema_fps = None
    t_prev = time.time()
    last_vis = np.zeros((480, 640, 3), dtype=np.uint8)

    try:
        while True:
            ret, frame = cap.read()
            now = time.time()
            dt = max(now - t_prev, 1e-6)
            t_prev = now
            inst_fps = 1.0 / dt
            ema_fps = inst_fps if ema_fps is None else (0.9 * ema_fps + 0.1 * inst_fps)

            if not ret or frame is None:
                vis = last_vis.copy()
                cv2.putText(vis, 'Empty frame. Skipping...', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('L2CS-Net (gaze)', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            H, W = frame.shape[:2]

            # --- 4) 추론 ---
            try:
                with torch.inference_mode():
                    results = gaze_pipeline.step(frame)  # 내부: 얼굴검출+전처리+추론

                # L2CS 기본 렌더(박스/화살표 등)
                vis = render(frame.copy(), results)

                # vis.py 기준: results.bboxes(list/np.ndarray of [x1,y1,x2,y2]),
                #              results.pitch(shape[0]==N), results.yaw(shape[0]==N)
                # → render 소스와 동일 가정 하에 직접 순회
                n_faces = int(getattr(results.pitch, "shape", [0])[0])
                n_faces = min(n_faces, len(results.bboxes))

                for i in range(n_faces):
                    bbox = results.bboxes[i]
                    pitch = float(results.pitch[i])
                    yaw   = float(results.yaw[i])

                    # render와 동일 안전 클램프 적용
                    x_min, y_min, x_max, y_max = safe_bbox_xyxy(bbox, W, H)

                    # 화살표와 '정확히 같은' 끝점 계산
                    cx, cy, ex, ey = arrow_endpoint_like_render(
                        (x_min, y_min, x_max, y_max),
                        pitch, yaw,
                        scale=0.5  # 필요하면 0.4~0.6 사이에서 조정해 render와 완전 일치
                    )

                    # 시점 점 = 화살표 끝점
                    cv2.circle(vis, (ex, ey), 6, (0, 0, 255), -1)  # 빨강 점
                    # (선택) 우리가 계산한 화살표도 같이 그려서 일치 확인
                    # cv2.arrowedLine(vis, (cx, cy), (ex, ey), (0, 255, 255), 2, tipLength=0.25)

                # 보조 정보
                cv2.putText(vis, f'faces: {n_faces}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            except ValueError:
                vis = frame.copy()
                cv2.putText(vis, 'No face detected', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            except Exception as e:
                vis = frame.copy()
                cv2.putText(vis, f'Error: {str(e)}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            # FPS(EMA) 표기
            fps_text = f'FPS: {ema_fps:.1f}' if ema_fps else 'FPS: ...'
            cv2.putText(vis, fps_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

            last_vis = vis
            cv2.imshow('L2CS-Net (gaze)', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
