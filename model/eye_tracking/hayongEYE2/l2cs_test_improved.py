# l2cs_test_improved.py
# -*- coding: utf-8 -*-
import cv2
import time
import torch
import numpy as np
from pathlib import Path
# Pipeline: L2CS-Net 시선 추정 전체 파이프라인(얼굴 검출→전처리→추론).
# render: 원본 프레임 위에 시선 결과(박스/벡터 등) 오버레이.
from l2cs import Pipeline, render 

def get_attr_or_key(obj, name, default=None):
    """obj가 객체든 dict든 안전하게 name을 가져온다."""
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    return default

def main():
    # --- 0) 장치/백엔드 설정 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] device: {device}')
    torch.backends.cudnn.benchmark = True  # 고정 해상도라면 유리

    # --- 1) 가중치 경로 검증 ---
    weights_path = Path(r"C:\Gukbi\hayongEYE2\models\L2CSNet_gaze360.pkl")
    if not weights_path.exists():
        raise FileNotFoundError(f'Weights not found: {weights_path}')

    # --- 2) L2CS 파이프라인 생성 ---
    gaze_pipeline = Pipeline(
        weights=weights_path,
        arch='ResNet50',
        device=device
    )
    # print(gaze_pipeline.model) # 모델 전체 구조


    # --- 3) 카메라 열기 ---
    cam_index = 0
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)  # Windows에서 안정적
    if not cap.isOpened():
        raise RuntimeError(f'Camera not opened (index={cam_index}). 다른 인덱스를 시도하세요.')

    # 해상도/코덱/FPS/버퍼 최적화(가능한 항목만 적용됨)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 최신 프레임 지향
    except Exception:
        pass

    # 워밍업 몇 프레임 읽기(노이즈/노출 안정화)
    for _ in range(5):
        cap.read()

    print('[INFO] Press Q to quit.')
    ema_fps = None
    t_prev = time.time()

    # (옵션) 마지막 정상 프레임 백업
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
                # 안전한 빈 화면 표시
                vis = last_vis.copy()
                cv2.putText(vis, 'Empty frame. Skipping...', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('L2CS-Net (gaze)', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # --- 4) 추론 (얼굴 0개 예외 안전) ---
            try:
                with torch.inference_mode():
                    results = gaze_pipeline.step(frame)  # 내부: 얼굴검출+전처리+추론
                vis = render(frame.copy(), results)
                faces = get_attr_or_key(results, 'faces', [])
                try:
                    faces_found = len(faces) if faces is not None else 0
                except TypeError:
                    faces_found = 0
                cv2.putText(vis, f'faces: {faces_found}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            except ValueError:
                # 예: "need at least one array to stack" (얼굴 미검출)
                vis = frame.copy()
                cv2.putText(vis, 'No face detected', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            except Exception as e:
                vis = frame.copy()
                cv2.putText(vis, f'Error: {str(e)}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            # --- 5) FPS 오버레이(EMA) ---
            cv2.putText(vis, f'FPS: {ema_fps:.1f}' if ema_fps else 'FPS: ...', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

            last_vis = vis  # 빈 프레임 발생 시 대비
            cv2.imshow('L2CS-Net (gaze)', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
