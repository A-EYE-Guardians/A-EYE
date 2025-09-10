# l2cs_test.py
# -*- coding: utf-8 -*-
import cv2
import torch
from pathlib import Path
from l2cs import Pipeline, render
import time

def main():
    # 0) 장치 선택
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] device: {device}')

    # 1) 가중치 절대경로 (필요시 수정)
    weights_path = Path(r"C:\Gukbi\hayongEYE2\models\L2CSNet_gaze360.pkl")
    if not weights_path.exists():
        raise FileNotFoundError(f'Weights not found: {weights_path}')

    # 2) L2CS 파이프라인 생성
    gaze_pipeline = Pipeline(
        weights=weights_path,
        arch='ResNet50',
        device=device
    )

    # 3) 웹캠 열기 (인덱스 0↔1↔2 바꿔가며 테스트)
    cam_index = 0
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)  # Windows 안정성
    if not cap.isOpened():
        raise RuntimeError(f'Camera not opened (index={cam_index}). 다른 인덱스를 시도하세요.')

    # 해상도 낮추기 (검출 성공률↑)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print('[INFO] Press Q to quit.')
    fps_t0, fps_cnt = time.time(), 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            cv2.imshow("L2CS-Net (No frame)", (255 * (frame if frame is not None else 
                         255 * (0))).__class__)  # noop to avoid lint
            print('[WARN] Empty frame. Skipping...')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # 4) L2CS 파이프라인 실행 (얼굴 0개일 때 예외 방지)
        try:
            results = gaze_pipeline.step(frame)  # 내부에서 얼굴검출+전처리+추론
            # 성공 시 결과를 그려줍니다.
            vis = render(frame.copy(), results)
            faces_found = len(getattr(results, 'faces', [])) if hasattr(results, 'faces') else -1
            cv2.putText(vis, f'faces: {faces_found}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        except ValueError as e:
            # 보통 "need at least one array to stack" 여기로 들어옵니다.
            vis = frame.copy()
            cv2.putText(vis, 'No face detected', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        except Exception as e:
            # 다른 예외는 로그만 찍고 프레임 표시
            vis = frame.copy()
            cv2.putText(vis, f'Error: {str(e)}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # FPS 표기
        fps_cnt += 1
        if time.time() - fps_t0 >= 1.0:
            cv2.putText(vis, f'FPS: {fps_cnt}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            fps_t0 = time.time()
            fps_cnt = 0

        cv2.imshow('L2CS-Net (gaze)', vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
