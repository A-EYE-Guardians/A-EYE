# l2cs_level1_gazepoint.py
# -*- coding: utf-8 -*-
"""
L2CS-Net 시선 결과(yaw/pitch)로 '시점 점'을 이미지 평면에 표시하는 데모 (레벨 1)
- 필요 패키지: opencv-python, torch, numpy, l2cs
- Windows 웹캠: CAP_DSHOW 사용
- 파이프라인에서 얼굴/각도 필드명이 달라도 최대한 안전하게 추출하는 헬퍼 포함
"""

import cv2
import time
import math
import torch
import numpy as np
from pathlib import Path
from l2cs import Pipeline, render


# -----------------------------
# 유틸: 안전한 getattr/키 접근
# -----------------------------
def get_attr_or_key(obj, name, default=None):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    return default


# ------------------------------------------
# yaw/pitch 추출(여러 배포본 호환을 위한 안전한 시도)
# ------------------------------------------
def extract_yaw_pitch_faces(results):
    """
    결과 객체에서 [yaws], [pitchs], [faces]를 최대한 안전하게 뽑아낸다.
    반환: (yaws_deg(list[float]), pitchs_deg(list[float]), faces(list))
    - 각도 단위는 '도(degree)' 로 맞춰 반환
    - faces 는 (x1,y1,x2,y2) 또는 dict 포함 가능 (렌더링용으로 중심점만 사용)
    """
    # 후보 키들(배포본에 따라 다름)
    yaw_keys   = ["yaws", "yaw_list", "yaw", "yaws_deg"]
    pitch_keys = ["pitchs", "pitch_list", "pitch", "pitchs_deg"]
    faces_keys = ["faces", "bboxes", "boxes", "face_boxes"]

    yaws = None
    for k in yaw_keys:
        v = get_attr_or_key(results, k, None)
        if v is not None:
            yaws = v
            break

    pitchs = None
    for k in pitch_keys:
        v = get_attr_or_key(results, k, None)
        if v is not None:
            pitchs = v
            break

    faces = None
    for k in faces_keys:
        v = get_attr_or_key(results, k, None)
        if v is not None:
            faces = v
            break

    # 타입 표준화(list)
    def to_list(x):
        if x is None:
            return None
        if isinstance(x, (list, tuple)):
            return list(x)
        if hasattr(x, "tolist"):
            try:
                return x.tolist()
            except Exception:
                pass
        # 단일 스칼라일 수도 있음 → 리스트로 감싸기
        return [x]

    yaws  = to_list(yaws)
    pitchs = to_list(pitchs)
    faces = faces if isinstance(faces, (list, tuple)) else ([] if faces is None else [faces])

    # 길이 맞추기(안전)
    n = 0
    if yaws is not None:   n = max(n, len(yaws))
    if pitchs is not None: n = max(n, len(pitchs))
    if faces is not None:  n = max(n, len(faces))

    if yaws is None:   yaws = [0.0] * n
    if pitchs is None: pitchs = [0.0] * n
    if faces is None:  faces = [None] * n

    # 라디안 → 도 변환(대부분 L2CS는 '도'지만, 혹시 모를 케이스 방어)
    def to_degree_list(vals):
        out = []
        for v in vals:
            try:
                v = float(v)
            except Exception:
                v = 0.0
            # 라디안일 가능성: 절댓값이 3.14 근처 범위
            if abs(v) <= math.pi + 0.1:
                # -99~+99도 대신 -3.14~+3.14가 들어왔다면 → 라디안으로 판단
                # 하지만 이미 도일 수도 있으니, -3~+3 사이만 라디안으로 간주(경계형성)
                if abs(v) <= 3.2:
                    v = math.degrees(v)
            out.append(v)
        return out

    yaws_deg   = to_degree_list(yaws)
    pitchs_deg = to_degree_list(pitchs)

    return yaws_deg, pitchs_deg, list(faces)


# --------------------------------
# yaw/pitch(도) → 3D 시선 단위벡터
# --------------------------------
def yawpitch_to_vec(yaw_deg: float, pitch_deg: float):
    """
    L2CS 관례(yaw 좌우, pitch 상하)를 단위벡터로 변환
    (카메라 좌표계 기준, render와 동일 방향성 가정)
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    x = -math.cos(pitch) * math.sin(yaw)
    y = -math.sin(pitch)
    z = -math.cos(pitch) * math.cos(yaw)
    # 정규화(혹시 모를 수치 오차)
    n = (x*x + y*y + z*z) ** 0.5
    if n == 0:
        return (0.0, 0.0, -1.0)
    return (x/n, y/n, z/n)


# ------------------------------------------------
# 레벨 1: 이미지 평면으로 '시점 점' 좌표를 계산(픽셀)
# ------------------------------------------------
def imagepoint_from_gaze(frame_shape, face_box, yaw_deg, pitch_deg, gain=250):
    """
    frame_shape: (H, W, C)
    face_box   : (x1,y1,x2,y2) 또는 dict/None. 얼굴 중심을 기준점으로 사용.
    yaw_deg, pitch_deg: L2CS 추정 각(도)
    gain: 벡터를 화면에서 얼마나 뻗을지(픽셀 스케일)
    """
    H, W = frame_shape[:2]

    # 얼굴 bbox → 중심점 계산 (dict 지원)
    cx, cy = W // 2, H // 2
    if face_box is not None:
        if isinstance(face_box, dict):
            # 흔한 필드명 대응
            x1 = face_box.get("x1") or face_box.get("xmin") or face_box.get("left")
            y1 = face_box.get("y1") or face_box.get("ymin") or face_box.get("top")
            x2 = face_box.get("x2") or face_box.get("xmax") or face_box.get("right")
            y2 = face_box.get("y2") or face_box.get("ymax") or face_box.get("bottom")
        else:
            try:
                x1, y1, x2, y2 = face_box
            except Exception:
                x1 = y1 = 0
                x2, y2 = W - 1, H - 1
        try:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
        except Exception:
            pass

    vx, vy, vz = yawpitch_to_vec(yaw_deg, pitch_deg)

    # 레벨 1: 매우 단순화 - x,y만 사용해서 이미지 평면으로 "점" 투영
    px = int(round(cx + gain * vx))
    py = int(round(cy + gain * vy))

    # 경계 클램프
    px = max(0, min(W - 1, px))
    py = max(0, min(H - 1, py))
    return (px, py)


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

            # --- 4) 추론 ---
            try:
                with torch.inference_mode():
                    results = gaze_pipeline.step(frame)  # 내부: 얼굴검출+전처리+추론

                # 기본 렌더(화살표 등)
                vis = render(frame.copy(), results)

                # yaw/pitch/face 추출(여러 구현 호환)
                yaws_deg, pitchs_deg, faces = extract_yaw_pitch_faces(results)

                # 첫 얼굴 기준으로 '시점 점' 찍기(여러 명이면 반복문으로 그려도 됨)
                if len(yaws_deg) > 0 and len(pitchs_deg) > 0:
                    yaw0 = float(yaws_deg[0])
                    pitch0 = float(pitchs_deg[0])
                    face0 = faces[0] if len(faces) > 0 else None

                    px, py = imagepoint_from_gaze(vis.shape, face0, yaw0, pitch0, gain=250)
                    cv2.circle(vis, (px, py), 6, (0, 0, 255), -1)  # 시점 점(빨강)
                    cv2.putText(vis, f'GazePt:({px},{py})  Y:{yaw0:.1f}  P:{pitch0:.1f}',
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)

                # FPS/얼굴 수 오버레이(가능한 경우)
                faces_count = len(faces) if faces is not None else 0
                cv2.putText(vis, f'faces: {faces_count}', (10, 30),
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
