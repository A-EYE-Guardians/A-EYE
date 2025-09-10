# Dual-Camera Iris Gaze v2.2 (Single-Cam Ready)

## 무엇이 달라졌나?

- **Single-Cam 모드**(`--single_cam`) 추가: eye 카메라 **한 대로도** 9점 캘리브레이션과 호모그래피 파이프라인을 테스트할 수 있습니다.
- **월드 캔버스**를 eye 프레임으로 사용하고, 그 위에 캘리 타깃/시선 마커/가상 ArUco를 오버레이합니다.
- 호모그래피 테스트 3가지:
  1. **Identity**: 정규좌표 → 픽셀 1:1
  2. **Manual ROI**: `m`키로 TL→TR→BR→BL 4점을 클릭하여 시야 내 사각형을 정의
  3. **Virtual ArUco**: `v`키로 가상 ArUco 4개(TL=0,TR=1,BR=2,BL=3) 오버레이 후 자동 검출(opencv-contrib-python 필요)

v2.1의 핵심(눈-정렬 좌표계, 원맞춤, 깜박임 게이팅, 분리 EMA) 그대로 유지.

---

## 설치

```bash
python -m venv venv
source venv/bin/activate   # Windows: .\venv\Scripts\activate
pip install --upgrade pip
pip install mediapipe opencv-python numpy
# ArUco(가상/실물) 사용 시
pip install opencv-contrib-python
```

## 실행 예

- Single-Cam:

```bash
python single_camera_iris_gaze_v2_2.py --single_cam --eye_cam 0 --flip_eye --feat_gain 1.2 --blink_gate
```

- Dual-Cam(예전과 동일):

```bash
python single_camera_iris_gaze_v2_2.py --world_cam 0 --eye_cam 1 --flip_eye
```

## 조작키

- `q` 종료, `h` HUD, `d` Eye 축/홍채 표시, `r` EMA 리셋
- `c` 캘리브 on/off → 초록 점을 바라보고 `SPACE`로 샘플(9점)
- `g` 보정 on/off, `S` 저장, `L` 로드
- `a` 호모그래피 on/off (H가 준비되어야 적용)
- `m` 수동 ROI 모드 토글: TL→TR→BR→BL **순서로 4번 클릭**하면 H가 설정됨
- `v` 가상 ArUco 오버레이 on/off (contrib 필요). 오버레이가 켜진 상태에서 자동으로 H 추정됨

## 팁

- Single-Cam에서 캘리브레이션은 **Eye 창에 보이는 초록 점**을 바라보며 진행합니다.
- `--feat_ema_alpha 0.45 --coord_ema_alpha 0.25 --feat_gain 1.2 --blink_gate` 권장
- 가상 ArUco는 파이프라인 테스트용입니다(실물 보드와 결과 값은 다릅니다).

## 라이선스

MIT (MediaPipe/OpenCV/Numpy는 각 라이선스 따름)
