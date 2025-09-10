# Dual-Camera Iris Gaze v2.1

**변경점(v2.1)**  
- 눈-정렬 좌표계(눈꼬리 축)에서의 **홍채 중심 상대좌표 (vx, vy)** 사용 → 눈만 굴려도 민감하게 반응  
- 홍채 중심은 iris 링에 **원 맞춤(Least-Squares)** 으로 계산 → 노이즈에 강함  
- **깜박임/부분가림 게이팅**(hv 대비 hv_ema 비율) 옵션 추가  
- **특징 EMA**와 **좌표 EMA**를 분리 설정 가능 (`--feat_ema_alpha`, `--coord_ema_alpha`)  
- **감도 스케일** `--feat_gain` 추가 (체감 튜닝/디버깅용, 보정 후에는 영향 작음)

---

## 설치

```powershell
# Windows 기준
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip

# 필수
pip install mediapipe opencv-python numpy

# (선택) ArUco 사용 시 필수
pip install opencv-contrib-python
```

## 실행 예

```bash
python dual_camera_iris_gaze_v2_1.py --world_cam 0 --eye_cam 1 --world_w 1280 --world_h 720 --eye_w 640 --eye_h 480 --flip_eye --feat_gain 1.2 --blink_gate
```

## 조작키

- `q` 종료, `h` 도움말 HUD, `d` 디버그(눈 축/홍채 표시), `r` EMA 리셋  
- `c` 캘리브 모드 on/off → 화면 초록 점을 **바라본 뒤 `SPACE`** 로 샘플 채집(9점)  
- `g` 보정 on/off, `S` 저장, `L` 로드  
- `a` ArUco 호모그래피 on/off (ID: TL=0, TR=1, BR=2, BL=3)  
- `b` 깜박임/부분가림 게이팅 on/off

## 주요 옵션

- `--feat_gain` : (vx,vy) 감도 부스트(1.0~1.5 권장). 보정 전 체감 튜닝에 유용.  
- `--feat_ema_alpha`, `--coord_ema_alpha` : 기본은 `--ema_alpha`를 사용.  
  - 예) `--feat_ema_alpha 0.45 --coord_ema_alpha 0.25`  
- `--blink_gate --blink_v_ratio 0.6` : hv < 0.6×hv_ema 이면 프레임을 무시(깜박임/가림 억제).

## 정확도 팁

1. 디버그(`d`) 켜고 눈 축과 홍채 중심이 잘 따라오는지 확인  
2. 9점 캘리브 시 각 점에서 **여러번 SPACE**로 샘플을 쌓아 평균화  
3. 조명이 반사되면 iris 검출이 흔들림 → 각도/조명 조정  
4. 월드 평면에 **ArUco 4코너**(0,1,2,3) 정확히 배치 → 호모그래피 on

## 라이선스

- 코드: MIT  
- MediaPipe / OpenCV / Numpy는 각 라이선스를 따릅니다.
