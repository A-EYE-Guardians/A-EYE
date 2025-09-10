# Single-Camera Iris Gaze v2.4

**핵심 개선점(v2.3 → v2.4)**

- **로컬 세로 정규화(Local-vertical normalization)**:  
  vy를 “현재 u(가로 위치)에서의 위/아래 눈꺼풀 경계까지 세로 반폭(gap_half)”로 정규화하여,  
  중앙/가장자리 등 위치에 따라 달라지는 세로 여백을 보정 → **상하 감도↑, 대각 응답 자연스러움↑**
- **화이트닝(Whitening) 프리보정**:  
  (증폭 특징)의 공분산을 이용해 선형 화이트닝 후 2차 다항 회귀 → **대각 왜곡/상관 완화**
- **깜박임 게이팅 개선(옵션)**:  
  기존 hv 대비 → **gap_half 기반**으로 닫힘/가림을 더 정확히 필터링
- **디버그 강화**:  
  Eye 디버그 창에 (u,v) 경계 곡선 + iris 위치 인셋 표시

---

## 1) 요구 사항

Windows 기준(권장):

```powershell
py -3.11 -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip
pip install mediapipe opencv-python opencv-contrib-python numpy
```

## 2) 실행 예시

단일 카메라(eye만, 화면=eye 프레임 복제)

```
python .\single_camera_iris_gaze_v2_4.py --single_cam --eye_cam 0 --flip_eye `
  --sens_gain_x 4.0 --sens_gain_y 6.0 --sens_gamma_x 0.8 --sens_gamma_y 0.7 --blink_gate
```

듀얼 카메라(월드/아이 분리)

```
python .\single_camera_iris_gaze_v2_4.py --world_cam 0 --eye_cam 1 --flip_eye --blink_gate
```

## 3) 조작 키

q : 종료
h : HUD on/off
d : 디버그 on/off (Eye 창에 (u,v) 인셋 표시)
r : EMA 리셋
c : 캘리브레이션 모드 on/off
SPACE : 캘리 타깃 샘플 채집(총 9점)
g : 보정(캘리브 적용) on/off
S : 캘리브 파일 저장 (--calib_path)
L : 캘리브 파일 로드
a : 호모그래피 on/off
m : 수동 ROI(호모그래피용 사변형 4점 클릭: TL→TR→BR→BL)
v : 가상 ArUco 오버레이 on/off (contrib 설치 필요)
n : Neutral(중립 오프셋 EMA) on/off
N : Neutral 즉시 리셋
X/x : X축 감도 +10% / -10%
Y/y : Y축 감도 +10% / -10%
