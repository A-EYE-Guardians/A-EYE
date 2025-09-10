# Single-Camera Iris Gaze v2.5

**v2.5 (v2.4 → v2.5) 업그레이드 요약**

- **확장 피처** 도입: `f = [vx, vy, ap, s, yaw, pitch]`
  - `vx, vy` : (v2.4) 로컬 세로 정규화 기반 눈-정렬 좌표
  - `ap` : aperture = `gap_half / hw` (u 위치별 세로 여백/가로폭 비)
  - `s` : scale = `1 / hw` (눈 크기/거리 변화 보정)
  - `yaw, pitch` : FaceMesh 기반 간이 머리자세 지표(옵션: `--use_head_pose`)
- **Ridge(릿지) 2차 다항 보정기**:
  - 화이트닝(μ, W) → 2차 다항(선형+자승+쌍곱) → L2(λ) 정규화
  - λ는 `--ridge_lambda`로 제어 (기본 1e-3)
- **One Euro Filter**로 최종 (sx, sy) 안정화:
  - `--euro_min_cutoff`, `--euro_beta`, `--euro_d_cutoff` 튜닝 가능

> v2.4의 “로컬 세로 정규화 + gap_half 깜박임 게이팅 + 호모그래피”는 그대로 유지됩니다.

---

## 1) 설치

Windows PowerShell 예시:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip
pip install mediapipe opencv-python opencv-contrib-python numpy
```

## 실행

단일 카메라(eye만, world=eye 복제)

```
python .\single_camera_iris_gaze_v2_5.py --single_cam --eye_cam 0 --flip_eye `
  --sens_gain_x 4.0 --sens_gain_y 4.0 --sens_gamma_x 0.8 --sens_gamma_y 0.8 `
  --blink_gate --ridge_lambda 1e-3 --use_head_pose `
  --euro_min_cutoff 1.2 --euro_beta 0.01 --euro_d_cutoff 1.0

```

듀얼 카메라(월드/아이 분리)

```
python .\single_camera_iris_gaze_v2_5.py --world_cam 0 --eye_cam 1 --flip_eye `
  --blink_gate --ridge_lambda 1e-3 --use_head_pose

```

```
3) 조작 키

q : 종료
h : HUD on/off
d : 디버그 on/off
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
```
