# Single-Camera Iris Gaze v2.3 — Aggressive Sensitivity

이 버전은 **눈동자의 작은 움직임도 월드 화면이 크게 반응**하도록 설계되었습니다.  
핵심은 (vx,vy) 특징에 대해 **데드존 → 감마 비선형(γ<1) → 축별 게인 → 포화**를 거쳐 증폭하고,  
그 **증폭된 특징을 캘리브레이션의 입력**으로 사용한다는 점입니다.

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

## 실행(단일 카메라 예)

```
python .\single_camera_iris_gaze_v2_3.py --single_cam --eye_cam 0 --flip_eye --sens_gain_x 4.0 --sens_gain_y 4.0 --sens_gamma_x 0.8 --sens_gamma_y 0.8 --blink_gate
```

## 파이프라인

```
Eye → FaceMesh → (vx,vy)  →  [Neutral EMA 제거]  →  SensAmplifier  →  {Calib ON: 2차다항} → (sx,sy)
{Calib OFF: tanh fallback} → (sx,sy)

(sx,sy) → [Homography H] → World 픽셀 표시
```

SensAmplifier

Deadzone: 미세 떨림 억제

Gamma(γ<1): 중심부 민감도 상승 (0.8 권장)

Gain: 축별 스케일

Saturation: 끝단 과민 방지

Neutral EMA (기본 ON)

사용자의 중립 시선 오프셋을 느리게 추정해 빼줍니다. 캘리브 중에는 자동으로 동결됩니다.

키 n으로 on/off, N으로 즉시 리셋

조작키

q 종료, h HUD, d Eye 축/홍채 표시, r EMA 리셋

c 캘리브 on/off → 초록 점을 바라보고 SPACE로 각 점을 샘플(총 9점)

g 보정 on/off, S 저장, L 로드

a 호모그래피 on/off, m 수동 ROI(TL→TR→BR→BL 클릭), v 가상 ArUco on/off(contrib 필요)

n Neutral on/off, N Neutral reset

X/x X축 감도 +10%/-10%, Y/y Y축 감도 +10%/-10%

튜닝 가이드

반응성 강화: --sens_gain_x/y를 3~6 범위로 조정. 너무 크면 끝단 포화가 잦아짐

중심 민감도: --sens_gamma_x/y를 0.7~0.95 (작을수록 중심에서 더 민감)

미세 떨림 억제: --sens_deadzone 0.01~0.03

캘리브레이션: 증폭된 특징을 기준으로 회귀하므로, 증폭 파라미터를 대략 맞춘 뒤 9점 캘리브 진행 권장

보정 OFF 테스트: g로 끄고 tanh fallback만으로도 큰 이동이 되는지 확인 → 이후 보정 ON으로 정밀화

주의

단일 카메라 모드에서는 Eye 프레임이 곧 월드 캔버스이므로, 호모그래피는 수동 ROI 또는 가상 ArUco로 실험용으로만 사용하세요.

실제 월드 카메라를 쓸 때는 물리적 마커(ArUco/체커보드)를 이용해 H를 측정하는 것을 권장합니다.
