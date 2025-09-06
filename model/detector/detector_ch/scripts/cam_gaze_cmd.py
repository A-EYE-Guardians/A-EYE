import os, time, math, webbrowser, tempfile
import cv2
import numpy as np
import torch
from ultralytics import YOLO

def annotate_all(frame, result, names, color=(0, 255, 0)):
    """모든 탐지 박스에 라벨과 점수를 그립니다."""
    if getattr(result, "boxes", None) is None:
        return
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label = f"{names[cls_id]} {conf:.2f}"

        # 박스
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 텍스트 가독성용 배경 + 라벨
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = y1 - 6 if y1 - 6 > th + 2 else y1 + th + 6
        cv2.rectangle(frame, (x1, y_text - th - 4), (x1 + tw + 6, y_text + 4), (0, 0, 0), -1)
        cv2.putText(frame, label, (x1 + 3, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)


# ---------- 설정 ----------
DWELL_SEC = 1.0          # 시선 머문 시간 임계
ZOOM_SCALE = 2.0         # 확대 배율
ZOOM_WIN = (240, 240)    # 확대 ROI 크기 (w,h)
STT_SECONDS = 4          # v키를 누른 뒤 녹음 길이(초)
CONF_THR = 0.5
IMGSZ = 640
WEIGHTS = "best.pt"
# --------------------------

# 시선(=마우스) 상태
gaze_xy = None           # (x, y)
_last_move_t = time.time()
dwell_locked = False
zoom_on = False          # "확대" 명령 토글
last_cmd_text = ""

def on_mouse(event, x, y, flags, param):
    global gaze_xy, _last_move_t, dwell_locked
    if event == cv2.EVENT_MOUSEMOVE:
        gaze_xy = (x, y)
        _last_move_t = time.time()
        dwell_locked = False

def select_device():
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def nearest_detection_to_gaze(result, gaze):
    """가장 가까운 박스(중심점 기준)와 거리 반환."""
    if result.boxes is None or gaze is None:
        return None, None
    gx, gy = gaze
    best, best_d = None, None
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        d = math.hypot(cx - gx, cy - gy)
        if best_d is None or d < best_d:
            best, best_d = box, d
    return best, best_d

def draw_gaze(frame, locked):
    if gaze_xy is None:
        return frame
    color = (0, 255, 255) if locked else (255, 255, 0)
    cv2.circle(frame, gaze_xy, 8, color, 2)
    return frame

def draw_zoom(frame):
    if not (zoom_on and gaze_xy):
        return None
    h, w = frame.shape[:2]
    gw, gh = ZOOM_WIN
    x, y = gaze_xy
    x1 = max(0, int(x - gw // 2))
    y1 = max(0, int(y - gh // 2))
    x2 = min(w, x1 + gw)
    y2 = min(h, y1 + gh)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    zoom = cv2.resize(roi, (int(gw * ZOOM_SCALE), int(gh * ZOOM_SCALE)))
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
    return zoom

def alias_to_coco(target_words):
    """‘커피’ → cup/bottle 처럼 COCO 이름 매핑 후보 리턴."""
    words = set(target_words)
    if "커피" in words:
        return ["cup", "bottle"]
    if "신발" in words or "운동화" in words or "shoes" in words:
        return ["shoe"]  # COCO에 'sports ball'은 공, 신발은 'shoe'
    return []

def execute_command(text, result, frame, names):
    """간단한 규칙기반 의도 파싱 → 액션 실행."""
    global zoom_on, last_cmd_text
    t = text.strip().lower()
    last_cmd_text = text

    # 1) 확대/줌
    if ("확대" in t) or ("줌" in t) or ("zoom" in t):
        zoom_on = not zoom_on
        return f"확대 토글: {'ON' if zoom_on else 'OFF'}"

    # 2) '커피 어딨어'류: 특정 카테고리 가장 가까운 박스 찾기
    if ("어디" in t) or ("어딨어" in t) or ("where" in t):
        # 키워드에서 카테고리 후보 뽑기
        candidates = alias_to_coco(t.split())  # 간단 분할
        if not candidates:
            candidates = ["cup", "bottle"]  # 기본
        # result에서 해당 카테고리만 골라, 시선과 가장 가까운 것 표시
        if result.boxes is not None:
            target = None; best_d = None
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                label = names[cls_id]
                if label not in candidates:
                    continue
                b = box.xyxy[0].tolist()
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                if gaze_xy is None: 
                    d = 0
                else:
                    d = math.hypot(cx - gaze_xy[0], cy - gaze_xy[1])
                if best_d is None or d < best_d:
                    target, best_d = (b, label), d
            if target:
                x1,y1,x2,y2 = map(int, target[0])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)
                cv2.putText(frame, f"{target[1]} (near gaze)", (x1, max(15,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
                if gaze_xy:
                    cv2.line(frame, gaze_xy, (int((x1+x2)/2), int((y1+y2)/2)), (0,0,255), 2)
                return f"가까운 {target[1]} 강조 표시"
        return "해당 물체를 찾지 못했습니다."

    # 3) '이거 검색/구글에 검색': 시선 근처 물체 라벨로 구글 검색 탭 오픈
    if ("검색" in t) or ("search" in t) or ("구글" in t):
        box, _ = nearest_detection_to_gaze(result, gaze_xy)
        if box is None:
            return "시선 근처 물체가 없습니다."
        cls_id = int(box.cls[0].item())
        label = names[cls_id]
        url = f"https://www.google.com/search?q={label}"
        try:
            webbrowser.open(url)
            return f"구글 검색: {label}"
        except Exception:
            return f"브라우저 열기 실패: {label}"

    # 매칭 안됨
    return "알아듣지 못했습니다. (예: '확대', '커피 어딨어', '이거 검색')"

def transcribe_whisper(seconds=STT_SECONDS, device="cpu"):
    """v 키로 누르면 음성 인식(한/영 혼용 가능)."""
    import sounddevice as sd, soundfile as sf, whisper
    sr = 16000
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()

    # 임시 wav로 저장 → Whisper
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        sf.write(f.name, audio, sr)
        model = whisper.load_model("base", device=device)
        res = model.transcribe(f.name, language="ko")
    try:
        os.unlink(f.name)
    except Exception:
        pass
    return res.get("text", "").strip()

def main():
    device = select_device()
    print(f"[INFO] Device: {device}")

    model = YOLO(WEIGHTS)
    names = model.names

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("A-EYE | gaze+cmd")
    cv2.setMouseCallback("A-EYE | gaze+cmd", on_mouse)

    prev_t = time.time(); fps = 0.0
    print("[INFO] q: 종료 / v: 음성명령(STT) / z: 확대 토글")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # dwell 체크
        global dwell_locked
        if time.time() - _last_move_t >= DWELL_SEC:
            dwell_locked = True

        # 추론
        results = model.predict(frame, conf=CONF_THR, imgsz=IMGSZ, device=device, verbose=False)
        r = results[0]

        annotate_all(frame, r, names)

        # 시선 표시
        draw_gaze(frame, dwell_locked)

        # 시선 근처 최단 박스 얇은 테두리로 보조 표시
        nb, _ = nearest_detection_to_gaze(r, gaze_xy)
        if nb is not None:
            b = nb.xyxy[0].int().tolist()
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 200, 255), 2)

        # 확대창
        zoom_img = draw_zoom(frame)

        # FPS
        now = time.time(); dt = now - prev_t; prev_t = now
        fps = 0.9 * fps + 0.1 * (1.0/dt) if dt > 0 else fps
        status = f"MPS:{'Y' if device=='mps' else 'N'} FPS:{fps:4.1f} Dwell:{'LOCK' if dwell_locked else '...'}"
        cv2.putText(frame, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        if last_cmd_text:
            cv2.putText(frame, f"CMD: {last_cmd_text}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,255,180), 2, cv2.LINE_AA)

        # 표시
        cv2.imshow("A-EYE | gaze+cmd", frame)
        if zoom_img is not None:
            cv2.imshow("Zoom", zoom_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('z'):
            zoom_on = not zoom_on
        elif key == ord('v'):
            # push-to-talk: v를 누르면 STT 실행
            try:
                print("[STT] 말씀하세요…")
                text = transcribe_whisper(seconds=STT_SECONDS, device=device if device in ("mps","cuda") else "cpu")
                print("[STT] ▶", text)
                msg = execute_command(text, r, frame, names)
                print("[CMD]", msg)
            except Exception as e:
                print("[STT ERROR]", e)

    cap.release()
    cv2.destroyAllWindows()
    if device == "mps":
        try: torch.mps.empty_cache()
        except: pass

if __name__ == "__main__":
    main()
