import os
import cv2
import time
import argparse
import torch
from ultralytics import YOLO

def select_device():
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def get_capture(source: int | str):
    # macOS에서 AVFoundation 백엔드가 안정적
    cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION) if isinstance(source, int) else cv2.VideoCapture(source)
    if not cap.isOpened():
        cap = cv2.VideoCapture(source)  # 자동 백엔드 재시도
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap

def draw_boxes(frame, result, names):
    if result.boxes is None:
        return frame
    for box in result.boxes:
        b = box.xyxy[0].int().tolist()
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label = f"{names[cls_id]} {conf:.2f}"
        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        cv2.putText(frame, label, (b[0], max(15, b[1]-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return frame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="웹캠 인덱스(예: 0) 또는 비디오 경로")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="YOLO 가중치 경로/이름")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    device = select_device()
    print(f"[INFO] Device: {device}")

    model = YOLO(args.weights)
    names = model.names

    cap = get_capture(source)
    if not cap or not cap.isOpened():
        raise SystemExit("❗️카메라/비디오 열기 실패. --source 값을 확인하세요.")

    prev_t = time.time()
    fps = 0.0

    print("[INFO] Press 'q' to quit, 'c' to capture frame into ./data/")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] 프레임을 가져올 수 없습니다. 종료합니다.")
            break

        results = model.predict(
            frame,
            conf=args.conf,
            imgsz=args.imgsz,
            device=device,
            verbose=False
        )

        r = results[0]
        frame = draw_boxes(frame, r, names)

        now = time.time()
        dt = now - prev_t
        prev_t = now
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps

        cv2.putText(frame, f"MPS:{'Y' if device=='mps' else 'N'}  FPS:{fps:5.1f}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("A-EYE | detection_ch (webcam)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            os.makedirs("data", exist_ok=True)
            save_path = f"data/capture_{int(time.time())}.jpg"
            cv2.imwrite(save_path, frame)
            print(f"[INFO] Saved: {save_path}")

    cap.release()
    cv2.destroyAllWindows()

    if device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

if __name__ == "__main__":
    main()
