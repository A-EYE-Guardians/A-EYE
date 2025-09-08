import argparse, time, math
import cv2
import numpy as np
from ultralytics import YOLOE  

# ----------------------------
# 유틸: 마스크 그리기 (반투명)
# ----------------------------
def draw_mask(image, mask, alpha=0.4):
    if mask is None:
        return image
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8)
    overlay = image.copy()
    color = (0, 255, 0)  # 초록 (OpenCV BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, thickness=cv2.FILLED)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

# ----------------------------
# 유틸: 시선(마우스)과 가장 가까운 박스 선택
# ----------------------------
def pick_gaze_target(boxes_xyxy, gaze_xy):
    if boxes_xyxy.size == 0:
        return -1
    cx = 0.5*(boxes_xyxy[:,0] + boxes_xyxy[:,2])
    cy = 0.5*(boxes_xyxy[:,1] + boxes_xyxy[:,3])
    d2 = (cx - gaze_xy[0])**2 + (cy - gaze_xy[1])**2
    return int(np.argmin(d2))

# ----------------------------
# 메인
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="웹캠 인덱스(예: 0) 또는 이미지/비디오 경로")
    ap.add_argument("--weights", default="yoloe-11s-seg-pf.pt", help="프롬프트-프리 가중치")
    ap.add_argument("--imgsz", default=640, type=int)
    ap.add_argument("--conf", default=0.25, type=float)
    ap.add_argument("--device", default=None, help="cuda:0 | mps | cpu (None=자동)")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    ap.add_argument("--agnostic_nms", action="store_true", help="Class-agnostic NMS")
    ap.add_argument("--max_det", type=int, default=30, help="Max detections per image")
    args = ap.parse_args()

    # 모델 로드: 프롬프트-프리 변형(내장 4,585 클래스 태그셋 기반) :contentReference[oaicite:2]{index=2}
    model = YOLOE(args.weights)

    # 비디오 소스 열기
    src = int(args.source) if (args.source.isdigit() and len(args.source) < 3) else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"❗ 소스를 열 수 없습니다: {args.source}")

    # 마우스(=시선) 좌표
    gaze = {"xy": (0, 0)}
    win = "YOLOE Prompt-Free (press q to quit)"
    cv2.namedWindow(win)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            gaze["xy"] = (x, y)
    cv2.setMouseCallback(win, on_mouse)

    # 프레임 루프
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 추론 (스트림 모드 아님: 한 프레임씩)
        t0 = time.time()
        results = model.predict(
            source=frame,            # 또는 infer_input (ROI 쓰는 버전이면)
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,                    # ← 추가
            agnostic_nms=args.agnostic_nms,  # ← 추가
            max_det=args.max_det,            # ← 추가
            device=args.device,
            stream=False,
            verbose=False
        )
        dt = (time.time() - t0) * 1000.0

        # 결과 파싱 (첫 배치만)
        res = results[0]
        im = frame.copy()

        boxes = res.boxes  # Boxes object
        masks = res.masks  # Segmentation masks (None 가능)
        names = res.names  # 클래스 id -> 라벨명 dict

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls  = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()

            # 시선과 가장 가까운 박스 선택
            gi = pick_gaze_target(xyxy, gaze["xy"])

            # 마스크 배열 정렬
            mask_arr = None
            if masks is not None and masks.data is not None:
                # (N, H, W) float32
                mask_arr = masks.data.cpu().numpy()

            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                c = cls[i]; p = conf[i]
                label = f"{names.get(c, str(c))} {p:.2f}"

                # 박스/라벨
                color = (0, 255, 255) if i == gi else (0, 200, 0)
                cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(im, label, (int(x1), max(0, int(y1)-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

                # 선택 객체 마스크 강조
                # if mask_arr is not None and i < mask_arr.shape[0] and i == gi:
                #     im = draw_mask(im, mask_arr[i].astype(np.float32), alpha=0.35)

        # 시선(마우스) 표시
        cv2.circle(im, (int(gaze["xy"][0]), int(gaze["xy"][1])), 5, (255, 255, 255), -1)
        cv2.putText(im, f"{dt:.1f} ms", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 220, 20), 2)

        cv2.imshow(win, im)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()




'''
python run_yoloe_pf.py --source 0 --iou 0.5 --conf 0.45 --agnostic_nms --max_det 30
'''