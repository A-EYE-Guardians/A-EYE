import os, sys, time, json, argparse
from math import hypot
from datetime import datetime
from pathlib import Path
import cv2, numpy as np

THIS = Path(__file__).resolve()
DETECTOR_CH = THIS.parent.parent
PROJECT_ROOT = DETECTOR_CH.parents[2]
if str(DETECTOR_CH) not in sys.path:
    sys.path.insert(0, str(DETECTOR_CH))

from detector import build_detector
from scripts.gaze_sources import build_gaze

WIN = "A-EYE • YOLO-World"

class Fixation:
    def __init__(self, fix_sec=3.0, px_thr=25):
        self.fix_sec = fix_sec; self.px_thr = px_thr
        self.anchor = None; self.t0 = None
    def update(self, x, y, now_ts):
        if self.anchor is None:
            self.anchor = (x, y); self.t0 = now_ts; return False
        d = hypot(x - self.anchor[0], y - self.anchor[1])
        if d <= self.px_thr:
            return (now_ts - self.t0) >= self.fix_sec
        else:
            self.anchor = (x, y); self.t0 = now_ts; return False

def ensure_dir(p): os.makedirs(p, exist_ok=True); return p

def draw_boxes(img, dets, color=(0,255,0), thickness=2):
    for d in dets:
        x1,y1,x2,y2 = map(int, d["xyxy"])
        cv2.rectangle(img,(x1,y1),(x2,y2),color,thickness)
        cv2.putText(img,f'{d["label"]} {d["score"]:.2f}',
                    (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img

def crop_roi(img, cx, cy, roi=256):
    h,w = img.shape[:2]
    x1=max(0, cx-roi//2); y1=max(0, cy-roi//2)
    x2=min(w, x1+roi);    y2=min(h, y1+roi)
    return img[y1:y2, x1:x2], (x1, y1)

def point_box_distance(px, py, box):
    x1,y1,x2,y2 = box
    if x1<=px<=x2 and y1<=py<=y2: return 0.0
    dx=max(x1-px, 0, px-x2); dy=max(y1-py, 0, py-y2)
    return float(np.hypot(dx,dy))

def pick_nearest(dets, x, y, tol=40):
    if not dets: return None
    best, best_d = None, 1e9
    for d in dets:
        x1,y1,x2,y2 = map(float, d["xyxy"])
        dmin = point_box_distance(x,y,(x1,y1,x2,y2))
        if dmin<best_d: best_d, best = dmin, d
    return best if best_d<=tol else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaze", default="mouse", choices=["mouse","eye"],
                    help="mouse=커서, eye=아이트래킹")
    ap.add_argument("--labels", default=str((DETECTOR_CH / "data/open_vocab.txt").resolve()))
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--source", default="0", help="웹캠 인덱스(0) 또는 동영상 경로")
    ap.add_argument("--roi", type=int, default=288)
    ap.add_argument("--fix", type=float, default=3.0)
    ap.add_argument("--px", type=int, default=25)
    ap.add_argument("--no-dot", action="store_true", help="시선 점 숨김")
    ap.add_argument("--no-fixbox", action="store_true", help="FIXATION 표시 숨김")
    ap.add_argument("--pick-tol", type=int, default=40)
    ap.add_argument("--auto-save", action="store_true")
    ap.add_argument("--speak", action="store_true", help="macOS 'say'로 라벨 읽기")
    args = ap.parse_args()

    det = build_detector("yoloworld", args.labels, conf_thr=args.conf, imgsz=args.imgsz)
    gaze = build_gaze(args.gaze)

    cap = cv2.VideoCapture(0 if args.source=="0" else args.source)
    if not cap.isOpened(): raise SystemExit("❗ 카메라/영상 열기 실패")

    cv2.namedWindow(WIN); gaze.attach(WIN)

    fix = Fixation(fix_sec=args.fix, px_thr=args.px)
    last_dets = []; last_gazed = None
    run_dir = ensure_dir(os.path.join(PROJECT_ROOT, "runs", "mux", datetime.now().strftime("%Y%m%d-%H%M%S")))
    print("[INFO] q 종료 / s 저장 / e 음성 / r 리셋")

    while True:
        ok, frame = cap.read()
        if not ok: break
        h,w = frame.shape[:2]
        gx, gy = gaze.get(w, h)

        if not args.no_dot:
            cv2.circle(frame, (int(gx), int(gy)), 6, (0,200,255), -1)

        triggered = fix.update(gx, gy, time.time())
        if triggered:
            roi_img, (ox, oy) = crop_roi(frame, gx, gy, roi=args.roi)
            dets_roi = det.infer(roi_img)
            for d in dets_roi:
                x1,y1,x2,y2 = d["xyxy"]; d["xyxy"]=(x1+ox, y1+oy, x2+ox, y2+oy)
            last_dets = dets_roi
            last_gazed = pick_nearest(last_dets, gx, gy, tol=args.pick_tol)

            if not args.no_fixbox:
                cv2.rectangle(frame, (ox, oy), (ox+roi_img.shape[1], oy+roi_img.shape[0]), (255,0,0), 2)
                cv2.putText(frame, "FIXATION", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)

            if args.auto_save:
                fn = datetime.now().strftime("%H%M%S_%f")
                out = draw_boxes(frame.copy(), last_dets, (0,255,0), 2)
                cv2.imwrite(os.path.join(run_dir, f"{fn}.jpg"), out)
                with open(os.path.join(run_dir, f"{fn}.json"), "w", encoding="utf-8") as f:
                    json.dump({"gaze":[gx,gy], "dets":last_dets, "gazed":last_gazed}, f, ensure_ascii=False, indent=2)

        vis = draw_boxes(frame.copy(), last_dets, (0,255,0), 2)
        if last_gazed:
            x1,y1,x2,y2 = map(int, last_gazed["xyxy"])
            cv2.rectangle(vis, (x1,y1), (x2,y2), (255,255,0), 3)
            cv2.putText(vis, "[GAZED]", (x1, max(0,y1-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        cv2.imshow(WIN, vis)
        k = cv2.waitKey(1) & 0xFF
        if k==ord('q'): break
        if k==ord('r'): last_dets=[]; last_gazed=None; fix.anchor=None; fix.t0=None
        if k==ord('s'):
            fn = datetime.now().strftime("%H%M%S_%f")
            cv2.imwrite(os.path.join(run_dir, f"{fn}.jpg"), vis)
            with open(os.path.join(run_dir, f"{fn}.json"), "w", encoding="utf-8") as f:
                json.dump({"gaze":[gx,gy], "dets":last_dets, "gazed":last_gazed}, f, ensure_ascii=False, indent=2)
            print(f"[SAVE] {fn}.jpg")
        if k==ord('e') and args.speak and last_gazed:
            os.system(f"say '{last_gazed['label']}'")

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
