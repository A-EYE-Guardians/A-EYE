import os, sys, time, json, argparse
from math import hypot
from datetime import datetime
from pathlib import Path
import cv2, numpy as np

# ---- detector.py 접근 보장 ----
THIS = Path(__file__).resolve()
DETECTOR_CH = THIS.parent.parent                 # .../detector_ch
PROJECT_ROOT = DETECTOR_CH.parents[2]            # .../A-EYE
if str(DETECTOR_CH) not in sys.path:
    sys.path.insert(0, str(DETECTOR_CH))

from detector import build_detector
from scripts.gaze_sources import build_gaze

WIN = "A-EYE • YOLO-World"

# ---------------- Fixation ----------------
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

# ---------------- Utils ----------------
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

# ---- 여기(유틸 바로 아래)에 추가된 유틸 2개가 들어갑니다 ----
def clamp(v, lo, hi): 
    return max(lo, min(hi, v))

def expand_box(box, margin, w, h):
    x1, y1, x2, y2 = map(float, box)
    dw = (x2 - x1) * margin
    dh = (y2 - y1) * margin
    nx1 = clamp(int(x1 - dw), 0, w-1)
    ny1 = clamp(int(y1 - dh), 0, h-1)
    nx2 = clamp(int(x2 + dw), 0, w-1)
    ny2 = clamp(int(y2 + dh), 0, h-1)
    return nx1, ny1, nx2, ny2
# -----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaze", default="mouse", choices=["mouse","eye"],
                    help="mouse=커서, eye=아이트래킹")
    ap.add_argument("--labels", default=str((DETECTOR_CH / "data/open_vocab.txt").resolve()))
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--source", default="0", help="웹캠 인덱스(0) 또는 동영상 경로")
    ap.add_argument("--roi", type=int, default=288)  # 기본 정사각 ROI(백업용)

    # ▼ 새로 추가된 옵션 3개
    ap.add_argument("--roi-ladder", type=str, default="192,288,384",
                    help="고정 발생 시 1차 스캔용 정사각 ROI 크기 후보(px, 콤마구분)")
    ap.add_argument("--roi-margin", type=float, default=0.2,
                    help="검출 박스 대비 가변 ROI 확장 비율")
    ap.add_argument("--refine-imgsz", type=int, default=640,
                    help="2차 정밀 스캔 입력 크기")

    ap.add_argument("--fix", type=float, default=3.0)
    ap.add_argument("--px", type=int, default=25)
    ap.add_argument("--no-dot", action="store_true", help="시선 점 숨김")
    ap.add_argument("--no-fixbox", action="store_true", help="FIXATION 표시 숨김")
    ap.add_argument("--pick-tol", type=int, default=40)
    ap.add_argument("--auto-save", action="store_true")
    ap.add_argument("--speak", action="store_true", help="macOS 'say'로 라벨 읽기")
    args = ap.parse_args()

    # Detector 준비: 빠른 스캔/정밀 스캔 분리
    det_fast = build_detector("yoloworld", args.labels, conf_thr=args.conf, imgsz=args.imgsz)
    det_refine = det_fast if args.refine_imgsz == args.imgsz else \
        build_detector("yoloworld", args.labels, conf_thr=args.conf, imgsz=args.refine_imgsz)

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
            # 1) 작은 정사각 ROI들(사다리)로 빠른 1차 스캔
            sizes = [int(s) for s in args.roi_ladder.split(",") if s.strip()]
            cand_box = None
            cand_dets = []
            for s in sizes:
                roi_img, (ox, oy) = crop_roi(frame, int(gx), int(gy), roi=s)
                dets_roi = det_fast.infer(roi_img)
                # 전역 좌표화
                for d in dets_roi:
                    x1, y1, x2, y2 = d["xyxy"]
                    d["xyxy"] = (x1+ox, y1+oy, x2+ox, y2+oy)
                picked = pick_nearest(dets_roi, gx, gy, tol=args.pick_tol)
                if picked:
                    cand_box = picked["xyxy"]
                    cand_dets = dets_roi
                    break

            # 2) 후보 없으면 사다리의 마지막 크기를 사용
            if cand_box is None:
                s = sizes[-1] if sizes else args.roi
                roi_img, (ox, oy) = crop_roi(frame, int(gx), int(gy), roi=s)
                dets_roi = det_fast.infer(roi_img)
                for d in dets_roi:
                    x1, y1, x2, y2 = d["xyxy"]
                    d["xyxy"] = (x1+ox, y1+oy, x2+ox, y2+oy)
                cand_dets = dets_roi
                picked = pick_nearest(cand_dets, gx, gy, tol=args.pick_tol)
                cand_box = picked["xyxy"] if picked else (ox, oy, ox+roi_img.shape[1], oy+roi_img.shape[0])

            # 3) 박스를 기반으로 가변 ROI 확장 → 2차 정밀 스캔
            x1, y1, x2, y2 = expand_box(cand_box, args.roi_margin, w, h)
            roi2 = frame[y1:y2, x1:x2]
            dets_ref = det_refine.infer(roi2)
            for d in dets_ref:
                bx1, by1, bx2, by2 = d["xyxy"]
                d["xyxy"] = (bx1 + x1, by1 + y1, bx2 + x1, by2 + y1)

            last_dets = dets_ref if dets_ref else cand_dets
            last_gazed = pick_nearest(last_dets, gx, gy, tol=args.pick_tol)

            if not args.no_fixbox:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
                cv2.putText(frame, "FIXATION", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)

            if args.auto_save:
                fn = datetime.now().strftime("%H%M%S_%f")
                out = draw_boxes(frame.copy(), last_dets, (0,255,0), 2)
                cv2.imwrite(os.path.join(run_dir, f"{fn}.jpg"), out)
                with open(os.path.join(run_dir, f"{fn}.json"), "w", encoding="utf-8") as f:
                    json.dump({
                        "gaze":[gx,gy], 
                        "final_roi":[int(x1),int(y1),int(x2),int(y2)],
                        "dets": last_dets,
                        "gazed": last_gazed
                    }, f, ensure_ascii=False, indent=2)

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


'''
cd A-EYE/model/detector/detector_ch
python -u scripts/cam_gaze_mux.py --gaze mouse --conf 0.30 --imgsz 640 \
  --roi-ladder 192,288,384 --roi-margin 0.2 --refine-imgsz 640 \
  --fix 3.0 --px 25


python -u scripts/cam_gaze_mux.py --gaze mouse --speak


'''