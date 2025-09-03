import argparse, os
from PIL import Image, ImageDraw
from ultralytics import YOLO

def draw_and_save(img_path, preds, out_path):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for b in preds.boxes:
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        conf = float(b.conf[0])
        cls  = int(b.cls[0])
        name = preds.names.get(cls, str(cls))
        draw.rectangle([x1, y1, x2, y2], outline=(0,255,0), width=3)
        draw.text((x1+3, y1+3), f"{name} {conf:.2f}", fill=(0,255,0))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path)
    print("[OK] saved ->", out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--out", default="outputs\\detect_result.jpg")
    ap.add_argument("--thr", type=float, default=0.25)
    ap.add_argument("--weights", default="yolov8n.pt")
    args = ap.parse_args()

    model = YOLO(args.weights)
    res = model.predict(args.img, imgsz=640, conf=args.thr, verbose=False)[0]
    draw_and_save(args.img, res, args.out)

if __name__ == "__main__":
    main()
