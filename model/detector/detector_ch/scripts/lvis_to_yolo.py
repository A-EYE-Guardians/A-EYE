# detection_ch/scripts/lvis_to_yolo.py
import os, json, argparse
from pathlib import Path
from collections import defaultdict
from urllib.parse import urlparse
from tqdm import tqdm
from PIL import Image

def load_json(p: Path):
    with p.open("r") as f:
        return json.load(f)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def build_cat_maps(categories, keep_names=None):
    """LVIS category id → 연속 YOLO id 매핑과 names 리스트 생성."""
    if keep_names:
        keep = [n.strip() for n in keep_names.split(",") if n.strip()]
        keep_set = set(keep)
        cats = [c for c in categories if c["name"] in keep_set]
        if not cats:
            raise SystemExit("❗ --keep 에 해당하는 카테고리가 없습니다.")
        # 사용자가 준 순서를 그대로 보존
        cats.sort(key=lambda c: keep.index(c["name"]))
    else:
        cats = list(categories)
        cats.sort(key=lambda c: c["id"])
    id_map = {c["id"]: i for i, c in enumerate(cats)}   # LVIS id → 0..K-1
    names  = [c["name"] for c in cats]
    return id_map, names

def index_image_paths(root: Path):
    """root 하위 모든 파일을 '파일명 → 절대경로'로 매핑(한 번만)."""
    idx = {}
    for p in root.rglob("*"):
        if p.is_file():
            idx[p.name] = p
    return idx

def resolve_filename(im: dict):
    """
    LVIS JSON에서 이미지 파일명을 최대한 튼튼하게 추출.
    우선순위: file_name → coco_url basename → coco_id → id
    """
    # 1) 표준 키
    fn = im.get("file_name")
    if fn:
        return Path(fn).name  # 혹시 경로가 포함돼 있으면 파일명만
    # 2) URL에서 파일명
    url = im.get("coco_url")
    if url:
        base = os.path.basename(urlparse(url).path)
        if base:
            return base
    # 3) coco_id 를 12자리 0패딩 jpg로
    if "coco_id" in im:
        try:
            return f"{int(im['coco_id']):012d}.jpg"
        except Exception:
            pass
    # 4) 마지막 대안: id를 12자리 jpg로
    try:
        return f"{int(im['id']):012d}.jpg"
    except Exception:
        return None

def to_yolo_line(bbox, W, H, yolo_cid):
    # LVIS bbox: [x, y, w, h] (pixel)
    x, y, w, h = bbox
    if w <= 1 or h <= 1:  # 너무 작은 박스 무시
        return None
    cx = x + w / 2.0
    cy = y + h / 2.0
    return f"{yolo_cid} {cx/W:.6f} {cy/H:.6f} {w/W:.6f} {h/H:.6f}"

def process_split(json_path: Path, images_dir: Path, labels_dir: Path, id_map, limit=None):
    data = load_json(json_path)
    images = data["images"]
    anns   = data["annotations"]

    # image_id → (file_name, width, height)
    imginfo = {}
    for im in images:
        fname = resolve_filename(im)
        if not fname:
            continue
        W = im.get("width")
        H = im.get("height")
        imginfo[im["id"]] = (fname, W, H)

    # image_id → anns
    ann_by_im = defaultdict(list)
    for a in anns:
        if a["category_id"] in id_map:
            ann_by_im[a["image_id"]].append(a)

    # 파일 경로 인덱스(한 번만 스캔)
    path_index = index_image_paths(images_dir)
    ensure_dir(labels_dir)

    stats = dict(total=len(images), written=0, empty=0, missing_imgs=0, no_path=0, need_probe=0)
    it = images if limit is None else images[:limit]
    for im in tqdm(it, desc=f"{json_path.name}"):
        im_id = im["id"]
        info = imginfo.get(im_id)
        if not info:
            stats["missing_imgs"] += 1
            continue

        fname, W, H = info
        # 이미지 경로 찾기
        img_path = path_index.get(Path(fname).name)
        if img_path is None:
            stats["missing_imgs"] += 1
            continue

        # JSON에 W/H가 없으면 실제 파일에서 한 번만 읽어서 보완
        if (not W) or (not H) or W == 0 or H == 0:
            try:
                with Image.open(img_path) as pil_im:
                    W, H = pil_im.size
                stats["need_probe"] += 1
            except Exception:
                stats["missing_imgs"] += 1
                continue

        lbl_path = labels_dir / (Path(fname).stem + ".txt")
        lines = []
        for a in ann_by_im.get(im_id, []):
            yid = id_map[a["category_id"]]
            line = to_yolo_line(a["bbox"], W, H, yid)
            if line:
                lines.append(line)

        if lines:
            lbl_path.write_text("\n".join(lines))
            stats["written"] += 1
        else:
            lbl_path.write_text("")  # 빈 파일 생성(호환성)
            stats["empty"] += 1

    return stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--val-json",   required=True)
    ap.add_argument("--img-root",   required=True, help=".../data/images (train, val 하위 포함)")
    ap.add_argument("--out-root",   default=None)
    ap.add_argument("--keep",       default="", help="쉼표구분 클래스만 유지(선택). 예: 'person,chair,laptop'")
    ap.add_argument("--yaml-name",  default="data_lvis.yaml")
    ap.add_argument("--limit",      type=int, default=None, help="테스트용: 각 split 최대 N장만 처리")
    args = ap.parse_args()

    img_root = Path(args.img_root)
    out_root = Path(args.out_root) if args.out_root else img_root.parent
    labels_train = out_root / "labels" / "train"
    labels_val   = out_root / "labels" / "val"
    ensure_dir(labels_train); ensure_dir(labels_val)

    # 카테고리 매핑 준비(Train JSON 기준)
    train_data = load_json(Path(args.train_json))
    id_map, names = build_cat_maps(train_data["categories"], args.keep)

    s_train = process_split(Path(args.train_json), img_root, labels_train, id_map, limit=args.limit)
    s_val   = process_split(Path(args.val_json),   img_root, labels_val,   id_map, limit=args.limit)

    yaml_path = out_root / args.yaml_name
    yaml_path.write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n" + "".join([f"  - {n}\n" for n in names])
    )

    print("\n=== DONE LVIS → YOLO ===")
    print("labels train :", labels_train)
    print("labels val   :", labels_val)
    print("data.yaml    :", yaml_path)
    print("classes      :", len(names))
    print("train stats  :", s_train)
    print("val stats    :", s_val)
    if s_train.get("missing_imgs", 0) or s_val.get("missing_imgs", 0):
        print("※ 경고: JSON에 있으나 이미지가 없는 항목이 있습니다. 이미지 경로/파일명을 다시 확인하세요.")

if __name__ == "__main__":
    main()
