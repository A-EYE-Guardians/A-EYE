import numpy as np
import torch
from typing import List, Dict, Any

# Ultralytics YOLO-World 사용
_HAS_YOLOWORLD = False
try:
    from ultralytics import YOLOWorld as _YOLO_CLASS
    _HAS_YOLOWORLD = True
except Exception:
    from ultralytics import YOLO as _YOLO_CLASS  # 구버전 호환

def _select_device_str() -> str | int:
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return 0
    return "cpu"

def _expand_labels(lines: List[str]) -> List[str]:
    out: List[str] = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"): continue
        out.extend([w.strip() for w in s.split("|") if w.strip()])
    return out or ["object"]

class YoloWorldBackend:
    """
    Ultralytics YOLO-World 래퍼
    - set_classes로 프롬프트(라벨 목록) 주입
    - infer(image_bgr) -> [{xyxy, score, label}]
    """
    def __init__(self, labels: List[str],
                 weights: str = "yolov8s-worldv2.pt",
                 conf_thr: float = 0.30,
                 imgsz: int = 640,
                 device: str|int|None = None):
        self.class_texts = _expand_labels(labels)
        self.device = device if device is not None else _select_device_str()
        self.conf_thr = conf_thr
        self.imgsz = imgsz

        self.model = _YOLO_CLASS(weights)
        set_classes = getattr(self.model, "set_classes", None)
        if callable(set_classes):
            set_classes(self.class_texts)
        else:
            raise RuntimeError(
                "Ultralytics 버전에서 set_classes를 지원하지 않습니다. `pip install -U ultralytics` 후 다시 시도하세요."
            )

    @torch.no_grad()
    def infer(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        results = self.model.predict(
            source=image_bgr, device=self.device, conf=self.conf_thr,
            imgsz=self.imgsz, verbose=False
        )
        if not results: return []
        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0: return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls  = boxes.cls.cpu().numpy().astype(int)
        names = r.names

        out: List[Dict[str, Any]] = []
        for i in range(xyxy.shape[0]):
            x1,y1,x2,y2 = map(float, xyxy[i].tolist())
            out.append({
                "xyxy": (x1,y1,x2,y2),
                "score": float(conf[i]),
                "label": names.get(int(cls[i]), "object")
            })
        return out
