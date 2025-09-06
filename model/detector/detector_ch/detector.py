from typing import List

def read_labels_txt(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

def build_detector(backend: str, labels_path: str, **kwargs):
    labels = read_labels_txt(labels_path)
    b = backend.lower()
    if b in ("yoloworld", "yolo-world", "yw"):
        # 패키지/로컬 둘 다 대응
        try:
            from .backends.yoloworld_backend import YoloWorldBackend
        except Exception:
            from backends.yoloworld_backend import YoloWorldBackend
        return YoloWorldBackend(labels=labels, **kwargs)
    raise ValueError(f"Unknown backend: {backend}")
