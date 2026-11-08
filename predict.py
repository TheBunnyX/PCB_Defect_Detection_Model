"""Save PCB detections from an image, folder, or video."""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime" / "ultralytics"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--weights", default=str(ROOT / "runs/detect/pcb_yolo26s/weights/best.pt"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if not Path(args.weights).is_file():
        raise SystemExit(f"Checkpoint not found: {args.weights}. Train first or pass --weights.")
    from ultralytics import YOLO
    for _ in YOLO(args.weights).predict(
        source=args.source, conf=args.conf, device=args.device,
        imgsz=640, save=True, save_txt=True, save_conf=True, stream=True,
        project=str(ROOT / "runs/detect"), name="predict",
    ):
        pass


if __name__ == "__main__":
    main()
