"""Measure real training throughput; exit before validation or saving weights."""
import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime/ultralytics"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    args = parser.parse_args()
    import torch
    import psutil
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import unset_deterministic

    measured = {"count": 0}
    def start(trainer):
        unset_deterministic()
        torch.backends.cudnn.benchmark = True
        if args.tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()

    def batch_end(trainer):
        torch.cuda.synchronize()
        measured["count"] += 1
        if measured["count"] == 5:
            measured["start"] = time.perf_counter()
        if measured["count"] == 25:
            elapsed = time.perf_counter() - measured["start"]
            result = {
                "requested_batch": args.batch, "batch": trainer.batch_size,
                "channels_last": args.channels_last, "tf32": args.tf32,
                "workers": args.workers, "cache": "disk", "warmup_batches": 5,
                "measured_batches": 20, "seconds": elapsed,
                "images_per_second": 20 * trainer.batch_size / elapsed,
                "peak_cuda_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
                "peak_cuda_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
                "available_ram_gb": psutil.virtual_memory().available / 2**30,
            }
            suffix = ("_nhwc" if args.channels_last else "") + ("_tf32" if args.tf32 else "")
            path = ROOT / f".runtime/benchmark_b{args.batch}_w{args.workers}{suffix}.json"
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print("BENCHMARK_RESULT " + json.dumps(result), flush=True)
            raise SystemExit(0)

    model = YOLO(str(ROOT / ".runtime/before-optimization-last.pt"))
    model.add_callback("on_train_start", start)
    model.add_callback("on_train_batch_end", batch_end)
    model.train(
        data=str(ROOT / ".runtime/pcb.yaml"), epochs=100, imgsz=640,
        batch=args.batch, workers=args.workers, device="0", cache="disk",
        amp=True, deterministic=False, plots=False, save=False,
        channels_last=args.channels_last,
        project=str(ROOT / "runs/benchmark"), name=f"b{args.batch}_w{args.workers}" + ("_nhwc" if args.channels_last else "") + ("_tf32" if args.tf32 else ""),
        seed=42, val=False,
    )


if __name__ == "__main__":
    main()
