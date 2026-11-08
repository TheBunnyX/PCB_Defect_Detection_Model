"""Train and evaluate a PCB defect detector on the local Roboflow export."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime" / "ultralytics"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--profile", choices=("baseline", "rtx5070"), default="baseline",
                        help="rtx5070: measured RTX 5070 12 GB preset (batch 32, workers 2, disk cache, fast NCHW)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--cache", choices=("none", "disk", "ram"))
    parser.add_argument("--fast", action="store_true", help="Enable cuDNN autotuning and relax deterministic execution")
    parser.add_argument("--tf32", action="store_true", help="Allow Tensor Core acceleration of FP32 matrix multiplication")
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="pcb_yolo26s")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--resume", type=Path, help="Resume an interrupted last.pt checkpoint")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()
    if args.profile == "rtx5070":
        args.batch = 32 if args.batch is None else args.batch
        args.workers = 2 if args.workers is None else args.workers
        args.cache = "disk" if args.cache is None else args.cache
        args.channels_last = False if args.channels_last is None else args.channels_last
        args.fast = True

    import torch
    import yaml
    from ultralytics import YOLO

    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable. Install CUDA-enabled PyTorch, or pass --device cpu.")
    # Absolute paths avoid dependencies on the current directory or YOLO settings.
    config = yaml.safe_load((ROOT / "Dataset" / "data.yaml").read_text(encoding="utf-8"))
    config["path"] = (ROOT / "Dataset").as_posix()
    runtime = ROOT / ".runtime"
    runtime.mkdir(exist_ok=True)
    data = runtime / "pcb.yaml"
    data.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    overrides = {}
    for key in ("batch", "workers"):
        if getattr(args, key) is not None:
            overrides[key] = getattr(args, key)
    if args.cache is not None:
        overrides["cache"] = False if args.cache == "none" else args.cache
    if args.channels_last is not None:
        overrides["channels_last"] = args.channels_last
    model = YOLO(str(args.resume.resolve()) if args.resume else args.model)
    if args.fast or args.tf32:
        def enable_fast_mode(trainer):
            if args.fast:
                from ultralytics.utils.torch_utils import unset_deterministic
                unset_deterministic()
                trainer.args.deterministic = False
                torch.backends.cudnn.benchmark = True
            if args.tf32:
                torch.set_float32_matmul_precision("high")
                torch.backends.cudnn.allow_tf32 = True
            gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
            print(f"Performance mode: GPU={gpu.name if gpu else 'CPU'}, AMP={trainer.args.amp}, "
                  f"batch={trainer.batch_size}, workers={trainer.args.workers}, cache={trainer.args.cache}, "
                  f"cuDNN benchmark={torch.backends.cudnn.benchmark}, "
                  f"matmul precision={torch.get_float32_matmul_precision()}", flush=True)
        model.add_callback("on_train_start", enable_fast_mode)
    if args.resume:
        model.train(resume=True, **overrides)
    else:
        model.train(
            data=str(data), epochs=args.epochs, imgsz=args.imgsz,
            batch=overrides.get("batch", 8), device=args.device, workers=overrides.get("workers", 4),
            project=str(ROOT / "runs" / "detect"), name=args.name,
            patience=25, seed=42, deterministic=not args.fast, cache=overrides.get("cache", False),
            fraction=args.fraction, close_mosaic=10, amp=True,
            channels_last=overrides.get("channels_last"),
        )
    best = Path(model.trainer.best)
    print(f"Best checkpoint: {best}")
    # The test split is reserved until training and model selection finish.
    if not args.skip_test:
        metrics = YOLO(str(best)).val(
            data=str(data), split="test", imgsz=model.trainer.args.imgsz, batch=model.trainer.batch_size,
            device=model.trainer.args.device, workers=model.trainer.args.workers,
            project=str(ROOT / "runs" / "detect"), name=f"{best.parent.parent.name}_test",
        )
        destination = best.parent.parent / "test_metrics.json"
        destination.write_text(json.dumps(metrics.results_dict, indent=2), encoding="utf-8")
        print(f"Test metrics: {destination}")


if __name__ == "__main__":
    main()
