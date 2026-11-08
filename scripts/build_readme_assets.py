"""Build GitHub README figures from a real training run and validation predictions."""
import argparse
import csv
import hashlib
import json
import os
import shutil
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime/ultralytics"))


def font(size, bold=False):
    from PIL import ImageFont
    for candidate in (
        Path("C:/Windows/Fonts") / ("seguisb.ttf" if bold else "segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu") / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
    ):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default(size=size)


def validation_examples(names):
    selected = {}
    used = set()
    for label in sorted((ROOT / "Dataset/valid/labels").glob("*.txt")):
        lines = [line.split() for line in label.read_text().splitlines() if line.strip()]
        classes = {int(line[0]) for line in lines}
        image = next((ROOT / "Dataset/valid/images" / (label.stem + suffix)
                      for suffix in (".jpg", ".jpeg", ".png")
                      if (ROOT / "Dataset/valid/images" / (label.stem + suffix)).is_file()), None)
        if image is None or image in used:
            continue
        for cls in sorted(classes):
            if cls in range(len(names)) and cls not in selected:
                selected[cls] = (image, len(lines))
                used.add(image)
                break
        if len(selected) == len(names):
            return [selected[i] for i in range(len(names))]
    raise SystemExit("Need validation images and labels covering every class. Download Dataset first.")


def training_figure(rows, best_epoch, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    epochs = [int(row["epoch"]) for row in rows]
    def values(key):
        return [float(row[key]) for row in rows]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), constrained_layout=True)
    fig.suptitle(f"PCB defect detection | Training snapshot: {epochs[-1]} epochs", fontsize=17, fontweight="bold")
    blue, teal, orange = "#2563eb", "#0d9488", "#ea580c"
    axes[0, 0].plot(epochs, values("metrics/mAP50(B)"), label="mAP50", color=blue, lw=2)
    axes[0, 0].plot(epochs, values("metrics/mAP50-95(B)"), label="mAP50-95", color=teal, lw=2)
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0, 0].set(title="Validation detection scores", ylabel="Average precision")
    best_value = float(next(row for row in rows if int(row["epoch"]) == best_epoch)["metrics/mAP50-95(B)"])
    best_percent = (Decimal(str(best_value)) * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    axes[0, 0].scatter([best_epoch], [best_value], marker="*", s=130, color=orange, zorder=5,
                       label=f"Best: {best_percent}% (epoch {best_epoch})")
    axes[0, 1].plot(epochs, values("metrics/precision(B)"), label="Precision", color=blue, lw=2)
    axes[0, 1].plot(epochs, values("metrics/recall(B)"), label="Recall", color=teal, lw=2)
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0, 1].set(title="Validation precision and recall", ylabel="Score")
    for ax, suffix, title in ((axes[1, 0], "box_loss", "Bounding-box loss"),
                              (axes[1, 1], "cls_loss", "Classification loss (log scale)")):
        ax.plot(epochs, values(f"train/{suffix}"), label="Train", color=blue, lw=2)
        ax.plot(epochs, values(f"val/{suffix}"), label="Validation", color=teal, lw=2)
        ax.set(title=title, ylabel="Loss (lower is better)")
    axes[1, 1].set_yscale("log")
    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.18)
        ax.legend(fontsize=8, loc="best", frameon=False)
    fig.savefig(destination, dpi=160, facecolor="white")
    plt.close(fig)


def demo_figures(results, selected, names, output, conf):
    from PIL import Image, ImageDraw, ImageOps
    navy, muted, accent = "#0b1220", "#a8b6cb", "#5eead4"
    panel_width, panel_height, gap = 600, 680, 20
    gallery = Image.new("RGB", (1260, 2260), navy)
    draw = ImageDraw.Draw(gallery)
    draw.text((20, 18), "PCB DEFECT DETECTION", font=font(34, True), fill="white")
    draw.text((20, 64), f"YOLO26s | validation examples | confidence >= {conf:.2f}", font=font(20), fill=muted)
    records, rendered = [], []
    for i, (result, (source, gt_count)) in enumerate(zip(results, selected)):
        annotated = Image.fromarray(result.plot(line_width=2, font_size=18)[..., ::-1])
        rendered.append(annotated)
        x, y = 20 + (i % 2) * (panel_width + gap), 112 + (i // 2) * (panel_height + gap)
        draw.text((x, y), f"0{i+1} / {names[i].upper().replace('_', ' ')}", font=font(23, True), fill=accent)
        gallery.paste(ImageOps.fit(annotated, (600, 600)), (x, y + 40))
        draw.text((x, y + 650), f"Predicted boxes: {len(result.boxes)}  |  Labeled boxes: {gt_count}",
                  font=font(17), fill=muted)
        boxes = result.boxes
        records.append({
            "class_example": names[i], "source": source.relative_to(ROOT).as_posix(),
            "ground_truth_box_count": gt_count,
            "predictions": [{"class": names[int(cls)], "confidence": float(score), "xyxy": xy}
                            for cls, score, xy in zip(boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist(),
                                                      boxes.xyxy.cpu().tolist())],
        })
    draw.text((20, 2220), "Images: TCC / Roboflow (CC BY 4.0) | Boxes and scores are model predictions",
              font=font(16), fill=muted)
    gallery.save(output / "demo_predictions.jpg", quality=92, subsampling=0)

    comparison = Image.new("RGB", (1260, 744), navy)
    draw = ImageDraw.Draw(comparison)
    draw.text((20, 15), "INPUT IMAGE", font=font(25, True), fill="white")
    draw.text((640, 15), "YOLO26s PREDICTION", font=font(25, True), fill=accent)
    comparison.paste(ImageOps.fit(Image.open(selected[0][0]).convert("RGB"), (600, 600)), (20, 65))
    comparison.paste(ImageOps.fit(rendered[0], (600, 600)), (640, 65))
    draw.text((20, 680), f"Validation example | best checkpoint | confidence >= {conf:.2f}", font=font(20), fill=muted)
    draw.text((20, 714), "Images: TCC / Roboflow (CC BY 4.0) | Actual model output", font=font(15), fill=muted)
    comparison.save(output / "demo_comparison.jpg", quality=92, subsampling=0)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/detect/pcb_yolo26s")
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    from ultralytics import YOLO
    import yaml

    run = args.run.resolve()
    weights = run / "weights/best.pt"
    if not weights.is_file():
        raise SystemExit(f"Missing trained checkpoint: {weights}. Train first.")
    with (run / "results.csv").open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    assets, snapshots = ROOT / "docs/assets", ROOT / "docs/results"
    assets.mkdir(parents=True, exist_ok=True)
    snapshots.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    names = [model.names[i] for i in range(len(model.names))]
    selected = validation_examples(names)
    results = model.predict(source=[str(image) for image, _ in selected], imgsz=640,
                            conf=args.conf, device=args.device, half=args.device != "cpu", save=False, verbose=False)
    records = demo_figures(results, selected, names, assets, args.conf)
    if selected[0][0].suffix.lower() in {".jpg", ".jpeg"}:
        shutil.copyfile(selected[0][0], assets / "demo_input.jpg")
    else:
        from PIL import Image
        Image.open(selected[0][0]).convert("RGB").save(assets / "demo_input.jpg", quality=95)
    training_figure(rows, int(best["epoch"]), assets / "training_curves.png")
    shutil.copy2(run / "results.csv", snapshots / "training_results.csv")
    training_args = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
    completed, planned = int(rows[-1]["epoch"]), int(training_args["epochs"])
    test_evaluated = (run / "test_metrics.json").is_file()
    status = "finished" if test_evaluated or completed >= planned else "snapshot"
    state_path = ROOT / ".runtime/training-process.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8-sig"))
        checkpoint_path = Path(state.get("checkpoint", ""))
        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT / checkpoint_path
        if (state.get("status") == "paused" and state.get("last_completed_epoch") == completed
                and checkpoint_path.resolve() == (run / "weights/last.pt").resolve()):
            status = "paused"
    checkpoint_epoch = model.ckpt.get("epoch", -1)
    summary = {
        "status": status, "completed_epochs": completed,
        "planned_epochs": planned, "early_stopping_patience": training_args["patience"],
        "metrics_split": "validation", "test_evaluated": test_evaluated,
        "best_epoch": int(best["epoch"]),
        "best_checkpoint_epoch": checkpoint_epoch + 1 if checkpoint_epoch >= 0 else None,
        "best_metrics": {key: float(value) for key, value in best.items() if key.startswith("metrics/")},
        "latest_metrics": {key: float(value) for key, value in rows[-1].items() if key.startswith("metrics/")},
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "demo": {"split": "validation", "confidence_threshold": args.conf, "image_size": 640,
                 "selection": "First distinct image in sorted validation labels for each class; no confidence-based selection",
                 "examples": records},
        "dataset_source": "https://universe.roboflow.com/tcc-r4j2r/pcb-la0tj/dataset/2",
        "image_license": "CC BY 4.0; TCC / Roboflow; prediction overlays and layout added",
    }
    (snapshots / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": summary["best_epoch"], "metrics": summary["best_metrics"],
                      "demo_box_counts": [len(record["predictions"]) for record in records]}, indent=2))
    print(f"README assets saved to {assets}")


if __name__ == "__main__":
    main()
