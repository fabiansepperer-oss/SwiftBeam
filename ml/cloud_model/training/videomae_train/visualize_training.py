#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "matplotlib is required for visualization. Install it and retry."
    ) from exc


def read_jsonl(path: Path) -> List[Dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_experiment(exp_dir: Path) -> Dict:
    logs_dir = exp_dir / "logs"
    results_dir = exp_dir / "results"
    train_log = read_jsonl(logs_dir / "train.log")
    val_log = read_jsonl(results_dir / "val_metrics.jsonl")
    return {
        "name": exp_dir.name,
        "dir": exp_dir,
        "train_log": train_log,
        "val_log": val_log,
    }


def series_from_train(log: List[Dict]) -> Tuple[List[int], List[float]]:
    steps = []
    losses = []
    for row in log:
        if "step" in row and "train_loss" in row:
            steps.append(int(row["step"]))
            losses.append(float(row["train_loss"]))
    return steps, losses


def series_from_val(log: List[Dict]) -> Tuple[List[int], List[float], List[float], List[float]]:
    epochs = []
    losses = []
    f1s = []
    accs = []
    for row in log:
        if "epoch" in row:
            epochs.append(int(row["epoch"]))
            losses.append(float(row.get("loss", 0.0)))
            f1s.append(float(row.get("f1", 0.0)))
            accs.append(float(row.get("acc", 0.0)))
    return epochs, losses, f1s, accs


def last_confusion_matrix(log: List[Dict]) -> Optional[np.ndarray]:
    if not log:
        return None
    for row in reversed(log):
        if "confusion_matrix" in row:
            return np.asarray(row["confusion_matrix"])
    return None


def plot_confusion(ax, cm: Optional[np.ndarray]) -> None:
    ax.set_title("Confusion Matrix (last val)")
    if cm is None or cm.size == 0:
        ax.text(0.5, 0.5, "no confusion matrix", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(int(v)), ha="center", va="center", fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_experiment(exp: Dict, out_dir: Path) -> Path:
    train_steps, train_loss = series_from_train(exp["train_log"])
    val_epochs, val_loss, val_f1, val_acc = series_from_val(exp["val_log"])
    cm = last_confusion_matrix(exp["val_log"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.set_title("Train Loss vs Step")
    if train_steps:
        ax.plot(train_steps, train_loss, color="#1f77b4")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
    else:
        ax.text(0.5, 0.5, "no train.log", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])

    ax = axes[0, 1]
    ax.set_title("Val Loss vs Epoch")
    if val_epochs:
        ax.plot(val_epochs, val_loss, color="#ff7f0e")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
    else:
        ax.text(0.5, 0.5, "no val_metrics.jsonl", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])

    ax = axes[1, 0]
    ax.set_title("Val F1 vs Epoch")
    if val_epochs:
        ax.plot(val_epochs, val_f1, color="#2ca02c")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1")
    else:
        ax.text(0.5, 0.5, "no val_metrics.jsonl", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])

    plot_confusion(axes[1, 1], cm)

    fig.suptitle(exp["name"])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = out_dir / f"{exp['name']}_summary.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def combine_pipeline(stage_a: Dict, stage_b: Dict) -> Dict:
    a_steps, a_loss = series_from_train(stage_a["train_log"])
    b_steps, b_loss = series_from_train(stage_b["train_log"])
    step_offset = a_steps[-1] if a_steps else 0
    b_steps = [s + step_offset for s in b_steps]

    a_epochs, a_vloss, a_vf1, a_vacc = series_from_val(stage_a["val_log"])
    b_epochs, b_vloss, b_vf1, b_vacc = series_from_val(stage_b["val_log"])
    epoch_offset = a_epochs[-1] if a_epochs else 0
    b_epochs = [e + epoch_offset for e in b_epochs]

    combined_train = [
        {"step": s, "train_loss": l} for s, l in zip(a_steps + b_steps, a_loss + b_loss)
    ]
    combined_val = []
    for e, l, f1, acc in zip(a_epochs, a_vloss, a_vf1, a_vacc):
        combined_val.append({"epoch": e, "loss": l, "f1": f1, "acc": acc})
    for e, l, f1, acc in zip(b_epochs, b_vloss, b_vf1, b_vacc):
        combined_val.append({"epoch": e, "loss": l, "f1": f1, "acc": acc})

    return {
        "name": f"{stage_a['name']}_to_{stage_b['name']}",
        "dir": stage_b["dir"],
        "train_log": combined_train,
        "val_log": combined_val,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize VideoMAE training logs")
    ap.add_argument("--base", default="outputs/exp_base_T16_192", help="Base experiment dir")
    ap.add_argument("--stagea", default="outputs/exp_stageA_head_only", help="Stage A experiment dir")
    ap.add_argument("--stageb", default="outputs/exp_stageB_unfreeze_last2", help="Stage B experiment dir")
    ap.add_argument("--out_dir", default="outputs/plots", help="Where to save PNGs")
    ap.add_argument("--skip_base", action="store_true", help="Skip base plots")
    ap.add_argument("--skip_stagea", action="store_true", help="Skip Stage A plots")
    ap.add_argument("--skip_stageb", action="store_true", help="Skip Stage B plots")
    ap.add_argument("--pipeline_only", action="store_true", help="Only plot Stage A -> Stage B pipeline")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "base": Path(args.base),
        "stagea": Path(args.stagea),
        "stageb": Path(args.stageb),
    }

    exps = {}
    for key, p in paths.items():
        if p.exists():
            exps[key] = load_experiment(p)

    saved = []

    if args.pipeline_only:
        if "stagea" in exps and "stageb" in exps:
            pipeline = combine_pipeline(exps["stagea"], exps["stageb"])
            saved.append(plot_experiment(pipeline, out_dir))
        else:
            print("[WARN] Stage A or Stage B logs not found; pipeline plot skipped.")
    else:
        if not args.skip_base and "base" in exps:
            saved.append(plot_experiment(exps["base"], out_dir))
        if not args.skip_stagea and "stagea" in exps:
            saved.append(plot_experiment(exps["stagea"], out_dir))
        if not args.skip_stageb and "stageb" in exps:
            saved.append(plot_experiment(exps["stageb"], out_dir))
        if "stagea" in exps and "stageb" in exps:
            pipeline = combine_pipeline(exps["stagea"], exps["stageb"])
            saved.append(plot_experiment(pipeline, out_dir))

    if saved:
        print("Saved plots:")
        for p in saved:
            print(f"  {p}")
    else:
        print("No plots saved (missing experiment directories or logs).")


if __name__ == "__main__":
    main()
