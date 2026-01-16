#!/usr/bin/env python3
import argparse
import math
import statistics
import time
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VideoFolderDataset
from eval import run_eval
from model import build_model
from utils import (
    append_jsonl,
    ensure_dir,
    load_config,
    save_checkpoint,
    save_json,
    seed_everything,
)


def make_loader(split_root: str, cfg: Dict, shuffle: bool) -> DataLoader:
    ds = VideoFolderDataset(
        split_root=split_root,
        num_frames=int(cfg["num_frames"]),
        input_size=int(cfg["input_size"]),
    )
    bs = int(cfg["train_batch_size"] if shuffle else cfg["val_batch_size"])
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=shuffle,
    )


def maybe_freeze(model, cfg: Dict) -> None:
    freeze_backbone = bool(cfg.get("freeze_backbone", False))
    if not freeze_backbone:
        return

    # For HF VideoMAE, backbone usually is model.videomae
    if hasattr(model, "videomae"):
        for p in model.videomae.parameters():
            p.requires_grad = False

    # Always keep classifier trainable
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True


def compute_class_weights(samples, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for s in samples:
        counts[int(s.label)] += 1.0
    total = counts.sum()
    # Inverse-frequency weights, normalized by number of classes.
    weights = total / (counts * max(1, num_classes))
    weights = torch.where(counts > 0, weights, torch.zeros_like(weights))
    return weights


def _sec_to_str(seconds: float) -> str:
    """Small helper to format seconds into a readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.1f}s"


def benchmark_steps(model, loader, optimizer, scaler, device, steps: int, amp: bool, loss_fn):
    if steps <= 0:
        raise ValueError("bench_steps must be positive")

    device_is_cuda = device.startswith("cuda")
    if device_is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    data_iter = iter(loader)

    # Warmup: ~10% of steps, clamped to [2, 10], but never consume all steps
    proposed_warmup = max(2, min(10, int(math.ceil(steps * 0.1))))
    warmup_steps = min(proposed_warmup, steps - 1) if steps > 1 else 0

    step_times = []
    load_times = []
    h2d_times = []
    compute_times = []
    batch_sizes = []

    load_start = time.perf_counter()

    for step_idx in range(steps):
        try:
            clips, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            load_start = time.perf_counter()
            clips, labels = next(data_iter)

        load_end = time.perf_counter()
        load_sec = load_end - load_start

        if device_is_cuda:
            torch.cuda.synchronize()
        h2d_start = time.perf_counter()
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if device_is_cuda:
            torch.cuda.synchronize()
        h2d_end = time.perf_counter()
        h2d_sec = h2d_end - h2d_start

        optimizer.zero_grad(set_to_none=True)
        if device_is_cuda:
            torch.cuda.synchronize()
        compute_start = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=amp):
            out = model(clips)
            logits = out.logits if hasattr(out, "logits") else out
            loss = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if device_is_cuda:
            torch.cuda.synchronize()
        compute_end = time.perf_counter()
        compute_sec = compute_end - compute_start

        step_sec = load_sec + h2d_sec + compute_sec

        if step_idx >= warmup_steps:
            step_times.append(step_sec)
            load_times.append(load_sec)
            h2d_times.append(h2d_sec)
            compute_times.append(compute_sec)
            batch_sizes.append(int(labels.size(0)))

        load_start = time.perf_counter()

    if not step_times:
        raise RuntimeError("No benchmark steps recorded after warmup. Try increasing bench_steps.")

    total_time = sum(step_times)
    step_count = len(step_times)
    avg_step = total_time / step_count
    metrics = {
        "bench_steps": steps,
        "warmup_steps": warmup_steps,
        "avg_step_sec": avg_step,
        "median_step_sec": statistics.median(step_times),
        "avg_load_sec": sum(load_times) / step_count,
        "avg_h2d_sec": sum(h2d_times) / step_count,
        "avg_compute_sec": sum(compute_times) / step_count,
        "steps_per_sec": step_count / total_time,
    }

    avg_batch = sum(batch_sizes) / step_count
    metrics["avg_batch_size"] = avg_batch
    metrics["clips_per_sec"] = avg_batch * metrics["steps_per_sec"]

    if device_is_cuda:
        metrics["peak_cuda_mem_alloc_bytes"] = torch.cuda.max_memory_allocated(device)
        metrics["peak_cuda_mem_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    else:
        metrics["peak_cuda_mem_alloc_bytes"] = None
        metrics["peak_cuda_mem_reserved_bytes"] = None

    return metrics


def print_benchmark_summary(stats: Dict) -> None:
    print("=== Benchmark Summary ===")
    print(
        f"Steps: {stats['bench_steps']} (warmup {stats.get('warmup_steps', 0)}) | "
        f"Avg step: {stats['avg_step_sec']:.4f}s | "
        f"Median step: {stats['median_step_sec']:.4f}s"
    )
    print(
        "Breakdown (avg): "
        f"load {stats['avg_load_sec']:.4f}s | "
        f"h2d {stats['avg_h2d_sec']:.4f}s | "
        f"compute {stats['avg_compute_sec']:.4f}s"
    )
    print(
        f"Throughput: {stats['steps_per_sec']:.2f} steps/s | "
        f"{stats['clips_per_sec']:.2f} clips/s (avg batch {stats['avg_batch_size']:.2f})"
    )
    alloc = stats.get("peak_cuda_mem_alloc_bytes")
    reserved = stats.get("peak_cuda_mem_reserved_bytes")
    if alloc is not None and reserved is not None:
        print(
            f"Peak CUDA memory: alloc {alloc / (1024 ** 2):.1f} MiB | "
            f"reserved {reserved / (1024 ** 2):.1f} MiB"
        )
    else:
        print("Peak CUDA memory: N/A (CPU)")

    est_epoch = stats.get("est_epoch_sec")
    est_total = stats.get("est_total_sec")
    steps_per_epoch = stats.get("steps_per_epoch")
    total_steps = stats.get("total_steps")
    epochs = stats.get("epochs")
    if est_epoch is not None and est_total is not None:
        print(
            f"Estimated epoch ({steps_per_epoch} steps): {_sec_to_str(est_epoch)} | "
            f"Estimated total ({total_steps} steps over {epochs} epochs): {_sec_to_str(est_total)}"
        )
    print("=========================")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to base config yaml")
    ap.add_argument("--override", default=None, help="Optional override yaml (stageA, stageB, etc)")
    ap.add_argument("--test_after", action="store_true", help="Run test set after training")
    ap.add_argument("--bench_steps", type=int, default=100, help="Number of steps to run benchmark over")
    ap.add_argument(
        "--bench_only",
        action="store_true",
        help="Run benchmark (bench_steps) to estimate throughput then exit before full training",
    )
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    seed_everything(int(cfg.get("seed", 1337)))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    outputs_root = Path(cfg["outputs_root"]) / cfg["exp_name"]
    ckpt_dir = outputs_root / "checkpoints"
    log_dir = outputs_root / "logs"
    res_dir = outputs_root / "results"
    ensure_dir(ckpt_dir)
    ensure_dir(log_dir)
    ensure_dir(res_dir)

    # Snapshot config
    save_json(ckpt_dir / "config_snapshot.json", cfg)

    train_root = str(Path(cfg["data_root"]) / "train")
    val_root = str(Path(cfg["data_root"]) / "val")
    test_root = str(Path(cfg["data_root"]) / "test")

    train_loader = make_loader(train_root, cfg, shuffle=True)
    val_loader = make_loader(val_root, cfg, shuffle=False)

    model, _ = build_model(cfg)
    maybe_freeze(model, cfg)
    model.to(device)

    # Optimizer over trainable params only
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))

    # Cosine schedule with warmup
    epochs = int(cfg["epochs"])
    warmup_epochs = int(cfg.get("warmup_epochs", 0))
    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)

    def lr_at(step: int) -> float:
        base = float(cfg["lr"])
        if warmup_steps > 0 and step < warmup_steps:
            return base * (step + 1) / warmup_steps
        # cosine
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return base * 0.5 * (1.0 + math.cos(math.pi * t))

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("amp", True)))
    if bool(cfg.get("use_class_weights", False)):
        weights = compute_class_weights(train_loader.dataset.samples, int(cfg["num_classes"]))
        print(f"[INFO] class counts: {weights.numel()} classes")
        print(f"[INFO] class weights: {weights.tolist()}")
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    if args.bench_only:
        bench_stats = benchmark_steps(
            model=model,
            loader=train_loader,
            optimizer=opt,
            scaler=scaler,
            device=device,
            steps=int(args.bench_steps),
            amp=bool(cfg.get("amp", True)),
            loss_fn=loss_fn,
        )
        steps_per_epoch = len(train_loader)
        total_steps = epochs * steps_per_epoch
        bench_stats.update(
            {
                "steps_per_epoch": steps_per_epoch,
                "total_steps": total_steps,
                "epochs": epochs,
                "est_epoch_sec": bench_stats["avg_step_sec"] * steps_per_epoch,
                "est_total_sec": bench_stats["avg_step_sec"] * total_steps,
            }
        )
        bench_path = log_dir / f"bench_{args.bench_steps}steps.json"
        save_json(bench_path, bench_stats)
        print_benchmark_summary(bench_stats)
        return

    best_val = -1.0
    best_epoch = -1
    bad_epochs = 0
    patience = int(cfg.get("early_stop_patience", 5))

    global_step = 0
    train_log_path = log_dir / "train.log"
    val_metrics_path = res_dir / "val_metrics.jsonl"

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}/{epochs}")

        running_loss = 0.0
        n = 0

        for clips, labels in pbar:
            clips = clips.to(device, non_blocking=True)   # [B,T,C,H,W]
            labels = labels.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=bool(cfg.get("amp", True))):
                out = model(clips)
                logits = out.logits if hasattr(out, "logits") else out
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            # LR update
            global_step += 1
            lr = lr_at(global_step)
            for pg in opt.param_groups:
                pg["lr"] = lr

            running_loss += float(loss.item()) * labels.size(0)
            n += labels.size(0)

            if global_step % int(cfg.get("log_every", 20)) == 0:
                line = {
                    "epoch": epoch,
                    "step": global_step,
                    "lr": lr,
                    "train_loss": running_loss / max(1, n),
                }
                append_jsonl(train_log_path, line)

            pbar.set_postfix(loss=running_loss / max(1, n), lr=lr)

        # Eval
        if epoch % int(cfg.get("eval_every", 1)) == 0:
            val_out = run_eval(model, val_loader, device)
            val_out["epoch"] = epoch
            append_jsonl(val_metrics_path, val_out)

            # Track best by F1 (you can switch to acc if you prefer)
            score = float(val_out["f1"])
            if score > best_val:
                best_val = score
                best_epoch = epoch
                bad_epochs = 0
                save_checkpoint(ckpt_dir / "best.pt", model, opt, scaler, epoch, best_val, cfg)
            else:
                bad_epochs += 1

        # Always save last
        if epoch % int(cfg.get("save_every", 1)) == 0:
            save_checkpoint(ckpt_dir / "last.pt", model, opt, scaler, epoch, best_val, cfg)

        if bad_epochs >= patience:
            break

    # Save summary
    save_json(res_dir / "best_val_summary.json", {"best_f1": best_val, "best_epoch": best_epoch})

    # Optional test
    if args.test_after and (Path(test_root).exists()):
        test_loader = make_loader(test_root, cfg, shuffle=False)
        test_out = run_eval(model, test_loader, device)
        save_json(res_dir / "test_metrics.json", test_out)


if __name__ == "__main__":
    main()
