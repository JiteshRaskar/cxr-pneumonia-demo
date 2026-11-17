"""Training (source domain only) and zero-shot evaluation (paper Sec. IV-F/G)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from .data import load_manifest, make_loader
from .metrics import bootstrap_ci, metrics_at, threshold_from_val
from .models import build_model
from .utils import get_device, load_json, save_json, seed_everything


def run_name(source, arch, norm, seed):
    return f"{source}_{arch}_{norm}_seed{seed}"


def _epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False):
    train = optimizer is not None
    model.train(train)
    tot, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False, desc="train" if train else "val"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = model(x)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            tot += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return tot / n, correct / n


@torch.no_grad()
def predict_probs(model, loader, device, amp=False):
    model.eval()
    ps = []
    for x, _ in tqdm(loader, leave=False, desc="predict"):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(x.to(device, non_blocking=True))
        ps.append(torch.softmax(out.float(), 1)[:, 1].cpu().numpy())
    return np.concatenate(ps)


def train_run(source, arch, norm, seed, out_root, manifest_dir, epochs=25, batch_size=32,
              lr=1e-4, patience=5, num_workers=4, pretrained=True, size=224, amp=True):
    seed_everything(seed)
    device = get_device()
    amp = amp and device.type == "cuda"
    run_dir = Path(out_root) / run_name(source, arch, norm, seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    tr = load_manifest(source, manifest_dir, "train")
    va = load_manifest(source, manifest_dir, "val")
    counts = np.bincount(tr["label"], minlength=2)
    weights = counts.sum() / (2.0 * np.maximum(counts, 1))  # class-weighted CE for imbalance
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    tl = make_loader(tr, True, norm, batch_size, num_workers, size)
    vl = make_loader(va, False, norm, batch_size, num_workers, size)

    model = build_model(arch, pretrained).to(device)  # all layers fine-tuned, same policy for every arch
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best, bad, hist = float("inf"), 0, []
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = _epoch(model, tl, criterion, device, opt, scaler, amp)
        va_loss, va_acc = _epoch(model, vl, criterion, device, amp=amp)
        hist.append(dict(epoch=ep, train_loss=tr_loss, train_acc=tr_acc, val_loss=va_loss, val_acc=va_acc))
        print(f"[{run_dir.name}] ep{ep:02d} tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} "
              f"va_loss={va_loss:.4f} va_acc={va_acc:.4f}")
        if va_loss < best - 1e-4:  # early stopping / model selection: SOURCE val loss only
            best, bad = va_loss, 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            bad += 1
            if bad >= patience:
                print("early stopping")
                break

    pd.DataFrame(hist).to_csv(run_dir / "history.csv", index=False)
    _plot_history(pd.DataFrame(hist), run_dir / "curves.png", run_dir.name)

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    p = predict_probs(model, vl, device, amp)
    thr = threshold_from_val(va["label"].values, p)  # fixed from here on
    save_json(run_dir / "threshold.json", {"threshold": thr})
    save_json(run_dir / "config.json", dict(source=source, arch=arch, norm=norm, seed=seed, size=size,
                                            pretrained=pretrained, epochs_run=len(hist), lr=lr,
                                            batch_size=batch_size, class_weights=weights.tolist()))
    return run_dir


def _plot_history(h, path, title):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(h.epoch, h.train_loss, label="train"); ax[0].plot(h.epoch, h.val_loss, label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(h.epoch, h.train_acc, label="train"); ax[1].plot(h.epoch, h.val_acc, label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def load_run(run_dir, device=None):
    """Returns (model, config, threshold)."""
    run_dir = Path(run_dir)
    device = device or get_device()
    cfg = load_json(run_dir / "config.json")
    model = build_model(cfg["arch"], pretrained=False)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    model.to(device).eval()
    return model, cfg, load_json(run_dir / "threshold.json")["threshold"]


def evaluate_run(run_dir, manifest_dir, targets, external_split="all", batch_size=64,
                 num_workers=4, n_boot=300, amp=True):
    """Zero-shot: weights + threshold fixed from source validation. No target tuning."""
    device = get_device()
    amp = amp and device.type == "cuda"
    run_dir = Path(run_dir)
    model, cfg, thr = load_run(run_dir, device)
    rows = []
    for tgt in targets:
        split = "test" if tgt == cfg["source"] else external_split
        df = load_manifest(tgt, manifest_dir, split)
        loader = make_loader(df, False, cfg["norm"], batch_size, num_workers, cfg["size"])
        p = predict_probs(model, loader, device, amp)
        y = df["label"].values
        m = metrics_at(y, p, thr)
        ci = bootstrap_ci(y, p, df["patient_id"].values, thr, n_boot=n_boot)
        row = dict(source=cfg["source"], target=tgt, arch=cfg["arch"], norm=cfg["norm"], seed=cfg["seed"],
                   eval_split=split, n=len(df), n_pos=int(y.sum()), n_patients=int(df["patient_id"].nunique()),
                   threshold=thr, **m)
        for k, (lo, hi) in ci.items():
            row[f"{k}_lo"], row[f"{k}_hi"] = lo, hi
        rows.append(row)
        out = df[["path", "label", "patient_id"]].copy()
        out["prob"] = p
        out.to_csv(run_dir / f"preds_{tgt}.csv", index=False)
        print(f"[{run_dir.name}] {cfg['source']}->{tgt} ({split}) auc={m['roc_auc']:.3f} f1={m['f1']:.3f}")
    save_json(run_dir / "metrics.json", rows)
    return rows
