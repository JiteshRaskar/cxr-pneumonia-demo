"""Grad-CAM (paper Sec. IV-J) -- exploratory interpretability only."""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from .data import build_transform, display_image
from .engine import load_run
from .models import gradcam_layer
from .utils import get_device


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(self._fwd)

    def _fwd(self, module, inp, out):
        self.acts = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x, target=1):
        """x: (1,3,H,W). target: class index (1 = pneumonia). Returns cam in [0,1], probs."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        logits[0, target].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, torch.softmax(logits.detach(), 1)[0].cpu().numpy()


def overlay(rgb, cam, alpha=0.4):
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    return (alpha * heat + (1 - alpha) * rgb).astype(np.uint8)


def explain_image(model, cfg, pil_img, device, target=1):
    x = build_transform(False, cfg["norm"], cfg["size"])(pil_img).unsqueeze(0).to(device)
    cam, probs = GradCAM(model, gradcam_layer(model, cfg["arch"]))(x, target)
    rgb = display_image(pil_img, cfg["norm"], cfg["size"])
    return overlay(rgb, cam), rgb, probs


def gradcam_gallery(run_dir, target, out_path, k=3):
    """Grid of the k most-confident TP / TN / FP / FN examples on `target`."""
    device = get_device()
    model, cfg, thr = load_run(run_dir, device)
    d = pd.read_csv(Path(run_dir) / f"preds_{target}.csv")
    d["pred"] = (d["prob"] >= thr).astype(int)
    d["conf"] = (d["prob"] - thr).abs()
    cats = {"TP": (1, 1), "TN": (0, 0), "FP": (0, 1), "FN": (1, 0)}  # (label, pred)
    fig, axes = plt.subplots(4, k, figsize=(3 * k, 12))
    for r, (name, (lab, pr)) in enumerate(cats.items()):
        sub = d[(d.label == lab) & (d.pred == pr)].sort_values("conf", ascending=False).head(k)
        for c in range(k):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(sub):
                row = sub.iloc[c]
                img = Image.open(row["path"]); img.load()
                ov, _, _ = explain_image(model, cfg, img, device, target=1)
                ax.imshow(ov)
                ax.set_title(f"{name}  p(pneu)={row['prob']:.2f}", fontsize=9)
    fig.suptitle(f"Grad-CAM (pneumonia class) | {Path(run_dir).name} -> {target}")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
