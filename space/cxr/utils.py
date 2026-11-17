import json
import random
from pathlib import Path

import numpy as np
import torch

DATASETS = {
    "A": "Chest X-Ray Images (Pneumonia) [pediatric]",
    "B": "NIH ChestX-ray14",
    "C": "CheXpert",
}
ARCHS = ["resnet50", "densenet121", "efficientnet_b0", "mobilenet_v2"]
METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)
