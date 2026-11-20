
# %% Imports library
import argparse
import contextlib
import copy
import gc
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

# %% database
os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parent / "torch_cache")
                      if "__file__" in globals() else "./torch_cache")

import numpy as np
import torch
import torch.backends.cudnn  
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

# %% Configuration
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_CFG = {
    "seed": 42,
    "data_dir": "./data",
    "out_dir": "./outputs",
    # data splits
    "train_per_class": 500,
    "val_per_class": 100,
    "test_total": 2000,          
    # optimisation (Run A settings, shared by ResNet-18, B and C)
    "batch_size": 32,            
    "fallback_batch_size": 16,   
    "epochs": 12,
    "warmup_epochs": 1,
    "lr": 3e-4,                  # used if the LR sweep is skipped
    "lr_grid": [1e-4, 3e-4, 1e-3],
    "lr_sweep_epochs": 3,        # shortened proxy runs; set to 12 for a full-length sweep
    "run_lr_sweep": True,
    "weight_decay": 0.05,
    "label_smoothing": 0.1,
    "dropout": 0.2,
    "fixed_size": 224,
    # progressive learning (Run B / C): 4 stages x 3 epochs
    "stages": [
        {"size": 128, "randaug": 5,  "dropout": 0.10},
        {"size": 160, "randaug": 8,  "dropout": 0.17},
        {"size": 192, "randaug": 12, "dropout": 0.23},
        {"size": 224, "randaug": 15, "dropout": 0.30},
    ],
    "mixup_alphas": [0.0, 0.05, 0.10, 0.20],   # Run C only
    "svm_C_grid": [1, 10],
    "num_workers": 0,            
    "smoke_test": False,
    "force": False,              
}

SMOKE_OVERRIDES = {
    "out_dir": "./outputs_smoke",
    "train_per_class": 20,
    "val_per_class": 10,
    "test_total": 200,
    "epochs": 1,                 
    "lr_sweep_epochs": 1,
    "batch_size": 32,
    "fallback_batch_size": 16,
}

ALL_STEPS = ["hog_svm", "lr_sweep", "resnet18", "effnet_fixed", "effnet_prog_size_only",
             "effnet_progressive", "effnet_prog_mixup", "test"]

# %% Utilities: seeding, device, AMP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    """Wait for queued GPU work so wall-clock timings are honest."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def empty_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def make_amp(device: torch.device):
    """Mixed precision only on CUDA. `torch.autocast(device_type="cuda")` is the current,
    non-deprecated spelling of `torch.cuda.amp.autocast`; behaviour is identical.
    On MPS/CPU training runs in plain fp32."""
    use_amp = device.type == "cuda"
    scaler = None
    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):      # older PyTorch
            scaler = torch.cuda.amp.GradScaler()

    def autocast():
        if use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    return autocast, scaler


def is_oom(err: BaseException) -> bool:
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", ())
    return isinstance(err, oom_cls) or "out of memory" in str(err).lower()


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def environment_info(device: torch.device) -> dict:
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "device": device.type,
        "amp": device.type == "cuda",
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_mem_gb"] = round(props.total_memory / 1024 ** 3, 2)
    return info

# %% Data: splits and datasets


class SubsetWithTransform(Dataset):
    """Index subset of a PIL-returning dataset with a swappable transform
    (needed so progressive learning can change resolution/augmentation per stage)."""

    def __init__(self, base, indices, transform=None):
        self.base = base
        self.indices = list(map(int, indices))
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def make_splits(train_targets, test_targets, cfg) -> dict:
    rng = np.random.default_rng(cfg["seed"])
    train_targets = np.asarray(train_targets)
    test_targets = np.asarray(test_targets)
    n_classes = int(train_targets.max()) + 1
    test_per_class = cfg["test_total"] // n_classes

    train_idx, val_idx, test_idx = [], [], []
    for c in range(n_classes):
        idx = rng.permutation(np.where(train_targets == c)[0])
        train_idx += idx[: cfg["train_per_class"]].tolist()
        val_idx += idx[cfg["train_per_class"]: cfg["train_per_class"] + cfg["val_per_class"]].tolist()
        tidx = rng.permutation(np.where(test_targets == c)[0])
        test_idx += tidx[:test_per_class].tolist()

    assert not set(train_idx) & set(val_idx), "train/val overlap"
    return {"train": sorted(train_idx), "val": sorted(val_idx), "test": sorted(test_idx)}


def find_cifar_root(data_dir) -> Path:
    """Return the folder that contains 'cifar-10-batches-py', accepting either that folder's
    parent (torchvision layout) or the 'cifar-10-batches-py' folder itself."""
    p = Path(data_dir).expanduser().resolve()
    if p.name == "cifar-10-batches-py":
        return p.parent
    return p


def load_data(cfg) -> dict:
    root = find_cifar_root(cfg["data_dir"])
    have_local = (root / "cifar-10-batches-py" / "data_batch_1").exists()
    if have_local:
        print(f"Using existing CIFAR-10 at {root / 'cifar-10-batches-py'} (no download)")
    else:
        print(f"CIFAR-10 not found in {root}; downloading once (~170 MB)...")
    # download=False when files exist: no network access; torchvision still verifies checksums
    base_train = datasets.CIFAR10(str(root), train=True, download=not have_local, transform=None)
    base_test = datasets.CIFAR10(str(root), train=False, download=not have_local, transform=None)
    splits = make_splits(base_train.targets, base_test.targets, cfg)
    save_json(splits, Path(cfg["out_dir"]) / "splits.json")
    print(f"Splits: train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")
    return {"base_train": base_train, "base_test": base_test, "splits": splits,
            "class_names": list(base_train.classes)}


def build_transform(size: int, train: bool, randaug_mag=None):
    t = []
    if train:
        # light, standard CIFAR augmentation applied to every DL run (A, B, C, ResNet-18)
        t += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    t.append(transforms.Resize((size, size), antialias=True))
    if train and randaug_mag:
        t.append(transforms.RandAugment(num_ops=2, magnitude=int(randaug_mag)))
    t += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(t)


def make_loader(ds, batch_size, shuffle, device, cfg, generator=None, drop_last=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                      num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"),
                      generator=generator)

# %% Models


def build_model(arch: str, num_classes: int = 10, dropout: float = 0.2, pretrained: bool = True):
    if arch == "efficientnet_v2_s":
        w = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_v2_s(weights=w)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True), nn.Linear(in_f, num_classes))
    elif arch == "resnet18":
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=w)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_f, num_classes))
    else:
        raise ValueError(f"unknown arch {arch}")
    return m


def set_head_dropout(model, p: float) -> None:
    head = model.classifier if hasattr(model, "classifier") else model.fc
    for mod in head.modules():
        if isinstance(mod, nn.Dropout):
            mod.p = p


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())

# %% Schedules (epoch plans + LR)


def fixed_schedule(epochs, size, dropout):
    return [{"stage": 1, "size": size, "randaug": None, "dropout": dropout, "mixup": 0.0}
            for _ in range(epochs)]


def progressive_schedule(stages, total_epochs, mixup_alphas=None, fixed_dropout=None):
    """Split total_epochs evenly across stages (12 -> 3 per stage). Always >= 1 epoch/stage.
    fixed_dropout: if given, only the image size grows; RandAugment is off and dropout stays at
    this value in every stage (size-only ablation, Run B0)."""
    n = len(stages)
    per = max(1, total_epochs // n)
    plan = []
    for i, st in enumerate(stages):
        reps = per + (total_epochs - per * n if i == n - 1 and total_epochs > per * n else 0)
        for _ in range(reps):
            size_only = fixed_dropout is not None
            plan.append({"stage": i + 1, "size": st["size"],
                         "randaug": None if size_only else st["randaug"],
                         "dropout": fixed_dropout if size_only else st["dropout"],
                         "mixup": float(mixup_alphas[i]) if mixup_alphas else 0.0})
    return plan


def warmup_cosine(optimizer, warmup_steps, total_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

# %% Metrics and plots


def compute_metrics(y_true, y_pred, class_names) -> dict:
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                          "support": int(s[i])} for i, c in enumerate(class_names)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "report": classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                        zero_division=0, digits=4),
    }


def mcnemar(y, p1, p2) -> dict:
    """McNemar test with continuity correction (chi-square, 1 dof).
    b = model 1 right & model 2 wrong, c = the reverse."""
    r1, r2 = (p1 == y), (p2 == y)
    b, c = int(np.sum(r1 & ~r2)), int(np.sum(~r1 & r2))
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "chi2": float(chi2), "p_value": float(math.erfc(math.sqrt(chi2 / 2)))}


def plot_confusion(cm, class_names, path, title):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_history(history, path, title):
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(ep, [h["train_loss"] for h in history], label="train")
    axes[0].plot(ep, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("Loss")
    axes[1].plot(ep, [h["train_acc"] for h in history], label="train")
    axes[1].plot(ep, [h["val_acc"] for h in history], label="val")
    axes[1].set_title("Accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

# %% Training loop (shared by ResNet-18, Run A, Run B, Run C)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, autocast, scaler,
                    mixup_alpha, rng):
    model.train()
    tot_loss, tot_correct, n = 0.0, 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        use_mix = mixup_alpha > 0
        if use_mix:
            lam = float(rng.beta(mixup_alpha, mixup_alpha))
            perm = torch.randperm(x.size(0), device=device)
            x = lam * x + (1 - lam) * x[perm]
            y_b = y[perm]
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            out = model(x)
            if use_mix:
                loss = lam * criterion(out, y) + (1 - lam) * criterion(out, y_b)
            else:
                loss = criterion(out, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()

        pred = out.argmax(1)
        if use_mix:
            correct = lam * (pred == y).sum().item() + (1 - lam) * (pred == y_b).sum().item()
        else:
            correct = (pred == y).sum().item()
        bs = x.size(0)
        tot_loss += loss.item() * bs
        tot_correct += correct
        n += bs
    return tot_loss / max(1, n), tot_correct / max(1, n)


@torch.no_grad()
def predict(model, loader, device, autocast, criterion=None):
    model.eval()
    ys, ps, tot_loss, n = [], [], 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast():
            out = model(x)
            if criterion is not None:
                tot_loss += criterion(out, y).item() * x.size(0)
        n += x.size(0)
        ys.append(y.cpu().numpy())
        ps.append(out.argmax(1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), tot_loss / max(1, n)


def train_dl_run(name, arch, plan, lr, batch_size, cfg, data, device, save_ckpt=True):
    out = Path(cfg["out_dir"])
    run_dir = out / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "checkpoints" / f"{name}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    autocast, scaler = make_amp(device)

    model = build_model(arch, len(data["class_names"]), plan[0]["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    train_ds = SubsetWithTransform(data["base_train"], data["splits"]["train"])
    val_ds = SubsetWithTransform(data["base_train"], data["splits"]["val"],
                                 build_transform(cfg["fixed_size"], train=False))
    val_loader = make_loader(val_ds, batch_size, False, device, cfg)

    steps_per_epoch = len(train_ds) // batch_size          # drop_last=True
    if steps_per_epoch == 0:
        raise ValueError("batch_size larger than the training set")
    scheduler = warmup_cosine(optimizer, cfg["warmup_epochs"] * steps_per_epoch,
                              len(plan) * steps_per_epoch)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history, best_acc, best_epoch = [], -1.0, -1
    train_time, total_start = 0.0, time.perf_counter()
    print(f"\n=== {name} | {arch} | lr={lr:g} | bs={batch_size} | epochs={len(plan)} | {device.type} ===")
    for epoch, spec in enumerate(plan, start=1):
        train_ds.transform = build_transform(spec["size"], train=True, randaug_mag=spec["randaug"])
        set_head_dropout(model, spec["dropout"])
        train_loader = make_loader(train_ds, batch_size, True, device, cfg, generator=gen, drop_last=True)

        sync(device)
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler, criterion,
                                          device, autocast, scaler, spec["mixup"], rng)
        sync(device)
        ep_time = time.perf_counter() - t0
        train_time += ep_time

        # validation always at the final resolution so all runs are comparable
        yv, pv, v_loss = predict(model, val_loader, device, autocast, criterion)
        v_acc = float(accuracy_score(yv, pv))
        v_f1 = float(f1_score(yv, pv, average="macro", zero_division=0))
        rec = {"epoch": epoch, "stage": spec["stage"], "size": spec["size"],
               "randaug": spec["randaug"], "dropout": spec["dropout"], "mixup": spec["mixup"],
               "lr_end": optimizer.param_groups[0]["lr"], "train_loss": tr_loss,
               "train_acc": tr_acc, "val_loss": v_loss, "val_acc": v_acc, "val_macro_f1": v_f1,
               "epoch_train_time_s": ep_time}
        history.append(rec)
        improved = v_acc > best_acc
        if improved:
            best_acc, best_epoch = v_acc, epoch
            if save_ckpt:
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, ckpt_path)
        print(f"ep {epoch:2d} [st{spec['stage']} {spec['size']}px ra={spec['randaug']} "
              f"do={spec['dropout']:.2f} mix={spec['mixup']:.2f}] "
              f"train {tr_loss:.3f}/{tr_acc:.3f}  val {v_loss:.3f}/{v_acc:.3f} f1={v_f1:.3f}  "
              f"{ep_time:.0f}s{'  *' if improved else ''}")

    summary = {
        "name": name, "arch": arch, "lr": lr, "batch_size": batch_size, "epochs": len(plan),
        "best_val_acc": best_acc, "best_epoch": best_epoch,
        "final_val_acc": history[-1]["val_acc"],
        "train_time_s": train_time, "total_time_s": time.perf_counter() - total_start,
        "params": count_params(model), "device": device.type,
        "peak_gpu_mem_gb": (torch.cuda.max_memory_allocated() / 1024 ** 3) if device.type == "cuda" else None,
        "checkpoint": str(ckpt_path) if save_ckpt else None,
    }
    save_json(history, run_dir / "history.json")
    save_json(summary, run_dir / "summary.json")
    plot_history(history, run_dir / "curves.png", name)
    del model, optimizer
    empty_cache(device)
    return summary


def run_dl(name, arch, plan, lr, cfg, data, device, save_ckpt=True):
    """Resumable wrapper with automatic batch-size fallback on out-of-memory."""
    summary_path = Path(cfg["out_dir"]) / "runs" / name / "summary.json"
    if summary_path.exists() and not cfg["force"]:
        print(f"[skip] {name} already finished -> {summary_path}")
        return load_json(summary_path)
    sizes = [cfg["batch_size"], cfg["fallback_batch_size"]]
    for i, bs in enumerate(sizes):
        oom = False
        try:
            return train_dl_run(name, arch, plan, lr, bs, cfg, data, device, save_ckpt)
        except RuntimeError as e:
            if not is_oom(e) or i == len(sizes) - 1:
                raise
            oom = True
        if oom:
            print(f"[OOM] batch size {bs} failed for {name}; retrying with {sizes[i + 1]}")
            empty_cache(device)

# %% Learning-rate selection on the validation split (EfficientNetV2-S, Run A settings)


def select_lr(cfg, data, device) -> float:
    path = Path(cfg["out_dir"]) / "selected_lr.json"
    if path.exists() and not cfg["force"]:
        lr = load_json(path)["selected_lr"]
        print(f"[skip] LR sweep done, selected lr={lr:g}")
        return lr
    if not cfg["run_lr_sweep"]:
        return cfg["lr"]
    results = {}
    for lr in cfg["lr_grid"]:
        plan = fixed_schedule(cfg["lr_sweep_epochs"], cfg["fixed_size"], cfg["dropout"])
        s = run_dl(f"lrsweep_effnet_lr{lr:g}", "efficientnet_v2_s", plan, lr, cfg, data, device,
                   save_ckpt=False)
        results[str(lr)] = {"best_val_acc": s["best_val_acc"], "train_time_s": s["train_time_s"]}
    best = max(results, key=lambda k: results[k]["best_val_acc"])
    save_json({"selected_lr": float(best), "sweep_epochs": cfg["lr_sweep_epochs"],
               "results": results}, path)
    print(f"LR sweep: {results} -> selected lr={float(best):g}")
    return float(best)

# %% Classical baseline: HOG + RBF-SVM


def hog_features(images_uint8):
    from skimage.color import rgb2gray
    from skimage.feature import hog
    return np.stack([hog(rgb2gray(img), orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
                     for img in images_uint8]).astype(np.float32)


def run_hog_svm(cfg, data) -> dict:
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    out = Path(cfg["out_dir"])
    summary_path = out / "runs" / "hog_svm" / "summary.json"
    if summary_path.exists() and not cfg["force"]:
        print(f"[skip] hog_svm already finished -> {summary_path}")
        return load_json(summary_path)

    X_all = data["base_train"].data
    y_all = np.asarray(data["base_train"].targets)
    tr, va = data["splits"]["train"], data["splits"]["val"]

    t0 = time.perf_counter()
    X_tr, X_va = hog_features(X_all[tr]), hog_features(X_all[va])
    feat_time = time.perf_counter() - t0
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_va = scaler.transform(X_tr), scaler.transform(X_va)
    print(f"\n=== hog_svm | feature dim={X_tr.shape[1]} | extraction {feat_time:.1f}s ===")

    grid, best = {}, None
    for C in cfg["svm_C_grid"]:
        t0 = time.perf_counter()
        clf = SVC(kernel="rbf", C=C, gamma="scale").fit(X_tr, y_all[tr])
        fit_time = time.perf_counter() - t0
        acc = float(accuracy_score(y_all[va], clf.predict(X_va)))
        grid[str(C)] = {"val_acc": acc, "fit_time_s": fit_time}
        print(f"C={C}: val acc={acc:.4f} ({fit_time:.1f}s)")
        if best is None or acc > best[1]:
            best = (C, acc, clf, fit_time)

    ckpt = out / "checkpoints" / "hog_svm.joblib"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "clf": best[2]}, ckpt)
    summary = {"name": "hog_svm", "best_C": best[0], "best_val_acc": best[1], "grid": grid,
               "train_time_s": feat_time + best[3], "feature_dim": int(X_tr.shape[1]),
               "checkpoint": str(ckpt)}
    save_json(summary, summary_path)
    return summary

# %% Final, single test-set evaluation


DL_RUNS = {   # step key -> (run name, arch, display name, train resolution)
    "resnet18": ("resnet18_fixed224", "resnet18", "ResNet-18 (fixed 224)", "224"),
    "effnet_fixed": ("A_effnetv2s_fixed224", "efficientnet_v2_s", "Run A: EffNetV2-S fixed 224", "224"),
    "effnet_prog_size_only": ("B0_effnetv2s_prog_size_only", "efficientnet_v2_s",
                              "Run B0: progressive size, fixed reg.", "128→224"),
    "effnet_progressive": ("B_effnetv2s_progressive", "efficientnet_v2_s",
                           "Run B: EffNetV2-S progressive", "128→224"),
    "effnet_prog_mixup": ("C_effnetv2s_prog_mixup", "efficientnet_v2_s",
                          "Run C: progressive + stage-wise Mixup", "128→224"),
}


def final_test(cfg, data, device) -> dict:
    out = Path(cfg["out_dir"])
    lock = out / "TEST_EVALUATED.lock"
    if lock.exists() and not cfg["force"]:
        raise RuntimeError(f"Test set already evaluated ({lock}). Results are in "
                           f"{out / 'test_results.json'}. Re-evaluating would break the "
                           f"single-use test protocol; pass --force only if you really must.")

    names = data["class_names"]
    test_idx = data["splits"]["test"]
    y_test = np.asarray(data["base_test"].targets)[test_idx]
    fig_dir = out / "figures"
    results, preds = {}, {}

    # trivial baselines
    rng = np.random.default_rng(cfg["seed"])
    preds["uniform_random"] = rng.integers(0, len(names), len(y_test))
    results["uniform_random"] = compute_metrics(y_test, preds["uniform_random"], names)
    results["uniform_random"].update(display="Uniform random guess", train_time_s=0.0, resolution="–")
    train_y = np.asarray(data["base_train"].targets)[data["splits"]["train"]]
    majority = int(np.bincount(train_y, minlength=len(names)).argmax())   # ties -> lowest index
    preds["majority_class"] = np.full_like(y_test, majority)
    results["majority_class"] = compute_metrics(y_test, preds["majority_class"], names)
    results["majority_class"].update(display=f"Majority class ('{names[majority]}')",
                                     train_time_s=0.0, resolution="–")

    # HOG + SVM
    svm_sum = out / "runs" / "hog_svm" / "summary.json"
    if svm_sum.exists():
        import joblib
        s = load_json(svm_sum)
        bundle = joblib.load(s["checkpoint"])
        X_te = bundle["scaler"].transform(hog_features(data["base_test"].data[test_idx]))
        preds["hog_svm"] = bundle["clf"].predict(X_te)
        results["hog_svm"] = compute_metrics(y_test, preds["hog_svm"], names)
        results["hog_svm"].update(display=f"HOG + RBF-SVM (C={s['best_C']})",
                                  best_val_acc=s["best_val_acc"], train_time_s=s["train_time_s"],
                                  resolution="32")

    # deep models: best-validation checkpoint, evaluated at 224
    autocast, _ = make_amp(device)
    test_ds = SubsetWithTransform(data["base_test"], test_idx,
                                  build_transform(cfg["fixed_size"], train=False))
    for key, (run, arch, display, res) in DL_RUNS.items():
        sp = out / "runs" / run / "summary.json"
        if not sp.exists():
            print(f"[warn] {run} not trained; skipped in test evaluation")
            continue
        s = load_json(sp)
        model = build_model(arch, len(names), pretrained=False)
        model.load_state_dict(torch.load(s["checkpoint"], map_location="cpu"))
        model.to(device)
        loader = make_loader(test_ds, s["batch_size"], False, device, cfg)
        yt, pt, _ = predict(model, loader, device, autocast)
        assert np.array_equal(yt, y_test)
        preds[run] = pt
        results[run] = compute_metrics(yt, pt, names)
        results[run].update(display=display, best_val_acc=s["best_val_acc"],
                            best_epoch=s["best_epoch"], train_time_s=s["train_time_s"],
                            params=s["params"], resolution=res, batch_size=s["batch_size"],
                            lr=s["lr"])
        del model
        empty_cache(device)

    for k, r in results.items():
        plot_confusion(r["confusion_matrix"], names, fig_dir / f"cm_{k}.png", r["display"])
        print(f"\n--- {r['display']} ---  acc={r['accuracy']:.4f}  macroF1={r['macro_f1']:.4f}")

    # paired significance tests between learned models (same 2,000 test images)
    preds = {k: np.asarray(v) for k, v in preds.items()}
    np.savez(out / "test_predictions.npz", y_true=y_test, **preds)
    keys = [k for k in preds if k not in ("uniform_random", "majority_class")]
    tests = {f"{a} vs {b}": mcnemar(y_test, preds[a], preds[b])
             for i, a in enumerate(keys) for b in keys[i + 1:]}
    save_json(tests, out / "mcnemar_tests.json")
    for pair, t in tests.items():
        print(f"McNemar {pair}: b={t['b']} c={t['c']} p={t['p_value']:.4f}")

    save_json(results, out / "test_results.json")
    lock.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    write_results_tables(results, names, out)
    return results

# %% Result tables (Markdown + LaTeX) and comparison figure


def write_results_tables(results, class_names, out):
    def fmt_t(s):
        return "–" if not s else f"{s / 60:.1f}"

    def val(r):
        return f"{100 * r['best_val_acc']:.2f}" if r.get("best_val_acc") is not None else "–"

    def params(r):
        return f"{r['params'] / 1e6:.1f}" if r.get("params") else "–"

    md = ["| Model | Train res. | Params (M) | Val Acc (%) | Test Acc (%) | Test Macro-F1 | Train time (min) |",
          "|---|---|---|---|---|---|---|"]
    tex = [r"\begin{table}[t]", r"\centering\small",
           r"\caption{CIFAR-10 results (5k train / 1k val / 2k test). Test set evaluated once.}",
           r"\label{tab:main}", r"\begin{tabular}{lcccccc}", r"\toprule",
           r"Model & Res. & Params (M) & Val Acc & Test Acc & Macro-F1 & Time (min) \\", r"\midrule"]
    for r in results.values():
        md.append(f"| {r['display']} | {r['resolution']} | {params(r)} | {val(r)} | "
                  f"{100 * r['accuracy']:.2f} | {r['macro_f1']:.4f} | {fmt_t(r.get('train_time_s'))} |")
        disp = r["display"].replace("_", r"\_").replace("&", r"\&")
        res = r["resolution"].replace("→", r"$\to$")
        tex.append(f"{disp} & {res} & {params(r)} & {val(r)} & {100 * r['accuracy']:.2f} & "
                   f"{r['macro_f1']:.3f} & {fmt_t(r.get('train_time_s'))} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    # per-class recall table for deep models
    deep = [k for k in results if k in {v[0] for v in DL_RUNS.values()}]
    pc = ["| Class | " + " | ".join(results[k]["display"] for k in deep) + " |",
          "|---|" + "---|" * len(deep)]
    for c in class_names:
        cells = [f"{results[k]['per_class'][c]['precision']:.2f} / {results[k]['per_class'][c]['recall']:.2f}"
                 for k in deep]
        pc.append(f"| {c} | " + " | ".join(cells) + " |")

    (out / "results_table.md").write_text(
        "\n".join(md) + "\n\n### Per-class precision / recall (test)\n\n" + "\n".join(pc) + "\n",
        encoding="utf-8")
    (out / "results_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\nTables written to {out / 'results_table.md'} and {out / 'results_table.tex'}")


def plot_val_comparison(cfg):
    out = Path(cfg["out_dir"])
    fig, ax = plt.subplots(figsize=(7, 4))
    drew = False
    for key, (run, _, display, _) in DL_RUNS.items():
        hp = out / "runs" / run / "history.json"
        if hp.exists():
            h = load_json(hp)
            ax.plot([e["epoch"] for e in h], [100 * e["val_acc"] for e in h], marker="o", ms=3, label=display)
            drew = True
    if not drew:
        plt.close(fig)
        return
    per = max(1, cfg["epochs"] // len(cfg["stages"]))
    for i in range(1, len(cfg["stages"])):
        ax.axvline(i * per + 0.5, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)  [evaluated at 224px]")
    ax.set_title("Validation accuracy per epoch (dotted = progressive stage boundaries)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figures" / "val_acc_comparison.png", dpi=200)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="EfficientNetV2-S progressive learning on CIFAR-10")
    p.add_argument("--smoke", action="store_true", help="tiny data, 1 epoch (1 per stage) per run")
    p.add_argument("--runs", nargs="+", default=ALL_STEPS, choices=ALL_STEPS,
                   help="pipeline steps to execute (default: all)")
    p.add_argument("--no-lr-sweep", action="store_true", help="skip the LR sweep, use lr=3e-4")
    p.add_argument("--lr-sweep-epochs", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--data-dir", default=None,
                   help="folder holding an existing CIFAR-10 copy (the folder that contains "
                        "'cifar-10-batches-py', or that folder itself); default ./data")
    p.add_argument("--force", action="store_true", help="re-run finished runs / allow test re-evaluation")
    args, _ = p.parse_known_args(argv)   # parse_known_args tolerates Jupyter's kernel args
    return args


def main(argv=None):
    args = parse_args(argv)
    cfg = copy.deepcopy(DEFAULT_CFG)
    if args.smoke:
        cfg.update(SMOKE_OVERRIDES, smoke_test=True)
    if args.no_lr_sweep:
        cfg["run_lr_sweep"] = False
    if args.lr_sweep_epochs:
        cfg["lr_sweep_epochs"] = args.lr_sweep_epochs
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    cfg["force"] = args.force

    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"])
    device = get_device()
    env = environment_info(device)
    print("Environment:", env)
    save_json(env, out / "environment.json")
    save_json(cfg, out / "config.json")

    data = load_data(cfg)
    steps = args.runs

    if "hog_svm" in steps:
        run_hog_svm(cfg, data)

    # select_lr runs the sweep, or reloads its saved result if it already ran
    if "lr_sweep" in steps or (out / "selected_lr.json").exists():
        lr = select_lr(cfg, data, device)
    else:
        lr = cfg["lr"]
        if any(s in steps for s in DL_RUNS):
            print(f"[warn] no LR sweep result found; using default lr={lr:g}")
    if any(s in steps for s in DL_RUNS):
        print(f"Using lr={lr:g} for all deep-learning runs")

    E = cfg["epochs"]
    plans = {
        "resnet18": fixed_schedule(E, cfg["fixed_size"], cfg["dropout"]),
        "effnet_fixed": fixed_schedule(E, cfg["fixed_size"], cfg["dropout"]),
        "effnet_prog_size_only": progressive_schedule(cfg["stages"], E,
                                                      fixed_dropout=cfg["dropout"]),
        "effnet_progressive": progressive_schedule(cfg["stages"], E),
        "effnet_prog_mixup": progressive_schedule(cfg["stages"], E, cfg["mixup_alphas"]),
    }
    for key, (run, arch, _, _) in DL_RUNS.items():
        if key in steps:
            run_dl(run, arch, plans[key], lr, cfg, data, device)

    plot_val_comparison(cfg)

    if "test" in steps:
        final_test(cfg, data, device)
    print(f"\nDone. Everything is in {out.resolve()}")


# main
if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        print("Notebook detected: call main(['--smoke']) for a smoke test or main([]) for the full run.")
    else:
        main()
