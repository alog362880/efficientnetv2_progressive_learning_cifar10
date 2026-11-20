import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.utils.data import DataLoader
from torchvision import datasets

from comp1818_efficientnetv2_cifar10 import (DL_RUNS, SubsetWithTransform, build_model,
                                             build_transform, find_cifar_root, get_device,
                                             hog_features, load_json, make_amp)

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
RUNS = [(run, arch, display) for run, arch, display, _ in DL_RUNS.values()]
DISPLAY = {run: display for run, _, display in RUNS}
DISPLAY["hog_svm"] = "HOG + RBF-SVM"
COLORS = {"hog_svm": "#7f7f7f", "resnet18_fixed224": "#1f77b4", "A_effnetv2s_fixed224": "#ff7f0e",
          "B0_effnetv2s_prog_size_only": "#2ca02c", "B_effnetv2s_progressive": "#d62728",
          "C_effnetv2s_prog_mixup": "#9467bd"}


def savefig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------- scores (one-off inference)
def compute_scores(cfg, out, split, device, recompute):
    cache = out / f"scores_{split}.npz"
    if cache.exists() and not recompute:
        print(f"Using cached scores {cache}")
        return dict(np.load(cache))

    splits = load_json(out / "splits.json")
    root = find_cifar_root(cfg["data_dir"])
    base = datasets.CIFAR10(str(root), train=(split == "val"), download=False)
    idx = splits[split]
    y = np.asarray(base.targets)[idx]
    ds = SubsetWithTransform(base, idx, build_transform(cfg["fixed_size"], train=False))
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    autocast, _ = make_amp(device)
    scores = {"y_true": y}

    for run, arch, display in RUNS:
        ckpt = out / "checkpoints" / f"{run}.pt"
        if not ckpt.exists():
            print(f"  [skip] {run}: no checkpoint")
            continue
        model = build_model(arch, len(CLASS_NAMES), pretrained=False)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.to(device).eval()
        probs = []
        with torch.no_grad():
            for x, _ in loader:
                with autocast():
                    logits = model(x.to(device))
                probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        scores[run] = np.concatenate(probs)
        print(f"  scored {display}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    svm_ckpt = out / "checkpoints" / "hog_svm.joblib"
    if svm_ckpt.exists():
        import joblib
        bundle = joblib.load(svm_ckpt)
        X = bundle["scaler"].transform(hog_features(base.data[idx]))
        scores["hog_svm"] = bundle["clf"].decision_function(X)   # one-vs-rest margins
        print("  scored HOG + RBF-SVM (decision-function margins, not probabilities)")

    np.savez(cache, **scores)

    # the reported test results must not change: compare with the single test evaluation
    ref_path = out / "test_predictions.npz"
    if split == "test" and ref_path.exists():
        ref = np.load(ref_path)
        assert np.array_equal(ref["y_true"], y), "test split differs from the evaluated one"
        for run, _, display in RUNS:
            if run in scores and run in ref:
                agree = float(np.mean(scores[run].argmax(1) == ref[run]))
                print(f"  check {display}: {100 * agree:.2f}% of labels identical to test run")
    return scores


# ---------------------------------------------------------------- figures
def plot_loss_curves(out, fig_dir):
    hist = {run: load_json(out / "runs" / run / "history.json") for run, _, _ in RUNS
            if (out / "runs" / run / "history.json").exists()}
    n = len(hist)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), sharey=True,
                             squeeze=False)
    for ax, (run, h) in zip(axes.flat, hist.items()):
        ep = [e["epoch"] for e in h]
        ax.plot(ep, [e["train_loss"] for e in h], "o-", ms=3, label="training")
        ax.plot(ep, [e["val_loss"] for e in h], "s-", ms=3, label="validation")
        ax.set_title(DISPLAY[run], fontsize=9)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    fig.supylabel("Loss (cross-entropy, label smoothing 0.1)", fontsize=10)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Training vs validation loss per epoch", y=1.01)
    fig.tight_layout()
    savefig(fig, fig_dir / "loss_curves.png")


def plot_val_metrics(out, cfg, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for run, _, display in RUNS:
        hp = out / "runs" / run / "history.json"
        if not hp.exists():
            continue
        h = load_json(hp)
        ep = [e["epoch"] for e in h]
        axes[0].plot(ep, [100 * e["val_acc"] for e in h], "o-", ms=3, color=COLORS[run], label=display)
        axes[1].plot(ep, [e["val_macro_f1"] for e in h], "o-", ms=3, color=COLORS[run], label=display)
    per = max(1, cfg["epochs"] // len(cfg["stages"]))
    for ax, title, ylab in [(axes[0], "Validation accuracy", "Accuracy (%)"),
                            (axes[1], "Validation macro-F1", "Macro-F1")]:
        for i in range(1, len(cfg["stages"])):
            ax.axvline(i * per + 0.5, color="grey", ls=":", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Validation metrics vs epoch (dotted = progressive stage boundaries)", y=1.02)
    fig.tight_layout()
    savefig(fig, fig_dir / "val_metrics_vs_epoch.png")


def plot_class_distribution(cfg, out, fig_dir):
    splits = load_json(out / "splits.json")
    root = find_cifar_root(cfg["data_dir"])
    tr_t = np.asarray(datasets.CIFAR10(str(root), train=True, download=False).targets)
    te_t = np.asarray(datasets.CIFAR10(str(root), train=False, download=False).targets)
    counts = {"train": np.bincount(tr_t[splits["train"]], minlength=10),
              "validation": np.bincount(tr_t[splits["val"]], minlength=10),
              "test": np.bincount(te_t[splits["test"]], minlength=10)}
    x = np.arange(10)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, (name, c) in enumerate(counts.items()):
        bars = ax.bar(x + (i - 1) * 0.27, c, 0.27, label=f"{name} (n={c.sum():,})")
        ax.bar_label(bars, fontsize=7, padding=1)
    ax.set_xticks(x, CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylabel("Number of images")
    ax.set_title("Class distribution of the CIFAR-10 subsets (class-balanced by design)")
    ax.set_ylim(0, max(c.max() for c in counts.values()) * 1.3)
    ax.legend(fontsize=8, ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, fig_dir / "class_distribution.png")


def one_hot(y):
    return np.eye(len(CLASS_NAMES), dtype=int)[y]


def ap_table(scores):
    Y = one_hot(scores["y_true"])
    table = {}
    for key, S in scores.items():
        if key == "y_true":
            continue
        per = {c: float(average_precision_score(Y[:, i], S[:, i])) for i, c in enumerate(CLASS_NAMES)}
        table[key] = {"display": DISPLAY[key], "per_class_AP": per,
                      "mAP_macro": float(np.mean(list(per.values()))),
                      "AP_micro": float(average_precision_score(Y.ravel(), S.ravel()))}
    return table


def plot_pr_per_class(scores, model, split, fig_dir, ap):
    Y, S = one_hot(scores["y_true"]), scores[model]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(CLASS_NAMES):
        p, r, _ = precision_recall_curve(Y[:, i], S[:, i])
        ax.plot(r, p, lw=1.3, color=cmap(i), label=f"{c} (AP {ap[model]['per_class_AP'][c]:.3f})")
    p, r, _ = precision_recall_curve(Y.ravel(), S.ravel())
    ax.plot(r, p, "k-", lw=2.5, label=f"all classes, micro (AP {ap[model]['AP_micro']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Precision-recall per class, one-vs-rest\n{DISPLAY[model]} ({split} set, "
                 f"mAP {ap[model]['mAP_macro']:.3f})", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)
    savefig(fig, fig_dir / f"pr_curves_{model}.png")


def plot_pr_all_models(scores, split, fig_dir, ap):
    Y = one_hot(scores["y_true"]).ravel()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for key in COLORS:
        if key not in scores:
            continue
        p, r, _ = precision_recall_curve(Y, scores[key].ravel())
        ax.plot(r, p, lw=1.8, color=COLORS[key], label=f"{DISPLAY[key]} (AP {ap[key]['AP_micro']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Micro-averaged precision-recall, all models ({split} set)", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)
    savefig(fig, fig_dir / "pr_micro_all_models.png")


def plot_f1_confidence(scores, model, split, fig_dir):
    Y, S = one_hot(scores["y_true"]).astype(bool), scores[model]
    t = np.linspace(0, 1, 201)
    f1 = np.zeros((len(CLASS_NAMES), len(t)))
    for i in range(len(CLASS_NAMES)):
        pos = S[:, i][None, :] >= t[:, None]                 # (thresholds, images)
        tp = (pos & Y[:, i][None, :]).sum(1)
        f1[i] = 2 * tp / np.maximum(pos.sum(1) + Y[:, i].sum(), 1)
    mean = f1.mean(0)
    best = int(np.argmax(mean))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(CLASS_NAMES):
        ax.plot(t, f1[i], lw=1, alpha=0.8, color=cmap(i), label=c)
    ax.plot(t, mean, "k-", lw=2.5, label=f"all classes: max {mean[best]:.3f} at {t[best]:.2f}")
    ax.axvline(t[best], color="k", ls=":", lw=1)
    ax.set_xlabel("Confidence threshold (softmax probability)")
    ax.set_ylabel("F1")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"F1 vs confidence, one-vs-rest\n{DISPLAY[model]} ({split} set)", fontsize=10)
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.grid(alpha=0.3)
    savefig(fig, fig_dir / f"f1_confidence_{model}.png")
    return {"best_threshold": float(t[best]), "best_mean_f1": float(mean[best])}


def plot_per_class_ap(ap, split, fig_dir):
    keys = [k for k in COLORS if k in ap]
    x = np.arange(len(CLASS_NAMES))
    width = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for j, key in enumerate(keys):
        vals = [ap[key]["per_class_AP"][c] for c in CLASS_NAMES]
        ax.bar(x + (j - (len(keys) - 1) / 2) * width, vals, width, color=COLORS[key],
               label=f"{DISPLAY[key]} (mAP {ap[key]['mAP_macro']:.3f})")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x, CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylabel("Average precision")
    ax.set_title(f"Per-class average precision, one-vs-rest ({split} set)")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, fig_dir / "per_class_ap.png")


def main():
    p = argparse.ArgumentParser(description="Report figures (no retraining)")
    p.add_argument("--out-dir", default="./outputs")
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--model", default="B_effnetv2s_progressive", choices=[r for r, _, _ in RUNS],
                   help="model used for the per-class PR and F1-confidence plots")
    p.add_argument("--recompute", action="store_true", help="ignore cached scores")
    args = p.parse_args()

    out = Path(args.out_dir)
    cfg = load_json(out / "config.json")
    fig_dir = out / "figures" / "report"
    device = get_device()

    print("Curves from training history:")
    plot_loss_curves(out, fig_dir)
    plot_val_metrics(out, cfg, fig_dir)
    plot_class_distribution(cfg, out, fig_dir)

    print(f"\nConfidence scores on the {args.split} split (frozen models, no training):")
    scores = compute_scores(cfg, out, args.split, device, args.recompute)
    ap = ap_table(scores)
    plot_pr_per_class(scores, args.model, args.split, fig_dir, ap)
    plot_pr_all_models(scores, args.split, fig_dir, ap)
    f1c = plot_f1_confidence(scores, args.model, args.split, fig_dir)
    plot_per_class_ap(ap, args.split, fig_dir)

    summary = {"split": args.split, "ap": ap, "f1_confidence": {args.model: f1c},
               "note": "HOG + SVM AP uses one-vs-rest SVM decision margins, not probabilities."}
    (fig_dir / f"ap_scores_{args.split}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nmAP (macro over classes):")
    for key, v in ap.items():
        print(f"  {v['display']:<42} {v['mAP_macro']:.4f}")
    print("\nIoU distribution: skipped. IoU needs predicted and ground-truth boxes (object "
          "detection); CIFAR-10 is image classification with no boxes.")


if __name__ == "__main__":
    main()
