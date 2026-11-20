"""Classify your own pictures with an already-trained model. No retraining.

Loads a saved checkpoint from outputs/checkpoints/ and prints the top predictions.
It never touches the CIFAR-10 test split, so the single-use test protocol is unaffected.

Usage (in Anaconda Prompt, after `conda activate F:\\conda_envs\\comp1818`):
    python predict_image.py my_cat.jpg
    python predict_image.py photo1.jpg photo2.png --run A_effnetv2s_fixed224
    python predict_image.py my_cat.jpg --no-downscale
    python predict_image.py test_picture          (a folder: classifies every picture in it)
    python predict_image.py test_picture --visualize   (adds a Grad-CAM heatmap + box picture)

Note on --visualize: this is a classifier, not an object detector. The box is drawn around
the region the Grad-CAM heatmap (Selvaraju et al., 2017) marks as most important for the
predicted class. It shows where the model looked, not a trained object location.

By default the picture is first shrunk to 32x32 and then enlarged to 224x224, exactly like
the CIFAR-10 training images, so the model sees the kind of input it was trained on.
--no-downscale feeds the full-resolution picture instead (useful to compare in the report).
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from comp1818_efficientnetv2_cifar10 import (DL_RUNS, IMAGENET_MEAN, IMAGENET_STD,
                                             build_model, get_device)

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]   # torchvision CIFAR-10 order
RUN_ARCH = {run: arch for run, arch, _, _ in DL_RUNS.values()}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def make_transform(downscale: bool):
    t = []
    if downscale:
        # match training data: square 32x32 image, then upscaled to 224
        t += [transforms.Resize(32, antialias=True), transforms.CenterCrop(32)]
    else:
        t += [transforms.Resize(224, antialias=True), transforms.CenterCrop(224)]
    t += [transforms.Resize((224, 224), antialias=True), transforms.ToTensor(),
          transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(t)


def gradcam_layer(model):
    """Last convolutional block: highest-level features that still keep spatial layout."""
    return model.features[-1] if hasattr(model, "features") else model.layer4[-1]


def predict_with_gradcam(model, x, class_idx=None):
    """Grad-CAM (Selvaraju et al., 2017). Returns (probs, cam in [0,1] at 224x224, class used).
    The heatmap shows which part of the input pushed the model towards that class."""
    store = {}

    def fwd_hook(_m, _i, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    h = gradcam_layer(model).register_forward_hook(fwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            c = int(probs.argmax()) if class_idx is None else class_idx
            logits[0, c].backward()
    finally:
        h.remove()
    act, grad = store["act"][0], store["grad"][0]            # (C, h, w)
    weights = grad.mean(dim=(1, 2), keepdim=True)            # channel importance
    cam = torch.relu((weights * act).sum(0))[None, None]
    cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear",
                                          align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return probs.detach().cpu(), cam.detach().cpu().numpy(), c


def box_from_cam(cam, threshold=0.5):
    """Box around the largest connected region where the heatmap >= threshold * max.
    Returns (x0, y0, x1, y1) in heatmap pixel coordinates."""
    import numpy as np
    from scipy import ndimage
    mask = cam >= threshold
    labels, n = ndimage.label(mask)
    if n == 0:
        return 0, 0, cam.shape[1], cam.shape[0]
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    ys, xs = np.where(labels == int(np.argmax(sizes)) + 1)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def save_visualisation(img, cam, box, label, prob, top_lines, out_path, threshold):
    """Left: full picture with the model's crop area and the box. Right: Grad-CAM overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import numpy as np

    w, h = img.size
    side = min(w, h)                                   # the centre square the model saw
    left, top = (w - side) / 2, (h - side) / 2
    s = side / cam.shape[0]                            # heatmap px -> original px
    x0, y0, x1, y1 = box
    bx, by, bw, bh = left + x0 * s, top + y0 * s, (x1 - x0) * s, (y1 - y0) * s

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5 * max(h / w, 0.6)))
    ax = axes[0]
    ax.imshow(img)
    ax.add_patch(patches.Rectangle((left, top), side, side, fill=False, ls="--",
                                   ec="white", lw=1.2, alpha=0.8))
    ax.add_patch(patches.Rectangle((bx, by), bw, bh, fill=False, ec="lime", lw=3))
    ax.text(bx, max(by - 6, 0), f"{label} {100 * prob:.1f}%", color="black", fontsize=12,
            weight="bold", va="bottom", bbox=dict(facecolor="lime", edgecolor="none", pad=2))
    ax.set_title("Prediction + Grad-CAM box (dashed = area the model sees)", fontsize=10)
    ax.axis("off")

    ax = axes[1]
    crop = np.asarray(img.crop((left, top, left + side, top + side)).resize(cam.shape[::-1]))
    ax.imshow(crop)
    ax.imshow(cam, cmap="jet", alpha=0.45)
    ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="lime", lw=2))
    ax.set_title(f"Grad-CAM for '{label}' (box: heat >= {threshold:.0%} of max)", fontsize=10)
    ax.axis("off")
    fig.text(0.5, 0.01, "   ".join(top_lines), ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Predict CIFAR-10 classes for your own images")
    p.add_argument("images", nargs="+", help="image file(s) or folder(s) of images: jpg, png, webp, ...")
    p.add_argument("--run", default="B_effnetv2s_progressive", choices=sorted(RUN_ARCH),
                   help="which trained model to use (default: Run B)")
    p.add_argument("--out-dir", default="./outputs", help="folder with checkpoints/")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--no-downscale", action="store_true",
                   help="do not shrink to 32x32 first (differs from training data)")
    p.add_argument("--visualize", action="store_true",
                   help="save a picture with a Grad-CAM heatmap and a box around the region "
                        "the model used for its decision")
    p.add_argument("--box-threshold", type=float, default=0.5,
                   help="heatmap level (fraction of max) used to draw the box (default 0.5)")
    p.add_argument("--vis-dir", default="./predictions", help="where --visualize saves pictures")
    args = p.parse_args()

    ckpt = Path(args.out_dir) / "checkpoints" / f"{args.run}.pt"
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}. Train the model first.")

    device = get_device()
    model = build_model(RUN_ARCH[args.run], len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(device).eval()
    tf = make_transform(downscale=not args.no_downscale)
    k = min(args.topk, len(CLASS_NAMES))

    print(f"Model: {args.run}  ({ckpt})  device={device.type}  "
          f"downscale={'off' if args.no_downscale else '32px'}")
    results = {}
    paths = []
    for item in args.images:                       # a folder means: every picture inside it
        p_item = Path(item)
        if p_item.is_dir():
            found = sorted(f for f in p_item.iterdir() if f.suffix.lower() in IMAGE_EXTS)
            if not found:
                print(f"\n[skip] no pictures ({', '.join(sorted(IMAGE_EXTS))}) in folder: {item}")
            paths += [str(f) for f in found]
        else:
            paths.append(item)

    for path in paths:
        if not Path(path).is_file():
            print(f"\n[skip] file not found: {path}\n"
                  f"       Tip: drag the picture into this window to paste its exact path.")
            continue
        try:
            img = Image.open(path).convert("RGB")  # handles PNG transparency / greyscale
        except OSError as e:
            print(f"\n[skip] not a readable image: {path} ({e})")
            continue
        x = tf(img).unsqueeze(0).to(device)
        if args.visualize:
            probs, cam, c = predict_with_gradcam(model, x)
        else:
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)[0].cpu()
        top = torch.topk(probs, k)
        print(f"\n{path}")
        for prob, idx in zip(top.values, top.indices):
            print(f"  {CLASS_NAMES[idx]:<11} {100 * prob:5.1f}%")
        results[str(path)] = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in top.indices}
        if args.visualize:
            box = box_from_cam(cam, args.box_threshold)
            lines = [f"{CLASS_NAMES[i]} {100 * float(probs[i]):.1f}%" for i in top.indices]
            suffix = "full" if args.no_downscale else "32px"
            out_path = Path(args.vis_dir) / f"{Path(path).stem}_{args.run}_{suffix}.png"
            save_visualisation(img, cam, box, CLASS_NAMES[c], float(probs[c]), lines,
                               out_path, args.box_threshold)
            print(f"  -> saved {out_path}")

    print("\nNote: the model can only answer with one of the 10 CIFAR-10 classes, "
          "even for pictures of something else.")
    return results


if __name__ == "__main__":
    main()
