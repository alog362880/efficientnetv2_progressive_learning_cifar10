"""Classify every frame of a video with an already-trained model and save an annotated copy.

No retraining; the CIFAR-10 test split is never touched. Reuses the preprocessing and
Grad-CAM code from predict_image.py, so frames are treated exactly like single pictures.

Usage (Anaconda Prompt, after `conda activate F:\\conda_envs\\comp1818`):
    python predict_video.py my_video.mp4
    python predict_video.py my_video.mp4 --visualize          (adds Grad-CAM heatmap + box)
    python predict_video.py my_video.mp4 --run A_effnetv2s_fixed224 --smooth 10

Output: predictions/<video name>_<run>[_gradcam].mp4 plus a .json summary of how often
each class was predicted.

Notes
* --smooth N averages the class probabilities over the last N frames. Each frame is
  classified independently, so without smoothing the label can flicker between frames.
* Like predict_image.py, this is a classifier: one label per frame. The --visualize box shows
  where the model looked (Grad-CAM), not one box per object.
"""
import argparse
import json
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from comp1818_efficientnetv2_cifar10 import build_model, get_device
from predict_image import (CLASS_NAMES, RUN_ARCH, box_from_cam, make_transform,
                           predict_with_gradcam)


def draw_label(frame, lines, scale):
    """Top-left text panel: first line large (current prediction), others smaller."""
    y = int(12 * scale)
    for i, text in enumerate(lines):
        fs = (0.9 if i == 0 else 0.6) * scale
        th = max(1, int((2 if i == 0 else 1) * scale))
        (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        x = int(10 * scale)
        cv2.rectangle(frame, (x - 4, y - 4), (x + tw + 4, y + tht + base + 4),
                      (0, 255, 0) if i == 0 else (40, 40, 40), -1)
        cv2.putText(frame, text, (x, y + tht), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (0, 0, 0) if i == 0 else (255, 255, 255), th, cv2.LINE_AA)
        y += tht + base + int(12 * scale)


def overlay_gradcam(frame, cam, box, label, prob, scale):
    """Blend the heatmap into the centre square the model saw and draw the box."""
    h, w = frame.shape[:2]
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    heat = cv2.applyColorMap((cv2.resize(cam, (side, side)) * 255).astype(np.uint8),
                             cv2.COLORMAP_JET)
    roi = frame[top:top + side, left:left + side]
    frame[top:top + side, left:left + side] = cv2.addWeighted(roi, 0.6, heat, 0.4, 0)
    cv2.rectangle(frame, (left, top), (left + side - 1, top + side - 1), (255, 255, 255), 1)
    s = side / cam.shape[0]
    x0, y0, x1, y1 = box
    p0 = (left + int(x0 * s), top + int(y0 * s))
    p1 = (left + int(x1 * s) - 1, top + int(y1 * s) - 1)
    cv2.rectangle(frame, p0, p1, (0, 255, 0), max(2, int(3 * scale)))
    cv2.putText(frame, f"{label} {100 * prob:.0f}%", (p0[0] + 4, max(p0[1] - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (0, 255, 0), max(1, int(2 * scale)),
                cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser(description="Classify the frames of a video (CIFAR-10 classes)")
    p.add_argument("video", help="video file: mp4, avi, mov, mkv, ...")
    p.add_argument("--run", default="B_effnetv2s_progressive", choices=sorted(RUN_ARCH),
                   help="which trained model to use (default: Run B)")
    p.add_argument("--out-dir", default="./outputs", help="folder with checkpoints/")
    p.add_argument("--vis-dir", default="./predictions", help="where the annotated video goes")
    p.add_argument("--no-downscale", action="store_true",
                   help="do not shrink frames to 32x32 first (differs from training data)")
    p.add_argument("--visualize", action="store_true", help="add Grad-CAM heatmap and box")
    p.add_argument("--box-threshold", type=float, default=0.5)
    p.add_argument("--smooth", type=int, default=5,
                   help="average probabilities over the last N frames (1 = off)")
    p.add_argument("--max-frames", type=int, default=None, help="stop early (quick tests)")
    args = p.parse_args()

    if not Path(args.video).is_file():
        raise SystemExit(f"Video not found: {args.video}\n"
                         f"Tip: drag the video into this window to paste its exact path.")
    ckpt = Path(args.out_dir) / "checkpoints" / f"{args.run}.pt"
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}. Train the model first.")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    device = get_device()
    model = build_model(RUN_ARCH[args.run], len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(device).eval()
    tf = make_transform(downscale=not args.no_downscale)

    out_dir = Path(args.vis_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(args.video).stem}_{args.run}{'_gradcam' if args.visualize else ''}"
    out_path = out_dir / f"{stem}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    scale = max(w, h) / 1000

    print(f"Model: {args.run}  device={device.type}  video={w}x{h} @ {fps:.1f} fps"
          f"{f', {n_total} frames' if n_total else ''}  smooth={args.smooth}")
    recent = deque(maxlen=max(1, args.smooth))
    counts, n, t0 = Counter(), 0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and n >= args.max_frames):
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        x = tf(img).unsqueeze(0).to(device)
        if args.visualize:
            probs, cam, _ = predict_with_gradcam(model, x)
        else:
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)[0].cpu()
        recent.append(probs.numpy())
        avg = np.mean(recent, axis=0)
        order = np.argsort(avg)[::-1]
        c = int(order[0])
        counts[CLASS_NAMES[c]] += 1

        if args.visualize:
            overlay_gradcam(frame, cam, box_from_cam(cam, args.box_threshold),
                            CLASS_NAMES[c], float(avg[c]), scale)
        draw_label(frame, [f"{CLASS_NAMES[c]} {100 * avg[c]:.1f}%"]
                   + [f"{CLASS_NAMES[i]} {100 * avg[i]:.1f}%" for i in order[1:3]], scale)
        writer.write(frame)
        n += 1
        if n % 50 == 0:
            done = f"{n}/{n_total}" if n_total else str(n)
            print(f"  frame {done}  ({n / (time.perf_counter() - t0):.1f} frames/s)")

    cap.release()
    writer.release()
    if n == 0:
        raise SystemExit("No frames could be read from the video.")

    secs = time.perf_counter() - t0
    summary = {"video": str(args.video), "run": args.run, "frames": n,
               "smooth": args.smooth, "downscale": not args.no_downscale,
               "seconds": round(secs, 1), "frames_per_second": round(n / secs, 1),
               "share_of_frames": {k: round(v / n, 4) for k, v in counts.most_common()}}
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nProcessed {n} frames in {secs:.0f}s ({n / secs:.1f} frames/s)")
    print("Predicted class share of frames:")
    for k, v in counts.most_common():
        print(f"  {k:<11} {100 * v / n:5.1f}%")
    print(f"\nSaved {out_path}\n      {out_dir / (stem + '.json')}")


if __name__ == "__main__":
    main()
