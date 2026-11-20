"""Deployment-cost benchmark for the trained COMP1818 models.

Measures, for each final model: parameter count, checkpoint size on disk, FLOPs per 224 px image,
inference latency (batch 1) and throughput (batch 32) on the GPU and the CPU, and peak GPU memory
during inference. Inputs are random tensors, so no CIFAR-10 test image is touched.

Usage (from the project root, in the comp1818 conda env):
    python benchmark_deployment.py
Writes outputs/deployment_benchmark.json (the report quotes the range over two runs, saved as
_run1/_run2).
"""
import json
import platform
import time
from pathlib import Path

import numpy as np
from skimage.feature import hog  # skimage.color crashes on import in this Windows env (scipy DLL)
import torch
import torch.nn as nn
import joblib
from torch.utils.flop_counter import FlopCounterMode
from torchvision import models

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "outputs" / "checkpoints"
OUT = ROOT / "outputs" / "deployment_benchmark.json"
SIZE = 224
WARMUP, REPS_GPU, REPS_CPU = 10, 50, 10


def build(arch):
    if arch == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(0.2, inplace=True), nn.Linear(m.classifier[1].in_features, 10))
    else:
        m = models.resnet18(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.fc.in_features, 10))
    return m


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def time_model(model, device, batch, reps, amp):
    x = torch.randn(batch, 3, SIZE, SIZE, device=device)
    ctx = (lambda: torch.autocast("cuda", dtype=torch.float16)) if amp else torch.no_grad
    for _ in range(WARMUP):
        with ctx():
            model(x)
    sync(device)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        with ctx():
            model(x)
        sync(device)
        times.append(time.perf_counter() - t0)
    med = float(np.median(times))
    return {"median_ms_per_batch": med * 1e3, "images_per_s": batch / med}


@torch.no_grad()
def gflops(model):
    model = model.cpu().eval()
    with FlopCounterMode(display=False) as fc:
        model(torch.randn(1, 3, SIZE, SIZE))
    return fc.get_total_flops() / 1e9


def bench_deep(name, arch, ckpt):
    model = build(arch)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    res = {"params": sum(p.numel() for p in model.parameters()),
           "checkpoint_mb": ckpt.stat().st_size / 1e6,
           "gflops_224": gflops(model)}
    torch.set_num_threads(torch.get_num_threads())
    res["cpu_fp32_b1"] = time_model(model.cpu(), torch.device("cpu"), 1, REPS_CPU, amp=False)
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        model = model.to(dev)
        for b in (1, 32):
            torch.cuda.reset_peak_memory_stats()
            r = time_model(model, dev, b, REPS_GPU, amp=True)
            r["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 1e6
            res[f"gpu_amp_b{b}"] = r
        model.cpu()
        torch.cuda.empty_cache()
    print(name, json.dumps(res, indent=1))
    return res


def rgb2gray(img):
    """Same luminance weights as skimage.color.rgb2gray (ITU-R 709), output in [0, 1]."""
    return (img.astype(np.float64) / 255.0) @ np.array([0.2125, 0.7154, 0.0721])


def bench_hog_svm(path):
    obj = joblib.load(path)
    scaler, clf = obj["scaler"], obj["clf"]
    imgs = np.random.default_rng(0).integers(0, 256, size=(32, 32, 32, 3), dtype=np.uint8)

    def predict(batch):
        f = np.stack([hog(rgb2gray(im), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                          block_norm="L2-Hys", feature_vector=True) for im in batch])
        return clf.predict(scaler.transform(f))

    for _ in range(3):
        predict(imgs[:1])
    out = {}
    for b in (1, 32):
        ts = []
        for _ in range(REPS_CPU):
            t0 = time.perf_counter()
            predict(imgs[:b])
            ts.append(time.perf_counter() - t0)
        med = float(np.median(ts))
        out[f"cpu_b{b}"] = {"median_ms_per_batch": med * 1e3, "images_per_s": b / med}
    res = {"checkpoint_mb": path.stat().st_size / 1e6, "n_support_vectors": int(clf.n_support_.sum()), **out}
    print("hog_svm", json.dumps(res, indent=1))
    return res


def main():
    torch.backends.cudnn.benchmark = True
    results = {
        "env": {"torch": torch.__version__, "cpu": platform.processor(), "cpu_threads": torch.get_num_threads(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "input_size": SIZE, "note": "random inputs; GPU uses fp16 autocast; median of repeated runs"},
        "resnet18": bench_deep("resnet18", "resnet18", CKPT / "resnet18_fixed224.pt"),
        "efficientnet_v2_s": bench_deep("efficientnet_v2_s", "efficientnet_v2_s", CKPT / "B_effnetv2s_progressive.pt"),
        "hog_svm": bench_hog_svm(CKPT / "hog_svm.joblib"),
    }
    OUT.write_text(json.dumps(results, indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
