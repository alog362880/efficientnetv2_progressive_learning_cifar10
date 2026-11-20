import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "outputs"
RUNS = {
    "ResNet-18": "resnet18_fixed224",
    "A": "A_effnetv2s_fixed224",
    "B0": "B0_effnetv2s_prog_size_only",
    "B": "B_effnetv2s_progressive",
    "C": "C_effnetv2s_prog_mixup",
}
SIZES = [128, 160, 192, 224]


def load(name, fname):
    with open(OUT / "runs" / name / fname, encoding="utf8") as f:
        return json.load(f)


def main():
    hist = {k: load(v, "history.json") for k, v in RUNS.items()}
    summ = {k: load(v, "summary.json") for k, v in RUNS.items()}
    t_a = summ["A"]["train_time_s"]

    print("== Training-step time")
    for k, s in summ.items():
        t = s["train_time_s"]
        per_stage = {}
        for e in hist[k]:
            per_stage.setdefault(e["size"], []).append(e["epoch_train_time_s"])
        stages = " / ".join(f"{np.mean(v):.1f}" for _, v in sorted(per_stage.items()))
        print(f"{k:10s} {t / 60:5.2f} min  ratio {t / t_a:.3f}  speed-up {t_a / t:.2f}x  "
              f"s/epoch [{stages}]  peak {s['peak_gpu_mem_gb']:.2f} GB")

    flops = sum((s / 224) ** 2 for s in SIZES) / len(SIZES)
    print(f"\nFLOPs ratio (equal epochs per stage): {flops:.3f}  -> {1 / flops:.2f}x")

    x, y = [], []
    for k in ["A", "B0", "B", "C"]:
        for e in hist[k]:
            x.append((e["size"] / 224) ** 2)
            y.append(e["epoch_train_time_s"])
    b, a = np.polyfit(x, y, 1)
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    pred = [a + b * (s / 224) ** 2 for s in SIZES]
    ratio = np.mean(pred) / pred[-1]
    print(f"Cost model: t = {a:.1f} s + {b:.1f} s * (S/224)^2   R^2 = {r2:.3f}  (n = {len(x)} epochs)")
    print(f"  predicted s/epoch {' / '.join(f'{p:.1f}' for p in pred)}; predicted ratio {ratio:.3f}; "
          f"overhead share at 224 px {a / pred[-1]:.1%}")

    print("\n== Train-val gap at best-validation epoch (pp)")
    for k, s in summ.items():
        e = hist[k][s["best_epoch"] - 1]
        print(f"{k:10s} epoch {e['epoch']:2d}  train {100 * e['train_acc']:.2f}  "
              f"val {100 * e['val_acc']:.2f}  gap {100 * (e['train_acc'] - e['val_acc']):+.2f}"
              + ("  (Mixup: not comparable)" if e["mixup"] > 0 else ""))

    with open(OUT / "test_results.json", encoding="utf8") as f:
        test = json.load(f)
    with open(OUT / "mcnemar_tests.json", encoding="utf8") as f:
        mc = json.load(f)
    print("\n== Test accuracy, 95% CI half-width, McNemar vs A")
    for k, v in RUNS.items():
        acc = test[v]["accuracy"]
        hw = 1.96 * np.sqrt(acc * (1 - acc) / 2000) * 100
        key = next((kk for kk in mc if RUNS["A"] in kk and v in kk and v != RUNS["A"]), None)
        p = f"p = {mc[key]['p_value']:.3g} (b={mc[key]['b']}, c={mc[key]['c']})" if key else ""
        print(f"{k:10s} acc {100 * acc:.2f} +/- {hw:.2f} pp  macro-F1 {test[v]['macro_f1']:.4f}  {p}")


if __name__ == "__main__":
    main()
