| Model | Train res. | Params (M) | Val Acc (%) | Test Acc (%) | Test Macro-F1 | Train time (min) |
|---|---|---|---|---|---|---|
| Uniform random guess | – | – | – | 10.30 | 0.1028 | – |
| Majority class ('airplane') | – | – | – | 10.00 | 0.0182 | – |
| HOG + RBF-SVM (C=1) | 32 | – | 54.30 | 54.10 | 0.5406 | 0.1 |
| ResNet-18 (fixed 224) | 224 | 11.2 | 92.90 | 91.85 | 0.9184 | 5.1 |
| Run A: EffNetV2-S fixed 224 | 224 | 20.2 | 96.50 | 95.35 | 0.9534 | 18.1 |
| Run B0: progressive size, fixed reg. | 128→224 | 20.2 | 96.10 | 95.05 | 0.9505 | 12.4 |
| Run B: EffNetV2-S progressive | 128→224 | 20.2 | 96.70 | 95.50 | 0.9549 | 12.9 |
| Run C: progressive + stage-wise Mixup | 128→224 | 20.2 | 96.10 | 95.35 | 0.9534 | 13.0 |

### Per-class precision / recall (test)

| Class | ResNet-18 (fixed 224) | Run A: EffNetV2-S fixed 224 | Run B0: progressive size, fixed reg. | Run B: EffNetV2-S progressive | Run C: progressive + stage-wise Mixup |
|---|---|---|---|---|---|
| airplane | 0.92 / 0.93 | 0.96 / 0.97 | 0.96 / 0.95 | 0.95 / 0.97 | 0.94 / 0.96 |
| automobile | 0.95 / 0.94 | 0.97 / 0.96 | 0.97 / 0.98 | 0.95 / 0.98 | 0.95 / 0.97 |
| bird | 0.91 / 0.89 | 0.96 / 0.94 | 0.96 / 0.94 | 0.96 / 0.95 | 0.97 / 0.94 |
| cat | 0.84 / 0.84 | 0.93 / 0.87 | 0.90 / 0.85 | 0.93 / 0.90 | 0.94 / 0.88 |
| deer | 0.92 / 0.94 | 0.97 / 0.96 | 0.95 / 0.98 | 0.96 / 0.96 | 0.98 / 0.95 |
| dog | 0.87 / 0.87 | 0.89 / 0.94 | 0.87 / 0.91 | 0.93 / 0.93 | 0.90 / 0.94 |
| frog | 0.94 / 0.94 | 0.96 / 0.98 | 0.97 / 0.98 | 0.97 / 0.98 | 0.95 / 0.99 |
| horse | 0.94 / 0.94 | 0.97 / 0.95 | 0.97 / 0.95 | 0.96 / 0.96 | 0.95 / 0.97 |
| ship | 0.95 / 0.95 | 0.99 / 0.98 | 0.98 / 0.98 | 0.98 / 0.96 | 0.98 / 0.96 |
| truck | 0.93 / 0.94 | 0.95 / 0.97 | 0.98 / 0.96 | 0.97 / 0.94 | 0.97 / 0.95 |
