| Model | Train res. | Params (M) | Val Acc (%) | Test Acc (%) | Test Macro-F1 | Train time (min) |
|---|---|---|---|---|---|---|
| Uniform random guess | – | – | – | 11.00 | 0.1102 | – |
| Majority class ('airplane') | – | – | – | 10.00 | 0.0182 | – |
| HOG + RBF-SVM (C=10) | 32 | – | 35.00 | 38.50 | 0.3756 | 0.0 |
| ResNet-18 (fixed 224) | 224 | 11.2 | 34.00 | 37.00 | 0.3204 | 0.0 |
| Run A: EffNetV2-S fixed 224 | 224 | 20.2 | 41.00 | 37.00 | 0.3500 | 0.0 |
| Run B: EffNetV2-S progressive | 128→224 | 20.2 | 65.00 | 67.00 | 0.6509 | 0.3 |
| Run C: progressive + stage-wise Mixup | 128→224 | 20.2 | 74.00 | 72.50 | 0.7232 | 0.1 |

### Per-class precision / recall (test)

| Class | ResNet-18 (fixed 224) | Run A: EffNetV2-S fixed 224 | Run B: EffNetV2-S progressive | Run C: progressive + stage-wise Mixup |
|---|---|---|---|---|
| airplane | 0.65 / 0.75 | 0.40 / 0.70 | 0.77 / 0.85 | 0.77 / 0.85 |
| automobile | 0.68 / 0.75 | 0.62 / 0.50 | 0.71 / 0.85 | 0.93 / 0.70 |
| bird | 0.00 / 0.00 | 0.00 / 0.00 | 0.59 / 0.50 | 0.71 / 0.50 |
| cat | 0.00 / 0.00 | 0.00 / 0.00 | 0.75 / 0.15 | 0.85 / 0.55 |
| deer | 1.00 / 0.05 | 0.57 / 0.60 | 0.71 / 0.60 | 0.76 / 0.65 |
| dog | 0.44 / 0.35 | 1.00 / 0.10 | 0.65 / 0.65 | 0.58 / 0.70 |
| frog | 1.00 / 0.15 | 1.00 / 0.40 | 0.63 / 0.85 | 0.86 / 0.90 |
| horse | 0.38 / 0.15 | 0.19 / 0.85 | 0.52 / 0.80 | 0.61 / 0.85 |
| ship | 0.53 / 0.50 | 0.67 / 0.20 | 0.87 / 0.65 | 0.81 / 0.65 |
| truck | 0.19 / 1.00 | 0.78 / 0.35 | 0.70 / 0.80 | 0.60 / 0.90 |
