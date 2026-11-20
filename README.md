# EfficientNetV2 progressive learning on CIFAR-10 (low data, laptop GPU)

Code, logs and results for a COMP1818 (University of Greenwich) coursework study:
**does EfficientNetV2's progressive learning still save training time without losing accuracy when
a pretrained model is fine-tuned on a small image dataset on limited hardware?**

Based on Tan & Le, *EfficientNetV2: Smaller Models and Faster Training*, ICML 2021
([paper](https://arxiv.org/abs/2104.00298)).

## What is compared

All deep models are ImageNet-pretrained torchvision checkpoints fine-tuned for 12 epochs
(AdamW, weight decay 0.05, 1 warm-up epoch + cosine decay, batch 32, label smoothing 0.1, mixed
precision, learning rate 1e-4 chosen by a 3-epoch sweep on Run A).

| Run | Model | Training |
|---|---|---|
| Baselines | uniform random, majority class | no training |
| HOG + SVM | HOG (32 px, grey) + RBF-SVM, C in {1, 10} | classical baseline |
| ResNet-18 | ResNet-18 | fixed 224 px, same settings as A |
| A | EfficientNetV2-S | fixed 224 px, dropout 0.2, no RandAugment |
| B0 | EfficientNetV2-S | image size 128 → 160 → 192 → 224, A's regularisation (size only) |
| B | EfficientNetV2-S | paper's progressive learning: size 128 → 224, RandAugment 5 → 15, dropout 0.10 → 0.30 |
| C | EfficientNetV2-S | B + Mixup alpha 0 → 0.05 → 0.10 → 0.20 |

Progressive runs use four 3-epoch stages. Validation accuracy is always measured at 224 px.

## Data

CIFAR-10 (<https://www.cs.toronto.edu/~kriz/cifar.html>) is downloaded automatically into
`./data` on the first run (it is not included in this repository). Stratified, non-overlapping
subsets with seed 42:

* train: 500 images per class (5,000) and validation: 100 per class (1,000), from the official training set
* test: 200 per class (2,000), from the official test set

The exact image indices are in `outputs/splits.json`.

## Setup

```bash
conda env create -f environment.yml
conda activate comp1818
```

`environment.yml` installs the CUDA 12.6 build of PyTorch. On a CPU-only machine, delete the
`--index-url` line first. The results were produced with Python 3.11, PyTorch 2.14.1 and
torchvision 0.29.1 on an NVIDIA RTX 2050 laptop GPU (4 GB); see `outputs/environment.json`.

## How to run

Run from the repository folder, in this order:

```bash
# 1. Quick check that everything works (tiny data, a few minutes)
python comp1818_efficientnetv2_cifar10.py --smoke

# 2. Full pipeline: HOG+SVM, LR sweep, ResNet-18, runs A, B0, B, C, then the single test evaluation
python comp1818_efficientnetv2_cifar10.py

# 3. Deployment cost (Table 3 of the report): size, FLOPs, CPU/GPU latency, GPU memory
python benchmark_deployment.py

# 4. Extra report figures (PR curves, per-class AP, loss curves) and mAP scores
python make_figures.py
```

Step 2 takes about 1.5 hours on the RTX 2050. Finished runs are skipped when it is started
again, so an interrupted run only repeats the current model. One step can be run on its own,
e.g. `--runs effnet_progressive` (steps: `hog_svm lr_sweep resnet18 effnet_fixed
effnet_prog_size_only effnet_progressive effnet_prog_mixup test`).

**The test set is evaluated only once.** After the `test` step, `outputs/TEST_EVALUATED.lock` is
created and the pipeline refuses to evaluate again. `--force` overrides this; do not use it to
reproduce the reported numbers, because re-testing after changes would leak the test set into
model choices.

### Trying the trained models on your own pictures or video

```bash
python predict_image.py my_picture.jpg                # top-3 classes (also accepts folders)
python predict_image.py my_picture.jpg --visualize    # adds a Grad-CAM heatmap and box
python predict_video.py my_video.mp4 --visualize      # annotated copy in ./predictions
```

These are classifiers: one label per image (or frame) from the 10 CIFAR-10 classes. The
Grad-CAM box shows where the model looked, not one box per object. Images are shrunk to
32 x 32 first to match the training data (`--no-downscale` turns this off).

## Trained models

The checkpoints (`outputs/checkpoints/*.pt`, about 45-82 MB each) are too large for the
repository and are attached to the GitHub Release instead:
**[add the Release link here]**. Download them into `outputs/checkpoints/` to use
`predict_*.py`, `benchmark_deployment.py` or `make_figures.py` without retraining.

## Results (2,000 test images, one run per model, seed 42)

| Model | Test acc. (%) | Macro-F1 | Training time (min) |
|---|---|---|---|
| HOG + RBF-SVM | 54.10 | 0.541 | 0.1 |
| ResNet-18 | 91.85 | 0.918 | 5.1 |
| A: EffNetV2-S, fixed 224 px | 95.35 | 0.953 | 18.1 |
| B0: progressive size only | 95.05 | 0.951 | 12.4 |
| B: progressive (paper) | 95.50 | 0.955 | 12.9 |
| C: B + Mixup ramp | 95.35 | 0.953 | 13.0 |

No pair among A, B0, B and C differs significantly (McNemar's test, all p > 0.3); every
EfficientNetV2-S variant beats ResNet-18 (p < 1e-6). Progressive learning cut training time for
the same 12 epochs by 29%. Full numbers: `outputs/test_results.json`,
`outputs/mcnemar_tests.json`, `outputs/results_table.md`.

## Repository layout

```
comp1818_efficientnetv2_cifar10.py   training pipeline, splits, single test evaluation, McNemar tests
benchmark_deployment.py              deployment cost benchmark (random inputs, no test images)
make_figures.py                      report figures and one-vs-rest AP / mAP
predict_image.py, predict_video.py   demo: classify your own pictures / video
environment.yml                      conda environment
outputs/
  splits.json                        exact train / val / test indices
  config.json, selected_lr.json      settings used and LR sweep result
  runs/<run>/                        per-epoch history.json, summary.json, curves.png
  test_results.json                  accuracy, macro-F1, per-class scores, confusion matrices
  mcnemar_tests.json                 pairwise significance tests
  deployment_benchmark_run*.json     deployment benchmark (two runs)
  figures/                           confusion matrices, validation curves, report figures
```

## Reference

Mingxing Tan and Quoc V. Le. EfficientNetV2: Smaller Models and Faster Training. In
*Proceedings of the 38th International Conference on Machine Learning (ICML)*, PMLR 139,
pp. 10096-10106, 2021.
