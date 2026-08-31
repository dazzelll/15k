# ForgeGate

## Project Overview

ForgeGate is a robust AI-generated image detection system designed to maintain high accuracy even when images undergo real-world post-processing transformations (JPEG compression, blur, resizing, noise, color adjustments, cropping). The system addresses the critical challenge that many forensic traces are destroyed by common image manipulations.

### Architecture

ForgeGate uses a multi-cue detection architecture with three key components:

1. **Semantic Stream** — Frozen CLIP ViT-B/16 (~86M parameters) that leverages distributional and photographic cues which survive image redistribution (inspired by UnivFD).
2. **Forensic Stream** — ResNet-18 operating on high-pass, neighboring-pixel, and mid-band spectrum maps to detect manipulation artifacts.
3. **Degradation-Aware Gate** — A lightweight module that computes image statistics (blur, blockiness, high-frequency energy, noise) and dynamically scales the forensic embedding based on estimated degradation severity.

The system is trained with protocol-matched augmentation using paired clean/transformed images with consistency loss, ensuring the model learns to maintain robustness across various image transformations.

**Technical Specifications:**
- Total parameters: ~97M (CLIP frozen at ~86M)
- Trainable parameters: ~12M
- Model size: Well under the 2B parameter cap

**Key Innovation:** The degradation-aware gate mechanism allows the model to adaptively rely more on semantic features when forensic traces are likely degraded by transformations, and more on forensic features when images are clean.

**Important:** Do not train on the organiser's WildFake demonstration split (COCO val2017 + DALL·E Advanced). Use it only for demos.

## Setup and Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

First CLIP load downloads OpenAI ViT-B/16 weights via `open_clip`.

## Data layout

Any tree with folder names that include `real` / `fake` (also `REAL`/`FAKE`, `aigc`, `authentic`):

```
data/train/{REAL,FAKE}/   # CIFAKE + SID_Set + WildFake (sampled, balanced)
data/val/{REAL,FAKE}/     # held-out splits of those same sources (checkpoint selection)
data/aigc_benchmark/      # unseen-generator TEST only; never in train or val
```

Licensed public sources (train on these, not the demo split):

- [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) — small, 32×32; good for a pipeline dry-run (images are upsampled to 224). Copy `env.example` to `.env` and set `KAGGLE_USERNAME` / `KAGGLE_KEY` from Kaggle → Account → Create New Token (`kaggle.json`).
- [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set) — diverse synthetic and tampered images; includes real images from OpenImages V7.
- [WildFake](https://modelscope.cn/datasets/hy2628982280/WildFake/summary) — large-scale hierarchical dataset with state-of-the-art generators. **Excludes validation subset (COCO val2017 + DALL·E Advanced) from training data** as per hackathon rules.
- [AIGC-Detection-Benchmark](https://huggingface.co/datasets/TheKernel01/AIGC-Detection-Benchmark) — Apache 2.0 licensed benchmark with 17 different AI generators (ADM, BigGAN, CycleGAN, DALLE2, GauGAN, GLIDE, Midjourney, ProGAN, SD14, SD15, SDXL, StarGAN, StyleGAN, StyleGAN2, VQDM, WhichFaceIsReal, Wukong). **Held-out test for cross-generator generalization** (not train, not val).

Download and prepare data:

```bash
# Default: ~36k train (6k/class × 3 sources), in-distribution val, AIGC test folder
python scripts/download_datasets.py

# Skip WildFake (e.g. no ModelScope) or skip the AIGC test set
python scripts/download_datasets.py --no-wildfake
python scripts/download_datasets.py --no-aigc-benchmark

# CIFAKE only
python scripts/download_datasets.py --no-sid --no-wildfake --no-aigc-benchmark

# Rebuild sampled folders from scratch
python scripts/download_datasets.py --force
```

The script downloads CIFAKE via KaggleHub, SID_Set via Hugging Face, WildFake via ModelScope, and AIGC-Detection-Benchmark via Hugging Face, then writes `data/train`, `data/val`, and `data/aigc_benchmark`. `.env` and images stay out of git. `~/.kaggle/kaggle.json` still works if `.env` is missing.

## Steps to Reproduce Results

### 1. Download and Prepare Data

```bash
# Download full training dataset (CIFAKE + SID_Set + WildFake, excludes demo split)
python scripts/download_datasets.py

# For demo video only (WildFake validation subset)
python scripts/download_demo.py
```

### 2. Train the Model

```bash
# Full training (5 epochs by default)
python train.py --train_dir data/train --val_dir data/val --config configs/default.yaml
```

Checkpoints are saved to `checkpoints/best.pt` (best validation score) and `checkpoints/last.pt` (latest epoch).

### 3. Evaluate on Validation Set

```bash
# In-distribution evaluation with robustness table
python evaluate.py --data_dir data/val --checkpoint checkpoints/best.pt --output outputs/robustness.csv
```

### 4. Evaluate on Unseen Generators

```bash
# Cross-generator generalization test
python evaluate.py --data_dir data/aigc_benchmark --checkpoint checkpoints/best.pt --output outputs/aigc_test.csv
```

### 5. Run Inference on New Images

```bash
# Batch inference on image directory
python infer.py --input_dir path/to/images --output outputs/preds.json --checkpoint checkpoints/best.pt
```

### 6. Demo (Optional)

```bash
# Run demo app
python app.py --checkpoint checkpoints/best.pt
```

## Train

## Inference (required deliverable)

Takes an image directory, writes JSON `{image_path, pred}` where `pred` is **P(AIGC)** in `[0, 1]`.

```bash
python infer.py --input_dir path/to/images --output outputs/preds.json --checkpoint checkpoints/best.pt
```

Without a checkpoint, CLIP **zero-shot prompts** are used so the demo still runs:

```bash
python infer.py --input_dir path/to/images --output outputs/preds.json --zeroshot
```

## Robustness table + gate ablation

```bash
python evaluate.py --data_dir data/val --checkpoint checkpoints/best.pt --output outputs/robustness.csv
# Also writes outputs/gate_vs_severity.png and outputs/reliability_clean.png
# Automatically calculates and displays Final Score: 0.50 * AUC_clean + 0.50 * AUC_robust
# Generates key_conditions.csv with: Clean, JPEG q30, Blur σ=2, Crop 80%, Resize 50%, Unseen gen.

python evaluate.py --data_dir data/aigc_benchmark --checkpoint checkpoints/best.pt --output outputs/aigc_test.csv
# Unseen-generator test (do not use this folder for train.py --val_dir)

python evaluate.py --data_dir data/val --checkpoint checkpoints/best.pt --ablation --output outputs/ablation.csv
# Adds semantic-only (g=0) / forensic-always (g=1) / full ForgeGate rows + ablation_insight.json
```

Training uses paired clean/transformed forwards with consistency loss and an explicit gate regularizer toward `1 - severity`. `best.pt` is selected on **protocol-augmented val AUC** (same mix as training, frozen per image). Temperature scaling is fit on that mix after the best epoch.

## Error analysis

```bash
python src/error_analysis.py --data_dir data/val --checkpoint checkpoints/best.pt
```

## Demo (video)

```bash
python app.py --checkpoint checkpoints/best.pt
# or: python app.py --zeroshot
```

Upload an image, optionally apply an official transform, read P(AIGC) and the forensic gate.

## Tools / stack

- Python, PyTorch, torchvision, open_clip, Gradio, scikit-learn, pandas
- Editor: Cursor / VS Code; optional Colab for GPU training

## Limitations and Future Improvements

### Current Limitations

1. **Generator Generalization**: Performance drops significantly on unseen AI generators not present in training data (e.g., newer models like DALL-E 3, Midjourney v6)

2. **Cross-Dataset Performance**: Model trained on specific datasets may not generalize well to images from different domains or demographics

3. **Computational Requirements**: Training requires GPU acceleration; inference is optimized but still benefits from GPU for batch processing

4. **Severe Transformations**: Severe JPEG compression (quality < 30) and extreme blur (σ=2) remain challenging despite the degradation-aware gate

5. **No Generator Attribution**: System provides binary classification only, cannot identify which specific generator created an image

6. **CIFAKE Limitations**: CIFAKE 32×32 images are not a fair generator-forensics test after upsampling to 224×224

### Future Improvements

Given more time and resources, the following improvements would be pursued:

1. **Enhanced Training Data**: Incorporate more diverse datasets including newer generators and varied demographic content

2. **Multi-Scale Features**: Extract features from multiple CLIP layers to capture both fine-grained and high-level patterns

3. **Test-Time Augmentation**: Apply multiple transformations during inference and aggregate predictions for improved robustness

4. **Specialized JPEG Handling**: Add a dedicated forensic branch optimized for heavily compressed images (NPR-only view for JPEG q=30)

5. **Generator Identification**: Extend to multi-class classification to identify specific AI generators

6. **Edge Deployment**: Optimize for mobile/embedded deployment with model quantization and pruning

7. **Active Learning**: Implement continuous learning pipeline to adapt to new generators

## Team Contributions

**Team Members:**
- Dazzel — <http://github.com/dazzelll>
- Zhi Ling — <http://github.com/zhilingggg>

**Key Contributions:**
- Architecture design and implementation of ForgeGate model
- Development of degradation-aware gate mechanism
- Implementation of protocol-matched training pipeline
- Creation of robustness evaluation framework
- Development of inference and evaluation scripts
- Comprehensive documentation and reproducibility setup
