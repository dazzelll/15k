# ForgeGate

Hackathon prototype for **robust detection of AI-generated images** after real-world post-processing (JPEG, blur, resize, noise, color jitter, crop).

The three-branch “RGB ViT + SRM ResNet + Canny ConvNeXt” stack is the right *forensics intuition* and the wrong *system* for this challenge. JPEG, blur, and thumbnail resize destroy the traces those extra backbones are built to see. ForgeGate keeps multi-cue detection, but:

1. **Semantic stream** — frozen CLIP ViT-B/16 (~86M, not trained). Distributional / photographic cues survive redistribution (UnivFD-style).
2. **Forensic stream** — ResNet-18 on high-pass + neighboring-pixel + **mid-band** spectrum maps, not raw SRM/PRNU.
3. **Degradation-aware gate** — cheap stats (blur, 8×8 blockiness, HF energy, noise) scale the forensic embedding; the gate is also supervised toward `1 − severity` from the known protocol transform.
4. **Protocol-matched training** — paired clean/transformed forwards with consistency loss + official transform table sampling.

Total parameters ≈ 97M (CLIP frozen). Trainable ≈ 12M. Under the **&lt;2B** cap.

Do **not** train on the organiser’s WildFake demonstration split (COCO val2017 + DALL·E Advanced). Use it only for demos.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

First CLIP load downloads OpenAI ViT-B/16 weights via `open_clip`.

## Data layout

Any tree with folder names that include `real` / `fake` (also `REAL`/`FAKE`, `aigc`, `authentic`):

```
data/train/real/*.jpg
data/train/fake/*.jpg
data/val/real/*.jpg
data/val/fake/*.jpg
```

Licensed public sources (train on these, not the demo split):

- [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) — small, 32×32; good for a pipeline dry-run (images are upsampled to 224). Copy `env.example` to `.env` and set `KAGGLE_USERNAME` / `KAGGLE_KEY` from Kaggle → Account → Create New Token (`kaggle.json`).
- [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set) — diverse synthetic and tampered images; includes real images from OpenImages V7.
- [WildFake](https://modelscope.cn/datasets/hy2628982280/WildFake/summary) — large-scale hierarchical dataset with state-of-the-art generators. **Excludes validation subset (COCO val2017 + DALL·E Advanced) from training data** as per hackathon rules.
- [AIGC-Detection-Benchmark](https://huggingface.co/datasets/TheKernel01/AIGC-Detection-Benchmark) — Apache 2.0 licensed benchmark with 17 different AI generators (ADM, BigGAN, CycleGAN, DALLE2, GauGAN, GLIDE, Midjourney, ProGAN, SD14, SD15, SDXL, StarGAN, StyleGAN, StyleGAN2, VQDM, WhichFaceIsReal, Wukong). **Use for testing cross-generator generalization** (not for training).

Download and prepare data:

```bash
# Download CIFAKE and SID_Set (default: 5000 images per class for hackathon-scale training)
python scripts/download_datasets.py

# Download all three datasets (CIFAKE + SID_Set + WildFake)
python scripts/download_datasets.py --wildfake

# Download and merge into single data/train directory
python scripts/download_datasets.py --wildfake --merge

# Custom sample size for faster training
python scripts/download_datasets.py --sample-size 2000 --wildfake --merge

# Only use CIFAKE (original behavior)
python scripts/download_datasets.py --no-sid

# Download AIGC-Detection-Benchmark for cross-generator testing (100 images per generator)
python scripts/download_datasets.py --aigc-benchmark --aigc-sample 100
```

The script downloads CIFAKE via KaggleHub, SID_Set via Hugging Face, WildFake via ModelScope, and AIGC-Detection-Benchmark via Hugging Face, then symlinks them to `data/` directories. `.env` and images stay out of git. `~/.kaggle/kaggle.json` still works if `.env` is missing.
- [WildFake](https://modelscope.cn/datasets/hy2628982280/WildFake/summary) — excluding the listed COCO val2017 / DALL·E Advanced demo subset.

## Train

```bash
python train.py --train_dir data/train --val_dir data/val --config configs/default.yaml
```

Checkpoints: `checkpoints/best.pt`, `checkpoints/last.pt`.

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

## Limitations / next

- Zero-shot CLIP is a baseline, not a detector. Train the gate + forensic head on SID_Set or WildFake (minus the demo split).
- CIFAKE 32×32 is not a fair generator-forensics test after upsampling.
- No generator-ID attribution; image-level only, as specified.
- Given more time: multi-layer CLIP patch features, test-time augmentation, and a second forensic view (NPR-only) for JPEG q=30.

## Team

Solo unless you add names here.
