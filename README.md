# ForgeGate

Hackathon prototype for **robust detection of AI-generated images** after real-world post-processing (JPEG, blur, resize, noise, color jitter, crop).

The three-branch “RGB ViT + SRM ResNet + Canny ConvNeXt” stack is the right *forensics intuition* and the wrong *system* for this challenge. JPEG, blur, and thumbnail resize destroy the traces those extra backbones are built to see. ForgeGate keeps multi-cue detection, but:

1. **Semantic stream** — frozen CLIP ViT-B/16 (~86M, not trained). Distributional / photographic cues survive redistribution (UnivFD-style).
2. **Forensic stream** — ResNet-18 on high-pass + neighboring-pixel + **mid-band** spectrum maps, not raw SRM/PRNU.
3. **Degradation-aware gate** — cheap stats (blur, 8×8 blockiness, HF energy, noise) scale the forensic embedding so the model does not rely on traces that are already gone.
4. **Protocol-matched training** — the official transform table is sampled during training. That is the main robustness lever.

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

- [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) — small, 32×32; good for a pipeline dry-run (images are upsampled to 224).
- [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set)
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

## Robustness table

```bash
python evaluate.py --data_dir data/val --checkpoint checkpoints/best.pt --output outputs/robustness.csv
```

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
- The gate is supervised only indirectly through detection loss; a small labelled-degradation auxiliary would make it more calibrated.
- No generator-ID attribution; image-level only, as specified.
- Given more time: test-time augmentation, temperature scaling on the protocol mix, and a second forensic view (NPR-only) for JPEG q=30.

## Team

Solo unless you add names here.
