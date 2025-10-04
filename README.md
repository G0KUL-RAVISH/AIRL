## Q1 — Vision Transformer (ViT) on CIFAR-10

### Overview
Trains a **Vision Transformer (ViT)** from scratch on CIFAR-10 and reports test accuracy.

### How to Run in Google Colab
1. Open `q1.ipynb` in Colab
2. Navigate to **Runtime → Change runtime type → Hardware accelerator → GPU**
3. Run all cells top to bottom (CIFAR-10 downloads automatically)

### Best Model Configuration
```json
{
  "seed": 42,
  "input_size": 32,
  "patch_size": 4,
  "num_classes": 10,
  "val_split": 5000,
  "randaugment": {
    "enabled": true,
    "N": 2,
    "M": 10
  },
  "mixup": {
    "p": 0.5,
    "alpha": 0.2
  },
  "cutmix": {
    "p": 0.0,
    "alpha": 1.0
  },
  "label_smoothing": 0.1,
  "embed_dim": 384,
  "depth": 12,
  "num_heads": 6,
  "mlp_ratio": 4.0,
  "drop_path": 0.1,
  "dropout": 0.0,
  "epochs": 100,
  "batch_size_target": 512,
  "grad_accum_steps": 1,
  "optimizer": {
    "name": "AdamW",
    "lr": 0.0006,
    "weight_decay": 0.1,
    "betas": [0.9, 0.999],
    "eps": 1e-8
  },
  "scheduler": {
    "type": "cosine",
    "warmup_epochs": 10
  },
  "ema_decay": 0.2,
  "grad_clip": 1.0,
  "use_amp": true,
  "eval_every": 5
}
```

### Results

| Config | Input | Patch | Dim | Depth | Heads | DropPath | Epochs | Test Acc (%) |
|--------|-------|-------|-----|-------|-------|----------|--------|--------------|
| vit_cifar10_v3_20251004-101042 | 32×32 | 4 | 384 | 12 | 6 | 0.1 | 100 | **71.93** |

### Analysis
- **Patch Size**: Smaller patches (p=4 → 65 tokens incl. CLS) preserve CIFAR-10 detail better than p=8 (17 tokens)
- **Depth/Width**: Small model (dim=384, depth=12, heads=6) outperforms tiny (dim=192) on Colab
- **Regularization**: RandAug, MixUp, Label Smoothing, and DropPath≈0.1 improve generalization
- **Schedule**: AdamW + warmup→cosine is effective; EMA (0.9998) could improve by +0.1–0.3%
- **Throughput**: 224×/p=16 (197 tokens) ≈ 9× more attention compute than 32×/p=4 (65 tokens)

---

## Q2 — Text-Driven Image Segmentation using SAM 2

### Overview
Performs **text-driven image segmentation** using open-source models (Grounding DINO/OWL-ViT + CLIP + SAM 2).

### Pipeline Steps
1. **Load Image** – Input image from URL or upload widget
2. **Text Prompt** – User provides zero-shot text description of target object
3. **Grounding DINO / OWL-ViT** – Generates bounding boxes for described object (OWL-ViT as fallback)
4. **CLIP Re-Ranking** – Filters and selects box most semantically similar to prompt
5. **SAM-2** – Produces precise segmentation mask for selected region
6. **Visualization** – Overlays mask on original image and displays outputs side by side

### How to Run in Google Colab
1. Open `q2.ipynb` in Colab
2. Navigate to **Runtime → Change runtime type → Hardware accelerator → GPU**
3. Run all cells top to bottom
4. Provide image URL
5. Enter text prompt describing the object to segment

### Limitations
- Handles **only single images** (no video propagation in this version)
- May fail for **ambiguous prompts** or very small/occluded objects
- Works best with **clear, descriptive prompts** (e.g., "brown dog", "red car")
- Requires **GPU runtime** for efficient inference
- Output quality depends on accuracy of text-to-region grounding
