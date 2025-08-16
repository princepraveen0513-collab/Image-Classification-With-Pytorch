# Image Classification with PyTorch — MLP Variants (Dropout & BatchNorm)
**Notebook:** `ImageClassificationwithPytorch.ipynb`  
~ By Guardians of the Galaxy

This project trains several **fully-connected (MLP)** image classifiers in PyTorch on a custom **10‑class** dataset. It includes data normalization computed from the training set, multiple model variants (**Baseline**, **Dropout**, **BatchNorm**, **Dropout+BatchNorm**), early stopping, checkpointing, and end‑to‑end evaluation.

---

## 🎯 Objectives
- Load and prepare an image dataset from a ZIP archive, compute **dataset mean/std** for normalization.
- Implement and compare multiple **MLP architectures** (no convolutions) with **Dropout** and **BatchNorm** variants.
- Train on GPU if available; track **loss** per epoch and evaluate on a held‑out test set.
- Report **accuracy** (and precision/recall where computed) and save the best model checkpoint.

---

## 🧪 Data
- **Size:** ~**4,844** images (from notebook output).  
- **Classes:** **10** (IDs 0–9; class distribution printed in notebook).  
- **Normalization:** computed mean/std used in `transforms.Normalize(...)`:
  - Mean: `tensor([0.6763, 0.5651, 0.4429])`
  - Std: `tensor([0.2441, 0.2925, 0.3333])`
- **Transforms:** `Resize` → `ToTensor` → `Normalize` (no heavy augmentation in the current run).

> The notebook unzips the dataset and expects a directory layout suitable for loading images (custom loader, not `ImageFolder`), then builds PyTorch `DataLoader`s.

---

## 🧠 Models
Custom `nn.Module` variants implemented:
- **BaselineNN** — plain MLP
- **Model1_Dropout** — adds `nn.Dropout`
- **Model3_BatchNorm** / **BatchNormNN** — adds `nn.BatchNorm1d`
- **Model4_Dropout_L2** — combined Dropout + (L2 if enabled in loss/optimizer)

> Layers used: `nn.Linear`, `nn.ReLU`, optional `nn.Dropout`, `nn.BatchNorm1d`  
> **No** `nn.Conv2d` is used; images are flattened into vectors.

---

## ⚙️ Training Setup
- **Device:** CUDA used when available
- **Loss:** `nn.CrossEntropyLoss()`
- **Optimizer:** `Adam (lr=0.001)`
- **Batch size:** `32` (with `num_workers=2`)
- **Epochs:** set to `10` (early stopping triggered around epoch 9 in one run)
- **Checkpointing:** best model saved via `torch.save(...)`
- **(Optional)** Early stopping logic present (triggered in runs shown)

---

## 📈 Results (from notebook run)
Best observed metrics from outputs:
- **Test Accuracy:** **0.999**
- **Test Precision:** **0.999**
- **Test Recall:** **0.999**

Training/validation losses are printed per epoch (see notebook for full logs). Results vary by model variant; BatchNorm/Dropout variants tend to converge faster and achieve the top scores in the provided runs.


---

## 📊 Visualizations & Reports
- **Per‑epoch**: Train/Test (validation) **loss**
- (Recommended) Add **confusion matrix** and `classification_report` to analyze per‑class performance; hooks exist for accuracy/precision/recall in code.

---

## 📁 Repository Structure
```text
├── ImageClassificationwithPytorch.ipynb   # Main notebook
└── README.md         # This file
```

If you export artifacts (best model) or figures, consider adding:
```text
├── checkpoints/      # Saved .pt/.pth files
└── images/           # Saved plots (loss curves, confusion matrix)
```

---

## 🚀 Setup & Run
**Dependencies:** `torch`, `torchvision`, `pillow`, `scikit-learn`, `matplotlib`, `pandas`

Install:
```bash
pip install -U torch torchvision torchaudio pillow scikit-learn matplotlib pandas
```

Run:
```bash
jupyter notebook "ImageClassificationwithPytorch.ipynb"
```

> Place (or unzip) your dataset where the notebook expects it. Update the data path variables in the first cells if needed.

---

## 💡 Next Steps
- **Data Augmentation:** add `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`, or `RandomResizedCrop` to improve robustness.
- **Schedulers:** try `StepLR`, `CosineAnnealingLR`, or `OneCycleLR` for better convergence.
- **Regularization:** confirm **L2** via `weight_decay` or explicit penalty in the loss; experiment with dropout rates.
- **Modeling:** compare against a small **CNN** or **pretrained** backbones (e.g., ResNet18) to benchmark MLPs.
- **Evaluation:** log a **confusion matrix**, **per‑class precision/recall/F1**, and **ROC/PR** (one‑vs‑rest) for deeper insight.
