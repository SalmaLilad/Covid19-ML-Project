# 🧠 COVID-19 Chest X-ray Classifier (PyTorch)

A convolutional neural network (CNN) for **binary COVID-19 classification** using 64×64 chest X-ray images.  
The model distinguishes **COVID** vs **Non-COVID** samples using supervised learning on publicly available datasets from the [University of Minnesota](http://www-users.math.umn.edu/~jwcalder/MCFAM/).

---

## 📦 Overview

This project demonstrates:
- Loading `.npz` image and label data from remote sources  
- Custom PyTorch `Dataset` for binary classification  
- CNN with 4 convolutional blocks + dropout regularization  
- Adaptive learning-rate scheduling via `ReduceLROnPlateau`  
- GPU support and checkpoint saving  

---

## 🧩 Dataset

- **Images:** `train_images64.npz`  
- **Labels:** `train_labels.npz`  
Each label corresponds to one of four classes, but the dataset is collapsed into a binary task:
> **1 → COVID, 0 → No-COVID**  
> (Labels `[2,3]` → COVID, `[0,1]` → Non-COVID)

---

## ⚙️ Model Architecture

| Layer | Details |
|-------|----------|
| Conv1 | 1 → 32 filters, BatchNorm, ReLU, MaxPool |
| Conv2 | 32 → 64 filters, BatchNorm, ReLU, MaxPool |
| Conv3 | 64 → 128 filters, BatchNorm, ReLU, MaxPool |
| Conv4 | 128 → 256 filters, BatchNorm, ReLU, MaxPool |
| FC1   | 512 neurons + Dropout(0.7) |
| FC2   | 2 outputs → LogSoftmax |

Loss: **Negative Log-Likelihood**  
Optimizer: **Adam (lr = 0.001)**  
Scheduler: **ReduceLROnPlateau(patience=2, factor=0.6)**

---

## 🚀 Training

```bash
python covid_classifier.py
