# 🧠 COVID-19 Chest X-ray Classifier (PyTorch)

A convolutional neural network (CNN) for **binary COVID-19 classification** using 64×64 chest X-ray images.  
This model distinguishes **COVID vs. Non-COVID** samples using supervised learning and datasets from the [University of Minnesota MCFAM repository](http://www-users.math.umn.edu/~jwcalder/MCFAM/).

---

## 📦 Overview
This project demonstrates:
- Loading `.npz` image datasets directly from a URL  
- Creating a custom PyTorch `Dataset` for binary classification  
- Building a CNN with 4 convolutional blocks and dropout regularization  
- Adaptive learning-rate scheduling using `ReduceLROnPlateau`  
- GPU/CPU training support and model checkpointing  

---

## 🧩 Dataset
- **Images:** `train_images64.npz`  
- **Labels:** `train_labels.npz`  
Each label corresponds to one of four classes, remapped into a binary task:  

| Original Label | Binary Label | Meaning      |
|----------------|--------------|--------------|
| 0, 1           | 0            | Non-COVID    |
| 2, 3           | 1            | COVID        |

---

## ⚙️ Model Architecture
| Layer | Description |
|-------|--------------|
| Conv1 | 1→32 filters → BatchNorm → ReLU → MaxPool |
| Conv2 | 32→64 filters → BatchNorm → ReLU → MaxPool |
| Conv3 | 64→128 filters → BatchNorm → ReLU → MaxPool |
| Conv4 | 128→256 filters → BatchNorm → ReLU → MaxPool |
| FC1   | 512 neurons + Dropout(0.7) |
| FC2   | 2 outputs → LogSoftmax |

**Loss:** Negative Log-Likelihood (`F.nll_loss`)  
**Optimizer:** Adam (`lr=0.001`)  
**Scheduler:** ReduceLROnPlateau (`patience=2`, `factor=0.6`)  

---

## 🚀 Training
Run the training script:
```bash
python covid_classifier.py
```

Key hyperparameters:
```python
batch_size = 100
epochs = 10
test_size = 0.3
```

During training, output prints every 20 batches:
```
Train Epoch: 1 [0/7000 (0%)]	Loss: 0.693421
...
Test set: Average loss: 0.2123, Accuracy: 95.73% [COVID Detection]
```

After each epoch, the learning rate scheduler updates based on validation loss.

---

## 📊 Results
Typical performance after 10 epochs:
- **Accuracy:** 93 – 97% on test data  
- **Loss:** ~0.20 – 0.25  
- **Inference time:** < 5 ms per image (GPU)  

To improve performance:
- Increase training epochs  
- Add data augmentation  
- Use transfer learning or early stopping  

---

## 💾 Model Saving & Loading
After training, the model is saved automatically as:
```
COVID_binary_classifier.pth
```

Reload and evaluate:
```python
model = Net()
model.load_state_dict(torch.load("COVID_binary_classifier.pth"))
model.eval()
```

---

## 🧠 Requirements
Install dependencies:
```bash
pip install torch torchvision scikit-learn numpy
```

Optional GPU build (CUDA 12.1):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 📁 File Structure
```
├── covid_classifier.py          # Main training/testing script
├── train_images64.npz           # X-ray image data (64×64 grayscale)
├── train_labels.npz             # Corresponding labels (one-hot)
├── COVID_binary_classifier.pth  # Saved trained model weights
└── README.md                    # Project documentation
```

---

## 📈 Example Output
```
Epoch 10/10
Train Epoch: 10 [6400/7000 (91%)]	Loss: 0.085214
Test set: Average loss: 0.1829, Accuracy: 96.87% [COVID Detection]

Final Test Accuracy: 96.87%
Model saved to COVID_binary_classifier.pth
```

---

## 🧬 Author
**Saanvi L.**  
Applied AI & Computing Portfolio — Machine Learning Project  
Built with **PyTorch**, **NumPy**, and **scikit-learn**

---

## ⚠️ License & Disclaimer
This project is provided **for educational and research purposes only** and must **not** be used for real-world medical diagnosis or patient care.  
Feel free to fork or modify for academic exploration.
