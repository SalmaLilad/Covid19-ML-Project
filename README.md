#  COVID-19 Chest X-ray Classifier (PyTorch)

This repository implements a **deep convolutional neural network (CNN)** that classifies **chest X-ray images** as either *COVID-positive* or *Non-COVID*.  
It was developed to explore the role of **AI in healthcare diagnostics**, focusing on model interpretability, data preprocessing, and generalization through batch normalization and dropout regularization.

This Project was built as part of the University of Minnesota 2025 Advanced AI and Machine Learning Summer Camp **Capstone Project**, it explores predictive capbability of COVID 19 using chest Xray images.


---

##  Overview

This project demonstrates the complete end-to-end process of developing a medical imaging classifier using **PyTorch**.  
It downloads and preprocesses datasets, builds a CNN from scratch, and uses adaptive learning-rate scheduling to improve convergence.  
The model is capable of achieving above 95% test accuracy on 64×64 X-ray images with minimal preprocessing.

### Key Highlights
-  Automatic dataset download  
-  4-layer convolutional architecture  
-  GPU support for accelerated training  
-  Adaptive learning rate scheduling with `ReduceLROnPlateau`  
-  Model saving and reloading for later inference

---

##  Dataset

This project uses open-access chest X-ray datasets hosted by the **University of Minnesota’s MCFAM repository**.  
Each image is a 64×64 grayscale scan, and labels are one-hot encoded across four categories, later collapsed into a binary classification.

| Original Label Index | Binary Label | Meaning |
|-----------------------|--------------|----------|
| 0, 1 | 0 | Non-COVID |
| 2, 3 | 1 | COVID |

### Data Download and Structure
The dataset is fetched automatically:
```python
urllib.request.urlretrieve('http://www-users.math.umn.edu/~jwcalder/MCFAM/train_images64.npz','train_images64.npz')
urllib.request.urlretrieve('http://www-users.math.umn.edu/~jwcalder/MCFAM/train_labels.npz','train_labels.npz')
```

### Output
```
Image shape: (10000, 64, 64)
Label shape: (10000, 4)
```
Each image is paired with a one-hot vector label and later transformed into binary form for classification.

---

##  Model Architecture

The classifier uses a **4-block convolutional neural network** that progressively extracts spatial patterns and texture gradients from X-ray data.  
Each layer includes **batch normalization** (for stability) and **ReLU activation** (for non-linearity), followed by **max pooling** to reduce dimensionality.  

| Layer | Description | Purpose |
|-------|--------------|----------|
| Conv1 | 1→32 filters → BatchNorm → ReLU → MaxPool | Extracts low-level features (edges, contours) |
| Conv2 | 32→64 filters → BatchNorm → ReLU → MaxPool | Captures mid-level textures |
| Conv3 | 64→128 filters → BatchNorm → ReLU → MaxPool | Identifies structural lung patterns |
| Conv4 | 128→256 filters → BatchNorm → ReLU → MaxPool | Detects high-level features for decision-making |
| FC1 | 512 neurons + Dropout(0.7) | Prevents overfitting, learns compressed representations |
| FC2 | 2 outputs → LogSoftmax | Outputs class probabilities for COVID / Non-COVID |

**Loss:** Negative Log-Likelihood (`F.nll_loss`)  
**Optimizer:** Adam (`lr=0.001`)  
**Scheduler:** ReduceLROnPlateau (`patience=2`, `factor=0.6`)

### Output During Model Setup
```
Image shape: torch.Size([1, 1, 64, 64])
Flattened feature size: 6400
Model successfully initialized.
```

---

##  Training

The training process is handled through custom `train()` and `test()` functions.  
The model iteratively optimizes its parameters by minimizing cross-entropy loss and adapts its learning rate based on validation performance.  
The dataset is automatically split into 70% training and 30% testing subsets for evaluation.

### Run Command
```bash
python covid_classifier.py
```

### Hyperparameters
```python
batch_size = 100
epochs = 10
test_size = 0.3
```

### Training Output Example
```
Train Epoch: 1 [0/7000 (0%)]	Loss: 0.693421
Train Epoch: 1 [2000/7000 (28%)]	Loss: 0.310529
Train Epoch: 1 [4000/7000 (57%)]	Loss: 0.221917
Test set: Average loss: 0.2123, Accuracy: 95.73% [COVID Detection]
```

**Interpretation:**  
As epochs progress, training loss steadily decreases while test accuracy rises, typically stabilizing between **93–97%**.

---

##  Results and Evaluation

After 10 epochs, the model achieves strong binary classification accuracy and generalization performance.  
Performance may vary depending on system hardware, random seed, and number of epochs.

| Metric | Typical Result |
|---------|----------------|
| Accuracy | 93–97% |
| Loss | 0.18–0.25 |
| Inference Speed | < 5 ms per image (GPU) |
| Learning Rate Adaptation | Automatic via scheduler |

### Sample Evaluation Output
```
Epoch 10/10
Train Epoch: 10 [6400/7000 (91%)]	Loss: 0.085214
Test set: Average loss: 0.1829, Accuracy: 96.87% [COVID Detection]
Final Test Accuracy: 96.87%
```

**Observation:**  
The model demonstrates excellent convergence with consistent validation accuracy and minimal overfitting due to dropout and normalization.

---

##  Model Saving and Reloading

After training, the model is automatically serialized to disk for reuse and deployment.  
This ensures reproducibility and allows continued inference without retraining.

### Code Snippet
```python
torch.save(model.state_dict(), "COVID_binary_classifier.pth")
```

### Reload for Inference
```python
model = Net()
model.load_state_dict(torch.load("COVID_binary_classifier.pth"))
model.eval()
```

### Output
```
Model weights loaded successfully.
Ready for COVID detection on new data.
```

---

##  Requirements and Environment Setup

The model runs efficiently on both **CPU and GPU** using Python 3.9+ and PyTorch ≥ 2.0.  
Dependencies can be installed with pip:

```bash
pip install torch torchvision scikit-learn numpy
```

Optional CUDA support (for NVIDIA GPUs):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Recommended Specs
- CPU: Intel i5/i7 or better  
- GPU: NVIDIA RTX or A100 (optional)  
- RAM: ≥ 8 GB  

---

##  File Structure

The repository follows a simple, modular layout to make navigation and experimentation straightforward.

```
├── covid_classifier.py          # Main training/testing script
├── train_images64.npz           # Downloaded X-ray images
├── train_labels.npz             # Corresponding labels
├── COVID_binary_classifier.pth  # Saved model weights
└── README.md                    # Project documentation
```

Each component can be replaced or extended—for example, substituting datasets or experimenting with CNN architectures.

---

##  Example Output Summary

Below is a sample summary of the full training cycle on GPU:

```
Image shape: (10000, 64, 64)
Label shape: (10000, 4)
Train Epoch: 1 [0/7000 (0%)]	Loss: 0.693421
Test set: Average loss: 0.2123, Accuracy: 95.73% [COVID Detection]
Epoch 10/10
Final Test Accuracy: 96.87%
Model saved to COVID_binary_classifier.pth
```

**Summary Insight:**  
The model achieves stable convergence after ~8–10 epochs and remains robust on unseen data, showing the effectiveness of regularization and balanced data sampling.

---

##  Code Components Overview

Each major function serves a distinct purpose in the training pipeline.

| Function | Purpose |
|-----------|----------|
| `CustomNumpyDataset` | Converts `.npz` arrays into PyTorch tensors and maps multi-class labels to binary form |
| `train()` | Executes forward/backward passes and updates network weights |
| `test()` | Evaluates performance and reports loss/accuracy metrics |
| `Net` | Defines CNN layers, feature extraction, and forward propagation |
| `ReduceLROnPlateau` | Dynamically adjusts learning rate when validation loss plateaus |

### Output Sample
```
Using device: cuda
Learning rate adjusted: 0.001 → 0.0006 after plateau
```

---

##  Author

**Saanvi ([@SalmaLilad](https://github.com/SalmaLilad))**  
Applied AI & Computing Portfolio — Machine Learning Project  
Developed to explore **AI’s role in real-world diagnostic modeling**, with a focus on accuracy, ethics, and reproducibility.  
Built with **PyTorch**, **NumPy**, and **scikit-learn**.

---

##  AI Assistance Disclosure

This README was prepared with the assistance of an AI agent to enhance structure, clarity, and technical documentation quality.  
All project code, architecture, and experimental results were implemented, and verified by me without any AI use. I provided the outline, structure and content for this ReadME and the AI assistant was used exclusively for documentation, drafting, formatting, and summarization purposes.

---

## ⚠️ License & Disclaimer

This repository is for **educational and research use only**.  
It is **not intended for clinical or diagnostic purposes**, and results should not be interpreted as medical advice.  
Researchers and students are encouraged to **fork, extend, or analyze** this work for non-commercial academic use.

---
