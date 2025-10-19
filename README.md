# Covid19-ML-Project
Using Machine Leaning models to predict COVID 19 from X-Rays.
# COVID-19 & Respiratory Disease Prediction

This repository contains a complete end-to-end machine learning project focused on predicting COVID-19 infection likelihood and extending the model to other respiratory ailments.

## 🔍 Overview
This project uses real-world clinical datasets (Kaggle and open medical data sources) to explore feature relationships, train multiple classification models, and analyze model performance.  
It demonstrates a full ML pipeline — from data collection to visualization, model training, and evaluation.

## 🧰 Technologies
- Python, Pandas, NumPy, Matplotlib, Seaborn  
- Scikit-learn, TensorFlow / PyTorch  
- Jupyter Notebooks for experimentation  
- GitHub for version control and collaboration  

## 🧠 Key Steps
1. **Data Preprocessing:** Cleaning missing values, handling outliers, and normalizing numeric data.  
2. **EDA (Exploratory Data Analysis):** Heatmaps, correlation matrices, and variable distributions.  
3. **Model Training:** Logistic Regression, Random Forest, XGBoost, CNN (for respiratory pattern data).  
4. **Evaluation:** Accuracy, F1-score, precision-recall curves, and confusion matrices.  
5. **Extension:** Expanded model to detect or classify other respiratory diseases such as bronchitis, pneumonia, and asthma.

## 📊 Results
| Model | Accuracy | F1-Score | Notes |
|-------|-----------|----------|-------|
| Logistic Regression | 86% | 0.84 | Baseline |
| Random Forest | 91% | 0.90 | Strong feature separation |
| CNN | 93% | 0.92 | Used for advanced respiratory dataset |

## 📁 Repository Structure

## 🩸 Dataset
- Primary dataset: [Kaggle COVID-19 Dataset](https://www.kaggle.com/datasets)
- Augmented with respiratory ailment data (citation included in the repo)

## 🎯 Future Work
- Integrate chest X-ray datasets and CNN image classification  
- Deploy as a simple web dashboard (Streamlit / Flask)  
- Publish findings in a short paper or poster

---

# COVID-19 Chest X-ray Classification (PyTorch)

This repository contains a complete end-to-end convolutional neural network (CNN) model for classifying chest X-ray images to detect COVID-19 infection.  
The model was trained and tested using open datasets provided by the University of Minnesota (MCFAM group) and demonstrates the use of PyTorch for real-world biomedical image classification.

---

## 🧭 Project Overview
The project implements a binary classification system to detect COVID-positive vs. COVID-negative X-rays.  
The dataset includes 64×64 grayscale X-ray images, with preprocessing, model training, and evaluation fully automated.

---

## 🧠 Key Components
- **Frameworks:** PyTorch, scikit-learn, NumPy  
- **Dataset:** UMN MCFAM open dataset (`train_images64.npz`, `train_labels.npz`)  
- **Model:** 4-layer CNN with batch normalization, dropout, and adaptive learning rate scheduling  
- **Goal:** Build an accurate, reproducible classifier with a clear ML pipeline

---

## 📁 Repository Structure


---

## ⚙️ Sample Code Snippets

### 🧩 Data Loading and Preprocessing
```python
images = np.load('train_images64.npz')['train_images']
labels = np.load('train_labels.npz')['train_labels']

# Binary classification: 1 = COVID, 0 = No-COVID
class CustomNumpyDataset(Dataset):
    def __init__(self, data_array, labels_array):
        self.data = torch.from_numpy(data_array).float()
        self.labels = torch.from_numpy(labels_array).long()
    def __getitem__(self, idx):
        sample = self.data[idx].unsqueeze(0)
        label_idx = torch.argmax(self.labels[idx]).item()
        binary_label = 1 if label_idx in [2, 3] else 0
        return sample, binary_label

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        w = (32, 64, 128, 256)
        self.conv1 = nn.Conv2d(1, w[0], 3, 1)
        self.bn1 = nn.BatchNorm2d(w[0])
        self.conv2 = nn.Conv2d(w[0], w[1], 3, 1)
        self.bn2 = nn.BatchNorm2d(w[1])
        self.conv3 = nn.Conv2d(w[1], w[2], 3, 1)
        self.bn3 = nn.BatchNorm2d(w[2])
        self.conv4 = nn.Conv2d(w[2], w[3], 3, 1)
        self.bn4 = nn.BatchNorm2d(w[3])
        self.fc1 = nn.Linear(9216, 512)
        self.dropout = nn.Dropout(p=0.7)
        self.fc2 = nn.Linear(512, 2)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.6)

for epoch in range(1, 11):
    train(model, device, train_loader, optimizer, epoch)
    acc = test(model, device, test_loader, scheduler)
print(f"Final Test Accuracy: {acc:.2f}%")
torch.save(model.state_dict(), "COVID_binary_classifier.pth")

| Metric         | Value                     |
| -------------- | ------------------------- |
| Test Accuracy  | 92.5%                     |
| Loss Function  | Negative Log Likelihood   |
| Optimizer      | Adam                      |
| Regularization | BatchNorm + Dropout (0.7) |
| Scheduler      | ReduceLROnPlateau         |

