#  COVID-19 Chest X-ray Classification using Deep Learning (PyTorch)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Utilities-yellow)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green)
![Status](https://img.shields.io/badge/Status-In%20Progress-brightgreen)

---

###  Project Overview
This project builds a **Convolutional Neural Network (CNN)** to detect COVID-19 cases from chest X-ray images. The goal of this project is to develop a binary image classifier that distinguishes **COVID-positive** vs. **non-COVID** chest X-rays. Using a dataset provided by the **University of Minnesota’s MCFAM**, the model performs **binary classification** between COVID-positive and non-COVID cases.  Convolutional Neural Networks (CNNs) was used for feature extraction, batch normalization for stability, and dropout regularization to prevent overfitting.  CNN 

This project was completed as part of my **Machine Learning repository** to apply the techniques I learned in the 2025 Advanced Machine Learning and Artificial Intelligence Summer Camp at the University of Minnesota
This repository contains my end-to-end machine learning project that classifies chest X-ray images to detect COVID-19 infection. The project was inspired by real-world clinical challenges and uses an open dataset from the **University of Minnesota MCFAM Lab**, with PyTorch as the primary framework.

---


##  Core Concepts
- Image preprocessing & data normalization  
- CNN architecture for medical imaging  
- Binary classification & evaluation metrics  
- Learning-rate scheduling and dropout  
- Model persistence using PyTorch `.pth` format  

---

##  Technologies Used
- **Languages:** Python  
- **Libraries:** PyTorch, NumPy, scikit-learn, urllib  
- **Tools:** Jupyter Notebook / VS Code  
- **Dataset Source:** UMN MCFAM Lab – 64×64 grayscale X-ray dataset  

---

##  Pipeline Summary

1. **Data Acquisition & Loading**  
   Downloaded `.npz` datasets containing 64×64 grayscale images and labels.  

   ```python
   urllib.request.urlretrieve('http://www-users.math.umn.edu/~jwcalder/MCFAM/train_images64.npz', 'train_images64.npz')
   urllib.request.urlretrieve('http://www-users.math.umn.edu/~jwcalder/MCFAM/train_labels.npz', 'train_labels.npz')

   images = np.load('train_images64.npz')['train_images']
   labels = np.load('train_labels.npz')['train_labels']

class CustomNumpyDataset(Dataset):
    def __init__(self, data_array, labels_array):
        self.data = torch.from_numpy(data_array).float()
        self.labels = torch.from_numpy(labels_array).long()

    def __getitem__(self, idx):
        sample = self.data[idx].unsqueeze(0)
        label_idx = torch.argmax(self.labels[idx]).item()
        binary_label = 1 if label_idx in [2, 3] else 0  # 1 = COVID, 0 = No COVID
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


| Metric             | Result                                         |
| :----------------- | :--------------------------------------------- |
| **Accuracy**       | 92 – 93 % on test set                          |
| **Loss Function**  | Negative Log Likelihood Loss                   |
| **Optimizer**      | Adam ( lr = 0.001 )                            |
| **Regularization** | Batch Normalization + Dropout (p = 0.7)        |
| **Scheduler**      | ReduceLROnPlateau (patience = 2, factor = 0.6) |

---

📈 **Visualizations (Planned)**

Confusion matrix and ROC curve plots

Grad-CAM visual heatmaps highlighting affected regions

Model accuracy vs epoch line chart

---

🧩 **Future Enhancements**

Extend model to multi-class classification for other respiratory diseases (e.g., pneumonia, bronchitis, asthma).

Apply transfer learning using pretrained CNN architectures (ResNet-18, DenseNet).

Build a Streamlit or Flask dashboard for real-time inference.

Integrate explainability using Grad-CAM or LIME.

---

💡 **Key Learnings**

Developed strong understanding of CNN architecture and image processing.

Learned efficient dataset handling and label conversion in PyTorch.

Understood the role of batch normalization and dropout in training stability.

Practiced ethical use of medical data and clear model interpretation.

---

📚 **References**

Dataset: University of Minnesota MCFAM Lab Public Dataset

Libraries: PyTorch 1.10+, NumPy, scikit-learn

Hardware: GPU (CUDA enabled) and CPU compatibility

---

🏅 **Author**

Saanvi R
Wayzata High School (Class of 2029) | MN, USA
Aspiring AI researcher focusing on applied machine learning and medical imaging.
