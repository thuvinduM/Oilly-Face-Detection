# Oilly-Face-Detection

## Overview
This project implements an **oil detection system** using **Deep Learning** with **ResNet-50**. The model classifies images into three categories:
- **Dry**
- **Normal**
- **Oily**

The system is built using **PyTorch** and utilizes **transfer learning** to achieve high accuracy. It processes images, trains a neural network, and evaluates its performance using **accuracy, confusion matrix, and classification reports**.

---

## Features
- ✅ **Custom dataset loader** to handle image processing
- ✅ **Pre-trained ResNet-50** for fast and accurate classification
- ✅ **Image transformations and normalization** for better model performance
- ✅ **Train-test split and DataLoader** for batch processing
- ✅ **Performance evaluation using accuracy, confusion matrix, and classification report**

---
Dataset - https://www.kaggle.com/datasets/shakyadissanayake/oily-dry-and-normal-skin-types-dataset
---

## Installation
To set up the environment and run the project, install the necessary dependencies.

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/your-username/oil-detection.git
cd oil-detection
```

### **Step 2: Install Required Libraries**
```bash
pip install -r requirements.txt
```

Alternatively, install dependencies manually:
```bash
pip install torch torchvision numpy pandas matplotlib seaborn opencv-python scikit-learn pillow
```

### **Step 3: Run the Notebook**
Open the Jupyter Notebook and execute the cells to train and evaluate the model.
```bash
jupyter notebook oil_detection.ipynb
```

---

## Dataset Structure
The dataset should be organized in the following format:
```
/dataset_directory/
    dry/
        image1.jpg
        image2.jpg
        ...
    normal/
        image1.jpg
        image2.jpg
        ...
    oily/
        image1.jpg
        image2.jpg
        ...
```
The dataset directory should contain **three folders** (`dry`, `normal`, `oily`), each containing relevant images.

---

## Model Architecture
The model is based on **ResNet-50**, a pre-trained convolutional neural network. The final fully connected layer is modified to classify three categories:
```python
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, 3)  # Modify final layer for 3 classes
```

### **Training Process**
The model is trained using:
- **CrossEntropyLoss** for multi-class classification
- **Adam optimizer** with a learning rate of `0.001`
- **Mini-batch training** using `DataLoader`

Training loop:
```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")
```

### **Evaluation Metrics**
After training, the model is tested, and performance is measured using:
- **Accuracy**
- **Confusion Matrix**
- **Classification Report**

Evaluation code:
```python
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy:.2f}")
print(confusion_matrix(all_labels, all_preds))
print(classification_report(all_labels, all_preds, target_names=["dry", "normal", "oily"]))
```

---

## Results & Performance
The model achieves **high accuracy** in classifying oil patterns. A sample confusion matrix helps visualize performance:
```
               Predicted
               Dry  Normal  Oily
Actual  Dry       50     2       3
        Normal    1     48       5
        Oily      4      3      47
```
A detailed classification report provides precision, recall, and F1-score for each category.

---

## Future Improvements
🚀 Possible enhancements for better performance:
- **Data Augmentation** to increase dataset size
- **Hyperparameter Tuning** for better optimization
- **Use of EfficientNet** for more accurate classification


![Screenshot 2025-03-16 202059](https://github.com/user-attachments/assets/2e2a0810-f338-4006-8b2d-20a0b8a55821)
