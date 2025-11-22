# 🌸 KNN Iris Classification – Train & Predict  
A simple machine learning project using **K-Nearest Neighbors (KNN)** to classify Iris flower species.  
This project includes:

- Training a KNN model on the Iris dataset  
- Saving the trained model using Joblib  
- Loading the model to make new predictions  

Perfect for beginners learning classification and model deployment basics.

---

## 📂 Project Files
```bash
|-- iris.csv
|-- knn_iris.ipynb → Training + evaluation + saving model
|-- using_model.ipynb → Loading model + predicting new samples
|-- knn_iris_model.sav → Saved trained model
```
## 🔧 Requirements

Install the required packages:

```bash
pip install pandas scikit-learn joblib
```
📘 1. Training the Model (knn_iris.ipynb)
👉 Load Dataset
```bash
import pandas as pd
dataset = pd.read_csv("iris.csv").values
```
👉 Separate Features & Labels
```bash
data = dataset[:, 0:4]      # Features
targets = dataset[:, 4]     # Labels
```

👉 Train/Test Split
```bash
from sklearn.model_selection import train_test_split
train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.2)
```

👉 Train KNN Model
```bash
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(train_data, train_targets)
```

👉 Evaluate Performance
```bash
from sklearn.metrics import accuracy_score
predicted_targets = model.predict(test_data)
acc = accuracy_score(test_targets, predicted_targets)
print("Test Accuracy:", acc)
```

👉 Save the Model
```bash
import joblib
joblib.dump(model, 'knn_iris_model.sav')
```

📗 2. Using the Saved Model (using_model.ipynb)
👉 Load the Model
```bash
import joblib
model = joblib.load('knn_iris_model.sav')
```

👉 Predict New Data
```bash
test_data = [[5.1, 3.5, 1.4, 0.2]]  # Sepal & petal measurements
result = model.predict(test_data)
print(result)
```

🎯 Output Example
```bash
['Iris-setosa'] - [0.]
```

🚀 Key Learnings

How to load and preprocess data

How KNN classifier works

Splitting data into training/testing

Measuring accuracy

Saving and loading machine learning models
