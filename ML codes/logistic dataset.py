import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset from CSV
data = pd.read_csv(r'C:\Users\Lenovo\Downloads\Iris.csv')

# Prepare features and labels
X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y = data['Species'].astype('category').cat.codes.values  # Convert species names to numeric codes

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# Initialize and train the Logistic Regression model
clf = LogisticRegression(max_iter=1000, random_state=0, multi_class='ovr')
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluate the model
acc = accuracy_score(y_test, y_pred) * 100
print(f"Logistic Regression model accuracy (in %): {acc:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred))

# ------------------- Visualization 1: Sigmoid Function -------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 100)
sigmoid_values = sigmoid(z)

plt.figure(figsize=(6, 4))
plt.plot(z, sigmoid_values, label="Sigmoid Function", color="red")
plt.axhline(0.5, linestyle="dashed", color="black", alpha=0.5)
plt.xlabel("z")
plt.ylabel("Sigmoid(z)")
plt.title("Sigmoid Activation Function")
plt.legend()
plt.grid()
plt.show()

# ------------------- Visualization 2: Confusion Matrix -------------------
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=data['Species'].unique(), yticklabels=data['Species'].unique())
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
