# Import necessary libraries
import numpy as np
import pandas as pd     # <-- NEW: import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset from CSV
data = pd.read_csv(r'C:\Users\Lenovo\Downloads\Iris.csv')  # <-- NEW: load CSV
# (r before string handles \ properly)

# Prepare features and labels
X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values  # Features
y = data['Species'].astype('category').cat.codes.values  # Labels (convert text labels to numbers)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create an SVM model (Support Vector Classifier)
svm = SVC(kernel='linear')

# Train the model
svm.fit(X_train, y_train)

# Predict on the test set
y_pred = svm.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Optional: Visualize the decision boundary (only using first two features)
X_2D = X[:, :2]
X_train_2D, X_test_2D, y_train_2D, y_test_2D = train_test_split(X_2D, y, test_size=0.3, random_state=42)

svm_2D = SVC(kernel='linear')
svm_2D.fit(X_train_2D, y_train_2D)

# Plot the decision boundaries
plt.figure(figsize=(10, 6))
h = .02
x_min, x_max = X_train_2D[:, 0].min() - 1, X_train_2D[:, 0].max() + 1
y_min, y_max = X_train_2D[:, 1].min() - 1, X_train_2D[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = svm_2D.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_train_2D[:, 0], X_train_2D[:, 1], c=y_train_2D, edgecolors='k', marker='o', s=100, cmap=plt.cm.Paired)
plt.title("SVM Decision Boundary (Linear Kernel) for 2D Iris Data (CSV Version)")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.show()
