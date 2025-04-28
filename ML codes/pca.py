import numpy as np
import matplotlib.pyplot as plt

# 1. Create or load data
# Let's create a simple 2D dataset for visualization
np.random.seed(42)
X = np.random.randn(100, 2)
X = X @ np.array([[3, 1], [1, 2]])  # Add some correlation

# 2. Standardize the data (zero mean)
X_meaned = X - np.mean(X, axis=0)

# 3. Calculate the covariance matrix
cov_mat = np.cov(X_meaned, rowvar=False)

# 4. Calculate eigenvalues and eigenvectors
eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

# 5. Sort eigenvalues and eigenvectors
sorted_index = np.argsort(eigen_values)[::-1]
eigen_values = eigen_values[sorted_index]
eigen_vectors = eigen_vectors[:, sorted_index]

# 6. Choose the number of principal components (let's pick 2 here)
n_components = 2
eigenvector_subset = eigen_vectors[:, 0:n_components]

# 7. Project data onto principal components
X_reduced = np.dot(X_meaned, eigenvector_subset)

# 8. Plot the reduced data
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.title('Data projected onto first 2 Principal Components')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid()
plt.show()
