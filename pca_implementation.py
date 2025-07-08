import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# Plot the PCA data
def plot_graph(pca):
    # Plot the explained variance ratio
    print("Plotting PCA data...")
    plt.figure(figsize=(10,6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA - Explained Variance vs Number of Components')
    plt.grid(True)
    plt.show()

# Plot the PCA 2D data
def plot_pca_2d(pca_2d_data, target):
    print("Plotting PCA 2D data...")
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_2d_data[:, 0], pca_2d_data[:, 1], c=target, cmap='viridis', edgecolor='k', s=50) # Use 'cancer.target' to color the points
    plt.title('PCA 2D Projection of Breast Cancer Dataset') 
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cancer Type')
    plt.grid(True)
    plt.show()

# Reduce the dataset into 2 PCA components
def reduce_to_2d(pca_data):
    # Reduce PCA data to 2D
    pca_2d = PCA(n_components=2)
    pca_2d_data = pca_2d.fit_transform(pca_data)
    return pca_2d_data

# logistic regression for prediction
def logistic_regression_predict(pca_data, target):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(pca_data, target, test_size=0.2, random_state=42)
    # Create a logistic regression model
    model = LogisticRegression()
    # Fit the model on the training data
    model.fit(X_train, y_train)
    # Make predictions on the test set
    predictions = model.predict(X_test)
    # Print classification report and confusion matrix
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    # Load the breast cancer dataset
    cancer = load_breast_cancer()
    print(f"Shape of dataset: {cancer.data.shape}")
    print(f"Number of features: {cancer.data.shape[1]}")
    print(f"Number of samples: {cancer.data.shape[0]}")
    print(f"Target names: {cancer.target_names}")
    print(f"Feature names: {cancer.feature_names}")
    
    # Create a DataFrame from the dataset
    df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
    print(df.head())
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Perform PCA
    pca = PCA()
    pca_data = pca.fit_transform(scaled_data)
    print(f"Shape of PCA: {pca_data.shape}")

    # Call function to Plot the explained variance ratio
    plot_graph(pca)

    # Call function to reduce the dataset into 2 PCA components
    pca_2d_data = reduce_to_2d(pca_data)
    print(f"Shape of PCA 2D data: {pca_2d_data.shape}")
    print(f"Feature names: {pca_2d_data.shape[0]} samples, {pca_2d_data.shape[1]} features")

    # Plot the PCA 2D data
    plot_pca_2d(pca_2d_data, cancer.target)

    # Call function to perform logistic regression prediction
    logistic_regression_predict(pca_2d_data, cancer.target)
    