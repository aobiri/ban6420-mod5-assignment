# ban6420-mod5-assignment
Nexford - Programming in R and Python, Module 5 Assignment

## PCA Implementation

This Python script implements Principal Component Analysis (PCA). 

Principal Component Analysis (PCA) is a powerful technique for reducing dimensionality and identifying the most informative features (essential variables) in a dataset — especially useful for exploratory data analysis and modeling.

### Features
- Data normalization
- Covariance matrix computation
- Visualization of explained variance

### Key Functions:
- fit(data): Computes the principal components and the explained variance from the input data.
- transform(data): Projects the input data onto the principal component space.
- fit_transform(data): Combines the fit and transform operations for convenience.

### Usage:
1. Import the PCA class from this module.
2. Create an instance of the PCA class.
3. Call the fit method with your dataset to compute the principal components.
4. Use the transform method to reduce the dimensionality of your dataset.

### Requirements
- Python 3.x
- Pandas
- Matplotlib

```bash
pip install pandas
pip install numpy matplotlib
pip install scikit-learn
```

Then, run the script with your dataset:

```bash
python pca_implementation.py
```

### Step-by-Step Execution
1) Executing the script starts with the loading of the cancer data:
```bash
    from sklearn.datasets import load_breast_cancer

    cancer = load_breast_cancer()
```
2) Standardize the data to have zero mean and unit variance:
```bash
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
```
3) Perform PCA:
```bash
    from sklearn.decomposition import PCA

    pca = PCA()
    pca_data = pca.fit_transform(scaled_data)
```
4) Call the plot_graph(pca) function to plot the explained variance ratio

5) Call to the function below to function to reduce the dataset into 2 PCA components. This is to help focus on the essentials, and provides the following:
    - Simplification of the dataset by converting it into principal components — combinations of features that capture the most variance.
    - Helping to identify the most informative variables (e.g., tumor size, diagnosis type, insurance status, location)
     ```bash
        reduce_to_2d(pca_data)
     ```

6) Call to the function below to visually represent the PCA components in 2D - referrals with similar characteristics cluster together.
   This helps to visually detect:
    - Groups needing urgent care
    - High-priority referral sources
    - Potential inefficiencies or overlaps
    ```bash
        plot_pca_2d(pca_2d_data, cancer.target) 
    ```

    ### Example
    You may notice that late-stage cancer referrals group closely on one side of the PCA graph — helping you allocate urgent resources more effectively.

7) Call to the the function below to train a logistic regression model to predict, for example, whether a tumor may be malignant or benign.
    ```bash
        logistic_regression_predict(pca_2d_data, cancer.target)
    ```

### Author
Albert Obiri-Yeboah
