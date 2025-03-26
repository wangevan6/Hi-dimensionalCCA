# Hi-dimensionalCCA
This repository includes a few well-known SCCA methods that could be used for high-dimensional data analysis. 
# Sparse Canonical Correlation Analysis (SCCA) and Related Methods

Welcome to this GitHub repository! This collection of Colab notebooks brings together implementations and examples of Sparse Canonical Correlation Analysis (SCCA), Principal Component Analysis (PCA), and other related methods, translated or developed in Python. These notebooks are designed to help users understand and apply these techniques to both synthetic and real-world data.

Below is an overview of the contents of this repository, including the purpose of each notebook and any notable details.

## Repository Contents

### 1. `SCCA_R_to_Python`
- **Description**: This notebook translates the original SCCA (Sparse Canonical Correlation Analysis) code from R to Python. It provides a faithful reproduction of the R implementation, allowing Python users to leverage SCCA for their analyses.
- **Purpose**: To bridge the gap between R and Python communities by making SCCA accessible in a Python environment.
- **Key Features**: Includes detailed comments mapping the R code to Python equivalents.

### 2. `COLAR`
- **Description**: A working implementation of COLAR (Canonical Correlation Analysis with Orthogonal Constraints) and PAM (Penalized Matrix Decomposition) in Python.
- **Purpose**: Demonstrates how these methods can be applied to extract correlated components from datasets with specific constraints.
- **Key Features**: Fully functional code with examples to illustrate usage.

- ### 2. `PMD`
- **Description**: A working implementation of COLAR (Canonical Correlation Analysis with Orthogonal Constraints) and PAM (Penalized Matrix Decomposition) in Python.
- **Purpose**: Demonstrates how these methods can be applied to extract correlated components from datasets with specific constraints.
- **Key Features**: Fully functional code with examples to illustrate usage.

### 3. `l0_deep_CCA_hyperparametertunning.ipynb`
- **Description**: Implements the `l0_deep_CCA` method, an extension of Deep Canonical Correlation Analysis (CCA) with an L0 regularization approach.
- **Purpose**: Explores the application of L0 regularization in deep CCA to enforce sparsity.
- **Note**: The error in this implementation is currently very high, indicating potential issues with convergence or parameter tuning that may require further investigation.

### 4. `l0_deep_CCA_real_data.ipynb`
- **Description**: Applies the `l0_deep_CCA` method to a real-world dataset.
- **Purpose**: Provides a practical example of how `l0_deep_CCA` performs on actual data, despite its high error.
- **Key Features**: Includes data preprocessing steps and visualization of results.

### 6. `SCCA_real_data.ipynb`
- **Description**: Applies the Sparse Canonical Correlation Analysis (SCCA) method to a real-world dataset.
- **Purpose**: Showcases SCCA’s ability to identify sparse, interpretable correlations between two sets of variables in practice.
- **Key Features**: Contains preprocessing, model fitting, and result interpretation steps tailored to real data.

## Getting Started
1. **Prerequisites**: These notebooks are designed to run in Google Colab. Ensure you have a Google account and access to Colab.
2. **Usage**: Open each `.ipynb` file in Colab, follow the instructions within, and run the cells sequentially. Some notebooks may require uploading datasets or installing dependencies (e.g., NumPy, SciPy, PyTorch).
3. **Datasets**: Where real data is used, either example datasets are provided or instructions are given for obtaining suitable data.

## Notes
- The `l0_deep_CCA.ipynb` notebook currently exhibits high error rates. Users are encouraged to experiment with hyperparameters or report issues if they identify potential fixes.
- Contributions, bug reports, or suggestions are welcome! Please open an issue or submit a pull request.

## License
This repository is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute the code as needed.

## Acknowledgments
- The original SCCA implementation in R inspired the Python translation.
- Thanks to the open-source community for providing tools and libraries that made this work possible.
