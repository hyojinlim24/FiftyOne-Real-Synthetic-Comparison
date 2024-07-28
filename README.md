## FiftyOne-Real-Synthetic-Comparison

### Description

This repository provides a comprehensive analysis of real and synthetic image data using the FiftyOne open-source tool. The project focuses on comparing datasets through keypoint analysis and image embeddings to evaluate differences and similarities. The analysis includes calculating metrics such as Frechet Inception Distance (FID) and Inception Score (IS) to quantify the quality of synthetic data.

### Features

- **Data Loading and Merging**: Load and merge real and synthetic datasets using FiftyOne.
- **Keypoint Analysis**: Extract and visualize keypoint data from images.
- **Image Embeddings**: Compute image embeddings using a pre-trained Inception v3 model.
- **Metric Calculation**: Calculate FID and Inception Score to compare real and synthetic images.

### Contents

- `notebooks/`: Jupyter notebooks for exploratory data analysis and image embedding computations.
- `src/`: Python scripts for modularized data loading, keypoint analysis, and image embeddings.
- `requirements.txt`: List of dependencies for the project.
- `README.md`: Project overview and setup instructions.

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/hyojinlim00/FiftyOne-Real-Synthetic-Comparison.git
   ```
2. Navigate to the project directory:
   ```bash
   cd FiftyOne-Real-Synthetic-Comparison
   ```
3. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venvScriptsactivate`
   ```
4. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

- To perform keypoint analysis, run the cells in `notebooks/eda.ipynb`.
- To compute image embeddings and evaluate metrics, run the cells in `notebooks/image_embeddings.ipynb`.

### Results

**Cluster Consistency**: Real data tends to have more consistent poses, while synthetic data includes more variability in the generation process.

**Accuracy and Precision**: The wide distribution in synthetic data suggests lower precision, potentially leading to lower accuracy.

**Data Quality**: There are numerous outliers in the synthetic data, indicating the need for preprocessing based on range criteria.

### Visualization of Real and Synthetic Data Features

**Datasets**:
- DIMS_REAL_ICMU_BODYKEYPOINT2D_002 (4,591 samples)
- DIMS_REAL_ICMU_SEATBELT_001 (19,684 samples)
- DIMS_SYN_ICMU_OBDBKPAGEOCL_000 (327,600 samples)
- Added "dataset_type" tag to each dataset.
- Merged the two datasets for unified management.

**Techniques Used**:
- PCA (Principal Component Analysis)
- t-SNE (t-distributed Stochastic Neighbor Embedding)
- UMAP (Uniform Manifold Approximation and Projection)

**Model**:
- Used CLIP (Contrastive Language–Image Pre-training) model with the ViT-B/32 (Vision Transformer) version.

**Results**:
- Visualization results indicate significant differences in the distribution of real and synthetic data, suggesting that synthetic data does not fully replicate some features of real data.
- Further analysis and experiments are needed to evaluate which features show differences and how these differences impact model performance.

**Reference Notes**:

- **t-SNE**: Preserves local structure in the data. It cannot interpret the distance between clusters at different ends of the plot reliably but can indicate similarity within clusters.
- **UMAP**: Claims to preserve both local and most of the global structure. It allows for interpreting both the distances between points and clusters. Both t-SNE and UMAP are stochastic and dependent on hyperparameters, meaning results can vary significantly between runs.

### Contact

For any questions or issues, please open an issue on GitHub.

