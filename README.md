# An Empirical Study on the Impact and Mitigation of Missing Data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete source code and experimental pipeline for the paper, **"Quantifying the Void: An Empirical Study on the Impact and Mitigation of Missing Data."** The full paper can be found at [LINK TO YOUR ARXIV PAPER HERE].

## 📖 About The Project

This project systematically investigates one of the most common challenges in data science: handling missing data. The core of this research is a controlled experiment designed to answer two key questions:

1.  How much does predictive model performance degrade as the amount of missing data increases?
2.  How effective are different imputation techniques at restoring that performance, and does the best technique change depending on the problem?

### Key Findings

Our experiments, conducted across three diverse datasets, revealed that:
* The impact of missing data is highly contingent on the **complexity and informational redundancy** of the dataset.
* There is **no universally superior imputation technique**; the optimal strategy is context-dependent.
* On complex problems, sophisticated methods like **MICE** become necessary as data quality worsens.
* On simpler problems, methods like **Mean/Median** can be surprisingly effective, and complex methods may even harm performance.

---

## 🔬 Methodology Overview

The experimental workflow is designed to be fully reproducible:

1.  **Baseline Establishment:** An XGBoost model is trained and evaluated on three pristine datasets (Bank Marketing, California Housing, and Iris) to establish a "gold standard" performance baseline.
2.  **Systematic Degradation:** The training data for each dataset is intentionally degraded by randomly removing 5%, 15%, and 30% of the values.
3.  **Imputation & Repair:** Four common imputation techniques are applied to each degraded dataset:
    * Mean Imputation
    * Median Imputation
    * k-Nearest Neighbors (k-NN) Imputation
    * Multivariate Imputation by Chained Equations (MICE)
4.  **Re-evaluation:** The XGBoost model is retrained on each newly imputed dataset and evaluated against the original, untouched test set.
5.  **Analysis:** The final performance scores are compared to the baseline to measure the efficacy of each imputation method under different conditions.

---

## 📂 Repository Structure

The project is organized into a series of modular Python scripts designed to be run in sequence:

├── data/
│   ├── bank-full.csv
│   ├── housing.csv
│   └── Iris.csv
├── preprocess.py       # Loads raw data, applies scaling and encoding
├── split.py            # Splits processed data into train/test sets
├── train.py            # Trains the XGBoost model on a given training set
├── evaluate.py         # Evaluates the trained model on the test set
├── degrade_data.py     # Intentionally introduces missing values
├── impute_data.py      # Applies imputation techniques to fix missing values
└── requirements.txt    # Lists all necessary Python libraries


---

## 🚀 Getting Started

To replicate the experiments, follow the steps below.

### Prerequisites

* Python 3.8+
* The datasets located in the `/data` directory.

### Installation

1.  Clone the repository to your local machine:
    ```sh
    git clone [https://github.com/your-username/missing-data-imputation-study.git](https://github.com/your-username/missing-data-imputation-study.git)
    cd missing-data-imputation-study
    ```
2.  Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

### Running an Experiment

To run a full experiment for a single dataset (e.g., Bank Marketing):

1.  **Configure `preprocess.py`** to point to the correct dataset file and target column.
2.  **Run the baseline pipeline:**
    ```sh
    python preprocess.py
    python split.py
    python train.py
    python evaluate.py
    ```
    *Carefully record the baseline score.*
3.  **Run the degradation and imputation loop:**
    * Configure and run `degrade_data.py` for a specific degradation level (e.g., 5%).
    * Configure and run `impute_data.py` for a specific imputation method (e.g., 'mean').
    * Run `train.py` and `evaluate.py` again, making sure they use the newly imputed training file.
    * Record the new score and compare it to the baseline.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
