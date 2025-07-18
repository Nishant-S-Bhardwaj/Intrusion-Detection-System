# Intrusion Detection System (IDS) using Machine Learning & Deep Learning
Intrusion Detection System using Machine Learning and Deep Learning across CIC-IDS2017, NSL-KDD, ToN-IoT, and UNSW-NB15 datasets.

A comprehensive Intrusion Detection System leveraging both classical Machine Learning and Deep Learning models across multiple cybersecurity benchmark datasets, including CIC-IDS2017, NSL-KDD, UNSW-NB15, and ToN-IoT.

> **Detects network attacks like DoS, Probe, R2L, U2R, Botnet, and more using supervised learning models.**

---

## 📖 Table of Contents
- [Project Overview](#project-overview)
- [Datasets Used](#datasets-used)
- [Project Structure](#project-structure)
- [Models & Techniques](#models--techniques)
- [Evaluation Metrics](#evaluation-metrics)
- [How to Run](#how-to-run)
- [Sample Results](#sample-results)
- [Technologies Used](#technologies-used)
- [Author](#author)
- [License](#license)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

---

## 📝 Project Overview
This project implements a robust Intrusion Detection System (IDS) that utilizes both classical machine learning and deep learning approaches to detect and classify various types of network intrusions. By leveraging multiple real-world and synthetic datasets, the system aims to provide a comparative analysis of model performance and feature selection techniques for cybersecurity applications.

---

## 🧪 Methodology

The methodology for this Intrusion Detection System project follows a systematic, end-to-end pipeline to ensure robust and reproducible results across multiple datasets and models. The main steps are:

1. **Data Acquisition**
   - Download and organize benchmark datasets: CIC-IDS2017, NSL-KDD, UNSW-NB15, and ToN-IoT.
   - Each dataset is stored in its dedicated folder with raw data, scripts, and results.

2. **Data Preprocessing**
   - Handle missing values, outliers, and inconsistent data entries.
   - Encode categorical features using label encoding or one-hot encoding as appropriate.
   - Normalize or standardize numerical features to ensure uniformity across features.
   - For some datasets, merge and filter test sets (see `merge_and_filter_test_sets.py` in `cic-2017/`).
   - Visualize data distributions and class imbalances (see `.png` plots in preprocessing folders).

3. **Feature Selection**
   - Apply statistical and model-based feature selection techniques:
     - ANOVA F-test (e.g., in `feature_selection.ipynb`)
     - XGBoost feature importance (see `feature_importance_xgboost.png`)
     - Correlation analysis to remove redundant features (see `corr_heatmap.png`)
   - Select top features for each dataset to improve model performance and reduce overfitting.

4. **Model Training**
   - Train classical machine learning models (Random Forest, XGBoost, Decision Tree, Logistic Regression, Naive Bayes) using scikit-learn and XGBoost libraries.
   - Train deep learning models (CNN) using TensorFlow/Keras, especially for complex datasets like UNSW-NB15.
   - Use cross-validation and hyperparameter tuning where appropriate.
   - Training scripts and notebooks are provided in each dataset folder (e.g., `nsl_kdd.ipynb`, `unsw_cnn_attack_type_full.py`).

5. **Model Evaluation**
   - Evaluate models using metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
   - Generate confusion matrices, ROC curves, and feature importance plots for each experiment.
   - Save all evaluation results as `.png` files in the respective dataset folders.

6. **Result Analysis & Comparison**
   - Compare model performance across datasets and feature selection strategies.
   - Analyze the impact of feature selection on accuracy and generalization.
   - Summarize findings in the README and through visualizations.

**References to scripts and notebooks:**
- Preprocessing and feature selection: `feature_selection.ipynb`, `preprocessing/` scripts
- Model training: `nsl_kdd.ipynb`, `unsw_cnn_attack_type_full.py`, `ton_iot_xgboost_multiclass_ids.py`
- Evaluation and visualization: `.png` result files in each dataset folder

This structured methodology ensures that the IDS is rigorously tested and benchmarked, providing reliable insights into the effectiveness of different models and feature selection techniques for intrusion detection.

---

## 📁 Datasets Used

| Dataset         | Description                                                                                       |
|----------------|---------------------------------------------------------------------------------------------------|
| **CIC-IDS2017**| Real-world network traffic simulating common attack scenarios (DoS, PortScan, BruteForce, Web, Infiltration, etc.) |
| **NSL-KDD**    | Improved version of KDD’99 eliminating redundant records and used for standard IDS benchmarking    |
| **UNSW-NB15**  | Covers modern attack types including Fuzzers, Backdoors, Exploits, Worms, etc.                    |
| **ToN-IoT**    | Real-time IoT telemetry dataset focused on smart home and industrial IoT attacks                   |

Each dataset folder contains scripts, notebooks, and result plots specific to that dataset.

---

## ⚙️ Project Structure

```plaintext
Project_Nishant_bhardwaj/
├── cic-2017/
│   ├── feature_selection.ipynb         # Feature selection and preprocessing for CIC-IDS2017
│   ├── preprocessing/                  # Additional preprocessing scripts and visualizations
│   ├── model_training/                 # (If present) Model training scripts for CIC-IDS2017
│   └── *.png                          # Result plots (confusion matrices, ROC curves, etc.)
├── nsl_kdd/
│   └── nsl_kdd.ipynb                   # Notebook for NSL-KDD experiments
│   └── *.png                          # Result plots
├── ton_iot/
│   └── ton_iot_xgboost_multiclass_ids.py # XGBoost multiclass classification for ToN-IoT
│   └── *.png                          # Result plots
├── unsw dataset/
│   ├── test_model.py                   # Testing models on UNSW-NB15
│   ├── unsw_cnn_attack_type_full.py    # CNN model for attack-type classification
│   ├── result.png, *.png               # Confusion matrices, ROC curves, etc.
│   └── result plots/                   # (If present) Additional result visualizations
├── wustl/                              # (If present) Additional dataset experiments
│   └── model.py, *.png
└── README.md                           # Project documentation
```

---

## 🧠 Models & Techniques

### ✅ Classical Machine Learning Models
- **Random Forest**: Ensemble learning for robust classification
- **XGBoost**: Gradient boosting for high performance
- **Decision Tree**: Used as a baseline
- **Logistic Regression**: For binary/multiclass classification
- **Naive Bayes**: Used in selected experiments

### ✅ Deep Learning
- **Convolutional Neural Network (CNN)**: For attack-type classification, especially on UNSW-NB15
- **Frameworks**: Keras and TensorFlow for deep learning models

---

## 🧪 Feature Selection

Feature selection is a critical step in building effective intrusion detection models. By selecting the most relevant features, we reduce dimensionality, improve model interpretability, and often boost classification performance. This project employs several feature selection techniques:

- **ANOVA F-test**: Used to identify features with the strongest relationship to the output variable. Applied in the `feature_selection.ipynb` notebook in the `cic-2017/` folder.
- **XGBoost Feature Importance**: Leverages the built-in feature importance scores from XGBoost models to rank and select top features. Visualizations and selection scripts are provided in each dataset folder (e.g., `feature_importance_xgboost.png`).
- **Correlation Analysis**: Examines the correlation matrix to remove highly correlated (redundant) features, improving model generalization. See correlation heatmaps in the `preprocessing/` subfolders (e.g., `corr_heatmap.png`).

Feature selection results and visualizations are available as `.png` files in the respective dataset folders. The selected features are then used for model training and evaluation, leading to improved accuracy and reduced overfitting.

---

## 📊 Evaluation Metrics

| Metric      | Description                                      |
|-------------|--------------------------------------------------|
| Accuracy    | Percentage of correctly classified samples        |
| Precision   | Ratio of true positives to predicted positives   |
| Recall      | Ratio of true positives to actual positives      |
| F1-Score    | Harmonic mean of Precision and Recall            |
| ROC-AUC     | Area under the ROC curve for multi-class detection |

- Evaluation is performed per dataset, with results saved as plots (confusion matrices, ROC curves, etc.) in each dataset folder.

---

## 🚀 How to Run

### ⚙️ Requirements
- Python 3.9+
- Install dependencies:

```bash
pip install -r requirements.txt
```

### 📌 Steps
1. **Clone the repository:**
   ```bash
git clone https://github.com/Nishant-S-Bhardwaj/Intrusion-Detection-System.git
cd Intrusion-Detection-System
   ```
2. **Run preprocessing (example for CIC-IDS2017):**
   ```bash
cd cic-2017
jupyter notebook feature_selection.ipynb
   ```
3. **Train models (e.g., Random Forest on NSL-KDD):**
   ```bash
cd nsl_kdd
jupyter notebook nsl_kdd.ipynb
   ```
4. **Test CNN on UNSW-NB15:**
   ```bash
cd "unsw dataset"
python test_model.py
   ```
5. **View results:**
   - All confusion matrices, ROC curves, and feature importance plots are saved as `.png` files in the respective dataset folders.

---

## 📈 Sample Results

| Dataset      | Model         | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------------|--------------|----------|-----------|--------|----------|---------|
| CIC-IDS2017  | Random Forest| 99.88%   | 99.88%    | 99.88% | 99.88%   | 0.98    |
| NSL-KDD      | XGBoost      | 99.31%   | 99.30%    | 99.31% | 99.30%   | 0.99    |
| UNSW-NB15    | CNN          | 97.48%   | 97.34%    | 97.48% | 97.36%   | 0.96    |

- 📌 All confusion matrices and ROC curves are available as `.png` files in each dataset folder.

---

## 📚 Technologies Used
- Python (3.9+)
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Pandas, NumPy
- Matplotlib, Seaborn
- Jupyter Notebooks

---

## 🧑‍💻 Author
**Nishant Bhardwaj**  
Final Year B.Tech (Electronics & Communication)  
NIT Raipur  
[GitHub](https://github.com/Nishant-S-Bhardwaj)

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing
Pull requests are welcome! Feel free to fork the repository and submit improvements, bug fixes, or new features.

---

## 📌 Acknowledgements
- Canadian Institute for Cybersecurity (for CIC-IDS2017)
- UNSW Canberra (for UNSW-NB15)
- University of New Brunswick (for NSL-KDD)
- CSIRO Data61 (for ToN-IoT)

---

## 🖼️ Screenshots & Visualizations
Want to see sample outputs? Check the `.png` files in each dataset folder for confusion matrices, ROC curves, and feature importance charts.

---

> **For any questions or suggestions, feel free to open an issue or contact the author via GitHub!**
