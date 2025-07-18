# 🔐 Intrusion Detection System (IDS) using Machine Learning & Deep Learning

A comprehensive Intrusion Detection System leveraging both classical Machine Learning and Deep Learning models across multiple cybersecurity benchmark datasets, including CIC-IDS2017, NSL-KDD, UNSW-NB15, and ToN-IoT.

> 🚨 **Detects network attacks like DoS, Probe, R2L, U2R, Botnet, and more using supervised learning models.**

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

### 🔍 Feature Selection
- **ANOVA F-test**
- **XGBoost Feature Importance**
- **Correlation Analysis**

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
