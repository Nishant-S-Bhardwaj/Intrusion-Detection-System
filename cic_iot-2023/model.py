import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedShuffleSplit

import warnings
warnings.filterwarnings("ignore")

import glob

# ---- CONFIG ----
DATASET_GLOB = 'Merged*.csv'  # Pattern for your files (Merged01.csv, Merged02.csv, ...)
TARGET_COLUMN = 'Label'       # Change if needed
N_FEATURES = 23
FS_SAMPLE_SIZE = 100_000      # Subsample size for feature selection to avoid memory errors
RF_SAMPLE_SIZE = 100_000      # Subsample size for Random Forest training
MLP_SAMPLE_SIZE = 100_000     # Subsample size for MLP training

# ---- LOAD AND MERGE DATA ----
all_files = sorted(glob.glob(DATASET_GLOB))[:10]  # Get up to 10 files
if not all_files:
    raise FileNotFoundError(f"No files matched the pattern {DATASET_GLOB}")
df_list = [pd.read_csv(f) for f in all_files]
df = pd.concat(df_list, ignore_index=True)
print(f"Merged {len(df_list)} files, total rows: {len(df)}")

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# ---- HANDLE MISSING AND INFINITE VALUES ----
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())
X = X.dropna(axis=1, how='all')

# Optional: Check for any remaining NaN or inf
if np.any(np.isnan(X.values)):
    print("Warning: NaNs remain in X after imputation.")
if np.any(np.isinf(X.values)):
    print("Warning: Infs remain in X after imputation.")

# Encode categorical target if needed
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)
    class_names = le.classes_
else:
    class_names = np.unique(y).astype(str)

# ---- FEATURE SELECTION (with subsampling for memory efficiency) ----
def select_features(X, y, method, k):
    if method == 'anova':
        selector = SelectKBest(f_classif, k=k)
    elif method == 'mi':
        selector = SelectKBest(mutual_info_classif, k=k)
    elif method == 'chi2':
        # chi2 requires non-negative features
        X_chi = X.copy()
        X_chi[X_chi < 0] = 0
        selector = SelectKBest(chi2, k=k)
        X = X_chi
    else:
        raise ValueError("Unknown method")
    X_new = selector.fit_transform(X, y)
    feature_idx = selector.get_support(indices=True)
    feature_scores = selector.scores_
    return X_new, feature_idx, feature_scores

# Subsample for feature selection if dataset is large
if len(X) > FS_SAMPLE_SIZE:
    X_fs_sample = X.sample(FS_SAMPLE_SIZE, random_state=42)
    y_fs_sample = y[X_fs_sample.index]
else:
    X_fs_sample = X
    y_fs_sample = y

# Choose best feature selection method
if np.issubdtype(X.dtypes[0], np.number):
    fs_methods = ['anova', 'mi']
else:
    fs_methods = ['chi2']

fs_results = {}
for method in fs_methods:
    print(f"Running feature selection: {method}")
    X_fs, idx, scores = select_features(X_fs_sample, y_fs_sample, method, N_FEATURES)
    fs_results[method] = {'X': X_fs, 'idx': idx, 'scores': scores}

# Pick the best method (highest mean score)
best_method = max(fs_results, key=lambda m: np.nanmean(fs_results[m]['scores']))
selected_idx = fs_results[best_method]['idx']
selected_scores = fs_results[best_method]['scores']
selected_features = X.columns[selected_idx]
X_selected = X.iloc[:, selected_idx]  # Use full data for training

print(f"Selected features with highest mean value ({best_method}): {list(selected_features)}")

# ---- FEATURE SELECTION CHART ----
plt.figure(figsize=(12, 6))
# Get the scores for the selected features only
selected_feature_scores = selected_scores[selected_idx]
# Sort selected features by their scores
sorted_idx = np.argsort(selected_feature_scores)[::-1]
plt.bar(range(len(selected_features)), selected_feature_scores[sorted_idx])
plt.xticks(range(len(selected_features)), selected_features[sorted_idx], rotation=45, ha='right')
plt.title(f"Feature Selection Scores ({best_method.upper()})")
plt.ylabel("Score")
plt.xlabel("Feature")
plt.tight_layout()
plt.savefig("feature_selection_chart.png")
plt.close()
print("Feature selection chart saved as feature_selection_chart.png.")

# ---- SPLIT DATA ----
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.3, random_state=42, stratify=y
)

# ---- STANDARDIZE ----
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---- CLASSIFIERS ----
models = {}

# Naive Bayes
models['Naive Bayes'] = GaussianNB()

# Random Forest (will be trained on a subsample)
models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42)

# XGBoost
models['XGBoost'] = XGBClassifier(eval_metric='mlogloss', random_state=42, use_label_encoder=False)

# DCNN (MLP for tabular)
def build_mlp(input_dim, n_classes):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_classes, activation='softmax' if n_classes > 2 else 'sigmoid'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy' if n_classes > 2 else 'binary_crossentropy',
                  metrics=['accuracy'])
    return model

# ---- TRAIN & EVALUATE ----
results = {}

# Naive Bayes (full data)
print("Training Naive Bayes...")
models['Naive Bayes'].fit(X_train, y_train)
y_pred = models['Naive Bayes'].predict(X_test)
y_proba = models['Naive Bayes'].predict_proba(X_test)
results['Naive Bayes'] = {'model': models['Naive Bayes'], 'y_pred': y_pred, 'y_proba': y_proba}

# Random Forest (subsample)
if len(X_train) > RF_SAMPLE_SIZE:
    idx_rf = np.random.choice(len(X_train), RF_SAMPLE_SIZE, replace=False)
    X_train_rf = X_train[idx_rf]
    y_train_rf = y_train[idx_rf]
else:
    X_train_rf = X_train
    y_train_rf = y_train

print("Training Random Forest on a subset...")
models['Random Forest'].fit(X_train_rf, y_train_rf)
y_pred = models['Random Forest'].predict(X_test)
y_proba = models['Random Forest'].predict_proba(X_test)
results['Random Forest'] = {'model': models['Random Forest'], 'y_pred': y_pred, 'y_proba': y_proba}

# XGBoost (subsample)
sss = StratifiedShuffleSplit(n_splits=1, train_size=RF_SAMPLE_SIZE, random_state=42)
for idx_xgb, _ in sss.split(X_train, y_train):
    X_train_xgb = X_train[idx_xgb]
    y_train_xgb = y_train[idx_xgb]

print("Training XGBoost on a subset...")
models['XGBoost'].fit(X_train_xgb, y_train_xgb)
y_pred = models['XGBoost'].predict(X_test)
y_proba = models['XGBoost'].predict_proba(X_test)
results['XGBoost'] = {'model': models['XGBoost'], 'y_pred': y_pred, 'y_proba': y_proba}

# DCNN/MLP (subsample)
if len(X_train) > MLP_SAMPLE_SIZE:
    idx_mlp = np.random.choice(len(X_train), MLP_SAMPLE_SIZE, replace=False)
    X_train_mlp = X_train[idx_mlp]
    y_train_mlp = y_train[idx_mlp]
else:
    X_train_mlp = X_train
    y_train_mlp = y_train

print("Training DCNN (MLP) on a subset...")
n_classes = len(np.unique(y))
mlp = build_mlp(X_train.shape[1], n_classes)
if n_classes > 2:
    y_train_cat = to_categorical(y_train_mlp)
    y_test_cat = to_categorical(y_test)
else:
    y_train_cat = y_train_mlp
    y_test_cat = y_test
mlp.fit(X_train_mlp, y_train_cat, epochs=30, batch_size=32, verbose=0)
y_pred_mlp = np.argmax(mlp.predict(X_test), axis=1) if n_classes > 2 else (mlp.predict(X_test) > 0.5).astype(int).flatten()
y_proba_mlp = mlp.predict(X_test)
results['DCNN'] = {'model': mlp, 'y_pred': y_pred_mlp, 'y_proba': y_proba_mlp}

# ---- EVALUATION & PLOTTING ----
from sklearn.metrics import classification_report

for name, res in results.items():
    print(f"\n=== {name} ===")
    # Print classification report
    report = classification_report(y_test, res['y_pred'], target_names=[str(c) for c in np.unique(y_test)], output_dict=True)
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Precision: {report['weighted avg']['precision']:.4f}")
    print(f"Recall: {report['weighted avg']['recall']:.4f}")
    print(f"F1-score: {report['weighted avg']['f1-score']:.4f}")
    for i, cname in enumerate([str(c) for c in np.unique(y_test)]):
        print(f"Class {cname}: Precision={report[cname]['precision']:.4f}, Recall={report[cname]['recall']:.4f}, F1={report[cname]['f1-score']:.4f}")
    # 1. Confusion Matrix (separate file)
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, res['y_pred'])
    labels = np.unique(y_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(np.arange(len(labels)))
    plt.yticks(np.arange(len(labels)))
    if 'le' in locals():
        label_names = le.inverse_transform(labels)
    else:
        label_names = labels
    plt.xticks(np.arange(len(labels)), label_names)
    plt.yticks(np.arange(len(labels)), label_names)
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{name.replace(' ', '_')}.png")
    plt.close()
    # 2. ROC Curve (separate file)
    if res['y_proba'] is not None:
        plt.figure(figsize=(8, 6))
        if n_classes > 2:
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test == i, res['y_proba'][:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"Class {label_names[i]} (AUC={roc_auc:.4f})")
        else:
            fpr, tpr, _ = roc_curve(y_test, res['y_proba'][:, 1] if res['y_proba'].ndim > 1 else res['y_proba'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f"{name} ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"roc_{name.replace(' ', '_')}.png")
        plt.close()
    # 3. Feature Importance/Ranking (separate file)
    plt.figure(figsize=(12, 6))
    if name == 'Random Forest':
        importances = res['model'].feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.bar(range(N_FEATURES), importances[indices])
        plt.xticks(range(N_FEATURES), selected_features[indices], rotation=45, ha='right')
        plt.title(f"{name} Feature Importance")
    elif name == 'Naive Bayes':
        if hasattr(res['model'], 'theta_'):
            importances = np.abs(res['model'].theta_).mean(axis=0)
            indices = np.argsort(importances)[::-1]
            plt.bar(range(N_FEATURES), importances[indices])
            plt.xticks(range(N_FEATURES), selected_features[indices], rotation=45, ha='right')
            plt.title(f"{name} Feature Means")
    elif name == 'DCNN':
        weights = res['model'].layers[0].get_weights()[0]
        importances = np.abs(weights).mean(axis=1)
        indices = np.argsort(importances)[::-1]
        plt.bar(range(N_FEATURES), importances[indices])
        plt.xticks(range(N_FEATURES), selected_features[indices], rotation=45, ha='right')
        plt.title(f"{name} Input Weight Magnitude")
    elif name == 'XGBoost':
        importances = res['model'].feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.bar(range(N_FEATURES), importances[indices])
        plt.xticks(range(N_FEATURES), selected_features[indices], rotation=45, ha='right')
        plt.title(f"{name} Feature Importance")
    plt.tight_layout()
    plt.savefig(f"feature_importance_{name.replace(' ', '_')}.png")
    plt.close()

print("All results plotted and saved as separate PNG files for each model and metric.")
# ---- END ----