import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import numpy as np

# Load training data
train_path = r'N:\DATASET\cic-2017\preprocessing\merged_common_features.csv'
df_train = pd.read_csv(train_path)

# Preprocess training data
for col in df_train.columns:
    if df_train[col].dtype in ['float64', 'int64']:
        df_train[col] = df_train[col].fillna(df_train[col].median())
    else:
        df_train[col] = df_train[col].fillna(df_train[col].mode()[0])

X_train = df_train.drop(columns=['Label'])
y_train = df_train['Label']

# Encode categorical features in training
for col in X_train.select_dtypes(include=['object', 'category']).columns:
    X_train[col] = LabelEncoder().fit_transform(X_train[col].astype(str))

if y_train.dtype == 'object' or y_train.dtype.name == 'category':
    y_train = LabelEncoder().fit_transform(y_train)

# Train model on all training data
model = GaussianNB()
model.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# XGBoost
xgb = XGBClassifier(eval_metric='mlogloss', random_state=42, use_label_encoder=False)
xgb.fit(X_train, y_train)

# DCNN/MLP
n_classes = len(np.unique(y_train))
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
mlp = build_mlp(X_train.shape[1], n_classes)
if n_classes > 2:
    y_train_cat = to_categorical(y_train)
else:
    y_train_cat = y_train
mlp.fit(X_train, y_train_cat, epochs=20, batch_size=32, verbose=0)

# Load merged test set
test_path = r'N:\DATASET\cic-2017\preprocessing\merged_test_set.csv'
df_test = pd.read_csv(test_path)

# Preprocess test data
for col in df_test.columns:
    if df_test[col].dtype in ['float64', 'int64']:
        df_test[col] = df_test[col].fillna(df_test[col].median())
    else:
        df_test[col] = df_test[col].fillna(df_test[col].mode()[0])

X_test = df_test.drop(columns=['Label'])
y_test = df_test['Label']

# Encode categorical features in test (fit new encoders for simplicity, but ideally use same as training)
for col in X_test.select_dtypes(include=['object', 'category']).columns:
    X_test[col] = LabelEncoder().fit_transform(X_test[col].astype(str))

if y_test.dtype == 'object' or y_test.dtype.name == 'category':
    y_test = LabelEncoder().fit_transform(y_test)

# Predict and evaluate
results = {}

# Naive Bayes
y_pred = model.predict(X_test)
results['Naive Bayes'] = y_pred

# Random Forest
rf_pred = rf.predict(X_test)
results['Random Forest'] = rf_pred

# XGBoost
xgb_pred = xgb.predict(X_test)
results['XGBoost'] = xgb_pred

# DCNN/MLP
mlp_pred = mlp.predict(X_test)
mlp_pred_label = np.argmax(mlp_pred, axis=1) if n_classes > 2 else (mlp_pred > 0.5).astype(int).flatten()
results['DCNN'] = mlp_pred_label

from sklearn.metrics import classification_report
for name, y_pred in results.items():
    print(f"\n=== {name} ===")
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Precision: {report['weighted avg']['precision']:.4f}")
    print(f"Recall: {report['weighted avg']['recall']:.4f}")
    print(f"F1-score: {report['weighted avg']['f1-score']:.4f}")
    for i, cname in enumerate([str(c) for c in np.unique(y_test)]):
        print(f"Class {cname}: Precision={report[cname]['precision']:.4f}, Recall={report[cname]['recall']:.4f}, F1={report[cname]['f1-score']:.4f}")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Test Confusion Matrix Heatmap')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()
    # ROC Curve
    if name == 'DCNN':
        y_proba = mlp_pred
    elif name == 'Random Forest':
        y_proba = rf.predict_proba(X_test)
    elif name == 'XGBoost':
        y_proba = xgb.predict_proba(X_test)
    elif name == 'Naive Bayes':
        y_proba = model.predict_proba(X_test)
    else:
        y_proba = None
    if y_proba is not None:
        plt.figure(figsize=(8, 6))
        if n_classes > 2:
            y_test_bin = label_binarize(y_test, classes=range(n_classes))
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.4f})")
        else:
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f"{name} ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.show()
