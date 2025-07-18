import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize

# --- Load encoders, scaler, and feature list ---
le_proto = joblib.load('le_proto.joblib')
le_state = joblib.load('le_state.joblib')
le_service = joblib.load('le_service.joblib')
le_attack_cat = joblib.load('le_attack_cat.joblib')
scaler = joblib.load('scaler.joblib')
selected_features = joblib.load('selected_features.joblib')

# --- Load test data ---
test = pd.read_csv(r"N:\DATASET\unsw dataset\CSV Files\Training and Testing Sets\UNSW_NB15_testing-set.csv")

# --- Drop unnecessary columns ---
drop_cols = ['srcip', 'dstip', 'Stime', 'Ltime']
test.drop(columns=drop_cols, inplace=True, errors='ignore')

# --- Encode categorical columns ---
for col, le_col in zip(['proto', 'state', 'service'], [le_proto, le_state, le_service]):
    if col in test.columns:
        test[col] = le_col.transform(test[col].astype(str))

# --- Encode target variable ---
test['attack_cat_encoded'] = le_attack_cat.transform(test['attack_cat'].astype(str))

# --- Separate features and labels ---
X_test = test.drop(columns=['attack_cat', 'Label', 'attack_cat_encoded'], errors='ignore')
y_test = test['attack_cat_encoded']

# --- Ensure all features are present and in correct order ---
# Use scaler.feature_names_in_ if available, else selected_features
feature_list = getattr(scaler, 'feature_names_in_', selected_features)
for col in feature_list:
    if col not in X_test.columns:
        X_test[col] = 0
X_test = X_test[feature_list]

# --- Scale features ---
X_test_flat = scaler.transform(X_test)
X_test_dcnn = X_test_flat.reshape(-1, X_test.shape[1], 1)

# --- Load models ---
dcnn_model = load_model('best_unsw_model.h5')
rf_model = joblib.load('rf_model.joblib')
xgb_model = joblib.load('xgb_model.joblib')

# --- Predict ---
y_pred_dcnn = dcnn_model.predict(X_test_dcnn)
y_pred_dcnn_labels = np.argmax(y_pred_dcnn, axis=1)
y_pred_rf = rf_model.predict(X_test_flat)
y_pred_rf_proba = rf_model.predict_proba(X_test_flat)
y_pred_xgb = xgb_model.predict(X_test_flat)
y_pred_xgb_proba = xgb_model.predict_proba(X_test_flat)

# --- Print classification reports ---
def print_report(name, y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print(f"\n{name} Classification Report:")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Precision: {report['weighted avg']['precision']:.4f}")
    print(f"Recall: {report['weighted avg']['recall']:.4f}")
    print(f"F1-score: {report['weighted avg']['f1-score']:.4f}")
    for cname in class_names:
        if cname in report:
            print(f"Class {cname}: Precision={report[cname]['precision']:.4f}, Recall={report[cname]['recall']:.4f}, F1={report[cname]['f1-score']:.4f}")
        else:
            print(f"Class {cname}: Precision=0.0000, Recall=0.0000, F1=0.0000 (not present in predictions or ground truth)")

class_names = le_attack_cat.classes_
for name, y_pred in [
    ('DCNN', y_pred_dcnn_labels),
    ('Random Forest', y_pred_rf),
    ('XGBoost', y_pred_xgb)
]:
    print_report(name, y_test, y_pred, class_names)
    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d',
                xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

# --- ROC curves for all models ---
n_classes = len(class_names)
y_test_bin = label_binarize(y_test, classes=range(n_classes))
proba_dict = {
    'DCNN': y_pred_dcnn,
    'Random Forest': y_pred_rf_proba,
    'XGBoost': y_pred_xgb_proba
}
for name, y_proba in proba_dict.items():
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'{name} ROC Curve (One-vs-Rest)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Feature Importance Chart (Random Forest) ---
try:
    rf = joblib.load('rf_feature_selector.joblib')
    importances = rf.feature_importances_
    feature_names = getattr(rf, 'feature_names_in_', None)
    if feature_names is None:
        feature_names = list(selected_features)
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 6))
    plt.title('Feature Importances (Random Forest)')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print('Feature importance chart could not be displayed:', e)
