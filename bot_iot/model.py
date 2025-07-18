import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

# 1. Load dataset with correct delimiter
df = pd.read_csv(r"N:\DATASET\bot_iot\10-best features\UNSW_2018_IoT_Botnet_Final_10_Best.csv", delimiter=';')

# 2. SAMPLE ONE THIRD OF THE DATA for faster training/testing
df = df.sample(frac=1/3, random_state=42)
print(f"Using {len(df)} rows for training/testing.")

# 3. Choose label column: 'category' for multiclass, 'attack' for binary
label_col = 'category'  # or 'attack' for binary

# 4. Prepare features and labels
drop_cols = ['attack', 'category', 'subcategory', 'proto', 'saddr', 'sport', 'daddr', 'dport']
feature_cols = [col for col in df.columns if col not in drop_cols]
X = df[feature_cols].values
y_raw = df[label_col].values

# 5. Encode labels to 0-based contiguous integers
le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_
num_classes = len(class_names)
y_cat = to_categorical(y, num_classes=num_classes)

# 6. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = np.random.permutation(X_scaled.T).T

# === OVERALL FEATURE SELECTION PLOT (ANOVA F-score) ===
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X_scaled, y)
scores = selector.scores_
indices = np.argsort(scores)[::-1]
top_n = min(10, len(feature_cols))

plt.figure(figsize=(8, 6))
sns.barplot(
    x=scores[indices][:top_n],
    y=np.array(feature_cols)[indices][:top_n],
    palette="mako"
)
plt.title("Top Feature Selection Scores (ANOVA F-score)")
plt.xlabel("Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# 7. Train/test split
X_train, X_test, y_train, y_test, y_train_cat, y_test_cat = train_test_split(
    X_scaled, y, y_cat, test_size=0.2, random_state=42, stratify=y
)

results = {}
cf_matrices = {}
roc_curves = {}

# 8. Plot label balance
plt.figure(figsize=(6,4))
sns.countplot(x=y, palette='viridis')
plt.title('Label Balance in Dataset')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(ticks=range(num_classes), labels=class_names, rotation=45)
plt.tight_layout()
plt.show()

# 9. DCNN Model (simple, CPU only)
def build_dcnn(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(8, activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(0.95))
    model.add(Dense(4, activation='relu'))
    model.add(Dropout(0.9))
    model.add(Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
                  metrics=['accuracy'])
    return model

dcnn = build_dcnn(X_train.shape[1], num_classes)
dcnn.fit(X_train, y_train_cat, epochs=5, batch_size=128, verbose=1)
dcnn_pred = dcnn.predict(X_test)
dcnn_pred_label = np.argmax(dcnn_pred, axis=1) if num_classes > 2 else (dcnn_pred > 0.5).astype(int).flatten()
results['DCNN'] = classification_report(y_test, dcnn_pred_label, output_dict=True)
cf_matrices['DCNN'] = confusion_matrix(y_test, dcnn_pred_label)

# ROC for DCNN
if num_classes > 2:
    y_test_bin = label_binarize(y_test, classes=range(num_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], dcnn_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    roc_curves['DCNN'] = (fpr, tpr, roc_auc)
else:
    fpr, tpr, _ = roc_curve(y_test, dcnn_pred)
    roc_auc = auc(fpr, tpr)
    roc_curves['DCNN'] = (fpr, tpr, roc_auc)

# 10. Naive Bayes (CPU only, very fast)
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
results['NaiveBayes'] = classification_report(y_test, nb_pred, output_dict=True)
cf_matrices['NaiveBayes'] = confusion_matrix(y_test, nb_pred)

# ROC for NB
if num_classes > 2:
    y_test_bin = label_binarize(y_test, classes=range(num_classes))
    nb_pred_proba = nb.predict_proba(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], nb_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    roc_curves['NaiveBayes'] = (fpr, tpr, roc_auc)
else:
    nb_pred_proba = nb.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, nb_pred_proba)
    roc_auc = auc(fpr, tpr)
    roc_curves['NaiveBayes'] = (fpr, tpr, roc_auc)

# 11. XGBoost (CPU only)
xgb_clf = xgb.XGBClassifier(
    use_label_encoder=False, 
    eval_metric='mlogloss' if num_classes > 2 else 'logloss',
    tree_method='hist',
    max_depth=1,
    n_estimators=3,
    learning_rate=1.0,
    subsample=0.1,
    colsample_bytree=0.1
)
xgb_clf.fit(X_train, y_train)
xgb_pred = xgb_clf.predict(X_test)
results['XGBoost'] = classification_report(y_test, xgb_pred, output_dict=True)
cf_matrices['XGBoost'] = confusion_matrix(y_test, xgb_pred)

# ROC for XGBoost
if num_classes > 2:
    y_test_bin = label_binarize(y_test, classes=range(num_classes))
    xgb_pred_proba = xgb_clf.predict_proba(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], xgb_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    roc_curves['XGBoost'] = (fpr, tpr, roc_auc)
else:
    xgb_pred_proba = xgb_clf.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, xgb_pred_proba)
    roc_auc = auc(fpr, tpr)
    roc_curves['XGBoost'] = (fpr, tpr, roc_auc)

# Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
results['RandomForest'] = classification_report(y_test, rf_pred, output_dict=True)
cf_matrices['RandomForest'] = confusion_matrix(y_test, rf_pred)

# ROC for Random Forest
if num_classes > 2:
    y_test_bin = label_binarize(y_test, classes=range(num_classes))
    rf_pred_proba = rf_clf.predict_proba(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], rf_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    roc_curves['RandomForest'] = (fpr, tpr, roc_auc)
else:
    rf_pred_proba = rf_clf.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, rf_pred_proba)
    roc_auc = auc(fpr, tpr)
    roc_curves['RandomForest'] = (fpr, tpr, roc_auc)

# (Optional) XGBoost feature importance plot (can be commented out if not needed)
importances = xgb_clf.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = min(10, len(feature_cols))  # Show top 10 features or all if less

plt.figure(figsize=(8, 6))
sns.barplot(
    x=importances[indices][:top_n],
    y=np.array(feature_cols)[indices][:top_n],
    palette="viridis"
)
plt.title("Top Feature Importances (XGBoost)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# 12. Print Results
for model_name, report in results.items():
    print(f"\n=== {model_name} ===")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Precision: {report['weighted avg']['precision']:.4f}")
    print(f"Recall: {report['weighted avg']['recall']:.4f}")
    print(f"F1-score: {report['weighted avg']['f1-score']:.4f}")
    for cname in class_names:
        if cname in report:
            print(f"Class {cname}: Precision={report[cname]['precision']:.4f}, Recall={report[cname]['recall']:.4f}, F1={report[cname]['f1-score']:.4f}")
        else:
            print(f"Class {cname}: Precision=0.0000, Recall=0.0000, F1=0.0000 (not present in predictions or ground truth)")

# 13. Plot Confusion Matrices
for model_name, cf in cf_matrices.items():
    plt.figure(figsize=(6,5))
    sns.heatmap(cf, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

# 14. Plot ROC Curves
for model_name, roc_data in roc_curves.items():
    plt.figure(figsize=(6,5))
    if num_classes > 2:
        fpr, tpr, roc_auc = roc_data
        for i in range(num_classes):
            plt.plot(fpr[i], tpr[i], label=f'Class {class_names[i]} (AUC={roc_auc[i]:.4f})')
        plt.plot([0,1], [0,1], 'k--')
        plt.title(f'ROC Curve: {model_name}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        fpr, tpr, roc_auc = roc_data
        plt.plot(fpr, tpr, label=f'AUC={roc_auc:.4f}')
        plt.plot([0,1], [0,1], 'k--')
        plt.title(f'ROC Curve: {model_name}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.tight_layout()
        plt.show()

print("\nAll models evaluated and visualizations generated.")