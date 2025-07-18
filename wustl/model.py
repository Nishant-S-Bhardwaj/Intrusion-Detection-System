import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
K.clear_session()

def build_dcnn(input_dim, num_classes):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
                  metrics=['accuracy'])
    return model

# 1. Load Data
DATA_PATH = "wustl-ehms-2020.csv"
df = pd.read_csv(DATA_PATH)
print(f"Dataset size: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Preprocessing
drop_cols = [
    'Dir', 'Flgs', 'SrcAddr', 'DstAddr', 'SrcMac', 'DstMac', 'Attack Category'
]
df = df.drop(columns=drop_cols, errors='ignore')
df = df.dropna()

cat_cols = df.select_dtypes(include=['object']).columns.tolist()
cat_cols = [col for col in cat_cols if col not in ['Label']]
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

X = df.drop('Label', axis=1)
y = df['Label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Remove constant features
nunique = X_train.apply(pd.Series.nunique)
constant_cols = nunique[nunique == 1].index.tolist()
if constant_cols:
    print(f"Removing constant columns: {constant_cols}")
    X_train = X_train.drop(columns=constant_cols)
    X_test = X_test.drop(columns=constant_cols)

# Remove highly correlated features (optional, threshold=0.98)
corr_matrix = X_train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]
if to_drop:
    print(f"Removing highly correlated columns: {to_drop}")
    X_train = X_train.drop(columns=to_drop)
    X_test = X_test.drop(columns=to_drop)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Feature Selection Methods

# 3.1 SelectKBest (ANOVA F-value)
skb = SelectKBest(score_func=f_classif, k=min(10, X_train.shape[1]))
X_train_skb = skb.fit_transform(X_train_scaled, y_train)
X_test_skb = skb.transform(X_test_scaled)
skb_features = X_train.columns[skb.get_support()]

# 3.2 Recursive Feature Elimination (RFE) with RandomForest (reduced complexity)
rfe_estimator = RandomForestClassifier(n_estimators=20, random_state=42)
# Ensure at least 2 features for RFE to avoid XGBoost errors
rfe = RFE(estimator=rfe_estimator, n_features_to_select=max(2, min(5, X_train.shape[1])))
X_train_rfe = rfe.fit_transform(X_train_scaled, y_train)
X_test_rfe = rfe.transform(X_test_scaled)
rfe_features = X_train.columns[rfe.get_support()]

# 3.3 Tree-based Feature Importance (RandomForest)
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train_scaled, y_train)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:min(10, X_train.shape[1])]
tree_features = X_train.columns[indices]
X_train_tree = X_train_scaled[:, indices]
X_test_tree = X_test_scaled[:, indices]

# 4. Classifiers
classifiers = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42),
}

feature_sets = {
    "SelectKBest": (X_train_skb, X_test_skb, skb_features),
    "RFE": (X_train_rfe, X_test_rfe, rfe_features),
    "TreeBased": (X_train_tree, X_test_tree, tree_features)
}

results = {}

print("Starting feature set loop")
for fs_name, (Xtr, Xte, feat_names) in feature_sets.items():
    print(f"{fs_name}: Number of features = {Xtr.shape[1]}")
    results[fs_name] = {}
    y_train_arr = y_train.values if hasattr(y_train, 'values') else y_train
    y_test_arr = y_test.values if hasattr(y_test, 'values') else y_test
    n_classes = len(np.unique(y_train_arr))
    # DCNN
    K.clear_session()
    dcnn = build_dcnn(Xtr.shape[1], n_classes)
    y_train_cat = to_categorical(y_train_arr, num_classes=n_classes)
    y_test_cat = to_categorical(y_test_arr, num_classes=n_classes)
    dcnn.fit(Xtr, y_train_cat, epochs=20, batch_size=32, verbose=0,
             callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
             validation_data=(Xte, y_test_cat))
    dcnn_pred = dcnn.predict(Xte)
    dcnn_pred_label = np.argmax(dcnn_pred, axis=1) if n_classes > 2 else (dcnn_pred[:, 0] > 0.5).astype(int)
    results[fs_name]["DCNN"] = {
        "y_pred": dcnn_pred_label,
        "y_proba": dcnn_pred,
        "clf": dcnn,
        "features": feat_names
    }
    # Sklearn classifiers
    for clf_name, clf in classifiers.items():
        try:
            clf.fit(Xtr, y_train)
            y_pred = clf.predict(Xte)
            y_proba = clf.predict_proba(Xte)
            results[fs_name][clf_name] = {
                "y_pred": y_pred,
                "y_proba": y_proba,
                "clf": clf,
                "features": feat_names
            }
        except Exception as e:
            print(f"Error for {fs_name} + {clf_name}: {e}")

# 5. Evaluation and Plots
n_classes = len(np.unique(y))
y_test_bin = label_binarize(y_test, classes=range(n_classes))

# Convert class labels to strings for reporting and plotting
class_names = [str(c) for c in le.classes_]

for fs_name in results:
    for clf_name in results[fs_name]:
        y_pred = results[fs_name][clf_name]["y_pred"]
        y_proba = results[fs_name][clf_name]["y_proba"]
        feat_names = results[fs_name][clf_name]["features"]

        print(f"\n=== {fs_name} + {clf_name} ===")
        print(f"{fs_name}: Xte.shape={Xte.shape}, y_test.shape={y_test.shape}")
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        print(f"Accuracy: {report['accuracy']:.4f}")
        print(f"Precision: {report['weighted avg']['precision']:.4f}")
        print(f"Recall: {report['weighted avg']['recall']:.4f}")
        print(f"F1-score: {report['weighted avg']['f1-score']:.4f}")
        # Optionally print per-class metrics
        for i, cname in enumerate(class_names):
            print(f"Class {cname}: Precision={report[cname]['precision']:.4f}, Recall={report[cname]['recall']:.4f}, F1={report[cname]['f1-score']:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix: {fs_name} + {clf_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()

        # ROC Curve (One-vs-Rest)
        if n_classes > 2:
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            plt.figure(figsize=(8,6))
            for i in range(n_classes):
                plt.plot(fpr[i], tpr[i], label=f'Class {class_names[i]} (AUC = {roc_auc[i]:.4f})')
            plt.plot([0,1],[0,1],'k--')
            plt.title(f'ROC Curve: {fs_name} + {clf_name}')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            fpr, tpr, _ = roc_curve(y_test, y_proba[:,1])
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(8,6))
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
            plt.plot([0,1],[0,1],'k--')
            plt.title(f'ROC Curve: {fs_name} + {clf_name}')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.tight_layout()
            plt.show()

        # Feature Importance/Ranking
        if clf_name != "DCNN":
            if hasattr(results[fs_name][clf_name]["clf"], "feature_importances_"):
                importances = results[fs_name][clf_name]["clf"].feature_importances_
                feat_names_arr = np.array(feat_names)
                plot_len = min(len(importances), len(feat_names_arr))
                plt.figure(figsize=(8,6))
                sns.barplot(x=importances[:plot_len], y=feat_names_arr[:plot_len])
                plt.title(f'Feature Importance: {fs_name} + {clf_name}')
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                plt.tight_layout()
                plt.show()
            elif hasattr(results[fs_name][clf_name]["clf"], "coef_"):
                importances = np.abs(results[fs_name][clf_name]["clf"].coef_).sum(axis=0)
                indices = np.argsort(importances)[::-1]
                plt.figure(figsize=(8,6))
                sns.barplot(x=importances[indices], y=np.array(feat_names)[indices])
                plt.title(f'Feature Coefficient Magnitude: {fs_name} + {clf_name}')
                plt.xlabel('Coefficient Magnitude')
                plt.ylabel('Feature')
                plt.tight_layout()
                plt.show()
        else:
            # DCNN input layer weights
            weights = results[fs_name][clf_name]["clf"].layers[0].get_weights()[0]
            importances = np.abs(weights).mean(axis=1)
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(8,6))
            sns.barplot(x=importances[indices], y=np.array(feat_names)[indices])
            plt.title(f'Feature Importance (Input Weights): {fs_name} + {clf_name}')
            plt.xlabel('Mean Abs Weight')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.show()

print("Review the plots and reported.")
# END OF SCRIPT