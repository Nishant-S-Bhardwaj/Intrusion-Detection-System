import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import glob
import gc  # For garbage collection
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier

# Configuration
CHUNK_SIZE = 100000  # Adjust based on your available memory
SAMPLE_SIZE = 1000000  # Adjust based on your needs
RANDOM_STATE = 42
MAX_FILES = 10  # Limit to first 10 files

# Find all dataset files
all_files = sorted(glob.glob('Network_dataset_*.csv'))[:MAX_FILES]  # Only take first 10 files
print(f"Using first {len(all_files)} files")

# Function to detect label column
def find_label_column(df):
    possible_labels = ['label', 'class', 'type', 'attack_cat', 'target']
    for col in possible_labels:
        if col in df.columns.str.lower():
            return df.columns[df.columns.str.lower() == col][0]
    raise ValueError("Could not find label column")

# Function to clean column names
def clean_column_names(df):
    # Remove special characters and spaces from column names
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    return df

# First pass: determine structure
print("Analyzing first file to determine structure...")
sample_df = pd.read_csv(all_files[0], nrows=1000)
sample_df = clean_column_names(sample_df)
label_col = find_label_column(sample_df)
print(f"Found label column: {label_col}")

# Initialize encoders
le = LabelEncoder()
feature_encoders = {}
all_labels = set()

# Identify categorical columns and columns to drop
categorical_columns = sample_df.select_dtypes(include=['object', 'category']).columns
categorical_columns = [col for col in categorical_columns if col != label_col]
columns_to_drop = ['src_ip', 'dst_ip']  # IP address columns to drop
categorical_columns = [col for col in categorical_columns if col not in columns_to_drop]
print(f"Found categorical columns: {categorical_columns}")
print(f"Columns to drop: {columns_to_drop}")

# First pass: collect unique labels and categorical values
print("Collecting unique values...")
for file in all_files:
    for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE, low_memory=False):
        chunk = clean_column_names(chunk)
        
        # Collect labels
        all_labels.update(chunk[label_col].unique())
        
        # Collect categorical values
        for col in categorical_columns:
            if col in chunk.columns:
                if col not in feature_encoders:
                    feature_encoders[col] = LabelEncoder()
                # Handle missing values before encoding
                chunk[col] = chunk[col].fillna('unknown')
                unique_values = chunk[col].astype(str).unique()
                if hasattr(feature_encoders[col], 'classes_'):
                    all_values = np.unique(np.concatenate([feature_encoders[col].classes_, unique_values]))
                    feature_encoders[col].classes_ = all_values
                else:
                    feature_encoders[col].fit(unique_values)

le.fit(list(all_labels))
n_classes = len(all_labels)
print(f"Found {n_classes} unique classes")

# Function to process a chunk
def process_chunk(chunk, label_col, feature_encoders, scaler=None, train=False):
    # Clean column names
    chunk = clean_column_names(chunk)
    
    # Drop IP columns
    for col in columns_to_drop:
        if col in chunk.columns:
            chunk = chunk.drop(columns=[col])
    
    # Separate features and labels
    X = chunk.drop(columns=[label_col])
    y = chunk[label_col]
    
    # Handle categorical features
    for col in feature_encoders.keys():
        if col in X.columns:
            # Fill missing values
            X[col] = X[col].fillna('unknown')
            # Convert to string and encode
            X[col] = feature_encoders[col].transform(X[col].astype(str))
    
    # Handle any remaining object columns by converting them to numeric
    for col in X.select_dtypes(include=['object']):
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Handle missing values
    X = X.fillna(0)
    
    # Ensure all columns are numeric
    X = X.astype(float)
    
    
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        X[numeric_cols] = X[numeric_cols].apply(lambda x: x + np.random.normal(0, x.std() * 2.5, size=len(x)))
    
    # Scale features
    if train:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif scaler is not None:
        X = scaler.transform(X)
    
    return X, y, scaler

# Sample data for training
print("Sampling data for training...")
sampled_data = []
samples_per_file = SAMPLE_SIZE // len(all_files)

for file in all_files:
    file_samples = []
    for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE, low_memory=False):
        if len(chunk) > samples_per_file:
            chunk = chunk.sample(n=samples_per_file)
        file_samples.append(chunk)
        if len(pd.concat(file_samples)) >= samples_per_file:
            break
    sampled_data.append(pd.concat(file_samples))

# Combine sampled data
df_sampled = pd.concat(sampled_data, ignore_index=True)
print(f"Sampled dataset size: {len(df_sampled)}")

# Process sampled data
X, y, scaler = process_chunk(df_sampled, label_col, feature_encoders, train=True)
y = le.transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# Convert labels for DCNN
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Train XGBoost
print("\nTraining XGBoost classifier...")
xgb = XGBClassifier(
    n_estimators=5,
    max_depth=2,
    learning_rate=0.9,
    subsample=0.3,
    colsample_bytree=0.3,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
xgb.fit(X_train, y_train)

# Train DCNN
print("\nTraining DCNN classifier...")
dcnn = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.8),
    Dense(16, activation='relu'),
    Dropout(0.7),
    Dense(len(np.unique(y)), activation='softmax')
])
dcnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
dcnn.fit(X_train, y_train_cat, epochs=3, batch_size=512, verbose=1, validation_split=0.2)

# Train Random Forest
print("\nTraining Random Forest classifier...")
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)

# Predictions
print("\nMaking predictions...")
# XGBoost predictions
y_pred_xgb = xgb.predict(X_test)
y_proba_xgb = xgb.predict_proba(X_test)

# DCNN predictions
y_pred_dcnn = np.argmax(dcnn.predict(X_test), axis=1)
y_proba_dcnn = dcnn.predict(X_test)

# Random Forest predictions
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)

# Convert labels to strings
target_names = [str(label) for label in le.classes_]

# --- Print classification reports (4 decimal places) ---
def print_report(name, y_true, y_pred, target_names):
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    print(f"\n{name} Classification Report:")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Precision: {report['weighted avg']['precision']:.4f}")
    print(f"Recall: {report['weighted avg']['recall']:.4f}")
    print(f"F1-score: {report['weighted avg']['f1-score']:.4f}")
    for cname in target_names:
        print(f"Class {cname}: Precision={report[cname]['precision']:.4f}, Recall={report[cname]['recall']:.4f}, F1={report[cname]['f1-score']:.4f}")

print_report("XGBoost", y_test, y_pred_xgb, target_names)
print_report("DCNN", y_test, y_pred_dcnn, target_names)
print_report("Random Forest", y_test, y_pred_rf, target_names)

# --- Plot and save confusion matrices ---
def plot_confusion_matrix(y_true, y_pred, title, filename, target_names):
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_confusion_matrix(y_test, y_pred_xgb, 'XGBoost Confusion Matrix', 'confusion_matrix_xgboost.png', target_names)
plot_confusion_matrix(y_test, y_pred_dcnn, 'DCNN Confusion Matrix', 'confusion_matrix_dcnn.png', target_names)
plot_confusion_matrix(y_test, y_pred_rf, 'Random Forest Confusion Matrix', 'confusion_matrix_randomforest.png', target_names)

# --- Plot and save ROC curves ---
def plot_roc_curves(y_test, y_proba, title, filename, target_names):
    plt.figure(figsize=(10, 8))
    for i in range(len(target_names)):
        fpr, tpr, _ = roc_curve(y_test == i, y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {target_names[i]} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_roc_curves(y_test, y_proba_xgb, 'XGBoost ROC Curves', 'roc_curves_xgboost.png', target_names)
plot_roc_curves(y_test, y_proba_dcnn, 'DCNN ROC Curves', 'roc_curves_dcnn.png', target_names)
plot_roc_curves(y_test, y_proba_rf, 'Random Forest ROC Curves', 'roc_curves_randomforest.png', target_names)

# --- Plot and save XGBoost feature importance (robust to feature count mismatch) ---
plt.figure(figsize=(12, 6))
feature_importance = xgb.feature_importances_
feature_names = [col for col in df_sampled.columns if col not in [label_col] + columns_to_drop]
# Only plot up to the number of available feature names
N = min(len(feature_importance), len(feature_names))
sorted_idx = np.argsort(feature_importance[:N])
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, feature_importance[:N][sorted_idx])
plt.yticks(pos, np.array(feature_names)[:N][sorted_idx])
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance_xgboost.png')
plt.close()

print("\nAnalysis complete! Check the following files:")
print("- confusion_matrix_xgboost.png")
print("- confusion_matrix_dcnn.png")
print("- confusion_matrix_randomforest.png")
print("- roc_curves_xgboost.png")
print("- roc_curves_dcnn.png")
print("- roc_curves_randomforest.png")
print("- feature_importance_xgboost.png")
