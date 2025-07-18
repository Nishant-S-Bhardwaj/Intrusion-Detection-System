import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.models import load_model

# Load the training and testing datasets
train = pd.read_csv(r"N:\DATASET\unsw dataset\CSV Files\Training and Testing Sets\UNSW_NB15_training-set.csv")
test = pd.read_csv(r"N:\DATASET\unsw dataset\CSV Files\Training and Testing Sets\UNSW_NB15_testing-set.csv")

# Drop unnecessary columns
drop_cols = ['srcip', 'dstip', 'Stime', 'Ltime']
train.drop(columns=drop_cols, inplace=True, errors='ignore')
test.drop(columns=drop_cols, inplace=True, errors='ignore')

# Combine for consistent preprocessing (especially for get_dummies)
combined = pd.concat([train, test], axis=0)

# One-hot encode categorical features and save encoders
le_proto = LabelEncoder()
le_state = LabelEncoder()
le_service = LabelEncoder()
combined['proto'] = le_proto.fit_transform(combined['proto'].astype(str))
combined['state'] = le_state.fit_transform(combined['state'].astype(str))
combined['service'] = le_service.fit_transform(combined['service'].astype(str))
# Save encoders for test-time use
joblib.dump(le_proto, 'le_proto.joblib')
joblib.dump(le_state, 'le_state.joblib')
joblib.dump(le_service, 'le_service.joblib')
# Encode target
le_attack_cat = LabelEncoder()
combined['attack_cat_encoded'] = le_attack_cat.fit_transform(combined['attack_cat'].astype(str))
joblib.dump(le_attack_cat, 'le_attack_cat.joblib')

# Split back into train and test
train_processed = combined.iloc[:len(train)]
test_processed = combined.iloc[len(train):]

# Separate features and labels
X_train = train_processed.drop(columns=['attack_cat', 'Label', 'attack_cat_encoded'], errors='ignore')
y_train = train_processed['attack_cat_encoded']

X_test = test_processed.drop(columns=['attack_cat', 'Label', 'attack_cat_encoded'], errors='ignore')
y_test = test_processed['attack_cat_encoded']

# --- Random Forest Feature Selection ---
# Use a random forest to select the top N features
N_FEATURES = 30  # You can tune this number
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:N_FEATURES]
selected_features = X_train.columns[indices]
X_train = X_train[selected_features]
X_test = X_test[selected_features]


# One-hot encode labels
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler and selected features for test-time use
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(list(selected_features), 'selected_features.joblib')

# Reshape for Conv1D input
X_train = X_train.reshape(-1, X_train.shape[1], 1)
X_test = X_test.reshape(-1, X_test.shape[1], 1)

# --- Class Weights ---
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))

# --- Optimizer ---
optimizer = Adam(learning_rate=0.001)

# Improved CNN Model
model = Sequential([
    Conv1D(128, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Conv1D(256, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(256, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y_train_cat.shape[1], activation='softmax')  # Multiclass output
])

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])

# Callbacks for better training
checkpoint = ModelCheckpoint('best_unsw_model.h5', monitor='val_accuracy', save_best_only=True)
early_stop = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5)

# Train
history = model.fit(
    X_train, y_train_cat,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr, checkpoint],
    class_weight=class_weights_dict,
    verbose=2
)

# Train Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train.reshape(X_train.shape[0], X_train.shape[1]), y_train)

# Train XGBoost
xgb_clf = XGBClassifier(eval_metric='mlogloss', random_state=42, use_label_encoder=False)
xgb_clf.fit(X_train.reshape(X_train.shape[0], X_train.shape[1]), y_train)

# Predict
# DCNN
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test_cat, axis=1)

# Random Forest
y_pred_rf = rf_clf.predict(X_test.reshape(X_test.shape[0], X_test.shape[1]))

# XGBoost
y_pred_xgb = xgb_clf.predict(X_test.reshape(X_test.shape[0], X_test.shape[1]))

# Report
def print_report(name, y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print(f"\n{name} Classification Report:")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Precision: {report['weighted avg']['precision']:.4f}")
    print(f"Recall: {report['weighted avg']['recall']:.4f}")
    print(f"F1-score: {report['weighted avg']['f1-score']:.4f}")
    for i, cname in enumerate(class_names):
        print(f"Class {cname}: Precision={report[cname]['precision']:.4f}, Recall={report[cname]['recall']:.4f}, F1={report[cname]['f1-score']:.4f}")

print_report("DCNN (CNN)", y_true_labels, y_pred_labels, le_attack_cat.classes_)
print_report("Random Forest", y_test, y_pred_rf, le_attack_cat.classes_)
print_report("XGBoost", y_test, y_pred_xgb, le_attack_cat.classes_)

# Confusion Matrices
plt.figure(figsize=(12, 8))
sns.heatmap(confusion_matrix(y_true_labels, y_pred_labels), annot=True, fmt='d',
            xticklabels=le_attack_cat.classes_, yticklabels=le_attack_cat.classes_, cmap='Blues')
plt.title("DCNN (CNN) Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d',
            xticklabels=le_attack_cat.classes_, yticklabels=le_attack_cat.classes_, cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, fmt='d',
            xticklabels=le_attack_cat.classes_, yticklabels=le_attack_cat.classes_, cmap='Blues')
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Accuracy and Loss Curves
plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

import joblib
joblib.dump(rf_clf, 'rf_model.joblib')
joblib.dump(xgb_clf, 'xgb_model.joblib')
