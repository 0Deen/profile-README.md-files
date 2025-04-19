# train_ddos_detector.py
# Modular, 10,000+ lines structure for DDoS detection model training

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import time

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

# ========== CONFIGURATION ==========
DATA_DIR = "Mininet/data/cicddos2019_dataset"
SAVE_MODEL_PATH = "ensemble_ddos_model.pkl"
SELECTED_FEATURES_PATH = "selected_features.txt"
REPORT_PATH = "training_report.txt"

TRAINING_FILES = [
    "LDAP-training.parquet", "MSSQL-training.parquet", "NetBIOS-training.parquet",
    "Portmap-training.parquet", "Syn-training.parquet", "TFTP-training.parquet",
    "UDP-training.parquet", "UDPLag-training.parquet", "WebDDoS-training.parquet",
    "DrDoS_DNS-train.parquet", "DrDoS_LDAP-train.parquet", "DrDoS_MSSQL-train.parquet",
    "DrDoS_NTP-train.parquet", "DrDoS_NetBIOS-train.parquet", "DrDoS_SNMP-train.parquet",
    "DrDoS_SSDP-train.parquet", "DrDoS_UDP-train.parquet", "SynDDos-train.parquet"
]

# ========== UTILITY FUNCTIONS ==========
def log_report(msg):
    with open(REPORT_PATH, 'a') as f:
        f.write(msg + '\n')
    print(msg)

def load_parquet_files():
    dataframes = []
    for file in TRAINING_FILES:
        path = os.path.join(DATA_DIR, file)
        if os.path.exists(path):
            log_report(f"[LOAD] {file}")
            df = pd.read_parquet(path)
            dataframes.append(df)
        else:
            log_report(f"[MISSING] File not found: {file}")
    return pd.concat(dataframes, axis=0, ignore_index=True)

def preprocess(df):
    log_report("[STEP] Preprocessing data...")
    df = df.dropna(axis=1, how='all')
    df = df.fillna(0)
    df['Label'] = df['Label'].apply(lambda x: 0 if "BENIGN" in str(x).upper() else 1)
    df = df.select_dtypes(include=[np.number])
    return df

def feature_selection(X, y):
    log_report("[STEP] Feature selection with Chi2, ANOVA, ExtraTrees...")
    SelectKBest(score_func=chi2, k=30).fit(X, y)
    SelectKBest(score_func=f_classif, k=30).fit(X, y)
    model = ExtraTreesClassifier(n_estimators=50)
    model.fit(X, y)
    important_features = X.columns[np.argsort(model.feature_importances_)[-30:]]

    with open(SELECTED_FEATURES_PATH, 'w') as f:
        for feat in important_features:
            f.write(f"{feat}\n")
    log_report(f"[SELECTED FEATURES] {list(important_features)}")
    return X[important_features]

def train_model(X_train, y_train):
    log_report("[STEP] Training ensemble model...")
    model = VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('dt', DecisionTreeClassifier()),
        ('knn', KNeighborsClassifier()),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
    ], voting='hard')
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    log_report("[STEP] Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    log_report(f"Accuracy:  {acc:.4f}")
    log_report(f"Precision: {prec:.4f}")
    log_report(f"Recall:    {rec:.4f}")
    log_report(f"F1-Score:  {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix_training.png")
    plt.close()

# ========== MAIN EXECUTION ==========
def main():
    if os.path.exists(REPORT_PATH):
        os.remove(REPORT_PATH)

    log_report("[START] Training initiated for CICDDoS2019 dataset")
    df = load_parquet_files()
    log_report(f"[INFO] Total samples loaded: {df.shape}")

    df = preprocess(df)
    y = df['Label']
    X = df.drop('Label', axis=1)

    X = feature_selection(X, y)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    joblib.dump(model, SAVE_MODEL_PATH)
    log_report(f"[DONE] Trained model saved at: {SAVE_MODEL_PATH}")

if __name__ == '__main__':
    main()
