import pandas as pd
import numpy as np
import random
import time
import string
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Configurations
ROWS = 2000
FLOW_TIMEOUTS = [5, 10, 15, 20, 30, 60, 90, 120]
FLAGS_RANGE = list(range(0, 11))
PROTOCOLS = [1, 6, 17]  # ICMP, TCP, UDP

ATTACK_LABELS = [
    "BENIGN",
    "DDoS",
    "PortScan",
    "UDP Flood",
    "TCP SYN Flood",
    "ICMP Flood",
    "Slowloris",
    "BruteForce",
    "Botnet"
]

FEATURES = [
    'tp_src', 'tp_dst', 'ip_proto', 'icmp_code', 'icmp_type',
    'flow_duration_sec', 'flow_duration_nsec',
    'idle_timeout', 'hard_timeout', 'flags',
    'packet_count', 'byte_count',
    'packet_count_per_second', 'packet_count_per_nsecond',
    'byte_count_per_second', 'byte_count_per_nsecond'
]

# Advanced Traffic Pattern Generator
def simulate_flow():
    proto = random.choice(PROTOCOLS)
    duration_sec = round(random.uniform(0.1, 30), 4)
    duration_nsec = round(random.uniform(1e3, 1e6), 4)

    packet_count = random.randint(1, 20000)
    byte_count = random.randint(200, 5_000_000)

    return {
        'tp_src': random.randint(1024, 65535),
        'tp_dst': random.choice([21, 22, 25, 53, 80, 110, 123, 443, 8080]),
        'ip_proto': proto,
        'icmp_code': -1 if proto != 1 else random.choice([0, 3, 5]),
        'icmp_type': -1 if proto != 1 else random.choice([0, 8, 11]),
        'flow_duration_sec': duration_sec,
        'flow_duration_nsec': duration_nsec,
        'idle_timeout': random.choice(FLOW_TIMEOUTS),
        'hard_timeout': random.choice(FLOW_TIMEOUTS),
        'flags': random.choice(FLAGS_RANGE),
        'packet_count': packet_count,
        'byte_count': byte_count,
        'packet_count_per_second': round(packet_count / duration_sec, 4),
        'packet_count_per_nsecond': round(packet_count / duration_nsec, 4),
        'byte_count_per_second': round(byte_count / duration_sec, 4),
        'byte_count_per_nsecond': round(byte_count / duration_nsec, 4),
        'label': random.choices(ATTACK_LABELS, weights=[0.45, 0.2, 0.1, 0.05, 0.05, 0.05, 0.04, 0.03, 0.03])[0]
    }

# 1. Generate Dataset
print("📦 Generating advanced synthetic flow traffic...")
data = [simulate_flow() for _ in range(ROWS)]
df = pd.DataFrame(data)
df.to_csv("FlowStatsfile.csv", index=False)
print("✅ FlowStatsfile.csv created.")

# 2. Preprocessing
print("🔧 Preprocessing and training ensemble model...")
X = df[FEATURES].copy()
y = df['label']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.25, random_state=42)

# 4. Define multiple base models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100),
    "AdaBoost": AdaBoostClassifier(n_estimators=100),
    "DecisionTree": DecisionTreeClassifier(max_depth=10),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel='rbf', probability=True)
}

# 5. Evaluate each model individually
for name, model in models.items():
    print(f"\n⚙️ Training: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ {name} Accuracy: {acc * 100:.2f}%")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 6. Voting Ensemble
ensemble = VotingClassifier(estimators=[
    ('rf', models['RandomForest']),
    ('gb', models['GradientBoosting']),
    ('adb', models['AdaBoost']),
    ('dt', models['DecisionTree'])
], voting='soft')

ensemble.fit(X_train, y_train)
y_ensemble_pred = ensemble.predict(X_test)
acc_ensemble = accuracy_score(y_test, y_ensemble_pred)

print("\n🎯 Ensemble VotingClassifier Performance:")
print(f"🔥 Final Accuracy: {acc_ensemble * 100:.2f}%")
print(confusion_matrix(y_test, y_ensemble_pred))
print(classification_report(y_test, y_ensemble_pred, target_names=label_encoder.classes_))

# 7. Save Encoded Labels + Feature Set
feature_meta = {
    "features": FEATURES,
    "label_encoder": dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
}

meta_path = "feature_metadata.json"
pd.Series(feature_meta).to_json(meta_path)
print(f"🧠 Feature metadata saved to {meta_path}")

# 8. Export test predictions
test_df = pd.DataFrame(X_test, columns=FEATURES)
test_df['label'] = label_encoder.inverse_transform(y_test)
test_df['predicted'] = label_encoder.inverse_transform(y_ensemble_pred)
test_df.to_csv("test_predictions.csv", index=False)
print("📄 Exported test predictions to test_predictions.csv")
