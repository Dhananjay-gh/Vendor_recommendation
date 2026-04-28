import pandas as pd
import numpy as np
import os
import json
import joblib
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Set directories
DATA_DIR = os.path.join(os.path.dirname(__file__), 'files')
SAVE_DIR = os.path.dirname(__file__)

print("Loading SAP Data...")
# Load SAP Tables
df_lfa1 = pd.read_csv(os.path.join(DATA_DIR, 'LFA1_Vendor_Master_General.csv'), 
                      low_memory=False, usecols=['LIFNR', 'NAME1', 'RISK_CLASS'])
df_bsak = pd.read_csv(os.path.join(DATA_DIR, 'BSAK_Cleared_Items.csv'), 
                      low_memory=False, usecols=['LIFNR', 'WRBTR', 'DAYS_OVERDUE_AT_CLEAR', 'CLEARED_LATE', 'BELNR'])
df_bsik = pd.read_csv(os.path.join(DATA_DIR, 'BSIK_Open_Items.csv'), 
                      low_memory=False, usecols=['LIFNR', 'WRBTR', 'BELNR'])

print("Processing Historical Data (BSAK)...")
# Process BSAK (Historical Cleared Items)
df_bsak['CLEARED_LATE_NUM'] = df_bsak['CLEARED_LATE'].apply(lambda x: 1 if x == 'X' else 0)
df_bsak['SPEND_VOL'] = df_bsak['WRBTR'].abs()

bsak_agg = df_bsak.groupby('LIFNR').agg({
    'DAYS_OVERDUE_AT_CLEAR': 'mean',
    'CLEARED_LATE_NUM': 'mean',
    'SPEND_VOL': 'sum',
    'BELNR': 'count'
}).reset_index()

bsak_agg.rename(columns={
    'DAYS_OVERDUE_AT_CLEAR': 'avg_days_overdue_hist',
    'CLEARED_LATE_NUM': 'late_ratio',
    'SPEND_VOL': 'total_spend_vol',
    'BELNR': 'transaction_count'
}, inplace=True)

print("Processing Open Items (BSIK)...")
# Process BSIK (Current Exposure)
df_bsik['OPEN_VOL'] = df_bsik['WRBTR'].abs()
bsik_agg = df_bsik.groupby('LIFNR').agg({
    'OPEN_VOL': 'sum',
    'BELNR': 'count'
}).reset_index()

bsik_agg.rename(columns={
    'OPEN_VOL': 'open_exposure',
    'BELNR': 'open_count'
}, inplace=True)

print("Merging Dataset...")
# Merge Everything
df_master = df_lfa1.merge(bsak_agg, on='LIFNR', how='inner') # Only train on vendors with history
df_master = df_master.merge(bsik_agg, on='LIFNR', how='left')

# Fill missing open exposure with 0
df_master['open_exposure'] = df_master['open_exposure'].fillna(0)
df_master['open_count'] = df_master['open_count'].fillna(0)
df_master['avg_days_overdue_hist'] = df_master['avg_days_overdue_hist'].fillna(0)

# Map target Risk Class
# Assuming A is low risk, D is critical risk
risk_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
df_master['RISK_CLASS_LABEL'] = df_master['RISK_CLASS'].map(risk_map)

# Drop rows with missing TARGET risk class
df_master = df_master.dropna(subset=['RISK_CLASS_LABEL'])

print(f"Final Dataset Size for Training: {df_master.shape}")

# Define ML features
features = ['avg_days_overdue_hist', 'late_ratio', 'total_spend_vol', 'open_exposure']
X = df_master[features].values
y = df_master['RISK_CLASS_LABEL'].astype(int).values

# Stratified 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit StandardScaler on X_train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.transform(X)   # full dataset for K-Means

print("Training XGBoost Classifier...")
xgb = XGBClassifier(
    n_estimators=100, 
    max_depth=4, 
    learning_rate=0.1, 
    eval_metric='mlogloss', 
    random_state=42
)
xgb.fit(X_train_scaled, y_train)

# Evaluate on held-out test set
test_preds = xgb.predict(X_test_scaled)
real_acc = np.mean(test_preds == y_test)
print(f"XGBoost Real Test Accuracy: {real_acc:.2%}")
print(classification_report(y_test, test_preds, target_names=['A','B','C','D']))

# Save real accuracy to model_metrics.json
with open(os.path.join(SAVE_DIR, 'model_metrics.json'), 'w') as f:
    json.dump({"xgb_test_accuracy": round(real_acc * 100, 1)}, f)

print("Generating K-Means Elbow Plot...")
# Elbow plot to validate k=4
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

inertias = []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 4), facecolor='#050810')
ax.set_facecolor('#0a0e1a')
ax.plot(list(k_range), inertias, color='#14f0a0', linewidth=2, marker='o',
        markerfacecolor='#14f0a0', markersize=6)
ax.axvline(x=4, color='#f0b840', linestyle='--', alpha=0.6, label='Current k=4')
ax.set_xlabel('Number of Clusters (k)', color='#94a3b8')
ax.set_ylabel('Inertia', color='#94a3b8')
ax.set_title('K-Means Elbow Plot — Validate Cluster Count', color='#f0f4ff')
ax.tick_params(colors='#94a3b8')
ax.legend(facecolor='#0a0e1a', labelcolor='#94a3b8')
for spine in ax.spines.values():
    spine.set_edgecolor('#2a3a5a')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'kmeans_elbow_plot.png'), dpi=120, facecolor='#050810')
plt.close()
print("Elbow plot saved → kmeans_elbow_plot.png. Review to validate k=4.")

print("Training K-Means Risk Clusters...")
# Train K-Means on full X_scaled (unsupervised — no train/test split needed)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_master['KMEANS_CLUSTER'] = kmeans.fit_predict(X_scaled)

unique, counts = np.unique(df_master['KMEANS_CLUSTER'], return_counts=True)
print("K-Means cluster sizes:", dict(zip(unique, counts)))

from sklearn.ensemble import IsolationForest

print("Training Isolation Forest Anomaly Detector...")
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,   # assume ~5% of vendors are genuine outliers
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_scaled)   # use the already-computed X_scaled from StandardScaler

# Score all vendors — lower (more negative) = more anomalous
# decision_function returns raw anomaly scores; convert to 0-1 range for usability
raw_scores = iso_forest.decision_function(X_scaled)          # negative = anomalous
anomaly_labels = iso_forest.predict(X_scaled)                # -1 = outlier, 1 = normal

# Normalize scores to 0-1 where 1.0 = most anomalous
min_s, max_s = raw_scores.min(), raw_scores.max()
normalized_scores = 1 - ((raw_scores - min_s) / (max_s - min_s))
normalized_scores = np.clip(normalized_scores, 0.0, 1.0)

df_master['ISOLATION_SCORE'] = np.round(normalized_scores, 4)
df_master['IS_OUTLIER'] = (anomaly_labels == -1).astype(int)

n_outliers = (anomaly_labels == -1).sum()
print(f"Isolation Forest complete. Outliers detected: {n_outliers} / {len(df_master)}")

print("Saving Models and Processed Data...")
# Save models and scaler
joblib.dump(xgb, os.path.join(SAVE_DIR, 'xgb_risk_model.pkl'))
joblib.dump(kmeans, os.path.join(SAVE_DIR, 'kmeans_risk_model.pkl'))
joblib.dump(scaler, os.path.join(SAVE_DIR, 'scaler.pkl'))
joblib.dump(iso_forest, os.path.join(SAVE_DIR, 'isolation_forest_model.pkl'))

# Save clean vendor list for the app dropdowns
df_master.to_csv(os.path.join(SAVE_DIR, 'processed_sap_vendors.csv'), index=False)

print("Preprocessing and Training Complete! Saved 4 items:")
print(" - xgb_risk_model.pkl")
print(" - kmeans_risk_model.pkl")
print(" - isolation_forest_model.pkl")
print(" - processed_sap_vendors.csv")
