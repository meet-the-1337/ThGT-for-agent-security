from train_xgb_baseline import engineer_features
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, auc, classification_report, confusion_matrix
import numpy as np

print("Loading data...")
df = pd.read_csv("agent_training_data.csv")

print("Engineering features...")
df = engineer_features(df)

split_idx = int(len(df) * 0.8)
df_train = df.iloc[:split_idx].copy()
df_test = df.iloc[split_idx:].copy()

train_counts = df_train.groupby(['tgt_type', 'action']).size().reset_index(name='count')
train_counts['action_rarity_for_dst'] = 1.0 / (train_counts['count'] + 1)

df_train = df_train.merge(train_counts[['tgt_type', 'action', 'action_rarity_for_dst']], on=['tgt_type', 'action'], how='left')
df_test = df_test.merge(train_counts[['tgt_type', 'action', 'action_rarity_for_dst']], on=['tgt_type', 'action'], how='left')
df_train['action_rarity_for_dst'] = df_train['action_rarity_for_dst'].fillna(1.0)
df_test['action_rarity_for_dst'] = df_test['action_rarity_for_dst'].fillna(1.0)

temporal_only_features = [
    'time_since_dst_last_event', 'time_since_src_last_event',
    'dst_incoming_count_last_20', 'dst_unique_sources_last_50',
    'src_to_dst_first_contact', 'hops_since_external_entry',
    'action_rarity_for_dst'
]

X_train = df_train[temporal_only_features]
y_train = df_train['is_anomaly']

X_test = df_test[temporal_only_features]
y_test = df_test['is_anomaly']

val_split_idx = int(len(X_train) * 0.85)
X_train_final = X_train.iloc[:val_split_idx]
y_train_final = y_train.iloc[:val_split_idx]
X_val = X_train.iloc[val_split_idx:]
y_val = y_train.iloc[val_split_idx:]

num_neg = (y_train_final == 0).sum()
num_pos = (y_train_final == 1).sum()
scale_pos_weight = num_neg / max(num_pos, 1)

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='aucpr',
    random_state=42,
    early_stopping_rounds=30
)

model.fit(
    X_train_final, y_train_final,
    eval_set=[(X_val, y_val)],
    verbose=False
)

y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)[:, 1]

precision, recall, _ = precision_recall_curve(y_test, y_scores)
pr_auc = auc(recall, precision)
print(f"PR-AUC (Ablation - Temporal Only): {pr_auc:.4f}")

high_prec_idx = np.where(precision >= 0.90)[0]
if len(high_prec_idx) > 0:
    recall_at_90_prec = np.max(recall[high_prec_idx])
    print(f"Recall at fixed precision >= 0.90: {recall_at_90_prec:.4f}")
else:
    print("Recall at fixed precision >= 0.90: N/A")

importances = model.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': temporal_only_features, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values('Importance', ascending=False).reset_index(drop=True)
print("\nTop Feature Importances:")
print(feature_imp_df.to_string())
