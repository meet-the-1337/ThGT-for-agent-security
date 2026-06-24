import pandas as pd
import numpy as np
from collections import deque
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, auc, classification_report, confusion_matrix
import json

def engineer_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 1. time_since_dst_last_event
    df['dst_last_time'] = df.groupby('target')['timestamp'].shift(1)
    df['time_since_dst_last_event'] = (df['timestamp'] - df['dst_last_time']).dt.total_seconds().fillna(0)

    # 2. time_since_src_last_event
    df['src_last_time'] = df.groupby('source')['timestamp'].shift(1)
    df['time_since_src_last_event'] = (df['timestamp'] - df['src_last_time']).dt.total_seconds().fillna(0)

    # 3, 4, 5, 6
    dst_incoming_count_last_20 = []
    dst_unique_sources_last_50 = []
    src_to_dst_first_contact = []
    hops_since_external_entry = []

    last_20_targets = deque(maxlen=20)
    last_50_edges = deque(maxlen=50) # store (src, dst)
    seen_edges = set()
    
    last_external_idx = 0
    
    # Loop over rows to calculate past-only features
    for i, row in df.iterrows():
        src = row['source']
        dst = row['target']
        src_type = row['src_type']
        
        # dst_incoming_count_last_20
        dst_incoming_count_last_20.append(sum(1 for t in last_20_targets if t == dst))
        last_20_targets.append(dst)
        
        # dst_unique_sources_last_50
        unique_srcs = len(set(s for s, d in last_50_edges if d == dst))
        dst_unique_sources_last_50.append(unique_srcs)
        last_50_edges.append((src, dst))
        
        # src_to_dst_first_contact
        edge = (src, dst)
        if edge not in seen_edges:
            src_to_dst_first_contact.append(1)
            seen_edges.add(edge)
        else:
            src_to_dst_first_contact.append(0)
            
        # hops_since_external_entry
        # Approximation: count rows since last event where source was user/scraper
        is_external = src_type == 'user' or 'scraper' in str(src_type).lower() or 'scraper' in src.lower()
        if is_external:
            last_external_idx = i
        hops_since_external_entry.append(i - last_external_idx)

    df['dst_incoming_count_last_20'] = dst_incoming_count_last_20
    df['dst_unique_sources_last_50'] = dst_unique_sources_last_50
    df['src_to_dst_first_contact'] = src_to_dst_first_contact
    df['hops_since_external_entry'] = hops_since_external_entry

    return df

def main():
    print("Loading data...")
    df = pd.read_csv("agent_training_data.csv")
    
    print("Engineering features (this may take a few seconds)...")
    df = engineer_features(df)
    
    # Chronological 80/20 split
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    # 7. action_rarity_for_dst (from training data only)
    train_counts = df_train.groupby(['tgt_type', 'action']).size().reset_index(name='count')
    train_counts['action_rarity_for_dst'] = 1.0 / (train_counts['count'] + 1)
    
    df_train = df_train.merge(train_counts[['tgt_type', 'action', 'action_rarity_for_dst']], on=['tgt_type', 'action'], how='left')
    df_test = df_test.merge(train_counts[['tgt_type', 'action', 'action_rarity_for_dst']], on=['tgt_type', 'action'], how='left')
    df_train['action_rarity_for_dst'] = df_train['action_rarity_for_dst'].fillna(1.0)
    df_test['action_rarity_for_dst'] = df_test['action_rarity_for_dst'].fillna(1.0)
    
    feature_cols = [
        'src_out_degree', 'tgt_in_degree', 'src_pagerank', 'tgt_pagerank',
        'time_since_dst_last_event', 'time_since_src_last_event',
        'dst_incoming_count_last_20', 'dst_unique_sources_last_50',
        'src_to_dst_first_contact', 'hops_since_external_entry',
        'action_rarity_for_dst'
    ]
    
    X_train = df_train[feature_cols]
    y_train = df_train['is_anomaly']
    
    X_test = df_test[feature_cols]
    y_test = df_test['is_anomaly']
    
    # Sanity checks
    print("\n--- Sanity Checks ---")
    train_pos_pct = y_train.mean() * 100
    test_pos_pct = y_test.mean() * 100
    print(f"Train split positive rate: {train_pos_pct:.2f}%")
    print(f"Test split positive rate: {test_pos_pct:.2f}%")
    if abs(train_pos_pct - test_pos_pct) > 5.0:
        print("FLAG: Test set positive rate differs wildly from train. Results may be misleading due to time boundary.")
        
    has_nan = X_train.isna().sum().sum() + X_test.isna().sum().sum()
    if has_nan > 0:
        print(f"WARNING: Found {has_nan} NaN values in engineered features.")
    else:
        print("Passed: No NaN/Inf values in engineered feature columns (NaNs from rolling windows filled with 0).")
        
    # Validation split for early stopping (last 15% of train chronologically)
    val_split_idx = int(len(X_train) * 0.85)
    X_train_final = X_train.iloc[:val_split_idx]
    y_train_final = y_train.iloc[:val_split_idx]
    X_val = X_train.iloc[val_split_idx:]
    y_val = y_train.iloc[val_split_idx:]
    
    # Class weights for XGBoost
    num_neg = (y_train_final == 0).sum()
    num_pos = (y_train_final == 1).sum()
    scale_pos_weight = num_neg / max(num_pos, 1)
    
    print("\nTraining XGBoost Classifier...")
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
    
    # Evaluation
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]
    
    print("\n--- XGBoost Model Evaluation ---")
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall, precision)
    print(f"1. PR-AUC: {pr_auc:.4f}")
    
    # Find recall at precision >= 0.90
    high_prec_idx = np.where(precision >= 0.90)[0]
    if len(high_prec_idx) > 0:
        # First index where precision >= 0.90 (since precision generally increases with threshold)
        # We want the maximum recall where precision >= 0.90
        valid_recalls = recall[high_prec_idx]
        recall_at_90_prec = np.max(valid_recalls)
        print(f"2. Recall at fixed precision >= 0.90: {recall_at_90_prec:.4f}")
    else:
        print("2. Recall at fixed precision >= 0.90: N/A (model never reached 0.90 precision)")
        
    print("\n3. Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Anomaly']))
    
    cm = confusion_matrix(y_test, y_pred)
    print("4. Confusion Matrix:")
    print(f"True Negatives:  {cm[0][0]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")
    print(f"True Positives:  {cm[1][1]}")
    
    print("\n5. Top Feature Importances:")
    importances = model.feature_importances_
    feature_imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
    feature_imp_df = feature_imp_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    print(feature_imp_df.head(15).to_string())
    
    # Save model
    model.save_model("xgb_baseline_model.json")
    print("\nModel saved to xgb_baseline_model.json")

if __name__ == "__main__":
    main()
