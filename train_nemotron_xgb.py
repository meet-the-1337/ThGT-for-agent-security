"""
train_nemotron_xgb.py
=====================
XGBoost classifier for detecting attack traces in the
Nemotron-AIQ Agentic Safety Dataset (OpenTelemetry spans).

DATA AUDIT FINDINGS (verified before writing this code):
─────────────────────────────────────────────────────────
Dataset structure across 4 splits:

  security_data/with_defense:    2600 traces — ALL attacks, ALL success=True
  security_data/without_defense: 2596 traces — ALL attacks, ALL success=True
  safety_data/with_defense:      2800 traces — 97 successful attacks, 2503 blocked, 200 benign
  safety_data/without_defense:   2800 traces — 559 successful attacks, 2041 blocked, 200 benign

Schema difference:
  - security_data manifests have: injection_string, template_name (NO attack_category/subcategory)
  - safety_data manifests have: attack_type, attack_category, attack_subcategory
  - Benign traces in safety_data have attack_category=None

Span-level findings:
  - Only 2 span_kinds: CHAIN, LLM (no real heterogeneity)
  - Only 4 span names: ai_researcher, generate_query, generate_summary, nvidia/llm-model
  - Each trace is a fully INDEPENDENT benchmark run (no shared state across traces)

CLASSIFICATION FRAMING:
  We focus on safety_data/with_defense where the most interesting and
  balanced task lives: distinguishing 200 benign + 2503 defended-attack
  traces from 97 traces where the attack succeeded (breached defense).

  Task A: attack_attempted (0) vs benign (1) — can we detect attack INTENT from trace structure?
  Task B: attack_succeeded (1) vs rest (0) — can we detect defense BYPASS from trace structure?

  Both are scored with PR-AUC on the MINORITY class.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    precision_recall_curve, auc, classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

DATA_BASE = "Nemotron-AIQ-Agentic-Safety-Dataset-1.0/data"


# ─────────────────────────────────────────────────────────────
# Step 1: Load manifest labels — FIXED label logic
# ─────────────────────────────────────────────────────────────

def load_manifest_labels():
    """
    Returns a dict: trace_id -> {label, attack_category, defense, dataset, ...}

    Label scheme:
      - 'benign': no attack at all (safety_data controls with attack_category=None)
      - 'blocked': attack attempted but defense blocked it (result.success=False)
      - 'successful_attack': attack breached the defense (result.success=True)
    """
    labels = {}

    # --- security_data: ALL traces are attacks (by dataset design) ---
    for defense in ["with_defense", "without_defense"]:
        path = f"{DATA_BASE}/security_data/{defense}/attack_manifest.jsonl"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                tid = rec["result"].get("trace_id")
                if not tid:
                    continue
                labels[tid] = {
                    "label": "successful_attack" if rec["result"]["success"] else "blocked",
                    "attack_template": rec["attack"].get("template_name", "unknown"),
                    "attack_category": "security_attack",
                    "defense": defense,
                    "dataset": "security_data",
                }

    # --- safety_data: mix of attacks and benign controls ---
    for defense in ["with_defense", "without_defense"]:
        path = f"{DATA_BASE}/safety_data/{defense}/attack_manifest.jsonl"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                tid = rec["result"].get("trace_id")
                if not tid:
                    continue

                cat = rec["attack"].get("attack_category")
                if cat is None:
                    # Genuine benign control trial
                    labels[tid] = {
                        "label": "benign",
                        "attack_template": "",
                        "attack_category": "benign",
                        "defense": defense,
                        "dataset": "safety_data",
                    }
                else:
                    labels[tid] = {
                        "label": "successful_attack" if rec["result"]["success"] else "blocked",
                        "attack_template": rec["attack"].get("attack_subcategory", "unknown"),
                        "attack_category": cat,
                        "defense": defense,
                        "dataset": "safety_data",
                    }

    return labels


# ─────────────────────────────────────────────────────────────
# Step 2: Extract features from each trace
# Each trace = one independent benchmark run → one feature row
# Memory is NOT shared across traces (confirmed disjoint).
# ─────────────────────────────────────────────────────────────

def extract_trace_features(spans):
    """
    Extract structural + temporal features from ONE trace (list of spans).
    """
    for s in spans:
        s["_start"] = pd.to_datetime(s.get("start_time", "2000-01-01"), utc=True)
        s["_end"] = pd.to_datetime(s.get("end_time", "2000-01-01"), utc=True)
        s["_duration"] = (s["_end"] - s["_start"]).total_seconds()

    spans_sorted = sorted(spans, key=lambda x: x["_start"])
    n_spans = len(spans_sorted)

    span_kinds = [s.get("span_kind", "") for s in spans_sorted]
    durations = [max(s["_duration"], 0) for s in spans_sorted]

    n_llm = sum(1 for k in span_kinds if k == "LLM")
    n_chain = sum(1 for k in span_kinds if k == "CHAIN")

    # Depth: max depth of parent-child tree
    span_id_map = {s.get("context.span_id"): i for i, s in enumerate(spans_sorted)}
    depths = {}
    def get_depth(span):
        sid = span.get("context.span_id")
        if sid in depths:
            return depths[sid]
        pid = span.get("parent_id")
        if pid is None:
            depths[sid] = 0
        elif pid in span_id_map:
            depths[sid] = get_depth(spans_sorted[span_id_map[pid]]) + 1
        else:
            depths[sid] = 1
        return depths[sid]

    for s in spans_sorted:
        get_depth(s)

    max_depth = max(depths.values()) if depths else 0
    avg_depth = np.mean(list(depths.values())) if depths else 0

    # Token counts
    prompt_tokens = [s.get("attributes.llm.token_count.prompt") or 0 for s in spans_sorted]
    completion_tokens = [s.get("attributes.llm.token_count.completion") or 0 for s in spans_sorted]
    total_tokens = [s.get("attributes.llm.token_count.total") or 0 for s in spans_sorted]

    # Input/output length (chars)
    input_lengths = [len(str(s.get("attributes.input.value") or "")) for s in spans_sorted]
    output_lengths = [len(str(s.get("attributes.output.value") or "")) for s in spans_sorted]

    # Total wall time of trace
    if n_spans >= 2:
        total_duration = (spans_sorted[-1]["_end"] - spans_sorted[0]["_start"]).total_seconds()
    else:
        total_duration = durations[0] if durations else 0

    # Root span input/output length
    root_spans = [s for s in spans_sorted if s.get("parent_id") is None]
    root_input_len = len(str(root_spans[0].get("attributes.input.value") or "")) if root_spans else 0
    root_output_len = len(str(root_spans[0].get("attributes.output.value") or "")) if root_spans else 0

    # Status codes
    statuses = [s.get("status_code", "") for s in spans_sorted]
    n_error = sum(1 for x in statuses if "error" in str(x).lower())
    n_ok = sum(1 for x in statuses if "ok" in str(x).lower())

    return {
        "n_spans": n_spans,
        "n_llm_spans": n_llm,
        "n_chain_spans": n_chain,
        "llm_ratio": n_llm / max(n_spans, 1),
        "max_depth": max_depth,
        "avg_depth": avg_depth,

        "total_duration_sec": total_duration,
        "avg_span_duration": np.mean(durations) if durations else 0,
        "max_span_duration": max(durations) if durations else 0,
        "min_span_duration": min(durations) if durations else 0,
        "std_span_duration": np.std(durations) if durations else 0,

        "total_prompt_tokens": sum(prompt_tokens),
        "total_completion_tokens": sum(completion_tokens),
        "total_all_tokens": sum(total_tokens),
        "avg_prompt_tokens": np.mean(prompt_tokens) if prompt_tokens else 0,
        "avg_completion_tokens": np.mean(completion_tokens) if completion_tokens else 0,

        "root_input_len": root_input_len,
        "root_output_len": root_output_len,
        "total_input_len": sum(input_lengths),
        "total_output_len": sum(output_lengths),
        "max_input_len": max(input_lengths) if input_lengths else 0,
        "max_output_len": max(output_lengths) if output_lengths else 0,
        "avg_input_len": np.mean(input_lengths) if input_lengths else 0,
        "avg_output_len": np.mean(output_lengths) if output_lengths else 0,

        "output_input_ratio": sum(output_lengths) / max(sum(input_lengths), 1),

        "n_error_spans": n_error,
        "n_ok_spans": n_ok,
        "error_rate": n_error / max(n_spans, 1),
    }


# ─────────────────────────────────────────────────────────────
# Step 3: Build feature DataFrame
# ─────────────────────────────────────────────────────────────

def build_dataset(labels):
    print("Building feature rows from trace files...")
    rows = []
    missing = 0

    for cat in ["security_data", "safety_data"]:
        for defense in ["with_defense", "without_defense"]:
            trace_dir = f"{DATA_BASE}/{cat}/{defense}/traces"
            if not os.path.exists(trace_dir):
                continue
            trace_files = glob.glob(f"{trace_dir}/trace_*.json")
            for tf in trace_files:
                tid = os.path.basename(tf).replace("trace_", "").replace(".json", "")
                if tid not in labels:
                    missing += 1
                    continue
                try:
                    with open(tf) as f:
                        spans = json.load(f)
                    if not spans:
                        continue
                    feats = extract_trace_features(spans)
                    feats["trace_id"] = tid
                    feats["label"] = labels[tid]["label"]
                    feats["dataset"] = labels[tid]["dataset"]
                    feats["defense"] = labels[tid]["defense"]
                    feats["attack_category"] = labels[tid]["attack_category"]
                    rows.append(feats)
                except Exception as e:
                    missing += 1

    print(f"  Loaded {len(rows)} traces ({missing} skipped/missing)")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# Step 4: Train + Evaluate — TWO tasks, scored correctly
# ─────────────────────────────────────────────────────────────

FEATURE_COLS = None  # set dynamically

def run_task(df, task_name, y_col, pos_label_name, pos_label_val):
    """
    Train XGBoost for one binary classification task.
    PR-AUC is always computed for the MINORITY class.
    """
    print(f"\n{'='*60}")
    print(f"  TASK: {task_name}")
    print(f"  Positive class = '{pos_label_name}' (label={pos_label_val})")
    print(f"{'='*60}")

    global FEATURE_COLS
    X = df[FEATURE_COLS]
    y = df[y_col]

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Sanity checks
    train_rate = y_train.mean() * 100
    test_rate = y_test.mean() * 100
    print(f"\n  Train: {len(y_train)} samples, {train_rate:.1f}% positive")
    print(f"  Test:  {len(y_test)} samples, {test_rate:.1f}% positive")

    no_skill_prauc = y_test.mean()  # baseline PR-AUC for positive class
    print(f"  No-skill PR-AUC baseline: {no_skill_prauc:.4f}")

    # Validation slice for early stopping
    val_size = int(len(X_train) * 0.15)
    X_val = X_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]
    X_train_f = X_train.iloc[:-val_size]
    y_train_f = y_train.iloc[:-val_size]

    # Scale pos weight for the MINORITY class
    n_neg = (y_train_f == 0).sum()
    n_pos = (y_train_f == 1).sum()
    spw = n_neg / max(n_pos, 1)
    print(f"  scale_pos_weight: {spw:.4f}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        random_state=42,
        early_stopping_rounds=30,
        device="cuda",
    )
    model.fit(X_train_f, y_train_f, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  Best iteration: {model.best_iteration}")

    # Predictions
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]

    # PR-AUC scored on the POSITIVE (minority) class
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall, precision)
    print(f"\n  1. PR-AUC (for '{pos_label_name}'): {pr_auc:.4f}  (no-skill baseline: {no_skill_prauc:.4f})")
    print(f"     Lift above baseline: {pr_auc - no_skill_prauc:.4f}")

    # Recall at precision >= 0.90
    high_prec_idx = np.where(precision >= 0.90)[0]
    if len(high_prec_idx) > 0:
        recall_at_90 = np.max(recall[high_prec_idx])
        print(f"  2. Recall at Precision >= 0.90: {recall_at_90:.4f}")
    else:
        print("  2. Recall at Precision >= 0.90: N/A")

    print(f"\n  3. Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Negative", pos_label_name]))

    cm = confusion_matrix(y_test, y_pred)
    print(f"  4. Confusion Matrix:")
    print(f"     TN: {cm[0][0]}  FP: {cm[0][1]}")
    print(f"     FN: {cm[1][0]}  TP: {cm[1][1]}")

    print(f"\n  5. Top 15 Feature Importances:")
    imp = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": model.feature_importances_})
    imp = imp.sort_values("Importance", ascending=False).reset_index(drop=True)
    print(imp.head(15).to_string())

    return model, pr_auc


def main():
    global FEATURE_COLS

    print("Loading manifest labels...")
    labels = load_manifest_labels()

    # Print label distribution
    from collections import Counter
    label_counts = Counter(v["label"] for v in labels.values())
    print(f"  Label distribution: {dict(label_counts)}")

    # Build features
    df = build_dataset(labels)

    # Sanity
    print(f"\n--- Sanity Checks ---")
    print(f"DataFrame shape: {df.shape}")
    print(f"Label distribution in df:")
    print(df["label"].value_counts().to_string())
    nan_count = df.isnull().sum().sum()
    print(f"NaN values: {nan_count} (filled with 0)")
    df = df.fillna(0)

    FEATURE_COLS = [c for c in df.columns if c not in
                    ["trace_id", "label", "dataset", "defense", "attack_category",
                     "is_attack", "is_successful_attack"]]

    # ══════════════════════════════════════════════════════════
    # TASK A: Detect BENIGN traces (minority) among all traces
    # Binary: benign=1, everything else=0
    # ══════════════════════════════════════════════════════════
    df["is_benign"] = (df["label"] == "benign").astype(int)

    model_a, prauc_a = run_task(
        df, "Detect Benign Traces (minority class)", "is_benign",
        pos_label_name="Benign", pos_label_val=1
    )

    # ══════════════════════════════════════════════════════════
    # TASK B: Detect SUCCESSFUL attacks (defense bypass)
    # Binary: successful_attack=1, everything else=0
    # This is the scientifically interesting question.
    # ══════════════════════════════════════════════════════════
    df["is_successful_attack"] = (df["label"] == "successful_attack").astype(int)

    model_b, prauc_b = run_task(
        df, "Detect Successful Attacks / Defense Bypass (minority in safety_data)",
        "is_successful_attack",
        pos_label_name="Successful Attack", pos_label_val=1
    )

    # ══════════════════════════════════════════════════════════
    # TASK C (ablation): Same as Task A but ONLY on safety_data
    #   Removes the confound of security_data vs safety_data folder
    # ══════════════════════════════════════════════════════════
    df_safety = df[df["dataset"] == "safety_data"].copy()
    df_safety["is_benign"] = (df_safety["label"] == "benign").astype(int)

    model_c, prauc_c = run_task(
        df_safety,
        "ABLATION: Detect Benign — safety_data ONLY (removes folder confound)",
        "is_benign",
        pos_label_name="Benign", pos_label_val=1
    )

    # ══════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"  Task A — Detect Benign (all data):           PR-AUC = {prauc_a:.4f}")
    print(f"  Task B — Detect Successful Attack (all data): PR-AUC = {prauc_b:.4f}")
    print(f"  Task C — Detect Benign (safety_data only):    PR-AUC = {prauc_c:.4f}")
    print()

    # Save best model
    model_b.save_model("nemotron_xgb_model.json")
    print("  Task B model saved to nemotron_xgb_model.json")
    df.to_csv("nemotron_features.csv", index=False)
    print("  Feature table saved to nemotron_features.csv")


if __name__ == "__main__":
    main()
