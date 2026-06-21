import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

DATA_FILE = "agent_training_data.csv"

def load_and_preprocess_data():
    print(f"Loading Agentic Graph dataset from {DATA_FILE}...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found. Please run agent_graph_loader.py first.")
        return None, None
        
    # Drop IDs and raw timestamps as they aren't features for the NN
    # We might use timestamp differences in future iterations for Temporal GNNs
    drop_cols = ['event_id', 'source', 'target', 'timestamp']
    df_features = df.drop(columns=drop_cols, errors='ignore')
    
    # Separate Labels
    if 'is_anomaly' not in df_features.columns:
        print("Error: 'is_anomaly' label missing from dataset.")
        return None, None
        
    y = df_features['is_anomaly']
    X_raw = df_features.drop(columns=['is_anomaly'])
    
    # Encode categorical features (actions, privileges, types)
    categorical_cols = ['action', 'src_type', 'tgt_type', 'src_privilege', 'tgt_privilege']
    X_encoded = X_raw.copy()
    
    label_encoders = {}
    for col in categorical_cols:
        if col in X_encoded.columns:
            le = LabelEncoder()
            # Convert to string to handle any NaNs or mixed types safely
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            label_encoders[col] = le
            
    # Scale numerical features (degrees, pagerank)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_encoded), columns=X_encoded.columns)
    
    return X_scaled, y

def train_baseline_model():
    X, y = load_and_preprocess_data()
    if X is None:
        return
        
    print(f"\n✅ Dataset loaded successfully: {len(X)} total interactions.")
    print(f"   - Anomalous (Prompt Injection/Exploit): {y.sum()}")
    print(f"   - Benign (Normal Behavior): {len(y) - y.sum()}")
    
    # Stratified split to ensure both classes are represented
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print("\nTraining Deep Neural Network (MLP) on Structural Graph Features...")
    # We use a Multi-Layer Perceptron as our baseline. 
    # In the future, this will be replaced with PyTorch Geometric (THGT)
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32), 
        activation='relu', 
        solver='adam', 
        max_iter=300, 
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    
    print("\n================================")
    print("      MODEL EVALUATION")
    print("================================")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Anomaly']))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Negatives (Correctly Benign): {cm[0][0]}")
    print(f"False Positives (False Alarms):    {cm[0][1]}")
    print(f"False Negatives (Missed Attacks):  {cm[1][0]}")
    print(f"True Positives (Blocked Attacks):  {cm[1][1]}")
    print("================================\n")
    print("Note: This is our structural baseline model. The next step in the research")
    print("is migrating this to PyTorch Geometric to create the Temporal Graph Transformer.")

if __name__ == "__main__":
    train_baseline_model()
