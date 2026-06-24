import json
import networkx as nx
import pandas as pd
from datetime import datetime

# ------------------------- CONFIG ------------------------- #
AGENT_LOGS_FILE = "agent_execution_logs.json"
LABELS_FILE = "agent_fraud_labels.csv"
OUTPUT_FEATURES = "agent_training_data.csv"

# ------------------------- GRAPH BUILDER ------------------------- #
def build_agent_graph(json_path):
    print(f"Loading Multi-Agent Graph from {json_path}...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_path} not found. Please run simulate_agent_graph.py first.")
        return nx.MultiDiGraph(), []
        
    G = nx.MultiDiGraph() # MultiDiGraph because agents can interact multiple times
    
    # Add Nodes
    for node in data.get("nodes", []):
        G.add_node(node["id"], type=node["type"], privilege=node["privilege"])
        
    # Add Edges (Temporal Events)
    for edge in data.get("edges", []):
        G.add_edge(edge["source"], edge["target"], 
                   event_id=edge["event_id"],
                   trace_id=edge.get("trace_id", "unknown"),
                   action=edge["action"],
                   timestamp=edge["timestamp"],
                   content=edge["content"])
                   
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} interactions.")
    return G, data["edges"]

def extract_features(G, edges):
    """
    Extract temporal and structural features for each event.
    """
    print("Extracting topological features...")
    features = []
    
    if G.number_of_nodes() == 0:
        return pd.DataFrame()
        
    # Calculate structural metrics
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    # PageRank identifies highly trusted/central agents. 
    # Anomaly detection often looks for low-pagerank agents exploiting high-pagerank agents.
    pagerank = nx.pagerank(nx.DiGraph(G)) 
    
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        
        feat = {
            "event_id": edge["event_id"],
            "source": source,
            "target": target,
            "action": edge["action"],
            "src_type": G.nodes[source].get("type", "unknown"),
            "tgt_type": G.nodes[target].get("type", "unknown"),
            "src_privilege": G.nodes[source].get("privilege", "unknown"),
            "tgt_privilege": G.nodes[target].get("privilege", "unknown"),
            "src_out_degree": out_degrees.get(source, 0),
            "tgt_in_degree": in_degrees.get(target, 0),
            "src_pagerank": pagerank.get(source, 0),
            "tgt_pagerank": pagerank.get(target, 0),
            "timestamp": edge["timestamp"],
            "trace_id": edge.get("trace_id", "unknown")
        }
        features.append(feat)
        
    return pd.DataFrame(features)

# ------------------------- MAIN ------------------------- #
def main():
    # 1. Build the graph from the synthetic logs
    G, edges = build_agent_graph(AGENT_LOGS_FILE)
    
    if G.number_of_nodes() == 0:
        return
        
    # 2. Extract Graph Features
    df_features = extract_features(G, edges)
    
    # 3. Load Labels and Merge
    print(f"Loading labels from {LABELS_FILE}...")
    try:
        df_labels = pd.read_csv(LABELS_FILE)
    except FileNotFoundError:
        print(f"Error: {LABELS_FILE} not found.")
        return
        
    # 4. Merge Features with Labels
    df_final = pd.merge(df_features, df_labels, on="event_id", how="inner")
    
    # 5. Save the processed dataset ready for ML Training
    df_final.to_csv(OUTPUT_FEATURES, index=False)
    print(f"Processed dataset saved to {OUTPUT_FEATURES}!")
    print("\nDataset Preview:")
    print(df_final[['event_id', 'source', 'target', 'action', 'is_anomaly']].head())

if __name__ == "__main__":
    main()
