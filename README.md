# Temporal Heterogeneous Graph Transformers (THGT) for Agentic AI Security

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch-Geometric-orange)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph-green)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow)

## 📌 The Core Problem: The "Temporal Confused Deputy"

As Large Language Models (LLMs) evolve into autonomous Multi-Agent Systems (MAS), traditional security perimeters (like prompt sanitization) are failing. Vulnerabilities in MAS are fundamentally **structural and temporal**. 

When an attacker successfully injects a malicious prompt into an external-facing agent (e.g., a Web Scraper), the payload often lies dormant. Steps later, a highly privileged internal agent (e.g., a Database Admin) may read the scraper's context and blindly execute a destructive command. Because the malicious instruction mutates as it crosses agent boundaries, standard static ML analysis fails to detect it.

## 🚀 Our Novel Solution

We model the Multi-Agent System's execution trace as a **Dynamic Heterogeneous Graph**:
*   **Nodes ($V$)**: Represent heterogeneous entities (Planner Agents, Web Tools, Vector DBs, Users).
*   **Edges ($E$)**: Represent temporal interactions (API calls, messages, context writes).

We introduce a **Temporal Heterogeneous Graph Transformer (THGT)** built on PyTorch Geometric. Our architecture explicitly tracks the inter-agent trust exploitation gap using:
1.  **Persistent Node Memory Banks (GRU)**: Maintains the evolving state of an agent. A "contaminated" agent carries the malicious context forward across dozens of intervening benign events.
2.  **Sequence Attention (Event Context Transformer)**: Explicitly attends back in time to catch dormant payloads interacting with active tool calls.
3.  **Heterogeneous Edge Types**: Separate learned weights because an anomaly on an `Agent->Tool` edge is structurally different from an `Agent->Agent` delegation.

---

## 📁 Repository Architecture

Our pipeline takes raw multi-agent execution traces and converts them into machine-learning-ready temporal datasets.

### 1. Data Generation & Red Teaming
*   `simulate_agent_graph.py`: A Python simulation engine that generates thousands of agentic workflows. It injects random "Red Team" prompt propagation attacks (simulating the Telephone Effect and Confused Deputy exploits).
*   *Outputs:* `agent_execution_logs.json` and `agent_fraud_labels.csv`

### 2. Topological Feature Extraction
*   `agent_graph_loader.py`: Parses the JSON trace data into a `networkx` Multi-Agent Graph. It calculates structural trust metrics (Agent PageRank, In-Degree, Out-Degree) to identify agents that are central to the ecosystem.
*   *Outputs:* `agent_training_data.csv`

### 3. Machine Learning Models
*   `train.py` **(The Baseline):** A standard Deep Neural Network (MLP) trained on the static topological features. It intentionally serves as our baseline. *Finding: It achieves ~88% accuracy but entirely fails on Recall (misses >75% of attacks) because it ignores temporal sequences.*
*   `train_thgt.py` **(The State-of-the-Art Solution):** The PyTorch Geometric THGT model. It processes the event stream chronologically, updating recurrent states to correctly identify dormant payloads before catastrophic tool execution occurs.

---

## 🛠️ Installation & Setup

This repository is optimized for Arch Linux (CachyOS) but runs on any modern Python environment.

### 1. Install System Dependencies (Arch/CachyOS)
```bash
sudo pacman -Sy python-pytorch python-pip python-scikit-learn python-networkx python-pandas
```

### 2. Install PyTorch Geometric
```bash
pip install torch_geometric --break-system-packages
```

---

## 🏃 Quick Start Guide

**1. Generate the Synthetic Agent Trace Dataset:**
```bash
python3 simulate_agent_graph.py
```

**2. Extract Graph Topological Features:**
```bash
python3 agent_graph_loader.py
```

**3. Run the THGT Model Evaluation:**
```bash
python3 train_thgt.py
```

---

## 🔬 Research & Evaluation Notes

Because of extreme class imbalance (attacks are rare), **Accuracy is a lying metric**. Our research evaluations focus exclusively on **PR-AUC** (Precision-Recall Area Under Curve) and Recall-at-fixed-Precision. 

Additionally, `train_thgt.py` uses chronological streaming. To prevent target leakage, the THGT architecture strictly updates memory states *after* making predictions, ensuring the model never "sees the future."