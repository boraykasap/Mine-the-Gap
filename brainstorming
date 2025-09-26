# 📄 Project Description for the Math/Algorithm Team

## 1. Dataset

We are given a CSV file representing the evolution of a **temporal graph**.
Each row describes a graph event:

```
src, dst, label, timestamp, event_type
```

* **src, dst**: graph nodes (e.g. router, port, concentrator).
* **label**: edge type (e.g. `HAS_PORT`, `DEPENDS_ON`, `INSTALLED_AT`).
* **timestamp**: time of the event.
* **event_type**: graph operation (`add` or `remove`).

👉 This defines a **dynamic heterogeneous graph**: edges are added or removed over time.

---

## 2. Goal

Develop an **unsupervised anomaly detection method** that identifies unusual changes during the graph evolution.

Types of anomalies:

* **Node-level**: suspicious node behavior.
* **Edge-level**: improbable or missing connections.
* **Graph-level**: unusual structural shifts.

Output:

* **Node anomaly scores**.
* **Edge anomaly scores**.
* **Graph anomaly scores**.
* A **UI prototype** to visualize and interact with anomalies.

---

## 3. Proposed Approach

### 3.1 Snapshot Construction

* Partition events into time windows (e.g. every Δt or every N events).
* For each window, construct the graph:

  $$
  G_t = (V_t, E_t)
  $$
* Result: a sequence of snapshots `[G_0, G_1, …, G_T]`.

---

### 3.2 Node Embeddings

* For each snapshot, compute node representations using a GNN (e.g. **GraphSAGE** or **RGCN**).
* Each node has an embedding:

  $$
  z_t(v) \in \mathbb{R}^d
  $$

---

### 3.3 Anomaly Scores

#### Node-level

* **Embedding drift**:

  $$
  S_\text{emb}(v) = \| z_t(v) - z_{t-1}(v) \|_2
  $$
* **Local feature changes** (e.g. degree, clustering coefficient):

  $$
  \Delta deg(v) = deg_t(v) - deg_{t-1}(v)
  $$
* Final node score = weighted combination of drift + local changes.

#### Edge-level

* Train a **link prediction model** on `G_{t-1}` (positive = existing edges, negative = sampled non-edges).
* Estimate probability $p(u,v)$ for an edge.
* Define anomaly score:

  * If edge exists in `G_t` but $p(u,v)$ is very low → suspicious.
  * If edge is missing in `G_t` but $p(u,v)$ is very high → suspicious.

#### Graph-level

* Compare distributions and global properties between snapshots:

  * Degree distribution divergence (KL divergence):

    $$
    S_{deg} = KL(P_{deg}^t \;\|\; P_{deg}^{t-1})
    $$
  * Change in number of connected components.
  * Change in triad density.
* Final graph score = weighted sum of these measures.

---

## 4. User Interface & Interaction

To make the results accessible and highlight their usefulness, we add a **lightweight UI** (e.g. Streamlit or Gradio). The UI should:

* **Overview tab**: show high-level KPIs (# node anomalies, # edge anomalies, graph anomaly score) and a simple time series chart of global anomaly scores.
* **Node anomalies tab**: table of top-K suspicious nodes with scores and basic explanations (e.g. degree change, embedding drift). Clicking a node reveals its local ego-network.
* **Edge anomalies tab**: table of suspicious edges (unexpected or missing) with anomaly scores.
* **Details view**: for a selected node/edge, show reasons for anomaly and its local subgraph.
* **Natural language query (optional)**: allow users to type simple queries like *“show suspicious routers in the last 10 minutes”*. This is parsed into filters and displays the corresponding results.

The UI does not need to be complex — its purpose is to demonstrate how anomaly scores can be inspected and used by non-ML experts.

---

## 5. Output

The system must produce:

1. **Node anomalies**: scores for each node.
2. **Edge anomalies**: scores for each edge.
3. **Graph anomalies**: scores for each snapshot.
4. **Interactive UI**: tables and simple graphs for exploration.

These scores provide a **ranking of anomalies** (higher = more suspicious).

---

## 6. Why This Fits the Evaluation Criteria

* **Technical impressiveness**: combines graph embeddings, link prediction, and temporal anomaly detection.
* **Reasonableness**: core algorithms (embedding drift, link prediction, graph statistics) are realistic to implement in the hackathon timeframe.
* **Innovation**: multi-level anomaly detection + lightweight natural language interface.
* **Prototype quality**: anomaly tables + ego-graph views are functional and demonstrate value.
* **Presentation**: results are interpretable and easy to show in slides/demo.
* **Integrity**: everything is computed from the given dataset, with transparent methods.

---

## 7. Why Unsupervised

We do not have labeled anomalies in the dataset.
The method learns typical graph patterns and highlights statistically improbable deviations.
This allows the system to detect **previously unseen anomalies** without explicit supervision.
