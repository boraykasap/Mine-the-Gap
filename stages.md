# 📄 Phases of Anomaly Detection in the Temporal Graph

## 1. Mathematical / Algorithmic Phase

The first step focuses on the **graph structure itself**, independent of the meaning of the nodes or edges.
The objective is to assign **numerical anomaly scores** that capture unusual behavior.

* **Node-level anomalies**
  A node is considered anomalous if its representation changes strongly between two snapshots.

  * Example metrics: embedding drift, degree variation, change in clustering coefficient.
  * Output: a score for each node, higher means more unusual.

* **Edge-level anomalies**
  An edge is anomalous if its presence or absence is surprising compared to past patterns.

  * If an edge exists but has low predicted probability → anomalous.
  * If an edge is missing but has high predicted probability → anomalous.
  * Output: a score for each edge.

* **Graph-level anomalies**
  The entire graph is anomalous if its global structure shifts significantly.

  * Measures: divergence in degree distribution, change in number of components, variation in triad density.
  * Output: a score per snapshot.

👉 At this phase the work is purely structural: detect deviations in how the graph evolves over time.

---

## 2. Semantic / Interpretative Phase

Once anomaly scores are computed, they can be **translated into domain meaning**.
This phase connects the numbers to the context of the network.

* **Node anomaly** → could mean a device or service that suddenly changes its connections (possible failure, misconfiguration).
* **Edge anomaly** → could mean a relationship that should not exist or one that disappeared unexpectedly.
* **Graph anomaly** → could mean a larger event affecting the entire network (fragmentation, sudden new clusters).

👉 This phase provides the human-understandable explanation of what the anomalies might represent in the real system.

---

## 3. Combined Workflow

1. Build snapshots of the graph from event data.
2. Compute embeddings and derive anomaly scores at node, edge, and graph levels.
3. Present anomaly scores in tables and visualizations.
4. Interpret anomalies in terms of network behavior for presentation and decision-making.
