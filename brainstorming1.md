# Dynamic Graph Anomaly Detection — Data & Modeling Choices (Backend)

This document explains **what we did on the data side**, the reasoning behind each choice, and the exact **artifacts** produced for downstream consumption (LLM + dashboard). UI details are intentionally omitted.

---

## 0) Objective

* **Unsupervised** anomaly detection on a **dynamic, heterogeneous graph** built from add/delete events.
* Cover three levels: **node**, **edge**, and **graph structure**.
* Produce **interpretable, quantitative signals** (scores, labels, metrics) consumable by other systems.

---

## 1) Input Data & Semantics

**Raw CSV schema**

```
src, dst, label, timestamp, event_type
```

* `event_type ∈ {add, delete}`.
* Graph treated as **undirected** for topology (can be switched to directed later).
* `label` is preserved as edge metadata (e.g., HAS_PORT, DEPENDS_ON) to support interpretability and neighbor dumps.

**Why undirected now?**

* For this MVP we focus on structure-driven drift and link plausibility. Undirected simplifies degree, components, clustering, and Node2Vec behavior. If directionality is crucial, we can adopt a directed encoder and adapt metrics.

---

## 2) Snapshotting Strategy (Temporal Batching)

**Choice:** fixed **event-count windows** of **1,000 events** per snapshot.

* Timestamp analysis showed highly variable event density (median≈4, p90≈16 per timestamp, large spike at t=0).
* Fixed wall-clock windows would produce uneven blocks → unstable training and noisy comparisons.
* Event windows ensure **stable per-step signal** and **predictable training time**.

**Result:** 11 full blocks of 1,000 + one remainder (404):

```
data/snapshots/snapshot_0.csv ... snapshot_11.csv
```

**Cumulative semantics:** The state at snapshot `t` is obtained by **applying all events up to and including `t`** on an empty graph (adds then deletes).

---

## 3) Graph Construction per Snapshot

* Maintain a **stable node index** via a global vocabulary:

  * `data/embeddings/node_vocab.json` includes `idx2id` (array) and `id2idx` (map).
  * New nodes are **appended** → indices are monotonic, enabling alignment across time.
* Build undirected `edge_index` by canonicalizing pairs `(u,v)` (sorted) from the cumulatively applied edge set.
* Infer **node type** from `node_id` prefix (e.g., `trunk-*`, `cpe-*`, `agreement-*`). Keep **edge labels** when present for interpretability.

---

## 4) Node Representation Learning (Per Snapshot)

**Encoder:** PyTorch Geometric **Node2Vec**

* Rationale: robust, unsupervised, fast on GPU, good for local/meso topology.
* Hyperparameters (good trade-off observed):

  * `dim=128`, `walk_length=20`, `walks_per_node=20`, `window=10`
  * `epochs=15`, `batch_size=1024`, `p=1.0`, `q=0.5` (slightly BFS-leaning)
* **Warm-start:** For `t>0`, copy overlapping rows from `Z_{t-1}` into `Z_t` to speed up convergence and smooth temporal geometry.
* Artifacts per `t`:

  * `data/embeddings/Z_t.pt` → `{ "embeddings": FloatTensor[n_t, d] }`
  * `data/embeddings/Z_t_preview.csv` (for quick inspection)
  * Global vocab: `data/embeddings/node_vocab.json`

**Why Node2Vec + warm-start?**

* We need **relative geometry stability** across time steps. Node2Vec is sufficiently stable with warm-start; we then add explicit alignment (next section).

---

## 5) Temporal Alignment & Drift

**Alignment:** **Orthogonal Procrustes**

* Embeddings are identifiable up to rotation. We align `Z_t` to `Z_{t-1}` on the common index range `[0..min(n_{t-1}, n_t)-1]` to obtain a rotation `R_t`.
* Compute per-node **drift** when the node exists in both snapshots:

  $\text{drift}_i(t) = \|\, (Z_t R_t)[i] - Z_{t-1}[i] \,\|_2$

**Degree & Novelty:**

* Recompute **degree** at `t-1` and `t` from cumulatively applied edge sets; $\Delta\deg_i(t) = \deg_i(t) - \deg_i(t-1)$.
* **Novelty:** node appears at `t` but not at `t-1` (by index) or transitions from degree 0 to >0.

These become core node features: `drift`, `degree`, `degree_change`, `novelty`.

---

## 6) Edge Anomalies (Two Classes)

**A) Present-but-improbable** (edge exists at `t` but link plausibility is low)

* Candidate set: edges in `E_t` (optionally downsampled).
* Scoring: derive $p(u,v)$ from embedding similarity (cosine/dot → normalization/sigmoid). Unsupervised prior, no decoder trained.
* Decision: flag if $p(u,v) \leq \tau_{present}$ (default 0.15). Rank by ascending `p`.

**B) Missing-but-expected** (edge absent at `t` but highly plausible or recently removed)

* **Removed edges** between `t-1` and `t`: treat as highly expected (`source="removed_edge"`), assign `p=1.0`.
* **KNN candidates:** for each node, top-k nearest neighbors in aligned space, excluding existing edges (`source="knn_candidate"`). Compute `p(u,v)` via same mapping.
* Decision: flag if $p(u,v) \geq \tau_{missing}$ (default 0.98). Cap candidate volume for speed.

**Why this split?**

* Removed edges are **hard evidence** of sudden change; we want them front and center.
* KNN captures **latent expectation** (structural similarity suggests a link).

**Artifacts:** `data/outputs/edges_t.json` — array of records like:

```json
{
  "src": "trunk-802ae17d",
  "dst": "concentrator-6b1ad12b",
  "edge_type": "DEPENDS_ON",
  "status": "missing_but_expected",
  "source": "removed_edge",  
  "probability": 1.0,
  "anomaly_score": 1.0
}
```

---

## 7) Graph-level Anomaly Signal

For each `t` vs `t-1` we compute:

* **JS divergence** of degree distributions $\text{JS}(P_{deg}(t), P_{deg}(t-1))$
* **Δ connected components** (absolute difference)
* **Δ global clustering coefficient** (absolute difference)

We scale and blend these into a **graph anomaly score** $\in [0,1]$. This drives the timeline spikes.

**Artifact:** `data/outputs/graph_series.json` — array of `{ snapshot, graph_anomaly_score, metrics, summary }`.

---

## 8) Node State Labeling (Triage)

Quantile-based thresholds (unsupervised, adaptive per snapshot):

**State**

* `isolated`: $\deg_{t-1}>0$ and $\deg_t=0$
* `new`: node appears at `t` (novelty)
* `reconfigured`: $\text{drift} \ge q_{0.90}(\text{drift})$ **or** $|\Delta\deg| \ge q_{0.90}(|\Delta\deg|)$
* `stable`: otherwise

**Severity**

* `critical`: `isolated` **or** $\text{removed\_edges\_count} \ge q_{0.90}$ **or** $|\Delta\deg| \ge q_{0.95}$ **or** $\text{drift} \ge q_{0.95}$
* `warning`: `reconfigured` **or** (`new` and $\deg_t \ge q_{0.90}(\deg)$) **or** $\text{removed\_edges\_count} \ge q_{0.75}$
* `info`: otherwise

**Composite node score**
$\text{score}_i = 0.5\,\widehat{\text{drift}} + 0.3\,\widehat{|\Delta\deg|} + 0.2\,\widehat{\text{removed}} + \text{novelty\_boost}$
(normalizations use robust ranges p75→p95; novelty gets a small bonus if high-degree)

**Artifact:** `data/outputs/node_state_t.json` — array of per-node records:

```json
{
  "node_id": "trunk-802ae17d",
  "node_type": "trunk",
  "degree_prev": 265,
  "degree": 67,
  "degree_change": -198,
  "drift": 2.27,
  "novelty": false,
  "removed_edges_count": 200,
  "state": "reconfigured",
  "severity": "critical",
  "anomaly_score": 0.93
}
```

**Fallback file (simpler):** `data/outputs/nodes_t.json` — top nodes with fewer fields; used only if `node_state_t.json` isn’t present.

---

## 9) Unified Snapshot Report

For fast hydration of a snapshot view, we also produce:

* `data/outputs/report_t.json` with:

  * `graph`: `{ snapshot, graph_anomaly_score, metrics, summary }`
  * `nodes.top`: top-N node records (subset)
  * `edges.present_but_improbable.top` and `edges.missing_but_expected.top`
  * `brief`: short natural-language summary (template-based)

---

## 10) On-demand Enrichment (for Explanations)

Generated only when requested:

* **Neighbors:** `data/outputs/neighbors_{t}_{node_id}.json` — current neighbors and degree at `t`.
* **Node trend:** `data/outputs/node_trend_{node_id}.json` — degree & drift across all snapshots.
* **Edge history:** `data/outputs/edge_history_{src}__{dst}.json` — first seen / removed-at indices.

These power LLM prompts and detailed drill-downs.

---

## 11) Thresholds & Tunables (CLI)

* Event window size (default 1000)
* Node2Vec: `dim, walk_length, walks_per_node, window, epochs, batch_size, p, q`
* Edge anomalies: `tau_present (default 0.15)`, `tau_missing (default 0.98)`, `knn`, `max_missing_candidates`
* Node state quantiles: p75/p90/p95 (can be adjusted to control prevalence)

All are exposed as script arguments to allow quick experimentation.

---

## 12) Rationale & Trade-offs

* **Unsupervised**: quantiles + similarity priors avoid need for labels.
* **Event windows**: stabilize signal per step and training time.
* **Node2Vec + warm-start + Procrustes**: simple, GPU-fast, temporally smooth → drift is meaningful and comparable.
* **Removed-edge emphasis**: operationally relevant; often aligns with real incidents.
* **Quantile thresholds**: robust to snapshot variability and scale.

**Limitations / Next steps**

* Train a light **link decoder** (e.g., MLP over pair features) for sharper edge probabilities.
* Add **directionality** and **relation types** explicitly (heterogeneous encoders).
* **Per-type calibration** (quantiles per node type like trunk/cpe).
* **Online/incremental** updates for near-real-time ingestion.

---

## 13) Validation Suggestions (for Math/ML)

* **Sanity**: correlate drift with |Δdegree|; verify JS spikes coincide with removed-edge surges.
* **Per-type** analysis: compare distributions across node types (trunk vs cpe vs agreement).
* **Ablations**: vary Node2Vec `p,q`, and compare cosine vs dot+sigmoid for $p(u,v)$ tails.
* **Threshold tuning**: adjust $\tau_{present}, \tau_{missing}$, and quantiles p90→p95 to control precision/recall on synthetic truth.

---

**Deliverables produced** (summary):

* `data/snapshots/snapshot_*.csv` — event-windowed blocks
* `data/embeddings/Z_*.pt`, `node_vocab.json` — aligned-ready embeddings & vocab
* `data/outputs/graph_series.json` — graph-level timeline
* `data/outputs/node_state_*.json` (preferred) and `nodes_*.json` (fallback)
* `data/outputs/edges_*.json` — edge anomalies (present/missing)
* `data/outputs/report_*.json` — snapshot aggregates
* On-demand: `neighbors_*`, `node_trend_*`, `edge_history_*`
