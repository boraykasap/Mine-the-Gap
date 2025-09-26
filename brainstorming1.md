Dynamic Graph Anomaly Detection — Data & Modeling Choices
0) Objective (what we’re optimizing for)

Unsupervised anomaly detection on a dynamic, heterogeneous graph built from add/delete events.

Cover three levels: node, edge, and whole-graph structure.

Produce interpretable, quantitative signals that can be consumed by an LLM (for explanation) and a dashboard (for triage).

1) Input data & event semantics

Raw CSV columns:

src, dst, label, timestamp, event_type


event_type ∈ {add, delete}.

We treat the graph as undirected for neighborhood structure (can be changed).

label is carried on edges when present (e.g., HAS_PORT, DEPENDS_ON). We keep label metadata for neighbor/edge dumps.

Why undirected?
For this MVP we focus on topology drift and link plausibility; undirected simplifies degree, components, clustering and Node2Vec behavior. If directionality is critical, we can switch to a directed encoder and adapt metrics.

2) Snapshotting strategy (temporal batching)
Choice: fixed event-count windows of 1,000 events

We analyzed timestamps: events per timestamp are highly variable (median 4, p90 ≈ 16; plus a massive spike at t=0).

Fixed “time” windows risk producing extremely uneven block sizes (unstable training).

Event windows give stable, reproducible blocks with enough signal per step while keeping training fast on GPU.

Result: 11 full blocks of 1,000 + a final remainder (404).
Files: data/snapshots/snapshot_0.csv, …, snapshot_11.csv.

Cumulative application: for any snapshot index t, the current graph state is defined by applying all events up to and including block t.

3) Graph construction per snapshot

For each snapshot t:

Start from empty E=∅, apply add/delete cumulatively up to t.

Maintain a stable node index using a global vocabulary:

data/embeddings/node_vocab.json with idx2id (array) and id2idx (map).

Indices are monotonic: new nodes append to the end → consistent row alignment across time.

Build a PyG graph: edge_index from the current E, undirected (we canonicalize pairs (u,v) as sorted).

We also keep:

Node type inferred from node_id prefix (e.g., trunk-*, cpe-*, agreement-*).

Edge label (when available) for neighbor dumps and edge reports.

4) Node representation learning (per snapshot)
Encoder: Node2Vec (PyTorch Geometric)

Tractable on our scale; robust unsupervised local+meso structure encoding.

Hyperparams (good trade-off observed on GPU):

dim=128, walk_length=20, walks_per_node=20, window=10,

epochs=15, batch_size=1024,

p=1.0, q=0.5 (slightly BFS-leaning to capture local role changes).

Warm-start: for t>0, we initialize shared rows in Z_t from Z_{t-1} (copies common rows by the stable index) → faster convergence and smoother temporal behavior.

Save:

data/embeddings/Z_t.pt with {"embeddings": torch.FloatTensor[n_t, d]}

A CSV preview for quick inspection: Z_t_preview.csv.

Rationale:
We need relative geometry stability between t-1 and t. Node2Vec is stable enough with warm-start, and we add explicit alignment (next section).

5) Temporal alignment & drift
Alignment: Orthogonal Procrustes

Embeddings are identifiable up to rotation. We align Z_t to Z_{t-1} on the common index range [0..min(n_{t-1}, n_t)-1]:

𝑅
𝑡
  
=
  
arg
⁡
min
⁡
𝑅
∈
𝑂
(
𝑑
)
∥
𝑍
𝑡
(
𝑐
𝑜
𝑚
𝑚
𝑜
𝑛
)
𝑅
−
𝑍
𝑡
−
1
(
𝑐
𝑜
𝑚
𝑚
𝑜
𝑛
)
∥
𝐹
R
t
	​

=arg
R∈O(d)
min
	​

∥Z
t
(common)
	​

R−Z
t−1
(common)
	​

∥
F
	​


Then compute node drift for any node i present in both:

drift
𝑖
(
𝑡
)
  
=
  
∥
 
(
𝑍
𝑡
𝑅
𝑡
)
[
𝑖
]
  
−
  
𝑍
𝑡
−
1
[
𝑖
]
 
∥
2
drift
i
	​

(t)=
	​

(Z
t
	​

R
t
	​

)[i]−Z
t−1
	​

[i]
	​

2
	​

Degree change:

Recompute degree at t-1 and t from the cumulatively applied event sets.

Δ
deg
⁡
𝑖
(
𝑡
)
=
deg
⁡
𝑖
(
𝑡
)
−
deg
⁡
𝑖
(
𝑡
−
1
)
Δdeg
i
	​

(t)=deg
i
	​

(t)−deg
i
	​

(t−1).

Novelty:

novelty=true if node exists at t but not at t-1 (by index) or had zero degree before and positive degree now.

These are the primary node features used downstream.

6) Edge anomaly methodology (two classes)

We detect two complementary edge anomaly types at each t:

A) Present-but-improbable edges

Edges that exist in the graph at t but have low link plausibility under the embedding model.

Candidate set: all edges in E_t (can be downsampled for speed).

Scoring: link probability 
𝑝
(
𝑢
,
𝑣
)
p(u,v) derived from embedding similarity. For MVP:

we use a similarity-to-probability mapping (cosine or dot followed by normalization/sigmoid).

These are unsupervised priors, not a trained decoder.

Decision rule: present edge is “improbable” if 
𝑝
(
𝑢
,
𝑣
)
≤
𝜏
present
p(u,v)≤τ
present
	​

, with default 
𝜏
present
=
0.15
τ
present
	​

=0.15.

Return top-K by ascending 
𝑝
p (most suspicious first).

B) Missing-but-expected edges

Edges that do not exist at t but are highly plausible or were recently removed.

Primary candidates: all removed edges between t-1 and t.
These represent sudden disappearances; we treat them as highly expected (source="removed_edge") and assign 
𝑝
=
1.0
p=1.0 (they existed at t-1).

Secondary candidates (KNN): for each node, its top-k nearest neighbors in the aligned space at t (or t-1, both are supported), excluding existing edges.
source="knn_candidate"; we compute 
𝑝
(
𝑢
,
𝑣
)
p(u,v) by the same similarity-to-probability map.

Decision rule: missing-but-expected if 
𝑝
(
𝑢
,
𝑣
)
≥
𝜏
missing
p(u,v)≥τ
missing
	​

, default 
𝜏
missing
=
0.98
τ
missing
	​

=0.98.

We cap candidate volume (--max-missing-candidates) and expose thresholds so you can tune precision/recall.

Why this split?

Removed edges are hard evidence; we want them highlighted by default.

KNN candidates capture latent expectations from structure (e.g., two devices in very similar positions that should be connected).

7) Graph-level anomaly signal

For each snapshot t (vs t-1) we compute:

JS divergence of degree distributions 
JS
(
𝑃
deg
⁡
(
𝑡
)
,
𝑃
deg
⁡
(
𝑡
−
1
)
)
JS(P
deg
	​

(t),P
deg
	​

(t−1)).

Δ connected components (absolute change in count).

Δ global clustering coefficient (absolute change).

We combine these into a normalized graph anomaly score 
∈
[
0
,
1
]
∈[0,1] (weighted blend after scaling to robust ranges). This drives the timeline spikes.

8) Node state labeling (for triage)

To make node lists actionable, we add state and severity using quantile-based thresholds (unsupervised, adaptive to each snapshot’s distribution):

state:

isolated: 
deg
⁡
𝑡
−
1
>
0
deg
t−1
	​

>0 and 
deg
⁡
𝑡
=
0
deg
t
	​

=0,

new: node appears at t (novelty),

reconfigured: 
drift
≥
𝑞
0.90
(
drift
)
drift≥q
0.90
	​

(drift) or 
∣
Δ
deg
⁡
∣
≥
𝑞
0.90
(
∣
Δ
deg
⁡
∣
)
∣Δdeg∣≥q
0.90
	​

(∣Δdeg∣),

stable: otherwise.

severity:

critical: isolated or 
removed_edges_count
≥
𝑞
0.90
removed_edges_count≥q
0.90
	​

 or 
∣
Δ
deg
⁡
∣
≥
𝑞
0.95
∣Δdeg∣≥q
0.95
	​

 or 
drift
≥
𝑞
0.95
drift≥q
0.95
	​

,

warning: reconfigured or (new with large 
deg
⁡
𝑡
≥
𝑞
0.90
(
deg
⁡
)
deg
t
	​

≥q
0.90
	​

(deg)) or 
removed_edges_count
≥
𝑞
0.75
removed_edges_count≥q
0.75
	​

,

info: otherwise.

Why quantiles?
No labels are available; quantiles give data-driven thresholds that adapt to snapshot variability without manual tuning.

9) Output files & schemas (what’s available)
Per snapshot t

Embeddings: data/embeddings/Z_t.pt → { "embeddings": FloatTensor[n_t, d] }.

Node anomalies (simple): data/outputs/nodes_{t}.json
Top nodes by unsupervised composite score (presence of fields depends on generator; used as fallback by the API).

Node state (rich): data/outputs/node_state_{t}.json
Array of records (one per node) with:

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


Edge anomalies: data/outputs/edges_{t}.json
Array of:

{
  "src": "trunk-802ae17d",
  "dst": "concentrator-6b1ad12b",
  "edge_type": "DEPENDS_ON",
  "status": "missing_but_expected",
  "source": "removed_edge" | "knn_candidate",
  "probability": 1.0,
  "anomaly_score": 1.0
}


Unified report: data/outputs/report_{t}.json
Aggregates graph metrics + top nodes/edges + brief summary. Used to hydrate one snapshot view quickly.

Across snapshots

Graph time series: data/outputs/graph_series.json
Array of { snapshot, graph_anomaly_score, metrics, summary }.

On-demand enrichment (for explanations)

Node neighbors: data/outputs/neighbors_{t}_{node_id}.json

Node trend: data/outputs/node_trend_{node_id}.json (per-t degree & drift)

Edge history: data/outputs/edge_history_{src}__{dst}.json (first seen / removed at)

10) Scoring details (how we rank)
Node anomaly score (composite)

Normalize each metric with robust ranges (p75→p95) and combine:

score
𝑖
=
0.5
⋅
drift
^
𝑖
+
0.3
⋅
∣
Δ
deg
⁡
∣
^
𝑖
+
0.2
⋅
removed_count
^
𝑖
+
novelty_boost
score
i
	​

=0.5⋅
drift
i
	​

+0.3⋅
∣Δdeg∣
	​

i
	​

+0.2⋅
removed_count
	​

i
	​

+novelty_boost

novelty_boost adds a small bonus if the node is new and high-degree.

Edge anomaly score

For “present-but-improbable”: use 
1
−
𝑝
(
𝑢
,
𝑣
)
1−p(u,v) or equivalently sort by ascending 
𝑝
p.

For “missing-but-expected”: use 
𝑝
(
𝑢
,
𝑣
)
p(u,v) and sort descending.

Removed edges get p=1.0 (explicit prior that they were highly expected before removal).

Graph score

Weighted blend (after normalization) of: JS(degree), Δcomponents, Δclustering.

All thresholds (
𝜏
present
τ
present
	​

, 
𝜏
missing
τ
missing
	​

, K in KNN, top-K limits) are CLI parameters to allow quick tuning.

11) Why these choices (trade-offs)

Unsupervised, label-free: quantiles and similarity-based link priors remove the need for labeled incidents.

Event windows: stabilize per-step training time and signal density.

Node2Vec + warm-start + Procrustes: simple, GPU-fast, and sufficiently temporal-smooth to make drift meaningful.

Removed-edge emphasis: aligns with operational interest (unexpected removals are often the incident).

Quantile thresholds: robust to scale and non-stationarities across snapshots.

Known limitations / future work:

A trained link decoder (e.g., shallow MLP on pairwise features or a contrastive objective) would sharpen edge probabilities.

Directed modeling and label-aware encoders for asymmetric relations.

Per-type calibration (separate quantiles per node type).

Incremental updates (mini-batch online Node2Vec or dynamic GNN) for near-real-time.

12) How you can validate/adjust (math/ML team)

Sanity checks:

Drift vs Δdegree correlation—ensure we don’t flag pure noise.

JS(degree) spikes coincide with high removed-edge counts.

Per-type distributions (e.g., trunk vs cpe) to assess calibration.

Ablations:

Change p,q (BFS/DFS bias) and see effect on drift sensitivity.

Try cosine vs dot+sigmoid mapping for 
𝑝
(
𝑢
,
𝑣
)
p(u,v); evaluate the tails.

Thresholds:

Adjust 
𝜏
present
τ
present
	​

, 
𝜏
missing
τ
missing
	​

 to match desired precision/recall on synthetic ground truth (if available).

Tweak quantiles (e.g., p90→p95) to control state/severity prevalence.
