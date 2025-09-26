import torch.nn as nn
import torch.nn.functional as F
from torch import no_grad
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCN, GAT
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from collections import defaultdict

from knowgraph import *

def train_anomaly_detector(model, data, epochs=200):
    """Train the GNN anomaly detector in an unsupervised manner"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        embeddings, reconstructed = model(data.x, data.edge_index)
        
        # Reconstruction loss (mean squared error)
        loss = F.mse_loss(reconstructed, data.x)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
    return losses

def compute_anomaly_scores(model, data):
    """Compute anomaly scores for nodes based on reconstruction error"""
    model.eval()
    with no_grad():
        embeddings, reconstructed = model(data.x, data.edge_index)
        
        # Compute reconstruction error for each node
        reconstruction_error = F.mse_loss(reconstructed, data.x, reduction='none')
        node_anomaly_scores = reconstruction_error.sum(dim=1)
        
        return node_anomaly_scores, embeddings

def detect_kg_anomalies(csv_data):
    """
    Main function to detect anomalies in a knowledge graph from CSV data
    
    Parameters:
    csv_data (str or DataFrame): CSV data with columns src, dst, label, timestamp, event_type
    
    Returns:
    dict: Anomaly scores for nodes and edges
    """
    
    # Handle input data
    if isinstance(csv_data, str):
        # If string, assume it's CSV content
        df = pd.read_csv(StringIO(csv_data))
    else:
        # If DataFrame, use directly
        df = csv_data.copy()
    
    # Initialize and fit the comprehensive detector
    detector = ComprehensiveAnomalyDetector()
    detector.fit(df)
    
    # Detect anomalies
    combined_scores, node_scores = detector.detect_anomalies(df)
    
    # Add scores to dataframe
    df['combined_anomaly_score'] = combined_scores
    
    # Results
    results = {
        'node_anomaly_scores': node_scores,
        'edge_anomaly_scores': df[['src', 'dst', 'label', 'timestamp', 'combined_anomaly_score']].to_dict('records'),
        'most_anomalous_nodes': sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:5],
        'most_anomalous_edges': df.nlargest(5, 'combined_anomaly_score')[['src', 'dst', 'label', 'combined_anomaly_score']].to_dict('records')
    }
    
    return results

    

class GNNAnomalyDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNAnomalyDetector, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
        # Reconstruction layer for anomaly scoring
        self.reconstruction = nn.Linear(output_dim, input_dim)
        
    def forward(self, x, edge_index):
        # GNN layers
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.5, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=0.5, training=self.training)
        h = F.relu(self.conv3(h, edge_index))
        
        # Reconstruction for anomaly detection
        reconstructed = self.reconstruction(h)
        
        return h, reconstructed

        
class TemporalAnomalyDetector:
    def __init__(self, window_size=2):
        self.window_size = window_size
        self.temporal_patterns = defaultdict(list)
        
    def build_temporal_patterns(self, df):
        """Build temporal patterns for edge types between node pairs"""
        # Group by timestamp and create graph snapshots
        for timestamp in df['timestamp'].unique():
            snapshot = df[df['timestamp'] == timestamp]
            for _, row in snapshot.iterrows():
                key = (row['src'], row['dst'], row['label'])
                self.temporal_patterns[key].append(timestamp)
                
    def detect_temporal_anomalies(self, df):
        """Detect anomalies based on temporal patterns"""
        anomaly_scores = []
        
        for _, row in df.iterrows():
            key = (row['src'], row['dst'], row['label'])
            timestamps = self.temporal_patterns[key]
            
            # Check if this edge appears at unusual times
            current_time = row['timestamp']
            
            # If this edge type rarely occurs, it's anomalous
            if len(timestamps) <= 1:
                score = 1.0
            else:
                # Calculate temporal distribution anomaly
                # Based on standard deviation of time intervals
                intervals = [timestamps[i+1] - timestamps[i] 
                            for i in range(len(timestamps)-1)]
                if intervals:
                    mean_interval = np.mean(intervals)
                    std_interval = np.std(intervals)
                    # If current time deviates significantly from expected pattern
                    if len(timestamps) > 1 and std_interval > 0:
                        expected_next = timestamps[-1] + mean_interval
                        temporal_deviation = abs(current_time - expected_next) / (std_interval + 1)
                        score = min(temporal_deviation, 1.0)
                    else:
                        score = 0.1
                else:
                    score = 0.5
                    
            anomaly_scores.append(score)
            
        return anomaly_scores


class EdgeAnomalyDetector:
    def __init__(self, kg):
        self.kg = kg
        self.node_edge_patterns = self._build_edge_patterns()
    
    def _build_edge_patterns(self):
        """Build normal edge patterns for each node"""
        patterns = defaultdict(lambda: defaultdict(int))
        
        # Count edge types for each node
        for node in self.kg.node_features:
            for edge_type, count in self.kg.node_features[node]['edge_types'].items():
                patterns[node][edge_type] = count
                
        return patterns
    
    def compute_edge_anomaly_scores(self, df):
        """Compute anomaly scores for edges based on unusual relationships"""
        edge_anomaly_scores = []
        
        for _, row in df.iterrows():
            src = row['src']
            dst = row['dst']
            edge_type = row['label']
            
            # Score based on how common this edge type is for the source node
            src_edge_count = self.node_edge_patterns[src][edge_type]
            dst_edge_count = self.node_edge_patterns[dst][edge_type]
            
            # Nodes with fewer connections are more anomalous
            src_total_connections = self.kg.node_features[src]['degree']
            dst_total_connections = self.kg.node_features[dst]['degree']
            
            # Anomaly score: low count of this edge type for these nodes
            # and low total connections makes it more anomalous
            score = 1.0 / (src_edge_count + dst_edge_count + 1) * \
                   (1.0 / (src_total_connections + dst_total_connections + 1))
            
            edge_anomaly_scores.append(score)
            
        return edge_anomaly_scores

        
class ComprehensiveAnomalyDetector:
    def __init__(self, node_weight=0.4, edge_weight=0.3, temporal_weight=0.3):
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        self.temporal_weight = temporal_weight
        self.kg = None
        self.model = None
        self.edge_detector = None
        self.temporal_detector = None
        
    def fit(self, df):
        """Fit the anomaly detector on the knowledge graph data"""
        # Build knowledge graph
        self.kg = DynamicKnowledgeGraph()
        pyg_data = self.kg.build_graph(df)
        
        # Initialize and train GNN model
        input_dim = pyg_data.x.shape[1]
        hidden_dim = 32
        output_dim = 16
        
        self.model = GNNAnomalyDetector(input_dim, hidden_dim, output_dim)
        train_anomaly_detector(self.model, pyg_data, epochs=1000)
        
        # Initialize edge detector
        self.edge_detector = EdgeAnomalyDetector(self.kg)
        
        # Initialize temporal detector
        self.temporal_detector = TemporalAnomalyDetector()
        self.temporal_detector.build_temporal_patterns(df)
        
        return self
    
    def detect_anomalies(self, df):
        """Detect anomalies in the knowledge graph"""
        # Node anomaly scores
        node_anomaly_scores, embeddings = compute_anomaly_scores(self.model, 
                                                                 self.kg._to_pyg_data())
        node_scores_dict = {}
        for node, idx in self.kg.node_to_idx.items():
            node_scores_dict[node] = node_anomaly_scores[idx].item()
            
        # Edge anomaly scores
        edge_anomaly_scores = self.edge_detector.compute_edge_anomaly_scores(df)
        
        # Temporal anomaly scores
        temporal_anomaly_scores = self.temporal_detector.detect_temporal_anomalies(df)
        
        # Combine scores
        combined_scores = []
        for i, row in df.iterrows():
            src_score = node_scores_dict.get(row['src'], 0)
            dst_score = node_scores_dict.get(row['dst'], 0)
            edge_score = edge_anomaly_scores[i]
            temporal_score = temporal_anomaly_scores[i]
            
            # Weighted combination
            combined_score = (self.node_weight * (src_score + dst_score) / 2 + 
                            self.edge_weight * edge_score + 
                            self.temporal_weight * temporal_score)
            combined_scores.append(combined_score)
            
        return combined_scores, node_scores_dict
