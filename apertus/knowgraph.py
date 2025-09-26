import torch
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

class DynamicKnowledgeGraph:
    def __init__(self):
        self.node_to_idx = {}
        self.idx_to_node = {}
        self.edge_type_to_idx = {}
        self.idx_to_edge_type = {}
        self.node_features = {}
        self.edges = []
        self.timestamps = []
        
    def build_graph(self, df):
        """Build knowledge graph from DataFrame"""
        node_idx = 0
        edge_type_idx = 0
        
        # Create node mappings
        all_nodes = set(df['src'].unique()) | set(df['dst'].unique())
        for node in all_nodes:
            if node not in self.node_to_idx:
                self.node_to_idx[node] = node_idx
                self.idx_to_node[node_idx] = node
                node_idx += 1
        
        # Create edge type mappings
        for edge_type in df['label'].unique():
            if edge_type not in self.edge_type_to_idx:
                self.edge_type_to_idx[edge_type] = edge_type_idx
                self.idx_to_edge_type[edge_type_idx] = edge_type
                edge_type_idx += 1
        
        # Build edges
        for _, row in df.iterrows():
            src_idx = self.node_to_idx[row['src']]
            dst_idx = self.node_to_idx[row['dst']]
            edge_type_idx = self.edge_type_to_idx[row['label']]
            self.edges.append([src_idx, dst_idx])
            self.timestamps.append(row['timestamp'])
        
        # Create node features (degree, edge type distribution, etc.)
        self._create_node_features(df)
        
        return self._to_pyg_data()
    
    def _create_node_features(self, df):
        """Create node features based on graph statistics"""
        # Initialize features for all nodes
        for node in self.node_to_idx:
            self.node_features[node] = {
                'degree': 0,
                'in_degree': 0,
                'out_degree': 0,
                'edge_types': defaultdict(int)
            }
        
        # Calculate features
        for _, row in df.iterrows():
            src = row['src']
            dst = row['dst']
            edge_type = row['label']
            
            # Update degrees
            self.node_features[src]['out_degree'] += 1
            self.node_features[dst]['in_degree'] += 1
            self.node_features[src]['degree'] += 1
            self.node_features[dst]['degree'] += 1
            
            # Update edge type distribution
            self.node_features[src]['edge_types'][edge_type] += 1
            self.node_features[dst]['edge_types'][edge_type] += 1
    
    def _to_pyg_data(self):
        """Convert to PyTorch Geometric Data format"""
        edge_index = torch.tensor(self.edges).t().contiguous()
        
        # Create feature matrix
        num_nodes = len(self.node_to_idx)
        num_edge_types = len(self.edge_type_to_idx)
        
        # Features: [degree, in_degree, out_degree, edge_type_features]
        x = torch.zeros((num_nodes, 3 + num_edge_types))
        
        for node, idx in self.node_to_idx.items():
            features = self.node_features[node]
            x[idx, 0] = features['degree']
            x[idx, 1] = features['in_degree']
            x[idx, 2] = features['out_degree']
            
            # Edge type features
            for edge_type, count in features['edge_types'].items():
                edge_type_idx = self.edge_type_to_idx[edge_type]
                x[idx, 3 + edge_type_idx] = count
        
        return Data(x=x, edge_index=edge_index)
