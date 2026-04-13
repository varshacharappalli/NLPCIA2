"""
structural_stream.py
Graph-based structural stream: four graph types, GCN encoder,
and pairwise sentence ordering evaluation.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')


# ── Similarity helper ─────────────────────────────────────────────────────────

def cosine_similarity(a, b):
    """Cosine similarity between two 1-D vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── Four graph types ──────────────────────────────────────────────────────────

def build_local_graph(n):
    """Adjacent-sentence edges: (i, i+1) with weight 1."""
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i][i + 1] = 1.0
        A[i + 1][i] = 1.0
    return A


def build_midrange_graph(n, window=3):
    """Edges between sentences within a sliding window."""
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, min(i + window + 1, n)):
            A[i][j] = 1.0
            A[j][i] = 1.0
    return A


def build_global_graph(embeddings, threshold=0.3):
    """Cosine-similarity edges above threshold."""
    n = len(embeddings)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim > threshold:
                A[i][j] = sim
                A[j][i] = sim
    return A


def build_entity_graph(sentences):
    """
    Shared-entity edges.
    Entities are approximated as: capitalised words or words longer than 4 characters.
    Edge weight = Jaccard overlap of entity sets.
    """
    n = len(sentences)
    A = np.zeros((n, n))

    def get_entities(sent):
        words = sent.split()
        return set(
            w.lower().strip('.,!?;:"\'-')
            for w in words
            if len(w) > 4 or (len(w) > 1 and w[0].isupper())
        )

    entities = [get_entities(s) for s in sentences]
    for i in range(n):
        for j in range(i + 1, n):
            union = entities[i] | entities[j]
            if not union:
                continue
            overlap = len(entities[i] & entities[j])
            if overlap > 0:
                score = overlap / len(union)
                A[i][j] = score
                A[j][i] = score
    return A


def merge_graphs(local, midrange, global_g, entity,
                 weights=(0.4, 0.25, 0.2, 0.15)):
    """Weighted combination of the four adjacency matrices."""
    return (weights[0] * local
            + weights[1] * midrange
            + weights[2] * global_g
            + weights[3] * entity)

# ── GCN (numpy) ───────────────────────────────────────────────────────────────

def gcn_layer(A, X, W, b, activation='relu'):
    """
    Single GCN propagation layer.
    Computes: H = act( D^{-1/2} A_hat D^{-1/2} X W + b )
    where A_hat = A + I (self-loops added).
    """
    n = A.shape[0]
    A_hat = A + np.eye(n)
    degree = A_hat.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / (np.sqrt(degree) + 1e-8))
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    Z = A_norm @ X @ W + b
    if activation == 'relu':
        return np.maximum(0.0, Z)
    elif activation == 'tanh':
        return np.tanh(Z)
    return Z


class GCNEncoder:
    """Two-layer Graph Convolutional Network for structural encoding."""

    def __init__(self, input_dim, hidden_dim=64, output_dim=64, seed=42):
        np.random.seed(seed)
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * scale2
        self.b2 = np.zeros(output_dim)

    def encode(self, A, X):
        """
        A : (n, n) combined adjacency matrix
        X : (n, input_dim) initial node features
        Returns: (n, output_dim) structural node embeddings
        """
        H1 = gcn_layer(A, X, self.W1, self.b1, activation='relu')
        H2 = gcn_layer(A, H1, self.W2, self.b2, activation='tanh')
        return H2
