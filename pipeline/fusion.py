"""
fusion.py
Gated MLP fusion module combining semantic and structural embeddings.
Compares semantic-only, structural-only, and fused pairwise accuracies.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')


# ── Gated Fusion Module ───────────────────────────────────────────────────────

class GatedFusion:
    """
    Gated MLP fusion.
    Both semantic and structural streams are projected to `output_dim`.
    A sigmoid gate (learned via random search) blends the two projections:
        fused = gate * p_sem + (1 - gate) * p_struct
    """

    def __init__(self, sem_dim, struct_dim, output_dim=128, seed=42):
        np.random.seed(seed)
        self.output_dim = output_dim

        # Projection matrices
        self.W_sem    = np.random.randn(sem_dim,    output_dim) * np.sqrt(2.0 / sem_dim)
        self.W_struct = np.random.randn(struct_dim, output_dim) * np.sqrt(2.0 / struct_dim)

        # Gate weights
        self.W_gate = np.random.randn(output_dim * 2, output_dim) * 0.01
        self.b_gate = np.zeros(output_dim)
        self.trained = False

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def project(self, sem_emb, struct_emb):
        """Project both streams to output_dim."""
        p_sem    = np.tanh(sem_emb    @ self.W_sem)
        p_struct = np.tanh(struct_emb @ self.W_struct)
        return p_sem, p_struct

    def fuse(self, sem_emb, struct_emb):
        """
        Fuse semantic and structural embeddings via gated combination.
        Inputs can be 1-D (single sentence) or 2-D (batch of sentences).
        """
        p_sem, p_struct = self.project(sem_emb, struct_emb)
        combined = np.concatenate([p_sem, p_struct], axis=-1)
        gate     = self.sigmoid(combined @ self.W_gate + self.b_gate)
        return gate * p_sem + (1.0 - gate) * p_struct

    def train_gate(self, sem_train, struct_train, labels=None,
                   n_iters=50, lr=0.01):
        """
        Train gate weights using random search (perturbation + keep-if-better).
        sem_train   : (N, sem_dim)
        struct_train: (N, struct_dim)
        """
        best_loss = float('inf')
        best_W    = self.W_gate.copy()
        best_b    = self.b_gate.copy()

        np.random.seed(42)
        for i in range(n_iters):
            noise_W = np.random.randn(*self.W_gate.shape) * 0.1
            noise_b = np.random.randn(*self.b_gate.shape) * 0.1
            self.W_gate += noise_W
            self.b_gate += noise_b

            fused  = self.fuse(sem_train, struct_train)
            target = np.tanh((sem_train @ self.W_sem + struct_train @ self.W_struct) / 2.0)
            loss   = float(np.mean((fused - target) ** 2))

            if loss < best_loss:
                best_loss = loss
                best_W    = self.W_gate.copy()
                best_b    = self.b_gate.copy()
            else:
                self.W_gate = best_W.copy()
                self.b_gate = best_b.copy()

        self.trained = True



