"""
decoding.py
Pairwise MLP scorer + tournament-ranking decoder for sentence ordering.
Reports pairwise accuracy, sequence accuracy, and mean Kendall Tau.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import warnings

warnings.filterwarnings('ignore')

from metrics import tournament_to_order


# ── Pairwise Scorer ───────────────────────────────────────────────────────────

class PairwiseScorer:
    """MLP-based pairwise scorer for sentence ordering."""

    def __init__(self):
        self.clf = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation='relu',
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )

    def fit(self, X, y):
        """Train the MLP scorer."""
        self.clf.fit(X, y)

    def predict_proba(self, X):
        """Return P(i before j) for each row in X."""
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, X):
        """Return hard binary predictions."""
        return self.clf.predict(X)

    def score(self, X, y):
        """Return classification accuracy."""
        return self.clf.score(X, y)

    def save(self, path):
        import joblib
        joblib.dump(self.clf, path)
        print(f"  Saved Scorer (MLP) to {path}")

    def load(self, path):
        import joblib
        import os
        if os.path.exists(path):
            self.clf = joblib.load(path)
            print(f"  Loaded Scorer (MLP) from {path}")
        else:
            print(f"  [Warning] Could not find {path}")


# ── Fused embedding helpers ───────────────────────────────────────────────────

def get_fused_embeddings(docs, fusion_module, semantic_encoder, gcn, tfidf_encoder):
    """Compute fused embeddings for each document."""
    from structural_stream import (
        build_local_graph, build_midrange_graph,
        build_global_graph, build_entity_graph, merge_graphs,
    )

    result = []
    for doc in docs:
        sents = doc['sentences']
        n     = len(sents)

        sem_embs    = semantic_encoder.encode_doc(doc)
        init_feats  = tfidf_encoder.encode_doc(doc)

        A_local     = build_local_graph(n)
        A_mid       = build_midrange_graph(n)
        A_global    = build_global_graph(init_feats)
        A_entity    = build_entity_graph(sents)
        A_combined  = merge_graphs(A_local, A_mid, A_global, A_entity)

        struct_embs = gcn.encode(A_combined, init_feats)
        fused       = fusion_module.fuse(sem_embs, struct_embs)

        result.append({
            'fused':     fused,
            'n':         n,
            'sentences': sents,
        })
    return result


def build_decoding_dataset(doc_embs):
    """Build pairwise (X, y) dataset from fused embeddings."""
    X, y = [], []
    for e in doc_embs:
        fused = e['fused']
        n     = e['n']
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                feat = np.concatenate([
                    fused[i], fused[j],
                    np.abs(fused[i] - fused[j]),
                    fused[i] * fused[j],
                ])
                X.append(feat)
                y.append(1 if i < j else 0)
    if not X:
        return np.zeros((0, 1)), np.zeros(0, dtype=int)
    return np.array(X), np.array(y)


# ── Tournament Decoding ───────────────────────────────────────────────────────

def predict_document_order(doc_emb, scorer):
    """
    Predict the ordering of sentences in a single document.
    Builds an (n x n) score matrix and converts it to a ranking via
    tournament_to_order (descending row-sum).
    """
    fused = doc_emb['fused']
    n     = doc_emb['n']

    if n == 1:
        return [0]

    score_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            feat = np.concatenate([
                fused[i], fused[j],
                np.abs(fused[i] - fused[j]),
                fused[i] * fused[j],
            ])
            prob = scorer.predict_proba(feat.reshape(1, -1))[0]
            score_matrix[i][j] = prob

    return tournament_to_order(score_matrix)


# ── Run Decoding ──────────────────────────────────────────────────────────────

def run_decoding(train_docs, test_docs, fusion_module,
                 semantic_encoder, gcn, tfidf_encoder):
    """
    Full decoding phase:
      1. Compute fused embeddings for train and test.
      2. Train MLP pairwise scorer.
      3. Decode sentence orders via tournament ranking.
      4. Compute and report pairwise accuracy, sequence accuracy, Kendall Tau.

    Returns:
        metrics – dict {pairwise_accuracy, sequence_accuracy, kendall_tau}
        scorer  – trained PairwiseScorer
    """
    print("  Computing fused embeddings...")
    train_embs = get_fused_embeddings(
        train_docs, fusion_module, semantic_encoder, gcn, tfidf_encoder
    )
    test_embs = get_fused_embeddings(
        test_docs, fusion_module, semantic_encoder, gcn, tfidf_encoder
    )

    print("  Training pairwise scorer (MLP)...")
    X_train, y_train = build_decoding_dataset(train_embs)
    X_test,  y_test  = build_decoding_dataset(test_embs)

    scorer = PairwiseScorer()
    scorer.fit(X_train, y_train)

    pairwise_acc = float(scorer.score(X_test, y_test))
    print(f"  Pairwise Scoring Accuracy: {pairwise_acc:.4f}")

    return {
        'pairwise_accuracy': pairwise_acc,
    }, scorer, test_embs
