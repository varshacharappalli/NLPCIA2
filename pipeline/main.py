"""
NLP CIA2 - Full Sentence Ordering Pipeline
==========================================
Preprocessing : C  (preprocessing/preprocess.c)
Pipeline       : Python (pipeline/)
Dataset        : ACL Anthology Network (AAN) Abstracts

Acanthropology (AAN) contains real-world research abstracts.
The pipeline evaluates how different semantic representations
(Word2Vec, TF-IDF, BERT, SBERT) impact the retrieval of the
logical order of sentences within these abstracts.
"""

import sys
import os
import warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression

from dataset_generator import generate_dataset
from data_loader import compile_c, preprocess_docs, create_train_test_split
from semantic_stream import (
    TFIDFEncoder, Word2VecEncoder, BERTEncoder,
    Word2VecTFIDFEncoder, RawTransformerEncoder, get_sentence_corpus,
    FineTunedDistilBERTEncoder
)
from structural_stream import (
    GCNEncoder, build_local_graph, build_midrange_graph,
    build_global_graph, build_entity_graph, merge_graphs
)
from fusion import GatedFusion
from decoding import run_decoding
from prediction_analyzer import analyze_sample_predictions

# ── Helpers ────────────────────────────────────────────────────────────────────

def print_banner(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def make_docs_with_sents(orig_docs, proc_results):
    out = []
    for orig, proc in zip(orig_docs, proc_results):
        sents = proc['preprocessed_sentences']
        if not sents or len(sents) != len(orig['sentences']):
            sents = orig['sentences']
        out.append({**orig, 'sentences': sents})
    return out





def gnn_fn(tfidf_enc, gcn):
    def fn(doc):
        n    = len(doc['sentences'])
        init = tfidf_enc.encode_doc(doc)
        A = merge_graphs(build_local_graph(n), build_midrange_graph(n),
                         build_global_graph(init), build_entity_graph(doc['sentences']))
        return gcn.encode(A, init)
    return fn





# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print_banner("NLP CIA2 -- Sentence Ordering Pipeline")

    # ── 1. Dataset ────────────────────────────────────────────────
    print_banner("Step 1: ACL Anthology Dataset Loading")
    docs = generate_dataset(seed=42, limit=500)
    train_docs, test_docs = create_train_test_split(docs, test_ratio=0.2, seed=42)
    print(f"  {len(docs)} ACL abstracts loaded | Train: {len(train_docs)} | Test: {len(test_docs)}")

    # ── 2. C Preprocessor ─────────────────────────────────────────
    print_banner("Step 2: C Preprocessor")
    compile_c()
    train_proc = preprocess_docs(train_docs)
    test_proc  = preprocess_docs(test_docs)
    train_full = make_docs_with_sents(train_docs, train_proc)
    test_full  = make_docs_with_sents(test_docs,  test_proc)
    seg_acc = sum(
        1 for o, p in zip(test_docs, test_proc)
        if p['stats'].get('num_sentences', 0) == len(o['sentences'])
    ) / len(test_docs)
    print(f"  Sentence segmentation accuracy: {seg_acc:.2%}")

    # ── 3. Fit all encoders ───────────────────────────────────────
    print_banner("Step 3: Fitting Encoders")
    corpus = get_sentence_corpus(train_full + test_full)

    tfidf_enc = TFIDFEncoder();   tfidf_enc.fit(corpus);  print("  [1/5] TF-IDF fitted")
    w2v_enc   = Word2VecEncoder(); w2v_enc.fit(corpus);   print("  [2/5] Word2Vec fitted")
    bert_enc  = BERTEncoder();    bert_enc.fit();         print("  [3/5] SBERT loaded")
    w2v_tfidf_enc = Word2VecTFIDFEncoder(w2v_enc, tfidf_enc); w2v_tfidf_enc.fit(corpus); print("  [4/5] Word2Vec+TFIDF fitted")
    raw_distilbert_enc = RawTransformerEncoder(); raw_distilbert_enc.fit(corpus); print("  [5/6] Raw DistilBERT loaded")
    ft_distilbert_enc = FineTunedDistilBERTEncoder(); ft_distilbert_enc.fit(train_full); print("  [6/6] Fine-Tuned DistilBERT fitted")

    import joblib
    print("  Saving base encoders...")
    joblib.dump(tfidf_enc, "tfidf_enc.pkl")
    joblib.dump(w2v_enc, "w2v_enc.pkl")
    ft_distilbert_enc.save("ft_distilbert.pt")
    print("  Evaluating sequence classification head performance for Fine-Tuned DistilBERT:")
    ft_distilbert_enc.evaluate_pairwise_accuracy(test_full)

    # ── 4. Main Embedding Evaluation ──────────────────────────────
    print_banner("Step 4: Embedding Evaluation (Pairwise Accuracy in Final Step)")

    EMBEDDINGS = [
        ('1. Word2Vec (Mean Pooled)',         w2v_enc),
        ('2. Word2Vec (TF-IDF Weighted)',     w2v_tfidf_enc),
        ('3. TF-IDF (Sparse Vector)',         tfidf_enc),
        ('4. Contextual Token (DistilBERT)',   raw_distilbert_enc),
        ('5. Sequence Domain (SBERT)',        bert_enc),
        ('6. Fine-Tuned Token (DistilBERT [CLS])', ft_distilbert_enc),
    ]

    # Initialize Prediction Report
    report_path = "test_predictions.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Test Prediction Analysis Report\n")
        f.write("This report analyzes the 'Actual vs Predicted' ordering for sample documents across different embedding methods.\n\n")

    for enc_name, enc in EMBEDDINGS:
        print(f"\n  Evaluating Embedding: {enc_name}")

        # Build main GCN
        sample = tfidf_enc.encode_doc(train_full[0])
        gcn_main = GCNEncoder(input_dim=sample.shape[1], hidden_dim=128, output_dim=64)

        # Build Fusion block for this embedding
        all_sem_tr = np.vstack([enc.encode_doc(d) for d in train_full])
        all_str_tr = np.vstack([gnn_fn(tfidf_enc, gcn_main)(d) for d in train_full])

        out_dim = min(128, all_sem_tr.shape[1])
        fusion_main = GatedFusion(all_sem_tr.shape[1], all_str_tr.shape[1], output_dim=out_dim)
        fusion_main.train_gate(all_sem_tr, all_str_tr, labels=None, n_iters=30)

        decoding_results, scorer, test_embs = run_decoding(
            train_full, test_full, fusion_main, enc, gcn_main, tfidf_enc
        )

        idx = enc_name.split('.')[0]
        scorer.save(f"scorer_{idx}.pkl")
        joblib.dump(gcn_main, f"gcn_{idx}.pkl")
        joblib.dump(fusion_main, f"fusion_{idx}.pkl")

        print(f"  Final Result ({enc_name}) -> Pairwise Ordering Accuracy: {decoding_results['pairwise_accuracy']:.2%}")

        # Generate Predictive Analysis
        analysis_content = analyze_sample_predictions(enc_name, test_embs, test_full, fusion_main, scorer, n_samples=5)
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(analysis_content)
            f.write("\n---\n\n")

    print(f"\n{'=' * 70}")
    print(f"  Pipeline complete.")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    main()
