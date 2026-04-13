import sys
import os
import joblib

# Ensure pipeline modules can be imported
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline'))

from pipeline.semantic_stream import (
    TFIDFEncoder, Word2VecEncoder, BERTEncoder,
    Word2VecTFIDFEncoder, RawTransformerEncoder, FineTunedDistilBERTEncoder
)
from pipeline.decoding import get_fused_embeddings, predict_document_order, PairwiseScorer

def print_banner(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def main():
    print_banner("Loading Saved Models")
    
    try:
        tfidf_enc = joblib.load("tfidf_enc.pkl")
        w2v_enc = joblib.load("w2v_enc.pkl")
        print("  [+] Loaded TF-IDF and Word2Vec base encoders")
    except Exception as e:
        print(f"  [!] Failed to load base encoders: {e}")
        print("  [!] Please run 'python pipeline/main.py' first to generate saved models.")
        return

    # Load transformer-based and composite encoders
    bert_enc = BERTEncoder()
    bert_enc.fit()
    
    w2v_tfidf_enc = Word2VecTFIDFEncoder(w2v_enc, tfidf_enc)
    w2v_tfidf_enc.fit([])
    
    raw_distilbert_enc = RawTransformerEncoder()
    raw_distilbert_enc.fit()
    
    ft_distilbert_enc = FineTunedDistilBERTEncoder()
    ft_distilbert_enc.load("ft_distilbert.pt")

    EMBEDDINGS = [
        ('1', 'Word2Vec (Mean Pooled)',         w2v_enc),
        ('2', 'Word2Vec (TF-IDF Weighted)',     w2v_tfidf_enc),
        ('3', 'TF-IDF (Sparse Vector)',         tfidf_enc),
        ('4', 'Contextual Token (DistilBERT)',  raw_distilbert_enc),
        ('5', 'Sequence Domain (SBERT)',        bert_enc),
        ('6', 'Fine-Tuned Token (DistilBERT [CLS])', ft_distilbert_enc),
    ]

    print_banner("Interactive Sentence Ordering")
    print("Enter a shuffled paragraph sentence by sentence.")
    print("Press ENTER on an empty line when you are finished.")

    while True:
        sentences = []
        i = 1
        print("\n--- New Input ---")
        while True:
            line = input(f"Sentence {i}: ").strip()
            if not line:
                break
            sentences.append(line)
            i += 1
            
        if not sentences:
            print("No sentences provided. Exiting.")
            break
            
        if len(sentences) == 1:
            print("Please enter at least 2 sentences.")
            continue

        doc = [{'sentences': sentences}]
        
        print("\n" + "="*50)
        print("Original Input Order:")
        for idx, s in enumerate(sentences):
            print(f"  [{idx}] {s}")
            
        print("\nPredicting ordered sequences...\n")
            
        for idx, name, enc in EMBEDDINGS:
            try:
                gcn = joblib.load(f"gcn_{idx}.pkl")
                fusion = joblib.load(f"fusion_{idx}.pkl")
                
                scorer = PairwiseScorer()
                scorer.load(f"scorer_{idx}.pkl")
                
                doc_embs = get_fused_embeddings(doc, fusion, enc, gcn, tfidf_enc)
                pred_order = predict_document_order(doc_embs[0], scorer)
                
                print(f"[{name}]")
                for pos, o in enumerate(pred_order):
                    print(f"  {pos+1}. {sentences[o]}")
                print()
                    
            except Exception as e:
                pass
                # Silently ignore missing models so we can run what's available
                # print(f"  [!] Missing models for {name}. Error: {e}")
                
        print("="*50)

if __name__ == '__main__':
    main()
