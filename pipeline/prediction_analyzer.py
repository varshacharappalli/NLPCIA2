"""
prediction_analyzer.py
Generates Actual vs Predicted ordering analysis for test documents.
"""
import numpy as np
from metrics import kendall_tau

def analyze_sample_predictions(name, test_embs, test_docs, fusion_module, scorer, n_samples=2):
    """
    Takes a list of document embeddings and ground truth docs,
    predicts their order, and returns a formatted Markdown analysis.
    """
    from decoding import predict_document_order
    
    output = [f"### Embedding: {name}\n"]
    
    # Select sample indices (first n_samples)
    indices = range(min(n_samples, len(test_docs)))
    
    for idx in indices:
        doc = test_docs[idx]
        emb = test_embs[idx]
        
        # Predict order [indices of sentences in predicted order]
        pred_order = predict_document_order(emb, scorer)
        actual_order = list(range(len(doc['sentences'])))
        
        # Calculate Kendall Tau for this specific doc
        tau = kendall_tau(pred_order, actual_order)
        
        output.append(f"#### Sample Document {idx+1} (ID: {doc['id']})")
        output.append(f"- **Kendall's Tau Score**: {tau:.4f}")
        output.append("| Order | Actual Sentence | Predicted Sentence | Match? |")
        output.append("| :--- | :--- | :--- | :--- |")
        
        # sentence-by-sentence comparison
        for i in range(len(doc['sentences'])):
            actual_text = doc['sentences'][i]
            # pred_order[i] is the index of the sentence the model thinks should be at rank i
            # But wait, usually pred_order is [sentence_idx_at_pos_0, sentence_idx_at_pos_1, ...]
            # So the sentence at rank i is doc['sentences'][pred_order[i]]
            pred_idx = pred_order[i]
            pred_text = doc['sentences'][pred_idx]
            match = "✅" if pred_idx == i else "❌"
            
            output.append(f"| {i+1} | {actual_text[:80]}... | {pred_text[:80]}... | {match} |")
        
        # Qualitative Analysis
        analysis = []
        if tau == 1.0:
            analysis.append("Perfect ordering.")
        else:
            if pred_order[0] == 0:
                analysis.append("Correctly identified the opening sentence.")
            else:
                analysis.append(f"Incorrect opening: Predicted sentence {pred_order[0]+1} as start.")
            
            if pred_order[-1] == len(actual_order) - 1:
                analysis.append("Correctly identified the concluding sentence.")
                
            # Check for simple swaps
            swaps = 0
            for i in range(len(actual_order) - 1):
                if pred_order[i] == i + 1 and pred_order[i+1] == i:
                    swaps += 1
            if swaps > 0:
                analysis.append(f"Contains {swaps} adjacent sentence swap(s).")
                
        output.append(f"\n**Analysis**: {' '.join(analysis)}\n")
        
    return "\n".join(output) + "\n"
