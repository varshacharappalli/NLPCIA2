```text
======================================================================
  NLP CIA2 -- Sentence Ordering Pipeline
======================================================================

======================================================================
  Step 1: ACL Anthology Dataset Loading
======================================================================
  [Loader] Reading abstract.csv...
  [Loader] Reached limit of 500 documents.
  [Loader] Successfully loaded 500 documents.
  500 ACL abstracts loaded | Train: 400 | Test: 100

======================================================================
  Step 2: C Preprocessor
======================================================================
  [INFO] gcc not found. C source exists at preprocessing/preprocess.c
         Using Python reimplementation of the same preprocessing logic.
         To use the actual C binary: install MinGW (choco install mingw)
         or Scoop (scoop install gcc), then re-run.
  Sentence segmentation accuracy: 100.00%

======================================================================
  Step 3: Fitting Encoders
======================================================================
  [1/5] TF-IDF fitted
  [2/5] Word2Vec fitted
  Loading BERT model: all-MiniLM-L6-v2 ...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 3650.36it/s]
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  |
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  |

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
  BERT model loaded.
  [3/5] SBERT loaded
  [4/5] Word2Vec+TFIDF fitted
  Loading Transformer model: distilbert-base-uncased ...
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 3358.96it/s]
DistilBertModel LOAD REPORT from: distilbert-base-uncased
Key                     | Status     |  |
------------------------+------------+--+-
vocab_projector.bias    | UNEXPECTED |  |
vocab_transform.bias    | UNEXPECTED |  |
vocab_layer_norm.bias   | UNEXPECTED |  |
vocab_transform.weight  | UNEXPECTED |  |
vocab_layer_norm.weight | UNEXPECTED |  |

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
  Transformer model distilbert-base-uncased loaded.
  [5/6] Raw DistilBERT loaded
  Preparing Siamese pairwise dataset for distilbert-base-uncased...
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 3356.54it/s]
DistilBertModel LOAD REPORT from: distilbert-base-uncased
Key                     | Status     |  |
------------------------+------------+--+-
vocab_projector.bias    | UNEXPECTED |  |
vocab_transform.bias    | UNEXPECTED |  |
vocab_layer_norm.bias   | UNEXPECTED |  |
vocab_transform.weight  | UNEXPECTED |  |
vocab_layer_norm.weight | UNEXPECTED |  |

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
  Fine-tuning distilbert-base-uncased (Siamese Margin Ranking)...
Training Siamese Network (Epoch 1/3): 100%|███████████████████████████████████████████████████████████| 508/508 [55:16<00:00,  6.53s/it] 
  Epoch 1/3 complete. Avg Loss: 0.3758
Training Siamese Network (Epoch 2/3): 100%|███████████████████████████████████████████████████████████| 508/508 [41:25<00:00,  4.89s/it] 
  Epoch 2/3 complete. Avg Loss: 0.1450
Training Siamese Network (Epoch 3/3): 100%|███████████████████████████████████████████████████████████| 508/508 [48:24<00:00,  5.72s/it] 
  Epoch 3/3 complete. Avg Loss: 0.0720
  [6/6] Fine-Tuned DistilBERT fitted
  Saving base encoders...
  Saved Siamese model weights to ft_distilbert.pt
  Evaluating sequence classification head performance for Fine-Tuned DistilBERT:
  Evaluating Siamese scalar scorer on test set...
  Siamese Scorer Pairwise Accuracy: 0.8117

======================================================================
  Step 4: Embedding Evaluation (Pairwise Accuracy in Final Step)
======================================================================

  Evaluating Embedding: 1. Word2Vec (Mean Pooled)
  Computing fused embeddings...
  Training pairwise scorer (MLP)...
  Pairwise Scoring Accuracy: 0.6415
  Saved Scorer (MLP) to scorer_1.pkl
  Final Result (1. Word2Vec (Mean Pooled)) -> Pairwise Ordering Accuracy: 64.15%

  Evaluating Embedding: 2. Word2Vec (TF-IDF Weighted)
  Computing fused embeddings...
  Training pairwise scorer (MLP)...
  Pairwise Scoring Accuracy: 0.6255
  Saved Scorer (MLP) to scorer_2.pkl
  Final Result (2. Word2Vec (TF-IDF Weighted)) -> Pairwise Ordering Accuracy: 62.55%

  Evaluating Embedding: 3. TF-IDF (Sparse Vector)
  Computing fused embeddings...
  Training pairwise scorer (MLP)...
  Pairwise Scoring Accuracy: 0.6287
  Saved Scorer (MLP) to scorer_3.pkl
  Final Result (3. TF-IDF (Sparse Vector)) -> Pairwise Ordering Accuracy: 62.87%

  Evaluating Embedding: 4. Contextual Token (DistilBERT)
  Computing fused embeddings...
  Training pairwise scorer (MLP)...
  Pairwise Scoring Accuracy: 0.7191
  Saved Scorer (MLP) to scorer_4.pkl
  Final Result (4. Contextual Token (DistilBERT)) -> Pairwise Ordering Accuracy: 71.91%

  Evaluating Embedding: 5. Sequence Domain (SBERT)
  Computing fused embeddings...
  Training pairwise scorer (MLP)...
  Pairwise Scoring Accuracy: 0.6468
  Saved Scorer (MLP) to scorer_5.pkl
  Final Result (5. Sequence Domain (SBERT)) -> Pairwise Ordering Accuracy: 64.68%

  Evaluating Embedding: 6. Fine-Tuned Token (DistilBERT [CLS])
  Computing fused embeddings...
  Training pairwise scorer (MLP)...
  Pairwise Scoring Accuracy: 0.8059
  Saved Scorer (MLP) to scorer_6.pkl
  Final Result (6. Fine-Tuned Token (DistilBERT [CLS])) -> Pairwise Ordering Accuracy: 80.59%

======================================================================
  Pipeline complete.
======================================================================
```
