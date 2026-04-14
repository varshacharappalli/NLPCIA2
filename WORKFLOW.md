# NLP CIA2 — Complete Workflow

## Overview

**Task:** Sentence Ordering — given a shuffled set of sentences from a research abstract, predict their correct logical order.

**Approach:** Compare 6 sentence embedding methods by passing each through an identical pipeline: Graph-based structural encoding → Gated Fusion → Pairwise MLP Decoder → Tournament Ranking.

**Dataset:** ACL Anthology Network (AAN) — real NLP research paper abstracts (~63k rows in abstract.csv)

**Split:** 500 documents loaded → 400 train / 100 test

---

## Entry Point: `pipeline/main.py`

This is the orchestrator. It does no ML math itself — it imports all modules and calls them in sequence. Running `python pipeline/main.py` triggers all 5 steps below.

Key setup in main.py:

- `sys.path.insert(0, os.path.dirname(__file__))` — lets all modules import each other without a package install
- `warnings.filterwarnings("ignore")` — suppresses sklearn/transformers verbosity
- Imports `LogisticRegression` from sklearn (unused in final code — leftover, not called)

---

## Step 1: Dataset Loading

### File: `pipeline/dataset_generator.py`

**Called from main.py as:** `docs = generate_dataset(seed=42, limit=500)`

`generate_dataset()` is a thin wrapper that calls `load_aan_dataset(seed=42, limit=500)`.

### `load_aan_dataset(csv_path, min_sentences=3, max_sentences=6, limit, seed)`

1. Opens `abstract.csv` using `csv.DictReader` — reads the `Abstract` column from each row
2. For each abstract text, calls `run_preprocessor(text, flags=["--no-lowercase", "--no-punct"])`
   - `--no-lowercase` and `--no-punct` = only do sentence segmentation here, keep original text
   - This gives raw sentences for the ground-truth ordering
3. Filters: only keeps docs with `3 <= len(sentences) <= 6`
   - Too few = trivial (nothing to order), too many = too complex for this scope
4. If a doc has more than 6 sentences: randomly picks a contiguous window of 6
   - `start = random.randint(0, len-6)`, then `sentences[start:start+6]`
5. Appends doc as `{id: row_index, topic: "ACL-Anthology", sentences: [...]}`
   - `sentences` = correct canonical order (ground truth)
6. Stops when `len(docs) == 500`

**Actual output:**

```
[Loader] Reading abstract.csv...
[Loader] Reached limit of 500 documents.
[Loader] Successfully loaded 500 documents.
500 ACL abstracts loaded | Train: 400 | Test: 100
```

---

## Step 2: Train/Test Split

### File: `pipeline/data_loader.py` — `create_train_test_split(docs, test_ratio=0.2, seed=42)`

1. `random.seed(42)` then `random.shuffle(docs_copy)`
2. `split = int(0.8 * 500) = 400`
3. Returns `docs[:400]` as train, `docs[400:]` as test
4. Same seed → same split on every run (reproducible experiments)

---

## Step 3: C Preprocessor

### File: `pipeline/data_loader.py`

Three functions work together:

### `compile_c()`

1. Checks if `preprocessing/preprocess.exe` already exists → use it directly if so
2. Otherwise: searches for `gcc` via `shutil.which("gcc")` and a list of common Windows paths (MinGW, MSYS2, Scoop, Chocolatey, conda)
3. If gcc found: runs `subprocess.run(["gcc", "-Wall", "-O2", "-o", "preprocess.exe", "preprocess.c"])`
4. If gcc not found: sets global `_USE_C_BINARY = False`, uses Python fallback

### `preprocess_docs(docs)`

- Joins each doc's sentences into one string: `" ".join(doc["sentences"])`
- Calls `run_preprocessor(text)` for each doc
- `run_preprocessor` routes to: C binary via `subprocess.run` (stdin pipe) OR `_python_preprocess()`
- Returns list of `{id, topic, original_sentences, preprocessed_sentences, tokens, stats}`

### `_python_preprocess(text)` — Pure Python fallback

Implements the same algorithm as preprocess.c:

**Segmentation (`_py_segment`):**

- Scans character by character for `.`, `!`, `?`
- On hitting punctuation: looks ahead for whitespace + uppercase letter or quote
- Skips if the word before the period is an abbreviation (mr, dr, prof, etc, fig, vs, ...)
- Skips decimal numbers (digit before `.` AND digit after)
- If valid boundary: slices sentence, appends, advances position

**Lowercase:** `str.lower()`

**Punctuation removal (`_py_remove_punct`):**

- Keeps: alphanumeric, whitespace, hyphens, apostrophes
- Replaces other punct with a space (deduped)

**Tokenization:** `str.split()`

### `make_docs_with_sents(orig_docs, proc_results)` — in main.py

- Replaces each doc's sentences with the preprocessed version
- Safety: if preprocessed sentence count != original, keeps original sentences
- Produces `train_full` and `test_full` used for all downstream encoding

**Segmentation accuracy check (main.py):**

```python
seg_acc = sum(
    1 for o, p in zip(test_docs, test_proc)
    if p["stats"].get("num_sentences", 0) == len(o["sentences"])
) / len(test_docs)
```

Fraction of test docs where preprocessor found the exact same sentence count as the original.

**Actual output (this machine — no gcc):**

```
[INFO] gcc not found. Using Python reimplementation.
Sentence segmentation accuracy: 100.00%
```

---

## Step 4: Fitting All Six Encoders

### File: `pipeline/semantic_stream.py`

Before the evaluation loop, all encoders are fitted on the combined corpus:

```python
corpus = get_sentence_corpus(train_full + test_full)  # flat list of all sentences
```

Fitting on both train+test ensures the vocabulary/model sees all text. The task is unsupervised fitting
(no labels used), so this does not cause data leakage.

---

### Encoder 1: `TFIDFEncoder`

**Concept:** Each sentence is a bag of words. TF-IDF weights each word by how often it appears
in this sentence (TF) vs how rarely it appears across all sentences (IDF).
Common words like "the" get low weight; rare technical terms get high weight.

**`fit(corpus)`**

```python
self.vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
self.vec.fit(corpus)
```

Builds a vocabulary of top 5000 terms (unigrams + bigrams) by TF-IDF score across corpus.

**`encode_doc(doc)`**

```python
return self.vec.transform(doc["sentences"]).toarray()  # shape: (n_sents, 5000)
```

Output: sparse vector where each dimension = TF-IDF weight of that term in this sentence.

**Limitation for ordering:** TF-IDF captures topic, not sequence.
"We propose X" and "X achieves Y%" share vocabulary but differ in position.

---

### Encoder 2: `Word2VecEncoder`

**Concept:** Each word has a dense 100-dim vector learned from co-occurrence in a sliding window.
Similar words cluster together in vector space. Sentence vector = mean of word vectors.

**`fit(corpus)`**

```python
tokenized = [sent.lower().split() for sent in corpus]
self.model = Word2Vec(sentences=tokenized, vector_size=100, window=5,
                     min_count=1, epochs=10, workers=1, seed=42)
```

Trains skip-gram/CBOW on the corpus. `window=5` = each word predicts 5 words on either side.

**`encode_doc(doc)`**

- For each sentence: `np.mean([model.wv[w] for w in words if w in model.wv], axis=0)`
- If no known words: returns zero vector (100-dim)

**Limitation:** Mean pooling loses word order. Sentences with same words in different order are identical.

---

### Encoder 3: `BERTEncoder` (SBERT)

**Concept:** `all-MiniLM-L6-v2` is a sentence-transformers model pretrained specifically
to produce semantically meaningful sentence embeddings via contrastive learning on NLI and STS tasks.

**`fit()`**

```python
self.model = SentenceTransformer("all-MiniLM-L6-v2")
```

Downloads/loads pretrained weights. No training on our corpus.

**`encode_doc(doc)`**

```python
return self.model.encode(sentences, show_progress_bar=False)  # shape: (n_sents, 384)
```

Internally: BERT tokenize → 6 transformer layers → mean pool → L2 normalize → 384-dim vector.

**Why better than Word2Vec:** Captures context (word meaning depends on surrounding words).
"The model performs well" vs "The model performs poorly" → different vectors.

---

### Encoder 4: `Word2VecTFIDFEncoder`

**Concept:** Word2Vec mean pooling, but each word's contribution is weighted by its TF-IDF score.
Rare, informative words get higher weight; stopwords get lower weight.

**`__init__(w2v_encoder, tfidf_encoder)`**

- Stores references to already-fitted Word2Vec model (`w2v_encoder.model`) and TF-IDF vectorizer
- No separate training — relies on the two pre-fitted encoders

**`_sentence_vector(sentence)`**

1. `tfidf_vec = self.tfidf.transform([sentence])` → sparse vector, one entry per vocab term
2. For each word: look up Word2Vec embedding + TF-IDF weight from sparse vector
3. `np.average(vecs, axis=0, weights=tfidf_weights)` → 100-dim weighted sentence vector

**Result in practice:** Slightly below Word2Vec plain (47.80% vs 52.20%)
TF-IDF weights may de-emphasize contextual function words that carry structural/positional meaning.

---

### Encoder 5: `RawTransformerEncoder`

**Concept:** Uses DistilBERT (distilled BERT — faster, 6 layers instead of 12) without fine-tuning.
Extracts token-level contextual embeddings and mean-pools them into a sentence vector.

**`fit(corpus)`**

```python
self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
self.model = AutoModel.from_pretrained("distilbert-base-uncased")
self.model.eval()
```

Loads pretrained weights. No fine-tuning.

**`encode(sentences)`**

```python
inputs = tokenizer(sentences, padding=True, truncation=True, max_length=128)
outputs = model(**inputs)  # last_hidden_state: (batch, seq_len, 768)

# Attention-mask-weighted mean pool:
mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
sum_emb  = torch.sum(outputs.last_hidden_state * mask, dim=1)
sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
mean_pooled = sum_emb / sum_mask  # shape: (batch, 768)
```

The attention mask ensures padding tokens (added to make sequences same length) are excluded from the mean.

**Why mean pool and not [CLS] token?**
The [CLS] token is only meaningful for classification if the model was explicitly trained on a
classification task (like BERT-for-classification). DistilBERT base was trained on masked LM,
so its [CLS] is not task-specific. Mean pooling over real tokens is more robust.

---

### Encoder 6: `FineTunedDistilBERTEncoder`

**Concept:** Fine-tunes DistilBERT on the actual ordering task using a Siamese network.
Two sentences are passed in simultaneously, and the model learns to assign a higher scalar score
to whichever sentence comes earlier in the document.

**`fit(train_docs)` — Dataset construction**

```python
for doc in train_docs:
    for i in range(n):
        for j in range(n):
            if i == j: continue
            texts_a.append(sents[i])
            texts_b.append(sents[j])
            labels.append(1.0 if i < j else -1.0)
```

All ordered pairs from all 400 training docs. Label +1 means "A comes before B", -1 means "B comes before A".

**Model architecture: `PairwiseRankingModel`**

```python
class PairwiseRankingModel(nn.Module):
    def __init__(self):
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.scorer = nn.Linear(768, 1)  # [CLS] -> scalar

    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        cls_a = self.distilbert(input_ids_a, attention_mask_a).last_hidden_state[:, 0, :]
        cls_b = self.distilbert(input_ids_b, attention_mask_b).last_hidden_state[:, 0, :]
        return self.scorer(cls_a).squeeze(-1), self.scorer(cls_b).squeeze(-1)
```

Both sentences share the same DistilBERT weights (Siamese = same model, different inputs).
[CLS] token (position 0) is extracted as a single 768-dim summary of the sentence.
Linear scorer projects 768-dim → 1 scalar = "how early does this sentence appear?"

**Loss: `MarginRankingLoss(margin=1.0)`**

```
loss = max(0, -y * (score_a - score_b) + margin)
```

- If y=+1 (A before B): penalizes when score_A < score_B + 1
- If y=-1 (B before A): penalizes when score_B < score_A + 1
- Margin=1.0 enforces a gap between scores, not just correct ordering

**Training:**

- Optimizer: `AdamW(lr=2e-5)` — small learning rate standard for transformer fine-tuning
- 3 epochs, batch_size=16
- Device: CUDA if available, else CPU

**Actual training output:**

```
Epoch 1/3 — Avg Loss: 0.4408  (106 batches, ~7 min on CPU)
Epoch 2/3 — Avg Loss: 0.1040  (~4.5 min)
Epoch 3/3 — Avg Loss: 0.0711  (~4.7 min)
```

Loss drops by 4x from epoch 1 to 3 — strong convergence.

**`evaluate_pairwise_accuracy(test_docs)`** — runs before the main loop

- Builds pairs (i < j) from test docs
- Predicts: is score_A > score_B?
- Measures accuracy of the scalar scorer alone (before MLP decoder)

```
Siamese Scorer Pairwise Accuracy: 74.84%
```

**`encode(sentences)`** — used during evaluation loop

```python
outputs = self.model.distilbert(**inputs)
return outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS], shape: (n, 768)
```

[CLS] is now meaningful because the whole model was trained to make [CLS] position-aware.

---

## Step 5: Main Evaluation Loop — runs once per encoder

```python
for enc_name, enc in EMBEDDINGS:
    # Build GCN (fresh each iteration, same init)
    sample = tfidf_enc.encode_doc(train_full[0])  # (n_sents, tfidf_dim)
    gcn_main = GCNEncoder(input_dim=sample.shape[1], hidden_dim=128, output_dim=64)

    # Stack all train embeddings for gate training
    all_sem_tr = np.vstack([enc.encode_doc(d) for d in train_full])       # (total_sents, sem_dim)
    all_str_tr = np.vstack([gnn_fn(tfidf_enc, gcn_main)(d) for d in train_full])  # (total_sents, 64)

    out_dim = min(128, all_sem_tr.shape[1])
    fusion_main = GatedFusion(all_sem_tr.shape[1], all_str_tr.shape[1], output_dim=out_dim)
    fusion_main.train_gate(all_sem_tr, all_str_tr, labels=None, n_iters=30)

    decoding_results, scorer, test_embs = run_decoding(
        train_full, test_full, fusion_main, enc, gcn_main, tfidf_enc)
```

Note: the GCN uses random-initialized weights that are **never updated** (no backprop on GCN).
It acts as a fixed structural feature extractor.

---

### 5a: Structural Encoding

### File: `pipeline/structural_stream.py`

**Why graphs?** Sentences have structural relationships beyond word content:
adjacency (what comes next), topic proximity (same subject nearby), entity coreference (same actors).
A graph captures these as edge weights between sentence nodes.

#### `build_local_graph(n)` — weight 0.40

```python
A[i][i+1] = A[i+1][i] = 1.0  for i in range(n-1)
```

Bidirectional edges between consecutive sentences. Captures narrative flow.
Highest weight because adjacent sentences are most strongly related structurally.

#### `build_midrange_graph(n, window=3)` — weight 0.25

```python
for i in range(n):
    for j in range(i+1, min(i+window+1, n)):
        A[i][j] = A[j][i] = 1.0
```

Connects sentences within a window of 3. Captures paragraph-level coherence.

#### `build_global_graph(embeddings, threshold=0.3)` — weight 0.20

```python
for i, j in all_pairs:
    sim = cosine_similarity(embeddings[i], embeddings[j])
    if sim > 0.3: A[i][j] = A[j][i] = sim
```

Uses TF-IDF embeddings to connect thematically similar sentences regardless of position.

#### `build_entity_graph(sentences)` — weight 0.15

```python
# entities = words > 4 chars OR capitalized
entities[i] & entities[j]
A[i][j] = len(intersection) / len(union)  # Jaccard similarity
```

Sentences sharing named entities (approximated as long/capitalized words) are connected.

#### `merge_graphs(local, midrange, global_g, entity)`

```python
return 0.4*local + 0.25*midrange + 0.2*global_g + 0.15*entity
```

Single combined adjacency matrix `A` of shape `(n, n)`.

#### `GCNEncoder` — 2-layer Graph Convolutional Network

Initialized with: `input_dim=sample.shape[1], hidden_dim=128, output_dim=64`
Weights are random (He initialization), never updated during the pipeline.

**`gcn_layer(A, X, W, b, activation)`**

```
A_hat  = A + I                              # add self-loops
degree = A_hat.sum(axis=1)                  # degree of each node
D_inv_sqrt = diag(1 / sqrt(degree))        # D^(-0.5)
A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt   # symmetric normalization
Z = A_norm @ X @ W + b                     # graph-diffused linear transform
return activation(Z)
```

Symmetric normalization prevents high-degree nodes (sentences connected to many others) from dominating.

**Forward pass through GCNEncoder:**

```
H1 = gcn_layer(A, X,  W1, b1, relu)   # (n, 5000) -> (n, 128)
H2 = gcn_layer(A, H1, W2, b2, tanh)   # (n, 128)  -> (n, 64)
```

Output: 64-dim structural embedding per sentence that encodes its graph neighborhood.

---

### 5b: Gated Fusion

### File: `pipeline/fusion.py` — class: `GatedFusion`

**Why fusion?** Semantic embeddings capture meaning; structural embeddings capture graph position.
A learned gate decides per-dimension how much to trust each stream.

**Initialization:**

```python
W_sem    = randn(sem_dim, output_dim) * sqrt(2/sem_dim)   # He init
W_struct = randn(64, output_dim)      * sqrt(2/64)
W_gate   = randn(output_dim*2, output_dim) * 0.01        # small init
b_gate   = zeros(output_dim)
```

**`project(sem_emb, struct_emb)`**

```python
p_sem    = tanh(sem_emb    @ W_sem)     # (n, output_dim)
p_struct = tanh(struct_emb @ W_struct)  # (n, output_dim)
```

Projects both to the same output_dim so they can be combined.

**`fuse(sem_emb, struct_emb)`**

```python
p_sem, p_struct = project(sem_emb, struct_emb)
combined = concat([p_sem, p_struct], axis=-1)      # (n, output_dim*2)
gate     = sigmoid(combined @ W_gate + b_gate)     # (n, output_dim) — values in (0,1)
fused    = gate * p_sem + (1 - gate) * p_struct    # (n, output_dim)
```

Gate near 1.0 = use semantic stream for that dimension.
Gate near 0.0 = use structural stream for that dimension.
Each output dimension can have a different gate value.

**`train_gate(sem_train, struct_train, labels=None, n_iters=50)`** (called with `n_iters=30` in main loop)
Uses random perturbation search (not gradient descent):

```python
for i in range(n_iters):  # n_iters=30 as called from main
    W_gate += randn_like(W_gate) * 0.1  # small random perturbation
    b_gate += randn_like(b_gate) * 0.1
    fused  = fuse(sem_train, struct_train)
    target = tanh((sem_train @ W_sem + struct_train @ W_struct) / 2)
    loss   = mean((fused - target) ** 2)
    if loss < best_loss: keep perturbation
    else: revert to best
```

Target = average of both projections. Gate learns to blend them to match this average.

---

### 5c: Pairwise Decoding

### File: `pipeline/decoding.py`

**`get_fused_embeddings(docs, fusion_module, semantic_encoder, gcn, tfidf_encoder)`**

For each document:

1. `sem_embs   = semantic_encoder.encode_doc(doc)` → `(n, sem_dim)`
2. `init_feats = tfidf_encoder.encode_doc(doc)` → `(n, 5000)` — GCN node features
3. Build 4 graphs on the document, merge → `A (n, n)`
4. `struct_embs = gcn.encode(A, init_feats)` → `(n, 64)`
5. `fused = fusion_module.fuse(sem_embs, struct_embs)` → `(n, output_dim)`

Returns: list of `{fused, n, sentences}` per doc

**`build_decoding_dataset(doc_embs)`**

For every ordered pair `(i, j)` where `i != j` in every doc:

```python
feat = concat([fused[i], fused[j], abs(fused[i]-fused[j]), fused[i]*fused[j]])
# feat dim = 4 * output_dim
label = 1 if i < j else 0
```

Why these 4 components:

- `fused[i]` — absolute embedding of sentence i (captures its position-independent properties)
- `fused[j]` — absolute embedding of sentence j
- `|fused[i]-fused[j]|` — element-wise difference (how different are they?)
- `fused[i]*fused[j]` — element-wise product (shared dimensions / interaction terms)

**`PairwiseScorer` (MLP)**

```python
MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu",
              max_iter=200, random_state=42,
              early_stopping=True, validation_fraction=0.1)
```

Trained on all pairwise features from 400 training docs.
`predict_proba(X)[:, 1]` = P(sentence i comes before j).

**`predict_document_order(doc_emb, scorer)`**

```python
score_matrix = zeros(n, n)
for i, j in all_pairs:
    feat = concat([fused[i], fused[j], abs(diff), product])
    score_matrix[i][j] = scorer.predict_proba(feat)[0]  # P(i before j)
return tournament_to_order(score_matrix)
```

---

### 5d: Tournament Decoding + Evaluation Metrics

### File: `pipeline/metrics.py`

**`tournament_to_order(score_matrix)`**

```python
row_scores = score_matrix.sum(axis=1)   # how many battles did sentence i win?
return list(argsort(-row_scores))       # sort sentences by descending win-count
```

Think of it like a round-robin tournament: each sentence "plays" all others.
The sentence that beats the most others should come first.

**`kendall_tau(pred_order, true_order)`**

```python
tau, _ = scipy.stats.kendalltau(pred_order, true_order)
```

Counts concordant pairs (same relative order in both) vs discordant pairs.
`tau = (concordant - discordant) / total_pairs`

- `+1.0` = perfect match
- `0.0` = random
- `-1.0` = completely reversed

**`run_decoding()` returns:**

- `{pairwise_accuracy}` — fraction of test sentence pairs ordered correctly
- `scorer` — trained MLP
- `test_embs` — fused embeddings for test docs

---

## Results

### Pairwise Ordering Accuracy — Test Set (100 docs)

| Rank | Encoder                     | Accuracy   |
| ---- | --------------------------- | ---------- |
| 1    | Fine-Tuned DistilBERT [CLS] | **75.47%** |
| 2    | Raw DistilBERT (mean pool)  | 66.98%     |
| 3    | SBERT (all-MiniLM-L6-v2)    | 64.47%     |
| 4    | Word2Vec (mean pool)        | 52.20%     |
| 5    | TF-IDF (sparse vector)      | 48.43%     |
| 6    | Word2Vec + TF-IDF weighted  | 47.80%     |

### Result Analysis (Important for Viva)

**1. Fine-tuning gives the biggest gain (+8.5 points over raw DistilBERT)**
Fine-tuned = 75.47%, Raw = 66.98%. The Siamese MarginRankingLoss explicitly trains
the model to assign higher scores to earlier sentences. The [CLS] token becomes a
position-aware "earliness score" rather than just a generic sentence summary.

**2. Contextual embeddings beat static by ~15–20 points**
All 3 transformer methods (64.47% – 75.47%) vs all 3 bag-of-words methods (47.80% – 52.20%).
Transformers understand that the same word means different things in different contexts.
Word2Vec and TF-IDF only count word occurrences — no understanding of sequence or meaning.

**3. SBERT nearly matches raw DistilBERT without any task fine-tuning**
SBERT (64.47%) is only 2.5 points behind DistilBERT (66.98%) despite no fine-tuning.
SBERT was pretrained on NLI + semantic similarity tasks which naturally involve sentence comparison,
making it well-suited for ordering even zero-shot.

**4. TF-IDF and Word2Vec hover near 50% (chance level)**
Confirms that word overlap carries no ordering signal for research abstracts.
Sentences at every position in an abstract share similar vocabulary.
You cannot tell "We propose X" (intro) from "X achieves 95% accuracy" (results) by word overlap alone.

**5. TF-IDF weighting on Word2Vec actually hurts (47.80% < 52.20%)**
The TF-IDF weighting boosts rare technical terms. These terms appear throughout the abstract
without positional pattern — so upweighting them destroys the weak positional signal Word2Vec had.

**6. Graph structure does not compensate for weak semantic representations**
All 6 encoders use the identical GCN + GatedFusion + MLP stack.
The 27.67-point spread (47.80% to 75.47%) is purely from the quality of the semantic encoder.
Structural information helps (all methods presumably beat a random 50% baseline with the GCN),
but cannot overcome fundamentally non-positional representations.

**7. Fine-tuned Siamese scorer is nearly as good as the full pipeline**
Siamese scorer standalone: 74.84%. Full pipeline with MLP decoder: 75.47%.
Difference: 0.63 points. The Siamese training itself captures most of the ordering signal.
The GCN + fusion + MLP add marginal gain. The fine-tuning is the key innovation.

---

## File-by-File Quick Reference

| File                     | What it does                                                                                                                                              |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `main.py`                | Orchestrator. Calls everything in order. Defines `gnn_fn()` (builds graphs + runs GCN per doc) and `make_docs_with_sents()` (swaps in cleaned sentences). |
| `dataset_generator.py`   | Reads abstract.csv, segments + filters, returns 100 docs in correct order.                                                                                |
| `data_loader.py`         | Compiles/runs C preprocessor or Python fallback. Does segmentation, lowercasing, punct removal. Also does train/test split.                               |
| `semantic_stream.py`     | All 6 encoder classes. Each has `fit(corpus)` and `encode_doc(doc)`.                                                                                      |
| `structural_stream.py`   | Builds 4 graph types per document, merges them, runs 2-layer GCN to produce 64-dim structural embeddings.                                                 |
| `fusion.py`              | GatedFusion: sigmoid gate trained by random search to blend semantic + structural embeddings.                                                             |
| `decoding.py`            | Builds pairwise feature dataset, trains MLP scorer, runs tournament decoding on test docs.                                                                |
| `metrics.py`             | `tournament_to_order` (row-sum ranking) and `kendall_tau` (scipy-based correlation metric).                                                               |
| `prediction_analyzer.py` | Generates `test_predictions.md` — actual vs predicted sentence tables with Kendall Tau per doc.                                                           |
