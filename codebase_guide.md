# NLP CIA2 Codebase Guide

This guide is intended for anyone reading the source code of this Sentence Ordering Pipeline. It provides a file-by-file breakdown of what every major class and function actually does to make the whole software work.

---

## 1. `pipeline/main.py`
This is your **Orchestrator**. It doesn't define much actual math or string-logic; instead, it imports all the building blocks and triggers them in sequence.

- **`main()`**: The root entrypoint. It runs:
  1. Dataset loading limitation (currently configured to 500 documents for higher statistical significance).
  2. Data splitting into 80/20 train/test batches.
  3. Preprocessing using the C binary bindings.
  4. Feeding the semantic data into all six encoder classes to fit/prepare them.
  5. Iterating over the 6 embeddings inside the `EMBEDDINGS` tuple, pushing them through the GCN, the Fusion gate, the Scorer MLP, and dumping the results to `test_predictions.md`.

---

## 2. `pipeline/dataset_generator.py`
Handles fetching dummy or raw data.
- **`generate_dataset(seed, limit)`**: Navigates to `abstract.csv`, drops empty/null items, sets a fixed random seed for reproducibility, shuffles the CSV, and extracts a specified limit of abstracts. It specifically targets sequential slices of abstracts (length 3 to 6 sentences) to act as standard matrices.

---

## 3. `pipeline/data_loader.py`
Handles memory preparation and formatting logic.
- **`compile_c(c_source_path, binary_path)`**: The program speeds up dataset parsing locally by utilizing a compiled `preprocess.exe` C binary. This function verifies if GCC/MinGW is available and automatically compiles it on Windows OS natively.
- **`preprocess_docs(docs)`**: Formats inputs into a raw string, opens a background subprocess `subprocess.run` to funnel the text through the high-speed C processor (tokenizing and standardizing the text), and processes the structural format outputted by the program. Fallbacks to Python string `.split()` processing if C fails.
- **`create_train_test_split(...)`**: Simple deterministic list slicing.

---

## 4. `pipeline/semantic_stream.py`
Contains all the Natural Language algorithms for "understanding" sentences.
Generally speaking, every class here supports:
- **`fit(corpus)`**: Preparing the space (building vocabulary limits, generating pairs, or doing nothing for pre-trained models).
- **`encode(sentences)`**: Translating a list of English sentences into dense numpy vectors.

### Detailed Model Descriptions

#### 1. Word2Vec (Mean Pooled)
- **Concept**: Learned word-level associations.
- **Implementation**: Trains a `gensim.models.Word2Vec` model on the provided corpus. Sentences are tokenized, and the embedding for each sentence is computed by taking the mathematical average (mean) of the vectors of its constituent words.
- **Strength**: Captures semantic similarity between words; fast to compute.

#### 2. Word2Vec (TF-IDF Weighted)
- **Concept**: Importance-aware word associations.
- **Implementation**: Uses the same Word2Vec model but incorporates a `TFIDFEncoder`. Instead of a simple mean, it calculates a weighted average where word vectors are multiplied by their TF-IDF scores before sum-pooling. 
- **Strength**: Reduces the impact of "stop words" and emphasizes content-heavy keywords uniquely identifying a sentence.

#### 3. TF-IDF (Sparse Vector)
- **Concept**: Term frequency-inverse document frequency.
- **Implementation**: Uses `sklearn.feature_extraction.text.TfidfVectorizer`. It maps sentences into a high-dimensional sparse space where each dimension represents a word or n-gram (1-2 words).
- **Strength**: Excellent at finding exact word matches and identifying unique sentence signatures.

#### 4. Contextual Token (DistilBERT)
- **Concept**: Attention-based contextual understanding.
- **Implementation**: Utilizes `distilbert-base-uncased` from HuggingFace. It processes the full sentence and extracts the `last_hidden_state`. It then applies a mean-pooling operation across all token embeddings (excluding padding via an attention mask) to get a 768-dimensional sentence vector.
- **Strength**: Understands word meanings based on their context within the sentence.

#### 5. Sequence Domain (SBERT)
- **Concept**: Sentence-level optimization.
- **Implementation**: Uses `sentence_transformers` with the `all-MiniLM-L6-v2` model. This is a BERT-based model specifically fine-tuned on millions of sentence pairs (NLI/STS tasks) to ensure that sentences with similar meanings are close in vector space.
- **Strength**: Best overall general-purpose sentence representation without custom training.

#### 6. Fine-Tuned Token (Siamese DistilBERT [CLS])
- **Concept**: Structure-aware fine-tuning.
- **Implementation**: A custom implementation where we fine-tune a DistilBERT model on our specific ACL dataset.
  - **Network**: A Siamese architecture that processes sentence pairs $(A, B)$ through shared weights.
  - **Loss**: Uses `nn.MarginRankingLoss` (Margin Ranking). It learns to assign a higher scalar score to sentence $A$ if it belongs before sentence $B$.
  - **Representation**: We extract the `[CLS]` token (index 0) from the final layer.
- **Strength**: The highest performer (80%+). It "knows" about the logical flow of research abstracts because it was trained to rank them.

---

## 5. `pipeline/structural_stream.py`
Handles Graph theory equations. How are the sentences spatially arranged?
- **`cosine_similarity()`**: Simple 1D vector mathematical dot product divisor used to deduce similarity.
- **`build_local_graph(n)`**: Connects nodes $(i, i+1)$ signifying adjacent sentences in chronological layouts.
- **`build_midrange_graph()`**: Connects an expanded chronological sliding window to trace narrative jumps.
- **`build_global_graph()`**: Uses embeddings to find mathematically similar sentences regardless of spacing matrix index.
- **`build_entity_graph()`**: Checks capitalization/named-entity extraction sets via Python mapping to connect actors/subjects.
- **`merge_graphs(...)`**: Accepts the four graphs above, multiplies them by constant float weight modifiers (e.g. `0.4` for local dependencies) and returns the standard unified matrix $A$.
- **`gcn_layer(...) / GCNEncoder`**: Iterates matrices through a 2-Level neural layer. It calculates `A_norm = D^-0.5 * A * D^-0.5`. It takes TF-IDF node features and pushes them through spatial propagation arrays using `relu` and `tanh` activation formulas.

---

## 6. `pipeline/fusion.py`
The "brain" of the combination logic.
- **`GatedFusion`**: A neural module wrapped in an Sklearn architecture. 
  - **`train_gate(sem, str, ...)`**: Uses random-search perturbations to build a blending equation `(gate_weights)` utilizing `sigmoid` formulas.
  - **`forward()` / `_call_()`**: Returns the final feature array derived from: `(Semantic * Gate) + (Structural * (1 - Gate))`.

---

## 7. `pipeline/metrics.py` & `pipeline/decoding.py`
The final judgment phase. 
**`decoding.py`**
- **`run_decoding(...)`**: This is where $O(N^2)$ expansion happens. We loop through the train dataset and configure tuples `(i, j, dist_diff, interaction)`. It then executes an `MLPClassifier` with layers `256, 128` to classify `1.0` if `i` is chronologically before `j`. It runs the test batches, sums probabilities per sentence, and calls `metrics.tournament_to_order()`.

**`metrics.py`**
- **`tournament_to_order(scores)`**: Using the raw outputted sum array from the decoder, this uses `np.argsort` inversions to spit out the clean integer ordering array (e.g `[3, 0, 1, 2]`).
- **`kendall_tau(pred, actual)`**: Generates the final evaluation integer metrics on "how many pairwise inversions occurred". `1.0` implies a perfectly ordered document.

---

## 8. `pipeline/prediction_analyzer.py`
The reporting subsystem.
- **`analyze_sample_predictions(...)`**: Parses evaluation runs and dumps qualitative formatted Markdown strings. Using index matching against the raw text elements, it visually highlights things like `First/Last Match`, `Kendall Tau score %`, and exactly where sentence swap mistakes occurred during testing.
