# NLP CIA2 - Sentence Ordering Pipeline

This project implements a comprehensive Natural Language Processing (NLP) pipeline for the task of **Sentence Ordering**. The goal is to take a randomly shuffled set of sentences from a research abstract and reconstruct their original, logical order.

This pipeline leverages a novel architecture by combining **Semantic Embeddings** (understanding what the text means) with **Structural Embeddings** (understanding how sentences connect to one another) through a learnable **Gated Fusion** module, culminating in a sequence decoding phase.

---

## Project Statistics and Results

The pipeline processes documents from the **ACL Anthology Network (AAN)** dataset. By default, the execution performs the following:

- **Total Documents Processed:** 500 real-world abstracts (configured via `limit` flag)
- **Train/Test Split:** 80% Training (400 abstracts), 20% Testing (100 abstracts)
- **Sentence Bounds:** Abstracts are filtered or windowed to manageable lengths (typically 3 to 6 sentences) to study complex pairwise combinations.

### Final Pairwise Ordering Accuracy
During the final test phase, the pipeline predicts whether sentence $i$ comes before sentence $j$. The most recent evaluation of the fusion architecture yielded the following pairwise ordering accuracies on the test set:

| Semantic Encoder Used | Pairwise Accuracy |
| :--- | :---: |
| **Word2Vec (Mean Pooled)** | 64.15% |
| **Word2Vec (TF-IDF Weighted)** | 62.55% |
| **TF-IDF (Sparse Vector)** | 62.87% |
| **Contextual Token (DistilBERT)** | 71.91% |
| **Sequence Domain (SBERT)** | 64.68% |
| **Fine-Tuned Token (Siamese DistilBERT [CLS])** | **80.59%** |

*Fine-tuning the DistilBERT base using a Siamese Network and structural Margin Ranking Loss results in the highest pairwise ordering accuracy, proving that deep sequence-dependency embeddings combined with structural graphs provide the best signal for logical document reasoning.*

---

## Detailed Architecture and Step-by-Step Flow

The program logic is orchestrated centrally by `pipeline/main.py`. Here is a breakdown of everything that happens behind the scenes:

### Step 1: Dataset Generation & Loading 
*(Scripts: `dataset_generator.py` and `data_loader.py`)*

1. **Extraction**: The pipeline reads `abstract.csv`. It filters out empty rows and targets a subset of the dataset (currently configured to 500 documents for the latest evaluation).
2. **C-Based Preprocessing**: A custom C binary (`preprocessing/preprocess.c`) is dynamically compiled and executed. It handles high-speed tokenisation, lowercasing, abbreviation-aware sentence segmentation, and punctuation stripping. (If the C compiler is missing, it falls back to a pure Python implementation).
3. **Splitting**: The preprocessed sentences are grouped into an 80/20 train/test split for the models to learn from.

### Step 2: Semantic Stream 
*(Script: `semantic_stream.py`)*

The Semantic Stream's job is to read words and output meaning. It maps sentences into pure numerical vectors. The pipeline tests five distinct methodologies:
1. **Word2Vec (Mean Pooled)**: Generates traditional word embeddings and averages them across the sentence.
2. **Word2Vec (TF-IDF Weighted)**: Averages word vectors, but multiplies them by TF-IDF scores so rare/important words dominate the sentence meaning vector.
3. **TF-IDF**: Pure sparse bag-of-words vectorization based on frequency.
4. **DistilBERT**: Uses Huggingface transformers to extract context-aware token embeddings, applying an attention mask and mean-pooling them into a dense 768-dimensional space.
5. **SBERT**: Uses the `all-MiniLM-L6-v2` Sentence-Transformer model, fine-tuned specifically for generating document-level representation.
6. **Fine-Tuned Siamese DistilBERT**: Custom iteration of DistilBERT fine-tuned dynamically on the training data. It uses a Siamese Dual-Encoder alongside PyTorch's `MarginRankingLoss`. It learns that $Score(Sentence_A) > Score(Sentence_B)$ if $A$ precedes $B$, thus permanently infusing structural rank into the raw `[CLS]` token numerical space.

### Step 3: Structural Stream 
*(Script: `structural_stream.py`)*

The Structural Stream doesn't just read the sentence; it analyzes the neighborhood. It uses graph logic to map sentence placement. 

For every document, the pipeline constructs four distinct Adjacency Graphs:
1. **Local Graph**: Connects immediate neighbors (Sentence 1 $\leftrightarrow$ Sentence 2).
2. **Midrange Graph**: Connects localized windows of sentences (e.g., Sentence 1 connects to 2, 3, and 4).
3. **Global Graph**: Uses cosine similarity to connect sentences that randomly sound alike, regardless of position.
4. **Entity Graph**: Strips sentences down to Named Entities/Long Words and connects sentences that share those specific entities (approximate Jaccard Overlap).

**Graph Convolutional Network (GCN)**: These four graphs are mathematically merged into a singular Adjacency Matrix ($A$). The matrix, alongside TF-IDF initial node features ($X$), is pushed through a 2-layer Graph Convolutional Network (GCN). The GCN iterates and outputs structural embeddings—dense vectors defining how a sentence relates to the document's broader web.

### Step 4: Gated MLP Fusion 
*(Script: `fusion.py`)*

Some documents are highly semantic (story-based); others are structural (list-based). The pipeline blends the Semantic Stream and Structural Stream dynamically:
- Vectors from both streams are projected and concatenated.
- They are passed through a Multi-Layer Perceptron (MLP) mapping layer which outputs a sigmoid gate threshold (`gate = \sigma(MLP(concat(sem, struct)))`).
- The final fused embedding is a weighted sum: `fused_emb = (gate * semantic_emb) + ((1 - gate) * structural_emb)`.
- The gate is trained via a random-search perturbation optimization to minimize loss across the training embeddings.

### Step 5: Pairwise Scoring & Decoding 
*(Scripts: `decoding.py` and `metrics.py`)*

Instead of predicting the absolute position of a sentence (which is highly unstable), the pipeline forms a tournament of $N^2$ matchups. 
1. **Feature Construction**: For every possible sentence pair $(i, j)$ in a document, the pipeline builds a feature vector comparing their fused embeddings: `concatenate( i, j, |i-j|, i*j )`.
2. **MLP Scorer**: A Deep Neural Network (MLPClassifier with layers 256, 128) trains on these features to output a binary probability: *Does sentence $i$ belong before sentence $j$?*
3. **Tournament Decoding**: For test documents, the MLP predicts the probability for every pair matchup. This builds a dense probability matrix. The pipeline sums the row probabilities for each sentence, effectively creating a "Tournament Ranking". Sort by the highest sum, and you get the predicted chronological order!

### Step 6: Prediction Analysis & Output 
*(Script: `prediction_analyzer.py`)*

The pipeline writes a comprehensive breakdown to `test_predictions.md`.
For sample documents, it compares the Predicted Sentence text side-by-side with the Actual Sentence text.
- It calculates **Kendall's Tau Score** (measuring chronological rank inversion).
- It performs Qualitative Checks, tagging if the model correctly identified the Introduction sentence, the Conclusion sentence, and flagging any localized neighbor swaps.

---

## Execution

To initiate the pipeline, generate data splits, train the embeddings, configure the GCN graphs, compute fusion, and output the analysis:

```bash
python pipeline/main.py
```

### Interactive Mode (New)
After running the pipeline at least once, the trained weights are saved to disk. You can then use the interactive tool to test custom sentence sets instantly:

```bash
python interactive.py
```
This utility loads the saved `.pt` and `.pkl` models and provides a real-time prompt to enter shuffled sentences and see the predicted order across all 6 embedding methods.

The terminal log will output step-by-step progress, graph construction metrics, and final scoring accuracies. Granular qualitative insights per document are saved into `test_predictions.md`.
