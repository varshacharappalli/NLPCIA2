# NLP CIA2 — Full Study Notes (End to End)

---

## The Big Picture

**Task:** Given a shuffled set of sentences from a research abstract, predict their correct order.

**Approach:** Compare 6 different ways of turning sentences into vectors (encoders), then pass each through the same pipeline: GCN → Gated Fusion → MLP Decoder → Tournament Ranking → Accuracy.

**Dataset:** 500 ACL research paper abstracts. 400 train, 100 test.

---

## What the pipeline does, in one line each:

1. **Encode** — convert sentences to vectors (6 different methods)
2. **GCN** — build graphs on the doc, extract structural vectors
3. **Gated Fusion** — blend semantic + structural into one fused vector
4. **MLP** — predict which sentence comes first for every pair
5. **Tournament** — rank sentences by total wins
6. **Accuracy** — compare predicted order to actual order

Steps 2-6 are **identical** for all 6 encoders. The accuracy difference comes purely from step 1.

---

## Step 1 — The 6 Encoders

### Encoder 1: TF-IDF

Each sentence becomes a vector of 5000 numbers. Each number = how important a word is in this sentence vs the whole corpus.

- Common words like "the" → low weight
- Rare technical terms → high weight
- Top 5000 terms kept, everything else ignored

**Why it fails for ordering (48%):**
Every sentence in a research abstract uses similar technical vocabulary. "model", "accuracy", "proposed" appear in sentence 1 AND sentence 5. Two sentences from completely different positions end up with nearly identical vectors — no ordering signal.

---

### Encoder 2: Word2Vec

Each word gets a 100-dim vector learned from co-occurrence in a sliding window of 5.

Words that appear near each other end up close in vector space:
- "neural" and "network" → close
- "paper" and "study" → close
- "neural" and "banana" → far

**Sentence vector = mean of all word vectors.**

Example:
```
"we propose a neural network"
→ grab vector for each word
→ average them
→ one 100-dim vector
```

**Why it fails (52%):**
- Averaging destroys word order — "model fails" and "fails model" → same vector
- Antonyms like "works" and "fails" appear in identical contexts → end up close in vector space
- No positional signal

---

### Encoder 3: SBERT (all-MiniLM-L6-v2)

Pretrained model specifically trained to produce meaningful sentence vectors.

**Internals:**

Input: "Our model fails here"

Step 1 — Tokenize:
```
[CLS] our model fails here [SEP]
```

Step 2 — Each token starts as a lookup vector

Step 3 — 6 transformer layers with attention:
Every token looks at every other token. After layer 6, "fails" has seen "model" and "here" — its vector encodes full sentence context. "fails" in "the model fails" gets a different vector than "fails" in "the system never fails."

Step 4 — Mean pool all token vectors → one 384-dim sentence vector

**Why better than Word2Vec:**
Same word = different vector depending on context. Word2Vec always gives "fails" the same vector.

**Why it still scores only 64%:**
SBERT was trained on semantic similarity, not ordering. In a research abstract, intro and conclusion talk about the same topic → similar vectors → model can't separate them by position.

---

### Encoder 4: Word2Vec + TF-IDF Weighted

Same as Word2Vec but each word's vector is weighted by its TF-IDF score instead of averaging equally.

**Why it's worse than plain Word2Vec (47% < 52%):**
TF-IDF upweights rare technical terms like "transformer", "attention", "BLEU". These appear throughout the abstract with no positional pattern. So you're giving more weight to exactly the words with no ordering signal, and downweighting common words like "we", "our", "this" which might actually hint at position ("we propose" = intro, "this shows" = conclusion).

---

### Encoder 5: Raw DistilBERT (mean pool)

Same transformer architecture as SBERT. DistilBERT = smaller, faster BERT (6 layers, 768-dim output). Pretrained on masked language modeling.

**Why mean pool and not [CLS]:**
[CLS] is only meaningful if the model was trained on a classification task. DistilBERT base was trained on masked LM — its [CLS] is not task-specific. Mean pooling over real tokens is more robust.

**Why it scores 67%, better than SBERT (64%):**
DistilBERT outputs 768-dim vectors vs SBERT's 384-dim. Richer representation gives the MLP more signal.

**Same core limitation:** no ordering-specific training.

---

### Encoder 6: Fine-Tuned DistilBERT (Siamese)

Same base model as encoder 5, but trained directly on the ordering task.

**Training data construction:**
For every pair of sentences (i, j) from 400 training docs:
- i comes before j → label = +1
- j comes before i → label = -1

**Siamese network:**
Two sentences go through the same DistilBERT simultaneously (shared weights). Each produces a [CLS] token → linear layer → single scalar "earliness score".

```
score_A, score_B = model(sentence_A, sentence_B)
```

**Loss (MarginRankingLoss):**
```
loss = max(0, -y * (score_A - score_B) + margin)
```
If A should come before B (y=+1): penalize when score_A < score_B + 1. Enforces a gap, not just correct ordering.

**Training:** AdamW lr=2e-5, 3 epochs, batch_size=16.

**Why [CLS] here but not in encoder 5:**
The entire model is fine-tuned to make [CLS] = "how early does this sentence appear." [CLS] is now position-aware — exactly what we need.

**Result: ~80%** — best of all 6 because it's the only encoder that directly learned ordering from data.

---

## Step 2 — GCN (Structural Encoding)

### Why graphs?

The encoder tells us what each sentence means. But sentences also have structural relationships — who's a neighbor, who shares topics, who mentions the same entities. Graphs capture these.

### Graphs 101

Nodes = sentences. Edges = relationships.

For 4 sentences, edges are stored in an adjacency matrix A (n×n):
```
     S1  S2  S3  S4
S1 [  0   1   0   0 ]
S2 [  1   0   1   0 ]
S3 [  0   1   0   1 ]
S4 [  0   0   1   0 ]
```
1 = edge exists, 0 = no edge.

---

### The 4 Graphs

**Local graph (weight 0.4)** — connects only immediate neighbors:
```
S1 — S2 — S3 — S4
```
S1 knows S2. S2 knows S1 and S3. Highest weight because adjacent sentences are most strongly related.

**Midrange graph (weight 0.25)** — connects within window of 3:
```
S1 connects to S2, S3, S4
S2 connects to S1, S3, S4
S3 connects to S1, S2, S4
```
Wider neighborhood — captures paragraph-level coherence.

**Global graph (weight 0.2)** — connects topically similar sentences:
Uses TF-IDF cosine similarity. If sim(Si, Sj) > 0.3 → draw edge with that similarity as weight.

Example: S1 = "We propose a neural model" and S4 = "Our neural model achieves 95%" → high cosine similarity → edge drawn even though far apart.

**Entity graph (weight 0.15)** — connects sentences sharing named entities:
Entities = capitalized words OR words longer than 4 chars (rough proxy).

Example:
- S1 = "The **Transformer** model was proposed" → entities = {transformer, model, proposed}
- S3 = "**Transformer** achieves state of the art" → entities = {transformer, achieves, state}
- Jaccard = shared/total = 1/5 = 0.2 → edge weight 0.2

---

### Merging the 4 Graphs

Each graph is a matrix. Merge = multiply each by its weight and add element by element.

Example with 3 sentences, local + midrange only:

**local:**
```
 0   1   0
 1   0   1
 0   1   0
```

**midrange:**
```
 0   1   1
 1   0   1
 1   1   0
```

**merged = 0.4 * local + 0.25 * midrange:**
```
S1-S2 = 0.4*1 + 0.25*1 = 0.65
S1-S3 = 0.4*0 + 0.25*1 = 0.25
S2-S3 = 0.4*1 + 0.25*1 = 0.65
```

**final A:**
```
 0     0.65  0.25
 0.65  0     0.65
 0.25  0.65  0
```

S1-S2 strong (0.65), S1-S3 weaker (0.25) because they're not direct neighbors.

The 0.4, 0.25, 0.2, 0.15 weights are hardcoded design choices — no mathematical derivation.

---

### GCN — What it does with the graph

Each sentence starts with its TF-IDF vector as node features: shape (n × 5000).

Each GCN layer updates each sentence's vector by mixing in its neighbors' vectors, weighted by the graph edges.

**Example:** S2 is connected to S1, S3, S4. After one GCN layer, S2's new vector = weighted average of S1, S2, S3, S4's vectors. S2 now "knows" what its neighbors look like.

**Math:**
```
H = activation( D^(-0.5) A_hat D^(-0.5) X W )
```
- `A_hat = A + I` — add self loops (each node keeps its own info)
- `D^(-0.5) A_hat D^(-0.5)` — normalize so high-degree nodes don't dominate
- `X W` — linear projection

**Two layers:**
- Layer 1 (relu): each sentence sees direct neighbors → (n, 5000) → (n, 128)
- Layer 2 (tanh): each sentence sees neighbors-of-neighbors → (n, 128) → (n, 64)

**Output:** 64-dim structural vector per sentence.

**Key point:** GCN weights are random and never updated. No backprop. The structural signal comes entirely from the graph topology — who is connected to whom.

---

## Step 3 — Gated Fusion

### What we have

Each sentence now has two vectors:
- Semantic vector — from encoder (captures meaning)
- Structural vector — 64-dim from GCN (captures graph neighborhood)

### The Problem

Simple average assumes both are equally useful for every dimension. Bad idea — some semantic dimensions might be more informative, others structural.

### The Gate

Gate = a vector of values between 0 and 1, one per output dimension.

```
fused = gate * semantic + (1 - gate) * structural
```

- gate = 1 → use semantic completely for that dimension
- gate = 0 → use structural completely
- gate = 0.6 → 60% semantic, 40% structural

### Example

Sentence S1 after encoding + GCN (3-dim for simplicity):

```
semantic   = [0.8, 0.2, 0.5]
structural = [0.1, 0.9, 0.3]
gate       = [0.7, 0.2, 0.9]
```

Gate meaning:
- dim 1 → trust semantic 70%, structural 30%
- dim 2 → trust semantic 20%, structural 80%
- dim 3 → trust semantic 90%, structural 10%

Apply:
```
dim1 = 0.7*0.8 + 0.3*0.1 = 0.56 + 0.03 = 0.59
dim2 = 0.2*0.2 + 0.8*0.9 = 0.04 + 0.72 = 0.76
dim3 = 0.9*0.5 + 0.1*0.3 = 0.45 + 0.03 = 0.48

fused = [0.59, 0.76, 0.48]
```

### Where does the gate come from?

```
combined = concat(semantic, structural) = [0.8, 0.2, 0.5, 0.1, 0.9, 0.3]  ← 6-dim
gate = sigmoid(combined @ W_gate + b_gate)
```

W_gate is a (6×3) matrix. Matrix multiply → 3-dim → sigmoid squishes to (0,1) → gate values.

### How is W_gate trained?

Random perturbation search (not gradient descent):
1. W_gate starts as small random numbers
2. Add small random noise to W_gate
3. Compute fused output, measure loss vs target
4. If loss improved → keep new W_gate
5. If loss got worse → revert
6. Repeat 30 times

**Output:** one fused vector per sentence that captures both meaning and structural position.

---

## Step 4 — MLP Pairwise Scorer

### Building the input feature

For every pair (i, j), concatenate 4 things:

```
[fused_i | fused_j | |fused_i - fused_j| | fused_i * fused_j]
```

Note: `|i - j|` and `i * j` are element-wise operations, not matrix multiply.

Example with 3-dim vectors:
```
fused_i    = [0.59, 0.76, 0.48]
fused_j    = [0.21, 0.44, 0.83]

difference = |i - j| = [0.38, 0.32, 0.35]
product    = i * j   = [0.59*0.21, 0.76*0.44, 0.48*0.83]
                     = [0.12,      0.33,       0.40]

final input = [0.59, 0.76, 0.48, 0.21, 0.44, 0.83, 0.38, 0.32, 0.35, 0.12, 0.33, 0.40]
```

12 numbers total (4 × 3-dim).

**Why these 4 components:**
- `fused_i`, `fused_j` — what each sentence looks like individually. Needed because two pairs could have the same difference vector but be completely different sentences.
- `|i - j|` — how different are they
- `i * j` — shared dimensions / interaction

### The MLP

```
12-dim input
    ↓
layer 1: 256 neurons (relu)
    ↓
layer 2: 128 neurons (relu)
    ↓
output: 1 number between 0 and 1
```

Output = P(sentence i comes before sentence j).

### Training (supervised)

Note: the encoding/fitting was unsupervised. The MLP is supervised.

For every pair (i, j) from 400 training docs:
- i actually comes before j → label = 1
- j actually comes before i → label = 0

MLP learns from thousands of labeled pairs.

### Two separate predictions per pair

P(S2 before S4) and P(S4 before S2) are two separate MLP runs with different inputs:

```
run 1: input = [fused_S2 | fused_S4 | |S2-S4| | S2*S4] → 0.6
run 2: input = [fused_S4 | fused_S2 | |S4-S2| | S4*S2] → 0.7
```

Order of concatenation is different → different input → different output. They don't have to sum to 1.

---

## Step 5 — Tournament Decoding

Build a score matrix. Row i, col j = P(i before j):

```
      S1    S2    S3    S4
S1  [  -   0.8   0.7   0.6 ]
S2  [ 0.2   -    0.4   0.3 ]
S3  [ 0.3  0.6    -    0.6 ]
S4  [ 0.4  0.7   0.4    -  ]
```

Sum each row (total score across all matchups):
```
S1 = 0.8 + 0.7 + 0.6 = 2.1
S2 = 0.2 + 0.4 + 0.3 = 0.9
S3 = 0.3 + 0.6 + 0.6 = 1.5
S4 = 0.4 + 0.7 + 0.4 = 1.5
```

Sort descending → predicted order: **S1 → S3 → S4 → S2**

Think of it as a round-robin tournament. Whoever beats the most others ranks highest.

---

## Step 6 — Accuracy

Compare predicted order vs actual order. Check every pair — did we get the relative order right?

Example:
```
actual    = [S1, S2, S3, S4]
predicted = [S1, S3, S4, S2]
```

```
S1 before S2 → correct ✓
S1 before S3 → correct ✓
S1 before S4 → correct ✓
S2 before S3 → wrong ✗
S2 before S4 → wrong ✗
S3 before S4 → correct ✓
```

4/6 correct → pairwise accuracy = 66.7%

Do this across all 100 test docs, average → final number.

---

## Final Results

| Encoder | Pairwise Accuracy |
|---|:---:|
| Fine-Tuned DistilBERT [CLS] | **80.59%** |
| Raw DistilBERT (mean pool) | 71.91% |
| SBERT (all-MiniLM-L6-v2) | 64.68% |
| Word2Vec (mean pool) | 64.15% |
| TF-IDF (sparse vector) | 62.87% |
| Word2Vec + TF-IDF weighted | 62.55% |

---

## Key Takeaways

**1. Fine-tuning wins** — explicitly training on ordering makes [CLS] position-aware. +8 points over raw DistilBERT.

**2. Contextual > static** — all transformer methods (64-80%) beat bag-of-words (62-64%). Same word, different context → different vector.

**3. TF-IDF weighting hurts Word2Vec** — upweights rare technical terms that appear everywhere in the abstract with no positional pattern.

**4. Graph structure alone can't save weak encoders** — all 6 use identical GCN + fusion + MLP. The spread in results is purely from the encoder quality.

**5. Encoding is unsupervised, MLP is supervised** — encoders fit on text without labels. MLP trained on ground truth sentence order from training docs.
