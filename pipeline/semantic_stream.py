"""
semantic_stream.py
Three sentence encoders (TF-IDF, Word2Vec, BERT) and pairwise classification
for semantic sentence ordering.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
import warnings

warnings.filterwarnings('ignore')


def get_sentence_corpus(docs):
    """Extract all sentences from a list of docs."""
    corpus = []
    for doc in docs:
        corpus.extend(doc['sentences'])
    return corpus


# ── TF-IDF Sentence Embeddings ────────────────────────────────────────────────

class TFIDFEncoder:
    """TF-IDF vectorizer wrapped as a sentence encoder."""

    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        self.vec = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range
        )
        self.fitted = False

    def fit(self, corpus):
        """Fit TF-IDF vectorizer on corpus (list of strings)."""
        self.vec.fit(corpus)
        self.fitted = True

    def encode(self, sentences):
        """Returns (n_sents, d) numpy array of TF-IDF sentence embeddings."""
        return self.vec.transform(sentences).toarray()

    def encode_doc(self, doc):
        """Encode all sentences in a document dict."""
        return self.encode(doc['sentences'])


# ── Word2Vec Sentence Embeddings ──────────────────────────────────────────────

class Word2VecEncoder:
    """Mean-pooled Word2Vec sentence encoder."""

    def __init__(self, vector_size=100, window=5, min_count=1, epochs=10):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model = None

    def fit(self, corpus):
        """Train Word2Vec on tokenized corpus."""
        try:
            from gensim.models import Word2Vec
            tokenized = [sent.lower().split() for sent in corpus]
            self.model = Word2Vec(
                sentences=tokenized,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                epochs=self.epochs,
                workers=1,
                seed=42,
            )
        except ImportError:
            print("  [Warning] gensim not found. Using random Word2Vec embeddings.")
            self.model = None

    def _sentence_vector(self, sentence):
        """Mean-pool word vectors for a sentence."""
        if self.model is None:
            rng = np.random.default_rng(abs(hash(sentence)) % (2 ** 31))
            return rng.standard_normal(self.vector_size).astype(np.float32)
        words = sentence.lower().split()
        vecs = []
        for w in words:
            if w in self.model.wv:
                vecs.append(self.model.wv[w])
        if not vecs:
            return np.zeros(self.vector_size, dtype=np.float32)
        return np.mean(vecs, axis=0)

    def encode(self, sentences):
        """Returns (n_sents, vector_size) numpy array."""
        return np.array([self._sentence_vector(s) for s in sentences])

    def encode_doc(self, doc):
        """Encode all sentences in a document dict."""
        return self.encode(doc['sentences'])


class Word2VecTFIDFEncoder:
    """TF-IDF weighted pooling Word2Vec sentence encoder."""
    
    def __init__(self, w2v_encoder, tfidf_encoder):
        self.w2v = w2v_encoder.model
        self.vector_size = w2v_encoder.vector_size
        self.tfidf = tfidf_encoder.vec
        if self.tfidf is not None and hasattr(self.tfidf, 'vocabulary_'):
            self.vocab = self.tfidf.vocabulary_
        else:
            self.vocab = {}

    def fit(self, corpus):
        pass # Relies on pre-fitted encoders given in __init__

    def _sentence_vector(self, sentence):
        import scipy.sparse
        if self.w2v is None:
            rng = np.random.default_rng(abs(hash(sentence)) % (2 ** 31))
            return rng.standard_normal(self.vector_size).astype(np.float32)
            
        words = sentence.lower().split()
        tfidf_vec = self.tfidf.transform([sentence])
        
        vecs = []
        weights = []
        
        for w in words:
            if w in self.w2v.wv:
                vecs.append(self.w2v.wv[w])
                weight = 1.0
                if w in self.vocab:
                    col_idx = self.vocab[w]
                    val = tfidf_vec[0, col_idx]
                    if val > 0:
                        weight = val
                weights.append(weight)
                
        if not vecs:
            return np.zeros(self.vector_size, dtype=np.float32)
            
        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(weights)) / len(weights)
            
        return np.average(vecs, axis=0, weights=weights)

    def encode(self, sentences):
        return np.array([self._sentence_vector(s) for s in sentences])

    def encode_doc(self, doc):
        return self.encode(doc['sentences'])


# ── BERT Sentence Embeddings ──────────────────────────────────────────────────

class RawTransformerEncoder:
    """Uses huggingface transformers (DistilBERT by default) to generate contextual token embeddings and mean-pools them."""
    
    def __init__(self, model_name='distilbert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def fit(self, corpus=None):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            print(f"  Loading Transformer model: {self.model_name} ...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            print(f"  Transformer model {self.model_name} loaded.")
        except ImportError:
            print("  [Warning] transformers or torch not found.")
            self.model = None

    def encode(self, sentences):
        import torch
        if self.model is None:
            rng = np.random.default_rng(42)
            # Default to 768 dims, but try to use model config if available
            dim = getattr(self.model.config, 'hidden_size', 768) if self.model else 768
            return rng.standard_normal((len(sentences), dim))
            
        inputs = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask, 1)
        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        return mean_pooled.numpy()

    def encode_doc(self, doc):
        return self.encode(doc['sentences'])

class BERTEncoder:
    """Sentence-Transformers BERT encoder with TF-IDF fallback."""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self._fallback_vec = None

    def fit(self, corpus=None):
        """Load pretrained sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"  Loading BERT model: {self.model_name} ...")
            self.model = SentenceTransformer(self.model_name)
            print("  BERT model loaded.")
        except ImportError:
            print("  [Warning] sentence-transformers not found. Using TF-IDF fallback for BERT.")
            self.model = None
            if corpus:
                self._fallback_vec = TfidfVectorizer(max_features=1000)
                self._fallback_vec.fit(corpus)

    def encode(self, sentences):
        """Returns (n_sents, embed_dim) numpy array."""
        if self.model is not None:
            return self.model.encode(sentences, show_progress_bar=False)
        # Fallback: TF-IDF + random projection to 384 dims
        if self._fallback_vec is not None:
            X = self._fallback_vec.transform(sentences).toarray()
        else:
            vec = TfidfVectorizer(max_features=1000)
            X = vec.fit_transform(sentences).toarray()
        rng = np.random.default_rng(42)
        proj = rng.standard_normal((X.shape[1], 384)) / np.sqrt(384)
        return X @ proj

    def encode_doc(self, doc):
        """Encode all sentences in a document dict."""
        return self.encode(doc['sentences'])


# ── Fine-Tuned DistilBERT Embeddings ──────────────────────────────────────────

class SiamesePairwiseDataset:
    def __init__(self, encodings_a, encodings_b, labels):
        self.encodings_a = encodings_a
        self.encodings_b = encodings_b
        self.labels = labels

    def __getitem__(self, idx):
        import torch
        item = {}
        for key, val in self.encodings_a.items():
            item[f'{key}_a'] = torch.tensor(val[idx])
        for key, val in self.encodings_b.items():
            item[f'{key}_b'] = torch.tensor(val[idx])
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)


class FineTunedDistilBERTEncoder:
    """Fine-tunes DistilBERT using a Siamese network with MarginRankingLoss, extracting high-quality [CLS] embeddings."""
    
    def __init__(self, model_name='distilbert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def fit(self, train_docs):
        try:
            import torch
            import torch.nn as nn
            from transformers import AutoTokenizer, DistilBertModel
            from torch.utils.data import DataLoader
            from tqdm import tqdm
        except ImportError:
            print("  [Warning] transformers or torch not found. Skipping fine-tuning.")
            return

        print(f"  Preparing Siamese pairwise dataset for {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        texts_a, texts_b, labels = [], [], []
        for doc in train_docs:
            sents = doc['sentences']
            n = len(sents)
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    texts_a.append(sents[i])
                    texts_b.append(sents[j])
                    # MarginRankingLoss: y=1 means score(A) > score(B). We want scores to be higher for earlier sentences.
                    labels.append(1.0 if i < j else -1.0)
                    
        encodings_a = self.tokenizer(texts_a, truncation=True, padding=True, max_length=128)
        encodings_b = self.tokenizer(texts_b, truncation=True, padding=True, max_length=128)
        
        dataset = SiamesePairwiseDataset(encodings_a, encodings_b, labels)
        
        # Build Siamese model
        class PairwiseRankingModel(nn.Module):
            def __init__(self, model_name):
                super().__init__()
                self.distilbert = DistilBertModel.from_pretrained(model_name)
                self.scorer = nn.Linear(self.distilbert.config.hidden_size, 1)
                
            def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
                out_a = self.distilbert(input_ids=input_ids_a, attention_mask=attention_mask_a)
                val_a = self.scorer(out_a.last_hidden_state[:, 0, :]).squeeze(-1)
                
                out_b = self.distilbert(input_ids=input_ids_b, attention_mask=attention_mask_b)
                val_b = self.scorer(out_b.last_hidden_state[:, 0, :]).squeeze(-1)
                
                return val_a, val_b
                
        self.model = PairwiseRankingModel(self.model_name)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        # We enforce score(A) > score(B) + margin if A is before B.
        criterion = nn.MarginRankingLoss(margin=1.0)
        
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        print(f"  Fine-tuning {self.model_name} (Siamese Margin Ranking)...")
        self.model.train()
        
        # Training for 3 epochs for better Siamese embeddings
        num_epochs = 3
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch in tqdm(dataloader, desc=f"Training Siamese Network (Epoch {epoch+1}/{num_epochs})"):
                optimizer.zero_grad()
                ida = batch['input_ids_a'].to(device)
                maska = batch['attention_mask_a'].to(device)
                idb = batch['input_ids_b'].to(device)
                maskb = batch['attention_mask_b'].to(device)
                y = batch['labels'].to(device)
                
                score_a, score_b = self.model(ida, maska, idb, maskb)
                loss = criterion(score_a, score_b, y)
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            print(f"  Epoch {epoch+1}/{num_epochs} complete. Avg Loss: {epoch_loss/len(dataloader):.4f}")

        
    def evaluate_pairwise_accuracy(self, test_docs):
        if self.model is None or self.tokenizer is None:
            return 0.5
            
        import torch
        from torch.utils.data import DataLoader
        
        texts_a, texts_b, labels = [], [], []
        # For evaluation, we only care about positive pairs (i < j) to see if score(A) > score(B)
        for doc in test_docs:
            sents = doc['sentences']
            n = len(sents)
            for i in range(n):
                for j in range(i+1, n):
                    texts_a.append(sents[i])
                    texts_b.append(sents[j])
                    labels.append(1.0)
        
        if not labels:
            return 0.5
            
        encodings_a = self.tokenizer(texts_a, truncation=True, padding=True, max_length=128)
        encodings_b = self.tokenizer(texts_b, truncation=True, padding=True, max_length=128)
        dataset = SiamesePairwiseDataset(encodings_a, encodings_b, labels)
        dataloader = DataLoader(dataset, batch_size=32)
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.eval()
        
        correct = 0
        total = 0
        
        print("  Evaluating Siamese scalar scorer on test set...")
        with torch.no_grad():
            for batch in dataloader:
                ida = batch['input_ids_a'].to(device)
                maska = batch['attention_mask_a'].to(device)
                idb = batch['input_ids_b'].to(device)
                maskb = batch['attention_mask_b'].to(device)
                
                score_a, score_b = self.model(ida, maska, idb, maskb)
                
                # We expect score_a > score_b because all targets are i < j
                preds = (score_a > score_b)
                correct += preds.sum().item()
                total += preds.size(0)
                
        acc = correct / total
        print(f"  Siamese Scorer Pairwise Accuracy: {acc:.4f}")
        return acc

    def encode(self, sentences):
        import torch
        if self.model is None:
            rng = np.random.default_rng(42)
            dim = 768
            return rng.standard_normal((len(sentences), dim))
            
        inputs = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=128)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        self.model.eval()
        with torch.no_grad():
            # Extract basic structure
            outputs = self.model.distilbert(**inputs)
            
        # Extract [CLS] token (first token)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return cls_embeddings.cpu().numpy()

    def encode_doc(self, doc):
        return self.encode(doc['sentences'])

    def save(self, model_path="ft_distilbert.pt"):
        import torch
        if self.model is not None:
            torch.save(self.model.state_dict(), model_path)
            print(f"  Saved Siamese model weights to {model_path}")

    def load(self, model_path="ft_distilbert.pt"):
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer, DistilBertModel
        import os
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        class PairwiseRankingModel(nn.Module):
            def __init__(self, model_name):
                super().__init__()
                self.distilbert = DistilBertModel.from_pretrained(model_name)
                self.scorer = nn.Linear(self.distilbert.config.hidden_size, 1)
                
            def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
                out_a = self.distilbert(input_ids=input_ids_a, attention_mask=attention_mask_a)
                val_a = self.scorer(out_a.last_hidden_state[:, 0, :]).squeeze(-1)
                
                out_b = self.distilbert(input_ids=input_ids_b, attention_mask=attention_mask_b)
                val_b = self.scorer(out_b.last_hidden_state[:, 0, :]).squeeze(-1)
                
                return val_a, val_b

        self.model = PairwiseRankingModel(self.model_name)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print(f"  Loaded fine-tuned weights from {model_path}")
        else:
            print(f"  [Warning] Could not find {model_path}")
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        self.model.eval()
