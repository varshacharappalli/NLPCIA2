"""
dataset_generator.py
Loads real abstracts from the ACL Anthology Network (AAN) dataset (abstract.csv).
Filters and segments results into a format suitable for sentence ordering.
"""

import csv
import os
import random
import sys

# Add parent directory to sys.path to allow imports from pipeline relative path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import run_preprocessor

def load_aan_dataset(csv_path='abstract.csv', min_sentences=3, max_sentences=6, limit=None, seed=42):
    """
    Load abstracts from abstract.csv and process them for sentence ordering.
    """
    if not os.path.exists(csv_path):
        # Try finding it in root if called within pipeline/
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'abstract.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Could not find {csv_path}")

    random.seed(seed)
    docs = []
    
    print(f"  [Loader] Reading {csv_path}...")
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            abstract_text = row.get('Abstract', '').strip()
            if not abstract_text:
                continue
            
            # Use the existing preprocessor (Python fallback or C binary)
            # This handles segmentation into sentences
            # We use --no-lowercase and --no-punct during initial loading to keep raw sentences for now
            # and let the pipeline's preprocessing steps handle formatting later.
            try:
                processed = run_preprocessor(abstract_text, flags=['--no-lowercase', '--no-punct'])
                sentences = processed.get('sentences', [])
            except Exception as e:
                # Fallback to simple split if preprocessor fails on specific text
                sentences = [s.strip() for s in abstract_text.split('.') if s.strip()]

            # Filtering for reasonable document lengths for the ordering task
            if len(sentences) >= min_sentences:
                # If too long, we take a window to keep it manageable
                if len(sentences) > max_sentences:
                    start_idx = random.randint(0, len(sentences) - max_sentences)
                    sentences = sentences[start_idx : start_idx + max_sentences]
                
                docs.append({
                    'id':        i,
                    'topic':     'ACL-Anthology',
                    'sentences': sentences, # Original (canonical) order
                })

            if limit and len(docs) >= limit:
                print(f"  [Loader] Reached limit of {limit} documents.")
                break

            if i % 5000 == 0 and i > 0:
                print(f"    Processed {i} rows...")

    print(f"  [Loader] Successfully loaded {len(docs)} documents.")
    return docs

def generate_dataset(seed=42, limit=None):
    """Bridge for existing pipeline calls - redirects to load_aan_dataset."""
    return load_aan_dataset(seed=seed, limit=limit)

if __name__ == '__main__':
    # Test loader
    docs = load_aan_dataset()
    if docs:
        print(f"Sample Document (ID {docs[0]['id']}):")
        for j, s in enumerate(docs[0]['sentences']):
            print(f"  [{j}] {s}")
