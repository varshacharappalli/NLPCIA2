"""
data_loader.py
Compile C preprocessor, run it on documents, parse output, and run ablation study.
"""

import subprocess
import os
import sys
import platform
import random
import numpy as np
from itertools import combinations

BINARY_NAME = 'preprocess.exe' if platform.system() == 'Windows' else 'preprocess'
BINARY_PATH = os.path.join(os.path.dirname(__file__), '..', 'preprocessing', BINARY_NAME)

# Global flag: True = use compiled C binary, False = Python fallback
_USE_C_BINARY = False


def _find_gcc():
    """Search for gcc in PATH and common Windows locations."""
    import shutil
    if shutil.which('gcc'):
        return 'gcc'
    # Common Windows locations (MinGW, MSYS2, Scoop, Chocolatey, conda)
    candidates = [
        r'C:\msys64\mingw64\bin\gcc.exe',
        r'C:\msys64\usr\bin\gcc.exe',
        r'C:\tools\mingw64\bin\gcc.exe',
        r'C:\mingw64\bin\gcc.exe',
        r'C:\mingw\bin\gcc.exe',
        r'C:\ProgramData\chocolatey\lib\mingw\tools\install\mingw64\bin\gcc.exe',
        r'C:\Scoop\apps\mingw\current\bin\gcc.exe',
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def compile_c():
    """Compile the C preprocessor. Falls back to Python implementation if gcc not found."""
    global _USE_C_BINARY
    preprocess_dir = os.path.join(os.path.dirname(__file__), '..', 'preprocessing')
    src = os.path.join(preprocess_dir, 'preprocess.c')
    out = os.path.join(preprocess_dir, BINARY_NAME)

    if os.path.exists(out):
        _USE_C_BINARY = True
        print("C preprocessor binary found, using compiled C.")
        return out

    gcc = _find_gcc()
    if gcc is None:
        print("  [INFO] gcc not found. C source exists at preprocessing/preprocess.c")
        print("         Using Python reimplementation of the same preprocessing logic.")
        print("         To use the actual C binary: install MinGW (choco install mingw)")
        print("         or Scoop (scoop install gcc), then re-run.")
        _USE_C_BINARY = False
        return None

    print(f"Compiling C preprocessor with: {gcc}")
    result = subprocess.run(
        [gcc, '-Wall', '-O2', '-o', out, src],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("GCC compilation error:", result.stderr)
        print("  Falling back to Python reimplementation.")
        _USE_C_BINARY = False
        return None

    print("Compiled successfully.")
    _USE_C_BINARY = True
    return out


# ── Python reimplementation of preprocess.c logic ────────────────────────────
# (same algorithm as C, used as fallback when gcc is unavailable)

_ABBREVS = {
    'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr',
    'vs', 'etc', 'inc', 'ltd', 'corp', 'fig', 'no',
    'st', 'ave', 'dept', 'est', 'approx', 'govt'
}


def _py_segment(text):
    """Rule-based sentence segmentation (mirrors C logic)."""
    sentences = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in '.!?':
            # Look ahead for whitespace + uppercase
            j = i + 1
            while j < n and text[j] == ' ':
                j += 1
            if j >= n or text[j].isupper() or text[j] in ('"', "'"):
                # Check abbreviation: find word before punctuation
                word_end = i
                word_start = i - 1
                while word_start > start and text[word_start].isalpha():
                    word_start -= 1
                if not text[word_start].isalpha():
                    word_start += 1
                word = text[word_start:word_end].lower()
                if word in _ABBREVS:
                    i += 1
                    continue
                # Check decimal number
                if (i > 0 and text[i - 1].isdigit() and
                        j < n and text[j].isdigit()):
                    i += 1
                    continue
                # Valid boundary
                sent = text[start:i + 1].strip()
                if sent:
                    sentences.append(sent)
                start = j
                i = j
                continue
        i += 1
    # Trailing sentence
    sent = text[start:].strip()
    if sent:
        sentences.append(sent)
    return sentences


def _py_lowercase(s):
    return s.lower()


def _py_remove_punct(s):
    out = []
    for ch in s:
        if ch in ("'", '-'):
            out.append(ch)
        elif ch.isalnum() or ch.isspace():
            out.append(ch)
        else:
            if out and out[-1] != ' ':
                out.append(' ')
    return ''.join(out)


def _py_tokenize(s):
    return s.split()


def _python_preprocess(text, flags=None):
    """Pure-Python preprocessing (same logic as preprocess.c)."""
    if flags is None:
        flags = []
    do_lowercase = '--no-lowercase' not in flags
    do_punct     = '--no-punct'     not in flags
    do_segment   = '--no-segment'   not in flags

    if do_segment:
        raw_sents = _py_segment(text)
    else:
        raw_sents = [s.strip() for s in text.split('\n') if s.strip()]
        if not raw_sents:
            raw_sents = [text.strip()] if text.strip() else []

    processed_sents = []
    all_tokens = []
    for s in raw_sents:
        p = s
        if do_lowercase:
            p = _py_lowercase(p)
        if do_punct:
            p = _py_remove_punct(p)
        processed_sents.append(p)
        all_tokens.append(_py_tokenize(p))

    total_tok = sum(len(t) for t in all_tokens)
    return {
        'sentences': processed_sents,
        'tokens': all_tokens,
        'stats': {'num_sentences': len(processed_sents), 'num_tokens': total_tok}
    }


def run_preprocessor(text, flags=None):
    """Call C binary (or Python fallback) with given flags, return parsed output dict."""
    if flags is None:
        flags = []
    if _USE_C_BINARY:
        binary = os.path.abspath(BINARY_PATH)
        result = subprocess.run(
            [binary] + flags,
            input=text, capture_output=True, text=True, timeout=30
        )
        return parse_output(result.stdout)
    else:
        return _python_preprocess(text, flags)


def parse_output(raw):
    """Parse C program output into dict with sentences, tokens, stats."""
    sections = {}
    current = None
    lines = raw.strip().split('\n')
    for line in lines:
        if line.endswith('_START'):
            current = line[:-6].lower()
            sections[current] = []
        elif line.endswith('_END'):
            current = None
        elif current is not None:
            sections[current].append(line)

    sentences = sections.get('sentences', [])
    tokens = [line.split() for line in sections.get('tokens', [])]
    stats = {}
    for s in sections.get('stats', []):
        if ':' in s:
            k, v = s.split(':', 1)
            try:
                stats[k] = int(v)
            except ValueError:
                stats[k] = v
    return {'sentences': sentences, 'tokens': tokens, 'stats': stats}


def preprocess_docs(docs, flags=None):
    """Preprocess all docs using C binary. Returns list of preprocessed dicts."""
    if flags is None:
        flags = []
    results = []
    for doc in docs:
        text = ' '.join(doc['sentences'])
        out = run_preprocessor(text, flags)
        results.append({
            'id':                     doc['id'],
            'topic':                  doc['topic'],
            'original_sentences':     doc['sentences'],
            'preprocessed_sentences': out['sentences'],
            'tokens':                 out['tokens'],
            'stats':                  out['stats'],
        })
    return results


def create_train_test_split(docs, test_ratio=0.2, seed=42):
    """Split docs into train and test sets."""
    random.seed(seed)
    shuffled = docs[:]
    random.shuffle(shuffled)
    split = int((1 - test_ratio) * len(shuffled))
    return shuffled[:split], shuffled[split:]
