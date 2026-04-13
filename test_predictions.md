# Test Prediction Analysis Report
This report analyzes the 'Actual vs Predicted' ordering for sample documents across different embedding methods.

### Embedding: 1. Word2Vec (Mean Pooled)

#### Sample Document 1 (ID: 39)
- **Kendall's Tau Score**: 0.3333
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | wordnets are lexico-semantic resources essential in many nlp tasks ... | wordnets for languages other than english tend to adopt unquestioningly wordnet'... | ❌ |
| 2 | princeton wordnet is the most widely known  and the most influential  among them... | princeton wordnet is the most widely known  and the most influential  among them... | ✅ |
| 3 | wordnets for languages other than english tend to adopt unquestioningly wordnet'... | wordnets are lexico-semantic resources essential in many nlp tasks ... | ❌ |
| 4 | we discuss a large wordnet constructed independently of wordnet  upon a model wi... | a mapping onto wordnet is under way  the large portions already linked open up a... | ❌ |
| 5 | a mapping onto wordnet is under way  the large portions already linked open up a... | we also try to characterise numerically a wordnet's aptitude for nlp application... | ❌ |
| 6 | we also try to characterise numerically a wordnet's aptitude for nlp application... | we discuss a large wordnet constructed independently of wordnet  upon a model wi... | ❌ |

**Analysis**: Incorrect opening: Predicted sentence 3 as start.

#### Sample Document 2 (ID: 360)
- **Kendall's Tau Score**: 0.3333
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | the biological text mining unit at bsc and cnio organized the first shared task ... | the shared task includes two tracks  one for ner offset and entity classificatio... | ❌ |
| 2 | the shared task includes two tracks  one for ner offset and entity classificatio... | we developed a pipeline system based on deep learning methods for this shared ta... | ❌ |
| 3 | we developed a pipeline system based on deep learning methods for this shared ta... | the biological text mining unit at bsc and cnio organized the first shared task ... | ❌ |
| 4 | evaluation conducted on the shared task data showed that our system achieves a m... | evaluation conducted on the shared task data showed that our system achieves a m... | ✅ |

**Analysis**: Incorrect opening: Predicted sentence 2 as start. Correctly identified the concluding sentence.

#### Sample Document 3 (ID: 379)
- **Kendall's Tau Score**: 0.6000
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | this paper is an outcome of ongoing research and presents an unsupervised method... | this paper is an outcome of ongoing research and presents an unsupervised method... | ✅ |
| 2 | the induction algorithm is based on modeling the cooccurrences of two or more wo... | the induction algorithm is based on modeling the cooccurrences of two or more wo... | ✅ |
| 3 | wsi takes place by detecting high-density components in the cooccurrence hypergr... | wsd assigns to each induced cluster a score equal to the sum of weights of its h... | ❌ |
| 4 | wsd assigns to each induced cluster a score equal to the sum of weights of its h... | our system participates in semeval-2007 word sense induction and discrimination ... | ❌ |
| 5 | our system participates in semeval-2007 word sense induction and discrimination ... | wsi takes place by detecting high-density components in the cooccurrence hypergr... | ❌ |

**Analysis**: Correctly identified the opening sentence.

#### Sample Document 4 (ID: 502)
- **Kendall's Tau Score**: 0.3333
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | verbs in romanian sometimes manifest local irregularities in the form of alterna... | we present a sequence tagging based method for learning stem alternations and en... | ❌ |
| 2 | we present a sequence tagging based method for learning stem alternations and en... | supervised training is based on a morphological dictionary  with a few regular e... | ❌ |
| 3 | supervised training is based on a morphological dictionary  with a few regular e... | verbs in romanian sometimes manifest local irregularities in the form of alterna... | ❌ |
| 4 | our best model improves upon previous machine learning approaches to romanian ve... | our best model improves upon previous machine learning approaches to romanian ve... | ✅ |

**Analysis**: Incorrect opening: Predicted sentence 2 as start. Correctly identified the concluding sentence.

#### Sample Document 5 (ID: 147)
- **Kendall's Tau Score**: -0.6667
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | we explore a rule system and a machine learning ml  approach to automatically ha... | here  the best f-scores for the rule and the ml systems amount to 34  and 19  re... | ❌ |
| 2 | in the lab condition  we test how feasible the automatic extraction of gres real... | in the lab condition  we test how feasible the automatic extraction of gres real... | ✅ |
| 3 | in the regu-londb condition  we investigate how robust both methodologies are by... | in the regu-londb condition  we investigate how robust both methodologies are by... | ✅ |
| 4 | here  the best f-scores for the rule and the ml systems amount to 34  and 19  re... | we explore a rule system and a machine learning ml  approach to automatically ha... | ❌ |

**Analysis**: Incorrect opening: Predicted sentence 4 as start.


---

### Embedding: 2. Word2Vec (TF-IDF Weighted)

#### Sample Document 1 (ID: 39)
- **Kendall's Tau Score**: 0.0667
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | wordnets are lexico-semantic resources essential in many nlp tasks ... | we discuss a large wordnet constructed independently of wordnet  upon a model wi... | ❌ |
| 2 | princeton wordnet is the most widely known  and the most influential  among them... | wordnets are lexico-semantic resources essential in many nlp tasks ... | ❌ |
| 3 | wordnets for languages other than english tend to adopt unquestioningly wordnet'... | we also try to characterise numerically a wordnet's aptitude for nlp application... | ❌ |
| 4 | we discuss a large wordnet constructed independently of wordnet  upon a model wi... | wordnets for languages other than english tend to adopt unquestioningly wordnet'... | ❌ |
| 5 | a mapping onto wordnet is under way  the large portions already linked open up a... | princeton wordnet is the most widely known  and the most influential  among them... | ❌ |
| 6 | we also try to characterise numerically a wordnet's aptitude for nlp application... | a mapping onto wordnet is under way  the large portions already linked open up a... | ❌ |

**Analysis**: Incorrect opening: Predicted sentence 4 as start.

#### Sample Document 2 (ID: 360)
- **Kendall's Tau Score**: 0.0000
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | the biological text mining unit at bsc and cnio organized the first shared task ... | we developed a pipeline system based on deep learning methods for this shared ta... | ❌ |
| 2 | the shared task includes two tracks  one for ner offset and entity classificatio... | the shared task includes two tracks  one for ner offset and entity classificatio... | ✅ |
| 3 | we developed a pipeline system based on deep learning methods for this shared ta... | the biological text mining unit at bsc and cnio organized the first shared task ... | ❌ |
| 4 | evaluation conducted on the shared task data showed that our system achieves a m... | evaluation conducted on the shared task data showed that our system achieves a m... | ✅ |

**Analysis**: Incorrect opening: Predicted sentence 3 as start. Correctly identified the concluding sentence.

#### Sample Document 3 (ID: 379)
- **Kendall's Tau Score**: 0.6000
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | this paper is an outcome of ongoing research and presents an unsupervised method... | this paper is an outcome of ongoing research and presents an unsupervised method... | ✅ |
| 2 | the induction algorithm is based on modeling the cooccurrences of two or more wo... | the induction algorithm is based on modeling the cooccurrences of two or more wo... | ✅ |
| 3 | wsi takes place by detecting high-density components in the cooccurrence hypergr... | wsd assigns to each induced cluster a score equal to the sum of weights of its h... | ❌ |
| 4 | wsd assigns to each induced cluster a score equal to the sum of weights of its h... | our system participates in semeval-2007 word sense induction and discrimination ... | ❌ |
| 5 | our system participates in semeval-2007 word sense induction and discrimination ... | wsi takes place by detecting high-density components in the cooccurrence hypergr... | ❌ |

**Analysis**: Correctly identified the opening sentence.

#### Sample Document 4 (ID: 502)
- **Kendall's Tau Score**: 0.3333
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | verbs in romanian sometimes manifest local irregularities in the form of alterna... | we present a sequence tagging based method for learning stem alternations and en... | ❌ |
| 2 | we present a sequence tagging based method for learning stem alternations and en... | supervised training is based on a morphological dictionary  with a few regular e... | ❌ |
| 3 | supervised training is based on a morphological dictionary  with a few regular e... | verbs in romanian sometimes manifest local irregularities in the form of alterna... | ❌ |
| 4 | our best model improves upon previous machine learning approaches to romanian ve... | our best model improves upon previous machine learning approaches to romanian ve... | ✅ |

**Analysis**: Incorrect opening: Predicted sentence 2 as start. Correctly identified the concluding sentence.

#### Sample Document 5 (ID: 147)
- **Kendall's Tau Score**: 0.0000
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | we explore a rule system and a machine learning ml  approach to automatically ha... | here  the best f-scores for the rule and the ml systems amount to 34  and 19  re... | ❌ |
| 2 | in the lab condition  we test how feasible the automatic extraction of gres real... | we explore a rule system and a machine learning ml  approach to automatically ha... | ❌ |
| 3 | in the regu-londb condition  we investigate how robust both methodologies are by... | in the lab condition  we test how feasible the automatic extraction of gres real... | ❌ |
| 4 | here  the best f-scores for the rule and the ml systems amount to 34  and 19  re... | in the regu-londb condition  we investigate how robust both methodologies are by... | ❌ |

**Analysis**: Incorrect opening: Predicted sentence 4 as start.


---

### Embedding: 3. TF-IDF (Sparse Vector)

#### Sample Document 1 (ID: 39)
- **Kendall's Tau Score**: 0.6000
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | wordnets are lexico-semantic resources essential in many nlp tasks ... | we discuss a large wordnet constructed independently of wordnet  upon a model wi... | ❌ |
| 2 | princeton wordnet is the most widely known  and the most influential  among them... | wordnets are lexico-semantic resources essential in many nlp tasks ... | ❌ |
| 3 | wordnets for languages other than english tend to adopt unquestioningly wordnet'... | princeton wordnet is the most widely known  and the most influential  among them... | ❌ |
| 4 | we discuss a large wordnet constructed independently of wordnet  upon a model wi... | wordnets for languages other than english tend to adopt unquestioningly wordnet'... | ❌ |
| 5 | a mapping onto wordnet is under way  the large portions already linked open up a... | a mapping onto wordnet is under way  the large portions already linked open up a... | ✅ |
| 6 | we also try to characterise numerically a wordnet's aptitude for nlp application... | we also try to characterise numerically a wordnet's aptitude for nlp application... | ✅ |

**Analysis**: Incorrect opening: Predicted sentence 4 as start. Correctly identified the concluding sentence.

#### Sample Document 2 (ID: 360)
- **Kendall's Tau Score**: 1.0000
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | the biological text mining unit at bsc and cnio organized the first shared task ... | the biological text mining unit at bsc and cnio organized the first shared task ... | ✅ |
| 2 | the shared task includes two tracks  one for ner offset and entity classificatio... | the shared task includes two tracks  one for ner offset and entity classificatio... | ✅ |
| 3 | we developed a pipeline system based on deep learning methods for this shared ta... | we developed a pipeline system based on deep learning methods for this shared ta... | ✅ |
| 4 | evaluation conducted on the shared task data showed that our system achieves a m... | evaluation conducted on the shared task data showed that our system achieves a m... | ✅ |

**Analysis**: Perfect ordering.

#### Sample Document 3 (ID: 379)
- **Kendall's Tau Score**: 0.6000
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | this paper is an outcome of ongoing research and presents an unsupervised method... | this paper is an outcome of ongoing research and presents an unsupervised method... | ✅ |
| 2 | the induction algorithm is based on modeling the cooccurrences of two or more wo... | the induction algorithm is based on modeling the cooccurrences of two or more wo... | ✅ |
| 3 | wsi takes place by detecting high-density components in the cooccurrence hypergr... | our system participates in semeval-2007 word sense induction and discrimination ... | ❌ |
| 4 | wsd assigns to each induced cluster a score equal to the sum of weights of its h... | wsi takes place by detecting high-density components in the cooccurrence hypergr... | ❌ |
| 5 | our system participates in semeval-2007 word sense induction and discrimination ... | wsd assigns to each induced cluster a score equal to the sum of weights of its h... | ❌ |

**Analysis**: Correctly identified the opening sentence.

#### Sample Document 4 (ID: 502)
- **Kendall's Tau Score**: 0.0000
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | verbs in romanian sometimes manifest local irregularities in the form of alterna... | supervised training is based on a morphological dictionary  with a few regular e... | ❌ |
| 2 | we present a sequence tagging based method for learning stem alternations and en... | we present a sequence tagging based method for learning stem alternations and en... | ✅ |
| 3 | supervised training is based on a morphological dictionary  with a few regular e... | verbs in romanian sometimes manifest local irregularities in the form of alterna... | ❌ |
| 4 | our best model improves upon previous machine learning approaches to romanian ve... | our best model improves upon previous machine learning approaches to romanian ve... | ✅ |

**Analysis**: Incorrect opening: Predicted sentence 3 as start. Correctly identified the concluding sentence.

#### Sample Document 5 (ID: 147)
- **Kendall's Tau Score**: 0.0000
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | we explore a rule system and a machine learning ml  approach to automatically ha... | in the regu-londb condition  we investigate how robust both methodologies are by... | ❌ |
| 2 | in the lab condition  we test how feasible the automatic extraction of gres real... | in the lab condition  we test how feasible the automatic extraction of gres real... | ✅ |
| 3 | in the regu-londb condition  we investigate how robust both methodologies are by... | we explore a rule system and a machine learning ml  approach to automatically ha... | ❌ |
| 4 | here  the best f-scores for the rule and the ml systems amount to 34  and 19  re... | here  the best f-scores for the rule and the ml systems amount to 34  and 19  re... | ✅ |

**Analysis**: Incorrect opening: Predicted sentence 3 as start. Correctly identified the concluding sentence.


---

### Embedding: 4. Contextual Token (DistilBERT)

#### Sample Document 1 (ID: 39)
- **Kendall's Tau Score**: 0.4667
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | wordnets are lexico-semantic resources essential in many nlp tasks ... | wordnets are lexico-semantic resources essential in many nlp tasks ... | ✅ |
| 2 | princeton wordnet is the most widely known  and the most influential  among them... | wordnets for languages other than english tend to adopt unquestioningly wordnet'... | ❌ |
| 3 | wordnets for languages other than english tend to adopt unquestioningly wordnet'... | princeton wordnet is the most widely known  and the most influential  among them... | ❌ |
| 4 | we discuss a large wordnet constructed independently of wordnet  upon a model wi... | we also try to characterise numerically a wordnet's aptitude for nlp application... | ❌ |
| 5 | a mapping onto wordnet is under way  the large portions already linked open up a... | a mapping onto wordnet is under way  the large portions already linked open up a... | ✅ |
| 6 | we also try to characterise numerically a wordnet's aptitude for nlp application... | we discuss a large wordnet constructed independently of wordnet  upon a model wi... | ❌ |

**Analysis**: Correctly identified the opening sentence. Contains 1 adjacent sentence swap(s).

#### Sample Document 2 (ID: 360)
- **Kendall's Tau Score**: 0.6667
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | the biological text mining unit at bsc and cnio organized the first shared task ... | the biological text mining unit at bsc and cnio organized the first shared task ... | ✅ |
| 2 | the shared task includes two tracks  one for ner offset and entity classificatio... | we developed a pipeline system based on deep learning methods for this shared ta... | ❌ |
| 3 | we developed a pipeline system based on deep learning methods for this shared ta... | the shared task includes two tracks  one for ner offset and entity classificatio... | ❌ |
| 4 | evaluation conducted on the shared task data showed that our system achieves a m... | evaluation conducted on the shared task data showed that our system achieves a m... | ✅ |

**Analysis**: Correctly identified the opening sentence. Correctly identified the concluding sentence. Contains 1 adjacent sentence swap(s).

#### Sample Document 3 (ID: 379)
- **Kendall's Tau Score**: 0.4000
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | this paper is an outcome of ongoing research and presents an unsupervised method... | this paper is an outcome of ongoing research and presents an unsupervised method... | ✅ |
| 2 | the induction algorithm is based on modeling the cooccurrences of two or more wo... | the induction algorithm is based on modeling the cooccurrences of two or more wo... | ✅ |
| 3 | wsi takes place by detecting high-density components in the cooccurrence hypergr... | our system participates in semeval-2007 word sense induction and discrimination ... | ❌ |
| 4 | wsd assigns to each induced cluster a score equal to the sum of weights of its h... | wsd assigns to each induced cluster a score equal to the sum of weights of its h... | ✅ |
| 5 | our system participates in semeval-2007 word sense induction and discrimination ... | wsi takes place by detecting high-density components in the cooccurrence hypergr... | ❌ |

**Analysis**: Correctly identified the opening sentence.

#### Sample Document 4 (ID: 502)
- **Kendall's Tau Score**: 0.6667
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | verbs in romanian sometimes manifest local irregularities in the form of alterna... | verbs in romanian sometimes manifest local irregularities in the form of alterna... | ✅ |
| 2 | we present a sequence tagging based method for learning stem alternations and en... | supervised training is based on a morphological dictionary  with a few regular e... | ❌ |
| 3 | supervised training is based on a morphological dictionary  with a few regular e... | we present a sequence tagging based method for learning stem alternations and en... | ❌ |
| 4 | our best model improves upon previous machine learning approaches to romanian ve... | our best model improves upon previous machine learning approaches to romanian ve... | ✅ |

**Analysis**: Correctly identified the opening sentence. Correctly identified the concluding sentence. Contains 1 adjacent sentence swap(s).

#### Sample Document 5 (ID: 147)
- **Kendall's Tau Score**: 0.6667
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | we explore a rule system and a machine learning ml  approach to automatically ha... | we explore a rule system and a machine learning ml  approach to automatically ha... | ✅ |
| 2 | in the lab condition  we test how feasible the automatic extraction of gres real... | in the regu-londb condition  we investigate how robust both methodologies are by... | ❌ |
| 3 | in the regu-londb condition  we investigate how robust both methodologies are by... | in the lab condition  we test how feasible the automatic extraction of gres real... | ❌ |
| 4 | here  the best f-scores for the rule and the ml systems amount to 34  and 19  re... | here  the best f-scores for the rule and the ml systems amount to 34  and 19  re... | ✅ |

**Analysis**: Correctly identified the opening sentence. Correctly identified the concluding sentence. Contains 1 adjacent sentence swap(s).


---

### Embedding: 5. Sequence Domain (SBERT)

#### Sample Document 1 (ID: 39)
- **Kendall's Tau Score**: 0.6000
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | wordnets are lexico-semantic resources essential in many nlp tasks ... | wordnets for languages other than english tend to adopt unquestioningly wordnet'... | ❌ |
| 2 | princeton wordnet is the most widely known  and the most influential  among them... | wordnets are lexico-semantic resources essential in many nlp tasks ... | ❌ |
| 3 | wordnets for languages other than english tend to adopt unquestioningly wordnet'... | princeton wordnet is the most widely known  and the most influential  among them... | ❌ |
| 4 | we discuss a large wordnet constructed independently of wordnet  upon a model wi... | a mapping onto wordnet is under way  the large portions already linked open up a... | ❌ |
| 5 | a mapping onto wordnet is under way  the large portions already linked open up a... | we discuss a large wordnet constructed independently of wordnet  upon a model wi... | ❌ |
| 6 | we also try to characterise numerically a wordnet's aptitude for nlp application... | we also try to characterise numerically a wordnet's aptitude for nlp application... | ✅ |

**Analysis**: Incorrect opening: Predicted sentence 3 as start. Correctly identified the concluding sentence. Contains 1 adjacent sentence swap(s).

#### Sample Document 2 (ID: 360)
- **Kendall's Tau Score**: 0.6667
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | the biological text mining unit at bsc and cnio organized the first shared task ... | the biological text mining unit at bsc and cnio organized the first shared task ... | ✅ |
| 2 | the shared task includes two tracks  one for ner offset and entity classificatio... | we developed a pipeline system based on deep learning methods for this shared ta... | ❌ |
| 3 | we developed a pipeline system based on deep learning methods for this shared ta... | the shared task includes two tracks  one for ner offset and entity classificatio... | ❌ |
| 4 | evaluation conducted on the shared task data showed that our system achieves a m... | evaluation conducted on the shared task data showed that our system achieves a m... | ✅ |

**Analysis**: Correctly identified the opening sentence. Correctly identified the concluding sentence. Contains 1 adjacent sentence swap(s).

#### Sample Document 3 (ID: 379)
- **Kendall's Tau Score**: 0.4000
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | this paper is an outcome of ongoing research and presents an unsupervised method... | this paper is an outcome of ongoing research and presents an unsupervised method... | ✅ |
| 2 | the induction algorithm is based on modeling the cooccurrences of two or more wo... | the induction algorithm is based on modeling the cooccurrences of two or more wo... | ✅ |
| 3 | wsi takes place by detecting high-density components in the cooccurrence hypergr... | our system participates in semeval-2007 word sense induction and discrimination ... | ❌ |
| 4 | wsd assigns to each induced cluster a score equal to the sum of weights of its h... | wsd assigns to each induced cluster a score equal to the sum of weights of its h... | ✅ |
| 5 | our system participates in semeval-2007 word sense induction and discrimination ... | wsi takes place by detecting high-density components in the cooccurrence hypergr... | ❌ |

**Analysis**: Correctly identified the opening sentence.

#### Sample Document 4 (ID: 502)
- **Kendall's Tau Score**: 0.3333
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | verbs in romanian sometimes manifest local irregularities in the form of alterna... | verbs in romanian sometimes manifest local irregularities in the form of alterna... | ✅ |
| 2 | we present a sequence tagging based method for learning stem alternations and en... | our best model improves upon previous machine learning approaches to romanian ve... | ❌ |
| 3 | supervised training is based on a morphological dictionary  with a few regular e... | we present a sequence tagging based method for learning stem alternations and en... | ❌ |
| 4 | our best model improves upon previous machine learning approaches to romanian ve... | supervised training is based on a morphological dictionary  with a few regular e... | ❌ |

**Analysis**: Correctly identified the opening sentence.

#### Sample Document 5 (ID: 147)
- **Kendall's Tau Score**: 1.0000
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | we explore a rule system and a machine learning ml  approach to automatically ha... | we explore a rule system and a machine learning ml  approach to automatically ha... | ✅ |
| 2 | in the lab condition  we test how feasible the automatic extraction of gres real... | in the lab condition  we test how feasible the automatic extraction of gres real... | ✅ |
| 3 | in the regu-londb condition  we investigate how robust both methodologies are by... | in the regu-londb condition  we investigate how robust both methodologies are by... | ✅ |
| 4 | here  the best f-scores for the rule and the ml systems amount to 34  and 19  re... | here  the best f-scores for the rule and the ml systems amount to 34  and 19  re... | ✅ |

**Analysis**: Perfect ordering.


---

### Embedding: 6. Fine-Tuned Token (DistilBERT [CLS])

#### Sample Document 1 (ID: 39)
- **Kendall's Tau Score**: 0.7333
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | wordnets are lexico-semantic resources essential in many nlp tasks ... | wordnets are lexico-semantic resources essential in many nlp tasks ... | ✅ |
| 2 | princeton wordnet is the most widely known  and the most influential  among them... | wordnets for languages other than english tend to adopt unquestioningly wordnet'... | ❌ |
| 3 | wordnets for languages other than english tend to adopt unquestioningly wordnet'... | princeton wordnet is the most widely known  and the most influential  among them... | ❌ |
| 4 | we discuss a large wordnet constructed independently of wordnet  upon a model wi... | we discuss a large wordnet constructed independently of wordnet  upon a model wi... | ✅ |
| 5 | a mapping onto wordnet is under way  the large portions already linked open up a... | we also try to characterise numerically a wordnet's aptitude for nlp application... | ❌ |
| 6 | we also try to characterise numerically a wordnet's aptitude for nlp application... | a mapping onto wordnet is under way  the large portions already linked open up a... | ❌ |

**Analysis**: Correctly identified the opening sentence. Contains 2 adjacent sentence swap(s).

#### Sample Document 2 (ID: 360)
- **Kendall's Tau Score**: 0.6667
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | the biological text mining unit at bsc and cnio organized the first shared task ... | the biological text mining unit at bsc and cnio organized the first shared task ... | ✅ |
| 2 | the shared task includes two tracks  one for ner offset and entity classificatio... | we developed a pipeline system based on deep learning methods for this shared ta... | ❌ |
| 3 | we developed a pipeline system based on deep learning methods for this shared ta... | the shared task includes two tracks  one for ner offset and entity classificatio... | ❌ |
| 4 | evaluation conducted on the shared task data showed that our system achieves a m... | evaluation conducted on the shared task data showed that our system achieves a m... | ✅ |

**Analysis**: Correctly identified the opening sentence. Correctly identified the concluding sentence. Contains 1 adjacent sentence swap(s).

#### Sample Document 3 (ID: 379)
- **Kendall's Tau Score**: 0.0000
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | this paper is an outcome of ongoing research and presents an unsupervised method... | our system participates in semeval-2007 word sense induction and discrimination ... | ❌ |
| 2 | the induction algorithm is based on modeling the cooccurrences of two or more wo... | the induction algorithm is based on modeling the cooccurrences of two or more wo... | ✅ |
| 3 | wsi takes place by detecting high-density components in the cooccurrence hypergr... | this paper is an outcome of ongoing research and presents an unsupervised method... | ❌ |
| 4 | wsd assigns to each induced cluster a score equal to the sum of weights of its h... | wsi takes place by detecting high-density components in the cooccurrence hypergr... | ❌ |
| 5 | our system participates in semeval-2007 word sense induction and discrimination ... | wsd assigns to each induced cluster a score equal to the sum of weights of its h... | ❌ |

**Analysis**: Incorrect opening: Predicted sentence 5 as start.

#### Sample Document 4 (ID: 502)
- **Kendall's Tau Score**: 1.0000
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | verbs in romanian sometimes manifest local irregularities in the form of alterna... | verbs in romanian sometimes manifest local irregularities in the form of alterna... | ✅ |
| 2 | we present a sequence tagging based method for learning stem alternations and en... | we present a sequence tagging based method for learning stem alternations and en... | ✅ |
| 3 | supervised training is based on a morphological dictionary  with a few regular e... | supervised training is based on a morphological dictionary  with a few regular e... | ✅ |
| 4 | our best model improves upon previous machine learning approaches to romanian ve... | our best model improves upon previous machine learning approaches to romanian ve... | ✅ |

**Analysis**: Perfect ordering.

#### Sample Document 5 (ID: 147)
- **Kendall's Tau Score**: 0.3333
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | we explore a rule system and a machine learning ml  approach to automatically ha... | we explore a rule system and a machine learning ml  approach to automatically ha... | ✅ |
| 2 | in the lab condition  we test how feasible the automatic extraction of gres real... | in the regu-londb condition  we investigate how robust both methodologies are by... | ❌ |
| 3 | in the regu-londb condition  we investigate how robust both methodologies are by... | here  the best f-scores for the rule and the ml systems amount to 34  and 19  re... | ❌ |
| 4 | here  the best f-scores for the rule and the ml systems amount to 34  and 19  re... | in the lab condition  we test how feasible the automatic extraction of gres real... | ❌ |

**Analysis**: Correctly identified the opening sentence.


---

