Sentiment analysis project for Neuromatch Academy Deep Learning.
# Text Modalities and Stress in Text  
**NMA NLP Project – Team Mawu-lise**  
_A pilot study using the Hippocorpus dataset to explore emotion, memory, and semantic decoding in language._

---

## Overview

This project investigates whether textual features—specifically word frequency and semantic content—can be used to classify **self-reported stress levels** and **memory modality types** (e.g., recollection, imagination) in narrative texts. We use the [Hippocorpus dataset](https://huggingface.co/datasets/allenai/hippocorpus), a rich collection of personal stories with associated metadata, to test the capacity of simple and interpretable models to classify cognitive and emotional attributes from language.

---

## Objectives

We aim to address the following research questions:

1. **Can we predict stress level from word frequency alone?**
2. **Can we classify memory modality types from frequency data?**
3. **Is time since the narrated event predictive of stress level?**
4. **Does semantic information (e.g. BERT embeddings) improve classification accuracy?**
5. **Are certain text segments (beginning, middle, end) more informative for classification?**

This pilot study will guide future work with more complex models and larger feature sets.

---

## Project Structure
<pre> ```text
├── data/ # Preprocessed data and feature representations
├── notebooks/ # Jupyter notebooks for EDA, modeling, evaluation
├── models/ # Trained models and experiment outputs
├── results/ # Evaluation results and figures
└── README.md # Project overview and instructions
```
</pre>

---

## Hypotheses

We formally test the following hypotheses:

- **H₀₁:** Word frequency does **not** predict stress level above chance.  
- **H₁₁:** Word frequency **does** predict stress level above chance.
<br>

- **H₀₂:** Word frequency does **not** predict memory modality.  
- **H₁₂:** Word frequency **does** predict memory modality.
<br>

- **H₀₃:** Time since event and stress level are uncorrelated.  
- **H₁₃:** There is a correlation between time since event and stress.
<br>

- **H₀₄:** Semantic information does **not** improve classification.  
- **H₁₄:** Semantic information **improves** classification over frequency-based models.
<br>

- **H₀₅:** No difference in classification accuracy across text segments.  
- **H₁₅:** Some segments (beginning, middle, end) are more informative than others.

---

## Data

We use a subset of the [Hippocorpus dataset](https://huggingface.co/datasets/allenai/hippocorpus), which includes:

- Full narratives, summaries, and surprising moments
- Self-reported ratings (e.g. stress, openness, importance)
- Memory modality labels (`memType`)
- Demographic info (age, gender, race)
- Metadata (time since event, work time, etc.)

---

## Methods

**Approach 1:**
* Preprocessing: Bag of words, TF-IDF
  * Advantage: simple, fast, out-of-the-box
* Classification (one of): Naive Bayes, Logistic Regression, SVMs

**Approach 2:**
* Feature extraction: Lexicon based transforms (LIWC)
* Processing: TF-IDF
* Classifier (one of): Naive Bayes, Logistic Regression, SVMs

**Approach 3:**
* Preprocessing: word embedings (word2vec, GloVe)
  * Advantage: capturing some semantic similarity (e.g., "frightened" and "scared")
* Feature extraction: averaging, semantic pooling
* Classifier (one of): Naive Bayes, Logistic Regression, SVMs

**Approach 4:**
* Preprocessing: word embedings (word2vec, GloVe)
* Classifier (one of): CNNs, RNNs
  * Advantage: $ \text{pattern of words} \xrightarrow{\text{signal}} \text{class} $
    * Undertands context (e.g, negation)
  * Drawback: hardware, time for hyperparm tuning, opaque

**Approach 5:**
* Embeddings: word2vec, Spacy, SBERT, Tranformer ...
* Dimensionality Reduction: UMAP, PCA ...
* Clustering: k-Means, HDBSCAN, BIRCH ...
* Tokenizer: CountVectorize, POS ...
* Weighting Scheme: c-TF-IDF, c-TF-IDF BM25, c-TF-IDF+Normalization ...
* Representation Tuning: GPT/T5, KeyBERT ...
  * Advantage: fast, performant

**Approach 6:**
* Classifier (one of): finetuning pré-trained transformers (BERT and alike)
  * Advantage: best in class mixture of Global semantics and contex nuances
  * Drawback: hardware, opaque

---

## Key Results (Coming Soon)


---

## Resources

- [📄 Hippocorpus Paper](https://www.pnas.org/doi/epdf/10.1073/pnas.2211715119)
- [📦 Dataset on HuggingFace](https://huggingface.co/datasets/allenai/hippocorpus)
- [💻 Team GitHub Repo](https://github.com/pinchunc/NMA_DL_SentimentAnalysis/tree/main)

---

## Team

This project is developed as part of the NMA NLP Track by **Team Mawu-lise**.

Contributors:
- Pin-Chun Chen
- Liam Hart
- Victor Martins
- Hamid Abuwarda
- Esteban Leon

---

## Future Directions

After the pilot, we plan to:
- Incorporate pretrained embeddings (e.g. BERT, GloVe)
- Investigate novel features for stress analysis

---

## Contact

Have questions or suggestions? Feel free to open an issue or contact us via GitHub.

---

_This README will evolve as the project progresses. Stay tuned!_
