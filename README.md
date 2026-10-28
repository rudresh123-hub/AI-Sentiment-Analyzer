# SentimentAI — AI Sentiment Analyzer & Smart Review Reply Generator

A production-grade Natural Language Processing (NLP) deep learning application built with **Python 3.11**, **Streamlit**, **TensorFlow/Keras**, and **Scikit-learn**.

SentimentAI evaluates customer reviews, classifies sentiment as **Positive**, **Negative**, or **Neutral / Uncertain** using both a **Multilayer Perceptron (MLP)** and a **Long Short-Term Memory (LSTM)** neural network, provides evidence-based explainability, and contextually generates tailored customer responses.

---

## 🌟 Key Application Features

1. **Executive Dashboard**:
   - Live session KPIs: Total Reviews Analyzed, Positive %, Negative %, Latest Verdict, Model Operational Status.
   - Session sentiment distribution donut chart.
   - Model consensus and agreement rate telemetry.
   - Clean empty-state handling when no reviews have been analyzed.

2. **Dedicated Review Analysis & Explainability**:
   - Single-review interactive analyzer with character & word counts.
   - Presets for rapid testing (Product & Delivery, Damaged Order, Support Staff, Entertainment, Ambiguous).
   - Side-by-side model verdict cards (MLP probability vs LSTM probability).
   - Consensus classification with transparent uncertainty policy near boundary or under model divergence.
   - Natural language explainability highlighting detected sentiment cues and reasoning.

3. **Smart Review Reply Generator**:
   - Rule-based contextual reply engine with aspect and entity extraction.
   - Tone selections: *Professional*, *Warm & Appreciative / Empathetic*, *Concise*, *Resolution / Next Steps*.
   - **Regenerate Reply** button and **Direct In-Place Editing** before copying.
   - One-click copy snippet.
   - Honest labeling: Not an external LLM; zero API fees and deterministic execution.

4. **Batch CSV Review Analysis**:
   - Bulk review analysis with CSV drag-and-drop.
   - Vectorized batch inference for both MLP and LSTM.
   - Automated enrichment: MLP prediction, LSTM prediction, Final Consensus, Confidence, Agreement.
   - Ground-truth evaluation if a `sentiment` column is present in the uploaded dataset.
   - Enriched CSV export via one-click download.

5. **Model Performance & Evaluation**:
   - Genuine test-set metrics (Accuracy, Precision, Recall, F1-Score) on stratified held-out data.
   - Comparative grouped bar chart.
   - Side-by-side Confusion Matrix heatmaps.
   - Test-set predictions table with ground truth vs predicted labels.

6. **Dataset Explorer & Custom Training**:
   - Interactive data table with sentiment filtering and keyword search.
   - Class distribution and character length distribution charts.
   - Optional CSV upload to retrain both models dynamically with rollback to default.

7. **Session Prediction History**:
   - Session-persistent history log stored in Streamlit session state.
   - Filter by sentiment class.
   - CSV export and safe history clearing.

---

## 🏗️ Neural Network Architectures

### Model A: Multilayer Perceptron (MLP)
- **Feature Space:** Term Frequency-Inverse Document Frequency (TF-IDF) bigram vectors (max 500 features).
- **Hidden Layer:** Dense layer with 32 units, Rectified Linear Unit activation ($\max(0, z)$).
- **Output Layer:** Dense layer with 1 unit, Sigmoid activation function ($\sigma(z) = \frac{1}{1 + e^{-z}}$).
- **Loss & Optimizer:** Binary Cross-Entropy with Adam ($\alpha=0.005$).
- **Function:** Evaluates global n-gram importance regardless of token order.

### Model B: Long Short-Term Memory (LSTM)
- **Input Layer:** Integer-encoded token sequences with pre-padding ($T=30$).
- **Embedding Layer:** 800-word vocabulary mapped into dense 32-dimensional continuous vectors.
- **Recurrent Layer:** 32 LSTM memory cells utilizing gated state updates (Forget, Input, Candidate, Output gates).
- **Output Layer:** Dense layer with 1 unit, Sigmoid activation function.
- **Loss & Optimizer:** Binary Cross-Entropy with Adam ($\alpha=0.005$).
- **Function:** Learns temporal dependencies and sequential token patterns.

---

## 🚀 Run Instructions (macOS with Python 3.11)

### 1. Navigate to Project Directory
```bash
cd /Users/rudreshsantoshnikhare/Downloads/AI_Sentiment_Analyzer
```

### 2. Activate Virtual Environment
```bash
source .venv/bin/activate
```

*(If creating a new virtual environment with Python 3.11: `python3.11 -m venv .venv && source .venv/bin/activate`)*

### 3. Upgrade pip and Install Dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Launch the Streamlit Application
```bash
streamlit run app.py
```

The application will open automatically at `http://localhost:8501`.

---

## 📁 Project Directory Structure

```text
AI_Sentiment_Analyzer/
├── .streamlit/
│   └── config.toml          # Streamlit theme configuration (Indigo/Light) & headless settings
├── app.py                   # Complete multi-view Streamlit application
├── requirements.txt         # Minimal dependency list
├── sample_reviews.csv       # Sample dataset for custom training / batch testing
└── README.md                # Documentation and viva guide
```

---

## 🎓 College Viva / Technical Q&A

1. **Q: Why does the LSTM use pre-padding (`padding='pre'`) instead of post-padding?**
   - **A**: With post-padding, the LSTM finishes its recurrent computation processing trailing zero vectors, which dampens cell activation states. With pre-padding, zeros are processed first and the actual words arrive at the end, so the final hidden state directly captures the sentence's concluding sentiment.
2. **Q: How is data leakage prevented?**
   - **A**: The `TfidfVectorizer` and the `Tokenizer` are fitted exclusively on the 75% training split. The held-out 25% test split is strictly transformed without re-fitting vocabulary or document frequencies.
3. **Q: How are ambiguous or neutral reviews identified?**
   - **A**: If average model probabilities land near the 50% decision boundary ($0.42 < P < 0.58$), or if the MLP and LSTM models produce conflicting verdicts with moderate confidence, the review is classified as "Neutral / Uncertain".
4. **Q: Is the reply generator an LLM?**
   - **A**: No. It is a contextual template engine that extracts specific feedback aspects (e.g. fast delivery, damaged packaging) and formats personalized replies without paid API dependencies.
