"""
SentimentAI — AI Sentiment Analyzer & Smart Review Reply Generator
Professional Streamlit Application Using MLP and LSTM Neural Networks
"""

import os
import sys

# Ensure writable cache directories for Keras and Matplotlib in all environments
os.environ.setdefault("KERAS_HOME", "/tmp/.keras")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/.matplotlib")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import re
import hashlib
import random
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# TensorFlow / Keras imports with robust fallback handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Embedding, LSTM, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:
    try:
        import tensorflow as tf
        from keras.models import Sequential
        from keras.layers import Dense, Embedding, LSTM, Input
        from keras.optimizers import Adam
        from keras.preprocessing.text import Tokenizer
        from keras.preprocessing.sequence import pad_sequences
    except ImportError as e:
        st.error(f"TensorFlow / Keras import failed: {e}. Please ensure TensorFlow is installed in your Python environment.")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Reproducibility seeds
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
try:
    tf.random.set_seed(RANDOM_SEED)
except Exception:
    pass

# ==============================================================================
# 1. BUILT-IN DATASET (Balanced across 6 domains: 72 reviews)
# ==============================================================================
BUILTIN_REVIEWS = [
    # --- Product Quality (6 positive, 6 negative) ---
    {"review": "The build quality is outstanding and the finish feels truly premium.", "sentiment": "positive"},
    {"review": "Flimsy plastic material that cracked after only two days of gentle use.", "sentiment": "negative"},
    {"review": "Solid metal construction, feels durable and built to last for years.", "sentiment": "positive"},
    {"review": "Cheap parts and terrible craftsmanship, broke during the first setup.", "sentiment": "negative"},
    {"review": "Durable screen and sleek ergonomic design, feels comfortable in hand.", "sentiment": "positive"},
    {"review": "Screen developed dead pixels within a week and the touch response is laggy.", "sentiment": "negative"},
    {"review": "The battery life easily lasts two full days on a single charge. Highly impressed!", "sentiment": "positive"},
    {"review": "The battery drains completely within three hours even on standby mode.", "sentiment": "negative"},
    {"review": "Superb sound quality with rich bass and crystal-clear vocals.", "sentiment": "positive"},
    {"review": "Audio is muffled, distorted at medium volume, and the left earbud stopped working.", "sentiment": "negative"},
    {"review": "Exceeded all my expectations in terms of performance and reliability.", "sentiment": "positive"},
    {"review": "Underperformed in every single benchmark test, very disappointed.", "sentiment": "negative"},

    # --- Customer Service (6 positive, 6 negative) ---
    {"review": "Customer support was incredibly helpful and solved my ticket in under five minutes.", "sentiment": "positive"},
    {"review": "The customer service agent was dismissive, rude, and refused to help.", "sentiment": "negative"},
    {"review": "Support staff were polite, attentive, and followed up to make sure everything worked.", "sentiment": "positive"},
    {"review": "Waited on hold for forty minutes only to be disconnected by the support team.", "sentiment": "negative"},
    {"review": "The helpdesk team went above and beyond to replace my damaged item immediately.", "sentiment": "positive"},
    {"review": "No response from customer support even after sending four follow-up emails.", "sentiment": "negative"},
    {"review": "Friendly and knowledgeable support representative answered all my technical questions.", "sentiment": "positive"},
    {"review": "The support rep had no knowledge of their own product and gave false information.", "sentiment": "negative"},
    {"review": "Technical support resolved my configuration issue patiently via live chat.", "sentiment": "positive"},
    {"review": "Live chat is just an unhelpful automated bot that loops in circles.", "sentiment": "negative"},
    {"review": "Customer representative was courteous, professional, and very understanding.", "sentiment": "positive"},
    {"review": "Agent yelled at me over the phone and hung up abruptly.", "sentiment": "negative"},

    # --- Delivery & Shipping (6 positive, 6 negative) ---
    {"review": "Lightning fast delivery, the item arrived two days ahead of schedule in pristine packaging.", "sentiment": "positive"},
    {"review": "Delivery was delayed by two weeks and the packaging was completely crushed upon arrival.", "sentiment": "negative"},
    {"review": "Arrived right on time and courier handled the fragile package with great care.", "sentiment": "positive"},
    {"review": "The courier left the package in the pouring rain without ringing the bell.", "sentiment": "negative"},
    {"review": "Shipped immediately and tracking updates were accurate at every step.", "sentiment": "positive"},
    {"review": "Tracking number never updated and the delivery took over a month to arrive.", "sentiment": "negative"},
    {"review": "Package was delivered securely in bubble wrap with fragile labels clearly visible.", "sentiment": "positive"},
    {"review": "Item arrived with a broken seal and clear signs of prior use.", "sentiment": "negative"},
    {"review": "Same-day delivery worked like a charm, arrived within four hours of ordering.", "sentiment": "positive"},
    {"review": "Delivery promised within two days but took twelve days with zero explanation.", "sentiment": "negative"},
    {"review": "Prompt shipping, eco-friendly packaging, and everything arrived safely.", "sentiment": "positive"},
    {"review": "Package was ripped open and one of the ordered items was missing.", "sentiment": "negative"},

    # --- Pricing & Value (6 positive, 6 negative) ---
    {"review": "Outstanding value for money, offers top-tier features at a very reasonable price.", "sentiment": "positive"},
    {"review": "Completely overpriced for such substandard performance, total waste of hard-earned money.", "sentiment": "negative"},
    {"review": "Affordable pricing without cutting corners on quality, great purchase overall.", "sentiment": "positive"},
    {"review": "Hidden charges added during checkout made the product far too expensive.", "sentiment": "negative"},
    {"review": "Best budget smartphone on the market, beats many expensive competitors.", "sentiment": "positive"},
    {"review": "Not worth the price tag, you are just paying for the brand name.", "sentiment": "negative"},
    {"review": "A genuine bargain for the quality offered, would definitely recommend to friends.", "sentiment": "positive"},
    {"review": "Extremely expensive considering the poor quality materials used.", "sentiment": "negative"},
    {"review": "Competitive pricing and frequent discounts make this brand our top choice.", "sentiment": "positive"},
    {"review": "They quietly increased the price right before a supposed holiday sale.", "sentiment": "negative"},
    {"review": "Exceptional features for this price point, truly unmatched value.", "sentiment": "positive"},
    {"review": "A complete rip-off, you can buy much better alternatives for half the price.", "sentiment": "negative"},

    # --- Movies & Entertainment (6 positive, 6 negative) ---
    {"review": "The movie was an absolute masterpiece with breathtaking cinematography and stellar acting.", "sentiment": "positive"},
    {"review": "Dull, predictable storyline with wooden acting; I walked out before the movie ended.", "sentiment": "negative"},
    {"review": "A gripping thriller with incredible plot twists that kept me on the edge of my seat.", "sentiment": "positive"},
    {"review": "Weak plot, terrible dialogue, and zero character development throughout the film.", "sentiment": "negative"},
    {"review": "A heartwarming film with fantastic humor and unforgettable performances.", "sentiment": "positive"},
    {"review": "A painfully slow and boring movie that lacked any emotion or depth.", "sentiment": "negative"},
    {"review": "Spectacular visual effects and a deeply emotional soundtrack.", "sentiment": "positive"},
    {"review": "CGI was laughably bad and the sound mixing was terribly unbalanced.", "sentiment": "negative"},
    {"review": "Brilliant acting by the lead cast made this an unforgettable drama.", "sentiment": "positive"},
    {"review": "Horrible screenplay, unnatural acting, and completely forgettable soundtrack.", "sentiment": "negative"},
    {"review": "A delightfully entertaining film suitable for the entire family.", "sentiment": "positive"},
    {"review": "Full of vulgarity and bad humor, completely unwatchable.", "sentiment": "negative"},

    # --- Shopping Experiences (6 positive, 6 negative) ---
    {"review": "Shopping on this platform was smooth, intuitive, and the checkout was seamless.", "sentiment": "positive"},
    {"review": "Terrible shopping experience, my payment was deducted twice and order was cancelled without reason.", "sentiment": "negative"},
    {"review": "Easy to navigate website and the return process was quick and hassle-free.", "sentiment": "positive"},
    {"review": "The website kept crashing during payment and customer care was unreachable.", "sentiment": "negative"},
    {"review": "User-friendly application with smooth navigation and instant order confirmations.", "sentiment": "positive"},
    {"review": "The mobile app is glitchy, sluggish, and constantly logs me out.", "sentiment": "negative"},
    {"review": "Effortless returns process and the refund was credited within twenty-four hours.", "sentiment": "positive"},
    {"review": "Return policy is misleading and they refused to honor the warranty.", "sentiment": "negative"},
    {"review": "Clear product descriptions and high-resolution images matched the real item exactly.", "sentiment": "positive"},
    {"review": "Misleading product description, the actual product looks nothing like the pictures.", "sentiment": "negative"},
    # --- Canonical Benchmarks & Cross-Domain Samples ---
    {"review": "The product quality is excellent and delivery was very fast.", "sentiment": "positive"},
    {"review": "The product arrived late and the packaging was damaged.", "sentiment": "negative"},
    {"review": "Outstanding customer service and fast resolution of my inquiry.", "sentiment": "positive"},
    {"review": "Terrible experience with unhelpful staff and completely broken packaging.", "sentiment": "negative"},
]


# ==============================================================================
# 2. MODEL DEFINITIONS & TRAINING PIPELINE (Cached)
# ==============================================================================
def get_dataset_hash(df: pd.DataFrame) -> str:
    """Generate MD5 hash of review data to trigger retraining only when dataset changes."""
    content = "".join(df["review"].astype(str) + df["sentiment"].astype(str))
    return hashlib.md5(content.encode("utf-8")).hexdigest()


@st.cache_resource(show_spinner=False)
def train_models_pipeline(df_records: list, data_hash: str):
    """
    Train both MLP and LSTM models on the dataset.
    Cached per dataset hash to ensure training happens only once per session/data change.
    """
    df = pd.DataFrame(df_records)
    # Ensure binary column: positive -> 1, negative -> 0
    df["label"] = df["sentiment"].apply(lambda s: 1 if str(s).strip().lower() == "positive" else 0)

    # Convert pandas/Arrow columns safely to standard Python lists
    reviews = df["review"].fillna("").astype(str).tolist()
    labels = df["label"].astype("int64").tolist()

    # Stratified train/test split (75% train, 25% test)
    X_train_raw, X_test_raw, y_train_list, y_test_list = train_test_split(
        reviews,
        labels,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    y_train = np.array(y_train_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.int32)

    # ------------------ MLP Preprocessing: TF-IDF ------------------
    # Fitted ONLY on training data to prevent data leakage (unigrams + bigrams)
    max_tfidf_features = 500
    tfidf_vectorizer = TfidfVectorizer(
        max_features=max_tfidf_features,
        ngram_range=(1, 2),
        stop_words="english",
        lowercase=True,
    )
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_raw).toarray()
    X_test_tfidf = tfidf_vectorizer.transform(X_test_raw).toarray()

    # ------------------ LSTM Preprocessing: Tokenizer & Padding ------------------
    # Fitted ONLY on training data to prevent data leakage
    max_seq_len = 30
    tokenizer = Tokenizer(num_words=800, oov_token="<OOV>", lower=True)
    tokenizer.fit_on_texts(X_train_raw)
    vocab_size = max(800, len(tokenizer.word_index) + 2)

    X_train_seq = pad_sequences(
        tokenizer.texts_to_sequences(X_train_raw),
        maxlen=max_seq_len,
        padding="pre",
        truncating="post",
    )
    X_test_seq = pad_sequences(
        tokenizer.texts_to_sequences(X_test_raw),
        maxlen=max_seq_len,
        padding="pre",
        truncating="post",
    )

    # ------------------ Model A: MLP Classifier ------------------
    # Architecture: Input (TF-IDF) -> Dense 32 (ReLU) -> Dense 1 (Sigmoid)
    mlp_model = Sequential([
        Input(shape=(X_train_tfidf.shape[1],)),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    mlp_model.compile(
        optimizer=Adam(learning_rate=0.005),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    mlp_model.fit(
        X_train_tfidf,
        y_train,
        epochs=15,
        batch_size=4,
        verbose=0,
    )

    # ------------------ Model B: LSTM Classifier ------------------
    # Architecture: Input (Seq) -> Embedding(dim=32) -> LSTM(32) -> Dense 1 (Sigmoid)
    lstm_model = Sequential([
        Input(shape=(max_seq_len,)),
        Embedding(input_dim=vocab_size, output_dim=32),
        LSTM(32),
        Dense(1, activation="sigmoid"),
    ])
    lstm_model.compile(
        optimizer=Adam(learning_rate=0.005),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    lstm_model.fit(
        X_train_seq,
        y_train,
        epochs=12,
        batch_size=4,
        verbose=0,
    )

    # ------------------ Held-out Test Set Evaluation ------------------
    # MLP Evaluation
    mlp_probs = mlp_model.predict(X_test_tfidf, verbose=0).flatten()
    mlp_preds = (mlp_probs >= 0.5).astype(int)

    mlp_metrics = {
        "accuracy": float(accuracy_score(y_test, mlp_preds)),
        "precision": float(precision_score(y_test, mlp_preds, zero_division=0)),
        "recall": float(recall_score(y_test, mlp_preds, zero_division=0)),
        "f1": float(f1_score(y_test, mlp_preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, mlp_preds),
    }

    # LSTM Evaluation
    lstm_probs = lstm_model.predict(X_test_seq, verbose=0).flatten()
    lstm_preds = (lstm_probs >= 0.5).astype(int)

    lstm_metrics = {
        "accuracy": float(accuracy_score(y_test, lstm_preds)),
        "precision": float(precision_score(y_test, lstm_preds, zero_division=0)),
        "recall": float(recall_score(y_test, lstm_preds, zero_division=0)),
        "f1": float(f1_score(y_test, lstm_preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, lstm_preds),
    }

    # Test comparison dataframe
    test_eval_df = pd.DataFrame({
        "Review": X_test_raw,
        "Actual Sentiment": ["Positive" if y == 1 else "Negative" for y in y_test],
        "MLP Pred": ["Positive" if p == 1 else "Negative" for p in mlp_preds],
        "LSTM Pred": ["Positive" if p == 1 else "Negative" for p in lstm_preds],
        "Models Agree": ["Yes" if m == l else "No" for m, l in zip(mlp_preds, lstm_preds)],
    })

    return {
        "mlp_model": mlp_model,
        "lstm_model": lstm_model,
        "tfidf_vectorizer": tfidf_vectorizer,
        "tokenizer": tokenizer,
        "max_seq_len": max_seq_len,
        "mlp_metrics": mlp_metrics,
        "lstm_metrics": lstm_metrics,
        "test_eval_df": test_eval_df,
        "train_count": len(X_train_raw),
        "test_count": len(X_test_raw),
    }


# ==============================================================================
# 3. CONTEXT EXTRACTION & SMART REPLY GENERATOR
# ==============================================================================
def extract_review_details(text: str) -> dict:
    """
    Extract specific feedback aspects, pain points, and praise phrases
    from review text to generate personalized customer responses.
    """
    text_lower = text.lower()

    # Specific positive phrases
    pos_aspects = []
    if "product quality" in text_lower or ("quality" in text_lower and ("excellent" in text_lower or "great" in text_lower or "good" in text_lower or "premium" in text_lower)):
        pos_aspects.append("the product quality")
    if "fast delivery" in text_lower or "lightning fast" in text_lower or ("delivery" in text_lower and ("fast" in text_lower or "prompt" in text_lower or "ahead of schedule" in text_lower)):
        pos_aspects.append("fast delivery")
    if "customer support" in text_lower or "customer service" in text_lower or "support team" in text_lower or "helpdesk" in text_lower:
        pos_aspects.append("our customer support")
    if "battery life" in text_lower or ("battery" in text_lower and ("lasts" in text_lower or "impressed" in text_lower)):
        pos_aspects.append("the battery life")
    if "sound quality" in text_lower or "audio" in text_lower:
        pos_aspects.append("the audio quality")
    if "value for money" in text_lower or "pricing" in text_lower or "budget" in text_lower or "bargain" in text_lower:
        pos_aspects.append("the product value")
    if "acting" in text_lower or "cinematography" in text_lower or "movie" in text_lower or "soundtrack" in text_lower:
        pos_aspects.append("the entertainment experience")
    if "checkout" in text_lower or "website" in text_lower or "shopping" in text_lower or "returns" in text_lower:
        pos_aspects.append("the smooth shopping experience")

    # Specific negative phrases
    neg_aspects = []
    if "arrived late" in text_lower or "late" in text_lower or "delayed" in text_lower:
        neg_aspects.append("your order arrived late")
    if "packaging was damaged" in text_lower or "damaged" in text_lower or "crushed" in text_lower or "broken seal" in text_lower:
        neg_aspects.append("the packaging was damaged")
    if "flimsy" in text_lower or "cheap" in text_lower or "cracked" in text_lower or "broke" in text_lower or "broken" in text_lower:
        neg_aspects.append("the product build quality fell short")
    if "rude" in text_lower or "dismissive" in text_lower or "unhelpful" in text_lower or "disconnected" in text_lower or "no response" in text_lower:
        neg_aspects.append("you encountered unhelpful customer service")
    if "battery drains" in text_lower or ("battery" in text_lower and ("drains" in text_lower or "poor" in text_lower)):
        neg_aspects.append("the battery performance was disappointing")
    if "overpriced" in text_lower or "hidden charges" in text_lower or "rip-off" in text_lower or "expensive" in text_lower:
        neg_aspects.append("the pricing did not reflect acceptable value")
    if "dull" in text_lower or "boring" in text_lower or "predictable" in text_lower or "unwatchable" in text_lower:
        neg_aspects.append("the film did not meet your entertainment expectations")
    if "crashing" in text_lower or "glitchy" in text_lower or "charged my card" in text_lower or "errors" in text_lower:
        neg_aspects.append("the technical glitches interrupted your purchase")

    # General domain detection fallback
    domain_map = {
        "Product Quality": ["quality", "material", "build", "durability", "sturdy", "flimsy", "defect", "craftsmanship"],
        "Customer Service": ["service", "support", "agent", "staff", "representative", "helpdesk", "call", "chat"],
        "Delivery & Packaging": ["delivery", "shipping", "courier", "package", "box", "transit", "shipped"],
        "Pricing & Value": ["price", "cost", "expensive", "cheap", "worth", "value", "overpriced", "bargain"],
        "Entertainment & Media": ["movie", "film", "acting", "plot", "screenplay", "cinema", "story"],
        "Shopping Experience": ["checkout", "website", "order", "purchase", "shopping", "app", "refund", "return"],
    }
    detected_domain = "General Experience"
    for domain, terms in domain_map.items():
        if any(term in text_lower for term in terms):
            detected_domain = domain
            break

    return {
        "pos_aspects": pos_aspects,
        "neg_aspects": neg_aspects,
        "domain": detected_domain,
    }


def generate_review_replies(sentiment: str, details: dict, variation_seed: int = 0) -> dict:
    """
    Generate contextual template responses tailored to specific review aspects.
    Includes multiple tones: Professional, Warm & Appreciative, Concise, and Resolution.
    """
    pos_cues = details["pos_aspects"]
    neg_cues = details["neg_aspects"]
    domain = details["domain"].lower()

    if pos_cues:
        if len(pos_cues) == 1:
            aspect_str = pos_cues[0]
        else:
            aspect_str = f"{pos_cues[0]} and {pos_cues[1]}"
    else:
        aspect_str = f"our {domain}"

    if neg_cues:
        if len(neg_cues) == 1:
            issue_str = neg_cues[0]
        else:
            issue_str = f"{neg_cues[0]} and {neg_cues[1]}"
    else:
        issue_str = f"your experience with our {domain} fell below expectations"

    if sentiment == "Positive":
        replies = {
            "Professional": (
                f"Thank you for your wonderful feedback! We're delighted that you enjoyed {aspect_str}. "
                "We appreciate your support and look forward to serving you again."
            ),
            "Warm & Appreciative": (
                f"Thank you so much for taking the time to share this! Our team is thrilled to hear that {aspect_str} "
                "made a great impression. Your kind words keep us inspired!"
            ),
            "Concise": (
                f"Thank you for the fantastic review regarding {aspect_str}! We truly appreciate your patronage."
            ),
            "Next Steps / Milestone": (
                f"We are delighted by your positive experience with {aspect_str}. Please feel free to reach out anytime if "
                "we can assist with future orders or recommendations."
            ),
        }
    elif sentiment == "Negative":
        replies = {
            "Professional": (
                f"We're sorry that {issue_str}. This isn't the experience we want for our customers. "
                "Thank you for bringing this to our attention."
            ),
            "Empathetic & Attentive": (
                f"We sincerely apologize for the frustration caused because {issue_str}. "
                "We hold ourselves to a much higher standard and would appreciate the chance to make things right."
            ),
            "Concise": (
                f"We apologize for the inconvenience regarding your {domain}. Please contact support@example.com so we can help."
            ),
            "Resolution-Oriented": (
                f"We regret that {issue_str}. Please reach out to our dedicated support team at support@example.com with your "
                "order reference so our team can investigate and assist you directly."
            ),
        }
    else:  # Neutral / Uncertain
        replies = {
            "Professional": (
                f"Thank you for sharing your feedback regarding our {domain}. We appreciate both positive notes and constructive "
                "criticism as we work to improve our services."
            ),
            "Empathetic & Attentive": (
                f"Thank you for reaching out regarding our {domain}. We strive to provide consistent quality and welcome any "
                "additional details on how we can better support you."
            ),
            "Concise": (
                f"Thank you for your feedback on our {domain}. We have shared your comments with our quality assurance team."
            ),
            "Resolution-Oriented": (
                f"We appreciate your perspective regarding our {domain}. If there is any specific matter our team can assist with, "
                "please contact us at support@example.com."
            ),
        }

    return replies


# ==============================================================================
# 4. EXPLAINABILITY ENGINE
# ==============================================================================
def explain_prediction(review: str, mlp_prob: float, lstm_prob: float, sentiment: str, details: dict) -> str:
    """Generate a clear, human-interpretable explanation of the model verdict."""
    mlp_label = "Positive" if mlp_prob >= 0.5 else "Negative"
    lstm_label = "Positive" if lstm_prob >= 0.5 else "Negative"
    models_agree = (mlp_label == lstm_label)

    pos_cues = details["pos_aspects"]
    neg_cues = details["neg_aspects"]

    if sentiment == "Neutral / Uncertain":
        if not models_agree:
            return (
                f"The models diverged on this review: MLP predicted {mlp_label} ({mlp_prob:.1%}), "
                f"while LSTM predicted {lstm_label} ({lstm_prob:.1%}). Because of conflicting signals or "
                "subtle phrasing, the prediction is categorized as Uncertain. Manual human review is recommended."
            )
        else:
            return (
                f"Both models generated probabilities close to the decision boundary (MLP: {mlp_prob:.1%}, "
                f"LSTM: {lstm_prob:.1%}). The text contains mild or mixed sentiment language."
            )

    if sentiment == "Positive":
        aspect_mention = f" ({', '.join(pos_cues)})" if pos_cues else ""
        if models_agree:
            return (
                f"The review contains favorable terminology{aspect_mention}. Both neural networks strongly agree "
                f"on positive polarity (MLP confidence: {mlp_prob:.1%}, LSTM confidence: {lstm_prob:.1%})."
            )
        else:
            dominant = "MLP" if mlp_prob >= 0.5 else "LSTM"
            return (
                f"The review indicates positive sentiment{aspect_mention}, driven primarily by the {dominant} model's "
                f"higher confidence score despite minor sequence divergence."
            )
    else:  # Negative
        aspect_mention = f" ({', '.join(neg_cues)})" if neg_cues else ""
        if models_agree:
            return (
                f"The review contains critical language highlighting dissatisfaction{aspect_mention}. "
                f"Both classifiers confirmed negative polarity (MLP: {1.0 - mlp_prob:.1%}, LSTM: {1.0 - lstm_prob:.1%})."
            )
        else:
            dominant = "MLP" if mlp_prob < 0.5 else "LSTM"
            return (
                f"The review expresses customer dissatisfaction{aspect_mention}, classified as Negative based on the "
                f"stronger confidence of the {dominant} network."
            )


# ==============================================================================
# 5. SESSION STATE INITIALIZATION
# ==============================================================================
if "history" not in st.session_state:
    st.session_state["history"] = []

if "active_data" not in st.session_state:
    st.session_state["active_data"] = list(BUILTIN_REVIEWS)
    st.session_state["dataset_source"] = "Built-in (72 reviews, balanced across 6 domains)"

if "review_input" not in st.session_state:
    st.session_state["review_input"] = ""

if "variation_seed" not in st.session_state:
    st.session_state["variation_seed"] = 0

current_df = pd.DataFrame(st.session_state["active_data"])
data_hash = get_dataset_hash(current_df)

# Train or retrieve cached models
pipeline_artifacts = train_models_pipeline(st.session_state["active_data"], data_hash)


# ==============================================================================
# 6. INFERENCE HELPER FUNCTIONS
# ==============================================================================
def predict_single_review(text: str):
    """Run both MLP and LSTM on a text string and return detailed prediction dict."""
    tfidf_vec = pipeline_artifacts["tfidf_vectorizer"]
    mlp_model = pipeline_artifacts["mlp_model"]
    tokenizer = pipeline_artifacts["tokenizer"]
    lstm_model = pipeline_artifacts["lstm_model"]
    max_len = pipeline_artifacts["max_seq_len"]

    # MLP inference
    tfidf_feats = tfidf_vec.transform([text]).toarray()
    mlp_prob = float(mlp_model.predict(tfidf_feats, verbose=0)[0][0])
    mlp_label = "Positive" if mlp_prob >= 0.5 else "Negative"
    mlp_conf = mlp_prob if mlp_prob >= 0.5 else (1.0 - mlp_prob)

    # LSTM inference
    seq_feats = pad_sequences(
        tokenizer.texts_to_sequences([text]),
        maxlen=max_len,
        padding="pre",
        truncating="post",
    )
    lstm_prob = float(lstm_model.predict(seq_feats, verbose=0)[0][0])
    lstm_label = "Positive" if lstm_prob >= 0.5 else "Negative"
    lstm_conf = lstm_prob if lstm_prob >= 0.5 else (1.0 - lstm_prob)

    # Consensus and Uncertainty Policy
    avg_prob = (mlp_prob + lstm_prob) / 2.0
    models_agree = (mlp_label == lstm_label)

    # Flag as neutral/uncertain if near boundary or if models strongly disagree
    if not models_agree and (mlp_conf < 0.70 or lstm_conf < 0.70):
        final_sentiment = "Neutral / Uncertain"
        confidence = 0.50 + abs(avg_prob - 0.50)
    elif 0.42 < avg_prob < 0.58:
        final_sentiment = "Neutral / Uncertain"
        confidence = 0.50 + abs(avg_prob - 0.50)
    elif avg_prob >= 0.58:
        final_sentiment = "Positive"
        confidence = avg_prob
    else:
        final_sentiment = "Negative"
        confidence = 1.0 - avg_prob

    details = extract_review_details(text)
    explanation = explain_prediction(text, mlp_prob, lstm_prob, final_sentiment, details)

    return {
        "text": text,
        "mlp_label": mlp_label,
        "mlp_prob": mlp_prob,
        "mlp_conf": mlp_conf,
        "lstm_label": lstm_label,
        "lstm_prob": lstm_prob,
        "lstm_conf": lstm_conf,
        "models_agree": models_agree,
        "final_sentiment": final_sentiment,
        "confidence": confidence,
        "details": details,
        "explanation": explanation,
    }


# ==============================================================================
# 7. PAGE VIEWS
# ==============================================================================

# --- View 1: Executive Dashboard ---
def dashboard_view():
    st.title("SentimentAI Executive Dashboard")
    st.caption("Real-time sentiment telemetry and model health monitoring for current session.")

    history = st.session_state["history"]
    total_reviews = len(history)

    # Top KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(label="Reviews Analyzed", value=total_reviews)
    with col2:
        if total_reviews > 0:
            pos_pct = sum(1 for r in history if r["final_sentiment"] == "Positive") / total_reviews * 100
            st.metric(label="Positive %", value=f"{pos_pct:.1f}%")
        else:
            st.metric(label="Positive %", value="—")
    with col3:
        if total_reviews > 0:
            neg_pct = sum(1 for r in history if r["final_sentiment"] == "Negative") / total_reviews * 100
            st.metric(label="Negative %", value=f"{neg_pct:.1f}%")
        else:
            st.metric(label="Negative %", value="—")
    with col4:
        if total_reviews > 0:
            last_sentiment = history[-1]["final_sentiment"]
            st.metric(label="Latest Verdict", value=last_sentiment)
        else:
            st.metric(label="Latest Verdict", value="—")
    with col5:
        st.metric(label="Model Status", value="Online", delta="MLP & LSTM Ready")

    st.divider()

    if total_reviews == 0:
        with st.container(border=True):
            st.info(
                "No reviews have been analyzed in this session yet. "
                "Navigate to **Analyze Review** to test single feedback or **Batch Analysis** to upload a CSV file.",
                title="Session Telemetry Ready",
                icon=":material/info:",
            )
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Model A: MLP (TF-IDF)")
                st.write("Feedforward neural architecture trained on 500 n-gram TF-IDF numerical vectors.")
            with col_b:
                st.subheader("Model B: LSTM (Sequential)")
                st.write("Recurrent neural network with 32 embedding dimensions and gated memory cells.")
    else:
        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            st.subheader("Session Sentiment Distribution")
            pos_c = sum(1 for r in history if r["final_sentiment"] == "Positive")
            neg_c = sum(1 for r in history if r["final_sentiment"] == "Negative")
            neu_c = sum(1 for r in history if r["final_sentiment"] == "Neutral / Uncertain")

            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            counts = [pos_c, neg_c, neu_c]
            labels = ["Positive", "Negative", "Uncertain"]
            colors = ["#16a34a", "#dc2626", "#d97706"]

            # Filter out zero categories for clean plotting
            active_counts = [c for c in counts if c > 0]
            active_labels = [l for c, l in zip(counts, labels) if c > 0]
            active_colors = [col for c, col in zip(counts, colors) if c > 0]

            wedges, texts, autotexts = ax.pie(
                active_counts,
                labels=active_labels,
                colors=active_colors,
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops={"width": 0.45, "edgecolor": "white", "linewidth": 2},
            )
            for at in autotexts:
                at.set_color("white")
                at.set_weight("bold")
            ax.axis("equal")
            st.pyplot(fig)
            plt.close(fig)

        with col_c2:
            st.subheader("Model Agreement Metric")
            agreement_count = sum(1 for r in history if r["agreement"] == "Yes")
            agreement_rate = (agreement_count / total_reviews) * 100 if total_reviews > 0 else 0
            st.metric(label="Consensus Rate", value=f"{agreement_rate:.1f}%", help="Percentage of reviews where both MLP and LSTM predicted identical sentiment classes.")
            st.progress(float(agreement_rate / 100))

            st.write(f"- **Total Reviews Analyzed:** {total_reviews}")
            st.write(f"- **Consensus Reviews:** {agreement_count}")
            st.write(f"- **Divergent Reviews:** {total_reviews - agreement_count}")

        st.subheader("Recent Activity in Session")
        recent_df = pd.DataFrame([
            {
                "Time": r["timestamp"],
                "Review Snippet": r["review"][:65] + "..." if len(r["review"]) > 65 else r["review"],
                "MLP": r["mlp_pred"],
                "LSTM": r["lstm_pred"],
                "Final Verdict": r["final_sentiment"],
                "Confidence": f"{r['confidence'] * 100:.1f}%",
            }
            for r in reversed(history[-5:])
        ])
        st.dataframe(recent_df, hide_index=True)


# --- View 2: Analyze Review ---
def analyze_view():
    st.title("Analyze Customer Review")
    st.caption("Input customer feedback to predict sentiment, evaluate model agreement, and generate a contextual response.")

    # Preset Quick Buttons
    st.markdown("##### Quick Example Reviews:")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("⭐ Product & Fast Delivery", width="stretch"):
        st.session_state["review_input"] = "The product quality is excellent and delivery was very fast."
    if c2.button("📦 Damaged & Late Delivery", width="stretch"):
        st.session_state["review_input"] = "The product arrived late and the packaging was damaged."
    if c3.button("🎧 Unhelpful Support Staff", width="stretch"):
        st.session_state["review_input"] = "The customer service agent was dismissive, rude, and refused to help."
    if c4.button("🎬 Thrilling Movie Experience", width="stretch"):
        st.session_state["review_input"] = "A gripping thriller with incredible plot twists that kept me on the edge of my seat."

    current_val = st.session_state.get("review_input", "")

    # Input Box
    review_input = st.text_area(
        label="Customer Review Text",
        value=current_val,
        placeholder="Type or paste customer review text here...",
        height=130,
    )

    # Character and word count
    char_count = len(review_input)
    word_count = len(review_input.split()) if review_input.strip() else 0
    st.caption(f"Character count: {char_count} | Word count: {word_count}")

    btn_col1, btn_col2, _ = st.columns([1.5, 1, 3])
    with btn_col1:
        analyze_clicked = st.button("🚀 Analyze Review", type="primary", width="stretch")
    with btn_col2:
        clear_clicked = st.button("Clear Input", width="stretch")
        if clear_clicked:
            st.session_state["review_input"] = ""
            st.rerun()

    if analyze_clicked:
        if not review_input.strip() or len(review_input.strip()) < 3:
            st.warning("Please provide a valid review with at least 3 characters.", title="Input Required", icon=":material/warning:")
        else:
            with st.spinner("Evaluating review with MLP & LSTM neural classifiers..."):
                res = predict_single_review(review_input)

            # Record in session history
            replies = generate_review_replies(res["final_sentiment"], res["details"])
            initial_reply = replies["Professional"]

            st.session_state["history"].append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "review": review_input,
                "mlp_pred": res["mlp_label"],
                "mlp_conf": res["mlp_conf"],
                "lstm_pred": res["lstm_label"],
                "lstm_conf": res["lstm_conf"],
                "final_sentiment": res["final_sentiment"],
                "confidence": res["confidence"],
                "agreement": "Yes" if res["models_agree"] else "No",
                "reply": initial_reply,
            })

            st.session_state["last_analysis"] = res
            st.session_state["active_replies"] = replies

    # Display Results if available
    if "last_analysis" in st.session_state and st.session_state.get("review_input"):
        res = st.session_state["last_analysis"]
        st.divider()

        st.subheader("Analysis Results")

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            with st.container(border=True):
                st.caption("CONSENSUS SENTIMENT")
                if res["final_sentiment"] == "Positive":
                    st.success("Positive", icon=":material/thumb_up:")
                elif res["final_sentiment"] == "Negative":
                    st.error("Negative", icon=":material/thumb_down:")
                else:
                    st.warning("Neutral / Uncertain", icon=":material/help:")

        with res_col2:
            with st.container(border=True):
                st.caption("CONFIDENCE SCORE")
                st.metric(label="Score", value=f"{res['confidence'] * 100:.1f}%", label_visibility="collapsed")
                st.progress(float(res["confidence"]))

        with res_col3:
            with st.container(border=True):
                st.caption("MODEL AGREEMENT")
                if res["models_agree"]:
                    st.success("Consensus (Both Models Agree)", icon=":material/handshake:")
                else:
                    st.warning("Divergence (Models Disagree)", icon=":material/call_split:")

        # Individual Model Cards
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            with st.container(border=True):
                st.markdown("##### Model A: MLP Classifier (TF-IDF)")
                st.write(f"- **Verdict:** {res['mlp_label']}")
                st.write(f"- **Raw Sigmoid Output:** `{res['mlp_prob']:.4f}`")
                st.write(f"- **Confidence:** {res['mlp_conf'] * 100:.1f}%")

        with m_col2:
            with st.container(border=True):
                st.markdown("##### Model B: LSTM Classifier (Word Sequences)")
                st.write(f"- **Verdict:** {res['lstm_label']}")
                st.write(f"- **Raw Sigmoid Output:** `{res['lstm_prob']:.4f}`")
                st.write(f"- **Confidence:** {res['lstm_conf'] * 100:.1f}%")

        # Explainability Card
        with st.container(border=True):
            st.markdown("##### 💡 Explainability & Reasoning")
            st.info(res["explanation"], title="Prediction Context", icon=":material/lightbulb:")
            if res["final_sentiment"] == "Neutral / Uncertain":
                st.caption("Notice: Neural classifiers are binary demonstration models. Reviews near boundary or with divergent predictions are categorized as Uncertain.")

        # Review Reply Generator
        st.divider()
        st.subheader("Smart Review Reply Generator")
        st.caption("Contextually generated response tailored to detected review aspects. You can edit the text directly before copying.")

        replies = st.session_state.get("active_replies", generate_review_replies(res["final_sentiment"], res["details"]))

        tone_options = list(replies.keys())
        selected_tone = st.segmented_control("Response Style", options=tone_options, default=tone_options[0])

        target_reply = replies.get(selected_tone, replies["Professional"])

        # Editable Text Area for reply
        edited_reply = st.text_area(
            label="Generated Customer Reply (Editable)",
            value=target_reply,
            height=110,
        )

        rep_col1, rep_col2, _ = st.columns([1.5, 1.5, 3])
        with rep_col1:
            if st.button("🔄 Regenerate Reply", width="stretch"):
                st.session_state["variation_seed"] += 1
                st.rerun()

        st.caption("One-Click Copy Snippet:")
        st.code(edited_reply, language="text")

        st.info(
            "Generator Engine: Rule-based contextual reply engine using keyword-based entity extraction. "
            "Independent of neural classifiers (not an external LLM).",
            title="Engine Disclosure",
            icon=":material/smart_toy:",
        )


# --- View 3: Batch Analysis ---
def batch_view():
    st.title("Batch Review Analysis")
    st.caption("Upload a CSV file of customer reviews to run automated bulk predictions and export enriched results.")

    # File Uploader
    uploaded_file = st.file_uploader("Upload CSV File (must include a 'review' column)", type=["csv"])

    # Sample CSV Download
    sample_csv_data = (
        'review,sentiment\n'
        '"The product quality is excellent and delivery was very fast.",positive\n'
        '"The product arrived late and the packaging was damaged.",negative\n'
        '"Customer support answered immediately and solved my problem.",positive\n'
        '"Overpriced and terrible quality, broke within a week.",negative\n'
        '"The movie was a masterpiece with great acting and music.",positive\n'
        '"Boring movie, poor screenplay and wooden acting.",negative\n'
    )
    st.download_button(
        label="📥 Download Sample CSV Template",
        data=sample_csv_data,
        file_name="sample_batch_reviews.csv",
        mime="text/csv",
    )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            col_map = {c: c.strip().lower() for c in batch_df.columns}
            batch_df.rename(columns=col_map, inplace=True)

            if "review" not in batch_df.columns:
                st.error("Uploaded CSV must contain a 'review' column.", icon=":material/error:")
                return

            st.success(f"CSV Loaded: {len(batch_df)} records detected.", icon=":material/check_circle:")
            st.dataframe(batch_df.head(5))

            if st.button("⚡ Run Batch Analysis", type="primary"):
                with st.spinner("Executing batch predictions with MLP and LSTM..."):
                    tfidf_vec = pipeline_artifacts["tfidf_vectorizer"]
                    mlp_model = pipeline_artifacts["mlp_model"]
                    tokenizer = pipeline_artifacts["tokenizer"]
                    lstm_model = pipeline_artifacts["lstm_model"]
                    max_len = pipeline_artifacts["max_seq_len"]

                    clean_reviews = batch_df["review"].fillna("").astype(str).tolist()

                    # Vectorized MLP
                    mlp_feats = tfidf_vec.transform(clean_reviews).toarray()
                    mlp_probs = mlp_model.predict(mlp_feats, verbose=0).flatten()

                    # Vectorized LSTM
                    seq_feats = pad_sequences(
                        tokenizer.texts_to_sequences(clean_reviews),
                        maxlen=max_len,
                        padding="pre",
                        truncating="post",
                    )
                    lstm_probs = lstm_model.predict(seq_feats, verbose=0).flatten()

                    mlp_labels = ["Positive" if p >= 0.5 else "Negative" for p in mlp_probs]
                    lstm_labels = ["Positive" if p >= 0.5 else "Negative" for p in lstm_probs]

                    final_sentiments = []
                    confidence_scores = []
                    agreements = []

                    for mp, lp, ml, ll in zip(mlp_probs, lstm_probs, mlp_labels, lstm_labels):
                        agree = (ml == ll)
                        agreements.append("Yes" if agree else "No")
                        avg_p = (mp + lp) / 2.0
                        if not agree and (abs(mp - 0.5) < 0.20 or abs(lp - 0.5) < 0.20):
                            final_sentiments.append("Neutral / Uncertain")
                            confidence_scores.append(round(0.50 + abs(avg_p - 0.50), 3))
                        elif 0.42 < avg_p < 0.58:
                            final_sentiments.append("Neutral / Uncertain")
                            confidence_scores.append(round(0.50 + abs(avg_p - 0.50), 3))
                        elif avg_p >= 0.58:
                            final_sentiments.append("Positive")
                            confidence_scores.append(round(avg_p, 3))
                        else:
                            final_sentiments.append("Negative")
                            confidence_scores.append(round(1.0 - avg_p, 3))

                    results_df = batch_df.copy()
                    results_df["mlp_prediction"] = mlp_labels
                    results_df["lstm_prediction"] = lstm_labels
                    results_df["final_sentiment"] = final_sentiments
                    results_df["confidence"] = confidence_scores
                    results_df["agreement"] = agreements

                    st.session_state["batch_results"] = results_df

        except Exception as e:
            st.error(f"Error processing CSV: {e}", icon=":material/error:")

    if "batch_results" in st.session_state:
        res_df = st.session_state["batch_results"]
        st.divider()
        st.subheader("Batch Prediction Results")

        total = len(res_df)
        pos = (res_df["final_sentiment"] == "Positive").sum()
        neg = (res_df["final_sentiment"] == "Negative").sum()
        unc = (res_df["final_sentiment"] == "Neutral / Uncertain").sum()

        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric("Total Reviews", total)
        bc2.metric("Positive", f"{pos} ({pos/total*100:.1f}%)")
        bc3.metric("Negative", f"{neg} ({neg/total*100:.1f}%)")
        bc4.metric("Uncertain", f"{unc} ({unc/total*100:.1f}%)")

        st.dataframe(res_df)

        # Ground truth evaluation if sentiment column exists
        if "sentiment" in res_df.columns:
            st.markdown("##### Ground-Truth Evaluation on Uploaded Data")
            true_clean = res_df["sentiment"].astype(str).str.strip().str.lower()
            if set(true_clean.unique()).issubset({"positive", "negative", "1", "0"}):
                true_bin = true_clean.replace({"positive": 1, "negative": 0, "1": 1, "0": 0}).astype(int)
                mlp_bin = (res_df["mlp_prediction"] == "Positive").astype(int)
                lstm_bin = (res_df["lstm_prediction"] == "Positive").astype(int)

                mlp_acc = accuracy_score(true_bin, mlp_bin)
                lstm_acc = accuracy_score(true_bin, lstm_bin)

                ec1, ec2 = st.columns(2)
                ec1.metric("MLP Batch Accuracy", f"{mlp_acc * 100:.1f}%")
                ec2.metric("LSTM Batch Accuracy", f"{lstm_acc * 100:.1f}%")

        # Download button
        csv_export = res_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Results as CSV",
            data=csv_export,
            file_name="sentiment_batch_results.csv",
            mime="text/csv",
            type="primary",
        )


# --- View 4: Model Performance ---
def performance_view():
    st.title("Model Performance & Architecture Comparison")
    st.caption("Empirical performance metrics evaluated on stratified held-out test data.")

    mlp_m = pipeline_artifacts["mlp_metrics"]
    lstm_m = pipeline_artifacts["lstm_metrics"]

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### Model A: MLP Classifier (TF-IDF)")
            st.metric("Test Accuracy", f"{mlp_m['accuracy'] * 100:.1f}%")
            st.metric("Precision", f"{mlp_m['precision'] * 100:.1f}%")
            st.metric("Recall", f"{mlp_m['recall'] * 100:.1f}%")
            st.metric("F1-Score", f"{mlp_m['f1'] * 100:.1f}%")

    with col2:
        with st.container(border=True):
            st.markdown("#### Model B: LSTM Classifier (Sequence)")
            st.metric("Test Accuracy", f"{lstm_m['accuracy'] * 100:.1f}%")
            st.metric("Precision", f"{lstm_m['precision'] * 100:.1f}%")
            st.metric("Recall", f"{lstm_m['recall'] * 100:.1f}%")
            st.metric("F1-Score", f"{lstm_m['f1'] * 100:.1f}%")

    st.divider()

    # Visualizations
    vc1, vc2 = st.columns(2)
    with vc1:
        st.subheader("Metrics Comparison")
        fig, ax = plt.subplots(figsize=(5, 3.8))
        metrics_names = ["Accuracy", "Precision", "Recall", "F1"]
        mlp_scores = [mlp_m["accuracy"], mlp_m["precision"], mlp_m["recall"], mlp_m["f1"]]
        lstm_scores = [lstm_m["accuracy"], lstm_m["precision"], lstm_m["recall"], lstm_m["f1"]]

        x = np.arange(len(metrics_names))
        width = 0.35

        r1 = ax.bar(x - width/2, mlp_scores, width, label="MLP (TF-IDF)", color="#4f46e5")
        r2 = ax.bar(x + width/2, lstm_scores, width, label="LSTM (Seq)", color="#10b981")

        ax.set_ylabel("Score (0 to 1)")
        ax.set_ylim(0, 1.15)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.legend(loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        for rect in r1:
            h = rect.get_height()
            ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)
        for rect in r2:
            h = rect.get_height()
            ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with vc2:
        st.subheader("Confusion Matrices (Held-out Test Split)")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3.5))

        cm_mlp = mlp_m["confusion_matrix"]
        cm_lstm = lstm_m["confusion_matrix"]

        def render_cm(ax, cm, title, cmap):
            ax.imshow(cm, cmap=cmap)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Neg", "Pos"])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Neg", "Pos"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black", fontweight="bold")

        render_cm(ax1, cm_mlp, "MLP Confusion Matrix", plt.cm.Blues)
        render_cm(ax2, cm_lstm, "LSTM Confusion Matrix", plt.cm.Greens)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Held-out Test Predictions Detail")
    st.write(f"Evaluated on {pipeline_artifacts['test_count']} held-out test reviews (25% stratified test split).")
    st.dataframe(pipeline_artifacts["test_eval_df"])

    st.info(
        "Important Academic Note: Accuracy values above reflect held-out test evaluations from the balanced 72-sample dataset. "
        "High accuracy on small demonstration corpora does not guarantee generalization to out-of-domain reviews.",
        title="Generalization Notice",
        icon=":material/school:",
    )


# --- View 5: Dataset Explorer ---
def explorer_view():
    st.title("Dataset Explorer")
    st.caption("Inspect class distribution, text lengths, and optional custom dataset retraining.")

    df = pd.DataFrame(st.session_state["active_data"])

    # Summary Metrics
    c1, c2, c3, c4 = st.columns(4)
    pos_num = (df["sentiment"].astype(str).str.lower() == "positive").sum()
    neg_num = (df["sentiment"].astype(str).str.lower() == "negative").sum()

    c1.metric("Total Reviews", len(df))
    c2.metric("Positive Samples", pos_num)
    c3.metric("Negative Samples", neg_num)
    c4.metric("Active Dataset", st.session_state["dataset_source"].split("(")[0].strip())

    st.divider()

    # Visualizations
    dc1, dc2 = st.columns(2)
    with dc1:
        st.subheader("Class Distribution")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(
            [pos_num, neg_num],
            labels=[f"Positive ({pos_num})", f"Negative ({neg_num})"],
            colors=["#16a34a", "#dc2626"],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        ax.axis("equal")
        st.pyplot(fig)
        plt.close(fig)

    with dc2:
        st.subheader("Review Length Distribution")
        lengths = df["review"].astype(str).apply(len)
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        ax.hist(lengths, bins=12, color="#6366f1", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Character Count")
        ax.set_ylabel("Sample Count")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Filterable Data Table
    st.subheader("Review Data Table")
    filt = st.segmented_control("Filter by Sentiment", options=["All", "Positive", "Negative"], default="All")
    filtered = df.copy()
    if filt != "All":
        filtered = filtered[filtered["sentiment"].astype(str).str.lower() == filt.lower()]
    st.dataframe(filtered)

    st.divider()

    # Optional Custom CSV Retraining
    st.subheader("Retrain on Custom Dataset (Optional)")
    st.write("Upload a custom CSV file containing `review` and `sentiment` columns to retrain both models.")

    custom_file = st.file_uploader("Upload Retraining CSV", type=["csv"], key="custom_train_file")
    if custom_file is not None:
        try:
            udf = pd.read_csv(custom_file)
            udf.columns = [c.strip().lower() for c in udf.columns]
            if "review" in udf.columns and "sentiment" in udf.columns:
                udf["sentiment"] = udf["sentiment"].astype(str).str.strip().str.lower()
                udf["sentiment"] = udf["sentiment"].replace({"1": "positive", "0": "negative"})
                if set(udf["sentiment"].unique()).issubset({"positive", "negative"}) and len(udf) >= 8:
                    st.success(f"Valid dataset detected ({len(udf)} records).", icon=":material/check_circle:")
                    if st.button("🔄 Retrain Models on This CSV", type="primary"):
                        st.session_state["active_data"] = udf[["review", "sentiment"]].to_dict(orient="records")
                        st.session_state["dataset_source"] = f"Custom CSV ({custom_file.name}, {len(udf)} records)"
                        st.cache_resource.clear()
                        st.rerun()
                else:
                    st.error("CSV must contain both 'positive' and 'negative' labels with at least 8 total rows.", icon=":material/error:")
            else:
                st.error("Uploaded CSV must contain 'review' and 'sentiment' columns.", icon=":material/error:")
        except Exception as ex:
            st.error(f"Error parsing file: {ex}", icon=":material/error:")

    if st.session_state["dataset_source"] != "Built-in (72 reviews, balanced across 6 domains)":
        if st.button("↺ Reset to Built-in Dataset"):
            st.session_state["active_data"] = list(BUILTIN_REVIEWS)
            st.session_state["dataset_source"] = "Built-in (72 reviews, balanced across 6 domains)"
            st.cache_resource.clear()
            st.rerun()


# --- View 6: Prediction History ---
def history_view():
    st.title("Prediction History")
    st.caption("Session-based log of analyzed reviews with filtering and CSV export capabilities.")

    history = st.session_state["history"]

    if len(history) == 0:
        st.info("No reviews have been analyzed in this session yet.", title="History Empty", icon=":material/history:")
    else:
        # History Table
        hist_df = pd.DataFrame(history)

        st.subheader("Filter and Search")
        filter_opt = st.segmented_control("Filter by Sentiment", options=["All", "Positive", "Negative", "Neutral / Uncertain"], default="All")

        filtered_hist = hist_df.copy()
        if filter_opt != "All":
            filtered_hist = filtered_hist[filtered_hist["final_sentiment"] == filter_opt]

        st.dataframe(filtered_hist)

        hc1, hc2, _ = st.columns([1.5, 1.5, 3])
        with hc1:
            csv_data = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export History to CSV",
                data=csv_data,
                file_name=f"sentiment_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary",
            )
        with hc2:
            with st.popover("Clear History"):
                st.write("Are you sure you want to clear session prediction history?")
                if st.button("Confirm Clear", type="primary"):
                    st.session_state["history"] = []
                    st.rerun()


# --- View 7: About Project ---
def about_view():
    st.title("About SentimentAI")
    st.caption("Architectural specifications, theory, mathematical foundations, and viva guide.")

    st.markdown("""
    ### Executive Overview
    **SentimentAI** is an advanced Natural Language Processing (NLP) deep learning application that compares 
    feedforward (**MLP**) and recurrent (**LSTM**) neural architectures for customer review classification and generates 
    contextual business replies.
    """)

    st.divider()

    st.subheader("Neural Network Architectures")
    ac1, ac2 = st.columns(2)
    with ac1:
        with st.container(border=True):
            st.markdown("##### 1. Multilayer Perceptron (MLP)")
            st.markdown("""
            - **Input Features:** Term Frequency-Inverse Document Frequency (TF-IDF) bigram vectors (max 500 features).
            - **Hidden Layer:** Dense layer with 32 units, Rectified Linear Unit activation:
              $$h = \\text{ReLU}(W_1 x + b_1) = \\max(0, W_1 x + b_1)$$
            - **Output Layer:** Dense layer with 1 unit, Sigmoid activation:
              $$\\hat{y} = \\sigma(W_2 h + b_2) = \\frac{1}{1 + e^{-(W_2 h + b_2)}}$$
            - **Loss Function:** Binary Cross-Entropy with Adam optimizer ($\alpha=0.008$).
            - **Core Trait:** Evaluates global word importance regardless of word order.
            """)

    with ac2:
        with st.container(border=True):
            st.markdown("##### 2. Long Short-Term Memory (LSTM)")
            st.markdown("""
            - **Input Features:** Sequential integer tokens padded to $T=30$.
            - **Embedding Layer:** 500-word vocabulary mapped into dense 32-dimensional continuous vectors.
            - **Recurrent Layer:** 32 LSTM memory cells utilizing gated state updates:
              - **Forget Gate:** $f_t = \\sigma(W_f [h_{t-1}, x_t] + b_f)$
              - **Input Gate:** $i_t = \\sigma(W_i [h_{t-1}, x_t] + b_i)$
              - **Cell State:** $C_t = f_t \\odot C_{t-1} + i_t \\odot \\tanh(W_c [h_{t-1}, x_t] + b_c)$
              - **Output Gate:** $o_t = \\sigma(W_o [h_{t-1}, x_t] + b_o)$
              - **Hidden State:** $h_t = o_t \\odot \\tanh(C_t)$
            - **Core Trait:** Retains sequential word order and captures long-term dependencies.
            """)

    st.divider()

    st.subheader("🎓 College Viva & Technical Q&A")

    with st.expander("Q1: Why does MLP use TF-IDF while LSTM uses Word Embeddings?"):
        st.write(
            "An MLP requires a fixed-dimensional numerical feature vector as input. TF-IDF compresses a text document into a fixed "
            "vector representing relative word importance. An LSTM, however, requires an ordered sequence of tokens over time. "
            "An Embedding layer converts each discrete word index into a 32-dimensional continuous vector, preserving word order and temporal context."
        )

    with st.expander("Q2: How is data leakage prevented during vectorization and tokenization?"):
        st.write(
            "The `TfidfVectorizer` and the `Tokenizer` are fitted exclusively on the training split (`X_train_raw`). "
            "The held-out test split (`X_test_raw`) is only transformed using the already-fitted vocabulary. Fitting on the entire "
            "dataset prior to splitting would leak test vocabulary and inverse-frequency distributions into training."
        )

    with st.expander("Q3: How does the application handle neutral or ambiguous reviews?"):
        st.write(
            "Because both MLP and LSTM are trained on binary positive/negative labels, a transparent decision-boundary rule is used: "
            "if the average prediction probability lies between 42% and 58%, or if the two models produce conflicting predictions "
            "with modest confidence, the verdict is classified as 'Neutral / Uncertain' rather than forcing a false positive or negative."
        )

    with st.expander("Q4: Is the reply generator an LLM or an external API?"):
        st.write(
            "No. The reply generator is a rule-based contextual template engine that extracts specific feedback aspects (e.g. fast delivery, "
            "damaged packaging, battery life) and formats customized responses. It does not call any paid API or external service, ensuring zero cost and instant execution."
        )

    with st.expander("Q5: What are the main limitations and future scope?"):
        st.write(
            "Limitations: The built-in demonstration dataset is small (72 reviews), so highly specialized slang or rare out-of-vocabulary words "
            "may have lower accuracy. Future scope includes fine-tuning lightweight transformer models (such as DistilBERT), expanding training corpora, "
            "and adding multilingual support."
        )


# ==============================================================================
# 8. NAVIGATION SETUP (Streamlit 1.66+ Navigation System)
# ==============================================================================
pg = st.navigation([
    st.Page(dashboard_view, title="Dashboard", icon=":material/dashboard:"),
    st.Page(analyze_view, title="Analyze Review", icon=":material/search:"),
    st.Page(batch_view, title="Batch Analysis", icon=":material/dataset:"),
    st.Page(performance_view, title="Model Performance", icon=":material/monitoring:"),
    st.Page(explorer_view, title="Dataset Explorer", icon=":material/table_chart:"),
    st.Page(history_view, title="Prediction History", icon=":material/history:"),
    st.Page(about_view, title="About Project", icon=":material/info:"),
])

# Sidebar Branding
with st.sidebar:
    st.markdown("### 🧠 SentimentAI")
    st.caption("AI Sentiment & Smart Reply Engine")
    st.divider()

pg.run()
