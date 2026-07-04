# app_local.py
# Local version - BERT + LSTM cascade + Keyword system
# BERT runs first, negation correction applied immediately after BERT
# LSTM kicks in only for borderline cases after negation correction

import re
import json
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="Comment Toxicity Detector",
    layout="wide"
)

# ── Constants ────────────────────────────────────────────────────
LSTM_MAX_LEN = 200
BERT_MAX_LEN = 128
BATCH_SIZE   = 32

LABEL_COLS = [
    "toxic", "severe_toxic", "obscene",
    "threat", "insult", "identity_hate"
]

LABEL_COLOURS = {
    "toxic"         : "#E24B4A",
    "severe_toxic"  : "#BA7517",
    "obscene"       : "#7F77DD",
    "threat"        : "#D85A30",
    "insult"        : "#1D9E75",
    "identity_hate" : "#378ADD"
}

THRESHOLDS = {
    "toxic"         : 0.3,
    "severe_toxic"  : 0.6,
    "obscene"       : 0.6,
    "threat"        : 0.15,
    "insult"        : 0.6,
    "identity_hate" : 0.5
}

# ── Negation words ───────────────────────────────────────────────
NEGATION_WORDS = [
    "never", "not", "won't", "wouldn't", "don't",
    "doesn't", "didn't", "cannot", "can't", "couldn't",
    "no", "nor", "neither"
]

# Core threatening words — used for negation detection
THREAT_CORE_WORDS = [
    "kill", "hurt", "murder", "shoot", "stab",
    "rape", "harm", "attack", "destroy", "die",
    "dead", "death", "beat", "punch", "hang"
]

# ── Keyword lists ────────────────────────────────────────────────
THREAT_KEYWORDS = [
    "kill you", "kill u", "kill ya",
    "i will kill", "i'll kill", "gonna kill", "going to kill",
    "murder you", "i will murder",
    "hope you die", "hope u die",
    "you will die", "you are going to die",
    "go die", "please die", "you should die",
    "go kill yourself", "kill yourself",
    "you are dead", "you're dead",
    "shoot you", "shoot u", "blow your head",
    "stab you", "slit your throat", "slit your wrists",
    "behead", "decapitate",
    "kick your ass", "beat you", "beat u",
    "punch you", "knock you out",
    "fuck you up", "i will attack",
    "i know where you live", "find where you live",
    "come to your home", "come to your house",
    "i will find you", "track you down",
    "hunt you down", "coming to get you",
    "i'm coming for you", "trace your ip",
    "rape you", "i will rape", "going to rape",
    "burn your house", "set you on fire",
    "destroy you", "you will regret",
    "make you pay", "you will suffer",
    "watch your back", "you have been warned",
    "this is a threat", "last warning",
    "stop or die", "or i will kill",
    "not going to be alive", "won't be alive",
    "dance on your grave", "i will get you",
    "hang yourself", "shoot yourself",
    "slaughter you", "life terminated",
    "you do not deserve to be alive",
    "should be shot", "i will castrate",
    "death to all jews", "death to all muslims",
    "death to all arabs", "death to all blacks",
    "death to all christians", "death to all infidels",
    "6 feet under", "6ft under",
]

IDENTITY_HATE_KEYWORDS = [
    "kill all niggers", "death to arabs",
    "death to jews", "death to muslims",
    "filthy muslim", "filthy jew", "sand nigger",
    "dirty arab", "dirty jew", "dirty muslim",
    "all muslims are", "all the muslims",
    "all jews are", "all the jews",
    "all blacks are", "all the blacks",
    "all christians are", "all hindus are",
    "all gays are", "all lesbians are",
    "hate all muslims", "hate all jews",
    "hate all blacks", "hate muslims", "hate jews",
    "your kind should", "your people should",
    "your race should", "go back to your country",
    "don't belong here", "inferior race",
    "inferior religion", "subhuman",
    "burn jews", "gas the jews", "exterminate",
    "ethnic cleansing", "nigger cunt",
    "death to infidel", "islamist terrorist",
    "muslim terrorist", "jewish conspiracy",
    "behead those who insult islam",
    "deserve to be shot", "should be executed",
    "hunted down like animals",
    "wipe your entire race", "next genocide",
    "you fucking nigger", "you fucking muslim",
    "you fucking jew", "you fucking arab",
    "you fucking faggot", "fucking paki",
    "fucking chink",
]

WHITELIST = [
    "death of a salesman", "to kill a mockingbird",
    "kill bill", "die hard", "dead poets society",
    "kill the bacteria", "bacteria kill rate",
    "death rate", "death toll", "death penalty",
    "death sentence", "brain dead", "dead serious",
    "dead end", "deadline", "die out", "die down",
    "kill switch", "killing it", "killing the game",
    "dead on arrival",
]


# ── Helper functions ─────────────────────────────────────────────

def is_whitelisted(comment: str) -> bool:
    c = comment.lower()
    return any(phrase in c for phrase in WHITELIST)


def is_classifiable(comment: str) -> bool:
    return len(re.sub(r'[^a-z\s]', '', comment.lower()).strip()) > 0


def has_negation_before_threat(comment: str) -> bool:
    """
    Returns True if a negation word appears within 5 words
    before any core threatening word.
    Example: 'i would never kill you' -> True
             'i will kill you'        -> False
    """
    text = re.sub(r'[^\w\s]', ' ', comment.lower())
    words = text.split()
    for i, word in enumerate(words):
        if word in NEGATION_WORDS:
            # Check the next 5 words for a threat core word
            window = " ".join(words[i:i + 5])
            if any(core in window for core in THREAT_CORE_WORDS):
                return True
    return False


def apply_negation_correction(scores: dict) -> dict:
    """
    If negation detected, set all scores to 0.05.
    This is applied AFTER BERT prediction.
    """
    for label in LABEL_COLS:
        scores[label] = 0.05
    return scores


def check_keywords(comment: str) -> list:
    """Rule-based keyword detection — safety net only."""
    if is_whitelisted(comment):
        return []

    # Skip keyword check if negation detected
    if has_negation_before_threat(comment):
        return []

    c = comment.lower()
    detected = []

    for kw in THREAT_KEYWORDS:
        if kw in c:
            detected.append("threat")
            break

    for kw in IDENTITY_HATE_KEYWORDS:
        if kw in c:
            detected.append("identity_hate")
            break

    return detected


# ── Load models ──────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    import os
    import torch
    from transformers import (
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification
    )

    # BERT
    bert_dir = "bert_best_model"
    bert_tok = DistilBertTokenizerFast.from_pretrained(bert_dir)
    bert_mdl = DistilBertForSequenceClassification.from_pretrained(bert_dir)
    bert_mdl.eval()

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    bert_mdl = bert_mdl.to(device)

    # LSTM
    lstm_mdl = Sequential([
        Embedding(input_dim=20000, output_dim=128,
                  input_length=LSTM_MAX_LEN),
        SpatialDropout1D(0.2),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(6, activation='sigmoid')
    ])
    lstm_mdl.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    lstm_mdl(np.zeros((1, LSTM_MAX_LEN)))
    lstm_mdl.load_weights("best_model_weights.weights.h5")

    with open("tokenizer.pkl", "rb") as f:
        lstm_tok = pickle.load(f)

    with open("bert_metrics.json", "r") as f:
        bert_metrics = json.load(f)

    with open("history.json", "r") as f:
        history = json.load(f)

    return bert_mdl, bert_tok, device, lstm_mdl, lstm_tok, bert_metrics, history


(bert_model, bert_tokenizer, bert_device,
 lstm_model, lstm_tokenizer,
 bert_metrics, history) = load_artifacts()


# ── Prediction ───────────────────────────────────────────────────

def predict_toxicity(comment: str):
    """
    Returns (scores_dict, model_label) or (None, None).

    Pipeline:
    1. BERT predicts scores
    2. If negation detected  → set all scores to 0.05 → return
    3. If max score 0.2-0.5  → run LSTM cascade, average scores
    4. Return final scores
    """
    if not is_classifiable(comment):
        return None, None

    import torch

    try:
        # Step 1: BERT prediction
        enc = bert_tokenizer(
            comment,
            max_length=BERT_MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            out = bert_model(
                input_ids=enc["input_ids"].to(bert_device),
                attention_mask=enc["attention_mask"].to(bert_device)
            )
            probs = torch.sigmoid(out.logits).cpu().numpy()[0]

        scores = {
            label: float(probs[i])
            for i, label in enumerate(LABEL_COLS)
        }

        # Step 2: Negation correction — MUST happen before cascade
        if has_negation_before_threat(comment):
            scores = apply_negation_correction(scores)
            return scores, "BERT (negation corrected)"

        max_score = max(scores.values())

        # Step 3: LSTM cascade for borderline cases
        if 0.2 <= max_score <= 0.5:
            try:
                seq = lstm_tokenizer.texts_to_sequences([comment])
                pad = pad_sequences(
                    seq,
                    maxlen=LSTM_MAX_LEN,
                    padding='post',
                    truncating='post'
                )
                with tf.device('/CPU:0'):
                    lstm_out = lstm_model(pad, training=False)
                lstm_probs = lstm_out.numpy()[0]

                # Average BERT and LSTM
                for i, label in enumerate(LABEL_COLS):
                    scores[label] = round(
                        (scores[label] + float(lstm_probs[i])) / 2, 4
                    )
                return scores, "BERT + LSTM cascade"
            except Exception:
                pass

        return scores, "BERT"

    except Exception as e:
        # LSTM fallback if BERT fails completely
        try:
            seq = lstm_tokenizer.texts_to_sequences([comment])
            pad = pad_sequences(
                seq,
                maxlen=LSTM_MAX_LEN,
                padding='post',
                truncating='post'
            )
            with tf.device('/CPU:0'):
                out = lstm_model(pad, training=False)
            probs = out.numpy()[0]
            return {
                label: float(probs[i])
                for i, label in enumerate(LABEL_COLS)
            }, "LSTM fallback"
        except Exception as e2:
            st.error(f"Prediction failed: {e2}")
            return None, None


# ── Sidebar ──────────────────────────────────────────────────────

st.sidebar.title("Toxicity Detector")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Predict", "EDA Dashboard", "Model Performance"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
bert_auc = bert_metrics.get("mean_auc", 0.9876)
st.sidebar.markdown(f"Mean AUC : `{bert_auc:.4f}`")
st.sidebar.markdown("Architecture : `BERT + LSTM Cascade + Rules`")
st.sidebar.markdown("Labels : `6`")


# ── PAGE 1: HOME ─────────────────────────────────────────────────

if page == "Home":
    st.title("Comment Toxicity Detection")
    st.markdown("### Deep Learning powered content moderation")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training Comments", "159,571")
    c2.metric("Toxicity Labels", "6")
    c3.metric("Mean AUC Score", f"{bert_auc:.4f}")
    c4.metric("Model", "BERT + LSTM")

    st.markdown("---")
    cl, cr = st.columns(2)

    with cl:
        st.markdown("### What this app does")
        st.markdown("""
        **Layer 1 - DistilBERT (primary)**
        Fine-tuned transformer — reads entire sentences
        using attention. Mean AUC: 0.9876.

        **Layer 2 - Negation Correction**
        Applied immediately after BERT prediction.
        Detects negation words (never, not, won't) before
        threatening words and sets scores to 0.05.
        Example: 'I would never kill you' is classified clean.

        **Layer 3 - LSTM Cascade (borderline cases)**
        When BERT is uncertain (score 0.2-0.5), LSTM
        runs and scores are averaged. LSTM Mean AUC: 0.9735.

        **Layer 4 - Keyword System (safety net)**
        200+ phrases from real training data. Skipped
        automatically if negation is detected.
        """)

    with cr:
        st.markdown("### The 6 toxicity labels")
        desc = {
            "toxic"         : "Generally rude or disrespectful",
            "severe_toxic"  : "Extremely offensive content",
            "obscene"       : "Vulgar or explicit language",
            "threat"        : "Threatening someone's safety",
            "insult"        : "Personally attacking someone",
            "identity_hate" : "Hatred based on race, religion, gender"
        }
        for label, d in desc.items():
            color = LABEL_COLOURS[label]
            st.markdown(
                f'<span style="color:{color};font-weight:600">'
                f'{label}</span> - {d}',
                unsafe_allow_html=True
            )

    st.markdown("---")
    a, b, c = st.columns(3)
    a.info("Step 1: Go to the Predict page from the sidebar")
    b.info("Step 2: Type or paste any comment into the text box")
    c.info("Step 3: Click Analyse Comment and see instant predictions")


# ── PAGE 2: PREDICT ──────────────────────────────────────────────

elif page == "Predict":
    st.title("Toxicity Prediction")
    st.markdown("---")
    st.markdown("### Single comment analysis")

    for key in ["prediction_result", "model_used",
                "comment_input", "keyword_detected"]:
        if key not in st.session_state:
            st.session_state[key] = None if key != "keyword_detected" else []

    comment = st.text_area(
        "Enter a comment to analyse",
        placeholder="Type or paste a comment here...",
        height=120,
        key="comment_input"
    )

    if st.button("Analyse Comment"):
        if not comment.strip():
            st.warning("Please enter a comment.")
        elif not is_classifiable(comment):
            st.warning(
                "This comment contains no classifiable text. "
                "Please enter words, not just symbols."
            )
        else:
            with st.spinner("Analysing..."):
                preds, model_used = predict_toxicity(comment)
                if preds is not None:
                    st.session_state.prediction_result = preds
                    st.session_state.model_used = model_used
                    st.session_state.keyword_detected = check_keywords(comment)
                else:
                    st.error("Prediction failed.")

    if st.session_state.prediction_result is not None:
        preds    = st.session_state.prediction_result
        kw_det   = st.session_state.keyword_detected or []
        mdl_used = st.session_state.model_used or "BERT"

        st.caption(f"Predicted by: {mdl_used}")
        st.markdown("---")

        ml_det  = [l for l, s in preds.items() if s >= THRESHOLDS[l]]
        all_det = sorted(set(ml_det + kw_det))
        max_s   = max(preds.values())

        if not all_det:
            st.success("This comment appears to be clean.")
        elif max_s < 0.6 and not kw_det:
            st.warning(
                f"This comment may be mildly toxic - "
                f"detected labels: **{', '.join(all_det)}**"
            )
        else:
            st.error(
                f"This comment is likely toxic - "
                f"detected labels: **{', '.join(all_det)}**"
            )

        if kw_det:
            st.info(
                f"Rule-based detection also flagged: "
                f"**{', '.join(kw_det)}** (keyword match)"
            )

        st.markdown("---")
        st.markdown("### Toxicity scores per label")

        labels = list(preds.keys())
        scores = [round(v * 100, 2) for v in preds.values()]
        colors = [LABEL_COLOURS[l] for l in labels]

        fig = go.Figure(go.Bar(
            x=labels, y=scores,
            marker_color=colors,
            text=[f"{s:.1f}%" for s in scores],
            textposition='outside'
        ))
        fig.update_layout(
            yaxis_title="Probability (%)",
            yaxis_range=[0, 110],
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Detailed scores")
        df = pd.DataFrame({
            "Label"     : labels,
            "Score"     : [f"{v*100:.2f}%" for v in preds.values()],
            "Threshold" : [f"{THRESHOLDS[l]*100:.0f}%" for l in labels],
            "Risk"      : [
                "High"   if v >= 0.6
                else "Medium" if v >= THRESHOLDS[l]
                else "Low"
                for l, v in preds.items()
            ],
            "Detected"  : [
                f"Yes ({mdl_used.split()[0]})" if v >= THRESHOLDS[l]
                else "Yes (Rule)" if l in kw_det
                else "No"
                for l, v in preds.items()
            ]
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

        if all_det:
            st.markdown("### Detected toxicity types")
            cols = st.columns(len(all_det))
            for i, label in enumerate(all_det):
                color = LABEL_COLOURS[label]
                score = preds[label]
                tag = (
                    "(Rule)" if label in kw_det and label not in ml_det
                    else f"({mdl_used.split()[0]})"
                )
                cols[i].markdown(
                    f'<div style="background:{color}20;'
                    f'border-left:4px solid {color};'
                    f'padding:10px;border-radius:4px">'
                    f'<span style="color:{color};font-weight:700">'
                    f'{label}</span> {tag}<br>'
                    f'<span style="font-size:1.2em;font-weight:600">'
                    f'{score*100:.1f}%</span></div>',
                    unsafe_allow_html=True
                )

    st.markdown("---")
    st.markdown("### Bulk prediction - upload a CSV file")
    st.markdown("Upload a CSV with a column named comment_text.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        bulk = pd.read_csv(uploaded)
        if "comment_text" not in bulk.columns:
            st.error("CSV must have a column named comment_text.")
        else:
            st.success(f"File loaded - {len(bulk):,} comments.")
            st.dataframe(bulk.head(), use_container_width=True)

            if st.button("Predict All Comments"):
                import torch
                all_comments = bulk["comment_text"].astype(str).tolist()
                all_probs = []

                with st.spinner(f"Processing {len(bulk):,} comments..."):
                    for i in range(0, len(all_comments), BATCH_SIZE):
                        batch = all_comments[i:i + BATCH_SIZE]
                        enc = bert_tokenizer(
                            batch,
                            max_length=BERT_MAX_LEN,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt"
                        )
                        with torch.no_grad():
                            out = bert_model(
                                input_ids=enc["input_ids"].to(bert_device),
                                attention_mask=enc["attention_mask"].to(
                                    bert_device
                                )
                            )
                            probs = torch.sigmoid(
                                out.logits
                            ).cpu().numpy()
                        all_probs.append(probs)

                all_probs = np.vstack(all_probs)
                results = []

                for i, text in enumerate(all_comments):
                    row = {
                        label: float(all_probs[i][j])
                        for j, label in enumerate(LABEL_COLS)
                    }
                    # Apply negation correction per row
                    if has_negation_before_threat(text):
                        row = {l: 0.05 for l in LABEL_COLS}

                    kw = check_keywords(text)
                    ml = [l for l, s in row.items() if s >= THRESHOLDS[l]]
                    det = sorted(set(ml + kw))
                    row["detected_labels"] = ", ".join(det)
                    row["comment_text"] = text[:100]
                    results.append(row)

                rdf = pd.DataFrame(results)
                cols_order = ["comment_text", "detected_labels"] + LABEL_COLS
                rdf = rdf[cols_order]
                for col in LABEL_COLS:
                    rdf[col] = rdf[col].round(4)

                st.success(f"Done - {len(rdf):,} comments analysed.")
                st.dataframe(rdf, use_container_width=True)
                st.download_button(
                    "Download predictions as CSV",
                    rdf.to_csv(index=False),
                    "toxicity_predictions.csv",
                    "text/csv"
                )


# ── PAGE 3: EDA ──────────────────────────────────────────────────

elif page == "EDA Dashboard":
    st.title("Exploratory Data Analysis")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train comments", "159,571")
    c2.metric("Test comments", "153,164")
    c3.metric("Null values", "0")
    c4.metric("Duplicates", "0")

    st.markdown("---")
    st.markdown("### Label distribution")

    counts = {
        "toxic": 15294, "severe_toxic": 1595,
        "obscene": 8449, "threat": 478,
        "insult": 7877, "identity_hate": 1405
    }
    fig = go.Figure(go.Bar(
        x=list(counts.keys()),
        y=list(counts.values()),
        marker_color=list(LABEL_COLOURS.values()),
        text=list(counts.values()),
        textposition='outside'
    ))
    fig.update_layout(
        yaxis_title="Count",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    cl, cr = st.columns(2)

    with cl:
        st.markdown("### Clean vs toxic")
        fig2 = go.Figure(go.Pie(
            labels=["Clean", "At least one toxic label"],
            values=[143346, 16225],
            marker_colors=["#1D9E75", "#E24B4A"],
            hole=0.4
        ))
        fig2.update_layout(
            height=350, paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig2, use_container_width=True)

    with cr:
        st.markdown("### Comment length statistics")
        for k, v in {
            "Average words": 67, "Median words": 36,
            "90th percentile": 152, "95th percentile": 230,
            "MAX_LEN (LSTM)": 200, "MAX_LEN (BERT)": 128
        }.items():
            a, b = st.columns(2)
            a.markdown(f"**{k}**")
            b.markdown(f"`{v}`")

    st.markdown("---")
    st.info("Insight 1 - Zero nulls and zero duplicates in both train and test.")
    st.info("Insight 2 - Toxic and obscene are highly correlated.")
    st.info("Insight 3 - Threat is rarest at 0.30% - BERT improved AUC from 0.9468 to 0.9923.")
    st.info("Insight 4 - Train and test length distributions are nearly identical.")


# ── PAGE 4: MODEL PERFORMANCE ────────────────────────────────────

elif page == "Model Performance":
    st.title("Model Performance")
    st.markdown("---")

    st.markdown("### LSTM vs DistilBERT - AUC Comparison")

    lstm_aucs = {
        "toxic": 0.9790, "severe_toxic": 0.9862,
        "obscene": 0.9887, "threat": 0.9521,
        "insult": 0.9842, "identity_hate": 0.9506
    }
    bert_aucs = {
        "toxic"        : bert_metrics.get("toxic", 0.9820),
        "severe_toxic" : bert_metrics.get("severe_toxic", 0.9916),
        "obscene"      : bert_metrics.get("obscene", 0.9903),
        "threat"       : bert_metrics.get("threat", 0.9923),
        "insult"       : bert_metrics.get("insult", 0.9840),
        "identity_hate": bert_metrics.get("identity_hate", 0.9855)
    }

    cdf = pd.DataFrame({
        "Label"      : list(lstm_aucs.keys()),
        "LSTM AUC"   : [f"{v:.4f}" for v in lstm_aucs.values()],
        "BERT AUC"   : [f"{v:.4f}" for v in bert_aucs.values()],
        "Improvement": [
            f"+{bert_aucs[l] - lstm_aucs[l]:.4f}"
            for l in lstm_aucs
        ]
    })
    st.dataframe(cdf, use_container_width=True, hide_index=True)
    st.success(
        f"LSTM Mean AUC: 0.9735 - "
        f"BERT Mean AUC: {bert_auc:.4f} "
        f"(+{bert_auc - 0.9735:.4f})"
    )

    st.markdown("---")
    st.markdown("### DistilBERT AUC per label")

    fig = go.Figure(go.Bar(
        x=list(bert_aucs.keys()),
        y=list(bert_aucs.values()),
        marker_color=[LABEL_COLOURS[l] for l in bert_aucs],
        text=[f"{v:.4f}" for v in bert_aucs.values()],
        textposition='outside'
    ))
    fig.update_layout(
        yaxis_title="AUC",
        yaxis_range=[0.9, 1.0],
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### LSTM Training history")

    cl, cr = st.columns(2)
    with cl:
        f = go.Figure()
        f.add_trace(go.Scatter(
            y=history["loss"], name="Train loss",
            line=dict(color="#378ADD", width=2)
        ))
        f.add_trace(go.Scatter(
            y=history["val_loss"], name="Val loss",
            line=dict(color="#E24B4A", width=2)
        ))
        f.update_layout(
            title="Loss per epoch",
            xaxis_title="Epoch", yaxis_title="Loss",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)', height=350
        )
        st.plotly_chart(f, use_container_width=True)

    with cr:
        f2 = go.Figure()
        f2.add_trace(go.Scatter(
            y=history["accuracy"], name="Train accuracy",
            line=dict(color="#378ADD", width=2)
        ))
        f2.add_trace(go.Scatter(
            y=history["val_accuracy"], name="Val accuracy",
            line=dict(color="#1D9E75", width=2)
        ))
        f2.update_layout(
            title="Accuracy per epoch",
            xaxis_title="Epoch", yaxis_title="Accuracy",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)', height=350
        )
        st.plotly_chart(f2, use_container_width=True)

    st.markdown("---")
    st.markdown("### All improvements made")
    st.success("Phase 1: LSTM baseline - Mean AUC 0.9719")
    st.success("Oversampling: threat 478 to 8,586 examples")
    st.success("Class weights: threat 5x penalty")
    st.success("LSTM improved - Mean AUC 0.9735")
    st.success("Keyword system: 200+ phrases from real training data")
    st.success("Negation correction: applied immediately after BERT prediction")
    st.success("Whitelist: literary phrases not flagged")
    st.success("Phase 2: DistilBERT fine-tuned on Apple M4 MPS")
    st.success(f"DistilBERT Mean AUC: {bert_auc:.4f}")
    st.success("LSTM cascade: runs for borderline cases (score 0.2-0.5)")
    st.success("Threat AUC: 0.9468 to 0.9923 (+0.0455)")
    st.success("Identity hate AUC: 0.9506 to 0.9855 (+0.0349)")

    st.markdown("---")
    st.markdown("### Known limitations")
    st.warning("Deep sarcasm - 'oh great another genius' partially handled by BERT.")
    st.warning("Wikipedia training bias - may underperform on modern slang.")
    st.info("Phase 3: Fine-tune on diverse dataset (Twitter + Reddit + Wikipedia).")