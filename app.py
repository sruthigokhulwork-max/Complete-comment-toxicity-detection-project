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
    page_icon="🛡️",
    layout="wide"
)

max_len = 200

label_cols = [
    "toxic", "severe_toxic", "obscene",
    "threat", "insult", "identity_hate"
]

label_colours = {
    "toxic"         : "#E24B4A",
    "severe_toxic"  : "#BA7517",
    "obscene"       : "#7F77DD",
    "threat"        : "#D85A30",
    "insult"        : "#1D9E75",
    "identity_hate" : "#378ADD"
}

THRESHOLDS = {
    "toxic"         : 0.3,
    "severe_toxic"  : 0.3,
    "obscene"       : 0.3,
    "threat"        : 0.15,
    "insult"        : 0.3,
    "identity_hate" : 0.3
}

NEGATIONS = [
    "never", "not", "won't", "wouldn't",
    "don't", "doesn't", "didn't", "cannot",
    "can't", "no", "nor", "neither",
    "i would never", "i will never",
    "i have never", "i'd never",
    "would never", "will never",
    "could never", "couldn't"
]

THREAT_KEYWORDS = [
    "kill you", "kill u", "kill ya", "kill ur",
    "i will kill", "i'll kill", "im going to kill",
    "i am going to kill", "gonna kill", "going to kill",
    "should be killed", "i kill you", "ill kill",
    "i will kill you", "murder you", "i will murder",
    "going to murder", "i am going to murder",
    "i'm going to kill", "i'm going to murder",
    "must die", "shall die",
    "hope you die", "hope u die", "i hope you die",
    "why dont you die", "why don't you die",
    "you will die", "you are going to die",
    "ur gonna die", "go die", "die you", "die bitch",
    "please die", "you should die", "deserve to die",
    "go kill yourself", "kill yourself",
    "you are dead", "you're dead",
    "your death is near", "days are numbered",
    "dead by now", "you will be dead",
    "drop dead", "choke and die",
    "i want you dead", "want you dead",
    "rot in hell", "burn in hell",
    "die in a fire", "go die in a fire",
    "die in hell", "perish",
    "your end will come", "blood from your throat",
    "shoot you", "shoot u", "with my gun",
    "blow your head", "blow up your house",
    "blow up", "i will blow",
    "shoot you in the head", "brains splatter",
    "bomb your", "fire bomb", "petrol bomb",
    "shoot your",
    "stab you", "stab u", "stabbing knife",
    "cut you", "cut ur", "cut off your",
    "slit your throat", "slit your wrists",
    "rip off your", "cut your heart out",
    "cut you into little pieces",
    "cut your balls", "i will stab",
    "stab you while you sleep",
    "a knife for you", "behead",
    "decapitate", "i will behead",
    "slice you", "slash your",
    "kick your ass", "kick ur ass", "kick yo ass",
    "beat you", "beat u", "beat ur", "beat yo",
    "punch you", "punch your face", "punch your",
    "smack you", "smack ya",
    "bash your", "knock you out", "knock u out",
    "fuck you up", "fuck u up",
    "i'll attack", "i will attack", "ill attack",
    "beat you to a bloody pulp",
    "smash you into", "stomp on",
    "kick the shit out of you",
    "i will punch",
    "i know where you live", "know where you live",
    "know where u live", "find out where you live",
    "find where you live", "come to your home",
    "come to your house", "come find you",
    "i will find you", "i'll find you", "ill find you",
    "track you down", "track u down",
    "hunt you down", "hunt u down",
    "coming to get you", "i am coming",
    "i'm coming for you", "coming for you",
    "i know who you are", "trace your ip",
    "find your location", "find where you are",
    "i will find you in real life",
    "look over your shoulder",
    "find you in real life",
    "gonna come after", "coming after your",
    "i will return", "im coming after",
    "rape you", "rape ur", "rape your",
    "i will rape", "going to rape", "gonna rape",
    "sexually abuse", "rape your family",
    "rape your children", "rape your mom",
    "going to rape you",
    "burn your house", "burn down your house",
    "set you on fire", "i will set you on fire",
    "light you on fire", "fire bombed",
    "burn you", "burn alive",
    "destroy you", "i will destroy",
    "i will terrorise you", "terrorize you",
    "i will blow the hell out of you",
    "wreak havoc",
    "you will regret", "you're gonna regret",
    "make you pay", "you will pay",
    "you will suffer", "make you suffer",
    "i will make sure you suffer",
    "slowly and painfully",
    "watch you bleed", "make you bleed",
    "you will bleed",
    "watch your back", "you have been warned",
    "this is a threat", "yes this is a threat",
    "last warning", "stop or die", "or die",
    "or i will kill", "or you die",
    "your life is over", "your life will be over",
    "dance on your grave", "piss on your grave",
    "not going to be alive", "wont be alive",
    "won't be alive", "won't let you live",
    "brutally murder", "death is near",
    "impending death", "i am out to get",
    "out to get you", "i will get you",
    "gonna get you", "going to get you",
    "i'll get you", "you will be sorry",
    "my revenge", "revenge will be",
    "with a vengeance", "sodomize",
    "i will hurt", "going to hurt", "gonna hurt",
    "hang yourself", "go hang yourself",
    "hang you", "i will hang you", "hang your corpse",
    "shoot yourself", "please shoot yourself",
    "slaughter you", "i will slaughter",
    "you shall burn", "you and your family shall burn",
    "i will savagely attack",
    "i will finish your life",
    "life terminated", "have your life terminated",
    "i will arrange to have your life terminated",
    "you do not deserve to be alive",
    "should be shot",
    "i will castrate", "roast your", "castrate you",
    "going to come round to your",
    "petrol bomb through your", "stuff a petrol bomb",
    "i am going to track you down", "tracking you down",
    "i will track you down", "i'm going to track you down",
    "i am going to come", "im going to come to your",
    "i will come to your house",
    "going to come to your house",
    "ill come to your house", "i'll come to your house",
    "send someone to", "i know where you are",
    "hack your computer", "destroy your c drive",
    "i will hack your", "6 feet under",
    "6ft under", "leave you 6ft",
]

IDENTITY_HATE_KEYWORDS = [
    "kill all niggers", "death to arabs",
    "death to jews", "death to muslims",
    "death to musulmans", "filthy muslim",
    "filthy jew", "filthy pig",
    "sand nigger", "dirty arab",
    "dirty jew", "dirty muslim",
    "muslim piece of shit",
    "fucking nigger", "fucking muslim",
    "fucking jew", "fucking arab",
    "all muslims are", "all the muslims",
    "all jews are", "all the jews",
    "all blacks are", "all the blacks",
    "all christians are", "all hindus are",
    "all gays are", "all lesbians are",
    "all arabs are", "all the arabs",
    "hate all muslims", "hate all jews",
    "hate all blacks", "hate all christians",
    "hate muslims", "hate jews",
    "your kind should", "your people should",
    "your race should", "your religion should",
    "go back to your country",
    "go back to china", "go back to",
    "don't belong here", "inferior race",
    "inferior religion", "subhuman",
    "you people are", "people like you should",
    "burn jews", "burning jews",
    "gas the jews", "exterminate",
    "wipe out all", "ethnic cleansing",
    "white trash", "nigger cunt",
    "kill all the jews", "kill jews",
    "death to niggers", "niggers should die",
    "all niggers", "fucking niggers",
    "death to infidel", "death to the infidel",
    "islamist terrorist", "muslim terrorist",
    "jewish conspiracy", "jew conspiracy",
    "behead those who insult islam",
    "filthy muslims", "filthy christians",
    "deserve to be shot", "should be executed",
    "should be taken outside and shot",
    "deserve the worst kind of death",
    "your type should", "scum like you",
    "hunted down like animals",
    "wipe your entire race", "next genocide",
    "we will kill you guys again",
    "won't let your children alive",
    "all anti-semites should die",
    "all communists should be shot",
    "nazis like you deserve",
    "you fucking nigger", "you fucking muslim",
    "you fucking jew", "you fucking arab",
    "you fucking faggot", "dirty semite",
    "you semite", "jewland",
    "get out of here you jewish",
    "jew son of a bitch",
    "jewish son of a bitch",
    "fucking paki", "fucking chink",
    "you fucking queer",
]

THREAT_INTENT_WORDS = [
    "kill", "murder", "shoot", "stab", "hurt",
    "die", "dead", "death", "bomb", "rape",
    "destroy", "hunt", "burn", "hang", "cut",
    "slash", "bleed", "corpse", "grave", "execute",
    "eliminate", "torture", "suffer", "pain",
    "coming for", "get you", "watch your back",
    "find you", "track", "your blood", "make you pay"
]

IDENTITY_INTENT_WORDS = [
    "hate all", "hate every", "kill all",
    "death to", "all should die", "deserve to die",
    "go back", "don't belong", "inferior",
    "subhuman", "exterminate", "cleanse",
    "nigger", "faggot", "sand", "filthy",
    "terrorist", "vermin", "parasite",
    "your kind", "your race", "your people"
]


def has_nearby_negation(text, keyword, window=4):
    """Check negation within 4 words before keyword.
    Strips punctuation first to handle quoted text."""
    clean_text = re.sub(r'[^\w\s]', ' ', text)
    words = clean_text.split()
    keyword_words = keyword.split()
    kw_len = len(keyword_words)

    for i in range(len(words) - kw_len + 1):
        if words[i:i + kw_len] == keyword_words:
            start = max(0, i - window)
            surrounding = " ".join(words[start:i])
            for neg in NEGATIONS:
                if neg in surrounding:
                    return True
    return False


def check_keywords(comment):
    """
    Three-layer hybrid detection:
    1. Exact keyword match with negation check
    2. Intent word scoring (threshold=3)
    3. Identity hate keyword match
    Keywords extracted from all 478 real threat comments
    and identity hate comments in the training dataset.
    """
    comment_lower = comment.lower()
    detected_by_keywords = []

    # Layer 1: Exact keyword match for threat
    threat_found = False
    for keyword in THREAT_KEYWORDS:
        if keyword in comment_lower:
            if not has_nearby_negation(comment_lower, keyword):
                threat_found = True
                break

    # Layer 2: Intent word scoring for indirect threats
    if not threat_found:
        threat_hits = sum(
            1 for word in THREAT_INTENT_WORDS
            if word in comment_lower
        )
        if threat_hits >= 3:
            threat_found = True

    if threat_found:
        detected_by_keywords.append("threat")

    # Layer 3: Exact keyword match for identity_hate
    identity_found = False
    for keyword in IDENTITY_HATE_KEYWORDS:
        if keyword in comment_lower:
            identity_found = True
            break

    # Layer 4: Intent word scoring for identity_hate
    if not identity_found:
        identity_hits = sum(
            1 for word in IDENTITY_INTENT_WORDS
            if word in comment_lower
        )
        if identity_hits >= 2:
            identity_found = True

    if identity_found:
        detected_by_keywords.append("identity_hate")

    return detected_by_keywords


def is_classifiable(comment: str) -> bool:
    """Check if comment has actual words after basic cleaning."""
    basic_clean = re.sub(r'[^a-z\s]', '', comment.lower()).strip()
    return len(basic_clean) > 0


@st.cache_resource
def load_artifacts():
    import os
    import gdown

    if not os.path.exists("best_model_weights.weights.h5"):
        gdown.download(
            "https://drive.google.com/uc?id=16wdi6tgm_GJ8atBTj8Q10jHqt9B8-gLO",
            "best_model_weights.weights.h5",
            quiet=False
        )

    model = Sequential([
        Embedding(input_dim=20000, output_dim=128, input_length=200),
        SpatialDropout1D(0.2),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(6, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    model(np.zeros((1, 200)))
    model.load_weights("best_model_weights.weights.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
    with open("history.json", "r") as f:
        history = json.load(f)

    return model, tokenizer, metrics, history


model, tokenizer, metrics, history = load_artifacts()


def predict_toxicity(comment: str) -> dict:
    """Predict toxicity probabilities for a single comment."""
    # Check if comment has classifiable content
    if not is_classifiable(comment):
        return None

    try:
        sequence = tokenizer.texts_to_sequences([comment])
        padded = pad_sequences(
            sequence,
            maxlen=max_len,
            padding='post',
            truncating='post'
        )
        with tf.device('/CPU:0'):
            predictions = model(padded, training=False)
        predictions_numpy = predictions.numpy()[0]
        return {
            label: float(predictions_numpy[i])
            for i, label in enumerate(label_cols)
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


# Sidebar
st.sidebar.title("Toxicity Detector")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Predict", "EDA Dashboard", "Model Performance"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
st.sidebar.markdown(f"Mean AUC : `{metrics.get('mean_auc', 'N/A')}`")
st.sidebar.markdown("Architecture : `LSTM + Rules`")
st.sidebar.markdown("Labels : `6`")


# PAGE 1 — HOME
if page == "Home":
    st.title("Comment Toxicity Detection")
    st.markdown("### Deep Learning powered content moderation")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Training Comments", value="159,571")
    with col2:
        st.metric(label="Toxicity Labels", value="6")
    with col3:
        st.metric(
            label="Mean AUC Score",
            value=f"{metrics.get('mean_auc', 0):.4f}"
        )
    with col4:
        st.metric(label="Model", value="LSTM")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### What this app does")
        st.markdown("""
        This application uses a **hybrid three-layer approach**:

        **Layer 1 — LSTM Deep Learning Model**
        Trained on 159,571 Wikipedia comments with oversampling
        and class weights to handle rare label imbalance.
        Mean AUC: 0.9735.

        **Layer 2 — Keyword Matching with Negation Handling**
        200+ threat phrases and 60+ identity hate phrases
        extracted from all 478 real threat comments in the
        training dataset. Negation check prevents false positives
        like "I would never kill you" being flagged as threat.

        **Layer 3 — Intent Word Scoring**
        Catches indirect threats by detecting 3+ threatening
        words together, even without exact keyword matches.
        """)

    with col_right:
        st.markdown("### The 6 toxicity labels")
        label_descriptions = {
            "toxic"         : "Generally rude or disrespectful",
            "severe_toxic"  : "Extremely offensive content",
            "obscene"       : "Vulgar or explicit language",
            "threat"        : "Threatening someone's safety",
            "insult"        : "Personally attacking someone",
            "identity_hate" : "Hatred based on race, religion, gender"
        }
        for label, desc in label_descriptions.items():
            color = label_colours[label]
            st.markdown(
                f'<span style="color:{color}; font-weight:600">'
                f'{label}</span> — {desc}',
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown("### How to use")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info("**Step 1**\n\nGo to the Predict page from the sidebar")
    with col_b:
        st.info("**Step 2**\n\nType or paste any comment into the text box")
    with col_c:
        st.info("**Step 3**\n\nClick Analyse and see instant predictions")


# PAGE 2 — PREDICT
elif page == "Predict":
    st.title("Toxicity Prediction")
    st.markdown("---")
    st.markdown("### Single comment analysis")

    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "comment_input" not in st.session_state:
        st.session_state.comment_input = ""
    if "keyword_detected" not in st.session_state:
        st.session_state.keyword_detected = []

    comment = st.text_area(
        label="Enter a comment to analyse",
        placeholder="Type or paste a comment here...",
        height=120,
        key="comment_input"
    )

    analyse_btn = st.button("🔍 Analyse Comment")

    if analyse_btn:
        if comment.strip() == "":
            st.warning("Please enter a comment before clicking Analyse.")
        elif not is_classifiable(comment):
            st.warning(
                " This comment contains no classifiable text "
                "(only symbols or numbers). "
                "Please enter a comment with actual words."
            )
        else:
            with st.spinner("Analysing comment..."):
                predictions = predict_toxicity(comment)
                if predictions is not None:
                    st.session_state.prediction_result = predictions
                    st.session_state.keyword_detected = check_keywords(comment)
                else:
                    st.error("Model failed to return predictions.")

    if st.session_state.prediction_result is not None:
        predictions = st.session_state.prediction_result
        keyword_detected = st.session_state.keyword_detected

        st.markdown("---")

        ml_detected = [
            label for label, score in predictions.items()
            if score >= THRESHOLDS[label]
        ]
        detected_labels = list(set(ml_detected + keyword_detected))

        max_label = max(predictions, key=predictions.get)
        max_score = predictions[max_label]

        if len(detected_labels) == 0:
            st.success("This comment appears to be clean.")
        elif max_score < 0.6 and len(keyword_detected) == 0:
            st.warning(
                f"This comment may be mildly toxic — "
                f"detected labels: **{', '.join(sorted(detected_labels))}**"
            )
        else:
            st.error(
                f"This comment is likely toxic — "
                f"detected labels: **{', '.join(sorted(detected_labels))}**"
            )

        if keyword_detected:
            st.info(
                f"⚠️ Rule-based detection also flagged: "
                f"**{', '.join(keyword_detected)}** (keyword match)"
            )

        st.markdown("---")
        st.markdown("### Toxicity scores per label")

        labels = list(predictions.keys())
        scores = [round(v * 100, 2) for v in predictions.values()]
        colors = [label_colours[l] for l in labels]

        fig = go.Figure(go.Bar(
            x=labels,
            y=scores,
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
        result_df = pd.DataFrame({
            "Label"      : labels,
            "Probability": [f"{v*100:.2f}%" for v in predictions.values()],
            "Threshold"  : [f"{THRESHOLDS[l]*100:.0f}%" for l in labels],
            "Risk Level" : [
                "High"   if v >= 0.6
                else "Medium" if v >= THRESHOLDS[label]
                else "Low"
                for label, v in predictions.items()
            ],
            "Detected"   : [
                "✅ Yes (ML)" if v >= THRESHOLDS[label]
                else "✅ Yes (Rule)" if label in keyword_detected
                else "❌ No"
                for label, v in predictions.items()
            ]
        })
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        if len(detected_labels) > 0:
            st.markdown("### Detected toxicity types")
            cols = st.columns(len(detected_labels))
            for i, label in enumerate(sorted(detected_labels)):
                with cols[i]:
                    color = label_colours[label]
                    score = predictions[label]
                    tag = "(Rule)" if (
                        label in keyword_detected
                        and label not in ml_detected
                    ) else "(ML)"
                    st.markdown(
                        f'<div style="background-color:{color}20; '
                        f'border-left: 4px solid {color}; '
                        f'padding: 10px; border-radius: 4px;">'
                        f'<span style="color:{color}; font-weight:700">'
                        f'{label}</span> {tag}<br>'
                        f'<span style="font-size:1.2em; font-weight:600">'
                        f'{score*100:.1f}%</span></div>',
                        unsafe_allow_html=True
                    )

    st.markdown("---")
    st.markdown("### Bulk prediction — upload a CSV file")
    st.markdown(
        "Upload a CSV with a column named `comment_text`. "
        "All comments processed in a single batch for speed."
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        bulk_df = pd.read_csv(uploaded_file)

        if "comment_text" not in bulk_df.columns:
            st.error("CSV must have a column named 'comment_text'.")
        else:
            st.success(f"File loaded — {len(bulk_df):,} comments found.")
            st.dataframe(bulk_df.head(), use_container_width=True)

            predict_bulk_btn = st.button("🔍 Predict All Comments")

            if predict_bulk_btn:
                with st.spinner(
                    f"Processing {len(bulk_df):,} comments in one batch..."
                ):
                    all_comments = bulk_df["comment_text"].astype(
                        str
                    ).tolist()

                    sequences = tokenizer.texts_to_sequences(all_comments)
                    padded = pad_sequences(
                        sequences,
                        maxlen=max_len,
                        padding='post',
                        truncating='post'
                    )

                    with tf.device('/CPU:0'):
                        all_preds = model(
                            padded, training=False
                        ).numpy()

                    results = []
                    for i, comment_text in enumerate(all_comments):
                        preds = {
                            label: float(all_preds[i][j])
                            for j, label in enumerate(label_cols)
                        }
                        kw = check_keywords(comment_text)
                        ml_det = [
                            l for l, s in preds.items()
                            if s >= THRESHOLDS[l]
                        ]
                        all_det = list(set(ml_det + kw))
                        preds["detected_labels"] = ", ".join(sorted(all_det))
                        preds["comment_text"] = comment_text[:100]
                        results.append(preds)

                results_df = pd.DataFrame(results)
                cols = ["comment_text", "detected_labels"] + label_cols
                results_df = results_df[cols]
                for col in label_cols:
                    results_df[col] = results_df[col].round(4)

                st.success(
                    f"✅ Predictions complete — "
                    f"{len(results_df):,} comments analysed."
                )
                st.dataframe(results_df, use_container_width=True)

                csv_output = results_df.to_csv(index=False)
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv_output,
                    file_name="toxicity_predictions.csv",
                    mime="text/csv"
                )


# PAGE 3 — EDA DASHBOARD
elif page == "EDA Dashboard":
    st.title("Exploratory Data Analysis")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Train comments", "159,571")
    with col2:
        st.metric("Test comments", "153,164")
    with col3:
        st.metric("Null values", "0")
    with col4:
        st.metric("Duplicates", "0")

    st.markdown("---")
    st.markdown("### Label distribution")
    st.markdown(
        "The dataset is heavily imbalanced. "
        "Toxic comments make up 9.58% while threats are only 0.30%. "
        "This is why we use ROC-AUC instead of accuracy."
    )

    label_counts = {
        "toxic": 15294, "severe_toxic": 1595,
        "obscene": 8449, "threat": 478,
        "insult": 7877, "identity_hate": 1405
    }

    fig_labels = go.Figure(go.Bar(
        x=list(label_counts.keys()),
        y=list(label_counts.values()),
        marker_color=list(label_colours.values()),
        text=list(label_counts.values()),
        textposition='outside'
    ))
    fig_labels.update_layout(
        yaxis_title="Number of comments",
        xaxis_title="Toxicity type",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    st.plotly_chart(fig_labels, use_container_width=True)

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Clean vs toxic comments")
        fig_pie = go.Figure(go.Pie(
            labels=["Clean comments", "At least one toxic label"],
            values=[159571 - 16225, 16225],
            marker_colors=["#1D9E75", "#E24B4A"],
            hole=0.4
        ))
        fig_pie.update_layout(
            height=350,
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.markdown("### Comment length statistics")
        st.markdown(" ")
        stats = {
            "Average words (train)"  : 67,
            "Median words (train)"   : 36,
            "90th percentile"        : 152,
            "95th percentile"        : 230,
            "MAX_LEN chosen"         : 200,
            "Coverage at MAX_LEN"    : "~93%"
        }
        for key, val in stats.items():
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**{key}**")
            with col_b:
                st.markdown(f"`{val}`")

    st.markdown("---")
    st.markdown("### Key EDA insights")
    st.info(
        "**Insight 1** — Zero null values and zero duplicate "
        "comments in both train and test sets."
    )
    st.info(
        "**Insight 2** — Toxic and obscene labels are highly "
        "correlated — most obscene comments are also labeled toxic."
    )
    st.info(
        "**Insight 3** — Threat is the rarest label at 0.30% "
        "(478 comments) — making it the hardest to detect, "
        "reflected in its AUC of 0.9521 (improved from 0.9468)."
    )
    st.info(
        "**Insight 4** — Train and test comment length "
        "distributions are nearly identical, confirming "
        "good generalisation."
    )


# PAGE 4 — MODEL PERFORMANCE
elif page == "Model Performance":
    st.title("Model Performance")
    st.markdown("---")

    st.markdown("### ROC-AUC scores per label")
    st.markdown(
        "AUC measures how well the model separates toxic from clean. "
        "1.0 = perfect. 0.5 = random guessing."
    )

    auc_labels = [k for k in metrics.keys() if k != "mean_auc"]
    auc_values = [metrics[k] for k in auc_labels]
    auc_colors = [label_colours[l] for l in auc_labels]

    fig_auc = go.Figure(go.Bar(
        x=auc_labels,
        y=auc_values,
        marker_color=auc_colors,
        text=[f"{v:.4f}" for v in auc_values],
        textposition='outside'
    ))
    fig_auc.update_layout(
        yaxis_title="AUC Score",
        yaxis_range=[0.9, 1.0],
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    st.plotly_chart(fig_auc, use_container_width=True)

    mean_auc = metrics.get("mean_auc", 0)
    if mean_auc >= 0.95:
        st.success(f"Mean AUC: {mean_auc:.4f} — Excellent performance")
    elif mean_auc >= 0.90:
        st.success(f"Mean AUC: {mean_auc:.4f} — Good performance")
    else:
        st.warning(f"Mean AUC: {mean_auc:.4f} — Acceptable performance")

    st.markdown("---")
    st.markdown("### Training history")
    st.markdown(
        "The model trained for 8 epochs before EarlyStopping "
        "triggered. Best weights were restored from Epoch 5."
    )

    col_left, col_right = st.columns(2)

    with col_left:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            y=history["loss"],
            name="Train loss",
            line=dict(color="#378ADD", width=2)
        ))
        fig_loss.add_trace(go.Scatter(
            y=history["val_loss"],
            name="Validation loss",
            line=dict(color="#E24B4A", width=2)
        ))
        fig_loss.update_layout(
            title="Loss per epoch",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=350
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    with col_right:
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            y=history["accuracy"],
            name="Train accuracy",
            line=dict(color="#378ADD", width=2)
        ))
        fig_acc.add_trace(go.Scatter(
            y=history["val_accuracy"],
            name="Validation accuracy",
            line=dict(color="#1D9E75", width=2)
        ))
        fig_acc.update_layout(
            title="Accuracy per epoch",
            xaxis_title="Epoch",
            yaxis_title="Accuracy",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=350
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown("---")
    st.markdown("### Model architecture")
    arch_data = {
        "Layer"        : [
            "Embedding", "SpatialDropout1D",
            "LSTM", "Dense"
        ],
        "Output Shape" : [
            "(None, 200, 128)", "(None, 200, 128)",
            "(None, 128)", "(None, 6)"
        ],
        "Purpose"      : [
            "Converts word IDs to 128-dimensional meaning vectors",
            "Drops 20% of word vectors to prevent overfitting",
            "Reads sequence word by word with memory — 128 units",
            "Outputs 6 independent probabilities via sigmoid"
        ]
    }
    st.dataframe(
        pd.DataFrame(arch_data),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    st.markdown("### Improvements made")
    st.success(
        " Oversampling: threat 478 → 8,586 examples; "
        "identity_hate 1,405 → 11,152 examples"
    )
    st.success(
        " Class weights: threat 5x penalty, "
        "identity_hate 2x penalty during training"
    )
    st.success(
        " Hybrid detection: LSTM + keyword matching "
        "(200+ phrases from real data) + intent word scoring"
    )
    st.success(
        " Negation handling: I would never kill you "
        "correctly classified as NOT a threat"
    )
    st.success(
        " Empty comment detection: gibberish/symbols-only "
        "comments flagged with helpful warning"
    )
    st.success(
        " Batch CSV prediction: all comments processed "
        "in a single model call for maximum speed"
    )
    st.success(" Mean AUC improved: 0.9719 → 0.9735")
    st.success(" Threat AUC improved: 0.9468 → 0.9521")

    st.markdown("---")
    st.markdown("### Known limitations & future improvements")
    st.warning(
        " **Sarcasm** — 'oh great, another genius' not detected. "
        "Requires BERT transformer model for full sentence "
        "context understanding."
    )
    st.warning(
        " **Wikipedia language bias** — model trained on "
        "Wikipedia comments (2017-2018). May underperform on "
        "modern social media language (Twitter, TikTok, gaming)."
    )
    st.warning(
        " **Deep implicit threats** — highly indirect language "
        "like 'your time is running short' may be missed. "
        "BERT would handle these semantically."
    )
    st.warning(
        " **Literary/academic false positives** — titles or "
        "academic discussion containing violent words may be "
        "flagged by keyword system."
    )
    st.info(
        " **Phase 2 upgrade path**: Replace LSTM with "
        "DistilBERT for semantic understanding, sarcasm detection, "
        "negation handling, and cross-domain generalisation."
    )