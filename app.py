import json
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
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

# Step 1: Loading thr pkl files 

@st.cache_resource
def load_artifacts():
    import os
    import gdown
    if not os.path.exists("best_model.keras"):
        gdown.download(
            "https://drive.google.com/uc?id=1l6qXGuwnqW84hbGjr_Z94KISxesYxp4D",
            "best_model.keras",
            quiet=False
        )
    model = load_model("best_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
    with open("history.json", "r") as f:
        history = json.load(f)
    return model, tokenizer, metrics, history

model, tokenizer, metrics, history = load_artifacts()


# Step 2: Prediction function
# Converts a raw comment string into 6 toxicity probabilities.

def predict_toxicity(comment: str) -> dict:
    try:
        # 1. Clean the text sequence
        sequence = tokenizer.texts_to_sequences([comment])
        padded = pad_sequences(
            sequence,
            maxlen=max_len,
            padding='post',
            truncating='post'
        )
        
        # 2. FORCE TensorFlow to run this predicting step cleanly
        with tf.device('/CPU:0'): # Forces it to CPU to prevent GPU driver lockups
            predictions = model(padded, training=False) # Faster/safer than .predict()
            
        # 3. Extract the array data safely
        predictions_numpy = predictions.numpy()[0]
        
        result = {
            label: float(predictions_numpy[i])
            for i, label in enumerate(label_cols)
        }
        return result
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Step 3: Sidebar

st.sidebar.title("🛡️ Toxicity Detector")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Predict", "EDA Dashboard", "Model Performance"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
st.sidebar.markdown(f"Mean AUC : `{metrics.get('mean_auc', 'N/A')}`")
st.sidebar.markdown("Architecture : `LSTM`")
st.sidebar.markdown("Labels : `6`")



# Step 3.1: PAGE 1 — HOME

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
        This application uses a deep learning LSTM model trained on
        159,571 Wikipedia comments to detect toxic content in real time.

        Given any comment, the model predicts the probability of it
        belonging to 6 toxicity categories simultaneously.

        The model achieves a **Mean AUC of 0.9719** — meaning it correctly
        identifies toxic content 97.19% of the time.
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



# Step 3.2: PAGE 2 — PREDICT
elif page == "Predict":

    st.title("Toxicity Prediction")
    st.markdown("---")

    st.markdown("### Single comment analysis")

    # 1. Properly initialize your session state variables
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
        
    if "comment_input" not in st.session_state:
        st.session_state.comment_input = ""

    # 2. Tie the text area directly to st.session_state using the 'key' parameter
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
        else:
            with st.spinner("Analysing comment..."):
                predictions = predict_toxicity(comment)
                if predictions is not None:
                    # 3. Save directly to session state inside the execution thread
                    st.session_state.prediction_result = predictions
                else:
                    st.error("Model failed to return predictions.")

    # 4. Read directly from session state to display the output
    if st.session_state.prediction_result is not None:
        predictions = st.session_state.prediction_result

        st.markdown("---")

        max_label = max(predictions, key=predictions.get)
        max_score = predictions[max_label]

        if max_score < 0.3:
            st.success("This comment appears to be clean.")
        elif max_score < 0.6:
            st.warning(
                f"This comment may be mildly toxic — "
                f"highest label: {max_label} ({max_score*100:.1f}%)"
            )
        else:
            st.error(
                f"This comment is likely toxic — "
                f"highest label: {max_label} ({max_score*100:.1f}%)"
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
            "Risk Level" : [
                "High"   if v >= 0.6 else
                "Medium" if v >= 0.3 else
                "Low"
                for v in predictions.values()
            ]
        })
        st.dataframe(result_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.markdown("### Bulk prediction — upload a CSV file")
    st.markdown(
        "Upload a CSV with a column named `comment_text`. "
        "The model will predict toxicity for every row."
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        bulk_df = pd.read_csv(uploaded_file)

        if "comment_text" not in bulk_df.columns:
            st.error(
                "CSV must have a column named 'comment_text'. "
                "Please check your file and try again."
            )
        else:
            st.success(f"File loaded — {len(bulk_df):,} comments found.")
            st.dataframe(bulk_df.head(), use_container_width=True)

            predict_bulk_btn = st.button("🔍 Predict All Comments")

            if predict_bulk_btn:
                progress = st.progress(0)
                results  = []
                total    = len(bulk_df)

                for idx, row in bulk_df.iterrows():
                    preds = predict_toxicity(str(row["comment_text"]))
                    preds["comment_text"] = str(row["comment_text"])[:100]
                    results.append(preds)
                    progress.progress((idx + 1) / total)

                results_df = pd.DataFrame(results)
                cols       = ["comment_text"] + label_cols
                results_df = results_df[cols]

                for col in label_cols:
                    results_df[col] = results_df[col].round(4)

                st.success("Predictions complete.")
                st.dataframe(results_df, use_container_width=True)

                csv_output = results_df.to_csv(index=False)
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv_output,
                    file_name="toxicity_predictions.csv",
                    mime="text/csv"
                )

# Step 3.3:PAGE 3 — EDA DASHBOARD

elif page == "EDA Dashboard":

    st.title("Exploratory Data Analysis")
    st.markdown("---")

    st.markdown("### Dataset overview")
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
        "toxic"         : 15294,
        "severe_toxic"  : 1595,
        "obscene"       : 8449,
        "threat"        : 478,
        "insult"        : 7877,
        "identity_hate" : 1405
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
        clean_count = 159571 - 16225
        toxic_count = 16225

        fig_pie = go.Figure(go.Pie(
            labels=["Clean comments", "At least one toxic label"],
            values=[clean_count, toxic_count],
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
        "**Insight 1** — The dataset has zero null values and zero "
        "duplicate comments. No data cleaning was required for missing data."
    )
    st.info(
        "**Insight 2** — Toxic and obscene labels are highly correlated. "
        "Most obscene comments are also labeled as toxic."
    )
    st.info(
        "**Insight 3** — Threat is the rarest label at 0.30%. "
        "This makes it the hardest for the model to detect — "
        "reflected in its lower AUC of 0.9468."
    )
    st.info(
        "**Insight 4** — Train and test comment length distributions "
        "are nearly identical, confirming the model will generalise well."
    )



# Step 3.4:PAGE 4 — MODEL PERFORMANCE

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
        "The model trained for 7 epochs before EarlyStopping triggered. "
        "Best weights were restored from Epoch 4."
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
            "Embedding",
            "SpatialDropout1D",
            "LSTM",
            "Dense"
        ],
        "Output Shape" : [
            "(None, 200, 128)",
            "(None, 200, 128)",
            "(None, 128)",
            "(None, 6)"
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