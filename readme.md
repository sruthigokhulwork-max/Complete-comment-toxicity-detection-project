# Comment Toxicity Detection

A hybrid deep learning project that detects toxic comments in real time using LSTM neural network combined with a rule-based keyword detection system.

Online platforms like Wikipedia, YouTube, and social media face a massive problem — toxic comments. Hiring humans to read every comment is impossible. This project builds an AI model that reads any comment and instantly predicts whether it is toxic, obscene, a threat, an insult, severely toxic, or identity hate.

Live App: https://sruthi-g-comment-toxicity-detection.hf.space

Train dataset — 159,571 comments
Test dataset — 153,164 comments

---
 Workflow

 Step 1: Import Libraries
The necessary libraries for handling data, building models, and creating the web application were imported.

 Step 2: Load Dataset
The dataset in CSV form was loaded using Pandas to understand and inspect the data and its columns.

 Step 3: Data Cleaning
- There were no null values and no duplicate values in both train and test datasets.
- Train and test comment IDs were checked for overlap — zero overlap confirmed, no data leakage.

 Step 4: Label Distribution
- toxic = 15,294
- severe_toxic = 1,595
- obscene = 8,449
- threat = 478 (rarest — only 0.30%)
- insult = 7,877
- identity_hate = 1,405

Since a single comment can belong to multiple categories at the same time, this is a multi-label classification problem, not multi-class.

 Step 5: Comment Length Analysis
- Comment length was measured for all train and test comments.
- Average length was 67 words.
- 90th percentile was 152 words.
- 95th percentile was 230 words.
- Based on this, MAX_LEN = 200 was chosen, which covers approximately 93% of all comments.

Step 6: Preprocessing

 Step 6.1: Oversampling (Class Imbalance Fix)
Before cleaning, rare labels were oversampled to fix class imbalance:
- threat: 478 → 8,586 examples (duplicated ×15)
- identity_hate: 1,405 → 11,152 examples (duplicated ×5)
- severe_toxic: 1,595 → 11,220 examples (duplicated ×4)
- Total training rows after oversampling: 180,146

Step 6.2: Text Cleaning
- All text was converted to lowercase.
- Newlines and carriage returns were removed.
- HTML tags were removed (Wikipedia comments contain editor markup).
- URLs were removed.
- Wikipedia markup like ==headings== and links were removed.
- Punctuation and numbers were removed using regex.
- Extra whitespace was collapsed into single spaces.
- 111 train comments and 2,137 test comments became empty after cleaning and were replaced with the placeholder word "unknown".

 Step 6.3: Tokenization
- A vocabulary of the top 20,000 most frequent words was built from train data only (never test data — to prevent data leakage).
- Each word was converted to a unique integer ID.
- Words outside the top 20,000 were replaced with a special OOV (Out Of Vocabulary) token.
- Total unique words found: 161,884. Only the top 20,000 were kept.

 Step 6.4: Padding
- Every comment was converted from words to a list of integers.
- All sequences were padded to exactly 200 numbers long using post-padding.
- Short comments got zeros added at the end (post-padding).
- Long comments got cut from the end (post-truncating).
- Why post and not pre? Because LSTM reads left to right — real words should come first, not padding zeros.
- Final shape of X_train: (180,146 × 200).

 Step 6.5: Saving
- tokenizer.pkl — the word dictionary saved using pickle.
- X_train.npy — padded train sequences saved using numpy.
- y_train.npy — train labels saved using numpy.
- X_test.npy — padded test sequences saved using numpy.

Step 7: Model Building
A deep learning LSTM model was built using TensorFlow and Keras with 4 layers:

1. Embedding layer — converts word IDs to 128-dimensional meaning vectors.
2. SpatialDropout1D(0.2) — randomly drops 20% of entire word vectors to prevent overfitting.
3. LSTM(128 units, dropout=0.2, recurrent_dropout=0.2)** — reads the comment word by word with memory of what came before.
4. Dense(6, sigmoid) — outputs 6 independent probabilities, one per toxicity label.

We used sigmoid and not softmax because a comment can belong to multiple labels at the same time (multi-label). Softmax forces all probabilities to sum to 100%, which only works when exactly one label applies.

- Loss function: binary_crossentropy (correct for independent yes/no predictions per label)
- Optimizer: Adam

 Step 8: Training the Model

Class weights were applied to handle remaining imbalance:**
- threat: 5.0 (5× penalty for missed threat detections)
- identity_hate: 2.0
- severe_toxic: 2.0
- toxic, obscene, insult: 1.0

The data was split into 90% train and 10% validation. The model was trained for a maximum of 10 epochs with batch size 256.

3 callbacks were used during training:
1. EarlyStopping — stopped training if validation loss did not improve for 3 consecutive epochs. restore_best_weights=True rolls back to the best epoch automatically.
2. ReduceLROnPlateau — halved the learning rate if model got stuck for 2 epochs (factor=0.5, min_lr=0.0001).
3. ModelCheckpoint — saved the best model weights automatically whenever validation loss improved.

Training stopped at Epoch 8. Best weights were restored from Epoch 5.

Step 9: Model Evaluation
We used ROC-AUC as our evaluation metric and not accuracy.

Why not accuracy?
The dataset is heavily imbalanced. Threat has only 0.30% positive cases. A model saying "not a threat" for every single comment would score 99.7% accuracy while detecting zero real threats. ROC-AUC honestly measures how well the model separates toxic from clean regardless of class imbalance.

AUC scores per label (after improvements):
- toxic — 0.9790
- severe_toxic — 0.9862
- obscene — 0.9887
- threat — 0.9521 (improved from 0.9468)
- insult — 0.9842
- identity_hate — 0.9506
- Mean AUC — 0.9735 (improved from 0.9719) — Excellent performance

 Step 10: Saving the Model
- best_model_weights.weights.h5 — only the trained weights saved (not full model) to avoid Keras version compatibility issues during deployment. Hosted on Google Drive and downloaded automatically via gdown at app startup.
- metrics.json — AUC scores saved for the Streamlit dashboard.
- history.json — training history (loss and accuracy per epoch) saved for the Streamlit dashboard.
- tokenizer.pkl — saved Keras tokenizer for use during prediction.

 Step 11: Hybrid Detection System

The app uses a three-layer hybrid approach to handle class imbalance in threat and identity_hate detection:

Layer 1 — LSTM Deep Learning Model**
Predicts 6 independent probability scores per comment.

Layer 2 — Keyword Matching with Negation Handling**
200+ threat phrases and 60+ identity hate phrases extracted by reading all 478 real threat comments from the training dataset.

Includes negation check — "I would never kill you" is NOT flagged as a threat because "never" appears within 4 words before the threat keyword "kill you".

Layer 3 — Intent Word Scoring**
Catches indirect threats by detecting 3 or more threatening intent words together in a comment, even when no exact keyword phrase matches.

Step 12: Building the Streamlit Application
An interactive 4-page web application was built where users can type any comment and get instant toxicity predictions.

1. Home — Project description, key metrics, label definitions, and how to use the app.
2. Predict — Type any comment and get instant toxicity scores with a bar chart, verdict (clean / mildly toxic / likely toxic), and a detailed scores table showing ML score, threshold, risk level, and whether each label was detected by ML or rule-based system. Also supports bulk CSV upload to predict multiple comments at once in a single batch.
3. EDA Dashboard — Label distribution chart, clean vs toxic pie chart, comment length statistics, and 4 key insights.
4. Model Performance — ROC-AUC scores per label, training history charts, model architecture table, list of improvements made, and known limitations with Phase 2 upgrade path.

Step 13: Deployment
The app was deployed on Hugging Face Spaces.

Why Hugging Face and not Streamlit Cloud?
TensorFlow 2.21.0 requires protobuf ≥ 6.31.1 but Streamlit Cloud's shared environment installs protobuf < 5, causing a version conflict at import time. Hugging Face Spaces allows full control over the environment through runtime.txt, packages.txt, and requirements.txt.

Live app:https://sruthi-g-comment-toxicity-detection.hf.space

---

 Known Limitations

- Sarcasm — "oh great, another genius" not detected. LSTM reads word by word and cannot understand irony. Requires BERT.
- Wikipedia language bias — model trained on 2017-2018 Wikipedia data. May underperform on modern social media language.
- Deep implicit threats — highly indirect language like "your time is running short" may be missed.
- Phase 2 upgrade — replace LSTM with DistilBERT for semantic understanding, sarcasm detection, and cross-domain generalisation.

---

 Tech Stack

Tool - Purpose 

 Python 3.11 - Core programming language 
 Pandas - Data loading and exploration 
 NumPy - Array operations and saving .npy files 
 Matplotlib / Seaborn - EDA charts 
 TensorFlow / Keras - LSTM model building and training
 Scikit-learn - ROC-AUC evaluation metric 
 Plotly - Interactive charts in Streamlit 
 Streamlit - Web application framework 
 Pickle - Tokenizer serialization 
 gdown - Downloading model weights from Google Drive 
 Hugging Face Spaces - Live deployment platform 
