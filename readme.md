 Comment Toxicity Detection

A hybrid deep learning project that detects toxic comments in real time using DistilBERT transformer model combined with an LSTM cascade system and a rule-based keyword detection system.

Online platforms like Wikipedia, YouTube, and social media face a massive problem — toxic comments. Hiring humans to read every comment is impossible. This project builds an AI system that reads any comment and instantly predicts whether it is toxic, obscene, a threat, an insult, severely toxic, or identity hate.

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

### Step 4: Label Distribution
- toxic = 15,294 (9.58%)
- obscene = 8,449 (5.29%)
- insult = 7,877 (4.94%)
- severe_toxic = 1,595 (1.00%)
- identity_hate = 1,405 (0.88%)
- threat = 478 (0.30%) — rarest label

Since a single comment can belong to multiple categories at the same time, this is a multi-label classification problem, not multi-class.

 Step 5: Comment Length Analysis
- Average comment length was 67 words.
- 90th percentile was 152 words.
- 95th percentile was 230 words.
- MAX_LEN = 200 was chosen for LSTM, covering approximately 93% of all comments.
- MAX_LEN = 128 was chosen for DistilBERT.

 Step 6: Preprocessing

 Step 6.1: Oversampling (Class Imbalance Fix)
Before cleaning, rare labels were oversampled to fix class imbalance:
- threat: 478 to 8,586 examples (duplicated x15)
- identity_hate: 1,405 to 11,152 examples (duplicated x5)
- severe_toxic: 1,595 to 11,220 examples (duplicated x4)
- Total training rows after oversampling: 180,146

 Step 6.2: Text Cleaning
- All text was converted to lowercase.
- Newlines and carriage returns were removed.
- HTML tags were removed.
- URLs were removed.
- Wikipedia markup like ==headings== and [[links]] were removed.
- Punctuation and numbers were removed using regex.
- Extra whitespace was collapsed into single spaces.
- 111 train comments and 2,137 test comments became empty after cleaning and were replaced with the placeholder word unknown.

 Step 6.3: Tokenization (for LSTM)
- A vocabulary of the top 20,000 most frequent words was built from train data only — never test data, to prevent data leakage.
- Each word was converted to a unique integer ID.
- Words outside the top 20,000 were replaced with a special OOV (Out Of Vocabulary) token.
- Total unique words found: 161,884. Only top 20,000 were kept.

 Step 6.4: Padding (for LSTM)
- All sequences were padded to exactly 200 numbers long using post-padding.
- Short comments got zeros added at the end.
- Long comments got cut from the end.
- Post padding was used because LSTM reads left to right — real words should come first.
- Final shape of X_train: (180,146 x 200).

 Step 6.5: Saving
- tokenizer.pkl — the word dictionary saved using pickle.
- X_train.npy — padded train sequences saved using numpy.
- y_train.npy — train labels saved using numpy.
- X_test.npy — padded test sequences saved using numpy.

 Step 7: Phase 1 - LSTM Model Building
A deep learning LSTM model was built using TensorFlow and Keras with 4 layers:

1. Embedding layer — converts word IDs to 128-dimensional meaning vectors.
2. SpatialDropout1D(0.2) — randomly drops 20% of entire word vectors to prevent overfitting.
3. LSTM(128 units, dropout=0.2, recurrent_dropout=0.2) — reads the comment word by word with memory of what came before.
4. Dense(6, sigmoid) — outputs 6 independent probabilities, one per toxicity label.

Sigmoid was used instead of softmax because a comment can belong to multiple labels at the same time. Softmax forces all probabilities to sum to 100%, which only works when exactly one label applies.

- Loss function: binary_crossentropy
- Optimizer: Adam

 Step 8: LSTM Training

Class weights were applied to handle remaining imbalance:
- threat: 5.0 (5x penalty for missed threat detections)
- identity_hate: 2.0
- severe_toxic: 2.0
- toxic, obscene, insult: 1.0

The data was split into 90% train and 10% validation. The model was trained for a maximum of 10 epochs with batch size 256.

3 callbacks were used:
1. EarlyStopping — stopped training if validation loss did not improve for 3 consecutive epochs. restore_best_weights=True rolls back to the best epoch automatically.
2. ReduceLROnPlateau — halved the learning rate if model got stuck for 2 epochs (factor=0.5, min_lr=0.0001).
3. ModelCheckpoint — saved the best model weights automatically whenever validation loss improved.

Training stopped at Epoch 8. Best weights were restored from Epoch 5.

 Step 9: LSTM Evaluation
ROC-AUC was used as the evaluation metric instead of accuracy.

Why not accuracy? The dataset is heavily imbalanced. Threat has only 0.30% positive cases. A model saying "not a threat" for every comment would score 99.7% accuracy while detecting zero real threats. ROC-AUC honestly measures how well the model separates toxic from clean regardless of class imbalance.

LSTM AUC scores per label:
- toxic — 0.9790
- severe_toxic — 0.9862
- obscene — 0.9887
- threat — 0.9521
- insult — 0.9842
- identity_hate — 0.9506
- Mean AUC — 0.9735 — Excellent performance

 Step 10: Phase 2 - DistilBERT Fine-Tuning

DistilBERT was chosen over full BERT because:
- DistilBERT is 40% smaller and 60% faster than BERT
- Retains 97% of BERT's performance
- Fits comfortably within Hugging Face free tier memory limits

DistilBERT was fine-tuned on 20% of the original training data (31,914 comments) using Apple M4 MPS acceleration. Training took approximately 2 hours.

Fine-tuning settings:
- Model: distilbert-base-uncased
- MAX_LEN: 128
- Batch size: 16
- Epochs: 3
- Learning rate: 2e-5
- Loss: BCEWithLogitsLoss with class weights (threat: 20x, identity_hate: 8x)
- Device: Apple M4 MPS (Metal Performance Shaders)

DistilBERT AUC scores per label (after fine-tuning):
- toxic — 0.9820
- severe_toxic — 0.9916
- obscene — 0.9903
- threat — 0.9923 (improved from 0.9521)
- insult — 0.9840
- identity_hate — 0.9855 (improved from 0.9506)
- Mean AUC — 0.9876 (improved from 0.9735)

 Step 11: Hybrid Detection System

The app uses a four-layer hybrid approach:

Layer 1 - DistilBERT (primary model)
Predicts 6 independent probability scores per comment. Reads the entire sentence simultaneously using attention mechanisms. Understands context better than LSTM.

Layer 2 - Negation Correction
Applied immediately after BERT prediction — before any other decision. Detects negation words (never, not, won't, can't, couldn't) appearing within 5 words before a core threatening word (kill, hurt, murder, shoot, stab etc.). If negation is detected, all scores are set to 0.05, correctly classifying "I would never kill you" as clean.

Layer 3 - LSTM Cascade (local version only)
When BERT score is borderline (between 0.2 and 0.5), LSTM runs as a second opinion and scores are averaged. This is used in the local version only — not on Hugging Face to avoid TensorFlow and PyTorch conflicts.

Layer 4 - Keyword Safety Net
200+ threat phrases and 60+ identity hate phrases extracted by reading all 478 real threat comments from the training dataset. Skipped automatically when negation is detected to avoid false positives.

 Step 12: Optimized Thresholds

Label-specific thresholds were set based on testing:
- toxic: 0.30
- severe_toxic: 0.60
- obscene: 0.60
- threat: 0.15 (lower because BERT scores threats conservatively)
- insult: 0.60
- identity_hate: 0.50

 Step 13: Building the Streamlit Application
An interactive 4-page web application was built.

1. Home — Project description, key metrics (AUC 0.9876), label definitions, and how to use.
2. Predict — Type any comment and get instant toxicity scores with a bar chart, verdict (clean, mildly toxic, likely toxic), detailed scores table showing score, threshold, risk level, and detection method (BERT or Rule). Also supports bulk CSV upload processed in batches of 32.
3. EDA Dashboard — Label distribution chart, clean vs toxic pie chart, comment length statistics, and 4 key insights.
4. Model Performance — LSTM vs DistilBERT AUC comparison table, per-label AUC chart, LSTM training history, all improvements made, and known limitations.

 Step 14: Deployment

Two versions of the app were built:

Local version (app_local.py) — BERT + LSTM cascade + keyword system. Uses Apple M4 MPS acceleration for fast inference. Full 4-layer hybrid system.

Hugging Face version (app.py) — DistilBERT only + keyword system. No LSTM to avoid TensorFlow and PyTorch conflicts on deployment server. Uses CPU inference.

The app was deployed on Hugging Face Spaces.

Why Hugging Face and not Streamlit Cloud? TensorFlow 2.21.0 requires protobuf 6.31.1 but Streamlit Cloud's shared environment installs an incompatible older version, causing import errors. Hugging Face Spaces allows full control over the environment through runtime.txt, packages.txt, and requirements.txt.

The DistilBERT model weights are saved as bert_best_model folder, zipped, and hosted on Google Drive. Downloaded automatically via gdown on first app startup.

Live app: https://sruthi-g-comment-toxicity-detection.hf.space
---

 AUC Improvement Summary

| Label - Baseline LSTM - Final DistilBERT - Improvement 

| toxic | 0.9756 | 0.9820 | +0.0064 |
| severe_toxic | 0.9869 | 0.9916 | +0.0047 |
| obscene | 0.9877 | 0.9903 | +0.0026 |
| threat | 0.9468 | 0.9923 | +0.0455 |
| insult | 0.9828 | 0.9840 | +0.0012 |
| identity_hate | 0.9519 | 0.9855 | +0.0336 |
| Mean AUC | 0.9719 | 0.9876 | +0.0157 |

---

 Tech Stack

 Tool - Purpose 

| Python 3.11 | Core programming language |
| Pandas | Data loading and exploration |
| NumPy | Array operations and saving .npy files |
| Matplotlib / Seaborn | EDA charts |
| TensorFlow / Keras | LSTM model building and training |
| PyTorch | DistilBERT fine-tuning and inference |
| Transformers (HuggingFace) | DistilBERT model and tokenizer |
| Scikit-learn | ROC-AUC evaluation metric |
| Plotly | Interactive charts in Streamlit |
| Streamlit | Web application framework |
| Pickle | Tokenizer serialization |
| gdown | Downloading model from Google Drive |
| Hugging Face Spaces | Live deployment platform |
