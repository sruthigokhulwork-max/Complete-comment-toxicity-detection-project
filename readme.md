---
title: Comment Toxicity Detection
emoji: 🛡️
colorFrom: red
colorTo: purple
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
---

Comment Toxicity Detection


 Comment Toxicity


<!-- Description -->
A deep learning project that detects toxic comments in real time using neural network.

Online platforms like Wikipedia, YouTube, and social media face a massive problem — toxic comments. Hiring humans to read every comment is impossible. This project builds an AI model that reads any comment and instantly predicts whether it is toxic, obscene, a threat, an insult, severely toxic, or identity hate.

<!-- Approach -->

Train dataset - 159,571 comments
test dataset -  153,164 comments

Step 1: The necessary libraries for handling the datas were imported.

Step 2: The dataset in the csv form was loaded to understand and inspect the data and its columns using pandas.

Step 3: Data Cleaning.
        -There were no null and duplicate values in the datasets.

Step 4: Label Distribution
        -toxic = 15,2949.
        -severe_toxic = 1,595.
        -obscene = 8,4495.
        -threat = 4780.
        -insult = 7,8774.
        -identity_hate = 1,405.

Step 5: Checking for comment lengths
        - Comment length was checked. Average was 67 words. 90th percentile was 152 words.
        - Based on this we decided MAX_LEN = 200 which covers 93% of all comment.

Step 6: Preprocessing 

Step 6.1: text cleaning
        - All text was converted to lowercase.
        - HTML tags were removed (Wikipedia comments contain editor markup).
        - URLs were removed.
        - Wikipedia markup like == headings == and links were removed.
        - Punctuation and numbers were removed using regex.
        - Extra spaces were removed.
        - 111 train comments and 2,137 test comments became empty after cleaning and were replaced with "unknown".

Step 6.2: Tokenization
        - A vocabulary of the top 20,000 most frequent words was built from train data only.
        - Each word was converted to a unique integer ID.
        - Words outside the top 20,000 were replaced with a special OOV (Out Of Vocabulary) token.
        - Total unique words found were 161,884. We kept only the top 20,000.

Step 6.3: Padding.
        - Every comment was converted from words to a list of integers.
        - All sequences were padded to exactly 200 numbers long.
        - Short comments got zeros added at the end.
        - Long comments got cut from the end.
        - Final shape of X_train was (159,571 x 200).

Step 6.4: Saving 
        - tokenizer.pkl — the word dictionary saved using pickle.
        - X_train.npy — padded train sequences saved using numpy.
        - y_train.npy — train labels saved using numpy.
        - X_test.npy — padded test sequences saved using numpy.

Step 7: Model Building
        - We built an LSTM model using TensorFlow and Keras.
        - The architecture has 4 layers.
            1. Embedding layer — converts word IDs to 128-dimensional meaning vectors.
            2. SpatialDropout1D — drops 20% of word vectors to prevent overfitting.
            3. LSTM — reads the comment word by word with memory of what came before.
            4. Dense layer — outputs 6 independent probabilities using sigmoid activation.
        - We used sigmoid and not softmax because a comment can belong to multiple labels at the same time.
        - Loss function used was binary crossentropy.
        - Optimizer used was Adam.

Step 8: Training the model
        - The data was split into 90% train and 10% validation.
        - The model was trained for a maximum of 10 epochs.
        - 3 callbacks were used during training.
            1. EarlyStopping — stopped training if validation loss did not improve for 3 epochs.
            2. ReduceLROnPlateau — reduced learning rate if model got stuck for 2 epochs.
            3. ModelCheckpoint — saved the best model automatically during training.
        - Training stopped at Epoch 7. Best weights were restored from Epoch 4.

Step 9: Model Evaluation
        - We used ROC-AUC as our evaluation metric and not accuracy.
        - Reason: the dataset is heavily imbalanced. Threat has only 0.30% positive cases. A model saying "not a threat" for everything would get 99.7% accuracy but detect nothing. ROC-AUC honestly measures how well the model separates toxic from clean regardless of imbalance.
        - AUC scores per label:
        toxic          — 0.9756
        severe_toxic   — 0.9869
        obscene        — 0.9877
        threat         — 0.9468
        insult         — 0.9828
        identity_hate  — 0.9519
        Mean AUC       — 0.9719 (Excellent performance)

Step 10: Saving the model
        - best_model.keras — the trained LSTM model saved using Keras.
        - metrics.json — AUC scores saved for the Streamlit dashboard.
        - history.json — training history (loss and accuracy per epoch) saved for the Streamlit dashboard.

Step 11: Building the streamlit application
        - We created an interactive 4 page web application where users can type any comment and get instant toxicity predictions.

                - 1. Home — Project description, key metrics, and how to use the app.
                - 2. Predict — Type any comment and get instant toxicity scores with a bar chart and risk table. Also has a bulk CSV upload option to predict multiple comments at once.
                - 3. EDA Dashboard — Label distribution chart, clean vs toxic pie chart, comment length statistics, and key insights.
                - 4. Model Performance — ROC-AUC scores per label, training history charts, and model architecture table.

<!-- Technologies used -->

Python — Core programming language
Pandas — Data loading and exploration
NumPy — Array operations
Matplotlib / Seaborn — EDA charts
TensorFlow / Keras — LSTM model building and training
Scikit-learn — ROC-AUC evaluation
Plotly — Interactive charts in Streamlit
Streamlit — Web application framework
Pickle — Tokenizer serialization