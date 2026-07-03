# Step 1: importing the necessary libraries
import re
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


label_cols= [
    "toxic", "severe_toxic", "obscene",
    "threat", "insult", "identity_hate"
]

max_words = 20000
max_len = 200

# --> Step 2: Loading the data

# We load both train and test here because both need the same
# cleaning and tokenization treatment.

print("\nLoading the data")

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

print(f"Train : {train.shape[0]:,} rows, {train.shape[1]} columns")
print(f"Test  : {test.shape[0]:,} rows, {test.shape[1]} columns")

# --> Step 3: Removing the duplicates

# We confirmed in EDA that there are no duplicates.But we still add this step here as a safety measure.

print("Removing the duplicates")

before_train = len(train)
before_test  = len(test)

train = train.drop_duplicates(subset="comment_text").reset_index(drop=True)
test  = test.drop_duplicates(subset="comment_text").reset_index(drop=True)

print(f"Train : removed {before_train - len(train)} duplicates")
print(f"Test  : removed {before_test  - len(test)}  duplicates")

# --> Step 4: Cleaning the texts

# here we basically remove the noises from the comments.
#  we comvert the comments to lower case since tokenization is case senstitive 

def clean_text(text):
    text = str(text)
#converting to lower case
    text = text.lower()
#removing newline and carriage return characters
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
#removing the HTML tags like <br>, <b>text</b>, <a href="...">
    text = re.sub(r'<.*?>', ' ', text)
#removing the URLs
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'www\.\S+', ' ', text)
#removing Wikipedia markups
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'={2,}.*?={2,}', ' ', text)

    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

print("\nCleaning train comments")
train["comment_clean"] = train["comment_text"].apply(clean_text)

print("\nCleaning test comments")
test["comment_clean"] = test["comment_text"].apply(clean_text)

print("Cleaned train and test comments")

# --> checking
print("\nBEFORE cleaning:")
print(" ", train["comment_text"].iloc[0][:150])
print("\nAFTER cleaning:")
print(" ", train["comment_clean"].iloc[0][:150])

# --> verifying empty comments
empty_train = (train["comment_clean"].str.strip() == "").sum()
empty_test  = (test["comment_clean"].str.strip()  == "").sum()
print(f"\nEmpty comments after cleaning — train: {empty_train}, test: {empty_test}")

# --> replacing with placeholder if empty comments exixts
if empty_train > 0:
    train["comment_clean"] = train["comment_clean"].replace("", "unknown")
if empty_test > 0:
    test["comment_clean"]  = test["comment_clean"].replace("", "unknown")

# Step 5: Tokenization

# Since, neural networks only understand numbers and not words.The Tokenizer builds a vocabulary,a dictionary that maps
# every unique word to a unique integer ID.
# Eg: {"the": 1, "you": 2, "is": 3, "hate": 312, "idiot": 847, ...}
# Then it converts every comment from words to a list of integers:
# "i hate you idiot" → [45, 312, 2, 847]

# oov_token="<OOV>"
#   OOV = Out Of Vocabulary
# The tokenizer learns the vocabulary from train data only.

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")

print("Building vocabulary from train data only")
tokenizer.fit_on_texts(train["comment_clean"])
# This reads all cleaned train comments and builds the word dictionary.

total_words = len(tokenizer.word_index)
print(f"Total unique words found : {total_words:,}")
print(f"Words kept (top)         : {max_words:,}")
print(f"Words ignored as OOV     : {total_words - max_words:,}")

# --> showing samples
print("\nSample vocabulary entries:")
sample = list(tokenizer.word_index.items())[:10]
for word, idx in sample:
    print(f"  '{word}' → {idx}")

# Step 6: Converting to sequence

# here  we convert the strings into integers
# eg: "you are an idiot" → [2, 17, 11, 847]

# We apply the SAME tokenizer to both train and test.

train_sequences = tokenizer.texts_to_sequences(train["comment_clean"])
test_sequences  = tokenizer.texts_to_sequences(test["comment_clean"])

# --> example
print("\nExample comment (cleaned):")
print(" ", train["comment_clean"].iloc[1][:100])
print("As integer sequence (first 15 numbers):")
print(" ", train_sequences[1][:15])

# Step 7: Padding

# Every comment is now a different length list of integers. Comment 1 might have 23 numbers. Comment 2 might have 150.
# Neural networks process data in batches — groups of comments processed simultaneously. 
# All items in a batch must be the SAME length, like rows in a matrix must have equal columns.

padding='post'
truncating='post'

x_train = pad_sequences(
    train_sequences,
    maxlen=max_len,
    padding='post',
    truncating='post'
)
x_test = pad_sequences(
    test_sequences,
    maxlen=max_len,
    padding='post',
    truncating='post'
)

print(f"X_train shape : {x_train.shape}")

print(f"X_test shape  : {x_test.shape}")

# --> Showing one padded example
print("\nFirst padded sequence (shows real numbers then trailing zeros):")
print(" ", x_train[1])


# Step 8: extracting the labels


y_train = train[label_cols].values.astype("float32")

print(f"y_train shape : {y_train.shape}")

# --> Showing one example — comment and its labels together
print("\nExample:")
print("  Comment :", train["comment_clean"].iloc[1][:80])
print("  Labels  :", dict(zip(label_cols, y_train[1])))

# Step 9: Saving 

# Saving tokenizer using pickle
# "wb" = write binary mode (required for pickle)
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("Saved :tokenizer.pkl")

# Saving numpy arrays
np.save("X_train.npy", x_train)
print("Saved :x_train.npy")

np.save("y_train.npy", y_train)
np.save("x_test.npy", x_test)


# Step 10: Final Summary

print(f"  Vocabulary size : {max_words:,} words")
print(f"  Sequence length : {max_len} per comment")
print(f"  x_train shape   : {x_train.shape} -  model input")
print(f"  y_train shape   : {y_train.shape} - model targets")
print(f"  x_test shape    : {x_test.shape} - for final predictions")


