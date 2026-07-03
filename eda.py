#  COMMENT TOXICITY

#  In this project, we are creating an AI MODEL that for identifying toxic comments in the online and 
# predict the likelihood of each comment being toxic using python.

# Step 1: Importing the necessary libraries.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Loading the datasets

# We have 2 datasets - train.csv,test.csv
# We load the both the dataset to explore them.

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# # Step 3: Data exploration

# # --> Step 3.1: looking into train.csv
print("TRAIN.CSV")

print(train.shape)
print(train.columns.tolist())
print(train.info())
print(train.describe())
print(train.head())

# --> checking for null values

print(train.isnull().sum())

# --> checking for duplicates

# 1. duplicate rows
train_duplicates = train.duplicated().sum()
print(train_duplicates)
# 2. duplicate comments
comment_train_duplicates = train["comment_text"].duplicated().sum()
print(comment_train_duplicates)

# # --> Step 3.2: label distribution

# # This is an important step to verify the imbalance in the labels.

label_cols = [ 
        "toxic", "severe_toxic", "obscene", 
        "threat", "insult", "identity_hate"
        ]

for col in label_cols:
    count = train[col].sum()
    pct   = count / len(train) * 100
    bar   = "█" * int(pct)
    print(f"  {col:<20} {count:>6,}  ({pct:5.2f}%)  {bar}")

# --> multiple comment check

# # Sometimes a single comment may have multiple labels, so here we check how many comments have more than one label.
train["total_labels"] = train[label_cols].sum(axis=1)

print("\nHow many labels per comment?")
print(train["total_labels"].value_counts().sort_index())
# this confirms our data is multilabel and hat the dataset is heavily imbalanced toward clean comments, with about
#  90% having zero toxic labels."

# --> exploring the actual comments

# # We explore the actual comment to know what type of noise exists so that the data cleaning can be done easily.

#  --> toxic comments
print("\nReading toxic comments")

toxic_sample = train[train["toxic"] == 1]["comment_text"].values[:3]
for i, c in enumerate(toxic_sample):
    print(f"\n[{i+1}] {c[:300]}")

# # #--> clean comments
print("\nReading clean comments")

clean_sample = train[train["toxic"] == 0]["comment_text"].values[:3]
for i, c in enumerate(clean_sample):
    print(f"\n[{i+1}] {c[:300]}")

# # --> knowing the lenth of the comment
train["word_count"] = train["comment_text"].apply(
    lambda x: len(str(x).split())
)

# print("\nTrain comment length (words):")
# print(f"  Mean  : {train['word_count'].mean():.0f}")
# print(f"  Median: {train['word_count'].median():.0f}")
# print(f"  Max   : {train['word_count'].max():,}")
# print(f"  90th% : {train['word_count'].quantile(0.90):.0f}")
# print(f"  95th% : {train['word_count'].quantile(0.95):.0f}")

# # --> Step 3.3: lookig into test.csv
print("TEST.CSV")

# print(test.shape)
# print(test.columns.tolist())
# print(test.info())
# print(test.describe())
# print(test.head())

# # --> checking for null
print(test.isnull().sum())

# --> checking for duplicates

# 1. duplicate rows
test_duplicates = test.duplicated().sum()
print(test_duplicates)
# 2. duplicate comments
comment_test_duplicates = test["comment_text"].duplicated().sum()
print(comment_test_duplicates)

# --> exploring the actual comments
# # We explore the actual comment to know what type of noise exists so that the data cleaning can be done easily.
print("\n Sample comments")

for i, c in enumerate(test["comment_text"].values[:3]):
    print(f"\n[{i+1}] {c[:300]}")

# -->knowing the length of the comment

test["word_count"] = test["comment_text"].apply(
    lambda x: len(str(x).split())
)

# print("\nTest comment length (words):")
print(f"  Mean  : {test['word_count'].mean():.0f}")
print(f"  Median: {test['word_count'].median():.0f}")
print(f"  Max   : {test['word_count'].max():,}")
print(f"  90th% : {test['word_count'].quantile(0.90):.0f}")
print(f"  95th% : {test['word_count'].quantile(0.95):.0f}")

# --> checking for overlaps
overlap = set(train["id"]).intersection(set(test["id"]))
print(f"\nID overlap between train and test: {len(overlap)}")
print("(Should be 0 — no comment should appear in both)")

# # --> Step 3.4: Comparisons

# # --> Label distribution using bar chart for TRAIN

counts = [train[col].sum() for col in label_cols]
plt.figure(figsize=(10, 5))
sns.barplot(x=label_cols, y=counts, hue=label_cols, palette="viridis", legend=False)
plt.title("Label distribution in training data")
plt.ylabel("Number of comments")
plt.xlabel("Toxicity type")
plt.tight_layout()
plt.savefig("label_distribution.png")
plt.show()

# # --> comment length comparison between train ad test

plt.figure(figsize=(10, 5))

plt.hist(
    train["word_count"].clip(upper=500),
    bins=50,
    alpha=0.6,
    label="Train"
)
plt.hist(
    test["word_count"].clip(upper=500),
    bins=50,
    alpha=0.6,
    color="orange",
    label="Test"
)
plt.axvline(
    x=200,
    color="red",
    linestyle="--",
    label="MAX_LEN = 200"
)
plt.title("Comment length: Train vs Test")
plt.xlabel("Word count (clipped at 500)")
plt.ylabel("Number of comments")
plt.legend()
plt.tight_layout()
plt.savefig("length_comparison.png")
plt.show()

# # --> Summary

print("EDA SUMMARY")

print(f"  Train size      : {len(train):,} comments")
print(f"  Test size       : {len(test):,} comments")
print(f"  Labels          : 6 (multi-label problem)")
print(f"  Null values     : 0 (no cleaning needed for nulls)")
print(f"  MAX_LEN chosen  : 200 words")
print(f"  Class imbalance : YES — use ROC-AUC not accuracy")
print(f"  ID overlap      : {len(overlap)} (data is clean)")

# --> Insights

# 1. Both the test and the train dataset has no null values, no duplicates. So our dataset are clean.

# 2. From the label distribution we found out that toxic comments make up to 9.58%, threat has about 0.30% which is the least
    #  so  we use the roc-auc method to detect the comments.

# 3. We read actual comments and saw HTML tags, Wikipedia markup, URLs, newlines mixed into the text.

# 4. We measured comment lengths and found 90th percentile at 152 words and 95th is 230 words. So, MAX_LEN=200(somewhere
#      between 90th and 95th).