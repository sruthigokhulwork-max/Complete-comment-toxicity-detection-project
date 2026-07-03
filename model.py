# LSTM model - Long Short-Term Memory model

#  We are choosing LSTM (Long Short-Term Memory) was chosen for this project because it reads comments the way a human does,
#  word by word, left to right, while maintaining a memory of everything it has read so far. This memory is controlled by three
#  internal gates that decide what information to keep, what to forget, and what to pass forward. This makes LSTM naturally 
#  suited for understanding context, tone shifts, and sequential patterns in text.

# Step 1: Importing the necessary libraries.
import numpy as np
import pickle
import os
import json
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split

# Step 2: Assigning constants
max_words = 20000
max_len = 200
embed_dim = 128
lstm_units = 128
batch_size = 256
epochs = 10

label_cols = [
    "toxic", "severe_toxic", "obscene",
    "threat", "insult", "identity_hate"
]

# Step 3: Loading the processed datas
x = np.load("X_train.npy")
y = np.load("y_train.npy")

print(f"X shape : {x.shape}")
print(f"y shape : {y.shape}")

# Step 4: Train validation split

# here we split out training data into 2 parts as train set(90%) and validation set(10%). this validation set is used only to check how 
# well its generalised.

x_train, x_val, y_train, y_val = train_test_split(
    x, y,
    test_size=0.1,
    random_state=42
)

print(f"x_train : {x_train.shape}")
print(f"x_val   : {x_val.shape}")
print(f"y_train : {y_train.shape}")
print(f"y_val   : {y_val.shape}")

# Step 5: Building the model

# We use the Sequential API — layers stacked one after another.
model = Sequential()

# there will be 4 layers 
#   --> embedding
#   --> SpatialdropoutID
#   --> LSTM
#   --> dense output

# ---> layer 1: embedding
model.add(Embedding(
    input_dim=max_words,
    output_dim=embed_dim,
    input_length=max_len
))

# --> layer 2: SpatialDroupoutID
model.add(SpatialDropout1D(0.2))

# ---> layer 3: LSTM 
model.add(LSTM(
    units=lstm_units,
    dropout=0.2,
    recurrent_dropout=0.2
))

# ---> layer 4: Dense output
model.add(Dense(units=6, activation='sigmoid'))

# Step 6: Compiling 
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# Step 7: Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=0.0001,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath='best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

print("Callbacks ready:")
print("  EarlyStopping     — stops if no improvement for 3 epochs")
print("  ReduceLROnPlateau — halves learning rate if stuck")
print("  ModelCheckpoint   — saves best model automatically")

# Step 8: Class weights to handle remaining class imbalance
# ─────────────────────────────────────────────────────────────────
# Even after oversampling, we add class weights as a second layer
# of imbalance correction. This tells the model to penalize itself
# MORE heavily when it misses rare label predictions.
#
# How class_weight works in multi-label:
# Keras applies class_weight per SAMPLE based on which labels
# that sample has. A threat comment with weight 5.0 contributes
# 5x more to the loss than a normal comment.
#
# Weights chosen based on inverse frequency after oversampling:
#   toxic        → 1.0  (34,919 examples — well learned)
#   severe_toxic → 2.0  (11,220 examples — moderate boost)
#   obscene      → 1.0  (24,192 examples — well learned)
#   threat       → 5.0  (8,586 examples  — needs extra attention)
#   insult       → 1.0  (23,766 examples — well learned)
#   identity_hate→ 2.0  (11,152 examples — moderate boost)
# ─────────────────────────────────────────────────────────────────

class_weight_dict = {
    0: 1.0,   # toxic
    1: 2.0,   # severe_toxic
    2: 1.0,   # obscene
    3: 5.0,   # threat — still needs most help
    4: 1.0,   # insult
    5: 2.0    # identity_hate
}

print("\nClass weights applied:")
for i, (label, weight) in enumerate(zip(label_cols, class_weight_dict.values())):
    print(f"  {label:<20} weight: {weight}")

# Step 9: Training the model
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stop, reduce_lr, checkpoint],
    class_weight=class_weight_dict,
    verbose=1
)

# Step 10: Plotting training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'],     label='Train loss',      color='blue')
axes[0].plot(history.history['val_loss'], label='Validation loss', color='orange')
axes[0].set_title('Loss per epoch')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(history.history['accuracy'],     label='Train accuracy',      color='blue')
axes[1].plot(history.history['val_accuracy'], label='Validation accuracy', color='orange')
axes[1].set_title('Accuracy per epoch')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.tight_layout()
plt.savefig("training_history.png")
plt.show()
print("Saved: training_history.png")

# Step 11: Evaluating the model by ROC-AUC
from sklearn.metrics import roc_auc_score

y_pred = model.predict(x_val, verbose=1)

print("\nROC-AUC per label:")
auc_scores = {}
for i, label in enumerate(label_cols):
    auc = roc_auc_score(y_val[:, i], y_pred[:, i])
    auc_scores[label] = round(float(auc), 4)
    print(f"  {label:<20} AUC: {auc:.4f}")

mean_auc = np.mean(list(auc_scores.values()))
auc_scores["mean_auc"] = round(float(mean_auc), 4)
print(f"\n  Mean AUC (overall): {mean_auc:.4f}")

if mean_auc >= 0.95:
    print("  Excellent model performance")
elif mean_auc >= 0.90:
    print("  Good model performance")
elif mean_auc >= 0.85:
    print("  Acceptable model performance")
else:
    print("  Model needs improvement")

# Step 12: Saving metrics and history
with open("metrics.json", "w") as f:
    json.dump(auc_scores, f, indent=4)
print("Saved: metrics.json")

history_dict = {
    "loss"         : history.history["loss"],
    "val_loss"     : history.history["val_loss"],
    "accuracy"     : history.history["accuracy"],
    "val_accuracy" : history.history["val_accuracy"]
}
with open("history.json", "w") as f:
    json.dump(history_dict, f, indent=4)
print("Saved: history.json")

# Step 13: Final Summary
print("\nTRAINING COMPLETE — SUMMARY")
print(f"  Best model saved : best_model.keras")
print(f"  Mean AUC         : {mean_auc:.4f}")
print(f"  Epochs run       : {len(history.history['loss'])}")
print(f"  Final val_loss   : {history.history['val_loss'][-1]:.4f}")