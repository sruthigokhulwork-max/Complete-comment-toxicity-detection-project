
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
x = np.load("x_train.npy")
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

# here it converts integer word IDs into dense vectors of meaning.
model.add(Embedding(
    input_dim=max_words,
    output_dim=embed_dim,
    input_length=max_len
))

# --> layer 2: SpatialDroupoutID

# here we randomly drop entire word embeddings during training.
model.add(SpatialDropout1D(0.2))
# Rate 0.2 means 20% of word vectors are zeroed out randomly.

# ---> layer 3: LSTM 

# it reads the comment word by word, left to right, and maintains a memory of what it has read so far.
model.add(LSTM(
    units=lstm_units,
    dropout=0.2,
    recurrent_dropout=0.2
))

# ---> layer 4: 

# it takes the LSTM's 128-number summary and produces 6 predictions.
# here we use sigmoid(it sequences each output between 0 and 1) since our data has multiple label classification.
model.add(Dense(units=6, activation='sigmoid'))

# Step 6: Compiling 

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
# this measures how wrong the model's prediction are.
# binary_crossentropy is used when each output have independent prediction.
#  lower the loss, better the model.
# optimizer = adam,the algorithm that adjusts the model's internal numbers
#   to reduce the loss after each batch.
# we track accuracy during training to watch progress.

model.summary()

# Step 7: Callbacks

# These are automatic helpers that monitor training and take action when certain conditions are met.
# --> earlystopping
# --> ReduceLROnPlateau
# --> modelcheckpoint

# ---> EarlyStopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)
# it monitors validation loss after each epoch.
# if it does not improve for 3 consecutive epochs,the training stops.
# restore_best_weights=True means it reverts to the best epoch's weights when training stops, not the last epoch's weights.

# ---> ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=0.0001,
    verbose=1
)
# LR = Learning Rate. It controls how big each adjustment step is.
# if validation loss stops improving for 2 epochs, then multiply the learning rate by 0.5 (make steps smaller).
# min_lr=0.0001 means it never reduces below this floor value.

# ---> ModelCheckpoint
checkpoint = ModelCheckpoint(
    filepath='best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)
# it saves the model to a file after every epoch where validation loss improved.
print("Callbacks ready:")
print("  EarlyStopping     — stops if no improvement for 3 epochs")
print("  ReduceLROnPlateau — halves learning rate if stuck")
print("  ModelCheckpoint   — saves best model automatically")

# Step 8: Traing the model.

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)
# model.fit() is the actual training call.

# Step 9: Plotting training history

#  history.history contains the loss and accuracy values recorded after every epoch during training.
# --> loss plot
# --> accuracy plot

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# ---> loss plot
axes[0].plot(history.history['loss'],     label='Train loss',      color='blue')
axes[0].plot(history.history['val_loss'], label='Validation loss', color='orange')
axes[0].set_title('Loss per epoch')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

# ---> accuracy plot
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

# Step 10: Evaluating the model by ROC-AUC

# roc_auc_score from sklearn calculates AUC for each label
# and we average them to get one final score.

from sklearn.metrics import roc_auc_score

# getting model prediction
y_pred = model.predict(x_val, verbose=1)

# calculating auc for each label.
print("\nROC-AUC per label:")
auc_scores = {}
for i, label in enumerate(label_cols):
    auc = roc_auc_score(y_val[:, i], y_pred[:, i])
    auc_scores[label] = round(float(auc), 4)
    print(f"  {label:<20} AUC: {auc:.4f}")
# y_val[:, i]  → true labels for label i (all rows, column i)
# y_pred[:, i] → predicted probabilities for label i

# calculating the auc for for all labels
mean_auc = np.mean(list(auc_scores.values()))
auc_scores["mean_auc"] = round(float(mean_auc), 4)
print(f"\n  Mean AUC (overall): {mean_auc:.4f}")

# Interpreting the result
if mean_auc >= 0.95:
    print("  Excellent model performance")
elif mean_auc >= 0.90:
    print("  Good model performance")
elif mean_auc >= 0.85:
    print("  Acceptable model performance")
else:
    print("  Model needs improvement")

# Step 11: Saving the metrics and the final model.

# --> Saving AUC scores as JSON
with open("metrics.json", "w") as f:
    json.dump(auc_scores, f, indent=4)
print("Saved: metrics.json")

# --> # Saving training history for Streamlit dashboard
history_dict = {
    "loss"         : history.history["loss"],
    "val_loss"     : history.history["val_loss"],
    "accuracy"     : history.history["accuracy"],
    "val_accuracy" : history.history["val_accuracy"]
}
with open("history.json", "w") as f:
    json.dump(history_dict, f, indent=4)
print("Saved: history.json")

# Step 12: Summary

print("\nTRAINING COMPLETE — SUMMARY")

print(f"  Best model saved : best_model.keras")
print(f"  Mean AUC         : {mean_auc:.4f}")
print(f"  Epochs run       : {len(history.history['loss'])}")
print(f"  Final val_loss   : {history.history['val_loss'][-1]:.4f}")



