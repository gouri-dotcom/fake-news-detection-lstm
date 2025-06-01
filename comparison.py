import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, GRU, Dense, Bidirectional, Concatenate, Conv1D, GlobalMaxPooling1D, SimpleRNN
from tensorflow.keras.optimizers import Adam
import itertools

# Load and label the data
real_df = pd.read_csv("C:/Users/HP/Desktop/dataset/true/true.csv")
real_df['label'] = 0
fake_df = pd.read_csv("C:/Users/HP/Desktop/dataset/fake/fake.csv")
fake_df['label'] = 1
df = pd.concat([real_df, fake_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)

# Prepare data
texts = df['text'].astype(str).values
y = df['label'].values

# Tokenize
max_words = 5000
max_len = 100
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_len)

# Split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, y, test_size=0.2, random_state=42)

# Embedding Layer
embedding_dim = 100

def build_model(name):
    input_layer = Input(shape=(max_len,))
    x = Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len)(input_layer)

    if name == "LSTM":
        x = LSTM(64)(x)
    elif name == "GRU":
        x = GRU(64)(x)
    elif name == "BiLSTM":
        x = Bidirectional(LSTM(64))(x)
    elif name == "RNN":
        x = SimpleRNN(64)(x)
    elif name == "CNN":
        x = Conv1D(128, 5, activation='relu')(x)
        x = GlobalMaxPooling1D()(x)
    elif name == "Hybrid":
        lstm_out = LSTM(64)(x)
        bilstm_out = Bidirectional(LSTM(64))(x)
        x = Concatenate()([lstm_out, bilstm_out])

    output_layer = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

models = ["LSTM", "GRU", "BiLSTM", "RNN", "CNN", "Hybrid"]
results = {}

for m in models:
    print(f"Training {m} model...")
    model = build_model(m)
    model.fit(X_train, y_train, batch_size=128, epochs=3, validation_split=0.1, verbose=0)
    y_proba = model.predict(X_test).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    results[m] = {
        "fpr": fpr, "tpr": tpr, "auc": auc_score,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "conf_matrix": cm
    }

# Plot ROC Curve
plt.figure(figsize=(10, 7))
for m in models:
    plt.plot(results[m]['fpr'], results[m]['tpr'], label=f"{m} (AUC = {results[m]['auc']:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison of Deep Learning Models")
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Confusion Matrices
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, m in enumerate(models):
    sns.heatmap(results[m]['conf_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(f"Confusion Matrix - {m}")
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')
plt.tight_layout()
plt.show()

# Tabular Comparison
comparison_df = pd.DataFrame({
    model: {
        "Accuracy": results[model]['accuracy'],
        "Precision": results[model]['precision'],
        "Recall": results[model]['recall'],
        "F1-Score": results[model]['f1'],
        "AUC": results[model]['auc']
    } for model in models
}).T

print("\nModel Performance Summary:")
print(comparison_df.round(4))
