import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dropout, Dense
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load Data ---
fake = pd.read_csv("C:/Users/HP/Desktop/dataset/fake/fake.csv")
real = pd.read_csv("C:/Users/HP/Desktop/dataset/true/true.csv")
fake["label"] = 1
real["label"] = 0

df = pd.concat([fake, real]).sample(frac=1.0, random_state=42).reset_index(drop=True)
X = df["text"].astype(str)
y = df["label"]

# --- Tokenization ---
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X)

sequences = tokenizer.texts_to_sequences(X)
maxlen = 300  # Increased to handle more context
X_pad = pad_sequences(sequences, maxlen=maxlen)

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, stratify=y, random_state=42)

# --- Hybrid Model ---
model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=maxlen),
    LSTM(64, return_sequences=True),
    Bidirectional(LSTM(32)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# --- Train (Epochs Increased to 5) ---
history = model.fit(X_train, y_train, epochs=5, batch_size=256, validation_split=0.1, verbose=1)

# --- Evaluation ---
y_pred = (model.predict(X_test) > 0.5).astype(int)

print("\n--- Evaluation Metrics (Hybrid Model) ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Hybrid Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# --- Training Curves ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- Prediction Function ---
def predict_news(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen)
    pred = model.predict(padded)[0][0]
    label = "FAKE" if pred > 0.5 else "REAL"
    print(f"🧾 Prediction: {label} ({pred:.2f} confidence)")

# --- Interactive Prediction Loop ---
print("\n🧪 Enter news headlines or paragraphs to classify them as FAKE or REAL.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("📰 Enter news text: ")
    if user_input.lower() == "exit":
        print("👋 Exiting prediction loop. Goodbye!")
        break
    else:
        predict_news(user_input)
"""
samples:
U.S. hopes to pressure Myanmar to permit Rohingya repatriation,"WASHINGTON (Reuters) - The United States hopes its determination that ethnic cleansing  occurred against the Rohingya will raise pressure on Myanmar’s military and civilian leadership to respond to the crisis and allow displaced people to return home, a U.S. official said on Wednesday. “The determination does indicate we feel it was ... organized planned and systematic,” a senior U.S. official told reporters on a conference call. “It does not point the finger at any specific group, but there is a limited number of groups that can be involved in that planning and organization.” ",politicsNews,"November 22, 2017 
Democratic Senate Leader Chuck Schumer has said it is premature to consider impeachment. Even some of O’Rourke’s supporters say impeachment talk is counterproductive as long as Republicans control Congress. “Otherwise, in my view, it’s just chest-beating,” said Nikki Redpath, a Houston-area homemaker and O’Rourke campaign volunteer. O’Rourke is seen as the favorite to win the Democratic nomination in March but analysts say his progressive views could prove a liability as he tries to reverse his party’s long losing streak in the Lone Star State. Trump finished nine percentage points ahead of Democrat Hillary Clinton in Texas last year and the state has not elected a Democratic governor or senator since 1994. Democrats have lost recent statewide elections by double-digit margins and have struggled to recruit top-tier candidates for major races. Still, O’Rourke’s anti-Trump message has resonated with oil-industry executive Katherine Stovring, who said she used to vote for candidates from both parties but now has been motivated to work for Democratic candidates as a way to stop Trump.  “I’m looking for ways to engage. This is our democracy at risk,” she said. Texas Republican strategist Matt Mackowiak said he thought O’Rourke would be trounced by Cruz unless voters turn en masse against Trump nationally. “He’s a more interesting candidate than the traditional sacrificial lamb the Democrats put up,” Mackowiak said. “But he’s far too liberal to be elected statewide.” In an era where differences between Republicans and Democrats are stark, candidates like O’Rourke have little incentive to moderate their positions, said James Henson, director of the Texas Politics Project at the University of Texas. At this point there is little downside for O’Rourke to make polarizing statements on impeachment and other issues.     “I think you can expect to hear a lot more of that as the campaign unfolds,” he said. ",politicsNews,"November 22, 2017 
"""