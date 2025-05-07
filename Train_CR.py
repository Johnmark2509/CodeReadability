import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import json
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Set dataset path
DATA_PATH = "D:/Code Readability/Code_Readability_Dataset/DeepCRM/Code Snippet Repository/"

# Readability rule-based labeling function
def analyze_readability(code: str) -> str:
    lines = [line for line in code.splitlines() if line.strip()]
    num_lines = len(lines)
    if num_lines == 0:
        return "unreadable"

    avg_length = sum(len(line) for line in lines) / num_lines
    comment_lines = sum(1 for line in lines if line.strip().startswith("//") or line.strip().startswith("/*") or line.strip().startswith("*"))
    comment_ratio = comment_lines / num_lines

    indent_levels = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    avg_indent = sum(indent_levels) / len(indent_levels) if indent_levels else 0

    nesting_score = sum(line.count("{") + line.count("}") for line in lines) / num_lines

    if avg_length < 100 and comment_ratio > 0.05 and avg_indent > 2 and nesting_score < 1.5:
        return "readable"
    else:
        return "unreadable"

# Load and label files
texts, labels = [], []

print("\U0001F4C1 Scanning Java files...")
for folder in tqdm(os.listdir(DATA_PATH), desc="Loading Projects"):
    folder_path = os.path.join(DATA_PATH, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".java"):
                fpath = os.path.join(folder_path, file)
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                    texts.append(code)
                    labels.append(analyze_readability(code))

print(f"\n✅ Labeled {len(texts)} Java files.")

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
joblib.dump(label_encoder, "label_encoder.pkl")

# Tokenize code
tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)

# Save tokenizer
tokenizer_json = tokenizer.to_json()
with open("tokenizer.json", "w", encoding="utf-8") as f:
    json.dump(tokenizer_json, f)

# Pad sequences
X = tf.keras.preprocessing.sequence.pad_sequences(X, padding='post', maxlen=1000)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN model
from tensorflow.keras.layers import LeakyReLU
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=64, input_length=1000),
    tf.keras.layers.Conv1D(64, 5),
    LeakyReLU(alpha=0.01),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64),
    LeakyReLU(alpha=0.01),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best_coderead.h5', monitor='val_accuracy', save_best_only=True)
]

model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32, callbacks=callbacks)

# Evaluate and save
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {acc:.2%}")
model.save("coderead.h5")
print("\U0001F4BE Model saved as coderead.h5")

# Classification Report and Confusion Matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

true_labels = label_encoder.inverse_transform(y_test)
pred_labels = label_encoder.inverse_transform(y_pred_classes)

print("\n📊 Classification Report:")
print(classification_report(true_labels, pred_labels))

cm = confusion_matrix(true_labels, pred_labels, labels=label_encoder.classes_)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()