#Dataset: https://www.kaggle.com/datasets/rmisra/news-category-dataset

# ============================
# Library import
# ============================
import numpy as np
import json
import tensorflow as tf
from transformers import AutoTokenizer
import os
import time
from warnings import filterwarnings
import matplotlib.pyplot as plt
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
filterwarnings('ignore')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


USE_LEMMATIZATION = True
USE_STEMMING = True
REMOVE_STOPWORDS = True
MIN_TOKEN_LEN = 2
NUMBER_TOKEN = "<NUM>"
# ============================
# Parameters
# ============================
JSON_PATH = "News_Category_Dataset_v3.json"
TOKENIZER_NAME = "distilbert-base-uncased"
DATASET_PATH = "tokenized_dataset.npz"
MAX_LEN = 192
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 5e-4
REGULARIZATION_RATE = 1e-4
limit_per_category = 1000
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
from_bert = False
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"]) if USE_LEMMATIZATION else None
except Exception:
    nlp = None
    print("[WARN] spaCy model 'en_core_web_sm' not loaded. Run: python -m spacy download en_core_web_sm")

try:
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = set()
STEMMER = SnowballStemmer("english")

# ====== text preprocessing function ======
def preprocess_text_py(text: str) -> str:

    if not isinstance(text, str):
        text = str(text)
    txt = text.strip().lower()

    txt = re.sub(r"http\S+|www\.\S+", " ", txt)
    txt = re.sub(r"\S+@\S+\.\S+", " ", txt)

    txt = re.sub(r"\b\d+(\.\d+)?\b", f" {NUMBER_TOKEN} ", txt)

    txt = re.sub(r"[^a-z0-9'\s<>]", " ", txt)

    txt = re.sub(r"\s+", " ", txt).strip()

    tokens = txt.split()

    cleaned = []
    for t in tokens:
        if len(t) < MIN_TOKEN_LEN:
            continue
        if REMOVE_STOPWORDS and t in STOPWORDS:
            continue
        cleaned.append(t)

    if not cleaned:
        return ""

    if USE_LEMMATIZATION and nlp is not None:
        doc = nlp(" ".join(cleaned))
        lemmas = []
        for tok in doc:
            l = tok.lemma_.lower().strip()
            if l == "-pron-" or not l:
                continue
            if REMOVE_STOPWORDS and l in STOPWORDS:
                continue
            if len(l) >= MIN_TOKEN_LEN:
                lemmas.append(l)
        cleaned = lemmas if lemmas else cleaned

    if USE_STEMMING:
        stems = []
        for t in cleaned:
            try:
                s = STEMMER.stem(t)
                if len(s) >= MIN_TOKEN_LEN:
                    stems.append(s)
            except Exception:
                stems.append(t)
        cleaned = stems

    return " ".join(cleaned)


# ============================
# Get categories
# ============================
def extract_categories(json_path):
    categories = set()
    with open(json_path, "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            obj = json.loads(line)
            categories.add(obj["category"])
            line = f.readline()
    categories = sorted(list(categories))
    cat2id = {c: i for i, c in enumerate(categories)}
    id2cat = {i: c for c, i in cat2id.items()}
    return cat2id, id2cat

cat2id, id2cat = extract_categories(JSON_PATH)
NUM_CLASSES = len(cat2id)
print(f"[INFO] Number of categories: {NUM_CLASSES}")

# ============================
# Load and preprocess full dataset
# ============================
CACHE_FILE = "preprocessed_texts.npz"

if os.path.exists(CACHE_FILE):
    print("[INFO] Loading preprocessed data from cache...")
    npz = np.load(CACHE_FILE, allow_pickle=True)
    texts = npz["texts"]
    input_ids = npz["input_ids"]
    labels = npz["labels"]
else:
    print("[INFO] Preprocessing and tokenizing full dataset...")

    texts = []
    labels = []
    cat_counts = {}

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cat = obj["category"]
            if cat_counts.get(cat, 0) < limit_per_category:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
                label = cat2id[cat]
                text = f"{obj['headline']} {obj['short_description']}"
                clean_text = preprocess_text_py(text)
                if not clean_text:
                    clean_text = "empty"
                texts.append(clean_text)
                labels.append(label)

    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_tensors="np"
    )
    input_ids = encodings["input_ids"]
    labels = np.array(labels, dtype=np.int32)

    np.savez(CACHE_FILE, input_ids=input_ids, texts=texts, labels=labels)
    print(f"[INFO] Saved preprocessed dataset to {CACHE_FILE}")

# ============================
# train/val split
# ============================
dataset_size = len(input_ids)
train_size = int(dataset_size * 0.8)
indices = np.arange(dataset_size)
np.random.shuffle(indices)

train_idx = indices[:train_size]
val_idx = indices[train_size:]

X_train = input_ids[train_idx]
y_train = labels[train_idx]
X_val = input_ids[val_idx]
y_val = labels[val_idx]

y_train = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=NUM_CLASSES)

train_dataset = tf.data.Dataset.from_tensor_slices(({"input_ids": X_train}, y_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices(({"input_ids": X_val}, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def predict_text(text, model, from_bert=False):
    clean_text = preprocess_text_py(text)
    if not from_bert:
        encoding = tokenizer(
            clean_text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            add_special_tokens=True,
            return_tensors="np"
        )
        arg = encoding["input_ids"]
    else:
        arg = compute_bert_embeddings([clean_text])
    logits = model.predict(arg)
    label_id = np.argmax(logits, axis=1)[0]
    prob = logits[0][label_id]
    return id2cat[label_id], prob

i = int(input("Choose a model: "))
if i == 1:
    # ============================
    # Logistic regression model
    # ============================
    inputs = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_ids")
    embeddings = tf.keras.layers.Embedding(
        input_dim=tokenizer.vocab_size,
        output_dim=64,
        input_length=MAX_LEN
    )(inputs)
    flatten = tf.keras.layers.Flatten()(embeddings)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(REGULARIZATION_RATE))(flatten)
    path = "logistic"

if i == 2:
    # ============================
    # LSTM model
    # ============================
    inputs = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_ids")
    embeddings = tf.keras.layers.Embedding(
        input_dim=tokenizer.vocab_size,
        output_dim=128,
        input_length=MAX_LEN,
        mask_zero=True
    )(inputs)
    reccurent = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))(embeddings)
    drop = tf.keras.layers.Dropout(0.3)(reccurent)
    lin = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(REGULARIZATION_RATE))(drop)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(REGULARIZATION_RATE))(lin)
    path = "LSTM"

if i == 3:
    # ============================
    # BERT model (DistilBERT)
    # ============================
    from transformers import TFAutoModel
    from tqdm import tqdm
    bert_model = TFAutoModel.from_pretrained("bert-base-uncased", from_pt=True)
    CACHE_FILE = "preprocessed_texts.npz"
    EMB_FILE = "bert_embeddings.npz"
    from_bert = True
    def compute_bert_embeddings(texts, model_name=TOKENIZER_NAME, batch_size=BATCH_SIZE, max_len=MAX_LEN, pooling="cls", save=False):
        print(f"[INFO] Computing BERT embeddings using {model_name} ...")
        model = TFAutoModel.from_pretrained(model_name, from_pt=True)
        all_embeddings = []
        n = len(texts)
        for start in tqdm(range(0, n, batch_size)):
            batch_texts = list(texts[start:start + batch_size])
            enc = tokenizer(batch_texts,
                            padding="max_length",
                            truncation=True,
                            max_length=max_len,
                            return_tensors="tf")
            outputs = model(enc["input_ids"], attention_mask=enc["attention_mask"], training=False)
            last_hidden = outputs.last_hidden_state
            if pooling == "cls":
                batch_emb = last_hidden[:, 0, :]
            else:
                mask = tf.cast(tf.expand_dims(enc["attention_mask"], -1), tf.float32)
                batch_emb = tf.reduce_sum(last_hidden * mask, axis=1) / tf.reduce_sum(mask, axis=1)
            all_embeddings.append(batch_emb.numpy())
        embeddings = np.concatenate(all_embeddings, axis=0)
        if save:
            np.savez_compressed(EMB_FILE, embeddings=embeddings, labels=labels)
            print(f"[INFO] Saved embeddings to {EMB_FILE} shape={embeddings.shape}")
        return embeddings

    if os.path.exists(EMB_FILE):
        print("[INFO] Loading precomputed BERT embeddings...")
        data = np.load(EMB_FILE, allow_pickle=True)
        embeddings = data["embeddings"]
        labels = data["labels"]
    else:
        start = time.time()
        embeddings = compute_bert_embeddings(texts, save=True)
        end = time.time()
        print(f"[INFO] Embeddings computation took {end - start} seconds")
    X_train = embeddings[train_idx]
    X_val = embeddings[val_idx]
    inputs = tf.keras.layers.Input(shape=(X_train.shape[1],))
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    REGULARIZATION_RATE = 1e-4
    x = tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(REGULARIZATION_RATE))(inputs)
    x0 = tf.keras.layers.Dropout(0.3)(x)
    x1 = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(REGULARIZATION_RATE))(x0)
    x2 = tf.keras.layers.Dropout(0.3)(x1)
    x3 = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(REGULARIZATION_RATE))(x2)
    x4 = tf.keras.layers.Dropout(0.3)(x3)
    x5 = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(REGULARIZATION_RATE))(
        x4)
    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(REGULARIZATION_RATE)
    )(x5)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    path = "BERT"
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE).prefetch(
        tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(
        tf.data.AUTOTUNE)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================
# Training
# ============================
start = time.time()
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)
end = time.time()
training_time = end - start
history = model.history.history
# ============================
# Prediction
# ============================

print("[TEST]", predict_text("Apple presented new iPhone", model, from_bert=from_bert))
model.save(path + "_model.keras")
print("Training time: {:.4f} sec".format(training_time))
epochs = np.linspace(1, EPOCHS, EPOCHS)
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(epochs, history['accuracy'], label='Training Accuracy', marker='o')
ax1.plot(epochs, history['val_accuracy'], label='Validation Accuracy', marker='o')
ax2.plot(epochs, history['loss'], label='Training Loss', marker='o')
ax2.plot(epochs, history['val_loss'], label='Validation Loss', marker='o')
ax1.set(xlabel='Epochs', ylabel='Loss')
ax2.set(xlabel='Epochs', ylabel='Accuracy')
legend1 = ax1.legend(loc='upper right', prop={'size': 8})
legend2 = ax2.legend(loc='upper right', prop={'size': 8})
plt.tight_layout()
plt.show()