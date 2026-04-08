import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv('IMDB Dataset.csv')

df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

df['text'] = df['review'].str.lower()

df_subset = df.sample(10000, random_state=42).reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(
    df_subset['text'], df_subset['sentiment'], test_size=0.2, random_state=42
)

start_bow = time.time()
bow_vec = CountVectorizer(max_features=5000)
X_train_bow = bow_vec.fit_transform(X_train)
X_test_bow = bow_vec.transform(X_test)

model_bow = LinearSVC()
model_bow.fit(X_train_bow, y_train)
y_pred_bow = model_bow.predict(X_test_bow)
acc_bow = accuracy_score(y_test, y_pred_bow)
time_bow = time.time() - start_bow

print(f"BoW + SVM Accuracy: {acc_bow:.4f}")


start_tfidf = time.time()
tfidf_vec = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vec.fit_transform(X_train)
X_test_tfidf = tfidf_vec.transform(X_test)

model_tfidf = LinearSVC()
model_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = model_tfidf.predict(X_test_tfidf)
acc_tfidf = accuracy_score(y_test, y_pred_tfidf)
time_tfidf = time.time() - start_tfidf

print(f"TF-IDF + SVM Accuracy: {acc_tfidf:.4f}")


tokenized_train = [text.split() for text in X_train]
tokenized_test = [text.split() for text in X_test]


start_w2v = time.time()
w2v_model = Word2Vec(
    sentences=tokenized_train,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)
time_w2v_train = time.time() - start_w2v

def get_sentence_vector(words, model):
    vectors = [model.wv[word] for word in words if word in model.wv]
    if not vectors:
        return np.zeros(100)
    return np.mean(vectors, axis=0)

X_train_w2v = np.array([get_sentence_vector(words, w2v_model) for words in tokenized_train])
X_test_w2v = np.array([get_sentence_vector(words, w2v_model) for words in tokenized_test])

model_nn = Sequential([
    Dense(64, activation='relu', input_shape=(100,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_nn.fit(X_train_w2v, y_train, epochs=5, batch_size=32, verbose=0)
loss, acc_w2v = model_nn.evaluate(X_test_w2v, y_test, verbose=0)

print(f"Word2Vec + NN Accuracy: {acc_w2v:.4f}")

methods = ['BoW + SVM', 'TF-IDF + SVM', 'Word2Vec + NN']
accuracies = [acc_bow, acc_tfidf, acc_w2v]
times = [time_bow, time_tfidf, time_w2v_train]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(methods, accuracies, color=['skyblue', 'lightgreen', 'salmon'])
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy Score")
plt.ylim(0.8, 0.9)

plt.subplot(1, 2, 2)
plt.bar(methods, times, color=['skyblue', 'lightgreen', 'salmon'])
plt.title("Training/Vectorization Time")
plt.ylabel("Seconds")

plt.tight_layout()
plt.show()
