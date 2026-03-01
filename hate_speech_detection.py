"""
Hate Speech Detection - NLP Assignment
=======================================
Uses: Pandas, NLTK, Scikit-learn

Tasks covered:
  1. Data Loading and Exploration
  2. Text Preprocessing (Tokenization, Stopword Removal, Lemmatization)
  3. Feature Extraction (BoW and TF-IDF)
  4. Model Training and Evaluation

Outputs:
  - solution.csv: Predictions for the test dataset
  - class_distribution.png: Bar plot of HS class distribution
  - Console output: EDA stats, classification reports, analysis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import re
import warnings
warnings.filterwarnings('ignore')

# NLTK imports
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Scikit-learn imports
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score


# ============================================================
# TASK 1: Data Loading and Exploration 
# ============================================================
print("=" * 70)
print("TASK 1: Data Loading and Exploration")
print("=" * 70)

# (a) Load the dataset
train_df = pd.read_csv('/home/prabal/Downloads/train_data.csv')
test_df = pd.read_csv('/home/prabal/Downloads/test.csv')

print("\n--- (a) First few rows of the training dataset ---")
print(train_df.head(10))

print(f"\nTraining dataset shape: {train_df.shape}")
print(f"Test dataset shape: {test_df.shape}")
print(f"\nColumn names: {list(train_df.columns)}")
print(f"\nData types:\n{train_df.dtypes}")
print(f"\nMissing values:\n{train_df.isnull().sum()}")

# (b) EDA: Class distribution
print("\n--- (b) Class Distribution ---")
class_counts = train_df['HS'].value_counts().sort_index()
print(f"\nHS = 0 (Not Hate Speech): {class_counts[0]} ({class_counts[0]/len(train_df)*100:.1f}%)")
print(f"HS = 1 (Hate Speech):     {class_counts[1]} ({class_counts[1]/len(train_df)*100:.1f}%)")
print(f"Total samples:            {len(train_df)}")

# Visualize class distribution using a bar plot
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#2ecc71', '#e74c3c']
bars = ax.bar(['Not Hate Speech (HS=0)', 'Hate Speech (HS=1)'],
              [class_counts[0], class_counts[1]],
              color=colors, edgecolor='black', linewidth=0.8)
for bar, count in zip(bars, [class_counts[0], class_counts[1]]):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 30,
            f'{count}\n({count/len(train_df)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=12)
ax.set_xlabel('Class', fontsize=13)
ax.set_ylabel('Count', fontsize=13)
ax.set_title('Distribution of Hate Speech Classes in Training Data', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(class_counts) * 1.2)
plt.tight_layout()
plt.savefig('/home/prabal/Downloads/class_distribution.png', dpi=150)
plt.close()
print("\nBar plot saved to: /home/prabal/Downloads/class_distribution.png")


# ============================================================
# TASK 2: Text Preprocessing 
# ============================================================
print("\n" + "=" * 70)
print("TASK 2: Text Preprocessing")
print("=" * 70)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    """
    Preprocess a single text string:
      (a) Tokenization using NLTK's word_tokenize
      (b) Stop word removal using NLTK's stopwords
      (c) Lemmatization using NLTK's WordNetLemmatizer

    Choice of Lemmatization over Stemming:
    - Lemmatization produces valid dictionary words (e.g., "running" -> "run")
      while stemming may produce non-words (e.g., "running" -> "run", but
      "studies" -> "studi").
    - Lemmatization considers the context and part of speech, leading to
      more meaningful word forms.
    - For hate speech detection, preserving word meaning is important for
      accurate classification.
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove mentions and hashtags symbols (keep the word)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # (a) Tokenization
    tokens = word_tokenize(text)

    # (b) Stop word removal
    tokens = [token for token in tokens if token not in stop_words]

    # (c) Lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Filter out very short tokens
    tokens = [token for token in tokens if len(token) > 1]

    return ' '.join(tokens)


# Apply preprocessing to train and test data
print("\nApplying preprocessing to training data...")
train_df['processed_text'] = train_df['text'].apply(preprocess_text)

print("Applying preprocessing to test data...")
test_df['processed_text'] = test_df['text'].apply(preprocess_text)

# Show examples of preprocessing
print("\n--- Preprocessing Examples ---")
for i in range(3):
    print(f"\nOriginal:  {train_df['text'].iloc[i][:100]}...")
    print(f"Processed: {train_df['processed_text'].iloc[i][:100]}...")

print(f"\n--- Why Lemmatization over Stemming? ---")
print("""
Lemmatization was chosen over stemming because:
1. It produces valid dictionary words (e.g., 'studies' -> 'study', not 'studi').
2. It considers the morphological analysis of words, preserving meaning.
3. For hate speech detection, maintaining word semantics is crucial for
   the classifier to learn meaningful patterns.
4. Though slightly slower than stemming, the accuracy improvement justifies it.
""")


# ============================================================
# TASK 3: Feature Extraction 
# ============================================================
print("=" * 70)
print("TASK 3: Feature Extraction")
print("=" * 70)

X_train_text = train_df['processed_text']
y_train = train_df['HS']
X_test_text = test_df['processed_text']

# Split training data for evaluation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_text, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# --- (a) Bag-of-Words (BoW) ---
print("\n--- (a) Bag-of-Words Representation ---")

# Experiment with different max_features
for max_feat in [5000, 10000, 15000]:
    bow_temp = CountVectorizer(max_features=max_feat)
    bow_temp.fit(X_tr)
    print(f"  max_features={max_feat}: vocabulary size = {len(bow_temp.vocabulary_)}")

# Use optimal parameters
bow_vectorizer = CountVectorizer(max_features=10000, min_df=2, max_df=0.95)
X_tr_bow = bow_vectorizer.fit_transform(X_tr)
X_val_bow = bow_vectorizer.transform(X_val)
X_train_bow_full = bow_vectorizer.fit_transform(X_train_text)
X_test_bow = bow_vectorizer.transform(X_test_text)

print(f"\nFinal BoW parameters: max_features=10000, min_df=2, max_df=0.95")
print(f"  Training BoW matrix shape: {X_tr_bow.shape}")
print(f"  Validation BoW matrix shape: {X_val_bow.shape}")

# Re-fit on split data for evaluation
bow_vectorizer_eval = CountVectorizer(max_features=10000, min_df=2, max_df=0.95)
X_tr_bow = bow_vectorizer_eval.fit_transform(X_tr)
X_val_bow = bow_vectorizer_eval.transform(X_val)


# --- (b) TF-IDF ---
print("\n--- (b) TF-IDF Representation ---")

# Experiment with different ngram_range values
print("\nExperimenting with different ngram_range values:")
for ngram in [(1, 1), (1, 2), (1, 3)]:
    tfidf_temp = TfidfVectorizer(max_features=10000, ngram_range=ngram)
    X_temp = tfidf_temp.fit_transform(X_tr)
    print(f"  ngram_range={ngram}: feature matrix shape = {X_temp.shape}")

# Use optimal parameters
tfidf_vectorizer_eval = TfidfVectorizer(
    max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.95
)
X_tr_tfidf = tfidf_vectorizer_eval.fit_transform(X_tr)
X_val_tfidf = tfidf_vectorizer_eval.transform(X_val)

# Full training set TF-IDF for final predictions
tfidf_vectorizer_full = TfidfVectorizer(
    max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.95
)
X_train_tfidf_full = tfidf_vectorizer_full.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer_full.transform(X_test_text)

print(f"\nFinal TF-IDF parameters: max_features=10000, ngram_range=(1,2), min_df=2, max_df=0.95")
print(f"  Training TF-IDF matrix shape: {X_tr_tfidf.shape}")
print(f"  Validation TF-IDF matrix shape: {X_val_tfidf.shape}")

print(f"\n--- Impact of ngram_range ---")
print("""
- (1,1) Unigrams only: Captures individual word frequencies. Simple but
  misses phrases like 'not good' or 'go home'.
- (1,2) Unigrams + Bigrams: Captures two-word combinations, improving
  context understanding. Better for hate speech detection as many slurs
  and hateful phrases are multi-word.
- (1,3) Up to Trigrams: Captures longer phrases but increases feature
  dimensionality significantly, risking overfitting with limited data.

We chose (1,2) as the optimal balance between capturing contextual
patterns and avoiding excessive dimensionality.
""")


# ============================================================
# TASK 4: Model Training and Evaluation 
# ============================================================
print("=" * 70)
print("TASK 4: Model Training and Evaluation")
print("=" * 70)

# --- (a) Train classifiers on both BoW and TF-IDF features ---

classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes': MultinomialNB(),
    'Linear SVM': LinearSVC(max_iter=2000, random_state=42),
}

results = {}

print("\n--- Training and Evaluating Models ---\n")

for name, clf in classifiers.items():
    print(f"{'='*50}")
    print(f"Classifier: {name}")
    print(f"{'='*50}")

    # Train on BoW features
    clf_bow = type(clf)(**clf.get_params())
    clf_bow.fit(X_tr_bow, y_tr)
    y_pred_bow = clf_bow.predict(X_val_bow)
    acc_bow = accuracy_score(y_val, y_pred_bow)

    print(f"\n  [BoW Features] Accuracy: {acc_bow:.4f}")
    print(f"  Classification Report (BoW):")
    print(classification_report(y_val, y_pred_bow, target_names=['Not HS (0)', 'HS (1)'], indent=4))

    # Train on TF-IDF features
    clf_tfidf = type(clf)(**clf.get_params())
    clf_tfidf.fit(X_tr_tfidf, y_tr)
    y_pred_tfidf = clf_tfidf.predict(X_val_tfidf)
    acc_tfidf = accuracy_score(y_val, y_pred_tfidf)

    print(f"  [TF-IDF Features] Accuracy: {acc_tfidf:.4f}")
    print(f"  Classification Report (TF-IDF):")
    print(classification_report(y_val, y_pred_tfidf, target_names=['Not HS (0)', 'HS (1)'], indent=4))

    results[name] = {
        'bow_accuracy': acc_bow,
        'tfidf_accuracy': acc_tfidf,
        'clf_bow': clf_bow,
        'clf_tfidf': clf_tfidf,
    }

# --- Summary Table ---
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"\n{'Classifier':<25} {'BoW Accuracy':>15} {'TF-IDF Accuracy':>18}")
print("-" * 60)
for name, res in results.items():
    print(f"{name:<25} {res['bow_accuracy']:>15.4f} {res['tfidf_accuracy']:>18.4f}")

# --- Discussion: BoW vs TF-IDF ---
print(f"\n--- Impact of BoW vs TF-IDF on Performance ---")
print("""
BoW (Bag-of-Words):
  - Simply counts word occurrences — common words dominate.
  - May give high weight to frequent but non-discriminative words.

TF-IDF (Term Frequency-Inverse Document Frequency):
  - Weights terms by their importance — reduces impact of common words.
  - Highlights rare but discriminative terms (e.g., specific slurs).
  - Generally performs better for hate speech detection because hateful
    terms tend to be distinctive and TF-IDF amplifies their signal.

Conclusion: TF-IDF typically provides better or comparable performance
to BoW for text classification tasks, especially when combined with
bigrams (ngram_range=(1,2)) to capture multi-word patterns.
""")


# ============================================================
# Generate solution.csv with predictions for the test dataset
# ============================================================
print("=" * 70)
print("GENERATING SOLUTION FILE")
print("=" * 70)

# Select the best model - use Logistic Regression with TF-IDF (generally robust)
best_model_name = max(results.keys(), key=lambda k: results[k]['tfidf_accuracy'])
print(f"\nBest model based on TF-IDF accuracy: {best_model_name}")
print(f"Validation accuracy: {results[best_model_name]['tfidf_accuracy']:.4f}")

# Train the best model on the FULL training set with TF-IDF
best_clf = type(results[best_model_name]['clf_tfidf'])(
    **results[best_model_name]['clf_tfidf'].get_params()
)
best_clf.fit(X_train_tfidf_full, y_train)

# Predict on test data
test_predictions = best_clf.predict(X_test_tfidf)

# Create solution dataframe
solution_df = pd.DataFrame({
    'id': test_df['id'],
    'HS': test_predictions
})

# Save solution
solution_df.to_csv('/home/prabal/Downloads/solution.csv', index=False)
print(f"\nSolution saved to: /home/prabal/Downloads/solution.csv")
print(f"Total test predictions: {len(solution_df)}")
print(f"Predicted distribution:")
print(f"  HS = 0: {(test_predictions == 0).sum()} ({(test_predictions == 0).mean()*100:.1f}%)")
print(f"  HS = 1: {(test_predictions == 1).sum()} ({(test_predictions == 1).mean()*100:.1f}%)")

print("\n" + "=" * 70)
print("ALL TASKS COMPLETED SUCCESSFULLY!")
print("=" * 70)
