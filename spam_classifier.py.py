# Spam Email Classifier using TF-IDF and Naive Bayes

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ------------------------------
# STEP 1: Create Sample Dataset
# ------------------------------

data = {
    'label': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam'],
    'message': [
        'Hello, how are you?',
        'Win money now!!!',
        'Let us meet tomorrow',
        'Congratulations! You won a prize',
        'Are you coming to class?',
        'Claim your free gift now'
    ]
}

df = pd.DataFrame(data)

# ------------------------------
# STEP 2: Split Data
# ------------------------------

X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ------------------------------
# STEP 3: Convert Text to TF-IDF
# ------------------------------

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ------------------------------
# STEP 4: Train Naive Bayes Model
# ------------------------------

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ------------------------------
# STEP 5: Test Accuracy
# ------------------------------

y_pred = model.predict(X_test_tfidf)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# ------------------------------
# STEP 6: Predict New Message
# ------------------------------

print("\nEnter a message to check if it is Spam or Ham:")
new_message = input()

new_message_tfidf = vectorizer.transform([new_message])
prediction = model.predict(new_message_tfidf)

print("Prediction:", prediction[0])
