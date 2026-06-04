import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Dataset
data = pd.read_csv("emails.csv")

# Features and Labels
X = data["text"]
y = data["label"]

# Convert text into numerical features
vectorizer = TfidfVectorizer()
X_features = vectorizer.fit_transform(X)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

# User Input
email = input("\nEnter Email Text: ")

email_features = vectorizer.transform([email])

result = model.predict(email_features)

print("\nPrediction:", result[0])
