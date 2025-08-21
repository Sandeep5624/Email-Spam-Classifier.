# email_spam_classifier.py

import pandas as pd
import string
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --------------------------
# 1. Load Dataset
# --------------------------
# You can replace this with your own dataset (CSV with 'text' and 'label')
# Example: Kaggle SMS Spam Collection
df = pd.read_csv("email_spam (1).csv", encoding='latin-1')

# For Kaggle SMS dataset, keep only required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# --------------------------
# 2. Preprocessing
# --------------------------
def clean_text(text):
    text = text.lower()                           # lowercase
    text = re.sub(r'\d+', '', text)               # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = text.strip()                           # remove extra spaces
    return text

df['text'] = df['text'].apply(clean_text)

# Encode labels (ham = 0, spam = 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# --------------------------
# 3. Split dataset
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# --------------------------
# 4. Feature Extraction (TF-IDF)
# --------------------------
vectorizer = TfidfVectorizer(max_features=3000)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# --------------------------
# 5. Train Model
# --------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# --------------------------
# 6. Evaluate
# --------------------------
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --------------------------
# 7. Prediction Demo
# --------------------------
def predict_email(email):
    email = clean_text(email)
    email_vec = vectorizer.transform([email])
    prediction = model.predict(email_vec)[0]
    return "Spam" if prediction == 1 else "Not Spam"

# Example Predictions
print("\nTest Predictions:")
print("👉", predict_email("Congratulations! You have won $1000 cash prize. Claim now!"))
print("👉", predict_email("Hi, let’s meet for lunch tomorrow."))



# Save the model and vectorizer
joblib.dump(model, "spam_classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model & Vectorizer saved successfully!")

