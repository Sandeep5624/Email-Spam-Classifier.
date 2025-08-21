# 📧 Email Spam Classifier

This is a Machine Learning project that classifies emails as **Spam** or **Not Spam (Ham)** using Natural Language Processing (NLP) techniques.  
It is built with **Python, Scikit-learn, and Streamlit** for the web interface.

---

## 🚀 Features
- Preprocesses email text using **TF-IDF Vectorization**
- Trains a **Naive Bayes classifier** for spam detection
- Interactive **Streamlit Web App** to test emails in real-time
- Achieves high accuracy on the dataset

---

## 🛠️ Tech Stack
- **Python 3.x**
- **Scikit-learn**
- **Pandas & NumPy**
- **Streamlit** (for UI)
- **Joblib** (for saving models)

---

## 📂 Project Structure
Email-Spam-Classifier/
│-- app.py # Streamlit app
│-- train.py # Script to train model
│-- spam_classifier.pkl # Trained ML model
│-- vectorizer.pkl # TF-IDF vectorizer
│-- streamlit.txt # Dependencies
│-- README.md # Project Documentation

1. Clone the repository:
   git clone https://github.com/Sandeep5624/Email-Spam-Classifier.git
   
   cd Email-Spam-Classifier

Install dependencies:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run app.py

Enter an email text in the input box, and the model will predict whether it is Spam or Not Spam ✅

📊 Dataset
The model was trained on the Spam/Ham dataset containing labeled email texts.

Preprocessing includes:
Removing punctuation & stopwords
Lowercasing
TF-IDF vectorization

🔮 Future Enhancements
Improve accuracy with advanced models (e.g., SVM, LSTM, BERT)
Deploy the app on Streamlit Cloud / Heroku
Support for multiple languages
