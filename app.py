import streamlit as st
import joblib
import pandas as pd

# Load the saved model and vectorizer
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(page_title="📧 Email Spam Classifier", layout="centered")

st.title("📧 Email Spam Classifier")
st.write("This app predicts whether an email is **Spam** or **Not Spam**.")

# ---- Single Email Prediction ----
st.subheader("✉️ Test a Single Email")

user_input = st.text_area("Enter email text here:", height=150)

if st.button("Predict"):
    if user_input.strip() != "":
        # Transform the input text before prediction
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        result = "🚨 Spam" if prediction == 1 else "✅ Not Spam"
        st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter some email text to classify.")

# ---- Bulk Email Prediction from CSV ----
st.subheader("📂 Test with Dataset (Upload CSV)")

uploaded_file = st.file_uploader("Upload a CSV file with emails", type=["csv"])

if uploaded_file is not None:
    # Try reading with utf-8, fallback to latin-1
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding="latin-1")

    st.write("📄 Uploaded Dataset Preview:")
    st.dataframe(df.head())

    # Let user select which column contains the email/text data
    text_column = st.selectbox(
        "Select the column containing email/text data:",
        options=df.columns.tolist()
    )

    if text_column:
        # Transform all emails before prediction
        email_vecs = vectorizer.transform(df[text_column].astype(str))
        df["Prediction"] = model.predict(email_vecs)
        df["Prediction"] = df["Prediction"].map({1: "🚨 Spam", 0: "✅ Not Spam"})

        st.write("🔎 Results:")
        st.dataframe(df)

        # Optionally download results
        csv_out = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Predictions as CSV",
            data=csv_out,
            file_name="predictions.csv",
            mime="text/csv",
        )
