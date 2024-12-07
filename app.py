import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Model va vectorizerni yuklash
with open("amazon.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Matnni tozalash funksiyasi
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Belgilarni olib tashlash
    text = text.lower()  # Kichik harflar
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Streamlit interfeysi
st.title("Sentiment Analysis App")
st.write("Bu dastur Amazon Fine Food sharhlari uchun sentimentni (ijobiy yoki salbiy) aniqlashga yordam beradi.")

# Foydalanuvchi kiriti
user_input = st.text_area("Sharhingizni yozing:", "")

if st.button("Natijani ko‘rish"):
    if user_input:
        # Matnni tozalash
        cleaned_input = clean_text(user_input)
        
        # TF-IDF xususiyatlariga aylantirish
        input_vectorized = vectorizer.transform([cleaned_input])
        
        # Sentimentni bashorat qilish
        prediction = model.predict(input_vectorized)
        sentiment = "Ijobiy" if prediction[0] == 1 else "Salbiy"
        
        # Natijani chiqarish
        st.subheader(f"Natija: {sentiment}")
    else:
        st.error("Iltimos, sharh matnini kiriting!")
