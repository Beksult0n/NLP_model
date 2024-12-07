import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
import nltk

# Model va vectorizerni yuklash
with open("amazon.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Belgilarni olib tashlash
    text = text.lower()  # Kichik harflar
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Streamlit interfeysi
st.title("Sharhlarni ijobiy yoki salbiyligini aniqlash")
st.write("Bu dastur Amazon Fine Food sharhlari uchun sentimentni (ijobiy yoki salbiy) aniqlashga yordam beradi.")

# Faylni yuklash (txt formatida)
uploaded_file = st.file_uploader("Sharhlar faylini yuklang (txt formatida)", type=["txt"])

if uploaded_file is not None:
    # Faylni o'qish
    text_data = uploaded_file.getvalue().decode("utf-8")  # Faylni matnga aylantirish
    
    # Fayl tarkibini ko‘rsatish (birinchi 500 ta belgi)
    st.write("Fayl tarkibi (boshqa belgilarga qarang):")
    st.text(text_data[:500])  # Faylning birinchi 500 ta belgisi

    # Fayldagi har bir izohni tahlil qilish
    # Agar faylda bir nechta satr bo'lsa, ularni qator bo‘yicha ajratamiz
    reviews = text_data.split("\n")
    
    results = []
    for review in reviews:
        review = review.strip()  # Har qanday ortiqcha bo'sh joylarni olib tashlash
        if review:  # Faqat bo'sh bo'lmagan satrlar bilan ishlash
            cleaned_text = clean_text(review)
            
            # TF-IDF vektoriga aylantirish
            input_vectorized = vectorizer.transform([cleaned_text])
            
            # Sentimentni bashorat qilish
            prediction = model.predict(input_vectorized)
            sentiment = "Ijobiy" if prediction[0] == 1 else "Salbiy"
            
            results.append((review, sentiment))
    
    # Natijalarni chiqarish
    if results:
        st.write("Sentiment natijalari:")
        for review, sentiment in results:
            st.write(f"Sharh: {review}")
            st.write(f"Sentiment: {sentiment}")
    else:
        st.write("Faylda faqat bo'sh satrlar bor.")
else:
    # Foydalanuvchi matn kiritishi uchun qism
    user_input = st.text_area("Sharhingizni yozing:", "")
    
    if st.button("Natijani ko'rish"):
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
