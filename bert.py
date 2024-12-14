import streamlit as st
from transformers import pipeline
import nltk
import re
from nltk.corpus import stopwords

# BERT modeli yordamida sentiment analiz
sentiment_model = pipeline("sentiment-analysis")

nltk.download('stopwords')
stop_words = set(stopwords.words('english')) - {'not'}

# Almashtirish uchun lug‘at
replacement_dict = {
    "not good": "bad",
    "don't like": "dislike"
}

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Belgilarni olib tashlash
    text = text.lower()  # Kichik harflar
    words = text.split()
    for old, new in replacement_dict.items():     
        text = text.replace(old, new)
    text = ' '.join(word for word in text.split() if word not in stop_words)    
    return " ".join(words)

# Streamlit interfeysi
st.title("Sharhlarni ijobiy yoki salbiyligini aniqlash")
st.write("Bu dastur BERT modeli yordamida sharhlarning sentimentini (ijobiy yoki salbiy) aniqlashga yordam beradi.")

# Faylni yuklash (txt formatida)
uploaded_file = st.file_uploader("Sharhlar faylini yuklang (txt formatida)", type=["txt"])

if uploaded_file is not None:
    # Faylni o'qish
    text_data = uploaded_file.getvalue().decode("utf-8")  # Faylni matnga aylantirish
    
    # Fayl tarkibini ko‘rsatish (birinchi 500 ta belgi)
    st.write("Fayl tarkibi (boshqa belgilarga qarang):")
    st.text(text_data[:500])

    reviews = text_data.split("\n")
    
    results = []
    for review in reviews:
        review = review.strip()  # Har qanday ortiqcha bo'sh joylarni olib tashlash
        if review:  # Faqat bo'sh bo'lmagan satrlar bilan ishlash
            cleaned_text = clean_text(review)
            
            # Sentimentni bashorat qilish
            result = sentiment_model(cleaned_text)
            sentiment = result[0]['label']
            
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
            
            # Sentimentni bashorat qilish
            result = sentiment_model(cleaned_input)
            sentiment = result[0]['label']
            
            # Natijani chiqarish
            st.subheader(f"Natija: {sentiment}")
        else:
            st.error("Iltimos, sharh matnini kiriting!")
