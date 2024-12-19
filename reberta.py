import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import nltk
import re
from nltk.corpus import stopwords

# RoBERTa modelini va tokenizerni yuklash
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base')

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
    text = ' '.join(word for word in words if word not in stop_words)    
    return " ".join(words)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    sentiment = "Positive" if predicted_class == 1 else "Negative"
    return sentiment

# Streamlit interfeysi
st.title("Sharhlarni ijobiy yoki salbiyligini aniqlash")
st.write("Bu dastur Amazon Fine Food sharhlari uchun RoBERTa modelidan foydalanib sentimentni (ijobiy yoki salbiy) aniqlashga yordam beradi.")

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
            sentiment = predict_sentiment(cleaned_text)
            
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
            sentiment = predict_sentiment(cleaned_input)
            
            # Natijani chiqarish
            st.subheader(f"Natija: {sentiment}")
        else:
            st.error("Iltimos, sharh matnini kiriting!")
