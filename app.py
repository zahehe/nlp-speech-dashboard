import os
import tempfile
import streamlit as st
import whisper
import spacy
import nltk
import torch
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS

# Setup NLTK
nltk.download('vader_lexicon', quiet=True)

st.set_page_config(page_title="Voice Insights NLP", layout="wide")
st.title("🎤 Speech Insights & Analysis Dashboard")

@st.cache_resource
def load_models():
    # 1. Speech Recognition
    stt_model = whisper.load_model("base")
    
    # 2. Keyphrase / Entity NLP
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        
    # 3. Direct Summarizer Loading (Bypasses Pipeline Task Registry Errors)
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # 4. Sentiment Analyzer
    sia = SentimentIntensityAnalyzer()
    
    return stt_model, nlp, (tokenizer, model), sia

with st.spinner("Loading NLP Models..."):
    stt_model, nlp, (tokenizer, model), sia = load_models()

def generate_summary(text):
    if not text.strip():
        return "No text available to summarize."
    inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=60, min_length=15, length_penalty=2.0, num_beams=2, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# File Uploader
uploaded_file = st.file_uploader("Upload an Audio File (.wav, .mp3, .m4a)", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(uploaded_file.read())
        temp_audio_path = temp_audio.name

    st.audio(temp_audio_path)

    if st.button("Process Audio"):
        with st.spinner("Transcribing with Whisper..."):
            result = stt_model.transcribe(temp_audio_path)
            transcript = result["text"].strip()

        st.success("Analysis Complete!")
        
        word_count = len(transcript.split())
        char_count = len(transcript)

        # Sentiment
        scores = sia.polarity_scores(transcript)
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = "Positive 😊"
        elif compound <= -0.05:
            sentiment = "Negative 😞"
        else:
            sentiment = "Neutral 😐"

        # Keywords
        doc = nlp(transcript)
        keywords = list({token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop})

        # Summary
        if word_count > 25:
            summary = generate_summary(transcript)
        else:
            summary = transcript if transcript else "No speech detected."

        # Text-To-Speech Output
        tts = gTTS(text=summary, lang='en')
        tts_path = "summary_speech.mp3"
        tts.save(tts_path)

        # Output Layout
        st.divider()
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📝 Transcript")
            st.write(transcript if transcript else "No transcript generated.")
            st.caption(f"**Metrics:** {word_count} words | {char_count} characters")

            st.subheader("📄 Summary")
            st.info(summary)
            
            st.subheader("🔊 Audio Summary (TTS)")
            st.audio(tts_path)

        with col2:
            st.subheader("😊 Sentiment Analysis")
            st.metric(label="Overall Sentiment", value=sentiment, delta=f"Score: {compound:.2f}")

            st.subheader("🔑 Key Concepts")
            st.write(", ".join([f"`{kw}`" for kw in keywords[:10]]) if keywords else "None found")

    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)