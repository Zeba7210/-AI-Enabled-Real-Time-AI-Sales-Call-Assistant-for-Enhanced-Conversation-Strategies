pip install gradio whisper transformers torch sentencepiece


import gradio as gr
import whisper
import torch
from transformers import pipeline
import re




print("Loading AI models...")

# Whisper STT
whisper_model = whisper.load_model("small")

# Sentiment
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

# NER
ner_model = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    grouped_entities=True
)

# Intent
intent_model = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

print("✅ All models loaded successfully")







def speech_to_text(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]







def speaker_diarization(transcript):
    sentences = re.split(r'[.?!]', transcript)
    data = []

    for i, s in enumerate(sentences):
        if len(s.strip()) < 3:
            continue

        speaker = "Sales Rep" if i % 2 == 0 else "Customer"

        data.append({
            "speaker": speaker,
            "sentence": s.strip()
        })

    return data






def nlp_pipeline(data):
    intents = [
        "greeting",
        "product inquiry",
        "price negotiation",
        "feature inquiry",
        "purchase intent",
        "objection",
        "closing"
    ]

    output = []

    for row in data:
        text = row["sentence"]

        sentiment = sentiment_model(text)[0]
        intent = intent_model(text, intents)
        entities = ner_model(text)

        output.append({
            "speaker": row["speaker"],
            "sentence": text,
            "sentiment": sentiment["label"],
            "intent": intent["labels"][0],
            "entities": entities
        })

    return output








CRM_PROFILE = {
    "customer_name": "Rahul Sharma",
    "location": "Bangalore",
    "company": "ABC Tech",
    "past_purchases": ["Basic CRM"],
    "budget_range": "₹50,000 – ₹80,000"
}











def ai_reasoning(nlp_data):
    next_question = "Can you tell me more about your needs?"
    objection_handling = "Explain value and benefits."
    product = "Standard CRM Plan"
    insight = "Customer is exploring options."

    for row in nlp_data:
        if row["intent"] == "price negotiation":
            objection_handling = "Offer discount, EMI, or annual plan."
        if row["intent"] == "purchase intent":
            next_question = "Shall I proceed with the booking?"
            product = "Enterprise CRM Plan"
        if row["intent"] == "objection":
            insight = "Customer has concerns – needs reassurance."

    return {
        "CRM_Profile": CRM_PROFILE,
        "Next_Question": next_question,
        "Objection_Handling": objection_handling,
        "Product_Recommendation": product,
        "Sales_Insight": insight
    }











def run_full_pipeline(audio):
    transcript = speech_to_text(audio)
    diarized = speaker_diarization(transcript)
    nlp_result = nlp_pipeline(diarized)
    reasoning = ai_reasoning(nlp_result)

    return transcript, nlp_result, reasoning











with gr.Blocks(title="AI Sales Call Assistant") as app:
    gr.Markdown("# 🎧 AI Sales Call Intelligence System")

    with gr.Tab("📁 Upload Call Recording"):
        upload_audio = gr.Audio(type="filepath", label="Upload WAV Call")
        upload_btn = gr.Button("▶ Analyze Uploaded Call")

    with gr.Tab("🎤 Live Call Recording"):
        live_audio = gr.Audio(type="filepath", sources=["microphone"], label="Record Live Call")
        live_btn = gr.Button("▶ Analyze Live Call")

    transcript_box = gr.Textbox(label="📜 Transcript", lines=6)
    nlp_box = gr.JSON(label="📊 Milestone 3: Sentiment + Intent + NER")
    reasoning_box = gr.JSON(label="🧠 Milestone 4: AI Reasoning Layer")

    upload_btn.click(run_full_pipeline, upload_audio,
                     [transcript_box, nlp_box, reasoning_box])

    live_btn.click(run_full_pipeline, live_audio,
                   [transcript_box, nlp_box, reasoning_box])

app.launch(share=True)
