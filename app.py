import streamlit as st
import azure.cognitiveservices.speech as speechsdk
import requests
import json
import base64

# -------------------------
# 🔐 YOUR KEYS
# -------------------------
SPEECH_KEY = "577IXgzDaBsgyxHGxDjcxVKHlhW7MpTz6EApueKJNAdVVo89Ew9RJQQJ99BKAC3pKaRXJ3w3AAAYACOGxpcA"
SPEECH_REGION = "eastasia"

OPENROUTER_API_KEY = "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"   # ← put your key here
OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct"

# -------------------------
# 🎤 Azure Speech-to-Text
# -------------------------
def azure_stt():
    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION
    )
    speech_config.speech_recognition_language = "ur-PK"  # or auto detect using language auto detection config

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    st.info("🎙️ Listening...")
    result = recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    else:
        return "Speech not recognized."

# -------------------------
# 🤖 OpenRouter LLM
# -------------------------
def openrouter_chat(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, json=payload, headers=headers)
    data = response.json()

    return data["choices"][0]["message"]["content"]

# -------------------------
# 🔊 Azure Text-to-Speech
# -------------------------
def azure_tts(text):
    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION
    )
    speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"

    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

    result = synthesizer.speak_text_async(text).get()
    audio_data = result.audio_data

    return audio_data

# -------------------------
# 🖥️ Streamlit UI
# -------------------------
st.title("🎤 OpenRouter + Azure Voice Chatbot")

if st.button("Click to Speak"):
    text = azure_stt()
    st.write("🗣️ You said:", text)

    if text.strip():
        answer = openrouter_chat(text)
        st.write("🤖 AI:", answer)

        audio_bytes = azure_tts(answer)
        st.audio(audio_bytes, format="audio/wav")
