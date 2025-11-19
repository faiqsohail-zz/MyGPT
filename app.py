import streamlit as st
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()

# -----------------------------
# Azure Credentials
# -----------------------------
SPEECH_KEY = os.getenv("SPEECH_KEY")
SPEECH_REGION = os.getenv("SPEECH_REGION")

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
OPENAI_DEPLOYMENT = os.getenv("OPENAI_DEPLOYMENT")  # YOUR GPT DEPLOYMENT NAME

# Azure OpenAI Client
client = AzureOpenAI(
    api_key=OPENAI_KEY,
    azure_endpoint=OPENAI_ENDPOINT,
    api_version="2024-05-01-preview"
)

# -----------------------------
# Speech-to-Text (Auto-Detect Language)
# -----------------------------
def azure_stt_from_bytes(audio_bytes):

    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION
    )

    # 🔥 Enable auto language detection  
    auto_detect_source_language_config = (
        speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=["en-US", "ur-PK", "ar-SA", "hi-IN"]
        )
    )

    # Create push stream
    stream = speechsdk.audio.PushAudioInputStream()
    stream.write(audio_bytes)
    stream.close()

    audio_config = speechsdk.AudioConfig(stream=stream)

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
        auto_detect_source_language_config=auto_detect_source_language_config
    )

    result = recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text

    return "Could not recognize speech."


# -----------------------------
# LLM Chat
# -----------------------------
def chat_llm(user_msg):
    response = client.chat.completions.create(
        model=OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a multilingual helpful assistant."},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=300
    )
    return response.choices[0].message.content


# -----------------------------
# Text-to-Speech (Auto Voice Based on Language)
# -----------------------------
def auto_voice_for_language(text):

    if any(ch in text for ch in "اأإءؤئة"):
        return "ar-SA-HamedNeural"
    if any(ch in text for ch in "ںچڈڑے"):
        return "ur-PK-UzmaNeural"
    if any(ch in text for ch in "अआइईउऊ"):
        return "hi-IN-MadhurNeural"
    return "en-US-JennyNeural"


def azure_tts(text):

    voice = auto_voice_for_language(text)

    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION
    )

    speech_config.speech_synthesis_voice_name = voice

    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    audio_config = speechsdk.audio.AudioOutputConfig(filename=temp_wav)

    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=audio_config
    )

    synthesizer.speak_text_async(text).get()

    return temp_wav


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎙️ Azure Speech → GPT → Voice Assistant (Auto Language)")

audio_data = st.audio_input("Click to record your voice")

if audio_data:
    st.write("🔄 Processing...")

    audio_bytes = audio_data.getvalue()

    # Step 1: Speech-to-Text
    text = azure_stt_from_bytes(audio_bytes)
    st.success(f"🗣️ You said: {text}")

    # Step 2: GPT Response
    reply = chat_llm(text)
    st.info(f"🤖 GPT: {reply}")

    # Step 3: Text-to-Speech
    wav_file = azure_tts(reply)

    with open(wav_file, "rb") as f:
        st.audio(f.read(), format="audio/wav")
