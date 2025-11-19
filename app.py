import streamlit as st
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# --------------------------
# Azure Credentials
# --------------------------

# Azure Speech Service
speech_key = os.getenv("SPEECH_KEY")
speech_region = os.getenv("SPEECH_REGION")

# Azure OpenAI
openai_endpoint = os.getenv("OPENAI_ENDPOINT")
openai_key = os.getenv("OPENAI_API_KEY")
deployment_name = os.getenv("OPENAI_DEPLOYMENT")


client = AzureOpenAI(
    api_key=openai_key,
    azure_endpoint=openai_endpoint,
    api_version="2024-05-01-preview"
)


# --------------------------
# Auto Language Detect STT
# --------------------------

def azure_stt_from_audio_bytes(audio_bytes):
    # Speech config
    speech_config = speechsdk.SpeechConfig(
        subscription=speech_key,
        region=speech_region
    )

    # AUTO DETECT language
    auto_detect = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
        languages=["en-US", "ur-PK", "ar-SA", "hi-IN"]
    )

    # Stream for audio bytes
    stream = speechsdk.audio.PushAudioInputStream()
    stream.write(audio_bytes)
    stream.close()

    audio_config = speechsdk.AudioConfig(stream=stream)

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
        auto_detect_source_language_config=auto_detect
    )

    result = recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        detected_lang = result.properties[
            speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
        ]
        return result.text, detected_lang

    return "Speech not recognized.", "unknown"


# --------------------------
# GPT Conversation
# --------------------------

def chat_llm(text, detected_lang):
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {
                "role": "system",
                "content": f"You are a multilingual assistant. Reply in the same language as the user. User is speaking: {detected_lang}."
            },
            {"role": "user", "content": text},
        ]
    )
    return response.choices[0].message.content


# --------------------------
# Azure Text To Speech (auto-language voice)
# --------------------------

def azure_tts(text, language):
    speech_config = speechsdk.SpeechConfig(
        subscription=speech_key,
        region=speech_region
    )

    # Dynamic multilingual voices
    voice_map = {
        "en-US": "en-US-JennyNeural",
        "ur-PK": "ur-PK-UzmaNeural",
        "ar-SA": "ar-SA-HamedNeural",
        "hi-IN": "hi-IN-SwaraNeural"
    }

    voice = voice_map.get(language, "en-US-JennyNeural")
    speech_config.speech_synthesis_voice_name = voice

    audio_config = speechsdk.audio.AudioOutputConfig(
        filename="response.wav"
    )

    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    synthesizer.speak_text_async(text).get()

    return "response.wav"


# --------------------------
# Streamlit UI
# --------------------------

st.title("🎤 Azure STT + GPT + TTS (Multilingual Auto Detect)")
st.write("Speak in **English, Urdu, Arabic, or Hindi** — the system detects automatically.")

audio_input = st.audio_input("Click to record your voice")

if audio_input:
    st.write("⏳ Processing speech...")
    audio_bytes = audio_input.getvalue()

    # Speech to Text
    text, lang = azure_stt_from_audio_bytes(audio_bytes)

    st.success(f"🗣 You said ({lang}): {text}")

    # LLM response
    response = chat_llm(text, lang)
    st.info(f"🤖 GPT: {response}")

    # Text-to-Speech
    audio_file = azure_tts(response, lang)

    st.audio(open(audio_file, "rb").read(), format="audio/wav")
