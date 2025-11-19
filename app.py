import streamlit as st
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI
import tempfile

# -----------------------------
#  YOUR AZURE SECRETS (NO ENV)
# -----------------------------
SPEECH_KEY = "577IXgzDaBsgyxHGxDjcxVKHlhW7MpTz6EApueKJNAdVVo89Ew9RJQQJ99BKAC3pKaRXJ3w3AAAYACOGxpcA"
SPEECH_REGION = "eastasia"

OPENAI_API_KEY = "AktWqc6lXxksNEDOc3dlzwT4GI6zCKdDfqZDg0olzNng1FHXWmojJQQJ99BKACqBBLyXJ3w3AAABACOGKcQ2"
OPENAI_ENDPOINT = "https://navttcopenai.openai.azure.com/"
OPENAI_DEPLOYMENT = "NAVTTCOPENAI"

# -----------------------------
#  Azure OpenAI Client
# -----------------------------
client = AzureOpenAI(
    api_key=OPENAI_API_KEY,
    api_version="2024-05-01-preview",
    azure_endpoint=OPENAI_ENDPOINT,
)

# -----------------------------
#  Speech-to-Text (Auto Lang)
# -----------------------------
def azure_stt_from_audio(audio_bytes):
    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION
    )

    # Auto language detection (adds Urdu, English, Arabic)
    auto_detect = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
        languages=["en-US", "ur-PK", "ar-SA"]
    )

    # Create stream
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
        return result.text
    else:
        return "Speech not recognized."


# -----------------------------
#  ChatGPT (Azure OpenAI)
# -----------------------------
def chat_llm(msg):
    response = client.chat.completions.create(
        model=OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a multilingual helpful AI assistant."},
            {"role": "user", "content": msg},
        ]
    )
    return response.choices[0].message.content


# -----------------------------
#  Text-to-Speech
# -----------------------------
def azure_tts(text):
    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION
    )
    speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"

    # Save to temporary wav file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio_config = speechsdk.audio.AudioOutputConfig(filename=tmp.name)

        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=audio_config
        )
        synthesizer.speak_text_async(text).get()

        return tmp.name


# -----------------------------
#  STREAMLIT UI
# -----------------------------
st.title("🎤 Azure Voice → GPT → Voice Chatbot (Auto-Language)")

audio_input = st.audio_input("Click to record your voice 🎙️")

if audio_input:
    st.write("⏳ Processing your voice...")
    audio_bytes = audio_input.getvalue()

    # STT
    text = azure_stt_from_audio(audio_bytes)
    st.success(f"🗣️ You said: {text}")

    # GPT
    answer = chat_llm(text)
    st.info(f"🤖 GPT: {answer}")

    # TTS
    wav_file = azure_tts(answer)

    st.audio(wav_file, format="audio/wav")
