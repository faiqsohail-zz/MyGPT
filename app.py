import streamlit as st
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Azure Speech Credentials
speech_key = os.getenv("SPEECH_KEY")
speech_region = os.getenv("SPEECH_REGION")

# Azure OpenAI Credentials
client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-05-01-preview",
    azure_endpoint=os.getenv("OPENAI_ENDPOINT")
)

deployment = os.getenv("OPENAI_DEPLOYMENT")


# -------------------------------
# Speech to Text
# -------------------------------
def azure_tts(text):
    speech_config = speechsdk.SpeechConfig(
        subscription=speech_key, region=speech_region
    )
    speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"

    audio_config = speechsdk.audio.AudioOutputConfig(filename="output.wav")

    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=audio_config
    )

    synthesizer.speak_text_async(text).get()
    return "output.wav"


# -------------------------------
# LLM Response (Azure OpenAI)
# -------------------------------
def chat_llm(user_msg):
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_msg},
        ]
    )
    return response.choices[0].message.content


# -------------------------------
# Text to Speech
# -------------------------------
def azure_tts(text):
    speech_config = speechsdk.SpeechConfig(
        subscription=speech_key, region=speech_region
    )
    speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"

    audio_config = speechsdk.audio.AudioOutputConfig(filename="output.wav")

    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=audio_config
    )

    synthesizer.speak_text_async(text).get()

    return "output.wav"


# -------------------------------
# Streamlit App UI
# -------------------------------
st.title("🎤 Azure Speech-to-Text + GPT + Text-to-Speech Chatbot")
st.write("Speak → LLM → Listen")

if st.button("Start Recording"):
    text = azure_stt()
    st.success(f"🗣️ You said: {text}")

    # LLM reply
    reply = chat_llm(text)
    st.info(f"🤖 LLM Response: {reply}")

    # Convert to speech
    audio_file = azure_tts(reply)

    # Play audio
    audio_bytes = open(audio_file, "rb").read()
    st.audio(audio_bytes, format="audio/wav")
