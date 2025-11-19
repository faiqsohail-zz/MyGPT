import streamlit as st
import azure.cognitiveservices.speech as speechsdk
import tempfile
import openai

# -----------------------------
#  AZURE SPEECH CONFIG
# -----------------------------
SPEECH_KEY = "577IXgzDaBsgyxHGxDjcxVKHlhW7MpTz6EApueKJNAdVVo89Ew9RJQQJ99BKAC3pKaRXJ3w3AAAYACOGxpcA"
SPEECH_REGION = "eastasia"

# -----------------------------
#  OPENROUTER CONFIG
# -----------------------------
OPENROUTER_API_KEY = "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"
openai.api_key = OPENROUTER_API_KEY
openai.api_base = "https://openrouter.ai/api/v1"
openai.api_type = "open_router"

# -----------------------------
#  AZURE SPEECH-TO-TEXT (AUTO LANG)
# -----------------------------
def azure_stt_from_audio(audio_bytes):
    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION
    )

    auto_detect = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
        languages=["en-US", "ur-PK", "ar-SA"]
    )

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
    return result.text if result.reason == speechsdk.ResultReason.RecognizedSpeech else None

# -----------------------------
#  OPENROUTER GPT CHAT
# -----------------------------
def chat_llm(msg):
    if not msg:
        return "I couldn't hear you clearly."

    response = openai.chat.completions.create(
        model="gpt-4o-mini",  # OpenRouter-supported model
        messages=[
            {"role": "system", "content": "You are a multilingual helpful AI assistant."},
            {"role": "user", "content": msg},
        ]
    )
    return response.choices[0].message.content

# -----------------------------
#  AZURE TEXT-TO-SPEECH
# -----------------------------
def azure_tts(text):
    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION
    )
    speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"

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
st.title("🎤 Azure Voice → OpenRouter GPT → Voice Chatbot (Auto-Language)")

# Record audio input
audio_input = st.audio_input("Click to record your voice 🎙️")

if audio_input:
    st.write("⏳ Processing your voice...")
    audio_bytes = audio_input.getvalue()

    # STT
    text = azure_stt_from_audio(audio_bytes)
    if text:
        st.success(f"🗣️ You said: {text}")

        # GPT
        answer = chat_llm(text)
        st.info(f"🤖 GPT: {answer}")

        # TTS — only run if we have an answer
        wav_file = azure_tts(answer)
        st.audio(wav_file, format="audio/wav")
    else:
        st.error("❌ Speech not recognized. Please try again.")

    # TTS
    wav_file = azure_tts(answer)
    st.audio(wav_file, format="audio/wav")
