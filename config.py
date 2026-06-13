"""
Sona Configuration
Central configuration settings.
"""

# ─── LLM Providers ───────────────────────────────────────────
DEFAULT_PROVIDER = "Local"
LLM_TEMPERATURE = 0.7

# Local (Ollama)
LOCAL_LLM_MODEL = "llama3.1:8b"
LOCAL_LLM_BASE_URL = "http://localhost:11434"

# OpenAI (cloud)
OPENAI_MODEL = "gpt-5-nano"

# ─── Speech-to-Text (STT) ────────────────────────────────────
LOCAL_STT_MODEL_SIZE = "base"
LOCAL_STT_DEVICE = "cpu"
LOCAL_STT_COMPUTE_TYPE = "int8"

OPENAI_STT_MODEL = "whisper-1"

# ─── Text-to-Speech (TTS) ────────────────────────────────────
LOCAL_TTS_VOICE = "af_heart"
LOCAL_TTS_LANG = "de"
LOCAL_TTS_SPEED = 1.0
LOCAL_TTS_SAMPLE_RATE = 24000

OPENAI_TTS_MODEL = "tts-1"
OPENAI_TTS_VOICE = "nova"

# ─── Supported Languages ──────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "de": {"name": "Deutsch", "flag": "🇩🇪", "local_tts_lang": "de", "local_tts_voice": "af_heart", "openai_tts_voice": "nova"},
    # Soon
    # "en": {"name": "English", "flag": "🇬🇧", "local_tts_lang": "en-us", "local_tts_voice": "af_heart", "openai_tts_voice": "alloy"},
    # "fr": {"name": "Français", "flag": "🇫🇷", "local_tts_lang": "fr", "local_tts_voice": "af_heart", "openai_tts_voice": "nova"},
    # "es": {"name": "Español", "flag": "🇪🇸", "local_tts_lang": "es", "local_tts_voice": "af_heart", "openai_tts_voice": "nova"},
}

DEFAULT_LANGUAGE = "de"

# ─── CEFR Levels ─────────────────────────────────────────
CEFR_LEVELS = {
    "A1": "Beginner - Basic expressions and simple sentences",
    "A2": "Elementary - Everyday conversations and simple dialogues",
    "B1": "Intermediate - Fluent speaking on everyday topics",
    "B2": "Upper Intermediate - Discussing complex topics and giving opinions",
    "C1": "Advanced - Fluent expression on academic and professional subjects",
    "C2": "Proficient - Near-native fluency",
}

DEFAULT_LEVEL = "A2"
