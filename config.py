"""
Sona Configuration
Central configuration settings.
"""

# ─── Ollama / LLM ────────────────────────────────────────────
LLM_MODEL = "llama3.1:8b"
LLM_BASE_URL = "http://localhost:11434"
LLM_TEMPERATURE = 0.7

# ─── Faster-Whisper / STT ────────────────────────────────────
STT_MODEL_SIZE = "base"
STT_DEVICE = "cpu"
STT_COMPUTE_TYPE = "int8"

# ─── Kokoro / TTS ────────────────────────────────────────────
TTS_VOICE = "af_heart"
TTS_LANG = "de"
TTS_SPEED = 1.0
TTS_SAMPLE_RATE = 24000

# ─── Supported Languages ──────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "de": {"name": "Deutsch", "flag": "🇩🇪", "tts_lang": "de", "tts_voice": "af_heart"},
    # Soon
    # "en": {"name": "English", "flag": "🇬🇧", "tts_lang": "en-us", "tts_voice": "af_heart"},
    # "fr": {"name": "Français", "flag": "🇫🇷", "tts_lang": "fr", "tts_voice": "af_heart"},
    # "es": {"name": "Español", "flag": "🇪🇸", "tts_lang": "es", "tts_voice": "af_heart"},
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
