import uuid
import hashlib
import logging
import streamlit as st

import config
from managers.stt_manager import STTManager
from managers.tts_manager import TTSManager
from managers.llm_manager import LLMManager
from prompts.system_prompts import get_system_prompt

# ─── Logging ──────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(name)s │ %(message)s")
logger = logging.getLogger("sona")


# ─── Streamlit Page Settings ─────────────────────────────────
st.set_page_config(
    page_title="Sona",
    #page_icon="🗣️",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ──────────────────────────────────── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header ──────────────────────────────────── */
    .sona-header {
        text-align: center;
        padding: 0.5rem 1rem 0.2rem;
        margin-bottom: 0.2rem;
    }
    .sona-header h1 {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.1rem;
    }
    .sona-header p {
        font-size: 0.9rem;
        opacity: 0.6;
        margin: 0;
    }

    /* ── Status Badge ────────────────────────────── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.5rem auto;
        animation: fadeIn 0.3s ease;
    }
    .status-recording { background: rgba(239, 68, 68, 0.15); color: #ef4444; }
    .status-transcribing { background: rgba(245, 158, 11, 0.15); color: #f59e0b; }
    .status-thinking { background: rgba(99, 102, 241, 0.15); color: #6366f1; }
    .status-speaking { background: rgba(16, 185, 129, 0.15); color: #10b981; }
    .status-ready { background: rgba(107, 114, 128, 0.1); color: #6b7280; }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-4px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ── Sidebar Styling ─────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(102,126,234,0.05) 0%, rgba(118,75,162,0.05) 100%);
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stRadio label {
        font-weight: 500;
    }

    /* ── Chat Messages ───────────────────────────── */
    .stChatMessage {
        border-radius: 12px !important;
        margin-bottom: 0.5rem !important;
    }

    /* ── Welcome Card ────────────────────────────── */
    .welcome-card {
        text-align: center;
        padding: 1.5rem 1.5rem;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(102,126,234,0.08) 0%, rgba(118,75,162,0.08) 100%);
        border: 1px solid rgba(102,126,234,0.15);
        margin: 0.5rem 0;
    }
    .welcome-card h3 {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .welcome-card p {
        opacity: 0.7;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* ── Level Badge ─────────────────────────────── */
    .level-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        letter-spacing: 0.5px;
    }

    /* ── Audio Input Styling ─────────────────────── */
    .stAudioInput {
        margin-top: 0.5rem;
    }

    /* ── Divider ─────────────────────────────────── */
    .sidebar-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102,126,234,0.3), transparent);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Managers with Caching ──────────────────────────────
# Streamlit reruns this script on every interaction; cache_resource keeps one
# shared instance per manager so heavy models (Whisper, Kokoro, LLM) are not
# reloaded on each rerun. Call these loaders anywhere — cache returns the same object.
@st.cache_resource(show_spinner="Loading Whisper Speech-to-Text model...")
def load_stt_manager() -> STTManager:
    return STTManager()


@st.cache_resource(show_spinner="Loading Kokoro Text-to-Speech model...")
def load_tts_manager() -> TTSManager:
    return TTSManager()


@st.cache_resource(show_spinner="Connecting to the LLM...")
def load_llm_manager() -> LLMManager:
    return LLMManager()


# ─── Start Session State ─────────────────────────────────────
def init_session_state():
    """Sets the default values for session state variables."""

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "level" not in st.session_state:
        st.session_state.level = config.DEFAULT_LEVEL
    if "language" not in st.session_state:
        st.session_state.language = config.DEFAULT_LANGUAGE
    if "status" not in st.session_state:
        st.session_state.status = "ready"
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "last_audio_id" not in st.session_state:
        st.session_state.last_audio_id = None


# ─── Sidebar ──────────────────────────────────────────────────
def render_sidebar(llm: LLMManager):
    """Sidebar Components"""
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # Language Selection
        lang_options = {
            key: value["name"] 
            for key, value in config.SUPPORTED_LANGUAGES.items()
        }
        selected_lang = st.selectbox(
            "Language",
            options=list(lang_options.keys()),
            format_func=lambda x: lang_options[x],
            index=list(lang_options.keys()).index(st.session_state.language),
            key="lang_selector",
        )

        # CEFR level
        level_options = {
            level: f"{level} - {desc}"
            for level, desc in config.CEFR_LEVELS.items()
        }
        selected_level = st.selectbox(
            "Language Level (CEFR)",
            options=list(level_options.keys()),
            format_func=lambda x: level_options[x],
            index=list(level_options.keys()).index(st.session_state.level),
            key="level_selector",
        )

        # Change detection
        if selected_lang != st.session_state.language or selected_level != st.session_state.level:
            st.session_state.language = selected_lang
            st.session_state.level = selected_level

            # Update system prompt
            new_system_prompt = get_system_prompt(st.session_state.level, st.session_state.language)
            llm.set_system_prompt(new_system_prompt)

            # Update TTS language
            tts = load_tts_manager()
            tts.lang = config.SUPPORTED_LANGUAGES.get(selected_lang, {}).get("tts_lang", selected_lang)
            tts.voice = config.SUPPORTED_LANGUAGES.get(selected_lang, {}).get("tts_voice", config.TTS_VOICE)

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # Current status indicator
        level = st.session_state.level
        lang_info = config.SUPPORTED_LANGUAGES.get(st.session_state.language, {})

        st.markdown(
            f"**Active Chat**\n\n"
            f"Language: **{lang_info.get('name', st.session_state.language)}**\n\n"
            f"Level: **{level}**",
        )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # New Chat Button
        if st.button("New Chat  ", use_container_width=True, type="secondary"):
            llm.clear_history(st.session_state.session_id)
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.status = "ready"
            st.session_state.last_audio_id = None
            st.rerun()


# ─── Status Badge ─────────────────────────────────────────────
_STATUS_MAP = {
    "ready": ("status-ready", "Waiting for voice input..."),
    "recording": ("status-recording", "Recording..."),
    "transcribing": ("status-transcribing", "Transcribing audio..."),
    "thinking": ("status-thinking", "Generating response..."),
    "speaking": ("status-speaking", "Preparing voice response..."),
}


def render_status():
    """Renders the status badge."""
    css_class, label = _STATUS_MAP.get(
        st.session_state.status,
        ("status-ready", "Ready"),
    )
    st.markdown(
        f'<div style="text-align:center">'
        f'<span class="status-badge {css_class}">{label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─── Chat Render ──────────────────────────────────────────────
def render_chat():
    """Renders all chat messages."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=msg.get("avatar")):
            st.markdown(msg["content"])
            if msg.get("audio"):
                st.audio(msg["audio"], format="audio/wav")


# ─── Audio Processing Pipeline ────────────────────────────────
def process_audio(audio_bytes: bytes, stt: STTManager, llm: LLMManager, tts: TTSManager):
    """
    Full audio processing pipeline:
    Audio → STT → LLM → TTS → Chat
    """
    lang_code = st.session_state.language

    # 1. STT: Audio → Text
    st.session_state.status = "transcribing"
    with st.spinner("Converting speech to text..."):
        user_text = stt.transcribe(audio_bytes, language=lang_code)

    if not user_text.strip():
        st.warning("No speech detected. Please try again.")
        st.session_state.status = "ready"
        return

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_text,
        "avatar": "🗣️",
    })

    # 2. LLM: Text → Response
    st.session_state.status = "thinking"
    with st.spinner("Generating response..."):
        ai_response = llm.get_response(user_text, st.session_state.session_id)

    # 3. TTS: Response → Audio
    st.session_state.status = "speaking"
    with st.spinner("Preparing voice response..."):
        try:
            audio_response = tts.synthesize_to_bytes(ai_response)
        except Exception as e:
            logger.error("TTS error: %s", e)
            audio_response = None

    # Add AI message
    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_response,
        "avatar": "🤖",
        "audio": audio_response,
    })

    st.session_state.status = "ready"


# ─── Main Application ─────────────────────────────────────────
def main():
    init_session_state()

    # Header
    st.markdown(
        '<div class="sona-header">'
        '<h1>🗣️ Sona</h1>'
        #'<p>Push-to-Talk Language Practice Assistant</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Load Managers
    stt = load_stt_manager()
    tts = load_tts_manager()
    llm = load_llm_manager()

    # Set system prompt on first run
    if "prompt_initialized" not in st.session_state:
        prompt = get_system_prompt(st.session_state.level, st.session_state.language)
        llm.set_system_prompt(prompt)
        st.session_state.prompt_initialized = True

    # Sidebar
    render_sidebar(llm)

    # Welcome card (if no messages yet)
    if not st.session_state.messages:
        lang_info = config.SUPPORTED_LANGUAGES.get(st.session_state.language, {})
        lang_name = lang_info.get("name", st.session_state.language)

        st.markdown(
            f'<div class="welcome-card">'
            f'<h3>Ready to practice speaking?</h3>'
            f'Language: <strong>{lang_name}</strong></p>'
            f'Level: <strong>{st.session_state.level}</strong></p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Show chat messages
    render_chat()

    # Status badge
    render_status()

    # 🎙️ Audio Input (Push-to-Talk)
    audio_value = st.audio_input(
        "🎙️ Record to speak",
        key="audio_recorder",
    )

    # Audio processing
    if audio_value is not None:
        # Read bytes and compute a content hash to avoid re-processing
        audio_bytes = audio_value.read()

        audio_hash = hashlib.md5(audio_bytes).hexdigest()
        if audio_hash != st.session_state.last_audio_id:
            st.session_state.last_audio_id = audio_hash
            process_audio(audio_bytes, stt, llm, tts)
            st.rerun()
  


if __name__ == "__main__":
    main()
