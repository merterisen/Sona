import uuid
import hashlib
import base64
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
    /* ── Global Typography ───────────────────────── */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', Roboto, sans-serif;
    }

    /* ── Header ──────────────────────────────────── */
    .sona-header {
        text-align: center;
        padding: 0.75rem 1rem 0.25rem;
        margin-bottom: 0.25rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
    .sona-header h1 {
        font-size: 1.9rem;
        font-weight: 700;
        color: var(--text-color);
        letter-spacing: -0.02em;
        margin-bottom: 0;
    }

    /* ── Status Badge ────────────────────────────── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 500;
        margin: 0.5rem auto;
        background: transparent;
        border: 1px solid;
    }
    /* Status colors: border + text only, transparent fill */
    .status-recording   { color: #c0392b; border-color: #c0392b; }
    .status-transcribing { color: #b7770d; border-color: #b7770d; }
    .status-thinking    { color: var(--primary-color); border-color: var(--primary-color); }
    .status-speaking    { color: #1a7a4a; border-color: #1a7a4a; }
    .status-ready {
        color: var(--text-color);
        border-color: rgba(128, 128, 128, 0.45);
        opacity: 0.7;
    }

    /* ── Sidebar ─────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: var(--background-color);
        border-right: 1px solid rgba(128, 128, 128, 0.2);
    }
    .sidebar-divider {
        height: 1px;
        background: rgba(128, 128, 128, 0.2);
        margin: 0.85rem 0;
    }

    /* ── Chat Messages — WhatsApp Style ─────────── */
    .stChatMessage {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin-bottom: 0.6rem !important;
    }

    /* Bubble wrapper */
    .chat-bubble-wrapper {
        display: flex;
        width: 100%;
        margin-bottom: 0.5rem;
    }
    .chat-bubble-wrapper.user {
        justify-content: flex-end;
    }
    .chat-bubble-wrapper.assistant {
        justify-content: flex-start;
    }

    /* The actual bubble */
    .chat-bubble {
        max-width: 72%;
        padding: 0.65rem 1rem;
        border-radius: 18px;
        font-size: 0.92rem;
        line-height: 1.55;
        word-break: break-word;
        background: rgba(128, 128, 128, 0.15);
        color: var(--text-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .chat-bubble.user {
        border-bottom-right-radius: 4px;
    }
    .chat-bubble.assistant {
        border-bottom-left-radius: 4px;
    }

    /* Audio player inside bubble */
    .chat-bubble audio {
        display: block;
        margin-top: 0.45rem;
        width: 100%;
        height: 32px;
        border-radius: 8px;
    }

    /* Hide Streamlit's native chat avatars & containers when using custom bubbles */
    .stChatMessage [data-testid="stChatMessageAvatarUser"],
    .stChatMessage [data-testid="stChatMessageAvatarAssistant"] {
        display: none !important;
    }

    /* ── Welcome Card ────────────────────────────── */
    .welcome-card {
        text-align: center;
        padding: 1.5rem;
        border-radius: 4px;
        background: transparent;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin: 0.75rem 0;
    }
    .welcome-card h3 {
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: var(--text-color);
    }
    .welcome-card p {
        color: var(--text-color);
        opacity: 0.65;
        font-size: 0.875rem;
        line-height: 1.55;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Managers with Caching ──────────────────────────────
# Streamlit reruns this script on every interaction; cache_resource keeps one
# shared instance per manager so heavy models (Whisper, Kokoro, LLM) are not
# reloaded on each rerun. Call these loaders anywhere — cache returns the same object.
@st.cache_resource(show_spinner="Loading Whisper Speech-to-Text model...")
def load_stt_manager(provider: str, api_key: str | None = None) -> STTManager:
    return STTManager(provider=provider, api_key=api_key)


@st.cache_resource(show_spinner="Loading Kokoro Text-to-Speech model...")
def load_tts_manager(provider: str, api_key: str | None = None) -> TTSManager:
    return TTSManager(provider=provider, api_key=api_key)


@st.cache_resource(show_spinner="Connecting to the LLM...")
def load_llm_manager(provider: str, api_key: str | None = None) -> LLMManager:
    return LLMManager(provider=provider, api_key=api_key)


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
    if "provider" not in st.session_state:
        st.session_state.provider = config.DEFAULT_PROVIDER
    if "api_key" not in st.session_state:
        st.session_state.api_key = None
    if "status" not in st.session_state:
        st.session_state.status = "ready"

    if "last_audio_id" not in st.session_state:
        st.session_state.last_audio_id = None


# ─── Sidebar ──────────────────────────────────────────────────
def _start_new_chat():
    """Applies the currently selected settings and starts a fresh chat session."""
    # Apply the sidebar selections as the new active settings
    selected_lang = st.session_state.get("lang_selector", st.session_state.language)
    selected_level = st.session_state.get("level_selector", st.session_state.level)
    selected_provider = st.session_state.get("provider_selector", st.session_state.provider)
    selected_api_key = st.session_state.get("api_key_input", st.session_state.api_key)

    st.session_state.language = selected_lang
    st.session_state.level = selected_level
    st.session_state.provider = selected_provider
    st.session_state.api_key = selected_api_key if selected_provider == "OpenAI" else None

    # Clear prompt_initialized so system prompt is set for the new LLM
    st.session_state.pop("prompt_initialized", None)

    # Update TTS language/voice
    tts = load_tts_manager(st.session_state.provider, st.session_state.api_key)
    tts.lang = config.SUPPORTED_LANGUAGES.get(selected_lang, {}).get("local_tts_lang", selected_lang)
    tts.voice = config.SUPPORTED_LANGUAGES.get(selected_lang, {}).get("local_tts_voice", config.LOCAL_TTS_VOICE)
    tts.openai_voice = config.SUPPORTED_LANGUAGES.get(selected_lang, {}).get("openai_tts_voice", config.OPENAI_TTS_VOICE)

    # Clear old LLM history before switching
    old_llm = load_llm_manager(st.session_state.provider, st.session_state.api_key)
    old_llm.clear_history(st.session_state.session_id)

    # Clear chat history and create a new session
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.status = "ready"
    st.session_state.last_audio_id = None


def render_sidebar():
    """Sidebar Components"""
    with st.sidebar:
        st.markdown("### Settings")
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

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # LLM Provider
        selected_provider = st.selectbox(
            "LLM Provider",
            options=["Local", "OpenAI"],
            index=["Local", "OpenAI"].index(st.session_state.provider),
            key="provider_selector",
        )

        # API Key — only shown when OpenAI is selected in the dropdown
        if selected_provider == "OpenAI":
            st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-...",
                key="api_key_input",
            )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # Disable New Chat when OpenAI is selected but no API key entered
        new_chat_disabled = (
            selected_provider == "OpenAI"
            and not st.session_state.get("api_key_input")
        )

        # New Chat Button — applies selected settings and starts fresh
        if st.button("New Chat", use_container_width=True, type="secondary", disabled=new_chat_disabled):
            _start_new_chat()
            st.rerun()

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # Active Chat info
        level = st.session_state.level
        lang_info = config.SUPPORTED_LANGUAGES.get(st.session_state.language, {})

        st.markdown(
            f"**Active Chat**\n\n"
            f"Provider: **{st.session_state.provider}**\n\n"
            f"Language: **{lang_info.get('name', st.session_state.language)}**\n\n"
            f"Level: **{level}**",
        )


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
    """Renders all chat messages as WhatsApp-style bubbles.

    User messages appear on the right, assistant messages on the left.
    The most recent assistant audio is auto-played via a hidden HTML element.
    A unique key (based on the audio hash) is embedded in the element ID so
    the browser always treats it as a new element and plays it.
    """
    last_assistant_audio = None
    last_assistant_audio_hash = None
    last_assistant_mime = "audio/wav"

    for msg in st.session_state.messages:
        role = msg["role"]  # "user" or "assistant"
        content = msg["content"]
        audio_bytes = msg.get("audio")
        audio_mime = msg.get("audio_mime", "audio/wav")

        # Build audio HTML if present
        audio_html = ""
        if audio_bytes:
            b64 = base64.b64encode(audio_bytes).decode()
            audio_html = (
                f'<audio controls src="data:{audio_mime};base64,{b64}" '
                f'style="display:block;margin-top:0.45rem;width:100%;height:32px;"></audio>'
            )
            if role == "assistant":
                last_assistant_audio = b64
                last_assistant_audio_hash = hashlib.md5(audio_bytes).hexdigest()
                last_assistant_mime = audio_mime

        bubble_html = (
            f'<div class="chat-bubble-wrapper {role}">'
            f'<div class="chat-bubble {role}">'
            f'{content}'
            f'{audio_html}'
            f'</div></div>'
        )
        st.markdown(bubble_html, unsafe_allow_html=True)

    # Auto-play the latest assistant audio response.
    # Use a unique element ID derived from the audio hash so the browser
    # always sees it as a fresh element and triggers playback.
    if last_assistant_audio and last_assistant_audio_hash:
        autoplay_html = (
            f'<audio id="autoplay-{last_assistant_audio_hash}" autoplay '
            f'src="data:{last_assistant_mime};base64,{last_assistant_audio}" '
            f'style="display:none;"></audio>'
        )
        st.markdown(autoplay_html, unsafe_allow_html=True)


# ─── Audio Processing Pipeline ────────────────────────────────
def process_audio(audio_bytes: bytes, stt: STTManager, llm: LLMManager, tts: TTSManager):
    """
    Full audio processing pipeline:
    Audio → STT → LLM → TTS → Chat
    """
    import time

    lang_code = st.session_state.language
    pipeline_start = time.perf_counter()

    audio_size_kb = len(audio_bytes) / 1024
    logger.info("⏱️ Pipeline started — audio size: %.1f KB", audio_size_kb)

    # 1. STT: Audio → Text
    st.session_state.status = "transcribing"
    with st.spinner("Converting speech to text..."):
        t0 = time.perf_counter()
        user_text = stt.transcribe(audio_bytes, language=lang_code)
        stt_duration = time.perf_counter() - t0
        logger.info("⏱️ STT completed in %.2f seconds", stt_duration)

    if not user_text.strip():
        st.warning("No speech detected. Please try again.")
        st.session_state.status = "ready"
        return

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_text,
    })

    # 2. LLM: Text → Response
    st.session_state.status = "thinking"
    with st.spinner("Generating response..."):
        t0 = time.perf_counter()
        ai_response = llm.get_response(user_text, st.session_state.session_id)
        llm_duration = time.perf_counter() - t0
        logger.info("⏱️ LLM completed in %.2f seconds", llm_duration)

    # 3. TTS: Response → Audio
    st.session_state.status = "speaking"
    with st.spinner("Preparing voice response..."):
        try:
            t0 = time.perf_counter()
            audio_response, audio_mime = tts.synthesize_to_bytes(ai_response)
            tts_duration = time.perf_counter() - t0
            logger.info("⏱️ TTS completed in %.2f seconds", tts_duration)
        except Exception as e:
            logger.error("TTS error: %s", e)
            audio_response = None
            audio_mime = None
            tts_duration = 0

    total_duration = time.perf_counter() - pipeline_start
    logger.info(
        "⏱️ PIPELINE TOTAL: %.2f seconds (STT=%.2f, LLM=%.2f, TTS=%.2f)",
        total_duration, stt_duration, llm_duration, tts_duration,
    )

    # Add AI message
    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_response,
        "audio": audio_response,
        "audio_mime": audio_mime,
    })

    st.session_state.status = "ready"


# ─── Main Application ─────────────────────────────────────────
def main():
    init_session_state()

    # Header
    st.markdown(
        '<div class="sona-header">'
        '<h1>Sona</h1>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Load Managers (cached per provider + api_key combination)
    provider = st.session_state.provider
    api_key = st.session_state.api_key

    stt = load_stt_manager(provider, api_key)
    tts = load_tts_manager(provider, api_key)
    llm = load_llm_manager(provider, api_key)

    # Set system prompt on first run
    if "prompt_initialized" not in st.session_state:
        prompt = get_system_prompt(st.session_state.level, st.session_state.language)
        llm.set_system_prompt(prompt)
        st.session_state.prompt_initialized = True

    # Sidebar
    render_sidebar()

    # Welcome card (if no messages yet)
    if not st.session_state.messages:
        lang_info = config.SUPPORTED_LANGUAGES.get(st.session_state.language, {})
        lang_name = lang_info.get("name", st.session_state.language)

        st.markdown(
            f'<div class="welcome-card">'
            f'<h3>Ready to practice speaking?</h3>'
            f'<p>Provider: <strong>{provider}</strong></p>'
            f'<p>Language: <strong>{lang_name}</strong></p>'
            f'<p>Level: <strong>{st.session_state.level}</strong></p>'
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
        audio_bytes = audio_value.read()
        audio_hash = hashlib.md5(audio_bytes).hexdigest()
        if audio_hash != st.session_state.last_audio_id:
            st.session_state.last_audio_id = audio_hash
            process_audio(audio_bytes, stt, llm, tts)
            st.rerun()



if __name__ == "__main__":
    main()
