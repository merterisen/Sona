"""
Sona — AI Language Practice Partner
Entry point for the Streamlit application.

This is the thin UI layer. All business logic lives in core/ and providers/.
"""

import hashlib
import logging

import streamlit as st

import config
from core.session import SonaSession
from core.pipeline import AudioPipeline
from providers import create_providers
from prompts.system_prompts import get_system_prompt
from ui.styles import get_css
from ui.components import render_header, render_sidebar, render_chat, render_status, render_welcome

# ─── Logging ──────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(name)s │ %(message)s")
logger = logging.getLogger("sona")


# ─── Streamlit Page Settings ─────────────────────────────────
st.set_page_config(
    page_title="Sona",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── Apply Custom CSS ─────────────────────────────────────────
st.markdown(get_css(), unsafe_allow_html=True)


# ─── Provider Caching ─────────────────────────────────────────
# Streamlit reruns the script on every interaction; cache_resource keeps
# one shared instance so heavy models are not reloaded each time.

@st.cache_resource(show_spinner="Loading providers...")
def _load_providers(provider_name: str, api_key: str | None = None):
    """Create and cache provider instances."""
    return create_providers(provider_name, api_key)


# ─── Session State Initialization ─────────────────────────────

def _get_session() -> SonaSession:
    """Get or create the SonaSession stored in Streamlit session_state."""
    if "sona_session" not in st.session_state:
        st.session_state.sona_session = SonaSession()
    return st.session_state.sona_session


# ─── New Chat Handler ─────────────────────────────────────────

def _handle_new_chat(new_settings: dict, session: SonaSession) -> None:
    """Apply new settings, clear history, and prepare fresh providers."""
    old_provider = session.provider_name
    old_api_key = st.session_state.get("_current_api_key")

    # Clear old LLM history before switching
    _, _, old_llm = _load_providers(old_provider, old_api_key)
    old_llm.clear_history(session.session_id)

    # Reset session with new settings
    session.reset(
        language=new_settings["language"],
        level=new_settings["level"],
        provider_name=new_settings["provider"],
    )

    # Store api_key for provider caching
    st.session_state._current_api_key = new_settings.get("api_key")

    # Update TTS voice for the new language
    lang_info = config.SUPPORTED_LANGUAGES.get(new_settings["language"], {})
    new_api_key = new_settings.get("api_key")
    _, new_tts, new_llm = _load_providers(new_settings["provider"], new_api_key)

    if new_settings["provider"] == "Local":
        tts_voice = lang_info.get("local_tts_voice", config.LOCAL_TTS_VOICE)
        tts_lang = lang_info.get("local_tts_lang", new_settings["language"])
    else:
        tts_voice = lang_info.get("openai_tts_voice", config.OPENAI_TTS_VOICE)
        tts_lang = new_settings["language"]

    new_tts.update_voice(tts_voice, tts_lang)

    # Set system prompt for the new LLM
    prompt = get_system_prompt(new_settings["level"], new_settings["language"])
    new_llm.set_system_prompt(prompt)

    # Mark prompt as initialized
    st.session_state._prompt_initialized = True


# ─── Main Application ─────────────────────────────────────────

def main():
    session = _get_session()

    # Load providers (cached)
    api_key = st.session_state.get("_current_api_key")
    stt, tts, llm = _load_providers(session.provider_name, api_key)

    # Set system prompt on first run
    if not st.session_state.get("_prompt_initialized"):
        prompt = get_system_prompt(session.level, session.language)
        llm.set_system_prompt(prompt)
        st.session_state._prompt_initialized = True

    # Header
    render_header()

    # Sidebar — returns new settings dict if "New Chat" was pressed
    new_settings = render_sidebar(session)
    if new_settings is not None:
        _handle_new_chat(new_settings, session)
        st.rerun()

    # Welcome card (if no messages yet)
    if not session.messages:
        render_welcome(session)

    # Show chat messages
    render_chat(session)

    # Status badge
    render_status(session)

    # 🎙️ Audio Input (Push-to-Talk)
    audio_value = st.audio_input(
        "🎙️ Record to speak",
        key="audio_recorder",
    )

    # Audio processing
    if audio_value is not None:
        audio_bytes = audio_value.read()
        audio_hash = hashlib.md5(audio_bytes).hexdigest()

        if audio_hash != session.last_audio_id:
            session.last_audio_id = audio_hash

            # Build pipeline and process
            pipeline = AudioPipeline(stt=stt, llm=llm, tts=tts)

            def update_status(status: str):
                session.status = status

            result = pipeline.process(
                audio_bytes=audio_bytes,
                session_id=session.session_id,
                language=session.language,
                on_status=update_status,
            )

            if result is not None:
                session.add_message("user", result.user_text)
                session.add_message(
                    "assistant",
                    result.ai_response,
                    audio=result.audio_bytes,
                    audio_mime=result.audio_mime,
                )

            st.rerun()


if __name__ == "__main__":
    main()
