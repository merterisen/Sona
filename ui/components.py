"""
Sona UI Components — Streamlit render functions.

Each function handles one UI concern. All business logic is delegated
to the core layer; these functions only do rendering and user interaction.
"""

import base64
import hashlib

import streamlit as st

import config
from core.session import SonaSession


# ─── Status Badge ─────────────────────────────────────────────

_STATUS_MAP = {
    "ready":        ("status-ready",        "Waiting for voice input..."),
    "recording":    ("status-recording",    "Recording..."),
    "transcribing": ("status-transcribing", "Transcribing audio..."),
    "thinking":     ("status-thinking",     "Generating response..."),
    "speaking":     ("status-speaking",     "Preparing voice response..."),
}


def render_header() -> None:
    """Renders the Sona header."""
    st.markdown(
        '<div class="sona-header">'
        '<h1>Sona</h1>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_status(session: SonaSession) -> None:
    """Renders the status badge based on session state."""
    css_class, label = _STATUS_MAP.get(
        session.status,
        ("status-ready", "Ready"),
    )
    st.markdown(
        f'<div style="text-align:center">'
        f'<span class="status-badge {css_class}">{label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_welcome(session: SonaSession) -> None:
    """Renders the welcome card when no messages exist."""
    lang_info = config.SUPPORTED_LANGUAGES.get(session.language, {})
    lang_name = lang_info.get("name", session.language)

    st.markdown(
        f'<div class="welcome-card">'
        f'<h3>Ready to practice speaking?</h3>'
        f'<p>Provider: <strong>{session.provider_name}</strong></p>'
        f'<p>Language: <strong>{lang_name}</strong></p>'
        f'<p>Level: <strong>{session.level}</strong></p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_chat(session: SonaSession) -> None:
    """
    Renders all chat messages as WhatsApp-style bubbles.

    User messages appear on the right, assistant messages on the left.
    The most recent assistant audio is auto-played via a hidden HTML element.
    """
    last_assistant_audio = None
    last_assistant_audio_hash = None
    last_assistant_mime = "audio/wav"

    for msg in session.messages:
        role = msg["role"]
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

    # Auto-play the latest assistant audio response
    if last_assistant_audio and last_assistant_audio_hash:
        autoplay_html = (
            f'<audio id="autoplay-{last_assistant_audio_hash}" autoplay '
            f'src="data:{last_assistant_mime};base64,{last_assistant_audio}" '
            f'style="display:none;"></audio>'
        )
        st.markdown(autoplay_html, unsafe_allow_html=True)


def render_sidebar(session: SonaSession) -> dict | None:
    """
    Renders the sidebar with settings and returns new-chat config if triggered.

    Returns:
        A dict with new settings if "New Chat" was pressed, otherwise None.
    """
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
            index=list(lang_options.keys()).index(session.language),
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
            index=list(level_options.keys()).index(session.level),
            key="level_selector",
        )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # LLM Provider
        selected_provider = st.selectbox(
            "LLM Provider",
            options=["Local", "OpenAI"],
            index=["Local", "OpenAI"].index(session.provider_name),
            key="provider_selector",
        )

        # API Key — only shown when OpenAI is selected
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

        # New Chat Button
        new_chat_pressed = st.button(
            "New Chat",
            use_container_width=True,
            type="secondary",
            disabled=new_chat_disabled,
        )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # Active Chat info
        lang_info = config.SUPPORTED_LANGUAGES.get(session.language, {})
        st.markdown(
            f"**Active Chat**\n\n"
            f"Provider: **{session.provider_name}**\n\n"
            f"Language: **{lang_info.get('name', session.language)}**\n\n"
            f"Level: **{session.level}**",
        )

    if new_chat_pressed:
        return {
            "language": selected_lang,
            "level": selected_level,
            "provider": selected_provider,
            "api_key": st.session_state.get("api_key_input") if selected_provider == "OpenAI" else None,
        }

    return None
