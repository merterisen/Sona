"""
SonaSession — Framework-agnostic conversation session state.

Manages messages, settings, and session lifecycle without any
dependency on Streamlit, Gradio, or other UI frameworks.
"""

import uuid
import logging

import config

logger = logging.getLogger(__name__)


class SonaSession:
    """Holds all state for a single conversation session."""

    def __init__(
        self,
        language: str = config.DEFAULT_LANGUAGE,
        level: str = config.DEFAULT_LEVEL,
        provider_name: str = config.DEFAULT_PROVIDER,
    ):
        self.session_id: str = str(uuid.uuid4())
        self.messages: list[dict] = []
        self.language: str = language
        self.level: str = level
        self.provider_name: str = provider_name
        self.status: str = "ready"
        self.last_audio_id: str | None = None

        logger.info(
            "Session created — id=%s, lang=%s, level=%s, provider=%s",
            self.session_id[:8], language, level, provider_name,
        )

    def add_message(
        self,
        role: str,
        content: str,
        audio: bytes | None = None,
        audio_mime: str | None = None,
    ) -> None:
        """Append a message to the conversation history."""
        msg = {"role": role, "content": content}
        if audio is not None:
            msg["audio"] = audio
            msg["audio_mime"] = audio_mime
        self.messages.append(msg)

    def reset(
        self,
        language: str | None = None,
        level: str | None = None,
        provider_name: str | None = None,
    ) -> None:
        """Clear messages and start a fresh session, optionally updating settings."""
        self.session_id = str(uuid.uuid4())
        self.messages = []
        self.status = "ready"
        self.last_audio_id = None

        if language is not None:
            self.language = language
        if level is not None:
            self.level = level
        if provider_name is not None:
            self.provider_name = provider_name

        logger.info(
            "Session reset — id=%s, lang=%s, level=%s, provider=%s",
            self.session_id[:8], self.language, self.level, self.provider_name,
        )
