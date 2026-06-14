"""
Provider Base Classes — Abstract interfaces for STT, TTS, and LLM providers.

All concrete providers (Local, OpenAI, future Realtime) implement these ABCs.
This ensures the core pipeline and UI layer are fully provider-agnostic.
"""

from abc import ABC, abstractmethod


class BaseSTT(ABC):
    """Abstract Speech-to-Text provider."""

    @abstractmethod
    def transcribe(self, audio_bytes: bytes, language: str | None = None) -> str:
        """
        Convert audio bytes to text.

        Args:
            audio_bytes: Raw audio data (WAV format).
            language: Optional target language code (e.g., "de").

        Returns:
            Transcribed text string.
        """
        ...


class BaseTTS(ABC):
    """Abstract Text-to-Speech provider."""

    @abstractmethod
    def synthesize(self, text: str) -> tuple[bytes, str]:
        """
        Convert text to audio bytes.

        Args:
            text: Text to synthesize.

        Returns:
            Tuple of (audio_bytes, mime_type).
        """
        ...

    @abstractmethod
    def update_voice(self, voice: str, language: str) -> None:
        """
        Update the voice and language settings.

        Args:
            voice: Voice identifier.
            language: Language code for synthesis.
        """
        ...


class BaseLLM(ABC):
    """Abstract Large Language Model provider."""

    @abstractmethod
    def set_system_prompt(self, prompt: str) -> None:
        """
        Set or update the system prompt.

        Args:
            prompt: System prompt text.
        """
        ...

    @abstractmethod
    def get_response(self, user_message: str, session_id: str = "default") -> str:
        """
        Generate a response to the user's message.

        Args:
            user_message: The user's input text.
            session_id: Conversation session identifier.

        Returns:
            Generated response text.
        """
        ...

    @abstractmethod
    def clear_history(self, session_id: str = "default") -> None:
        """
        Clear conversation history for a session.

        Args:
            session_id: Conversation session identifier.
        """
        ...
