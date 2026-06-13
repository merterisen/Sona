"""
STTManager – Speech-to-Text (Faster-Whisper or OpenAI)
Converts audio data to text.
"""

import io
import tempfile
import logging

import numpy as np
import soundfile as sf
import openai

import config

logger = logging.getLogger(__name__)


class STTManager:
    """Speech-to-Text manager (Faster-Whisper or OpenAI)."""

    def __init__(
        self,
        provider: str = config.DEFAULT_PROVIDER,
        api_key: str | None = None,
        model_size: str = config.LOCAL_STT_MODEL_SIZE,
        device: str = config.LOCAL_STT_DEVICE,
        compute_type: str = config.LOCAL_STT_COMPUTE_TYPE,
    ):
        self._provider = provider
        logger.info(
            "Starting STTManager - provider=%s",
            provider
        )

        if provider == "OpenAI":
            self.client = openai.OpenAI(api_key=api_key)
            self.model_name = config.OPENAI_STT_MODEL
        else:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
            )
        logger.info("STTManager ready.")

    def transcribe(self, audio_bytes: bytes, language: str | None = None) -> str:
        """
        Converts audio bytes to text.

        Args:
            audio_bytes: Raw audio data in WAV format (bytes).
            language: Target language code (e.g., "de"). If None, detects automatically.

        Returns:
            Transcribed text.
        """
        # Streamlit audio_input returns WAV bytes - write to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()

            if self._provider == "OpenAI":
                with open(tmp.name, "rb") as audio_file:
                    kwargs = {}
                    if language:
                        kwargs["language"] = language
                    response = self.client.audio.transcriptions.create(
                        model=self.model_name,
                        file=audio_file,
                        **kwargs
                    )
                    text = response.text
                logger.info("STT completed (OpenAI) - text='%s'", text[:80])
            else:
                segments, info = self.model.transcribe(
                    tmp.name,
                    beam_size=5,
                    language=language,
                )

                text = " ".join(segment.text.strip() for segment in segments)

                detected_lang = info.language
                prob = info.language_probability
                logger.info(
                    "STT completed (Local) - language=%s (%.1f%%), text='%s'",
                    detected_lang, prob * 100, text[:80],
                )

        return text.strip() 