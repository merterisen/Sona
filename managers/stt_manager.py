"""
STTManager – Speech-to-Text (Faster-Whisper)
Converts audio data to text.
"""

import io
import tempfile
import logging

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

import config

logger = logging.getLogger(__name__)


class STTManager:
    """Speech-to-Text manager based on Faster-Whisper."""

    def __init__(
        self,
        model_size: str = config.STT_MODEL_SIZE,
        device: str = config.STT_DEVICE,
        compute_type: str = config.STT_COMPUTE_TYPE,
    ):
        logger.info(
            "Starting STTManager - model=%s, device=%s, compute=%s",
            model_size, device, compute_type,
        )
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

            segments, info = self.model.transcribe(
                tmp.name,
                beam_size=5,
                language=language,
            )

            text = " ".join(segment.text.strip() for segment in segments)

        detected_lang = info.language
        prob = info.language_probability
        logger.info(
            "STT completed - language=%s (%.1f%%), text='%s'",
            detected_lang, prob * 100, text[:80],
        )
        return text.strip() 