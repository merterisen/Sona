"""
TTSManager – Text-to-Speech (Kokoro ONNX)
Converts text to speech.
"""

import io
import logging

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

import config

logger = logging.getLogger(__name__)


class TTSManager:
    """Text-to-Speech manager based on Kokoro ONNX."""

    def __init__(
        self,
        voice: str = config.TTS_VOICE,
        lang: str = config.TTS_LANG,
        speed: float = config.TTS_SPEED,
    ):
        logger.info(
            "Starting TTSManager – voice=%s, lang=%s, speed=%.1f",
            voice, lang, speed,
        )
        self._voice = voice
        self._lang = lang
        self._speed = speed

        from huggingface_hub import hf_hub_download

        # Kokoro model – downloads from HuggingFace and caches on first run
        logger.info("Downloading Kokoro model files from HuggingFace...")
        model_path = hf_hub_download(
            repo_id="fastrtc/kokoro-onnx",
            filename="kokoro-v1.0.onnx"
        )
        voices_path = hf_hub_download(
            repo_id="fastrtc/kokoro-onnx",
            filename="voices-v1.0.bin"
        )
        logger.info("Kokoro model files loaded: model=%s, voices=%s", model_path, voices_path)

        self._kokoro = Kokoro(model_path, voices_path)
        logger.info("TTSManager ready.")

    @property
    def voice(self) -> str:
        return self._voice

    @voice.setter
    def voice(self, value: str):
        self._voice = value

    @property
    def lang(self) -> str:
        return self._lang

    @lang.setter
    def lang(self, value: str):
        self._lang = value

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """
        Converts text to speech.

        Args:
            text: Text to be converted to speech.

        Returns:
            (samples, sample_rate) tuple.
        """
        logger.info("Synthesizing TTS – text='%s'", text[:80])
        samples, sample_rate = self._kokoro.create(
            text,
            voice=self._voice,
            speed=self._speed,
            lang=self._lang,
        )
        logger.info(
            "TTS completed – %d samples, sr=%d, duration=%.1fs",
            len(samples), sample_rate, len(samples) / sample_rate,
        )
        return samples, sample_rate

    def synthesize_to_bytes(self, text: str) -> bytes:
        """
        Returns the audio as WAV bytes.
        Can be used directly with Streamlit st.audio().

        Args:
            text: Text to be converted to speech.

        Returns:
            Audio bytes in WAV format.
        """
        samples, sample_rate = self.synthesize(text)

        buffer = io.BytesIO()
        sf.write(buffer, samples, sample_rate, format="WAV")
        buffer.seek(0)
        return buffer.read()
