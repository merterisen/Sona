"""
TTSManager – Text-to-Speech (Kokoro ONNX or OpenAI)
Converts text to speech.
"""

import io
import logging

import numpy as np
import soundfile as sf

import config

logger = logging.getLogger(__name__)


class TTSManager:
    """Text-to-Speech manager (Kokoro ONNX or OpenAI)."""

    def __init__(
        self,
        provider: str = config.DEFAULT_PROVIDER,
        api_key: str | None = None,
        voice: str = config.LOCAL_TTS_VOICE,
        lang: str = config.LOCAL_TTS_LANG,
        speed: float = config.LOCAL_TTS_SPEED,
    ):
        self._provider = provider
        logger.info("Starting TTSManager - provider=%s", provider)

        self._voice = voice
        self._lang = lang
        self._speed = speed
        self._openai_voice = config.OPENAI_TTS_VOICE

        if provider == "OpenAI":
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            self.model_name = config.OPENAI_TTS_MODEL
        else:
            from kokoro_onnx import Kokoro
            from huggingface_hub import hf_hub_download

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

    @property
    def openai_voice(self) -> str:
        return self._openai_voice

    @openai_voice.setter
    def openai_voice(self, value: str):
        self._openai_voice = value

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """
        Converts text to speech using local Kokoro model.
        """
        logger.info("Synthesizing TTS (Local) – text='%s'", text[:80])
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

    def synthesize_to_bytes(self, text: str) -> tuple[bytes, str]:
        """
        Synthesizes text to audio.

        Returns:
            Tuple of (audio_bytes, mime_type).
        """
        if self._provider == "OpenAI":
            logger.info("Synthesizing TTS (OpenAI) – voice=%s, text='%s'", self._openai_voice, text[:80])
            response = self.client.audio.speech.create(
                model=self.model_name,
                voice=self._openai_voice,
                input=text,
                speed=0.9,
                response_format="mp3"
            )
            return response.content, "audio/mpeg"
        else:
            samples, sample_rate = self.synthesize(text)
            buffer = io.BytesIO()
            sf.write(buffer, samples, sample_rate, format="WAV")
            buffer.seek(0)
            return buffer.read(), "audio/wav"
