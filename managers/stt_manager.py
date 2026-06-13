"""
STTManager – Speech-to-Text (Faster-Whisper or OpenAI)
Converts audio data to text.
"""

import io
import tempfile
import logging

import numpy as np
import soundfile as sf

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
            import openai
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

    def _downsample_to_16k_mono(self, audio_bytes: bytes) -> bytes:
        """Resample audio to 16 kHz mono WAV to minimize upload size."""
        data, sr = sf.read(io.BytesIO(audio_bytes))
        # Convert stereo to mono
        if data.ndim > 1:
            data = data.mean(axis=1)
        # Resample to 16 kHz if needed
        target_sr = 16000
        if sr != target_sr:
            # Simple linear interpolation resampling (no scipy needed)
            ratio = target_sr / sr
            num_samples = int(len(data) * ratio)
            indices = np.linspace(0, len(data) - 1, num_samples)
            data = np.interp(indices, np.arange(len(data)), data).astype(np.float32)
            sr = target_sr
        # Write compact WAV
        buf = io.BytesIO()
        sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        logger.info(
            "Audio downsampled: %.1f KB → %.1f KB (16kHz mono)",
            len(audio_bytes) / 1024, buf.getbuffer().nbytes / 1024,
        )
        return buf.read()

    def transcribe(self, audio_bytes: bytes, language: str | None = None) -> str:
        """
        Converts audio bytes to text.

        Args:
            audio_bytes: Raw audio data in WAV format (bytes).
            language: Target language code (e.g., "de"). If None, detects automatically.

        Returns:
            Transcribed text.
        """
        if self._provider == "OpenAI":
            # Downsample to reduce upload size
            compact_audio = self._downsample_to_16k_mono(audio_bytes)
            audio_file = io.BytesIO(compact_audio)
            audio_file.name = "audio.wav"

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
            # Local Whisper needs a temp file
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
                    "STT completed (Local) - language=%s (%.1f%%), text='%s'",
                    detected_lang, prob * 100, text[:80],
                )

        return text.strip() 