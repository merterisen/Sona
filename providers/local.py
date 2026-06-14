"""
Local Providers — Faster-Whisper (STT), Kokoro ONNX (TTS), Ollama (LLM).

These run entirely on the local machine without cloud API calls.
"""

import io
import logging
import tempfile
from collections import defaultdict

import numpy as np
import soundfile as sf

import config
from providers.base import BaseSTT, BaseTTS, BaseLLM

logger = logging.getLogger(__name__)


# ─── Local STT (Faster-Whisper) ──────────────────────────────

class LocalSTT(BaseSTT):
    """Speech-to-Text using Faster-Whisper (local)."""

    def __init__(
        self,
        model_size: str = config.LOCAL_STT_MODEL_SIZE,
        device: str = config.LOCAL_STT_DEVICE,
        compute_type: str = config.LOCAL_STT_COMPUTE_TYPE,
    ):
        from faster_whisper import WhisperModel

        logger.info(
            "Loading LocalSTT (Faster-Whisper) — model=%s, device=%s",
            model_size, device,
        )
        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        logger.info("LocalSTT ready.")

    def transcribe(self, audio_bytes: bytes, language: str | None = None) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()

            segments, info = self._model.transcribe(
                tmp.name,
                beam_size=5,
                language=language,
            )

            text = " ".join(segment.text.strip() for segment in segments)

            logger.info(
                "STT completed (Local) — lang=%s (%.1f%%), text='%s'",
                info.language, info.language_probability * 100, text[:80],
            )

        return text.strip()


# ─── Local TTS (Kokoro ONNX) ─────────────────────────────────

class LocalTTS(BaseTTS):
    """Text-to-Speech using Kokoro ONNX (local)."""

    def __init__(
        self,
        voice: str = config.LOCAL_TTS_VOICE,
        lang: str = config.LOCAL_TTS_LANG,
        speed: float = config.LOCAL_TTS_SPEED,
    ):
        from kokoro_onnx import Kokoro
        from huggingface_hub import hf_hub_download

        logger.info("Loading LocalTTS (Kokoro) — voice=%s, lang=%s", voice, lang)

        model_path = hf_hub_download(
            repo_id="fastrtc/kokoro-onnx",
            filename="kokoro-v1.0.onnx",
        )
        voices_path = hf_hub_download(
            repo_id="fastrtc/kokoro-onnx",
            filename="voices-v1.0.bin",
        )

        self._kokoro = Kokoro(model_path, voices_path)
        self._voice = voice
        self._lang = lang
        self._speed = speed
        logger.info("LocalTTS ready.")

    def update_voice(self, voice: str, language: str) -> None:
        self._voice = voice
        self._lang = language
        logger.info("LocalTTS voice updated — voice=%s, lang=%s", voice, language)

    def synthesize(self, text: str) -> tuple[bytes, str]:
        logger.info("Synthesizing (Local) — text='%s'", text[:80])

        samples, sample_rate = self._kokoro.create(
            text,
            voice=self._voice,
            speed=self._speed,
            lang=self._lang,
        )

        logger.info(
            "TTS completed — %d samples, sr=%d, duration=%.1fs",
            len(samples), sample_rate, len(samples) / sample_rate,
        )

        buffer = io.BytesIO()
        sf.write(buffer, samples, sample_rate, format="WAV")
        buffer.seek(0)
        return buffer.read(), "audio/wav"


# ─── Local LLM (Ollama) ──────────────────────────────────────

class LocalLLM(BaseLLM):
    """LLM using Ollama (local)."""

    def __init__(
        self,
        model: str = config.LOCAL_LLM_MODEL,
        base_url: str = config.LOCAL_LLM_BASE_URL,
        temperature: float = config.LLM_TEMPERATURE,
    ):
        from langchain_ollama import ChatOllama

        logger.info("Loading LocalLLM (Ollama) — model=%s", model)
        self._llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
        )
        self._system_prompt = "You are a helpful assistant."
        self._histories: dict[str, list] = defaultdict(list)
        logger.info("LocalLLM ready.")

    def set_system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt
        logger.info("System prompt updated.")

    def get_response(self, user_message: str, session_id: str = "default") -> str:
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        logger.info("LLM request — session=%s, message='%s'", session_id, user_message[:80])

        history = self._histories[session_id]
        history.append(HumanMessage(content=user_message))

        messages = [SystemMessage(content=self._system_prompt)] + history
        response = self._llm.invoke(messages)
        result = response.content

        history.append(AIMessage(content=result))
        logger.info("LLM response — '%s'", result[:80])
        return result

    def clear_history(self, session_id: str = "default") -> None:
        self._histories.pop(session_id, None)
        logger.info("History cleared for session %s.", session_id)
