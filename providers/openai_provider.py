"""
OpenAI Providers — OpenAI Whisper (STT), OpenAI TTS, OpenAI Chat (LLM).

Cloud-based providers using the OpenAI API.
"""

import io
import logging
from collections import defaultdict

import numpy as np
import soundfile as sf

import config
from providers.base import BaseSTT, BaseTTS, BaseLLM

logger = logging.getLogger(__name__)


# ─── OpenAI STT (Whisper API) ────────────────────────────────

class OpenAISTT(BaseSTT):
    """Speech-to-Text using OpenAI Whisper API."""

    def __init__(self, api_key: str, model: str = config.OPENAI_STT_MODEL):
        import openai

        logger.info("Loading OpenAISTT — model=%s", model)
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model
        logger.info("OpenAISTT ready.")

    def _downsample_to_16k_mono(self, audio_bytes: bytes) -> bytes:
        """Resample audio to 16 kHz mono WAV to minimize upload size."""
        data, sr = sf.read(io.BytesIO(audio_bytes))

        # Convert stereo to mono
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Resample to 16 kHz if needed
        target_sr = 16000
        if sr != target_sr:
            ratio = target_sr / sr
            num_samples = int(len(data) * ratio)
            indices = np.linspace(0, len(data) - 1, num_samples)
            data = np.interp(indices, np.arange(len(data)), data).astype(np.float32)
            sr = target_sr

        buf = io.BytesIO()
        sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)

        logger.info(
            "Audio downsampled: %.1f KB → %.1f KB (16kHz mono)",
            len(audio_bytes) / 1024, buf.getbuffer().nbytes / 1024,
        )
        return buf.read()

    def transcribe(self, audio_bytes: bytes, language: str | None = None) -> str:
        compact_audio = self._downsample_to_16k_mono(audio_bytes)
        audio_file = io.BytesIO(compact_audio)
        audio_file.name = "audio.wav"

        kwargs = {}
        if language:
            kwargs["language"] = language

        response = self._client.audio.transcriptions.create(
            model=self._model,
            file=audio_file,
            **kwargs,
        )

        text = response.text
        logger.info("STT completed (OpenAI) — text='%s'", text[:80])
        return text.strip()


# ─── OpenAI TTS ───────────────────────────────────────────────

class OpenAITTS(BaseTTS):
    """Text-to-Speech using OpenAI TTS API."""

    def __init__(
        self,
        api_key: str,
        model: str = config.OPENAI_TTS_MODEL,
        voice: str = config.OPENAI_TTS_VOICE,
    ):
        import openai

        logger.info("Loading OpenAITTS — model=%s, voice=%s", model, voice)
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model
        self._voice = voice
        logger.info("OpenAITTS ready.")

    def update_voice(self, voice: str, language: str) -> None:
        self._voice = voice
        logger.info("OpenAITTS voice updated — voice=%s", voice)

    def synthesize(self, text: str) -> tuple[bytes, str]:
        logger.info("Synthesizing (OpenAI) — voice=%s, text='%s'", self._voice, text[:80])

        response = self._client.audio.speech.create(
            model=self._model,
            voice=self._voice,
            input=text,
            speed=0.9,
            response_format="mp3",
        )

        return response.content, "audio/mpeg"


# ─── OpenAI LLM (Chat API) ───────────────────────────────────

class OpenAILLM(BaseLLM):
    """LLM using OpenAI Chat API via LangChain."""

    def __init__(
        self,
        api_key: str,
        model: str = config.OPENAI_LLM_MODEL,
        temperature: float = config.LLM_TEMPERATURE,
    ):
        from langchain_openai import ChatOpenAI

        logger.info("Loading OpenAILLM — model=%s", model)
        self._llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
        )
        self._system_prompt = "You are a helpful assistant."
        self._histories: dict[str, list] = defaultdict(list)
        logger.info("OpenAILLM ready.")

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
