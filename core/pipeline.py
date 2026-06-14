"""
AudioPipeline — Framework-agnostic STT → LLM → TTS pipeline.

Orchestrates the full audio processing flow without any UI dependencies.
Returns a structured result that any UI layer can render.
"""

import time
import logging
from dataclasses import dataclass, field

from providers.base import BaseSTT, BaseTTS, BaseLLM

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a full audio processing pipeline run."""
    user_text: str
    ai_response: str
    audio_bytes: bytes | None = None
    audio_mime: str | None = None
    timings: dict[str, float] = field(default_factory=dict)


class AudioPipeline:
    """
    Orchestrates: Audio → STT → LLM → TTS.

    Fully decoupled from UI frameworks. Accepts provider instances
    through constructor injection.
    """

    def __init__(self, stt: BaseSTT, llm: BaseLLM, tts: BaseTTS):
        self._stt = stt
        self._llm = llm
        self._tts = tts

    def process(
        self,
        audio_bytes: bytes,
        session_id: str,
        language: str,
        on_status: callable | None = None,
    ) -> PipelineResult | None:
        """
        Run the full audio pipeline.

        Args:
            audio_bytes: Raw audio data (WAV).
            session_id: Conversation session identifier.
            language: Target language code (e.g., "de").
            on_status: Optional callback to report status changes
                       (e.g., "transcribing", "thinking", "speaking").

        Returns:
            PipelineResult with all outputs and timings, or None if no speech detected.
        """
        def _set_status(status: str):
            if on_status:
                on_status(status)

        timings = {}
        pipeline_start = time.perf_counter()
        audio_size_kb = len(audio_bytes) / 1024
        logger.info("⏱️ Pipeline started — audio size: %.1f KB", audio_size_kb)

        # 1. STT: Audio → Text
        _set_status("transcribing")
        t0 = time.perf_counter()
        user_text = self._stt.transcribe(audio_bytes, language=language)
        timings["stt"] = time.perf_counter() - t0
        logger.info("⏱️ STT completed in %.2f seconds", timings["stt"])

        if not user_text.strip():
            logger.info("No speech detected.")
            _set_status("ready")
            return None

        # 2. LLM: Text → Response
        _set_status("thinking")
        t0 = time.perf_counter()
        ai_response = self._llm.get_response(user_text, session_id)
        timings["llm"] = time.perf_counter() - t0
        logger.info("⏱️ LLM completed in %.2f seconds", timings["llm"])

        # 3. TTS: Response → Audio
        _set_status("speaking")
        audio_response = None
        audio_mime = None
        try:
            t0 = time.perf_counter()
            audio_response, audio_mime = self._tts.synthesize(ai_response)
            timings["tts"] = time.perf_counter() - t0
            logger.info("⏱️ TTS completed in %.2f seconds", timings["tts"])
        except Exception as e:
            logger.error("TTS error: %s", e)
            timings["tts"] = 0.0

        timings["total"] = time.perf_counter() - pipeline_start
        logger.info(
            "⏱️ PIPELINE TOTAL: %.2f seconds (STT=%.2f, LLM=%.2f, TTS=%.2f)",
            timings["total"], timings["stt"], timings["llm"], timings["tts"],
        )

        _set_status("ready")

        return PipelineResult(
            user_text=user_text,
            ai_response=ai_response,
            audio_bytes=audio_response,
            audio_mime=audio_mime,
            timings=timings,
        )
