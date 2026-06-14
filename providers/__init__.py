"""
Sona Providers — Factory for creating provider instances.

Usage:
    from providers import create_providers
    stt, tts, llm = create_providers("Local")
    stt, tts, llm = create_providers("OpenAI", api_key="sk-...")
"""

from providers.base import BaseSTT, BaseTTS, BaseLLM


def create_providers(
    provider_name: str,
    api_key: str | None = None,
) -> tuple[BaseSTT, BaseTTS, BaseLLM]:
    """
    Factory function that creates a matched set of STT, TTS, and LLM providers.

    Args:
        provider_name: Provider identifier ("Local" or "OpenAI").
        api_key: API key (required for cloud providers).

    Returns:
        Tuple of (stt, tts, llm) provider instances.

    Raises:
        ValueError: If provider_name is not recognized.
    """
    if provider_name == "Local":
        from providers.local import LocalSTT, LocalTTS, LocalLLM
        return LocalSTT(), LocalTTS(), LocalLLM()

    elif provider_name == "OpenAI":
        if not api_key:
            raise ValueError("OpenAI provider requires an API key.")
        from providers.openai_provider import OpenAISTT, OpenAITTS, OpenAILLM
        return OpenAISTT(api_key), OpenAITTS(api_key), OpenAILLM(api_key)

    else:
        raise ValueError(
            f"Unknown provider: '{provider_name}'. "
            f"Supported providers: Local, OpenAI"
        )


__all__ = ["BaseSTT", "BaseTTS", "BaseLLM", "create_providers"]
