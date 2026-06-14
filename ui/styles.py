"""
Sona UI Styles — All CSS for the Streamlit interface.

Extracted from app.py to keep the UI layer clean and maintainable.
"""


def get_css() -> str:
    """Returns the full CSS for the Sona Streamlit UI."""
    return """
<style>
    /* ── Global Typography ───────────────────────── */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', Roboto, sans-serif;
    }

    /* ── Header ──────────────────────────────────── */
    .sona-header {
        text-align: center;
        padding: 0.75rem 1rem 0.25rem;
        margin-bottom: 0.25rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
    .sona-header h1 {
        font-size: 1.9rem;
        font-weight: 700;
        color: var(--text-color);
        letter-spacing: -0.02em;
        margin-bottom: 0;
    }

    /* ── Status Badge ────────────────────────────── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 500;
        margin: 0.5rem auto;
        background: transparent;
        border: 1px solid;
    }
    /* Status colors: border + text only, transparent fill */
    .status-recording   { color: #c0392b; border-color: #c0392b; }
    .status-transcribing { color: #b7770d; border-color: #b7770d; }
    .status-thinking    { color: var(--primary-color); border-color: var(--primary-color); }
    .status-speaking    { color: #1a7a4a; border-color: #1a7a4a; }
    .status-ready {
        color: var(--text-color);
        border-color: rgba(128, 128, 128, 0.45);
        opacity: 0.7;
    }

    /* ── Sidebar ─────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: var(--background-color);
        border-right: 1px solid rgba(128, 128, 128, 0.2);
    }
    .sidebar-divider {
        height: 1px;
        background: rgba(128, 128, 128, 0.2);
        margin: 0.85rem 0;
    }

    /* ── Chat Messages — WhatsApp Style ─────────── */
    .stChatMessage {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin-bottom: 0.6rem !important;
    }

    /* Bubble wrapper */
    .chat-bubble-wrapper {
        display: flex;
        width: 100%;
        margin-bottom: 0.5rem;
    }
    .chat-bubble-wrapper.user {
        justify-content: flex-end;
    }
    .chat-bubble-wrapper.assistant {
        justify-content: flex-start;
    }

    /* The actual bubble */
    .chat-bubble {
        max-width: 72%;
        padding: 0.65rem 1rem;
        border-radius: 18px;
        font-size: 0.92rem;
        line-height: 1.55;
        word-break: break-word;
        background: rgba(128, 128, 128, 0.15);
        color: var(--text-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .chat-bubble.user {
        border-bottom-right-radius: 4px;
    }
    .chat-bubble.assistant {
        border-bottom-left-radius: 4px;
    }

    /* Audio player inside bubble */
    .chat-bubble audio {
        display: block;
        margin-top: 0.45rem;
        width: 100%;
        height: 32px;
        border-radius: 8px;
    }

    /* Hide Streamlit's native chat avatars & containers when using custom bubbles */
    .stChatMessage [data-testid="stChatMessageAvatarUser"],
    .stChatMessage [data-testid="stChatMessageAvatarAssistant"] {
        display: none !important;
    }

    /* ── Welcome Card ────────────────────────────── */
    .welcome-card {
        text-align: center;
        padding: 1.5rem;
        border-radius: 4px;
        background: transparent;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin: 0.75rem 0;
    }
    .welcome-card h3 {
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: var(--text-color);
    }
    .welcome-card p {
        color: var(--text-color);
        opacity: 0.65;
        font-size: 0.875rem;
        line-height: 1.55;
        margin: 0.2rem 0;
    }
</style>
"""
