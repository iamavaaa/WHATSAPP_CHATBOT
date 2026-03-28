import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

try:
    # Works with: python -m src.app
    from src.rag_engine import RAGEngine
except ModuleNotFoundError:
    # Works with: python src/app.py
    from rag_engine import RAGEngine


_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

app = Flask(__name__)
rag = None


def get_rag() -> RAGEngine:
    global rag
    if rag is None:
        app.logger.info("Initializing RAG engine...")
        rag = RAGEngine()
    return rag


@app.get("/")
def index() -> tuple[str, int]:
    return (
        "WhatsApp RAG bot is running. Use GET /health for status and POST /whatsapp for Twilio webhook.",
        200,
    )


@app.get("/health")
def health() -> tuple[str, int]:
    return "ok", 200


def _whatsapp_reply() -> str:
    incoming_text = (request.form.get("Body") or "").strip()
    user_id = request.form.get("From", "unknown-user")

    if not incoming_text:
        reply_text = "Please send a message so I can help."
    else:
        try:
            reply_text = get_rag().answer(user_id=user_id, query=incoming_text)
        except Exception as exc:
            # Keep user-facing message safe, but log exact error server-side.
            app.logger.exception("RAG generation failed: %s", exc)
            reply_text = (
                "I hit a temporary issue while generating your answer. "
                "Please try again in a few seconds."
            )

    twilio_response = MessagingResponse()
    twilio_response.message(reply_text)
    return str(twilio_response)


@app.post("/whatsapp")
def whatsapp_webhook() -> str:
    return _whatsapp_reply()


@app.post("/")
def whatsapp_webhook_root_fallback() -> str:
    # Helps when Twilio sandbox is misconfigured to call the root URL.
    return _whatsapp_reply()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
