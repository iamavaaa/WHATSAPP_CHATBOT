import os
from typing import Dict, List

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


class RAGEngine:
    """RAG engine with simple in-memory conversation history per user."""

    def __init__(self) -> None:
        # Render/free serverless-like environments can time out if we load local
        # embedding + vector DB eagerly. Allow disabling local retrieval by env.
        render_env = os.getenv("RENDER", "")
        use_local_vector_default = "false" if render_env else "true"
        use_local_vector = os.getenv("USE_LOCAL_VECTOR_DB", use_local_vector_default).lower() == "true"

        self._retriever = None
        if use_local_vector:
            chroma_dir = os.getenv("CHROMA_DB_DIR", "data/chroma_db")
            self._embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self._vector_db = Chroma(
                persist_directory=chroma_dir,
                embedding_function=self._embeddings,
            )
            self._retriever = self._vector_db.as_retriever(search_kwargs={"k": 4})

        self._llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            temperature=0.2,
        )
        self._history: Dict[str, List[str]] = {}
        self._max_turns = 6
        self._max_context_chars = 3500
        self._max_history_chars = 1200

    def _fallback_from_context(self, docs: list) -> str:
        """
        Used when the LLM is rate-limited/quota-exhausted.
        Keeps the WhatsApp demo working by answering directly from retrieved text.
        """
        if not docs:
            return (
                "I’m having trouble generating a response right now (LLM quota exceeded). "
                "Could you clarify your question a bit?"
            )

        top = docs[0].page_content or ""
        snippet = top.strip()
        if len(snippet) > self._max_context_chars:
            snippet = snippet[: self._max_context_chars] + "..."

        return (
            "My AI is temporarily unavailable (LLM quota exceeded), but I found relevant info in the dataset:\n\n"
            f"{snippet}\n\n"
            "Reply with a follow-up like: “more details” or ask a more specific question."
        )

    def _history_text(self, user_id: str) -> str:
        items = self._history.get(user_id, [])
        if not items:
            return "No previous messages."
        text = "\n".join(items[-self._max_turns :])
        return text[-self._max_history_chars :]

    def _update_history(self, user_id: str, user_message: str, bot_message: str) -> None:
        turns = self._history.setdefault(user_id, [])
        turns.append(f"User: {user_message}")
        turns.append(f"Bot: {bot_message}")
        if len(turns) > self._max_turns * 2:
            self._history[user_id] = turns[-self._max_turns * 2 :]

    def answer(self, user_id: str, query: str) -> str:
        docs = self._retriever.invoke(query) if self._retriever else []
        context = "\n\n".join(doc.page_content for doc in docs) if docs else "No relevant context found."
        if len(context) > self._max_context_chars:
            context = context[: self._max_context_chars] + "..."

        prompt = f"""
You are a WhatsApp support assistant using RAG.

Rules:
- Use only the provided context when possible.
- If context is missing, say you are not sure and ask for clarification.
- Keep answers concise and readable for WhatsApp.
- Use bullet points when listing items.

Conversation history:
{self._history_text(user_id)}

Retrieved context:
{context}

Current user message:
{query}
"""
        try:
            response = self._llm.invoke(prompt)
            text = response.content if isinstance(response.content, str) else str(response.content)
        except Exception:
            # Gemini quota failures are expected on free tiers. Return a context-only answer.
            text = self._fallback_from_context(docs)
        self._update_history(user_id, query, text)
        return text
