import os
import logging
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import Pinecone


load_dotenv()
logger = logging.getLogger(__name__)


class RAGEngine:
    """RAG engine with simple in-memory conversation history per user."""

    def __init__(self) -> None:
        self._retriever: Optional[object] = None
        self._pinecone_index = None
        self._pinecone_namespace = os.getenv("PINECONE_NAMESPACE", "")
        self._pinecone_embedder: Optional[HuggingFaceEmbeddings] = None
        self._init_retriever()

        self._llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            temperature=0.2,
        )
        self._history: Dict[str, List[str]] = {}
        self._max_turns = 6
        self._max_context_chars = 3500
        self._max_history_chars = 1200

    def _init_retriever(self) -> None:
        use_pinecone = os.getenv("USE_PINECONE", "true").lower() == "true"
        if use_pinecone:
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
            if pinecone_api_key and pinecone_index_name:
                pc = Pinecone(api_key=pinecone_api_key)
                self._pinecone_index = pc.Index(pinecone_index_name)
                self._pinecone_embedder = HuggingFaceEmbeddings(
                    model_name=os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")
                )
                return

        # Fallback: local vector DB (useful for local development)
        render_env = os.getenv("RENDER", "")
        use_local_vector_default = "false" if render_env else "true"
        use_local_vector = os.getenv("USE_LOCAL_VECTOR_DB", use_local_vector_default).lower() == "true"
        if use_local_vector:
            chroma_dir = os.getenv("CHROMA_DB_DIR", "data/chroma_db")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_db = Chroma(
                persist_directory=chroma_dir,
                embedding_function=embeddings,
            )
            self._retriever = vector_db.as_retriever(search_kwargs={"k": 4})

    def _fallback_from_context(self, docs: list[str]) -> str:
        """
        Used when the LLM is rate-limited/quota-exhausted.
        Keeps the WhatsApp demo working by answering directly from retrieved text.
        """
        if not docs:
            return (
                "I’m having trouble generating a response right now (LLM quota exceeded). "
                "Could you clarify your question a bit?"
            )

        top = docs[0] or ""
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
        docs: list[str] = []
        if self._pinecone_index is not None and self._pinecone_embedder is not None:
            try:
                query_vec = self._pinecone_embedder.embed_query(query)
                result = self._pinecone_index.query(
                    vector=query_vec,
                    top_k=4,
                    include_metadata=True,
                    namespace=self._pinecone_namespace,
                )
                matches = result.get("matches", []) if isinstance(result, dict) else getattr(result, "matches", [])
                for item in matches:
                    metadata = item.get("metadata", {}) if isinstance(item, dict) else getattr(item, "metadata", {})
                    text = metadata.get("text", "") if isinstance(metadata, dict) else ""
                    if text:
                        docs.append(text)
            except Exception as exc:
                logger.exception("Pinecone retrieval failed: %s", exc)
                docs = []
        elif self._retriever:
            docs = [doc.page_content for doc in self._retriever.invoke(query)]

        context = "\n\n".join(docs) if docs else "No relevant context found."
        if len(context) > self._max_context_chars:
            context = context[: self._max_context_chars] + "..."

        prompt = f"""
You are a WhatsApp support assistant using RAG.

Rules:
- Prioritize the provided context when it is relevant.
- If context is missing or weak, answer using your general knowledge and clearly say it is a general answer.
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
