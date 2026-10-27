"""Grounded LLM integration for Sentinel.

The LLM receives retrieved chunks and is explicitly restricted to those
chunks. Ollama must be running locally, for example::

    ollama run llama3.2:3b
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import requests


OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2:3b"


class LLMError(RuntimeError):
    """Raised when the Ollama request fails."""


def _build_context(chunks: Sequence[Mapping[str, Any]]) -> str:
    sources = []
    for number, chunk in enumerate(chunks, start=1):
        text = str(chunk.get("text", "")).strip()
        if text:
            title = str(chunk.get("title", "unknown source"))
            sources.append(f"[Source {number}: {title}]\n{text}")
    return "\n\n".join(sources)


def _generate_extractive_summary(
    question: str,
    chunks: Sequence[Mapping[str, Any]]
) -> str:
    """Generates a plain-English synthesis when Ollama is unavailable."""
    sections = []

    all_low = all(
        chunk.get("confidence") == "low" or chunk.get("is_out_of_domain")
        for chunk in chunks
    )
    if all_low:
        return "I do not have enough evidence to answer this question. The query does not appear in the verified medical textbooks."

    sections.append(
        f"Quick Summary in Plain English:\n"
        f"Here is an easy-to-understand, evidence-based guide regarding your query: \"{question}\"."
    )

    textbook_bullets = []
    for idx, chunk in enumerate(chunks, start=1):
        sentences = [
            sentence.strip()
            for sentence in chunk.get("text", "").split(".")
            if len(sentence.strip()) > 20
        ][:2]
        if sentences:
            textbook_bullets.append(f"- {'. '.join(sentences)}. [Source {idx}]")

    sections.append("Key Facts from Medical Textbooks:\n" + "\n".join(textbook_bullets))
    sections.append(
        "For personal medical concerns, discuss your situation with a healthcare provider."
    )

    return "\n\n".join(sections)


def answer_question(
    question: str,
    chunks: Sequence[Mapping[str, Any]],
    *,
    model: str = MODEL,
    ollama_url: str = OLLAMA_URL,
    timeout: float = 120.0,
    session: requests.Session | None = None,
) -> str:
    """Generate an answer from retrieved chunks only.

    ``chunks`` accepts the dictionaries returned by ``retrieve.retrieve``.
    """
    context = _build_context(chunks)
    if not context:
        raise LLMError("No retrieved context was provided")

    prompt = (
        "You are Sentinel, a medical assistant answering from retrieved evidence.\n"
        "Answer the question using ONLY the retrieved medical textbook sources below.\n\n"
        "Guidelines:\n"
        "- Use clear, accessible language and briefly explain medical terms.\n"
        "- Cite each factual statement with [Source N].\n"
        "- Do not speculate or use outside knowledge. If the sources are insufficient, "
        "say exactly: I do not have enough evidence to answer this question.\n\n"
        f"Question: {question}\n\nRetrieved sources:\n{context}"
    )

    if all(c.get("confidence") == "low" or c.get("is_out_of_domain") for c in chunks):
        return "I do not have enough evidence to answer this question. The query does not appear in the verified medical textbooks."

    client = session or requests.Session()
    try:
        response = client.post(
            f"{ollama_url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=(1.5, timeout),
        )
        response.raise_for_status()
        answer = str(response.json()["response"]).strip()
        if answer:
            return answer
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        # Fallback to extractive summary if Ollama is offline or timed out
        print(f"[LLM] Ollama unavailable ({exc}), generating extractive clinical summary...")
        return _generate_extractive_summary(question, chunks)

    return _generate_extractive_summary(question, chunks)


def answer_with_retrieval(question: str, top_k: int = 3) -> str:
    """Retrieve evidence and generate a grounded answer in one call."""
    from src.retrieve import retrieve

    return answer_question(question, retrieve(question, top_k=top_k))
