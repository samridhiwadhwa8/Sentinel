"""FastAPI backend for the Sentinel RAG and faithfulness pipeline.

Usage:
    python api.py
    # or
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Sentinel Retrieval API",
    description="Medical RAG pipeline with grounded LLM answers and a faithfulness classifier.",
    version="1.1.0",
)


class QueryRequest(BaseModel):
    question: str = Field(default="", description="The clinical query")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of evidence chunks to retrieve")


@app.get("/api/health")
def health_check():
    """Health check endpoint to verify backend connectivity."""
    return {"status": "ok", "service": "Sentinel Retrieval API", "version": "1.1.0"}


@app.post("/api/sentinel")
def query_sentinel(request: QueryRequest):
    """Executes the Sentinel Clinical RAG pipeline:

    1. Hybrid retrieval over MedRAG textbooks
    2. Grounded answer generation via local Ollama LLM
    3. Faithfulness classification and abstention
    """
    try:
        from src.sentinel import sentinel

        result = sentinel(question=request.question, top_k=request.top_k)
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Sentinel pipeline error: {exc}",
        ) from exc


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
