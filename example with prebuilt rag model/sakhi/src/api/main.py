from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
import asyncio
import logging
from pathlib import Path

# Ensure the root project directory is in the PYTHONPATH
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.rag import SakhiResources, build_graph
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger("sakhi.api")

# ── Global resources ──────────────────────────────────────────────────────────
resources: SakhiResources = None
pipeline: CompiledStateGraph = None
init_error: str = None


def _load_resources():
    """Synchronous initialization — runs in thread pool so it doesn't block."""
    logger.info("🚀 Starting up Sakhi API...")
    r = SakhiResources()
    p = build_graph(r)
    logger.info("✅ Sakhi API ready to serve requests.")
    return r, p


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan event handler (replaces deprecated on_event)."""
    global resources, pipeline, init_error
    try:
        # Run heavy initialization in thread pool — never blocks the event loop
        loop = asyncio.get_running_loop()
        resources, pipeline = await loop.run_in_executor(None, _load_resources)
    except Exception as e:
        init_error = str(e)
        logger.warning(f"⚠️ Sakhi pipeline uninitialized: {e}")
    yield
    # Cleanup (if needed in future)
    logger.info("🛑 Sakhi API shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sakhi AI Legal Companion API",
    description="Backend API for the Sakhi RAG engine",
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS — configurable via ALLOWED_ORIGINS env var ──────────────────────────
raw_origins = os.getenv("ALLOWED_ORIGINS", "")
if raw_origins.strip():
    ALLOWED_ORIGINS = [o.strip() for o in raw_origins.split(",") if o.strip()]
else:
    ALLOWED_ORIGINS = [
        "https://sakhi-alpha.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    chat_history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    answer: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if not pipeline:
        detail_msg = f"RAG pipeline is not ready: {init_error}" if init_error else "RAG pipeline is still initializing. Try again in a moment."
        raise HTTPException(status_code=503, detail=detail_msg)

    try:
        logger.info(f"  ❓ API received query: {request.query}")

        # Build chat history
        history = [
            msg.model_dump() if hasattr(msg, "model_dump") else msg.dict()
            for msg in request.chat_history
        ]

        # Count trailing clarification questions (ignoring initial assistant greeting)
        clarification_count = 0
        for msg in reversed(history):
            content = msg.get("content", "").strip()
            if msg["role"] == "assistant":
                if "Namaste" in content and "How can I assist" in content:
                    continue
                if content.endswith("?"):
                    clarification_count += 1
            elif msg["role"] == "user":
                continue
            else:
                break

        logger.info(f"  📊 Chat history: {len(history)} messages, {clarification_count} clarifications")

        initial_state = {
            "query":                  request.query,
            "corrected_query":        "",
            "understood_as":          "",
            "needs_clarification":    False,
            "clarification_question": None,
            "clarification_count":    clarification_count,
            "is_legal_query":         True,
            "search_queries":         [],
            "embeddings":             [],
            "chunks":                 [],
            "context":                "",
            "answer":                 "",
            "low_confidence":         False,
            "chat_history":           history,
        }

        # ✅ Run pipeline in thread pool — does NOT block the async event loop
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, pipeline.invoke, initial_state)

        answer = result.get("answer", "I'm sorry, I could not generate an answer.")
        return ChatResponse(answer=answer)

    except Exception as e:
        logger.error(f"❌ Error processing query: {e}")
        err_msg = str(e)
        if any(term in err_msg for term in ["401", "API Key", "Authentication", "api_key", "invalid_api_key", "Unauthorized"]):
            return ChatResponse(
                answer=(
                    "Namaste! 🙏\n\n"
                    "To enable my full AI legal intelligence and vector search, please configure your API keys in the `.env` file:\n\n"
                    "- **GROQ_API_KEY**: Get a free key at [console.groq.com](https://console.groq.com)\n"
                    "- **OPENROUTER_API_KEY**: Get a free key at [openrouter.ai/keys](https://openrouter.ai/keys)\n\n"
                    "Once added to `.env`, restart the app and I'll be ready to assist you! ✨"
                )
            )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        "status":   "healthy" if pipeline is not None else ("misconfigured" if init_error else "initializing"),
        "error":    init_error,
        "model":    "qwen/qwen3-embedding-8b",
        "llm":      os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        "version":  "2.0.0",
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port, reload=False)
