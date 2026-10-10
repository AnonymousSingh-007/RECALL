from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

app = FastAPI(title="RECALL", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = Path(__file__).parent / "frontend"


@app.get("/")
async def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Also serve CSS/JS at root level so relative paths in index.html work
@app.get("/style.css")
async def serve_css():
    return FileResponse(FRONTEND_DIR / "style.css", media_type="text/css")


@app.get("/app.js")
async def serve_js():
    return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "RECALL"}


@app.post("/query")
async def query_endpoint(file: UploadFile = File(...)):
    """Accept an audio file (multipart), run full RAG pipeline.

    Returns the complete RAGResponse when the harness is wired up.
    Falls back to STT-only mode if indices aren't loaded yet.
    """
    from backend import stt, guardrails
    from backend.latency import LatencyTracker

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    tracker = LatencyTracker()

    try:
        # --- STT ---
        with tracker.track("stt"):
            result = await stt.transcribe(
                audio_bytes,
                filename=file.filename or "audio.wav",
                content_type=file.content_type or "audio/wav",
            )

        query = result["transcript"]
        detected_language = result["language_code"]

        # --- Guardrails Layer 1: input safety ---
        is_safe, reason = guardrails.check_input(query)
        if not is_safe:
            raise HTTPException(status_code=400, detail=reason)

        # --- Guardrails Layer 3: language mismatch ---
        lang_ok, lang_reason = guardrails.check_language(query)
        if not lang_ok:
            raise HTTPException(status_code=400, detail=lang_reason)

        # --- Try full harness if indices are loaded ---
        try:
            from backend.harness import RAGHarness
            from backend.retriever.faiss_store import load_all_stores
            from backend.retriever.bm25_store import load_bm25

            faiss_store = load_all_stores()
            bm25_store = load_bm25()
            harness = RAGHarness(faiss_store, bm25_store)
            response = await harness.run(audio_bytes)
            return response.model_dump()

        except (ImportError, FileNotFoundError, Exception):
            # Indices not available yet — return STT-only response
            # with a mock answer shape so frontend can render it
            return {
                "answer": f"[STT-only mode] Transcript: {query}",
                "sources": [],
                "strategy_used": [],
                "confidence": 0.0,
                "grounded": True,
                "latency_breakdown": {
                    "stt_ms": tracker.stages.get("stt", 0),
                    "classify_ms": 0,
                    "retrieve_ms": 0,
                    "rerank_ms": 0,
                    "guardrail_ms": 0,
                    "generate_ms": 0,
                    "total_ms": tracker.total_ms,
                },
                "query_language": detected_language,
                "query_text": query,
            }

    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))