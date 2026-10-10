from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from pathlib import Path
import traceback

load_dotenv()

app = FastAPI(title="RECALL", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent / "frontend"

faiss_store = None
bm25_store = None
harness = None


@app.on_event("startup")
async def startup():
    global faiss_store, bm25_store, harness
    from backend.harness import RAGHarness
    from backend.retriever.faiss_store import load_all_stores
    from backend.retriever.bm25_store import load_bm25

    print("RECALL: loading indices...")
    faiss_store = load_all_stores()
    bm25_store = load_bm25()
    harness = RAGHarness(faiss_store, bm25_store)
    print("RECALL: all indices loaded, harness ready.")


@app.get("/")
async def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/style.css")
async def serve_css():
    return FileResponse(FRONTEND_DIR / "style.css", media_type="text/css")


@app.get("/app.js")
async def serve_js():
    return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "RECALL", "harness_ready": harness is not None}


@app.post("/query")
async def query_endpoint(file: UploadFile = File(...)):
    """Accept an audio file (multipart), run full RAG pipeline."""
    from backend.latency import LatencyTracker

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    if harness is None:
        raise HTTPException(status_code=503, detail="Harness not ready yet — server still starting up")

    try:
        response = await harness.run(audio_bytes)
        return response.model_dump()
    except ValueError as exc:
        # guardrail rejections (unsafe input, language mismatch, no relevant context)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        print(f"Harness failed: {exc}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")