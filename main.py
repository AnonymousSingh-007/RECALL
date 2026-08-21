from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="RECALL", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "RECALL"}


@app.post("/query")
async def query_endpoint(file: UploadFile = File(...)):
    """Accept an audio file (multipart), run STT, return transcript.

    Once the full pipeline is wired, this will return the complete
    RAGResponse. For Day 1, it returns the STT result so we can
    confirm the voice → text path works end-to-end.
    """
    from backend import stt

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        result = await stt.transcribe(
            audio_bytes,
            filename=file.filename or "audio.wav",
            content_type=file.content_type or "audio/wav",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return {
        "transcript": result["transcript"],
        "language_code": result["language_code"],
        "status": "stt_only",
    }