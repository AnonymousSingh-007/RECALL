import httpx
import os

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


async def transcribe(audio_bytes: bytes, filename: str = "audio.wav", content_type: str = "audio/wav", retries: int = 2) -> dict:
    """Call Sarvam STT API and return {"transcript": str, "language_code": str}.

    Sarvam REST API expects multipart/form-data with:
      - file: audio file (WAV/MP3/AAC/FLAC/OGG)
      - model: "saaras:v2" (recommended)

    Response JSON: {"request_id": ..., "transcript": ..., "language_code": ...}
    """
    key = SARVAM_API_KEY or os.getenv("SARVAM_API_KEY")
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    SARVAM_STT_URL,
                    headers={"api-subscription-key": key},
                    files={"file": (filename, audio_bytes, content_type)},
                    data={"model": "saaras:v3"},
                )
                resp.raise_for_status()
                body = resp.json()
                return {
                    "transcript": body["transcript"],
                    "language_code": body.get("language_code", "unknown"),
                }
        except (httpx.HTTPError, KeyError) as exc:
            if attempt == retries:
                raise RuntimeError(f"STT failed after {retries + 1} attempts: {exc}")
    raise RuntimeError("STT failed after retries")