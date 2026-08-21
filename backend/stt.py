import httpx
import os

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


async def transcribe(audio_bytes: bytes, retries: int = 2) -> str:
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    SARVAM_STT_URL,
                    headers={"api-subscription-key": SARVAM_API_KEY},
                    files={"file": ("audio.wav", audio_bytes, "audio/wav")},
                )
                resp.raise_for_status()
                return resp.json()["transcript"]
        except (httpx.HTTPError, KeyError):
            if attempt == retries:
                raise
    raise RuntimeError("STT failed after retries")