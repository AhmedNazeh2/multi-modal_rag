import asyncio
import io,os
import logging
from typing import Tuple
import numpy as np
import whisper
import edge_tts
from dotenv import load_dotenv

load_dotenv(override=True)

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE"))
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL")

logger = logging.getLogger(__name__)

# ---------------------------
# Load Whisper model for STT
# ---------------------------
try:
    whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
    logger.info(f"Loaded Whisper model '{WHISPER_MODEL_NAME}'.")
except Exception as e:
    logger.exception(f"Whisper model could not be loaded: {e}")
    whisper_model = None

# ---------------------------
# Female voices mapping
# ---------------------------
VOICE_MAP = {"ar": "ar-EG-SalmaNeural"}
DEFAULT_LANG = "ar"

# ---------------------------
# Speech-to-Text
# ---------------------------
async def transcribe_async(audio_data: np.ndarray) -> Tuple[str, str]:
    if whisper_model is None:
        logger.error("Whisper model is not loaded. STT disabled.")
        return "", DEFAULT_LANG

    loop = asyncio.get_event_loop()
    blocking_task = lambda: whisper_model.transcribe(audio_data, fp16=False, language="ar")
    logger.info("Starting Whisper transcription (Arabic)...")
    result = await loop.run_in_executor(None, blocking_task)
    text = result["text"].strip()
    logger.info(f"Transcription done. Text: '{text}'")
    return text, DEFAULT_LANG  

# ---------------------------
# Text-to-Speech (Edge TTS, Arabic female voice)
# ---------------------------
async def tts_async(text: str, lang: str = DEFAULT_LANG) -> bytes:
    voice_code = VOICE_MAP[DEFAULT_LANG]
    logger.info(f"Generating Arabic TTS with voice '{voice_code}'")

    communicate = edge_tts.Communicate(text, voice=voice_code)
    audio_buffer = io.BytesIO()

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_buffer.write(chunk["data"])

    audio_bytes = audio_buffer.getvalue()
    logger.info(f"TTS generation completed: {len(audio_bytes)} bytes")
    return audio_bytes

# ---------------------------
# Convert MP3 bytes to WAV bytes in-memory
# ---------------------------
def mp3_to_wav_bytes(mp3_bytes: bytes) -> bytes:
    from pydub import AudioSegment
    audio_segment = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
    wav_buffer = io.BytesIO()
    audio_segment.export(wav_buffer, format="wav")
    return wav_buffer.getvalue()