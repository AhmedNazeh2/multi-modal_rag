# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List
import uvicorn
import base64
from src.processor import DocumentProcessor
from src.embedding import EmbeddingManager
from src.chat import ChatManager
from pydub import AudioSegment
import soundfile as sf
import numpy as np
import io
import src.voice as voice
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
load_dotenv(override=True)


app = FastAPI(title="RAG Chat multi-modal PDF Chatbot with Voice Support")

processor = DocumentProcessor()
embedding_manager = EmbeddingManager()
chat_manager = ChatManager()

class UrlRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str 

@app.post("/upload/", summary="Upload documents to index")
async def upload_documents(files: List[UploadFile] = File(...)):
    all_docs = []
    for f in files:
        docs = processor.process_document(f)
        all_docs.extend(docs)
    if not all_docs:
        raise HTTPException(status_code=400, detail="No valid documents processed.")
    success = embedding_manager.create_embeddings(all_docs)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to create embeddings.")
    chat_manager.set_retriever(embedding_manager.retriever)
    return {"message": f"Processed {len(all_docs)} document chunks."}

@app.post("/chat/", response_model=QueryResponse, summary="Query the RAG model")
async def chat_endpoint(request: QueryRequest):
    if embedding_manager.retriever is None:
        raise HTTPException(status_code=400, detail="No documents have been indexed yet.")

    result = chat_manager.chain({"question": request.query})
    answer = result["answer"]

    return QueryResponse(answer=answer)


@app.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            audio_bytes = await websocket.receive_bytes()
            
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            audio_segment = audio_segment.set_channels(1)
            audio_segment = audio_segment.set_frame_rate(voice.SAMPLE_RATE)
            wav_bytes = io.BytesIO()
            audio_segment.export(wav_bytes, format="wav")
            audio_np, sr = sf.read(io.BytesIO(wav_bytes.getvalue()), dtype="float32")
            if audio_np.ndim > 1:
                audio_np = np.mean(audio_np, axis=1)
            
            transcript, lang_code = await voice.transcribe_async(audio_np)
            
            answer = chat_manager.generate_response(transcript, [])
            speech_bytes = await voice.tts_async(answer, lang_code)
            print(f"TTS_DEBUG: Speech bytes generated: {len(speech_bytes)} bytes.")
            await websocket.send_json({
                "type": "full_response",
                "transcript": transcript,
                "answer": answer,
                "audio": base64.b64encode(speech_bytes).decode("utf-8")
            })

    except WebSocketDisconnect:
        print("Client disconnected")
        

@app.post("/process-url/", summary="Process a URL and add content to index")
async def process_url_endpoint(request: UrlRequest):
    """
    Scrapes content from a URL and adds it to the RAG knowledge base.
    """
    url = request.url
    print(f"Received URL for processing: {url}")
    
    docs = processor.process_url(url)
    
    if not docs:
        raise HTTPException(status_code=400, detail="Failed to extract content from URL or URL is empty.")
    success = embedding_manager.create_embeddings(docs)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update embeddings with URL content.")
    chat_manager.set_retriever(embedding_manager.retriever)
    
    return {
        "message": f"Successfully processed URL. Added {len(docs)} chunks to the knowledge base.",
        "source": url
    }        


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
