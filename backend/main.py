import os
import json
import logging
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Import our RAG Engine and Database Engine
from backend.rag_engine import RAGEngine
from backend.database import init_db, save_message, save_document_record, clear_document_records, get_sessions, delete_session, get_messages

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot-backend")

# Initialize database
try:
    init_db()
except Exception as db_err:
    logger.warning(f"Database init warning: {db_err}")

# Load environment variables
# Look for .env first in backend directory, then root
backend_env = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(backend_env):
    load_dotenv(backend_env)
else:
    load_dotenv()

app = FastAPI(title="Aura AI Chatbot API", version="1.0.0")

# Setup CORS for development frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global RAG Engine
rag_engine = RAGEngine()

# Get API key from env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    model: str = "openai/gpt-oss-120b"
    temperature: float = 0.7
    rag_enabled: bool = False

@app.get("/api/status")
async def get_status():
    """Verify that backend is online and Groq API key is valid."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {
            "status": "warning",
            "message": "GROQ_API_KEY is missing from backend/.env"
        }
    
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        client.models.list()
        return {
            "status": "ok",
            "message": "Groq API key verified & online.",
            "model_configured": "llama-3.3-70b-versatile"
        }
    except Exception as e:
        err_str = str(e)
        logger.warning(f"Groq status check error: {err_str}")
        if "PermissionDeniedError" in err_str or "not allowed by policy" in err_str or "401" in err_str or "403" in err_str:
            return {
                "status": "error",
                "message": "GROQ_API_KEY is invalid or restricted by Groq policy. Please provide a new key from console.groq.com"
            }
        return {
            "status": "error",
            "message": f"Groq API Key error: {err_str}"
        }

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a text or PDF file and index it for RAG search."""
    try:
        content = await file.read()
        filename = file.filename
        
        if not (filename.endswith('.txt') or filename.endswith('.pdf')):
            raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported.")
            
        success = rag_engine.add_document(filename, content)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process or index the document.")
            
        return {
            "message": f"Successfully uploaded and indexed '{filename}'",
            "filename": filename,
            "chunks_count": len(rag_engine.chunks)
        }
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"File upload error: {str(e)}")

@app.get("/api/documents")
async def list_documents():
    """Get the names of all currently loaded documents and chunk stats."""
    return {
        "documents": rag_engine.document_names,
        "total_chunks": len(rag_engine.chunks)
    }

@app.delete("/api/documents")
async def clear_documents():
    """Clear all documents from the RAG search index."""
    rag_engine.clear()
    return {"message": "All documents cleared from memory index."}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint supporting SSE streaming output, history, and RAG injection."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set on the server .env")

    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    except ImportError:
        raise HTTPException(status_code=500, detail="langchain-groq dependencies not properly installed")

    # 1. Base system instructions
    system_prompt = (
        "You are Aura, an intelligent, versatile, and highly capable AI Assistant. "
        "Your primary objective is to provide direct, clear, accurate, and comprehensive answers "
        "to the user's questions and prompts. "
        "Do NOT respond by asking counter-questions back to the user unless explicit clarification is strictly required. "
        "Format code blocks cleanly using markdown with specific language specifiers. "
        "Maintain a polite, helpful, engaging, and professional tone at all times. "
        "If applicable, at the very end of your response, you may optionally list 2-3 brief suggested follow-up questions under a '### 💡 Suggested Follow-ups' section."
    )

    # 2. Integrate RAG if enabled
    retrieved_contexts = []
    if request.rag_enabled:
        contexts_with_scores = rag_engine.retrieve(request.message, top_k=3)
        if contexts_with_scores:
            context_blocks = []
            for chunk, score in contexts_with_scores:
                context_blocks.append(f"[Source: {chunk['source']} | Rel: {score:.2f}]\n{chunk['text']}")
                retrieved_contexts.append({
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "score": score
                })
            
            system_prompt += (
                "\n\nYou have access to the following retrieved document contexts. Use this information to answer "
                "the user's request accurately. Cite sources explicitly when relevant. If the answer is not present in the contexts "
                "but is standard general knowledge, answer using your knowledge while noting that it was not found in the uploaded text.\n\n"
                "=== RETRIEVED CONTEXT ===\n" + "\n\n".join(context_blocks) + "\n========================="
            )

    # 3. Build chat history messages
    messages = [SystemMessage(content=system_prompt)]
    
    # Append conversation history (skipping empty content placeholders)
    for msg in request.history:
        if not msg.content or not msg.content.strip():
            continue
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))
            
    # Append current user prompt
    messages.append(HumanMessage(content=request.message))

    # 4. Stream response generator
    def response_streamer():
        # Send RAG sources first as a metadata event
        if request.rag_enabled and retrieved_contexts:
            yield f"data: {json.dumps({'type': 'rag_sources', 'sources': retrieved_contexts})}\n\n"

        models_to_try = [request.model, "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b", "groq/compound"]
        seen = set()
        models_list = [m for m in models_to_try if not (m in seen or seen.add(m))]

        success = False
        last_error = ""

        for selected_model in models_list:
            try:
                llm = ChatGroq(
                    api_key=api_key,
                    model=selected_model,
                    temperature=request.temperature,
                    streaming=True
                )
                
                stream_generated = False
                for chunk in llm.stream(messages):
                    if chunk.content:
                        stream_generated = True
                        yield f"data: {json.dumps({'type': 'content', 'text': chunk.content})}\n\n"
                
                if stream_generated:
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    success = True
                    break
            except Exception as e:
                logger.warning(f"Model {selected_model} stream attempt failed: {e}")
                last_error = str(e)
                continue

        if not success:
            logger.error(f"All model streaming attempts failed: {last_error}")
            yield f"data: {json.dumps({'type': 'error', 'error': f'Groq Stream Error: {last_error}'})}\n\n"

    return StreamingResponse(response_streamer(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    # Read host and port from env
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)
