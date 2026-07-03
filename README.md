# Aura Agent: High-Performance React + FastAPI Chatbot Sandbox

A premium, recruiter-ready AI Chatbot web sandbox featuring real-time stream responses, a custom zero-dependency Retrieval-Augmented Generation (RAG) search index, and a modern glassmorphic interface.

Designed to showcase modern software engineering best practices, microservice architectures, and custom semantic text retrieval implementations.

## 🚀 Key Features

* **⚡ Lightning Fast Streaming**: Uses Server-Sent Events (SSE) from a Python backend to stream LLM responses token-by-token.
* **🔍 Custom In-Memory RAG Engine**: Pure Python TF-IDF Vectorizer and Cosine Similarity search index. No database configurations needed—chunking, indexing, and matching happen dynamically in memory.
* **🎙️ Voice Dictation**: Integrated native web browser Speech-to-Text API for hands-free queries.
* **🎨 Glassmorphic Interface**: A bespoke frontend styled with pure CSS (no bloated frameworks) using backdrop blur effects, neon gradient glows, and smooth responsive transitions.
* **⚙️ Decoupled Architecture**: Separation of concerns between a React + TypeScript SPA frontend and a FastAPI backend service.

---

## 🛠️ Architecture & Data Flow

```mermaid
graph TD
    User([User]) <--> |React UI| FE[Frontend: React + Vite + TS]
    FE <--> |JSON API / SSE Stream| BE[Backend: FastAPI]
    BE <--> |Query & Retrieval| RAG[Custom RAG Search Index]
    BE <--> |Streaming API Client| Groq[Groq Llama 3.3 LLM]
    
    subgraph Context Ingestion (RAG)
        UploadFile[Upload PDF / TXT] --> BE
        BE --> |Parse & Chunk| RAG
    end
```

---

## 💻 Tech Stack

* **Frontend**: React 18, Vite, TypeScript, Lucide Icons, Marked (Markdown Parser).
* **Backend**: FastAPI, Uvicorn, LangChain-Groq, python-dotenv, PyPDF (for PDF context extraction).
* **Styling**: Vanilla CSS3, Google Fonts (Outfit, Inter, Fira Code), CSS variables.

---

## ⚙️ Quick Start

### 1. Backend Setup

1. Open your terminal in the `backend/` directory.
2. Install Python dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Set your Groq API key in the `.env` file inside the `backend` folder:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```
4. Run the backend server:
   ```bash
   uvicorn backend.main:app --reload
   ```

The backend API will start running at [http://127.0.0.1:8000](http://127.0.0.1:8000). You can explore the interactive API docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 2. Aura Agent Frontend Setup

1. Open your terminal in the `Aura agent/` directory.
2. Install node modules:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```

Vite will serve the frontend at [http://localhost:5173](http://localhost:5173). All `/api` requests are proxied directly to the FastAPI server automatically.

---

## 💡 Code Highlights for Recruiters

* **Zero-Dependency Vector Search**: In `backend/rag_engine.py`, the tokenization, stop-word removal, TF-IDF weights, L2 normalization, and cosine-similarity computations are coded in pure Python to demonstrate strong mathematical foundations of information retrieval.
* **Low-Latency Streaming**: Standard HTTP streaming in `backend/main.py` sends text tokens instantly from Groq to the client, providing a fluid user experience.
* **Type-Safe Frontend**: Written in TypeScript using React functional components, hooks, and clean state management.
