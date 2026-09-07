import os
import json
import sqlite3
import logging
from typing import List, Dict, Optional

logger = logging.getLogger("chatbot-database")

DB_PATH = os.getenv("DB_PATH", os.path.join(os.path.dirname(__file__), "aura_agent.db"))

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database tables for sessions, messages, and uploaded documents."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Messages table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions (id) ON DELETE CASCADE
        )
    ''')

    # Documents table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploaded_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            chunk_count INTEGER NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")

def save_session(session_id: str, title: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_sessions (id, title) VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET title=excluded.title",
        (session_id, title)
    )
    conn.commit()
    conn.close()

def get_sessions() -> List[Dict]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, created_at FROM chat_sessions ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [{"id": r["id"], "title": r["title"], "created_at": r["created_at"]} for r in rows]

def delete_session(session_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()

def save_message(session_id: str, role: str, content: str, sources: Optional[List] = None):
    conn = get_db_connection()
    cursor = conn.cursor()
    # Ensure session exists
    cursor.execute("INSERT OR IGNORE INTO chat_sessions (id, title) VALUES (?, ?)", (session_id, "New Chat"))
    
    sources_str = json.dumps(sources) if sources else None
    cursor.execute(
        "INSERT INTO chat_messages (session_id, role, content, sources_json) VALUES (?, ?, ?, ?)",
        (session_id, role, content, sources_str)
    )
    conn.commit()
    conn.close()

def get_messages(session_id: str) -> List[Dict]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT role, content, sources_json FROM chat_messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
    rows = cursor.fetchall()
    conn.close()
    
    result = []
    for r in rows:
        msg = {"role": r["role"], "content": r["content"]}
        if r["sources_json"]:
            try:
                msg["sources"] = json.loads(r["sources_json"])
            except Exception:
                pass
        result.append(msg)
    return result

def save_document_record(filename: str, chunk_count: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO uploaded_documents (filename, chunk_count) VALUES (?, ?) ON CONFLICT(filename) DO UPDATE SET chunk_count=excluded.chunk_count",
        (filename, chunk_count)
    )
    conn.commit()
    conn.close()

def get_document_records() -> List[Dict]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT filename, chunk_count, uploaded_at FROM uploaded_documents")
    rows = cursor.fetchall()
    conn.close()
    return [{"filename": r["filename"], "chunk_count": r["chunk_count"]} for r in rows]

def clear_document_records():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM uploaded_documents")
    conn.commit()
    conn.close()
