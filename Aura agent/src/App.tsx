import React, { useState, useEffect, useRef } from 'react';
import { 
  Bot, User, Send, Mic, MicOff, Settings, 
  Upload, Trash2, FileText, CheckCircle, 
  XCircle, AlertTriangle, Sparkles, RefreshCw, Cpu
} from 'lucide-react';
import { marked } from 'marked';

// Types matching backend models
interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  sources?: Array<{
    text: string;
    source: string;
    score: number;
  }>;
}

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputText, setInputText] = useState('');
  const [model, setModel] = useState('llama-3.3-70b-versatile');
  const [temperature, setTemperature] = useState(0.7);
  const [ragEnabled, setRagEnabled] = useState(false);
  const [documents, setDocuments] = useState<string[]>([]);
  const [totalChunks, setTotalChunks] = useState(0);
  
  // UI states
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState<'online' | 'offline' | 'checking'>('checking');
  const [isRecording, setIsRecording] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom of chat
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  // Check backend status & get indexed documents on mount
  useEffect(() => {
    checkBackendStatus();
    fetchDocuments();
  }, []);

  const checkBackendStatus = async () => {
    setBackendStatus('checking');
    try {
      const response = await fetch('/api/status');
      const data = await response.json();
      if (response.ok && data.status === 'ok') {
        setBackendStatus('online');
        setErrorMessage('');
      } else {
        setBackendStatus('offline');
        setErrorMessage(data.message || 'Groq API Key is not configured in .env');
      }
    } catch (e) {
      setBackendStatus('offline');
      setErrorMessage('Failed to connect to backend server.');
    }
  };

  const fetchDocuments = async () => {
    try {
      const response = await fetch('/api/documents');
      if (response.ok) {
        const data = await response.json();
        setDocuments(data.documents || []);
        setTotalChunks(data.total_chunks || 0);
      }
    } catch (e) {
      console.error('Error fetching documents:', e);
    }
  };

  // Upload document for RAG
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setErrorMessage('');
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (response.ok) {
        // Automatically enable RAG when document is successfully uploaded
        setRagEnabled(true);
        fetchDocuments();
      } else {
        setErrorMessage(data.detail || 'Failed to upload document.');
      }
    } catch (e) {
      setErrorMessage('Network error during file upload.');
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  // Clear RAG memory
  const handleClearDocuments = async () => {
    try {
      const response = await fetch('/api/documents', {
        method: 'DELETE',
      });
      if (response.ok) {
        setDocuments([]);
        setTotalChunks(0);
        setRagEnabled(false);
      }
    } catch (e) {
      console.error('Error clearing documents:', e);
    }
  };

  // Speech recognition handler
  const startSpeechRecognition = () => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Speech recognition is not supported in this browser. Try Chrome or Safari.");
      return;
    }

    const rec = new SpeechRecognition();
    rec.continuous = false;
    rec.interimResults = false;
    rec.lang = 'en-US';

    rec.onstart = () => setIsRecording(true);
    rec.onend = () => setIsRecording(false);
    rec.onerror = (e: any) => {
      console.error("Speech Recognition Error:", e);
      setIsRecording(false);
    };
    rec.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript;
      setInputText(prev => prev + (prev ? ' ' : '') + transcript);
    };

    rec.start();
  };

  // Send Chat message
  const handleSendMessage = async (e?: React.FormEvent, textOverride?: string) => {
    if (e) e.preventDefault();
    const queryText = textOverride || inputText;
    if (!queryText.trim() || loading) return;

    // Clear input
    if (!textOverride) setInputText('');

    const newMessages: ChatMessage[] = [
      ...messages,
      { role: 'user', content: queryText }
    ];
    setMessages(newMessages);
    setLoading(true);
    setErrorMessage('');

    // Pre-insert Assistant loading bubble
    setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: queryText,
          history: messages.map(m => ({ role: m.role, content: m.content })),
          model,
          temperature,
          rag_enabled: ragEnabled
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Error communicating with Groq');
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder('utf-8');
      let assistantReply = '';
      let extractedSources: any[] = [];

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n');
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.substring(6));
                
                if (data.type === 'rag_sources') {
                  extractedSources = data.sources;
                } else if (data.type === 'content') {
                  assistantReply += data.text;
                  setMessages(prev => {
                    const updated = [...prev];
                    updated[updated.length - 1] = {
                      role: 'assistant',
                      content: assistantReply,
                      sources: extractedSources.length > 0 ? extractedSources : undefined
                    };
                    return updated;
                  });
                } else if (data.type === 'error') {
                  throw new Error(data.error);
                }
              } catch (err) {
                // Ignore parsing errors for incomplete stream segments
              }
            }
          }
        }
      }
    } catch (e: any) {
      console.error('Chat error:', e);
      setErrorMessage(e.message || 'Failed to get answer from Groq LLM.');
      // Remove loading message if it failed completely
      setMessages(prev => prev.slice(0, -1));
    } finally {
      setLoading(false);
    }
  };

  // Helper to render markdown inside messages safely
  const getMarkdownHtml = (content: string) => {
    try {
      const parsed = marked.parse(content);
      // Wait: marked.parse returns string or Promise<string>. In older versions, it returns a string.
      // Cast it or assert it as a string to please TypeScript.
      return { __html: parsed as string };
    } catch (e) {
      return { __html: content };
    }
  };

  // Trigger quick pills
  const handlePillClick = (text: string) => {
    handleSendMessage(undefined, text);
  };

  return (
    <div className="app-container">
      {/* Sidebar (Left) */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo-icon">
            <Sparkles size={20} />
          </div>
          <div className="logo-info">
            <h1 className="logo-text">Aura Agent</h1>
          </div>
        </div>

        <div className="sidebar-scrollable">
          {/* Server status section */}
          <div className="sidebar-section">
            <h2 className="section-title">Backend Connection</h2>
            <div className="toggle-container" onClick={checkBackendStatus} style={{ cursor: 'pointer' }}>
              <div className="toggle-info">
                <span className="toggle-title" style={{ fontSize: '0.85rem' }}>
                  {backendStatus === 'online' ? 'System Online' : backendStatus === 'checking' ? 'Connecting...' : 'Disconnected'}
                </span>
                <span className="toggle-desc">Click to refresh status</span>
              </div>
              <div className={`status-dot ${backendStatus === 'online' ? 'online' : backendStatus === 'checking' ? 'checking' : 'offline'}`} />
            </div>
          </div>

          {/* RAG Controller section */}
          <div className="sidebar-section">
            <h2 className="section-title">Context & RAG</h2>
            
            <div className="toggle-container" onClick={() => setRagEnabled(!ragEnabled)}>
              <div className="toggle-info">
                <span className="toggle-title">Enable RAG</span>
                <span className="toggle-desc">Reference uploaded files</span>
              </div>
              <label className="switch" onClick={(e) => e.stopPropagation()}>
                <input 
                  type="checkbox" 
                  checked={ragEnabled}
                  onChange={() => setRagEnabled(!ragEnabled)}
                />
                <span className="slider"></span>
              </label>
            </div>

            {/* Drag & Drop Upload Zone */}
            <div 
              className="file-upload-zone"
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload className="file-upload-icon" size={24} />
              <span className="file-upload-text">
                {uploading ? 'Processing File...' : 'Upload Knowledge File'}
              </span>
              <span className="file-upload-subtext">PDF or Text (Max 5MB)</span>
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileUpload} 
                style={{ display: 'none' }}
                accept=".txt,.pdf"
              />
            </div>

            {/* Indexed docs stats */}
            {documents.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginTop: '4px' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: 600 }}>
                  Active Index ({documents.length} Docs, {totalChunks} Chunks)
                </span>
                
                <div className="document-list">
                  {documents.map((doc, i) => (
                    <div className="document-item" key={i}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <FileText size={14} style={{ color: 'var(--primary)' }} />
                        <span className="document-name">{doc}</span>
                      </div>
                    </div>
                  ))}
                </div>

                <button 
                  className="clear-docs-btn" 
                  onClick={handleClearDocuments}
                >
                  <Trash2 size={14} />
                  Clear Index
                </button>
              </div>
            )}
          </div>
        </div>
      </aside>

      {/* Main Chat Feed (Right) */}
      <main className="chat-area">
        {/* Header */}
        <header className="chat-header">
          <div className="chat-title-info">
            <h2 className="chat-title">Interactive Sandbox</h2>
            <div className="connection-badge">
              <span style={{ color: 'var(--text-muted)' }}>RAG Status:</span>
              <span style={{ color: ragEnabled ? 'var(--accent-cyan)' : 'var(--text-muted)', fontWeight: 500 }}>
                {ragEnabled ? 'ACTIVE' : 'INACTIVE'}
              </span>
            </div>
          </div>
          
          <button className="icon-btn" onClick={() => setMessages([])} title="Clear conversation history">
            <Trash2 size={18} />
          </button>
        </header>

        {/* Messages feed */}
        <div className="messages-container">
          {errorMessage && (
            <div style={{ 
              background: 'rgba(239, 68, 68, 0.1)', 
              border: '1px solid rgba(239, 68, 68, 0.2)', 
              color: 'var(--accent-red)',
              padding: '12px 16px',
              borderRadius: '12px',
              fontSize: '0.85rem',
              display: 'flex',
              alignItems: 'center',
              gap: '10px'
            }}>
              <AlertTriangle size={16} />
              <span>{errorMessage}</span>
            </div>
          )}

          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">
                <Sparkles size={48} />
              </div>
              <h3 className="empty-state-title">Welcome to Aura Agent</h3>
              <p className="empty-state-desc">
                An advanced interactive AI Chatbot web sandbox with document indexing support. Upload documents, enable context reference, and chat.
              </p>
              
              <div style={{ marginTop: '12px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                Developer Note: Powered by FastAPI and React. Code ready for GitHub square contributions!
              </div>
            </div>
          ) : (
            messages.map((msg, index) => (
              <div className={`message ${msg.role}`} key={index}>
                <div className="avatar">
                  {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                </div>
                
                <div className="message-content-wrapper">
                  <div className="message-bubble">
                    {msg.content === '' && loading && index === messages.length - 1 ? (
                      <div className="typing-indicator">
                        <span className="typing-dot"></span>
                        <span className="typing-dot"></span>
                        <span className="typing-dot"></span>
                      </div>
                    ) : (
                      <div dangerouslySetInnerHTML={getMarkdownHtml(msg.content)} />
                    )}
                  </div>

                  {/* Render Sources if present */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div style={{ paddingLeft: '8px' }}>
                      <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600 }}>
                        Retrieved Contexts:
                      </span>
                      <div className="sources-indicator">
                        {msg.sources.map((src, sIdx) => (
                          <div className="source-tag" key={sIdx} title={src.text}>
                            <FileText size={10} />
                            <span>{src.source} ({Math.round(src.score * 100)}% match)</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input area */}
        <div className="chat-input-wrapper">
          {messages.length === 0 && (
            <div className="pills-container">
              <button 
                className="pill" 
                onClick={() => handlePillClick("What are Groq llama models and how are they so fast?")}
              >
                🚀 About Groq Models
              </button>
              <button 
                className="pill" 
                onClick={() => handlePillClick("How is Generative AI connected to Data Science?")}
              >
                📊 GenAI & Data Science
              </button>
              <button 
                className="pill" 
                onClick={() => handlePillClick("How does the RAG system index and retrieve search contexts?")}
              >
                🔍 Explain RAG here
              </button>
            </div>
          )}

          <form onSubmit={handleSendMessage} className="input-container">
            <textarea
              className="chat-input"
              placeholder={
                backendStatus === 'online' 
                  ? "Ask anything... (Use shift+enter for multi-lines)" 
                  : "Connecting to server backend..."
              }
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
              rows={1}
              disabled={backendStatus !== 'online'}
            />
            
            <div className="action-buttons">
              <button
                type="button"
                className={`icon-btn mic-btn ${isRecording ? 'active' : ''}`}
                onClick={startSpeechRecognition}
                disabled={backendStatus !== 'online'}
                title="Voice Dictation"
              >
                {isRecording ? <MicOff size={18} /> : <Mic size={18} />}
              </button>
              
              <button
                type="submit"
                className="icon-btn send-btn"
                disabled={!inputText.trim() || loading || backendStatus !== 'online'}
                title="Send query"
              >
                <Send size={16} />
              </button>
            </div>
          </form>
        </div>
      </main>
    </div>
  );
}

export default App;
