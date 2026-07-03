import re
import math
from typing import List, Dict, Tuple
from pypdf import PdfReader

class RAGEngine:
    def __init__(self):
        # List of chunks, each chunk is a dict: {"text": str, "source": str, "id": int}
        self.chunks: List[Dict] = []
        # Inverted index for TF-IDF calculations
        self.vocab: Dict[str, int] = {}  # word -> index
        self.idf: Dict[str, float] = {}  # word -> idf value
        self.chunk_vectors: List[Dict[int, float]] = []  # List of sparse TF-IDF vectors (idx -> val)
        self.document_names: List[str] = []

    def clean_text(self, text: str) -> List[str]:
        """Tokenize text and remove punctuation, converting to lowercase."""
        words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        # Remove very common words (stop words) to improve search quality
        stop_words = {
            'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'arent',
            'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 
            'can', 'cannot', 'could', 'did', 'do', 'does', 'doing', 'dont', 'down', 'during', 'each', 'few', 
            'for', 'from', 'further', 'had', 'has', 'have', 'having', 'he', 'hed', 'hell', 'hes', 'her', 'here',
            'heres', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'hows', 'i', 'id', 'ill', 'im', 'ive',
            'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'lets', 'me', 'more', 'most', 'mustnt', 'my', 
            'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 
            'ours', 'ourselves', 'out', 'over', 'own', 'same', 'she', 'shed', 'shell', 'shes', 'should', 'so', 
            'some', 'such', 'than', 'that', 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 
            'there', 'theres', 'these', 'they', 'theyd', 'theyll', 'theyre', 'theyve', 'this', 'those', 'through', 
            'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'wed', 'well', 'were', 'what', 'whats', 
            'when', 'whereto', 'where', 'which', 'while', 'who', 'whom', 'why', 'with', 'would', 'you', 'your', 
            'yours', 'yourself', 'yourselves'
        }
        return [w for w in words if w not in stop_words and len(w) > 2]

    def add_document(self, filename: str, content: bytes):
        """Extract text, chunk it, and update the TF-IDF index."""
        text = ""
        if filename.endswith('.pdf'):
            try:
                # Use io.BytesIO to read PDF from memory bytes
                import io
                pdf_file = io.BytesIO(content)
                reader = PdfReader(pdf_file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            except Exception as e:
                print(f"Error reading PDF {filename}: {e}")
                return False
        else:
            # Assume text/plain
            try:
                text = content.decode('utf-8', errors='ignore')
            except Exception as e:
                print(f"Error reading text file {filename}: {e}")
                return False

        if not text.strip():
            return False

        # Add to document names list
        if filename not in self.document_names:
            self.document_names.append(filename)

        # Chunk the text: 500 characters, 100 overlap
        chunk_size = 500
        overlap = 100
        start = 0
        doc_chunks = []
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if len(chunk_text) > 20:  # Skip tiny chunks
                doc_chunks.append({
                    "text": chunk_text,
                    "source": filename,
                    "id": len(self.chunks) + len(doc_chunks)
                })
            start += (chunk_size - overlap)

        self.chunks.extend(doc_chunks)
        self._rebuild_tfidf_index()
        return True

    def _rebuild_tfidf_index(self):
        """Re-calculate TF-IDF vectors for all chunks in memory."""
        self.vocab = {}
        self.chunk_vectors = []
        
        if not self.chunks:
            return

        # 1. Build vocabulary and count doc frequency (DF)
        df: Dict[str, int] = {}  # word -> count of documents containing it
        chunk_tokenized: List[List[str]] = []

        for chunk in self.chunks:
            tokens = self.clean_text(chunk["text"])
            chunk_tokenized.append(tokens)
            
            # Record unique tokens in this chunk for DF
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] = df.get(token, 0) + 1
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

        # 2. Compute IDF for each word
        num_chunks = len(self.chunks)
        self.idf = {}
        for token, count in df.items():
            # Standard IDF formula with smoothing
            self.idf[token] = math.log((1 + num_chunks) / (1 + count)) + 1.0

        # 3. Build TF-IDF vectors for each chunk
        for tokens in chunk_tokenized:
            tf: Dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            
            vector: Dict[int, float] = {}
            # Compute TF-IDF weights and normalize
            length_sq = 0.0
            for token, count in tf.items():
                word_idx = self.vocab[token]
                # Log-frequency weighting for TF
                tf_val = 1 + math.log(count)
                tfidf_val = tf_val * self.idf[token]
                vector[word_idx] = tfidf_val
                length_sq += tfidf_val ** 2
            
            # L2 Normalization
            length = math.sqrt(length_sq)
            if length > 0:
                normalized_vector = {idx: val / length for idx, val in vector.items()}
            else:
                normalized_vector = {}
                
            self.chunk_vectors.append(normalized_vector)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """Query the index and return top_k most similar chunks with similarity score."""
        if not self.chunks or not self.vocab:
            return []

        query_tokens = self.clean_text(query)
        if not query_tokens:
            return []

        # Build query TF-IDF vector
        query_tf: Dict[str, int] = {}
        for token in query_tokens:
            if token in self.vocab:  # Only count words in our vocabulary
                query_tf[token] = query_tf.get(token, 0) + 1

        if not query_tf:
            return []

        query_vector: Dict[int, float] = {}
        length_sq = 0.0
        for token, count in query_tf.items():
            word_idx = self.vocab[token]
            tf_val = 1 + math.log(count)
            tfidf_val = tf_val * self.idf[token]
            query_vector[word_idx] = tfidf_val
            length_sq += tfidf_val ** 2

        length = math.sqrt(length_sq)
        if length > 0:
            query_vector = {idx: val / length for idx, val in query_vector.items()}
        else:
            return []

        # Calculate cosine similarity with all chunks
        results: List[Tuple[Dict, float]] = []
        for i, chunk_vector in enumerate(self.chunk_vectors):
            similarity = 0.0
            # Dot product of sparse vectors
            for idx, val in query_vector.items():
                if idx in chunk_vector:
                    similarity += val * chunk_vector[idx]
            
            if similarity > 0.0:
                results.append((self.chunks[i], similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def clear(self):
        """Reset the RAG system."""
        self.chunks = []
        self.vocab = {}
        self.idf = {}
        self.chunk_vectors = []
        self.document_names = []
