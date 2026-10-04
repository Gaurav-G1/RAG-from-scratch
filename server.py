#!/usr/bin/env python3
"""
server.py — Browser Web UI & API Server for RAG from Scratch
============================================================
Serves a modern Tailwind CSS interactive web interface where you can
ask questions against your indexed documents directly in the browser.

USAGE:
    python server.py
    python server.py --port 8080 --model gemini-3.8-flash
    python main.py --web
"""

import os
import sys
import json
import time
import socket
import argparse
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn

# Fix console encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv()

# Global state
pipeline_state = {
    "embedding_model": None,
    "vector_store": None,
    "chain_cache": {},
    "total_vectors": 0,
    "indexed_docs": [],
    "index_path": "faiss_index",
    "data_dir": "data/sample_docs",
    "default_model": "gemini-3.8-flash",
    "status": "initializing",  # "initializing", "ready", "error"
    "status_message": "Starting engine...",
    "error": None,
}


def background_init(index_path="faiss_index", data_dir="data/sample_docs", model_name="gemini-3.8-flash"):
    """Initialize the RAG pipeline in background thread."""
    try:
        pipeline_state["status_message"] = "Loading embedding model (all-MiniLM-L6-v2)..."
        print(f"🔢 {pipeline_state['status_message']}", flush=True)

        from src.embedder import get_embedding_model
        from src.vector_store import load_vector_store, create_vector_store, save_vector_store
        from src.document_loader import load_documents
        from src.chunker import chunk_documents

        pipeline_state["embedding_model"] = get_embedding_model("all-MiniLM-L6-v2")

        # Discover document names in data directory
        if os.path.exists(data_dir):
            pipeline_state["indexed_docs"] = [
                f for f in os.listdir(data_dir)
                if not f.startswith(".") and os.path.isfile(os.path.join(data_dir, f))
            ]

        # Check if FAISS index exists
        index_file = os.path.join(index_path, "index.faiss")
        if os.path.exists(index_file):
            pipeline_state["status_message"] = f"Loading FAISS index from '{index_path}'..."
            print(f"📂 {pipeline_state['status_message']}", flush=True)
            pipeline_state["vector_store"] = load_vector_store(index_path, pipeline_state["embedding_model"])
        else:
            pipeline_state["status_message"] = f"Building FAISS index from '{data_dir}'..."
            print(f"🆕 {pipeline_state['status_message']}", flush=True)
            docs = load_documents(data_dir)
            if not docs:
                raise ValueError(f"No documents found in '{data_dir}' to build vector index.")
            chunks = chunk_documents(docs)
            pipeline_state["vector_store"] = create_vector_store(chunks, pipeline_state["embedding_model"])
            save_vector_store(pipeline_state["vector_store"], index_path)

        pipeline_state["total_vectors"] = pipeline_state["vector_store"].index.ntotal
        pipeline_state["status"] = "ready"
        pipeline_state["status_message"] = f"Ready ({pipeline_state['total_vectors']} chunks indexed)"
        print(f"✅ RAG Engine Ready! Total vectors: {pipeline_state['total_vectors']}", flush=True)

    except Exception as e:
        pipeline_state["status"] = "error"
        pipeline_state["error"] = str(e)
        pipeline_state["status_message"] = f"Failed to initialize: {e}"
        print(f"❌ Initialization error: {e}", flush=True)


def get_cached_chain(model_name: str, k: int = 3):
    """Retrieve or build a QA chain cached by model and k."""
    cache_key = f"{model_name}:{k}"
    if cache_key in pipeline_state["chain_cache"]:
        return pipeline_state["chain_cache"][cache_key]

    from src.retriever import get_retriever
    from src.generator import build_qa_chain

    retriever = get_retriever(pipeline_state["vector_store"], k=k)
    chain = build_qa_chain(retriever, model_name=model_name, debug=False)
    pipeline_state["chain_cache"][cache_key] = chain
    return chain


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    daemon_threads = True


class RAGRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler serving UI and REST API."""

    def do_GET(self):
        """Handle GET requests."""
        if self.path in ("/", "/index.html"):
            self.serve_file("static/index.html", "text/html; charset=utf-8")
            return

        if self.path == "/api/status":
            self.send_json({
                "ready": pipeline_state["status"] == "ready",
                "status": pipeline_state["status"],
                "status_message": pipeline_state["status_message"],
                "total_vectors": pipeline_state["total_vectors"],
                "documents": pipeline_state["indexed_docs"],
                "default_model": pipeline_state["default_model"],
                "has_gemini": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
                "has_openai": bool(os.getenv("OPENAI_API_KEY")),
                "error": pipeline_state["error"],
            })
            return

        if self.path.startswith("/static/"):
            file_path = self.path.lstrip("/")
            if os.path.exists(file_path):
                content_type = "text/plain"
                if file_path.endswith(".html"):
                    content_type = "text/html; charset=utf-8"
                elif file_path.endswith(".css"):
                    content_type = "text/css; charset=utf-8"
                elif file_path.endswith(".js"):
                    content_type = "application/javascript; charset=utf-8"
                elif file_path.endswith(".json"):
                    content_type = "application/json"
                self.serve_file(file_path, content_type)
                return

        self.send_error(404, "File not found")

    def do_POST(self):
        """Handle POST requests."""
        if self.path == "/api/query":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")

            try:
                data = json.loads(body) if body else {}
            except json.JSONDecodeError:
                self.send_json({"error": "Invalid JSON body"}, status=400)
                return

            question = data.get("question", "").strip()
            if not question:
                self.send_json({"error": "Question cannot be empty"}, status=400)
                return

            if pipeline_state["status"] != "ready":
                self.send_json({
                    "error": f"RAG pipeline is still warming up: {pipeline_state['status_message']}. Please try again in a few seconds."
                }, status=503)
                return

            model_name = data.get("model", pipeline_state["default_model"])
            try:
                k = int(data.get("k", 3))
                k = max(1, min(k, 10))
            except (ValueError, TypeError):
                k = 3

            start_time = time.time()
            try:
                chain = get_cached_chain(model_name=model_name, k=k)
                result = chain.invoke({"input": question})

                answer = result.get("answer", "")
                raw_sources = result.get("context", [])

                sources = []
                seen = set()
                for doc in raw_sources:
                    src_name = doc.metadata.get("source", "Unknown")
                    src_clean = os.path.basename(src_name)
                    page = doc.metadata.get("page", None)
                    key = f"{src_clean}_{page}"
                    if key not in seen:
                        seen.add(key)
                        sources.append({
                            "source": src_clean,
                            "page": page + 1 if isinstance(page, int) else page,
                            "snippet": doc.page_content[:300].strip(),
                            "content": doc.page_content.strip(),
                        })

                elapsed_ms = int((time.time() - start_time) * 1000)

                self.send_json({
                    "question": question,
                    "answer": answer,
                    "sources": sources,
                    "latency_ms": elapsed_ms,
                    "model": model_name,
                    "k": k,
                })

            except Exception as e:
                elapsed_ms = int((time.time() - start_time) * 1000)
                print(f"Error handling query '{question}': {e}", flush=True)
                self.send_json({
                    "error": str(e),
                    "latency_ms": elapsed_ms,
                }, status=500)
            return

        self.send_error(404, "API endpoint not found")

    def send_json(self, payload, status=200):
        """Helper to send JSON response."""
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(encoded)

    def serve_file(self, path, content_type):
        """Helper to send a static file."""
        try:
            with open(path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

    def log_message(self, format, *args):
        """Suppress standard request logging unless 404 or 500."""
        if args and str(args[1]) in ("404", "500"):
            super().log_message(format, *args)


def find_available_port(start_port=8000, max_attempts=20):
    """Find an available TCP port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    return start_port


def start_server(port=8000, open_browser=True, index_path="faiss_index", data_dir="data/sample_docs", model="gemini-3.8-flash"):
    """Start the HTTP server and open the browser."""
    actual_port = find_available_port(port)
    server_address = ("0.0.0.0", actual_port)

    pipeline_state["index_path"] = index_path
    pipeline_state["data_dir"] = data_dir
    pipeline_state["default_model"] = model

    has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))

    if has_gemini and not model.startswith("gemini"):
        pipeline_state["default_model"] = "gemini-3.8-flash"
    elif has_openai and not has_gemini and not model.startswith("gpt"):
        pipeline_state["default_model"] = "gpt-3.5-turbo"

    # Start background initialization thread
    init_thread = threading.Thread(
        target=background_init,
        kwargs={"index_path": index_path, "data_dir": data_dir, "model_name": pipeline_state["default_model"]},
        daemon=True,
    )
    init_thread.start()

    httpd = ThreadedHTTPServer(server_address, RAGRequestHandler)
    local_url = f"http://localhost:{actual_port}"

    print("=" * 60, flush=True)
    print(f"  🌐 RAG Web UI is running at: {local_url}", flush=True)
    print("=" * 60, flush=True)
    print("  • Open the URL in your browser to ask questions.", flush=True)
    print("  • Press Ctrl+C in terminal to stop the server.\n", flush=True)

    if open_browser:
        def open_browser_delayed():
            time.sleep(1.0)
            try:
                webbrowser.open(local_url)
            except Exception:
                pass
        threading.Thread(target=open_browser_delayed, daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped gracefully. Goodbye! 👋", flush=True)
        httpd.server_close()


def main():
    parser = argparse.ArgumentParser(description="Run the RAG Web UI Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on (default: 8000)")
    parser.add_argument("--model", default="gemini-3.8-flash", help="Default LLM model name")
    parser.add_argument("--data-dir", default="data/sample_docs", help="Directory of source documents")
    parser.add_argument("--index-path", default="faiss_index", help="Directory of FAISS vector store")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")

    args = parser.parse_args()
    start_server(
        port=args.port,
        open_browser=not args.no_browser,
        index_path=args.index_path,
        data_dir=args.data_dir,
        model=args.model,
    )


if __name__ == "__main__":
    main()
