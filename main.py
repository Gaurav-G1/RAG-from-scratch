#!/usr/bin/env python3
# main.py
#
# RAG FROM SCRATCH — COMPLETE PIPELINE
# ======================================
# This file ties together all 6 steps of the RAG (Retrieval-Augmented Generation)
# pipeline into a single runnable script.
#
# THE 6 STEPS:
#   1. LOAD     → Read .pdf/.txt/.docx files from disk into LangChain Documents
#   2. CHUNK    → Split large documents into smaller overlapping chunks
#   3. EMBED    → Convert each chunk to a vector using a HuggingFace model
#   4. INDEX    → Store all vectors in a FAISS index (saved to disk for reuse)
#   5. RETRIEVE → Given a user question, find the top-k most relevant chunks
#   6. GENERATE → Pass the question + retrieved chunks to an LLM for a grounded answer
#
# USAGE:
#   # Single question mode:
#   python main.py --question "What are the main topics in these documents?"
#
#   # Interactive mode (loops until you type 'quit'):
#   python main.py
#
#   # Use a local Ollama model instead of OpenAI:
#   python main.py --model ollama/llama3
#
#   # Debug mode (shows full prompt sent to LLM and retrieved chunks):
#   python main.py --debug --question "What is the refund policy?"
#
#   # Specify a different data folder or index location:
#   python main.py --data-dir my_docs/ --index-path my_index/

import os
import sys
import argparse

# Ensure Unicode characters (box drawings, emojis) print reliably on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# python-dotenv loads KEY=VALUE pairs from your .env file into os.environ.
# This is the standard way to manage API keys without hardcoding them in source code.
from dotenv import load_dotenv

# Import each step of our pipeline from the src/ package
from src.document_loader import load_documents
from src.chunker import chunk_documents
from src.embedder import get_embedding_model, embed_text
from src.vector_store import get_or_create_vector_store
from src.retriever import get_retriever, retrieve_chunks
from src.generator import build_qa_chain


def parse_args():
    """
    Parse command-line arguments.

    argparse is Python's built-in library for CLI argument handling.
    It automatically generates --help text from the descriptions below.
    """
    parser = argparse.ArgumentParser(
        description="RAG from Scratch — Ask questions about your documents using AI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --question "What are the main topics?"
  python main.py --model ollama/llama3 --question "Summarize the documents"
  python main.py --debug --question "What is the refund policy?"
  python main.py --data-dir /path/to/docs --index-path /path/to/index
        """,
    )

    parser.add_argument(
        "--data-dir",
        default="data/sample_docs",
        help="Path to folder containing .pdf, .txt, or .docx files. "
             "Default: data/sample_docs",
    )

    parser.add_argument(
        "--index-path",
        default="faiss_index",
        help="Path to save/load the FAISS vector index. "
             "Default: faiss_index (created automatically on first run).",
    )

    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="LLM to use for answer generation. "
             "Options: gemini-1.5-flash, gemini-2.0-flash, gpt-3.5-turbo, gpt-4, ollama/llama3. "
             "Default: gpt-3.5-turbo (or gemini-1.5-flash if GEMINI_API_KEY is set)",
    )

    parser.add_argument(
        "--question",
        default=None,
        help="A single question to answer and exit. "
             "If omitted, starts an interactive Q&A loop.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: print the full prompt sent to LLM and "
             "detailed chain steps.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of chunks to retrieve per query (top-k). Default: 3.",
    )

    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch the interactive browser UI (HTML + Tailwind CSS).",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the browser web server (default: 8000).",
    )

    return parser.parse_args()


def run_pipeline(args):
    """
    Execute the full RAG pipeline end-to-end.

    This function orchestrates all 6 steps, printing clear separators between
    each phase so you can follow along and understand what's happening.
    """

    print("=" * 60)
    print("  RAG FROM SCRATCH — PIPELINE STARTING")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # LOAD ENVIRONMENT VARIABLES
    # -------------------------------------------------------------------------
    # .env is NOT committed to git (see .gitignore). Copy .env.example → .env
    # and fill in your OPENAI_API_KEY before running with an OpenAI model.
    load_dotenv()

    # Warn early if an API key is missing and not using a local model
    has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    if not args.model.startswith("ollama/") and not has_gemini and not has_openai:
        print(
            "\n⚠️  WARNING: No API key found in your environment.\n"
            "   Add GEMINI_API_KEY or OPENAI_API_KEY to your .env file,\n"
            "   or use a local model with --model ollama/llama3\n"
        )

    # -------------------------------------------------------------------------
    # LAUNCH WEB UI IF --web FLAG IS GIVEN
    # -------------------------------------------------------------------------
    if getattr(args, "web", False):
        from server import start_server
        start_server(
            port=args.port,
            index_path=args.index_path,
            data_dir=args.data_dir,
            model=args.model,
        )
        return

    # -------------------------------------------------------------------------
    # STEP 3 FIRST: LOAD EMBEDDING MODEL
    # -------------------------------------------------------------------------
    print("\n" + "─" * 60)
    print("STEP: Loading embedding model")
    print("─" * 60)
    print("  Model: all-MiniLM-L6-v2 (free, local, no API key needed)")

    embedding_model = get_embedding_model("all-MiniLM-L6-v2")

    # -------------------------------------------------------------------------
    # CHECK FOR EXISTING FAISS VECTOR STORE
    # -------------------------------------------------------------------------
    index_file = os.path.join(args.index_path, "index.faiss")
    if os.path.exists(index_file):
        print("\n" + "─" * 60)
        print("STEP: Loading existing FAISS vector store")
        print("─" * 60)
        print(f"  Index location: {args.index_path}/")
        print(f"  Tip: Delete '{args.index_path}/' to force a full rebuild.")
        from src.vector_store import load_vector_store
        vector_store = load_vector_store(args.index_path, embedding_model)
    else:
        # STEP 1: LOAD DOCUMENTS
        print("\n" + "─" * 60)
        print("STEP 1/6: Loading documents")
        print("─" * 60)
        print(f"  Source directory: {args.data_dir}")

        documents = load_documents(args.data_dir)

        if not documents:
            print(
                "\n❌ No documents loaded. Please add .pdf, .txt, or .docx files to:\n"
                f"   {args.data_dir}\n"
                "\nThen re-run: python main.py"
            )
            return

        # STEP 2: CHUNK DOCUMENTS
        print("\n" + "─" * 60)
        print("STEP 2/6: Chunking documents")
        print("─" * 60)

        chunks = chunk_documents(
            documents,
            chunk_size=500,
            chunk_overlap=50,
        )

        # BUILD AND SAVE VECTOR STORE
        print("\n" + "─" * 60)
        print("STEP 4/6: Building FAISS vector store")
        print("─" * 60)
        vector_store = get_or_create_vector_store(
            chunks=chunks,
            embedding_model=embedding_model,
            path=args.index_path,
        )

    # -------------------------------------------------------------------------
    # STEP 5: SET UP RETRIEVER
    # -------------------------------------------------------------------------
    print("\n" + "─" * 60)
    print("STEP 5/6: Configuring retriever")
    print("─" * 60)
    print(f"  Retrieval strategy: cosine similarity, top-k={args.k}")

    retriever = get_retriever(vector_store, k=args.k)

    # -------------------------------------------------------------------------
    # STEP 6: BUILD QA CHAIN (LLM + RETRIEVER)
    # -------------------------------------------------------------------------
    print("\n" + "─" * 60)
    print("STEP 6/6: Building QA chain (LLM + Retriever)")
    print("─" * 60)

    qa_chain = build_qa_chain(
        retriever=retriever,
        model_name=args.model,
        debug=args.debug,
    )

    print("\n" + "=" * 60)
    print("  PIPELINE READY — Let's ask some questions!")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Q&A PHASE: Single question or interactive loop
    # -------------------------------------------------------------------------

    if args.question:
        # Single question mode — answer it and exit
        ask_question(qa_chain, args.question, args.debug)
    else:
        # Interactive mode — keep asking until the user types 'quit' or 'exit'
        print("\n💬 Interactive Q&A Mode")
        print("   Type your question and press Enter.")
        print("   Type 'quit' or 'exit' to stop.\n")

        # Sample question to get the user started
        sample_question = "What are the main topics covered in these documents?"
        print(f"   💡 Sample question: {sample_question}\n")

        while True:
            try:
                question = input("Your question: ").strip()
            except (KeyboardInterrupt, EOFError):
                # Handle Ctrl+C gracefully
                print("\n\nGoodbye! 👋")
                break

            if not question:
                print("   (Please type a question, or 'quit' to exit)")
                continue

            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye! 👋")
                break

            ask_question(qa_chain, question, args.debug)


def ask_question(qa_chain, question: str, debug: bool = False):
    """
    Ask a single question and print the answer with source attribution.

    Args:
        qa_chain:      The assembled retrieval chain (from generator.build_qa_chain).
        question (str): The question to ask.
        debug (bool):   If True, print source document details.
    """

    print(f"\n❓ Question: {question}")
    print("   (Retrieving relevant chunks and generating answer...)\n")

    try:
        # .invoke() runs the full chain:
        #   question → embed → FAISS search → retrieve chunks → fill prompt → LLM → answer
        result = qa_chain.invoke({"input": question})

        # The result dict has:
        #   result["answer"]   → the LLM's answer string
        #   result["context"]  → list of Document objects used as context
        answer = result["answer"]
        source_docs = result.get("context", [])

        print(f"💡 Answer:\n{answer}")

        # Show which source documents contributed to this answer
        if source_docs:
            print("\n📚 Sources used:")
            seen_sources = set()
            for doc in source_docs:
                source = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "")
                page_info = f", page {page}" if page != "" else ""
                source_key = f"{source}{page_info}"

                # Deduplicate — a source file may appear multiple times (different chunks)
                if source_key not in seen_sources:
                    print(f"   • {source_key}")
                    seen_sources.add(source_key)

                    # In debug mode, show the actual chunk text used
                    if debug:
                        print(f"     Context: {doc.page_content[:150]}...")

    except Exception as e:
        print(f"\n❌ Error generating answer: {e}")
        print(
            "\nCommon causes:\n"
            "  • Missing OPENAI_API_KEY (check your .env file)\n"
            "  • Ollama not running (start with: ollama serve)\n"
            "  • Model not pulled (run: ollama pull llama3)\n"
            "  • Network connectivity issues\n"
        )

    print()  # blank line for readability between questions


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
