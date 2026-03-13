# ingest.py — instalar:
# pip install -r requirements.txt

import os, json
from pathlib import Path
from dotenv import load_dotenv
from paddleocr import PaddleOCR
from docling.document_converter import DocumentConverter
import psycopg2
import ollama

# ── Load environment variables ──────────────────────────
load_dotenv()

# ── config ─────────────────────────────────────────────
DB_URL  = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/docs_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHUNK_SIZE  = int(os.getenv("CHUNK_SIZE", "800"))
MIN_TEXT_LENGTH = int(os.getenv("MIN_TEXT_LENGTH", "40"))

# ── 1. OCR con GPU ─────────────────────────────────────
def ocr_pdf(pdf_path: str) -> str:
    ocr = PaddleOCR(use_angle_cls=True, lang="es", use_gpu=True)
    result = ocr.ocr(pdf_path, cls=True)
    lines = []
    for page in result:
        if page:
            for line in page:
                lines.append(line[1][0])   # solo el texto
    return "\n".join(lines)

# ── 2. Estructura con Docling ───────────────────────────
def structure_text(pdf_path: str) -> list[dict]:
    converter = DocumentConverter()
    doc = converter.convert(pdf_path)
    chunks = []
    for element, _level in doc.document.iterate_items():
        text = getattr(element, "text", "")
        if text and len(text.strip()) > MIN_TEXT_LENGTH:
            chunks.append({
                "text": text.strip(),
                "section": getattr(element, "label", "unknown"),
            })
    return chunks

# ── 3. Embeddings vía Ollama ────────────────────────────
def embed(text: str) -> list[float]:
    res = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return res["embedding"]

# ── 4. Guardar en pgvector ──────────────────────────────
def save_chunks(chunks: list[dict], doc_name: str):
    conn = psycopg2.connect(DB_URL)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id       SERIAL PRIMARY KEY,
            doc_name TEXT,
            section  TEXT,
            content  TEXT,
            embedding VECTOR(768)
        )
    """)
    for chunk in chunks:
        vec = embed(chunk["text"])
        cur.execute(
            "INSERT INTO documents (doc_name, section, content, embedding) "
            "VALUES (%s, %s, %s, %s)",
            (doc_name, chunk["section"], chunk["text"], vec)
        )
    conn.commit()
    cur.close()
    conn.close()
    print(f"✓ {len(chunks)} chunks guardados para '{doc_name}'")

# ── main ────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    pdf_path = sys.argv[1]          # python ingest.py mi_documento.pdf
    doc_name = Path(pdf_path).stem

    print("→ Extrayendo estructura con Docling...")
    chunks = structure_text(pdf_path)

    if not chunks:
        print("→ Docling no extrajo texto (PDF imagen pura), usando OCR...")
        raw = ocr_pdf(pdf_path)
        # chunking manual por párrafos
        paragraphs = [p.strip() for p in raw.split("\n\n") if len(p.strip()) > MIN_TEXT_LENGTH]
        chunks = [{"text": p, "section": "ocr"} for p in paragraphs]

    print(f"→ {len(chunks)} chunks. Guardando en pgvector...")
    save_chunks(chunks, doc_name)