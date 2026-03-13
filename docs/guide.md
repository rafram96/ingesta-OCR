# Setup RAG con PDFs escaneados — desde cero con Open WebUI

> Punto de partida: solo tienes Open WebUI corriendo. Esta guía instala todo lo demás.

---

## Índice

- [Setup RAG con PDFs escaneados — desde cero con Open WebUI](#setup-rag-con-pdfs-escaneados--desde-cero-con-open-webui)
  - [Índice](#índice)
  - [1. Requisitos previos](#1-requisitos-previos)
  - [2. Levantar servicios con Docker](#2-levantar-servicios-con-docker)
  - [3. Configurar pgvector](#3-configurar-pgvector)
  - [4. Instalar Ollama y modelos](#4-instalar-ollama-y-modelos)
  - [5. Instalar dependencias Python](#5-instalar-dependencias-python)
  - [6. Probar el pipeline de ingesta](#6-probar-el-pipeline-de-ingesta)
  - [7. Instalar la Function en Open WebUI](#7-instalar-la-function-en-open-webui)
  - [8. Verificación end-to-end](#8-verificación-end-to-end)
  - [9. Comandos de mantenimiento](#9-comandos-de-mantenimiento)
  - [Resumen de puertos y credenciales](#resumen-de-puertos-y-credenciales)

---

## 1. Requisitos previos

```bash
# Verificar que tienes Docker y Docker Compose
docker --version          # >= 24.0
docker compose version    # >= 2.20

# Verificar GPU NVIDIA
nvidia-smi

# Verificar que el runtime NVIDIA está disponible para Docker
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi
```

Si el último comando falla, instalar el toolkit:

```bash
# NVIDIA Container Toolkit (Ubuntu/Debian)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## 2. Levantar servicios con Docker

Crear el archivo `docker-compose.yml` en tu directorio de trabajo:

```yaml
# docker-compose.yml
services:

  postgres:
    image: pgvector/pgvector:pg16
    container_name: rag_postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: raguser
      POSTGRES_PASSWORD: ragpass
      POSTGRES_DB: docs_db
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  ollama:
    image: ollama/ollama:latest
    container_name: rag_ollama
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  pgdata:
  ollama_data:
```

```bash
# Levantar ambos servicios
docker compose up -d

# Verificar que están corriendo
docker compose ps

# Ver logs si algo falla
docker compose logs postgres
docker compose logs ollama
```

---

## 3. Configurar pgvector

```bash
# Conectarse a PostgreSQL
docker exec -it rag_postgres psql -U raguser -d docs_db
```

Dentro del cliente `psql`, ejecutar:

```sql
-- Habilitar extensión vectorial
CREATE EXTENSION IF NOT EXISTS vector;

-- Tabla principal de documentos
CREATE TABLE IF NOT EXISTS documents (
    id        SERIAL PRIMARY KEY,
    doc_name  TEXT NOT NULL,
    section   TEXT,
    content   TEXT NOT NULL,
    embedding VECTOR(768),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Índice HNSW para búsqueda rápida (~50ms con miles de chunks)
CREATE INDEX ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Verificar
\d documents
```

```bash
# Salir del cliente psql
\q
```

---

## 4. Instalar Ollama y modelos

```bash
# Descargar el modelo de embeddings (768 dimensiones — debe coincidir con la tabla)
docker exec -it rag_ollama ollama pull nomic-embed-text

# Descargar el LLM principal (en español rinde mejor que Mistral)
docker exec -it rag_ollama ollama pull qwen2.5:7b

# Verificar modelos instalados
docker exec -it rag_ollama ollama list
```

Probar que los embeddings funcionan:

```bash
curl http://localhost:11434/api/embeddings \
  -d '{"model": "nomic-embed-text", "prompt": "prueba de embedding"}' | \
  python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Dimensiones: {len(d[\"embedding\"])}')"
# Esperado: Dimensiones: 768
```

---

## 5. Instalar dependencias Python

```bash
# Crear entorno virtual (recomendado)
python3 -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

# Dependencias base
pip install psycopg2-binary ollama

# OCR con GPU (PaddleOCR)
pip install paddlepaddle-gpu paddleocr

# Estructura de documentos
pip install docling

# Verificar instalación de PaddleOCR con GPU
python3 -c "from paddleocr import PaddleOCR; print('PaddleOCR OK')"
```

> **Nota:** Si `paddlepaddle-gpu` falla, verificar la versión de CUDA con `nvidia-smi` y buscar el wheel correspondiente en https://www.paddlepaddle.org.cn/install/quick

---

## 6. Probar el pipeline de ingesta

Guardar este archivo como `ingest.py`:

```python
# ingest.py
import sys
from pathlib import Path
from paddleocr import PaddleOCR
from docling.document_converter import DocumentConverter
import psycopg2
import ollama

DB_URL      = "postgresql://raguser:ragpass@localhost:5432/docs_db"
EMBED_MODEL = "nomic-embed-text"

def ocr_pdf(pdf_path: str) -> list[dict]:
    """Fallback OCR puro para PDFs imagen sin texto embebido."""
    ocr = PaddleOCR(use_angle_cls=True, lang="es", use_gpu=True)
    result = ocr.ocr(pdf_path, cls=True)
    chunks = []
    for page_idx, page in enumerate(result or []):
        if page:
            text = "\n".join(line[1][0] for line in page)
            if len(text.strip()) > 40:
                chunks.append({"text": text.strip(), "section": f"pagina_{page_idx+1}"})
    return chunks

def structure_pdf(pdf_path: str) -> list[dict]:
    """Extrae estructura (secciones, tablas) con Docling."""
    converter = DocumentConverter()
    doc = converter.convert(pdf_path)
    chunks = []
    for element, _level in doc.document.iterate_items():
        text = getattr(element, "text", "")
        if text and len(text.strip()) > 40:
            chunks.append({
                "text": text.strip(),
                "section": getattr(element, "label", "unknown"),
            })
    return chunks

def embed(text: str) -> list[float]:
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]

def save_chunks(chunks: list[dict], doc_name: str):
    conn = psycopg2.connect(DB_URL)
    cur  = conn.cursor()
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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python ingest.py <archivo.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    doc_name = Path(pdf_path).stem

    print("→ Intentando extracción estructural con Docling...")
    chunks = structure_pdf(pdf_path)

    if not chunks:
        print("→ Sin texto estructurado. Usando OCR con GPU...")
        chunks = ocr_pdf(pdf_path)

    if not chunks:
        print("✗ No se pudo extraer texto del PDF.")
        sys.exit(1)

    print(f"→ {len(chunks)} chunks extraídos. Generando embeddings...")
    save_chunks(chunks, doc_name)
```

```bash
# Procesar un PDF
python ingest.py mi_documento.pdf

# Verificar que se guardaron los chunks
docker exec -it rag_postgres psql -U raguser -d docs_db \
  -c "SELECT doc_name, section, LEFT(content, 60) FROM documents LIMIT 10;"
```

---

## 7. Instalar la Function en Open WebUI

1. Ir a **Open WebUI → Admin Panel → Functions → nueva Function**
2. Pegar el siguiente código:

```python
# RAG Function para Open WebUI
import psycopg2
import ollama
from pydantic import BaseModel

class Pipe:
    class Valves(BaseModel):
        DB_URL: str = "postgresql://raguser:ragpass@localhost:5432/docs_db"
        EMBED_MODEL: str = "nomic-embed-text"
        TOP_K: int = 6

    def __init__(self):
        self.valves = self.Valves()

    def _search(self, query: str) -> list[str]:
        vec = ollama.embeddings(
            model=self.valves.EMBED_MODEL, prompt=query
        )["embedding"]
        conn = psycopg2.connect(self.valves.DB_URL)
        cur  = conn.cursor()
        cur.execute("""
            SELECT content, section,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (vec, vec, self.valves.TOP_K))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [f"[{r[1]}] {r[0]}" for r in rows]

    def pipe(self, user_message: str, model_id: str, messages: list, body: dict):
        chunks = self._search(user_message)
        context = "\n\n---\n\n".join(chunks)
        system = (
            "Usa SOLO el siguiente contexto para responder. "
            "Si la respuesta no está en el contexto, dilo claramente.\n\n"
            f"CONTEXTO:\n{context}"
        )
        body["messages"] = [
            {"role": "system", "content": system},
            *[m for m in messages if m["role"] != "system"]
        ]
        return body
```

3. Guardar y activar la Function.
4. En la conversación, seleccionar el modelo `qwen2.5:7b` y activar la Function desde el menú de herramientas.

> **Importante:** Open WebUI debe poder alcanzar `localhost:5432` y `localhost:11434`. Si Open WebUI corre en Docker, reemplazar `localhost` por `host.docker.internal` en los `Valves`.

---

## 8. Verificación end-to-end

```bash
# 1. Servicios corriendo
docker compose ps
# Esperado: rag_postgres Up, rag_ollama Up

# 2. Modelos disponibles
docker exec -it rag_ollama ollama list
# Esperado: nomic-embed-text, qwen2.5:7b

# 3. Chunks en la base de datos
docker exec -it rag_postgres psql -U raguser -d docs_db \
  -c "SELECT COUNT(*), doc_name FROM documents GROUP BY doc_name;"

# 4. Prueba de búsqueda manual
python3 - <<'EOF'
import psycopg2, ollama

vec = ollama.embeddings(model="nomic-embed-text", prompt="experiencia profesional")["embedding"]
conn = psycopg2.connect("postgresql://raguser:ragpass@localhost:5432/docs_db")
cur = conn.cursor()
cur.execute("""
    SELECT LEFT(content, 120), 1 - (embedding <=> %s::vector) AS sim
    FROM documents ORDER BY embedding <=> %s::vector LIMIT 3
""", (vec, vec))
for row in cur.fetchall():
    print(f"[{row[1]:.3f}] {row[0]}")
EOF
```

---

## 9. Comandos de mantenimiento

```bash
# Ver logs en tiempo real
docker compose logs -f ollama
docker compose logs -f postgres

# Detener todo
docker compose down

# Detener sin borrar datos
docker compose stop

# Reiniciar un servicio
docker compose restart ollama

# Eliminar todos los chunks de un documento
docker exec -it rag_postgres psql -U raguser -d docs_db \
  -c "DELETE FROM documents WHERE doc_name = 'nombre_sin_extension';"

# Ver cuántos chunks tiene cada documento
docker exec -it rag_postgres psql -U raguser -d docs_db \
  -c "SELECT doc_name, COUNT(*) AS chunks FROM documents GROUP BY doc_name ORDER BY chunks DESC;"

# Procesar varios PDFs en lote
for f in pdfs/*.pdf; do
  python ingest.py "$f"
done

# Backup de la base de datos
docker exec rag_postgres pg_dump -U raguser docs_db > backup_$(date +%Y%m%d).sql

# Restaurar backup
cat backup_20250101.sql | docker exec -i rag_postgres psql -U raguser -d docs_db
```

---

## Resumen de puertos y credenciales

| Servicio | Puerto | Usuario | Contraseña | Base de datos |
|----------|--------|---------|------------|---------------|
| PostgreSQL | 5432 | raguser | ragpass | docs_db |
| Ollama API | 11434 | — | — | — |
| Open WebUI | 3000 | (el tuyo) | (el tuyo) | — |