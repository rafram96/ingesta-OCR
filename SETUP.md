# ingesta-OCR Setup

## Instalación

1. **Copia el archivo de configuración:**
   ```bash
   cp .env.example .env
   ```

2. **Configura las variables en `.env`:**
   - `DATABASE_URL`: URL de conexión a PostgreSQL con pgvector
   - `EMBED_MODEL`: Modelo de Ollama para embeddings (ej: `nomic-embed-text`)
   - `CHUNK_SIZE`: Tamaño de fragmentos de texto (caracteres)
   - `MIN_TEXT_LENGTH`: Longitud mínima de texto para considerar válido

3. **Instala dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

## Uso

```bash
python ingest.py documento.pdf
```

El script:
1. Intenta extraer estructura con **Docling** (si es PDF moderno)
2. Si falla, usa **OCR con PaddleOCR** (para PDFs con imágenes)
3. Genera embeddings con **Ollama**
4. Guarda chunks en **PostgreSQL con pgvector**

## Requisitos previos

- **PostgreSQL** con extensión **pgvector** instalada
- **Ollama** ejecutándose en `localhost:11434` con el modelo configurado
- **GPU compatible** (para PaddleOCR y Ollama)
