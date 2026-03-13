# En Open WebUI → Admin → Functions → nueva Function (tipo "Filter" o "Pipe")

import psycopg2
import ollama
from pydantic import BaseModel

class Pipe:
    class Valves(BaseModel):
        DB_URL: str = "postgresql://user:pass@localhost:5432/docs_db"
        EMBED_MODEL: str = "nomic-embed-text"
        TOP_K: int = 6

    def __init__(self):
        self.valves = self.Valves()

    def _search(self, query: str) -> list[str]:
        """Busca los chunks más relevantes en pgvector."""
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
        cur.close(); conn.close()

        return [f"[{r[1]}] {r[0]}" for r in rows]

    def pipe(self, user_message: str, model_id: str, messages: list, body: dict):
        """Open WebUI llama esto antes de enviar al LLM."""
        chunks = self._search(user_message)
        context = "\n\n---\n\n".join(chunks)

        # Inyecta el contexto como system message
        system = (
            "Usa SOLO el siguiente contexto para responder. "
            "Si la respuesta no está en el contexto, dilo claramente.\n\n"
            f"CONTEXTO:\n{context}"
        )

        # Inserta/reemplaza el system prompt
        body["messages"] = [
            {"role": "system", "content": system},
            *[m for m in messages if m["role"] != "system"]
        ]
        return body