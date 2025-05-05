# search_utils.py

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pinecone import Pinecone
import psycopg2

from db_utils import get_pg_conn
import logging

# Embedder setup (exactly like RAG)
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"}
)

# Pinecone v3
pc = Pinecone(api_key="pcsk_5cxnbw_Km7NBkBocGHmR13BpxbiUkzLQ2sVmytwg5H1zZh3ekvs29WaREVobLmAe3KcdKM")
index = pc.Index("ip-link-clean-links")

def run_vector_search(query_text: str, page: int = 1, page_size: int = 10) -> list[dict]:
    cursor = None
    conn = None
    try:
        conn = get_pg_conn()
        cursor = conn.cursor()

        vector = embedder.embed_query(query_text)

        top_k = page * page_size
        results = index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            namespace=""
        )

        # paginate locally (Pinecone doesn’t support offset)
        matches = results["matches"][(page - 1) * page_size : page * page_size]

        docs = []
        for match in matches:
            source_id = match["metadata"].get("source_id")
            cursor.execute("SELECT * FROM ip_new WHERE id = %s", (source_id,))
            row = cursor.fetchone()

            if row:
                doc = {
                    "id": source_id,
                    # "score": match["score"],
                    "title": row[2],
                    # "metadata": match["metadata"]
                    # don't return content unless requested separately
                }
                docs.append(doc)

        return docs
    except Exception as e:
        if conn:
            conn.rollback()
        # print("DB ERROR in get_document_by_id:", e)
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_document_by_id(doc_id: str) -> dict:
    cursor = None
    conn = None
    try:
        conn = get_pg_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM ip_new WHERE id = %s", (doc_id,))
        row = cursor.fetchone()

        if not row:
            return None
        
        university_name = None
        university_id = row[1]
        if university_id:
            cursor.execute("SELECT name FROM universities_new WHERE id = %s", (university_id,))
            name_row = cursor.fetchone()
            if name_row:
                university_name = name_row[0]

        return {
            # "id": doc_id,
            "university" : university_name,
            "title": row[2],
            "url": row[3],
            "content": row[4],
            "contributors": row[5],
        }
    except Exception as e:
        if conn:
            conn.rollback()
        # print("DB ERROR in get_document_by_id:", e)
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()