# FastAPI Logic -

from fastapi import FastAPI, Request, HTTPException, Path, Depends, Query
from pydantic import BaseModel
from typing import List
from fastapi.responses import JSONResponse

import httpx
import os
import logging

import uuid
from rag_agent_phase2 import (
    precheck_node,
    query_extractor_node,
    find_node,
    generation_node,
)

from db_utils import (
    fetch_chat_history,
    create_new_conversation,
    update_chat_history,
    get_pg_conn,
)

from ip_search import (
    run_vector_search, 
    get_document_by_id
)

from langchain_core.documents import Document

def setup_logger(name="api_logger", log_file="logs/api_logs.log", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

         # Console logging (for Cloud Run)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger

# Create logger with a unique name
api_logger = setup_logger()

app = FastAPI()

# ==== Request models ====

class QueryRequest(BaseModel):
    message: str
    conversation_id: str = None

class GenerateRequest(BaseModel):
    conversation_id: str
    selected_doc_ids: List[int]
    intent: str
    custom_question: str = None
    external_context: str = None
    original_query: str

class SearchRequest(BaseModel):
    keyword: str
    rows: int = 10
    oppStatuses: str = "forecasted|posted"

class OpportunityDetailRequest(BaseModel):
    opportunityId: int

# --------- External API Endpoints ---------
GRANTS_SEARCH_URL = "https://api.training.grants.gov/v1/api/search2"
GRANTS_DETAIL_URL = "https://api.training.grants.gov/v1/api/fetchOpportunity"

@app.post("/grants/search")
async def search_grants(payload: SearchRequest):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(GRANTS_SEARCH_URL, json=payload.dict())
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/grants/opportunity")
async def fetch_grant_details(payload: OpportunityDetailRequest):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(GRANTS_DETAIL_URL, json=payload.dict())
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==== Query Endpoint ====

@app.post("/query")
def query_route(request: QueryRequest):
    cursor = None
    conn = None
    try:
        conn = get_pg_conn()
        cursor = conn.cursor()

        api_logger.info(f"Received query: {request.message}")
        if request.conversation_id:
            chat_history = fetch_chat_history(cursor, request.conversation_id)
            conversation_id = request.conversation_id
        else:
            conversation_id = create_new_conversation(cursor)
            chat_history = []

        precheck_result = precheck_node({
            "original_query": request.message,
            "chat_history": chat_history,
            "reset_context": True
        })

        if precheck_result.get("precheck_triggered"):
            update_chat_history(cursor, conversation_id, precheck_result.get("chat_history", []))
            conn.commit()
            return {
                "reply": precheck_result["final_answer"],
                "conversation_id": conversation_id,
                "retrieved_docs": [],
                "needs_user_selection": False
            }

        retrieval_state = {
            "original_query": request.message,
            "chat_history": precheck_result.get("chat_history", []),
            "reset_context": False,
            "needs_retrieval": True,
        }
        retrieval_state = query_extractor_node(retrieval_state)
        retrieval_state = find_node(retrieval_state, cursor)

        retrieved_docs = retrieval_state.get("retrieved_docs", [])
        doc_titles = [
            {"source_id": doc.metadata["source_id"], "title": doc.metadata.get("title", "Untitled")}
            for doc in retrieved_docs
        ]

        return {
            "conversation_id": conversation_id,
            "retrieved_docs": doc_titles,
            "needs_user_selection": True,
            "original_query": request.message,
            "rewritten_query": retrieval_state.get("search_query", ""),
        }
    
    except Exception as e:
        if conn:
            conn.rollback()
        api_logger.exception("Error in /query route", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")

    finally:
        if conn:
            conn.close()
        if cursor:
            cursor.close()


# # ==== Generate Endpoint ====
@app.post("/generate")
def generate_route(request: GenerateRequest):
    cursor = None
    conn = None
    try:
        conn = get_pg_conn()
        cursor = conn.cursor()

        api_logger.info(f"Generating response for conversation {request.conversation_id}")
        chat_history = fetch_chat_history(cursor, request.conversation_id)

        selected_docs = []
        for source_id in request.selected_doc_ids:
            cursor.execute("SELECT * FROM ip_new WHERE id = %s", (source_id,))
            row = cursor.fetchone()
            if row:
                doc = Document(
                    page_content=row[4],
                    metadata={"source_id": source_id, "title": row[2]}
                )
                selected_docs.append(doc)

        full_state = {
            "original_query": request.original_query,
            "chat_history": chat_history,
            "selected_docs": selected_docs,
            "intent": request.intent,
            "custom_question": request.custom_question,
            "external_context": request.external_context,
            "reset_context": False,
            "needs_retrieval": False,
        }

        final_state = generation_node(full_state)

        update_chat_history(cursor, request.conversation_id, final_state.get("chat_history", []))
        conn.commit()

        return {
            "reply": final_state["final_answer"],
            "conversation_id": request.conversation_id
        }
    
    except Exception as e:
        if conn:
            conn.rollback()
        api_logger.exception("Error in /generate route", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# # ==== IP Search Endpoint ====
@app.get("/search")
def search_endpoint(q: str, page: int = 1):
    try:
        results = run_vector_search(q, page)
        return {"page": page, "results": results}
    except Exception as e:
        api_logger.exception("Error in /search route", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


# # ==== Document Retrieval Endpoint ====
@app.get("/ip/{doc_id}")
def get_full_doc(doc_id: str):
    try:
        doc = get_document_by_id(doc_id)
        if not doc:
            api_logger.error(f"Document with ID {doc_id} not found")
            raise HTTPException(status_code=404, detail="Document not found")
        return doc
    except Exception as e:
        api_logger.error("Error in /ip/{doc_id} route, error: %s", e)
        # api_logger.exception(f"Failed to retrieve document with ID {doc_id}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# # ==== Health Check Endpoint ====
@app.get("/health")
def health_check():
    return {"status": "ok"}


# ==== Global Error Handler ====
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    api_logger.error(f"Unhandled error at {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"}
    )