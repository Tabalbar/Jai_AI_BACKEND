from dotenv import load_dotenv

import os
load_dotenv()

import json

from typing import TypedDict, Optional, List, Literal
# from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain_core.documents import Document
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
import psycopg2
from pinecone import Pinecone

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate

# Setup OpenAI, PG, Pinecone, Embedder, and cursor

from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_openai import ChatOpenAI  # ✅ Supports with_structured_output


llm = ChatOpenAI(
    model="gpt-4o",  # or "gpt-4o-mini"
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Embedder
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# # PG Database client
# import psycopg2

# pg_conn = psycopg2.connect(
#     dbname="iplink4",
#     user="postgres",
#     password="CrappassFunTime25!",
#     host="iplink-4.c0nm66oygoar.us-east-1.rds.amazonaws.com",
#     port="5432"
# )

# cursor = pg_conn.cursor()


# Logging
import logging

def setup_logger(name="app_logger1", log_file="logs/history_test_logs.log", level=logging.INFO):
    """Manually set up a logger for Streamlit apps."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger

# Actually create the logger
app_logger1 = setup_logger()

api_logger = logging.getLogger("api_logger") # Main API logger imported from main.py

# ----------------------
# 1. Define RAG State
# ----------------------
class RAGState(TypedDict):
    original_query: str
    # rewritten_query: Optional[str]
    final_answer: Optional[str]
    retrieved_docs: Optional[List[Document]]
    # retry_attempts: int
    # reflection_feedback: Optional[dict]
    # doc_check_feedback: Optional[dict]
    intent: Optional[Literal["summarize", "explain", "compare", "other"]] # Add more intents like analyze, novelty_check, etc.
    chat_history: List[dict]
    precheck_triggered: Optional[bool]
    # flow_attempts: int
    # rewrite_origin: Optional[Literal["doc_check", "reflection"]]
    # query_type: Optional[Literal["new_query", "follow_up"]] # new_query or follow_up
    search_query: Optional[str] # Exracted query to search in the vector store
    external_context: Optional[str]  # user input or uploaded content for comparison
    custom_question: Optional[str] # When user selects "other" for intent
    selected_docs: Optional[List[Document]] # Selected documents by the user from the retrieved list
    needs_retrieval: Optional[bool]  # NEW
    reset_context: Optional[bool]   # NEW


""" PRE-CHECK NODE"""
# ----------------------

from pydantic import BaseModel
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import ChatPromptTemplate

class PrecheckClassification(BaseModel):
    classification: Literal["chitchat", "query"]


def precheck_node(state: RAGState) -> RAGState:
    try:
        query = state["original_query"].strip()
    except Exception:
        api_logger.exception("Missing or invalid 'original_query' in precheck_node")
        raise ValueError("original_query missing or not a string")

    def apply_context_reset(state: RAGState) -> None:
        state["selected_docs"] = []
        state["external_context"] = None
        state["needs_retrieval"] = True
        state.setdefault("chat_history", []).append({
            "role": "system",
            "content": "--- New query started. You may ignore prior context if available. ---"
        })

    if state.get("reset_context"):
        apply_context_reset(state)

    try:
        precheck_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant classifying user input before any processing."),
        ("human", f"""
        Classify the following user input into exactly one of two categories:

        - **query**: The input refers to a specific topic, term, technology, or area of interest that could reasonably be used to perform a search — even if it is short or not a full question. Examples include keywords like "brain tumors", "3D printing", or "energy storage".

        - **chitchat**: The input is vague, overly broad, conversational, exploratory, or general in nature. This includes greetings, small talk, or open-ended language like "I'm just browsing" or "tell me what you can do".

        Input:
        "{query}"

        Respond with exactly one word: query or chitchat.
        No explanations. No extra words.
        """)
        ])

        structured_precheck_llm = llm.with_structured_output(PrecheckClassification)

        chain = precheck_prompt | structured_precheck_llm

        result = chain.invoke({"query": query})

        classification = result.classification.strip().lower()
    
    except Exception:
        api_logger.exception("LLM classification failed in precheck_node")
        raise ValueError("Precheck classification failed")


    if classification == "chitchat":
        try:
            friendly_prompt = f"""

            This message is casual:
            "{query}"

            You are an AI Assistant for IP-LINK — a platform that helps users explore and understand university intellectual property (IP), such as patents and licensable technologies. You have access to IP data across almost all major US Universities.

            Reply naturally and conversationally. If the user is asking who you are or what you can do, introduce yourself as the IP-LINK assistant. 
            """

            reply = llm.invoke(friendly_prompt)

            state.setdefault("chat_history", []).append({
                "role": "user",
                "content": query
            })

            state["chat_history"].append({
                "role": "assistant",
                "content": reply.content
            })
            return {**state, "final_answer": reply.content, "precheck_triggered": True}
        except Exception:
            api_logger.exception("LLM failed to generate friendly response in precheck_node")
            raise ValueError("Chitchat fallback LLM generation failed")

    return {**state, "precheck_triggered": False}


""" QUERY EXTRACTOR NODE"""
# ----------------------

class QueryExtractionOutput(BaseModel):
    rewritten_query: str

def query_extractor_node(state: RAGState) -> RAGState:
    try:
        original_query = state["original_query"].strip()
        if not original_query:
            raise ValueError("Missing original_query in state.")
    except Exception as e:
        api_logger.exception("Error accessing original_query in query_extractor_node")
        raise ValueError("Failed to extract original query.")
    
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
            "You are an assistant that extracts a clean, focused query suitable for document retrieval in a RAG (Retrieval-Augmented Generation) pipeline. "
            "Your goal is to remove irrelevant phrasing, greetings, or conversational filler, and produce a concise query that can be used for semantic search."),

            ("human", f'''
            User Input:
            "{original_query}"

            Extracted Search Query:
            ''')
        ])

        structured_llm = llm.with_structured_output(QueryExtractionOutput)
        chain = prompt | structured_llm

        result = chain.invoke({"original_query": original_query})
        state["search_query"] = result.rewritten_query.strip()
        return state
    except Exception as e:
        api_logger.exception("LLM query extraction failed in query_extractor_node")
        raise ValueError("Query extraction via LLM failed.")



""" FIND (RETRIEVAL) NODE"""
# ----------------------

def find_node(state: RAGState, cursor) -> RAGState: #Retriever
    query = state.get("search_query") or state["original_query"]
    docs = []

    try:
        vector = embedder.embed_query(query)
    except Exception as e:
        api_logger.exception("Error embedding query")
        raise ValueError("Query Embedding Failed!") 
    
    try:
        results = index.query(
            vector=vector, 
            top_k=10, 
            include_metadata=True, 
            namespace="")
    except Exception as e:
        api_logger.exception("Pinecone query failed in find_node")
        raise ValueError("Search failed.")    

    for match in results["matches"]:
        source_id = match["metadata"].get("source_id")
        try:
            cursor.execute("SELECT * FROM ip_new WHERE id = %s", (source_id,))
            row = cursor.fetchone()

            if row:
                doc = Document(
                    page_content=row[4],        # contents
                    metadata={
                        "source_id": source_id,
                        "title": row[2],        # assuming row[2] is title
                        **match["metadata"]
                    }
                )
                docs.append(doc)
        except Exception as e:
            api_logger.warning(f"Failed to fetch or build doc for source_id {source_id}: {e}")
            continue  # Skip broken doc, don’t crash whole thing

    # return {**state, "retrieved_docs": docs}
    state["retrieved_docs"] = docs
    return state


""" EXPLORE RESULTS NODE"""
# ----------------------

def explore_results_node(state: RAGState) -> RAGState:
    if not state.get("selected_docs"):
        # TEMPORARILY: select first 2 retrieved docs
        retrieved_docs = state.get("retrieved_docs", [])
        state["selected_docs"] = retrieved_docs[:2]

    if not state.get("intent"):
        # TEMPORARILY: auto-assign an intent
        state["intent"] = "compare"  # or "explain", "compare"

    # Optional for "compare" intent if only 1 doc selected
    if state["intent"] == "compare" and len(state["selected_docs"]) == 1:
        state["external_context"] = "User provided description of wearable biosensor."

    if state['intent'] == 'other':
        state['custom_question'] = "Which one of these is more cost-effective and why?"

    return state


""" GENERATION NODE"""
# ----------------------


def format_chat_history(chat_history, anchor_turns=2, recency_turns=4):
    """Formats chat history keeping ALL system messages + first N conversations + last M conversations without content-based elimination, only index-based."""
    
    if not chat_history:
        return ""

    # === Step 1: Keep ALL system messages + separate out user/assistant ===
    system_messages = [m for m in chat_history if m['role'].lower() == "system"]
    non_system_messages = [m for m in chat_history if m['role'].lower() != "system"]

    # === Step 2: Form conversations (User -> Assistant pairs) ===
    conversations = []
    temp = []

    for message in non_system_messages:
        temp.append(message)
        if message['role'].lower() == "assistant":
            conversations.append(temp)
            temp = []

    # === Step 3: Select anchors and recents by position (NO duplicate indexes)
    anchors = conversations[:anchor_turns]
    recents = conversations[anchor_turns:] if anchor_turns < len(conversations) else []

    # Get only the *last* N recents
    recents = recents[-recency_turns:] if recents else []

    # === Step 4: Merge together
    combined = system_messages + [m for conv in anchors for m in conv] + [m for conv in recents for m in conv]

    # === Step 5: Pretty print formatting ===
    formatted = []
    formatted.append("===== Cropped Chat History =====")

    turn_counter = 1
    current_turn = []

    for message in combined:
        role = message['role'].capitalize()
        content = message['content']

        if role == "User":
            if current_turn:
                formatted.append("\n".join(current_turn))
                turn_counter += 1
                current_turn = []

            formatted.append(f"\n==== Turn {turn_counter} ====")
            current_turn.append(f"({role}) {content}")

        elif role == "Assistant":
            current_turn.append(f"({role}) {content}")

        elif role == "System":
            formatted.append(f"(System) {content}")

    if current_turn:
        formatted.append("\n".join(current_turn))

    return "\n".join(formatted)


def generation_node(state: RAGState) -> RAGState:
    try:    
        original_query = state["original_query"]
        selected_docs = state.get("selected_docs", [])
        intent = state.get("intent")
        custom_question = state.get("custom_question")
        external_context = state.get("external_context", "")
        chat_history = state.get("chat_history", [])
    except Exception:
        api_logger.exception("Failed to access required fields in generation_node")
        raise ValueError("Invalid or incomplete state passed to generation_node")

    # -----------------------------
    # Validation
    # -----------------------------
    if not selected_docs or len(selected_docs) == 0:
        raise ValueError("No documents selected.")
    if len(selected_docs) > 2:
        raise ValueError("Currently only 1 or 2 documents can be selected.")
    if not intent:
        raise ValueError("Intent must be provided.")
    if intent == "other" and not custom_question:
        raise ValueError("Custom question must be provided when intent is 'other'.")

    # Format documents
    docs_text = "\n\n".join([
        f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(selected_docs)
    ])

    # Format history
    history_text = format_chat_history(chat_history)

    # app_logger1.info("\n===== Cropped Chat History Before Model Call =====")
    # app_logger1.info(history_text)

    # -----------------------------
    # Intent-Based Generation Logic
    # -----------------------------
    try:
        if intent in {"summarize", "explain"}:
            outputs = []
            for i, doc in enumerate(selected_docs):
                if intent == "summarize":
                    task = "Summarize the following document clearly and concisely in 3–5 sentences."
                elif intent == "explain":
                    task = "Explain the following document in simple, accessible language for a general audience."

                prompt = f"""
                You are a professional AI assistant on IP-LINK, a platform with access to intellectual property (IP) data from top research universities across the USA. Your role is to help users by answering their questions and continuing multi-turn conversations about university technologies, patents, and innovations in a clear, helpful, and engaging manner. You will be provided with the conversation history so far, the user's new question, and the document(s) you need to answer the question with. 

                Conversation so far:
                {history_text}

                User's new question:
                {task}

                Document{i+1}:
                {doc.page_content}
                """
                response = llm.invoke(prompt)
                outputs.append(f"Document {i+1}:\n{response.content.strip()}")

            output = "\n\n".join(outputs)

        elif intent == "compare":
            if len(selected_docs) == 2:
                prompt = f"""
                You are a professional AI assistant on IP-LINK, a platform with access to intellectual property (IP) data from top research universities across the USA. Your role is to help users by answering their questions and continuing multi-turn conversations about university technologies, patents, and innovations in a clear, helpful, and engaging manner. 

                Conversation so far:
                {history_text}

                Compare the following two documents:

                Document 1:
                {selected_docs[0].page_content}

                Document 2:
                {selected_docs[1].page_content}

                Highlight their differences and similarities in:
                - Purpose
                - Technology
                - Application
                """
            elif (external_context or "").strip():
                prompt = f"""
                You are continuing a conversation about university IPs.

                Conversation so far:
                {history_text}

                Compare the selected document with the provided external input.

                Selected Document:
                {selected_docs[0].page_content}

                External Input:
                {external_context}

                Highlight key similarities and differences in purpose, technology, and potential use cases.
                """
            else:
                raise ValueError("Compare requires 2 documents or 1 document + external context.")

            response = llm.invoke(prompt)
            output = response.content.strip()

        elif intent == "other":
            prompt = f"""
            You are continuing a multi-turn conversation with a user about university IP.

            Conversation so far:
            {history_text}

            User's new question:
            {custom_question}

            Documents:
            {docs_text}

            Answer clearly and concisely using only the information in the provided documents.
            """
            response = llm.invoke(prompt)
            output = response.content.strip()

        else:
            raise ValueError(f"Unsupported intent: {intent}")
        
    except Exception:
        api_logger.exception(f"LLM generation failed for intent: {intent}")
        raise ValueError("LLM generation failed.")

    # -----------------------------
    # Update Chat History
    # -----------------------------
    user_message = f'{original_query} (Intent: {intent})'
    if intent == "other" and custom_question:
        user_message += f' (Custom Question: {custom_question})'

    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": output})

    state["chat_history"] = chat_history

    # -----------------------------
    # Reset Control Flags
    # -----------------------------
    state["reset_context"] = False
    state["needs_retrieval"] = False
    state["final_answer"] = output

    return state


""" GRAPH """

from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda

# Initialize the graph
graph = StateGraph(RAGState)

# Add all your nodes
graph.add_node("precheck", RunnableLambda(precheck_node))
graph.add_node("query_extractor", RunnableLambda(query_extractor_node))
graph.add_node("find", RunnableLambda(find_node))
graph.add_node("explore_results", RunnableLambda(explore_results_node))
graph.add_node("generate", RunnableLambda(generation_node))

# Set the entry point
graph.set_entry_point("precheck")

# Add conditional edge after precheck
graph.add_conditional_edges("precheck", lambda state:
    "__end__" if state.get("precheck_triggered") else
    "query_extractor" if state.get("needs_retrieval") else
    "generate"
)

# # Conditional routing after query type classification
# graph.add_conditional_edges("query_type_classifier", lambda state:
#     "generate" if state["query_type"] == "follow_up" else "query_extractor"
# )

# Linear flow for new_query
graph.add_edge("query_extractor", "find")
graph.add_edge("find", "explore_results")
graph.add_edge("explore_results", "generate")

# Final output node
graph.add_edge("generate", "__end__")

# Compile the app
app = graph.compile()