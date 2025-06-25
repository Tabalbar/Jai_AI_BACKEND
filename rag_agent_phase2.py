"""
RAG Agent Phase 2 - Multi-Node Retrieval-Augmented Generation System

This module implements a RAG (Retrieval-Augmented Generation) system using LangGraph
that processes user queries through multiple specialized nodes. The system is designed for IP-LINK,
a platform that helps users explore university intellectual property (IP) data.

System Architecture:
1. Precheck: Classifies input as casual conversation vs. searchable query
2. Query Extraction: Converts natural language to clean search terms
3. Document Retrieval: Searches vector database for relevant IP documents
4. Result Exploration: Handles document selection and intent classification
5. Generation: Produces final answers based on user intent (summarize, explain, compare, other)

Key Features:
- Multi-turn conversation support with chat history management
- Intent-based response generation (summarize, explain, compare, custom)
- Document comparison capabilities (2 docs or 1 doc + external context)
- Context reset functionality for new query sessions
- Robust error handling and logging throughout the pipeline
"""

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

# ----------------------
# SYSTEM CONFIGURATION
# ----------------------

# Setup OpenAI, PG, Pinecone, Embedder, and cursor

from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_openai import ChatOpenAI  # ✅ Supports with_structured_output

# Initialize OpenAI LLM with structured output support
# This LLM is used throughout the pipeline for various classification and generation tasks
llm = ChatOpenAI(
    model="gpt-4o",  # or "gpt-4o-mini"
    temperature=0,   # Deterministic responses for consistency
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# Initialize Pinecone vector database client
# Used for semantic search of university IP documents
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Initialize sentence transformer embedder
# Converts text queries and documents into vector representations for similarity search
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# # PostgreSQL Database Configuration (Currently commented out)
# # Used to fetch full document content based on source IDs from Pinecone matches
# # Contains the actual IP document data (title, content, metadata)
# import psycopg2
# 
# pg_conn = psycopg2.connect(
#     dbname="iplink4",
#     user="postgres", 
#     password="CrappassFunTime25!",
#     host="iplink-4.c0nm66oygoar.us-east-1.rds.amazonaws.com",
#     port="5432"
# )
# 
# cursor = pg_conn.cursor()

# ----------------------
# LOGGING CONFIGURATION
# ----------------------

# Logging
import logging

def setup_logger(name="app_logger1", log_file="logs/history_test_logs.log", level=logging.INFO):
    """
    Manually set up a logger for Streamlit apps.
    
    Creates a file-based logger that tracks the RAG system's execution flow,
    including successful operations, errors, and debugging information.
    
    Args:
        name: Logger name identifier
        log_file: Path to log file (creates directory if needed)
        level: Logging level (INFO, DEBUG, ERROR, etc.)
    
    Returns:
        Configured logger instance
    """
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

# Create application logger for tracking RAG system operations
app_logger1 = setup_logger()

# Main API logger imported from main.py for cross-module logging consistency
api_logger = logging.getLogger("api_logger")

# ----------------------
# STATE DEFINITION
# ----------------------

class RAGState(TypedDict):
    """
    Central state object that flows through all nodes in the RAG pipeline.
    
    This TypedDict maintains all necessary information as the query progresses
    through different processing stages, enabling stateful multi-turn conversations
    and complex document interactions.
    
    Attributes:
        original_query: User's raw input query
        final_answer: Generated response to return to user
        retrieved_docs: Documents found via vector search
        intent: User's intention (summarize, explain, compare, other)
        chat_history: List of previous conversation turns
        precheck_triggered: Whether query was classified as chitchat
        search_query: Cleaned query optimized for vector search
        external_context: User-provided content for comparison
        custom_question: User's custom question when intent is "other"
        selected_docs: Documents chosen for final generation
        needs_retrieval: Flag indicating if vector search is needed
        reset_context: Flag to clear previous conversation context
    """
    original_query: str                                    # User's raw input
    final_answer: Optional[str]                           # Final generated response
    retrieved_docs: Optional[List[Document]]              # Documents from vector search
    intent: Optional[Literal["summarize", "explain", "compare", "other"]]  # User's intention
    chat_history: List[dict]                              # Conversation history
    precheck_triggered: Optional[bool]                    # Chitchat classification result
    search_query: Optional[str]                           # Extracted search query
    external_context: Optional[str]                       # User input for comparison
    custom_question: Optional[str]                        # Custom question for "other" intent
    selected_docs: Optional[List[Document]]               # User-selected documents
    needs_retrieval: Optional[bool]                       # Whether to perform vector search
    reset_context: Optional[bool]                         # Whether to reset conversation


# ----------------------
# NODE 1: PRECHECK
# ----------------------
"""
PRECHECK NODE - Input Classification and Chitchat Handling

Purpose:
- Classifies incoming user input as either "chitchat" or "query"
- Handles casual conversation without triggering the full RAG pipeline
- Manages context reset functionality for new query sessions
- Acts as the entry point and traffic controller for the system

Flow:
1. Check if context reset is requested and apply if needed
2. Use LLM to classify input as chitchat vs searchable query
3. If chitchat: Generate friendly response and end pipeline
4. If query: Continue to next node for processing

This optimization prevents unnecessary vector searches for casual conversation
and provides appropriate responses for non-search interactions.
"""

from pydantic import BaseModel
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import ChatPromptTemplate

class PrecheckClassification(BaseModel):
    """Structured output for precheck classification results."""
    classification: Literal["chitchat", "query"]


def precheck_node(state: RAGState) -> RAGState:
    """
    Classifies user input and handles chitchat vs. query routing.
    
    This node serves as the entry point for all user interactions, determining
    whether the input requires the full RAG pipeline or can be handled as
    casual conversation.
    
    Args:
        state: Current RAGState containing user's original query
        
    Returns:
        Updated RAGState with classification results and potentially final answer
        
    Raises:
        ValueError: If original_query is missing or LLM operations fail
    """
    # Extract and validate the user's query
    try:
        query = state["original_query"].strip()
    except Exception:
        api_logger.exception("Missing or invalid 'original_query' in precheck_node")
        raise ValueError("original_query missing or not a string")

    def apply_context_reset(state: RAGState) -> None:
        """
        Resets conversation context for new query sessions.
        
        Clears previous document selections, external context, and adds
        a system message indicating a fresh start.
        """
        state["selected_docs"] = []
        state["external_context"] = None
        state["needs_retrieval"] = True
        state.setdefault("chat_history", []).append({
            "role": "system",
            "content": "--- New query started. You may ignore prior context if available. ---"
        })

    # Apply context reset if requested
    if state.get("reset_context"):
        apply_context_reset(state)

    # Classify the input using structured LLM output
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

        # Use structured output for reliable classification
        structured_precheck_llm = llm.with_structured_output(PrecheckClassification)
        chain = precheck_prompt | structured_precheck_llm
        result = chain.invoke({"query": query})
        classification = result.classification.strip().lower()
    
    except Exception:
        api_logger.exception("LLM classification failed in precheck_node")
        raise ValueError("Precheck classification failed")

    # Handle chitchat classification
    if classification == "chitchat":
        try:
            # Generate friendly response for casual conversation
            friendly_prompt = f"""

            This message is casual:
            "{query}"

            You are an AI Assistant for IP-LINK — a platform that helps users explore and understand university intellectual property (IP), such as patents and licensable technologies. You have access to IP data across almost all major US Universities.

            Reply naturally and conversationally. If the user is asking who you are or what you can do, introduce yourself as the IP-LINK assistant. 
            """

            reply = llm.invoke(friendly_prompt)

            # Update chat history with the casual interaction
            state.setdefault("chat_history", []).append({
                "role": "user",
                "content": query
            })

            state["chat_history"].append({
                "role": "assistant",
                "content": reply.content
            })
            
            # End pipeline with chitchat response
            return {**state, "final_answer": reply.content, "precheck_triggered": True}
            
        except Exception:
            api_logger.exception("LLM failed to generate friendly response in precheck_node")
            raise ValueError("Chitchat fallback LLM generation failed")

    # Continue to next node for query processing
    return {**state, "precheck_triggered": False}


# ----------------------
# NODE 2: QUERY EXTRACTOR
# ----------------------
"""
QUERY EXTRACTOR NODE - Natural Language to Search Query Conversion

Purpose:
- Converts natural language user input into clean, focused search queries
- Removes conversational filler, greetings, and irrelevant phrasing
- Optimizes queries for semantic vector search in Pinecone
- Ensures consistent query format for downstream retrieval

This preprocessing step significantly improves retrieval quality by focusing
the search on core concepts rather than conversational elements.
"""

class QueryExtractionOutput(BaseModel):
    """Structured output for extracted search queries."""
    rewritten_query: str

def query_extractor_node(state: RAGState) -> RAGState:
    """
    Extracts a clean, focused search query from natural language input.
    
    This node processes the user's original query to create an optimized
    version suitable for semantic search in the vector database.
    
    Args:
        state: Current RAGState containing original_query
        
    Returns:
        Updated RAGState with extracted search_query
        
    Raises:
        ValueError: If original_query is missing or LLM extraction fails
    """
    # Validate input query
    try:
        original_query = state["original_query"].strip()
        if not original_query:
            raise ValueError("Missing original_query in state.")
    except Exception as e:
        api_logger.exception("Error accessing original_query in query_extractor_node")
        raise ValueError("Failed to extract original query.")
    
    # Extract clean search query using structured LLM output
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


# ----------------------
# NODE 3: FIND (RETRIEVAL)
# ----------------------
"""
FIND NODE - Vector Database Retrieval

Purpose:
- Performs semantic search in Pinecone vector database
- Retrieves top-k most relevant university IP documents
- Fetches full document content from PostgreSQL using source IDs
- Builds Document objects with content and metadata for downstream processing

This is the core retrieval component that connects user queries with
relevant university intellectual property documents through semantic similarity.
"""

def find_node(state: RAGState, cursor) -> RAGState:
    """
    Retrieves relevant documents from vector database based on search query.
    
    This node performs the core retrieval operation by:
    1. Converting search query to vector embedding
    2. Querying Pinecone for similar document vectors
    3. Fetching full document content from PostgreSQL
    4. Building Document objects for downstream processing
    
    Args:
        state: Current RAGState containing search_query
        cursor: PostgreSQL database cursor for content retrieval
        
    Returns:
        Updated RAGState with retrieved_docs populated
        
    Raises:
        ValueError: If embedding or vector search fails
    """
    # Use extracted search query or fallback to original
    query = state.get("search_query") or state["original_query"]
    docs = []

    # Convert query text to vector embedding
    try:
        vector = embedder.embed_query(query)
    except Exception as e:
        api_logger.exception("Error embedding query")
        raise ValueError("Query Embedding Failed!") 
    
    # Search Pinecone vector database for similar documents
    try:
        results = index.query(
            vector=vector, 
            top_k=10,                    # Retrieve top 10 most similar documents
            include_metadata=True,       # Include document metadata
            namespace=""                 # Use default namespace
        )
    except Exception as e:
        api_logger.exception("Pinecone query failed in find_node")
        raise ValueError("Search failed.")    

    # Fetch full document content from PostgreSQL for each match
    for match in results["matches"]:
        source_id = match["metadata"].get("source_id")
        try:
            # Query PostgreSQL for full document content
            cursor.execute("SELECT * FROM ip_new WHERE id = %s", (source_id,))
            row = cursor.fetchone()

            if row:
                # Build Document object with content and metadata
                doc = Document(
                    page_content=row[4],        # Document contents from database
                    metadata={
                        "source_id": source_id,
                        "title": row[2],        # Document title
                        **match["metadata"]     # Additional Pinecone metadata
                    }
                )
                docs.append(doc)
                
        except Exception as e:
            api_logger.warning(f"Failed to fetch or build doc for source_id {source_id}: {e}")
            continue  # Skip broken doc, don't crash whole thing

    # Update state with retrieved documents
    state["retrieved_docs"] = docs
    return state


# ----------------------
# NODE 4: EXPLORE RESULTS
# ----------------------
"""
EXPLORE RESULTS NODE - Document Selection and Intent Processing

Purpose:
- Manages document selection from retrieved results
- Handles intent classification (summarize, explain, compare, other)
- Sets up external context for comparison operations
- Prepares state for final generation based on user preferences

This node serves as a bridge between retrieval and generation, allowing for
flexible document interaction patterns and intent-based response customization.
"""

def explore_results_node(state: RAGState) -> RAGState:
    """
    Processes retrieved documents and prepares for generation based on user intent.
    
    This node handles the selection of relevant documents and configures
    the generation parameters based on the user's intended interaction type.
    
    Args:
        state: Current RAGState with retrieved_docs and potentially intent
        
    Returns:
        Updated RAGState with selected_docs, intent, and additional context
        
    Note:
        Currently implements temporary auto-selection logic. In production,
        this would integrate with frontend UI for user selection.
    """
    # Document Selection Logic
    # TODO: Replace with actual user selection from frontend UI
    if not state.get("selected_docs"):
        # TEMPORARILY: auto-select first 2 retrieved documents
        retrieved_docs = state.get("retrieved_docs", [])
        state["selected_docs"] = retrieved_docs[:2]

    # Intent Assignment Logic  
    # TODO: Replace with actual intent classification from user interaction
    if not state.get("intent"):
        # TEMPORARILY: auto-assign comparison intent for demonstration
        state["intent"] = "compare"  # Could be "summarize", "explain", "compare", "other"

    # Handle comparison with external context
    # For single document comparison, provide external context
    if state["intent"] == "compare" and len(state["selected_docs"]) == 1:
        # TODO: Replace with actual external context from user input
        state["external_context"] = "User provided description of wearable biosensor."

    # Handle custom questions for "other" intent
    if state['intent'] == 'other':
        # TODO: Replace with actual custom question from user
        state['custom_question'] = "Which one of these is more cost-effective and why?"

    return state


# ----------------------
# NODE 5: GENERATION
# ----------------------
"""
GENERATION NODE - Intent-Based Response Generation

Purpose:
- Generates final responses based on user intent and selected documents
- Supports multiple interaction modes: summarize, explain, compare, custom
- Manages chat history for multi-turn conversations
- Implements context-aware response formatting

This is the final processing node that synthesizes retrieved information
into user-friendly responses tailored to their specific intent and needs.
"""

def format_chat_history(chat_history, anchor_turns=2, recency_turns=4):
    """
    Formats chat history for context-aware generation while managing token limits.
    
    This function implements a sophisticated chat history compression strategy:
    - Keeps ALL system messages (important context markers)
    - Preserves first N conversations (anchor_turns) for context continuity
    - Includes last M conversations (recency_turns) for immediate relevance
    - Formats output in a readable turn-based structure
    
    Args:
        chat_history: List of message dictionaries with role and content
        anchor_turns: Number of early conversations to preserve
        recency_turns: Number of recent conversations to include
        
    Returns:
        Formatted string representation of compressed chat history
        
    Note:
        This approach balances context preservation with token efficiency,
        preventing context window overflow in long conversations.
    """
    
    if not chat_history:
        return ""

    # === Step 1: Separate system messages from user/assistant messages ===
    # System messages contain important context markers and should always be preserved
    system_messages = [m for m in chat_history if m['role'].lower() == "system"]
    non_system_messages = [m for m in chat_history if m['role'].lower() != "system"]

    # === Step 2: Group user/assistant messages into conversation pairs ===
    conversations = []
    temp = []

    for message in non_system_messages:
        temp.append(message)
        # Complete conversation when we hit an assistant response
        if message['role'].lower() == "assistant":
            conversations.append(temp)
            temp = []

    # === Step 3: Select conversations by position (no duplicate indexes) ===
    # Take early conversations for context anchoring
    anchors = conversations[:anchor_turns]
    # Take remaining conversations for recency selection
    recents = conversations[anchor_turns:] if anchor_turns < len(conversations) else []

    # Get only the *last* N recent conversations
    recents = recents[-recency_turns:] if recents else []

    # === Step 4: Merge all selected components ===
    combined = system_messages + [m for conv in anchors for m in conv] + [m for conv in recents for m in conv]

    # === Step 5: Format for readable output ===
    formatted = []
    formatted.append("===== Cropped Chat History =====")

    turn_counter = 1
    current_turn = []

    for message in combined:
        role = message['role'].capitalize()
        content = message['content']

        if role == "User":
            # Start new turn when we encounter a user message
            if current_turn:
                formatted.append("\n".join(current_turn))
                turn_counter += 1
                current_turn = []

            formatted.append(f"\n==== Turn {turn_counter} ====")
            current_turn.append(f"({role}) {content}")

        elif role == "Assistant":
            current_turn.append(f"({role}) {content}")

        elif role == "System":
            # System messages are shown independently
            formatted.append(f"(System) {content}")

    # Add final turn if exists
    if current_turn:
        formatted.append("\n".join(current_turn))

    return "\n".join(formatted)


def generation_node(state: RAGState) -> RAGState:
    """
    Generates final responses based on user intent and selected documents.
    
    This node implements intent-based generation logic supporting multiple
    interaction modes with context-aware chat history management.
    
    Supported Intents:
    - summarize: Concise summary of each selected document
    - explain: Detailed explanation in accessible language
    - compare: Comparative analysis of 2+ documents or 1 doc + external context
    - other: Custom question-based generation
    
    Args:
        state: Current RAGState with all necessary generation inputs
        
    Returns:
        Updated RAGState with final_answer and updated chat_history
        
    Raises:
        ValueError: If required inputs are missing or invalid
    """
    # Extract and validate required inputs
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
    # Input Validation
    # -----------------------------
    if not selected_docs or len(selected_docs) == 0:
        raise ValueError("No documents selected.")
    if not intent:
        raise ValueError("Intent must be provided.")
    if intent == "other" and not custom_question:
        raise ValueError("Custom question must be provided when intent is 'other'.")

    # Format documents for prompt injection
    docs_text = "\n\n".join([
        f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(selected_docs)
    ])

    # Format and compress chat history for context
    history_text = format_chat_history(chat_history)

    # Log formatted history for debugging (optional)
    # app_logger1.info("\n===== Cropped Chat History Before Model Call =====")
    # app_logger1.info(history_text)

    # -----------------------------
    # Intent-Based Generation Logic
    # -----------------------------
    try:
        if intent in {"summarize", "explain"}:
            """
            Handle single-document processing intents.
            Processes each selected document individually with appropriate prompting.
            """
            outputs = []
            
            for i, doc in enumerate(selected_docs):
                # Set task description based on intent
                if intent == "summarize":
                    task = "Summarize the following document clearly and concisely in 3–5 sentences."
                elif intent == "explain":
                    task = "Explain the following document in simple, accessible language for a general audience."

                # Generate response for individual document
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

            # Combine all document responses
            output = "\n\n".join(outputs)

        elif intent == "compare":
            """
            Handle document comparison intent.
            Supports flexible comparison of multiple documents or with external context.
            """
            if len(selected_docs) >= 2:
                # Multi-document comparison mode
                docs_for_comparison = "\n\n".join([
                    f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(selected_docs)
                ])
                
                prompt = f"""
                You are a professional AI assistant on IP-LINK, a platform with access to intellectual property (IP) data from top research universities across the USA. Your role is to help users by answering their questions and continuing multi-turn conversations about university technologies, patents, and innovations in a clear, helpful, and engaging manner. 

                Conversation so far:
                {history_text}

                Compare the following {len(selected_docs)} documents:

                {docs_for_comparison}

                Provide a comprehensive comparison highlighting:
                - Key similarities and differences
                - Purpose and applications of each technology
                - Technical approaches and methodologies
                - Potential advantages and limitations
                - Use cases and target markets
                """
                
            elif len(selected_docs) == 1 and (external_context or "").strip():
                # Single document + external context comparison mode
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
                raise ValueError("Compare requires either 2+ documents or 1 document + external context.")

            response = llm.invoke(prompt)
            output = response.content.strip()

        elif intent == "other":
            """
            Handle custom question intent.
            Uses user-provided custom question for flexible document interaction.
            """
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
    # Create descriptive user message including intent information
    user_message = f'{original_query} (Intent: {intent})'
    if intent == "other" and custom_question:
        user_message += f' (Custom Question: {custom_question})'

    # Add both user query and assistant response to history
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": output})

    state["chat_history"] = chat_history

    # -----------------------------
    # Reset Control Flags and Finalize
    # -----------------------------
    # Clear control flags for next interaction
    state["reset_context"] = False
    state["needs_retrieval"] = False
    
    # Set final answer for pipeline completion
    state["final_answer"] = output

    return state


# ----------------------
# GRAPH CONSTRUCTION
# ----------------------
"""
LANGGRAPH WORKFLOW DEFINITION

This section defines the complete RAG pipeline using LangGraph's StateGraph framework.
The graph connects all processing nodes with conditional routing logic to create
a flexible, multi-path workflow that adapts to different query types and user intents.

Graph Structure:
1. precheck (Entry Point)
   ├── → __end__ (if chitchat detected)
   ├── → generate (if follow-up query with existing context)
   └── → query_extractor (if new query requiring retrieval)

2. query_extractor → find → explore_results → generate → __end__

This design enables efficient processing by:
- Bypassing unnecessary steps for chitchat
- Reusing existing context for follow-up questions
- Performing full retrieval pipeline for new queries
"""

from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda

# Initialize the main graph with RAGState
graph = StateGraph(RAGState)

# Add all processing nodes to the graph
graph.add_node("precheck", RunnableLambda(precheck_node))
graph.add_node("query_extractor", RunnableLambda(query_extractor_node))
graph.add_node("find", RunnableLambda(find_node))
graph.add_node("explore_results", RunnableLambda(explore_results_node))
graph.add_node("generate", RunnableLambda(generation_node))

# Set the entry point - all queries start with precheck
graph.set_entry_point("precheck")

# Add conditional routing after precheck node
# This implements the core routing logic for the RAG system
graph.add_conditional_edges("precheck", lambda state:
    "__end__" if state.get("precheck_triggered") else           # End if chitchat
    "query_extractor" if state.get("needs_retrieval") else      # New query path
    "generate"                                                  # Follow-up path
)

# # Alternative routing for query type classification (currently unused)
# # This would enable different handling for new vs. follow-up queries
# graph.add_conditional_edges("query_type_classifier", lambda state:
#     "generate" if state["query_type"] == "follow_up" else "query_extractor"
# )

# Define linear flow for new query processing
# These edges create the main retrieval → generation pipeline
graph.add_edge("query_extractor", "find")
graph.add_edge("find", "explore_results")
graph.add_edge("explore_results", "generate")

# Final output - generation always leads to completion
graph.add_edge("generate", "__end__")

# Compile the graph into an executable application
# This creates the final RAG system ready for query processing
app = graph.compile()

"""
USAGE EXAMPLE:

# Initialize state with user query
initial_state = {
    "original_query": "Tell me about brain tumor detection technologies",
    "chat_history": [],
    "needs_retrieval": True
}

# Run the RAG pipeline
result = app.invoke(initial_state)

# Extract final answer
final_answer = result["final_answer"]
"""