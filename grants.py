from dotenv import load_dotenv
import os
load_dotenv()

import os
from langchain.chat_models import ChatOpenAI

from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Optional
from typing import Dict
import logging


llm = ChatOpenAI(
    model="gpt-4o",  # or "gpt-4o-mini"
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

api_logger = logging.getLogger("api_logger") # Main API logger imported from main.py

class GrantQueryExtractionOutput(BaseModel):
    search_term: str
    grant_topic: str


from typing import TypedDict

class RAGState(TypedDict, total=False):
    original_query: str
    search_term: Optional [str]
    grant_topic: Optional [str]
    selected_grant_ids: Optional [list[str]]  # if you're letting users pick
    retrieved_grants: Optional [list[dict]]   # raw or cleaned grant info
    user_intent: Optional [str]               # optional, from frontend or follow-up
    generation_response: Optional [str]       # final answer from the model


def grant_query_extractor_node(state: RAGState) -> RAGState:
    try:
        original_query = state["original_query"].strip()
        if not original_query:
            raise ValueError("Missing original_query in state.")
    except Exception as e:
        api_logger.exception("Error accessing original_query in grant_query_extractor_node")
        raise ValueError("Failed to extract original query.")

    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
            "You are a helpful assistant that extracts a structured grant search query from a user's input. "
            "Return only a clean search phrase for the API and a short summarized topic for context."),

            ("human", f'''
            User Query:
            "{original_query}"

            Return in JSON:
            {{
              "search_term": "...",
              "grant_topic": "..."
            }}
            ''')
        ])

        structured_llm = llm.with_structured_output(GrantQueryExtractionOutput)
        chain = prompt | structured_llm

        result = chain.invoke({"original_query": original_query})
        state["search_term"] = result.search_term.strip()
        state["grant_topic"] = result.grant_topic.strip()
        return state

    except Exception as e:
        api_logger.exception("LLM grant query extraction failed in grant_query_extractor_node")
        raise ValueError("Grant query extraction via LLM failed.")

