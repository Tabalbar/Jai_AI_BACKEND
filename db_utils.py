from dotenv import load_dotenv
import os
load_dotenv()

import uuid
import json

import psycopg2

import logging

def setup_logger(name="db_logger", log_file="logs/db_logs.log", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handler = logging.FileHandler(log_file, mode='a')
        handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Console logging (for Cloud Run)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger

db_logger = setup_logger()


def get_pg_conn():
    try:
        return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        connect_timeout=5,
        )
    except Exception as e:
        db_logger.exception("Failed to connect to PostgreSQL")
        raise


def fetch_chat_history(cursor, conversation_id):
    try:
        cursor.execute("SELECT chat_history FROM conversations WHERE conversation_id = %s", (conversation_id,))
        result = cursor.fetchone()
        if result and result[0]:
            return result[0]  # chat_history already stored as JSON/array
        return []
    except Exception as e:
        db_logger.exception(f"Failed to fetch chat history for conversation_id {conversation_id}")
        return []


def create_new_conversation(cursor):
    try:
        conversation_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO conversations (conversation_id, chat_history) VALUES (%s, %s)",
            (conversation_id, [])
        )
        return conversation_id
    except Exception as e:
        db_logger.exception("Failed to create new conversation")
        raise


def update_chat_history(cursor, conversation_id, chat_history):
    try:
        cursor.execute(
            "UPDATE conversations SET chat_history = %s WHERE conversation_id = %s",
            (json.dumps(chat_history), conversation_id)
        )
    except Exception as e:
        db_logger.exception(f"Failed to update chat history for conversation_id {conversation_id}")
        raise

