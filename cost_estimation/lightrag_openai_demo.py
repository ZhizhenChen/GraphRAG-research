
import os
import asyncio
import logging
import logging.config
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import logger, set_verbose_debug
import json

WORKING_DIR = "./dickens"

def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "lightrag_demo.log"))

    print(f"\nLightRAG demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

async def main():
    # Check if OPENAI_API_KEY environment variable exists
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY environment variable is not set. Please set this variable before running the program."
        )
        print("You can set the environment variable by running:")
        print("  export OPENAI_API_KEY='your-openai-api-key'")
        return  # Exit the async function

    try:
        # Clear old data files
        files_to_delete = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_doc_status.json",
            "kv_store_full_docs.json",
            "kv_store_text_chunks.json",
            "vdb_chunks.json",
            "vdb_entities.json",
            "vdb_relationships.json",
        ]

        for file in files_to_delete:
            file_path = os.path.join(WORKING_DIR, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleting old file:: {file_path}")

        # Initialize RAG instance
        rag = await initialize_rag()

        # Test embedding function
        test_text = ["This is a test string for embedding."]
        embedding = await rag.embedding_func(test_text)
        embedding_dim = embedding.shape[1]
        print("\n=======================")
        print("Test embedding function")
        print("========================")
        print(f"Test dict: {test_text}")
        print(f"Detected embedding dimension: {embedding_dim}\n\n")

        cpath = "tiktokenestimates/Datasets/Corpus/novel.json"
        with open(cpath, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        for c in corpus:
            try:
                await rag.ainsert(c.get("context").strip())
            except Exception as e:
                print(f"Error inserting context id {c.get('id')}: {e}")
        
        

        qpath = "tiktokenestimates/Datasets/Questions/novel_questions.json"

        # Perform hybrid search
        print("\n=====================")
        print("Query mode: hybrid")
        print("=====================")
        with open(qpath, "r", encoding="utf-8") as f:
            questions = json.load(f)

        out_dir = os.path.join(WORKING_DIR, "results")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "novel_results_hybrid.json")

        hybrid_results = []
        for q in questions[5:10]:
            id = q.get("id")
            question = q.get("question")
            record = {"id": id, "question": question, "answer": None, "data": {}}
            try:
                res = await rag.aquery_llm(
                        question.strip(), param=QueryParam(mode="hybrid"))
                if isinstance(res, dict):
                    # extract llm text (various possible keys)
                    llm_resp = res.get("llm_response") or res.get("llm") or {}
                    answer_text = (
                        llm_resp.get("content")
                        or llm_resp.get("text")
                        or res.get("text")
                        or res.get("answer")
                        or str(res)
                    )
                    data = res.get("data") or {}
                else:
                    answer_text = str(res)
                    data = {}
                chunks = data.get("chunks", []) or []
                entities = data.get("entities", []) or []
                relations = data.get("relationships", []) or data.get("relations", []) or []
                record["answer"] = answer_text
                record["data"]["chunks"] = chunks
                record["data"]["entities"] = entities
                record["data"]["relationships"] = relations

            except Exception as e:
                answer_text = f"ERROR: {e}"
                record["answer"] = answer_text
                record["data"] = {}
            hybrid_results.append(record)

        # save results
        with open(out_file, "w", encoding="utf-8") as outf:
            json.dump(hybrid_results, outf, ensure_ascii=False, indent=2)

        print(f"Saved results to {out_file}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()

if __name__ == "__main__":
    # Configure logging before running the main function
    configure_logging()
    asyncio.run(main())
    print("\nDone!")

