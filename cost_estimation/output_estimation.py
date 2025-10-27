import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger
import json

setup_logger("lightrag", level="INFO")

WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    # IMPORTANT: Both initialization calls are required!
    await rag.initialize_storages()  # Initialize storage backends
    await initialize_pipeline_status()  # Initialize processing pipeline
    return rag

async def main():
    try:
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
        # Perform naive search
        print("\n=====================")
        print("Query mode: naive")
        print("=====================")
        with open(qpath, "r", encoding="utf-8") as f:
            questions = json.load(f)

        naive_results = []
        out_dir = os.path.join(WORKING_DIR, "results")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "novel_results_naive.json")

        for q in questions[:5]:
            id = q.get("id")
            question = q.get("question")
            try:
                tmp = await rag.aquery(
                        question.strip(), param=QueryParam(mode="naive"))
                print(type(tmp))
                if isinstance(tmp, dict):
                        answer_text = tmp.get("text") or tmp.get("answer") or str(tmp)
                else:
                    answer_text = str(tmp)
            except Exception as e:
                answer_text = f"ERROR: {e}"
            naive_results.append({
                "id": id,
                "question": question,
                "answer": answer_text
            })

        # save results
        with open(out_file, "w", encoding="utf-8") as outf:
            json.dump(naive_results, outf, ensure_ascii=False, indent=2)

        print(f"Saved results to {out_file}")
        
        

        # Perform local search
        print("\n=====================")
        print("Query mode: local")
        print("=====================")
        print(
            await rag.aquery(
                "What are the top themes in this story?", param=QueryParam(mode="local")
            )
        )

        # Perform global search
        print("\n=====================")
        print("Query mode: global")
        print("=====================")
        print(
            await rag.aquery(
                "What are the top themes in this story?",
                param=QueryParam(mode="global"),
            )
        )

        # Perform hybrid search
        print("\n=====================")
        print("Query mode: hybrid")
        print("=====================")
        print(
            await rag.aquery(
                "What are the top themes in this story?",
                param=QueryParam(mode="hybrid"),
            )
        )
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())