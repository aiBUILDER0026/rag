from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List
import rag  # import your RAG logic here

app = FastAPI()

class QueryRequest(BaseModel):
    documents: str  # URL of the PDF
    questions: List[str]

@app.post("/hackrx/run")
async def run_query(request: QueryRequest, authorization: str = Header(None)):
    # Simple Bearer token auth check
    if authorization != "Bearer your-secure-api-key":
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 1. Process PDF -> text chunks
    chunks = rag.pdf_to_chunks(request.documents)

    # 2. Upsert to Pinecone only if index empty
    if rag.index.describe_index_stats().total_vector_count == 0:
        rag.upsert_chunks(chunks)

    # 3. Query and get answers
    answers = rag.query_index_and_answer(request.questions)
    return {"answers": answers}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


