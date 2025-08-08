import requests
import io
import pdfplumber
from sentence_transformers import SentenceTransformer
import pinecone
from llama_cpp import Llama
from fastapi import HTTPException

# Hardcoded Pinecone API key (replace with your key)
PINECONE_API_KEY = "pcsk_HQjF8_9D9xU8xdCQuP2WYY6aQTTFWTZecWUajFw5VBt1MfzBMJj8mABkc8jRzbpnGuuVR"

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if "policy-docs" not in pc.list_indexes().names():
    pc.create_index(
        name="policy-docs",
        dimension=384,  # dimension for 'all-MiniLM-L6-v2'
        metric="cosine"
    )
index = pc.Index("policy-docs")

# Initialize embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize LLM model
llm = Llama.from_pretrained(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",  # 4-bit quantized model file
    n_gpu_layers=20,
    n_ctx=2048,
)

def pdf_to_chunks(pdf_url: str):
    """Download PDF and extract text with page numbers."""
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()

        chunks = []
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    chunks.append({
                        "text": text,
                        "metadata": {"page": page.page_number, "source": pdf_url}
                    })
        return chunks

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF processing failed: {str(e)}")

def upsert_chunks(chunks):
    """Embed text chunks and upsert to Pinecone index."""
    embeddings = embed_model.encode([chunk["text"] for chunk in chunks])
    vectors = [{
        "id": f"chunk_{idx}",
        "values": emb.tolist(),
        "metadata": chunks[idx]["metadata"]
    } for idx, emb in enumerate(embeddings)]

    # Upsert in batches of 100
    for i in range(0, len(vectors), 100):
        batch = vectors[i:i+100]
        index.upsert(vectors=batch)

def query_index_and_answer(questions):
    """For each question, query Pinecone and generate answer with LLM."""
    answers = []

    for question in questions:
        query_embed = embed_model.encode(question)
        results = index.query(vector=query_embed.tolist(), top_k=3, include_metadata=True)

        # Prepare context for LLM prompt
        context = "\n".join(
            f"Page {match.metadata['page']}: {match.metadata.get('text', '')}"
            for match in results.matches
        )

        prompt = f"""Answer based on these policy clauses:
{context}

Question: {question}
Respond in JSON format:
{{"decision": "APPROVED/DENIED", "amount": number, "clauses": ["page X"], "explanation": "text"}}
"""

        answer = llm(prompt, max_tokens=200)
        answers.append(answer)

    return answers
