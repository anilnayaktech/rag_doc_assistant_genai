from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scripts.rag_pipeline import qa_chain
from scripts.safety import is_safe

app = FastAPI(title="GenAI RAG Chatbot")

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    if not is_safe(query.question):
        raise HTTPException(status_code=400, detail="Unsafe content detected!")
    
  #  answer = qa_chain.run(query.question)
  #  return {"answer": answer}
  # Use __call__() instead of .run()
    result = qa_chain(query.question)  # returns a dict
    return {"answer": result["result"], "sources": result["source_documents"]}

# Run: uvicorn app.main:app --reload
