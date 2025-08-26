from fastapi import FastAPI
from pydantic import BaseModel
from rag_agent import build_agent
from prompts import ANSWER_PROMPT
from tools import retrieve_tool

app = FastAPI(title="Agentic RAG (Free)")
agent = build_agent()

class Query(BaseModel):
    query: str

@app.post("/")
def ask(q: Query):
    retrieved = retrieve_tool.run(q.query)
    composed = ANSWER_PROMPT.format(question=q.query, context=retrieved)
    res = agent.invoke({"input": composed, "chat_history": []})
    output = res["output"] if isinstance(res, dict) and "output" in res else str(res)
    return {"answer": output}
