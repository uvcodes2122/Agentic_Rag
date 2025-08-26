
import os
from dotenv import load_dotenv

from langchain.agents import initialize_agent, AgentType
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools import retrieve_tool, web_search, calculator
from prompts import SYSTEM_PROMPT, ANSWER_PROMPT

load_dotenv()

MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

def build_agent():
    llm = ChatOllama(model=MODEL, temperature=0.2)
    tools = [retrieve_tool, web_search, calculator]
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Question: {input}"),
        ("assistant", "First think whether to use tools. If using, show plan then execute."),
    ])
    agent = initialize_agent(
        tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True, handle_parsing_errors=True, max_iterations=6
    )
    agent.prompt = prompt
    return agent

def pretty_context(query: str, retrieved: str) -> str:
    return f"\n\n[Retrieved Context]\n{retrieved}\n"

def main():
    print("\nAgentic RAG (free/local). Type 'exit' to quit.\n")
    agent = build_agent()
    chat_history = []
    while True:
        try:
            user = input("> ").strip()
        except EOFError:
            break
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break
        # Force a retrieve pass injected into the prompt (agent can still call tools as needed)
        retrieved = retrieve_tool.run(user)
        composed = ANSWER_PROMPT.format(question=user, context=retrieved)
        res = agent.invoke({"input": composed, "chat_history": chat_history})
        answer = res["output"] if isinstance(res, dict) and "output" in res else str(res)
        print("\n" + answer + "\n")
        chat_history.append(("human", user))
        chat_history.append(("ai", answer))

if __name__ == "__main__":
    main()
