
# # # import os
# # # from dotenv import load_dotenv

# # # from langchain.agents import create_react_agent, AgentExecutor
# # # from langchain_ollama import ChatOllama
# # # from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# # # from langchain.prompts import PromptTemplate
# # # from ollama import Tool
# # # from tools import retrieve_tool, web_search, calculator
# # # from prompts import SYSTEM_PROMPT, ANSWER_PROMPT

# # # load_dotenv()

# # # MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")


# # # from langchain.agents import create_react_agent, AgentExecutor, Tool
# # # from langchain.prompts import PromptTemplate
# # # from langchain_community.llms import Ollama

# # # def build_agent():
# # #     llm = Ollama(model="llama2")

# # #     tools = [
# # #         Tool(
# # #             name="Search",
# # #             func=lambda q: "Search results for: " + q,
# # #             description="Use this for searching"
# # #         )
# # #     ]

# # #     # Updated prompt with required input variables
# # #     template = """You are an intelligent agent.
# # # Use the following tools: {tools}
# # # Available tool names: {tool_names}

# # # Question: {input}

# # # Use the following reasoning and previous steps:
# # # {agent_scratchpad}
# # # """
# # #     prompt = PromptTemplate(
# # #         input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
# # #         template=template
# # #     )

# # #     agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
# # #     return AgentExecutor(agent=agent, tools=tools, verbose=True)

# # # # def build_agent():
# # # #     llm = Ollama(model="llama2")

# # # #     tools = [
# # # #         Tool(
# # # #             name="Search",
# # # #             func=lambda q: "Search results for: " + q,
# # # #             description="Use this for searching"
# # # #         )
# # # #     ]

# # # #     # ✅ Correct prompt with required variables
# # # #     template = """
# # # #     You are an AI assistant using the ReAct framework.
# # # #     You have access to the following tools:

# # # #     {tools}

# # # #     Use the following tool names: {tool_names}

# # # #     When answering, follow this format:

# # # #     Question: the input question you must answer
# # # #     Thought: think about what to do
# # # #     Action: the action to take, should be one of [{tool_names}]
# # # #     Action Input: the input to the action
# # # #     Observation: the result of the action
# # # #     ... (repeat Thought/Action/Observation as needed)
# # # #     Final Answer: the answer to the question

# # # #     Begin!

# # # #     Question: {input}
# # # #     {agent_scratchpad}
# # # #     """

# # # #     prompt = PromptTemplate(
# # # #         template=template,
# # # #         input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
# # # #     )

# # # #     agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
# # # #     return AgentExecutor(agent=agent, tools=tools, verbose=True)



# # # # def build_agent():
# # # #     # Free local LLM via Ollama
# # # #     llm = ChatOllama(model="llama2")  # You can change model to mistral or codellama

# # # #     tools = [
# # # #         Tool(
# # # #             name="Search",
# # # #             func=lambda q: "Search results for: " + q,
# # # #             description="Use this for searching"
# # # #         )
# # # #     ]

# # # #     # Create the prompt template
# # # #     template = """Answer the question as best as you can using the tools."""
# # # #     prompt = PromptTemplate(template=template, input_variables=[])

# # # #     # Create the agent
# # # #     agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
# # # #     agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# # # #     return agent_executor



# # # # def build_agent():
# # # #     llm = ChatOllama(model=MODEL, temperature=0.2)
# # # #     tools = [retrieve_tool, web_search, calculator]
# # # #     prompt = ChatPromptTemplate.from_messages([
# # # #         ("system", SYSTEM_PROMPT),
# # # #         MessagesPlaceholder(variable_name="chat_history"),
# # # #         ("human", "Question: {input}"),
# # # #         ("assistant", "First think whether to use tools. If using, show plan then execute."),
# # # #     ])
# # # #     agent = initialize_agent(


# # # #         tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
# # # #         verbose=True, handle_parsing_errors=True, max_iterations=6
# # # #     )
# # # #     agent.prompt = prompt
# # # #     return agent

# # # def pretty_context(query: str, retrieved: str) -> str:
# # #     return f"\n\n[Retrieved Context]\n{retrieved}\n"

# # # def main():
# # #     print("\nAgentic RAG (free/local). Type 'exit' to quit.\n")
# # #     agent = build_agent()
# # #     chat_history = []
# # #     while True:
# # #         try:
# # #             user = input("> ").strip()
# # #         except EOFError:
# # #             break
# # #         if not user:
# # #             continue
# # #         if user.lower() in {"exit", "quit"}:
# # #             break
# # #         # Force a retrieve pass injected into the prompt (agent can still call tools as needed)
# # #         retrieved = retrieve_tool.run(user)
# # #         composed = ANSWER_PROMPT.format(question=user, context=retrieved)
# # #         res = agent.invoke({"input": composed, "chat_history": chat_history})
# # #         answer = res["output"] if isinstance(res, dict) and "output" in res else str(res)
# # #         print("\n" + answer + "\n")
# # #         chat_history.append(("human", user))
# # #         chat_history.append(("ai", answer))

# # # if __name__ == "__main__":
# # #     main()


# # import os
# # from dotenv import load_dotenv
# # from langchain_ollama import ChatOllama
# # from langchain.agents import initialize_agent, AgentType
# # from langchain_community.tools import DuckDuckGoSearchRun
# # from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# # from tools import retrieve_tool, web_search, calculator
# # from prompts import SYSTEM_PROMPT, ANSWER_PROMPT

# # load_dotenv()

# # MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

# # def build_agent():
# #     # Use Ollama LLM
# #     llm = ChatOllama(model=MODEL, temperature=0.1)

# #     # Define tools
# #     tools = [retrieve_tool, web_search, calculator, DuckDuckGoSearchRun()]

# #     # Initialize ReAct agent
# #     agent = initialize_agent(
# #         tools=tools,
# #         llm=llm,
# #         agent=AgentType.OPENAI_MULTI_FUNCTIONS,  # Supports function calling
# #         verbose=True,
# #         handle_parsing_errors=True
# #     )
# #     return agent

# # def main():
# #     print("\nAgentic RAG (free/local). Type 'exit' to quit.\n")
# #     agent = build_agent()
# #     chat_history = []

# #     while True:
# #         try:
# #             user = input("> ").strip()
# #         except EOFError:
# #             break
# #         if not user:
# #             continue
# #         if user.lower() in {"exit", "quit"}:
# #             break

# #         retrieved = retrieve_tool.run(user)
# #         composed = ANSWER_PROMPT.format(question=user, context=retrieved)

# #         res = agent.invoke({"input": composed})
# #         print("\n" + res["output"] + "\n")

# # if __name__ == "__main__":
# #     main()

# import os
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.llms import Ollama
# from langchain.chains import RetrievalQA
# from langchain.tools import Tool
# from langchain.agents import create_openai_tools_agent, AgentExecutor
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema import SystemMessage

# # ----------------------------
# # ✅ CONFIGURATION
# # ----------------------------
# DB_DIR = "chroma_db"
# MODEL_NAME = "llama2"  # You can change this to "mistral" or any other Ollama model

# # ----------------------------
# # ✅ 1. Embeddings + Vector Store
# # ----------------------------
# print("[INFO] Loading embeddings and vector DB...")
# embeddings = OllamaEmbeddings(model=MODEL_NAME)
# vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# # ----------------------------
# # ✅ 2. LLM (Ollama)
# # ----------------------------
# print("[INFO] Initializing Ollama model...")
# llm = Ollama(model=MODEL_NAME)

# # ----------------------------
# # ✅ 3. RAG Chain (Retrieval-based Q&A)
# # ----------------------------
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     return_source_documents=True
# )

# # Wrap as a Tool
# rag_tool = Tool(
#     name="RAG-Answer",
#     func=lambda q: qa_chain.run(q),
#     description="Use this tool when you need to answer questions using the knowledge base."
# )

# # ----------------------------
# # ✅ 4. Create Agent using Tools
# # ----------------------------
# system_prompt = SystemMessage(content="You are an AI assistant with access to a knowledge base. Answer clearly.")
# prompt = ChatPromptTemplate.from_messages([system_prompt])

# agent = create_openai_tools_agent(
#     llm=llm,
#     tools=[rag_tool],
#     prompt=prompt
# )

# agent_executor = AgentExecutor(agent=agent, tools=[rag_tool], verbose=True)

# # ----------------------------
# # ✅ 5. Interactive Loop
# # ----------------------------
# print("\nAgentic RAG (Free & Local). Type 'exit' to quit.\n")
# while True:
#     query = input("> ")
#     if query.lower() in ["exit", "quit"]:
#         print("Goodbye!")
#         break
#     response = agent_executor.invoke({"input": query})
#     print("\n[AI]:", response["output"])

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain_community.chat_models import ChatOllama

# 1. Load documents
loader = TextLoader("data\knowledge_base.txt", encoding="utf-8")
documents = loader.load()

# 2. Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Store in FAISS vector database
db = FAISS.from_documents(docs, embeddings)

# 5. Create retriever
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 6. Initialize local LLM (Ollama)
llm = ChatOllama(model="llama3", temperature=0.3)  # Make sure Ollama is installed & running locally

# 7. Define RAG pipeline
template = """
You are a helpful assistant. Use the following context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = RunnableMap({
    "context": retriever,
    "question": RunnablePassthrough()
}) | prompt | llm

# 8. Start interactive Q&A
print("\nAgentic RAG (free/local). Type 'exit' to quit.\n")

while True:
    query = input("> ")
    if query.lower() == "exit":
        break
    answer = rag_chain.invoke(query)
    print("\nAnswer:", answer.content, "\n")


