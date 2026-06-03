"""
Agentic AI Research Assistant
RAG + ReAct Agent using LangChain + LangGraph + Google Gemini (free)
"""

import os
from dotenv import load_dotenv
load_dotenv()

# ========================================
# CONCEPT 1: ChatGoogleGenerativeAI (LLM)
# ========================================
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ========================================
# CONCEPT 2: PyPDFLoader (Document Loader)
# ========================================
from langchain_community.document_loaders import PyPDFLoader

pdf_path = input("Enter PDF path: ").strip().strip('"').strip("'")
loader = PyPDFLoader(pdf_path)
pages = loader.load()
print(f"✅ PDF loaded: {len(pages)} pages")

# ========================================
# CONCEPT 3: RecursiveCharacterTextSplitter
# ========================================
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(pages)
print(f"✅ Split into {len(chunks)} chunks")

# ========================================
# CONCEPT 4: HuggingFaceEmbeddings (Free, local)
# ========================================
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ========================================
# CONCEPT 5: FAISS (Vector Store)
# ========================================
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)
print("✅ Vector store ready (FAISS)")

# ========================================
# CONCEPT 6: as_retriever()
# ========================================
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ========================================
# CONCEPT 7: create_retriever_tool (RAG Tool)
# ========================================
from langchain_core.tools.retriever import create_retriever_tool

pdf_tool = create_retriever_tool(
    retriever,
    name="pdf_knowledge_base",
    description="Search the uploaded PDF document for information. Use this when the question is about the document's content.",
)

# ========================================
# CONCEPT 8: TavilySearch (Web Search Tool)
# ========================================
from langchain_tavily import TavilySearch

web_search = TavilySearch(max_results=3)

# ========================================
# CONCEPT 9: @tool decorator (Custom Tool)
# ========================================
from langchain_core.tools import tool

@tool
def save_answer(answer: str) -> str:
    """Save the final answer to answers.txt file."""
    with open("answers.txt", "a", encoding="utf-8") as f:
        f.write(answer + "\n\n---\n\n")
    return "✅ Answer saved to answers.txt"

# ========================================
# CONCEPT 10 + 11: create_react_agent (LangGraph)
# Modern replacement for old create_react_agent + AgentExecutor
# ========================================
from langgraph.prebuilt import create_react_agent

tools = [pdf_tool, web_search, save_answer]
agent_executor = create_react_agent(llm, tools)

tool_names = [t.name for t in tools]
print(f"✅ Agent ready with {len(tools)} tools: {tool_names}\n")

# ========================================
# Interactive Question Loop
# ========================================
print("Ask anything (type 'quit' to exit):\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit", "q"):
        print("👋 Goodbye!")
        break
    if not user_input:
        continue

    result = agent_executor.invoke({"messages": [("user", user_input)]})
    # Extract final answer from last message
    final = result["messages"][-1].content
    print(f"\n✅ Answer: {final}\n")
    print("-" * 60 + "\n")