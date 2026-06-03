# Agentic AI Research Assistant (RAG + ReAct) — Gemini Edition

A minimal (~90 lines) LangChain project combining **RAG** and a **ReAct Agent**.
Upload any PDF — the agent decides whether to search the document, the web, or both.
Powered by **Google Gemini** (free API available!).

## Quick Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API keys
cp .env.example .env
# Edit .env and fill in GOOGLE_API_KEY and TAVILY_API_KEY

# 4. Run the assistant
python main.py
```

## Get API Keys (Both FREE)
- **Google Gemini**: https://aistudio.google.com/app/apikey  ← free tier available
- **Tavily**: https://app.tavily.com  ← free tier: 1000 searches/month

## What Changed from OpenAI Version
| Component   | OpenAI (paid)         | Gemini (free tier)                     |
|-------------|-----------------------|----------------------------------------|
| LLM         | `ChatOpenAI` gpt-4o   | `ChatGoogleGenerativeAI` gemini-1.5-pro|
| Embeddings  | `OpenAIEmbeddings`    | `GoogleGenerativeAIEmbeddings`         |
| Env var     | `OPENAI_API_KEY`      | `GOOGLE_API_KEY`                       |
| Package     | `langchain-openai`    | `langchain-google-genai`               |

## Sample Interaction

```
Enter PDF path: company_report.pdf
✅ PDF loaded: 24 pages
✅ Split into 47 chunks
✅ Vector store ready (FAISS)
✅ Agent ready with 3 tools: [pdf_knowledge_base, web_search, save_answer]

Ask anything (type 'quit' to exit):

You: What does the document say about revenue growth?
✅ Answer: Revenue grew 34% YoY to $12.4M...

You: How does that compare to industry averages?
✅ Answer: Your 34% growth outperforms the 20-25% industry average...

You: quit
👋 Goodbye!
```

## LangChain Concepts Covered
`ChatGoogleGenerativeAI` · `PyPDFLoader` · `RecursiveCharacterTextSplitter`
`GoogleGenerativeAIEmbeddings` · `FAISS` · `as_retriever()` · `create_retriever_tool`
`TavilySearchResults` · `@tool decorator` · `hub.pull()` · `create_react_agent` · `AgentExecutor`