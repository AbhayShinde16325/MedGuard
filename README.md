# 📊 MarketMinds

**An AI-powered financial chatbot that answers complex market and company-related questions by combining real financial data with company reports using Retrieval-Augmented Generation (RAG).**

Think of it as a junior financial analyst that never gets tired.

---

## 🚀 What It Does

MarketMinds allows users to ask natural language questions like:

- *"How did Apple perform in its last earnings call?"*
- *"Summarize the key risks mentioned in Tesla's annual report."*
- *"Compare Infosys and TCS based on recent financials."*

The system intelligently decides whether to:

- Fetch live/structured market data
- Search company reports (PDFs)
- Combine both to generate a grounded, context-aware response

---

## 🧠 Architecture

```
User Query
   ↓
Intent Detection (API / Reports / Hybrid)
   ↓
Data Retrieval (Stock API or Vector DB)
   ↓
LLM + Context (RAG)
   ↓
Final Answer
```

The system uses:

- **Intent Router** - Determines if query needs market data, reports, or both
- **LLM Client** - Ollama-based (Mistral) for local, privacy-respecting inference
- **RAG Pipeline** - Vector embeddings + document retrieval for company reports
- **Financial Data Source** - API integration for real-time stock/market data

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.8+ |
| **Backend Framework** | FastAPI |
| **LLM** | Ollama (Mistral) |
| **Embeddings** | Sentence Transformers |
| **Vector Store** | Local (Chroma/FAISS) |
| **Frontend** | HTML/CSS/JavaScript |
| **Document Processing** | PyPDF2, LangChain |

---

## 📁 Project Structure

```
marketminds-chatbot/
├── backend/
│   └── app/
│       ├── main.py                 # FastAPI entry point
│       ├── config.py               # Configuration management
│       ├── chatbot/
│       │   ├── response_builder.py # RAG pipeline orchestration
│       │   └── router.py           # Intent routing logic
│       ├── data_sources/
│       │   └── financial_api.py    # External data integrations
│       ├── llm/
│       │   ├── llm_client.py       # Ollama client
│       │   └── prompt_templates.py # LLM prompts
│       ├── rag/
│       │   ├── embeddings.py       # Embedding generation
│       │   ├── ingest.py           # Document ingestion
│       │   └── retriever.py        # Vector search & retrieval
│       └── utils/
│           ├── pdf_utils.py        # PDF parsing
│           └── text_utils.py       # Text processing
├── frontend/
│   ├── index.html                  # Chat interface
│   ├── app.js                      # Frontend logic
│   └── style.css                   # Styling
├── data/
│   ├── raw/                        # Raw documents
│   ├── processed/                  # Processed data (CSVs)
│   └── vector_store/               # Vector embeddings store
├── tests/
│   └── test_basic_flow.py          # Integration tests
└── notebooks/
    └── experiments.ipynb           # Development & prototyping
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai) installed and running locally
- pip package manager

### 1. Clone & Navigate

```bash
git clone <your-repo-url>
cd marketminds-chatbot
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Ollama

```bash
# Start Ollama service (runs on localhost:11434)
ollama serve

# In another terminal, pull the Mistral model
ollama pull mistral
```

### 5. Prepare Data

Place earnings reports and company documents in `data/raw/` directory.

Run the ingestion pipeline:

```bash
python backend/app/rag/ingest.py
```

This will:
- Process PDFs
- Generate embeddings
- Store vectors in `data/vector_store/`

---

## 🏃 Running the Application

### Start the Backend

```bash
cd marketminds-chatbot
uvicorn backend.app.main:app --reload
```

The API will be available at `http://localhost:8000`

- **Chat endpoint**: `POST /query`
- **Health check**: `GET /health`
- **API docs**: `http://localhost:8000/docs`

### Access the Frontend

Open your browser and navigate to:

```
http://localhost:8000
```

---

## 📖 Usage Examples

### Via Chat Interface

Simply type your questions:

```
"What were Tesla's revenue and profit margins last quarter?"
"Which tech stocks are mentioned positively in recent earnings?"
"Summarize the risks disclosed by Meta"
```

### Via API

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "How did Apple perform in Q3?"}'
```

Response:

```json
{
  "answer": "Based on Apple's Q3 earnings report..."
}
```

---

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

---

## 🎯 Why MarketMinds?

Most financial tools analyze one data source at a time. **MarketMinds**:

✅ Combines structured + unstructured financial data  
✅ Performs cross-source reasoning  
✅ Abstracts complex workflows into one simple question  
✅ Mirrors how real financial analysts think  

---

## 📌 Current Status

- ✅ Core architecture designed & implemented
- ✅ FastAPI backend with RAG pipeline
- ✅ Ollama LLM integration
- ✅ Vector store setup
- ✅ Basic chat interface
- 🔄 In development: Enhanced intent detection, multi-company analysis

---

## 🔮 Future Improvements

- 📰 News integration & sentiment analysis
- 🔄 Multi-company comparisons
- 📊 Advanced financial ratio analysis
- 🤖 Multi-agent architecture for complex queries
- 📱 Mobile-friendly interface
- 🔐 User authentication & session management

---

## 🛠️ Development

### Code Structure

- **`chatbot/`** - Core reasoning and response generation
- **`data_sources/`** - External data integrations
- **`llm/`** - Language model interface
- **`rag/`** - Retrieval-augmented generation pipeline
- **`utils/`** - Helper utilities

### Adding New Features

1. Create feature branch: `git checkout -b feature/your-feature`
2. Implement in appropriate module
3. Add tests in `tests/`
4. Submit pull request

---

## 📝 Configuration

Edit `backend/app/config.py` to customize:

- Model selection (default: Mistral)
- Data directories
- API endpoints
- Debug mode

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Ollama connection refused | Ensure Ollama is running: `ollama serve` |
| Model not found | Pull model: `ollama pull mistral` |
| Port 8000 in use | Change port: `uvicorn ... --port 8001` |
| Vector store empty | Run: `python backend/app/rag/ingest.py` |

---

## 📄 License

This project is open source and available under the MIT License.

---

## 👤 Author

**Abhay Shinde**  
Computer Engineering Student | Data & AI Enthusiast

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:

- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

---

## 📚 Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Ollama Models](https://ollama.ai/library)
- [LangChain Documentation](https://langchain.readthedocs.io)
- [Vector Search Basics](https://www.pinecone.io/learn/vector-search)

---

**Built with ❤️ for financial data exploration**
