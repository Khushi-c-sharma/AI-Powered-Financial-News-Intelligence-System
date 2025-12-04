# Financial News Intelligence System 📰💹

> A **production-grade Multi-Agent RAG system** that transforms high-volume financial news into actionable market intelligence with real-time stock impact predictions.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-green.svg)](https://langchain.com/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)

🎯 Overview
This system processes financial news through a sophisticated pipeline of specialized AI agents, delivering:

- **Semantic deduplication** reducing noise by 60-80%
- **Stock impact predictions** with confidence scores (direction & magnitude)
- **Sub-2-second query responses** via Groq-powered synthesis
- **Multi-model entity extraction** achieving >90% precision
- **Hybrid ranking** combining semantic relevance with market impact

### Core Capabilities

- ✅ **Multi-agent orchestration** using LangGraph
- ✅ **Semantic deduplication** with clustering (60-80% noise reduction)
- ✅ **Named entity extraction** with >90% precision
- ✅ **Stock impact predictions** (direction, magnitude, confidence)
- ✅ **Hybrid search** combining semantic + impact scoring
- ✅ **Async orchestration** with reliable state polling
- ✅ **Pipeline safety** (race-condition fixes, memory leak patches, strong state management)
- ✅ **Database persistence** using SQLAlchemy + async SQLite
- ✅ **Sub-2-second query responses** via Groq-powered synthesis

---

## 🧩 Architecture

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LangGraph Orchestrator                           │
│                    (State Management & Workflow)                        │
└───┬─────────────┬──────────────┬─────────────┬──────────────┬──────────┘
    │             │              │             │              │
    ▼             ▼              ▼             ▼              ▼
┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐  ┌───────────┐
│Ingestion│  │Embedding │  │  Dedup  │  │ Entity   │  │  Stock    │
│ Agent   │→ │ Service  │→ │ Agent   │→ │Extraction│→ │  Impact   │
│ (8000)  │  │ (8002)   │  │ (8003)  │  │  (8004)  │  │  (8005)   │
└─────────┘  └──────────┘  └─────────┘  └──────────┘  └───────────┘
     │            │              │             │              │
     └────────────┴──────────────┴─────────────┴──────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Storage & Indexing   │
                    │      Agent (8006)      │
                    └────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │    Query Agent (8007)  │
                    │  + Orchestrator (8008) │
                    └────────────────────────┘
```

### Agent Pipeline Flow

**1. Ingestion Agent (Port 8000)**
- Fetches raw articles from RSS/API feeds
- Stores metadata and content in `raw_articles` table
- **Output**: `data/ingestion/raw_articles.db`

**2. Embedding Service (Port 8002)**
- Generates vector embeddings using SentenceTransformers
- Builds FAISS index with persistent ID mapping
- **Output**: `data/embeddings/vectors.db`, `faiss.index`, `faiss_id_map.json`
- **Status**: Marks articles as `processed`

**3. Deduplication Agent (Port 8003)**
- Performs semantic clustering (cosine similarity ≥0.88)
- Groups duplicate articles into story clusters
- **Output**: `data/dedup/clusters.db` (`story_clusters` table)
- **Blocking**: Waits for embedding completion

**4. Entity Extraction Agent (Port 8004)**
- Multi-model NER (primary + secondary models)
- Maps entities to stock symbols with confidence scores
- **Output**: `data/entities/entities.db` (`article_entities`, `stock_impacts`)
- **Blocking**: Processes articles after dedup

**5. Stock Impact Agent (Port 8005)**
- Sentiment analysis + event classification
- Predicts direction (up/down/neutral) and magnitude (0-1)
- **Output**: `article_stock_effects` table
- **Models**: FinBERT, Zero-shot classification

**6. Storage & Indexing Agent (Port 8006)**
- Consolidates stories from clusters
- Groq-powered summarization (Llama 3.3 70B)
- Builds final FAISS index for search
- **Output**: Canonical stories with summaries

**7. Query Agent (Port 8007)**
- Hybrid search (semantic + entity + impact)
- Groq reranking for top results
- **Response time**: 1-2s with reranking, 300-600ms without

**8. Orchestrator Service (Port 8008)**
- LangGraph-based workflow coordination
- Intent parsing and query routing
- Final answer synthesis

---

## 📂 Repository Structure

```
tradl-hackathon/
├── agents/
│   ├── ingestion_agent.py          # News ingestion (RSS/API)
│   ├── embedding_agent.py          # Vector generation + FAISS
│   ├── dedup_agent.py              # Clustering + deduplication
│   ├── entity_extraction_agent.py  # NER + stock mapping
│   ├── stock_impact_agent.py       # Sentiment + impact analysis
│   ├── storage_indexing_agent.py   # Story consolidation + Groq summarization
│   ├── query_agent.py              # Hybrid search + reranking
│   └── orchestrator_service.py     # LangGraph coordination
│
├── orchestrator/
│   └── main.py                     # Main LangGraph workflow
│
├── data/
│   ├── ingestion/
│   │   └── raw_articles.db         # Raw news articles
│   ├── embeddings/
│   │   ├── vectors.db              # Article embeddings
│   │   ├── faiss.index             # FAISS vector index
│   │   └── faiss_id_map.json       # ID mapping
│   ├── dedup/
│   │   └── clusters.db             # Story clusters
│   ├── entities/
│   │   └── entities.db             # Entities + stock impacts
│   ├── mappings/
│   │   ├── stock_mapping.json      # Company → Symbol mapping
│   │   └── sector_mapping.json     # Sector → Stocks mapping
│   └── storage/
│       └── stories.db              # Canonical stories
│
├── config/
│   └── sources.json                        
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🛠️ Key Features & Fixes

### ✓ Production-Ready Improvements

| Feature | Description | Status |
|---------|-------------|--------|
| **Stronger State Polling** | Orchestrator blocks until embedding completion before dedup | ✅ Fixed |
| **Database Model Unification** | Consistent schema across all agents | ✅ Fixed |
| **Race-Condition Fix** | Thread-safe initialization for entity matcher | ✅ Fixed |
| **Memory Leak Patch** | `query_latencies` bounded to prevent unbounded growth | ✅ Fixed |
| **Unicode Encoding** | Windows console UTF-8 support (no checkmark crashes) | ✅ Fixed |
| **FAISS ID Persistence** | ID map saved/loaded with index | ✅ Fixed |
| **Groq Fallback Logic** | Graceful degradation when rate-limited | ✅ Fixed |
| **Connection Pooling** | Async HTTP client with keep-alive (20-100 connections) | ✅ Added |
| **LRU Caching** | Entity extraction cached (1000 entries) | ✅ Added |
| **Incremental Indexing** | Only process new/updated stories | ✅ Added |

### Pipeline Safety Features

- **Async-safe operations** throughout
- **Database transactions** with rollback on failure
- **Retry logic** with exponential backoff (3 retries default)
- **Error isolation** (one agent failure doesn't crash pipeline)
- **Health checks** for all services
- **Diagnostic endpoints** for debugging

---

## 🔧 Requirements

### System Requirements
- **Python 3.10+**
- **4GB+ RAM** (8GB+ recommended)
- **2GB+ disk space** for models and data
- **Internet connection** for model downloads and Groq API

### Core Dependencies
```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
langgraph>=0.0.40
sqlalchemy>=2.0.0
aiosqlite>=0.19.0
httpx>=0.25.0
pydantic>=2.0.0
python-dotenv>=1.0.0
```

### AI/ML Dependencies
```
transformers>=4.35.0
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4  # or faiss-gpu for CUDA
groq>=0.4.0
torch>=2.1.0
```

### Data Processing
```
feedparser>=6.0.10
beautifulsoup4>=4.12.0
aiohttp>=3.9.0
numpy>=1.24.0
pandas>=2.0.0
```

### Install All Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Create `.env` file in project root:

```env
# ============================================
# REQUIRED CONFIGURATION
# ============================================

# Groq API (get from https://console.groq.com)
GROQ_API_KEY=gsk_your_api_key_here

# ============================================
# AGENT ENDPOINTS
# ============================================

STORAGE_BASE=http://localhost:8006
INGESTION_BASE=http://localhost:8000

# ============================================
# MODEL CONFIGURATION
# ============================================

# Embedding Models
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# NER Models (Entity Extraction)
PRIMARY_MODEL=Jean-Baptiste/roberta-large-ner-english
SECONDARY_MODEL=dslim/bert-base-NER-uncased

# Groq Models
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_MAX_RETRIES=3
GROQ_TIMEOUT=30

# ============================================
# PERFORMANCE TUNING
# ============================================

# Batch Processing
BATCH_SIZE=64
MAX_CANDIDATES=50
TOP_K_RETURN=10

# Deduplication
SENT_SIM_THRESHOLD=0.88

# Scoring Weights
HYBRID_ENTITY_BOOST=0.15
HYBRID_IMPACT_BOOST=0.25
RERANK_WEIGHT=0.55
HYBRID_WEIGHT=0.45

# Caching
CACHE_TTL_SECONDS=300
RATE_LIMIT_PER_MINUTE=100

# ============================================
# OPTIONAL: REDIS CACHE
# ============================================

# REDIS_URL=redis://localhost:6379
```

---

## ▶️ Running the System

### Option 1: Individual Services (Development)

Start each agent in a separate terminal:

```bash
# Terminal 1 - Ingestion
uvicorn agents.ingestion_agent:app --reload --port 8000

# Terminal 2 - Embedding
uvicorn agents.embedding_agent:app --reload --port 8002

# Terminal 3 - Deduplication
uvicorn agents.dedup_agent:app --reload --port 8003

# Terminal 4 - Entity Extraction
uvicorn agents.entity_extraction_agent:app --reload --port 8004

# Terminal 5 - Stock Impact
uvicorn agents.stock_impact_agent:app --reload --port 8005

# Terminal 6 - Storage & Indexing
uvicorn agents.storage_indexing_agent:app --reload --port 8006

# Terminal 7 - Query Agent
uvicorn agents.query_agent:app --reload --port 8007

# Terminal 8 - Orchestrator
uvicorn agents.orchestrator_service:app --reload --port 8008
```

### Option 2: Using the Orchestrator

The LangGraph orchestrator manages the entire pipeline:

```bash
python orchestrator/main.py
```

This will:
1. Start the multi-agent pipeline
2. Poll agent states
3. Ensure proper ordering (embedding → dedup → entity extraction → impact)
4. Process financial news in real-time

---

## 🚦 Initial Data Pipeline

Run these steps in order for first-time setup:

### Step 1: Ingest Articles

```powershell
# Load RSS feeds from sources.json
$sources = Get-Content "sources.json" | ConvertFrom-Json

foreach ($source in $sources.sources) {
    $body = @{
        feed_url    = $source.url
        source_name = $source.name
        category    = $source.category
    } | ConvertTo-Json
    
    Invoke-RestMethod -Uri "http://localhost:8000/ingest/rss" `
        -Method Post -ContentType "application/json" -Body $body
    
    Start-Sleep -Seconds 1
}
```

### Step 2: Generate Embeddings

```powershell
# Process all pending articles (wait for completion)
Invoke-RestMethod -Uri "http://localhost:8002/embed/process?limit=200" -Method Post
```

**✅ Checkpoint**: Verify embeddings created
```powershell
Invoke-RestMethod -Uri "http://localhost:8002/embed/stats"
# Should show: total_embeddings > 0
```

### Step 3: Run Deduplication

```powershell
# Cluster duplicate articles (blocks until embedding complete)
Invoke-RestMethod -Uri "http://localhost:8003/dedup/run" `
    -Method Post -ContentType "application/json" `
    -Body '{"force_rebuild": false, "similarity_threshold": 0.88}'
```

**✅ Checkpoint**: Verify clusters created
```powershell
Invoke-RestMethod -Uri "http://localhost:8003/dedup/stats"
# Should show: total_clusters > 0
```

### Step 4: Extract Entities & Map Stocks

```powershell
# Extract entities and generate stock impacts
Invoke-RestMethod -Uri "http://localhost:8004/entities/extract" `
    -Method Post -ContentType "application/json" `
    -Body '{"limit": 200}'
```

**✅ Checkpoint**: Verify entities extracted
```powershell
Invoke-RestMethod -Uri "http://localhost:8004/entities/stats"
# Should show: total_entities > 0, total_stock_impacts > 0
```

### Step 5: Analyze Stock Impacts

```powershell
# Run sentiment & impact prediction
Invoke-RestMethod -Uri "http://localhost:8005/impact/run" `
    -Method Post -ContentType "application/json" `
    -Body '{"limit": 200}'
```

### Step 6: Build Search Index

```powershell
# Create enhanced index with Groq summarization
$body = @{
    rebuild_index = $false
    enhanced      = $true
    limit         = $null
    incremental   = $true
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8006/index/all" `
    -Method Post -ContentType "application/json" -Body $body
```

**✅ Checkpoint**: Verify index built
```powershell
Invoke-RestMethod -Uri "http://localhost:8006/diagnostics"
# Should show: faiss_index_loaded: true, vector_count > 0
```

---

## 🔍 Querying the System

### Web Interface

Open browser to: **http://localhost:8008**

### API Examples

**Basic Query:**
```powershell
$body = @{ query = "Latest HDFC Bank dividend announcements" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8008/query" `
    -Method Post -ContentType "application/json" -Body $body
```

**Company-Specific Query:**
```powershell
$body = @{
    query = "TCS cloud business growth Q3 2024"
    k = 10
    rerank_with_groq = $true
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8007/query" `
    -Method Post -ContentType "application/json" -Body $body
```

**Portfolio Monitoring:**
```powershell
$body = @{
    query = "Recent developments affecting my holdings"
    portfolio = @("HDFCBANK", "TCS", "INFY", "RELIANCE", "ICICIBANK")
    k = 20
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8007/query" `
    -Method Post -ContentType "application/json" -Body $body
```

### Response Format

```json
{
  "query": "HDFC Bank dividend news",
  "returned": 5,
  "processing_time_ms": 1247.3,
  "results": [
    {
      "story_id": "abc-123",
      "canonical_title": "HDFC Bank announces 15% dividend increase",
      "snippet": "HDFC Bank's board approved...",
      "final_score": 0.95,
      "vector_score": 0.82,
      "hybrid_score": 0.87,
      "rerank_score": 0.98,
      "impacts": [
        {
          "symbol": "HDFCBANK",
          "name": "HDFC Bank Ltd",
          "confidence": 1.0,
          "type": "direct",
          "direction": "up",
          "magnitude": 0.75,
          "reasoning": "Direct mention of HDFC Bank"
        }
      ],
      "entities": {
        "companies": ["HDFC Bank"],
        "sectors": ["Banking", "Financial Services"]
      },
      "reason": "Direct company mention with positive corporate action"
    }
  ],
  "debug": {
    "companies": ["HDFC Bank"],
    "sectors": ["Banking"],
    "regulators": [],
    "events": ["dividend"],
    "themes": []
  },
  "cached": false
}
```

---

## 📊 API Documentation

Interactive API docs for each service:

| Service | Interactive Docs | Description |
|---------|-----------------|-------------|
| Ingestion | http://localhost:8000/docs | RSS/API feed ingestion |
| Embedding | http://localhost:8002/docs | Vector generation + FAISS |
| Deduplication | http://localhost:8003/docs | Story clustering |
| Entity Extraction | http://localhost:8004/docs | NER + stock mapping |
| Stock Impact | http://localhost:8005/docs | Sentiment + prediction |
| Storage & Indexing | http://localhost:8006/docs | Story consolidation |
| Query Agent | http://localhost:8007/docs | Hybrid search + reranking |
| Orchestrator | http://localhost:8008/docs | LangGraph coordination |

---

## 🐛 Troubleshooting

### Common Issues

**1. Unicode Encoding Errors (Windows)**
```powershell
# Already fixed in code, but if issues persist:
chcp 65001  # Set console to UTF-8
```

**2. Model Download Failures**
```bash
# Pre-download models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

**3. FAISS Index Corruption**
```powershell
# Rebuild index
Invoke-RestMethod -Uri "http://localhost:8002/embed/rebuild-faiss" -Method Post
```

**4. Groq Rate Limits**
```env
# Reduce concurrent requests in .env
MAX_CANDIDATES=30
GROQ_RERANK_MAX_CANDIDATES=20
GROQ_MAX_RETRIES=5
```

**5. Memory Issues**
```env
# Reduce batch sizes
BATCH_SIZE=32
MAX_CANDIDATES=30
```

### Diagnostic Commands

```powershell
# Check all services
foreach ($port in 8000,8002,8003,8004,8005,8006,8007,8008) {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:$port/health"
        Write-Host "Port $port`: " -NoNewline -ForegroundColor Green
        Write-Host $response.status
    } catch {
        Write-Host "Port $port`: OFFLINE" -ForegroundColor Red
    }
}
```

---

## 📈 Performance Benchmarks

Tested on: Intel i7-11th Gen, 16GB RAM, Windows 11

| Operation | Latency | Throughput | Notes |
|-----------|---------|------------|-------|
| Article Ingestion | 200-500ms | 5-10/sec | With HTML parsing |
| Embedding Generation | 50-100ms | 10-20/sec | Batch size 64 |
| Deduplication | 2-5s | 100-200 articles/run | FAISS k-NN search |
| Entity Extraction | 150-300ms | 3-7/sec | Multi-model NER |
| Stock Impact | 100-200ms | 5-10/sec | Sentiment + events |
| Query (no rerank) | 300-600ms | 2-3/sec | Vector + hybrid |
| Query (with rerank) | 1-2s | 0.5-1/sec | + Groq reranking |
| Full Pipeline | 30-60s | 100 articles | End-to-end |

---

## 🗺️ Roadmap

### Completed ✅
- [x] Multi-agent orchestration with LangGraph
- [x] Production-ready embedding pipeline
- [x] Semantic deduplication with FAISS
- [x] Multi-model entity extraction
- [x] Stock impact prediction
- [x] Hybrid search with reranking
- [x] Race-condition and memory leak fixes
- [x] Comprehensive error handling

### In Progress 🚧
- [ ] Real-time streaming ingestion (WebSocket)
- [ ] Advanced visualization dashboard
- [ ] Historical trend analysis
- [ ] ML-based impact prediction (beyond heuristics)

### Future Enhancements 🔮
- [ ] Multi-language support (Hindi, Chinese, Spanish)
- [ ] Portfolio backtesting integration
- [ ] Mobile app (iOS/Android)
- [ ] Slack/Discord bot interface
- [ ] Integration with trading platforms
- [ ] Advanced alert system

---

**Built with ❤️ for the Tradl Hackathon**

*If you find this project useful, please star ⭐ the repository!*

