# Financial News Intelligence System – Architecture

This document describes the complete architecture of the **Financial News Intelligence System**, including pipeline flow, agent responsibilities, databases, orchestration patterns, query processing, and data lifecycle.

---

## 🏗️ 1. High-Level Architecture

The system is composed of two major subsystems:

1. **Data Processing Pipeline** - Ingestion, embedding, deduplication, and entity extraction
2. **Query & Retrieval System** - Hybrid search and answer generation

### 1.1 Data Processing Flow

```
┌──────────────────┐
│  Ingestion Agent │  ← Fetches financial news
└────────┬─────────┘
         │ Raw Articles
         ▼
┌──────────────────┐
│ Embedding Agent  │  ← Generates vector embeddings
└────────┬─────────┘
         │ Embeddings Ready
         ▼
┌──────────────────┐
│   Dedup Agent    │  ← Clusters similar articles
└────────┬─────────┘
         │ Clustered Articles
         ▼
┌──────────────────┐
│  Entity Agent    │  ← Extracts entities & enriches
└────────┬─────────┘
         ▼
  Enriched Documents
```

### 1.2 Query Processing Flow

```
┌──────────────────┐
│   User Query     │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Extract Targets  │  ← Parse entities, topics, intent
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Hybrid Query     │  ← Keyword + Vector search
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Generate Answer  │  ← LLM synthesis with context
└────────┬─────────┘
         ▼
   Final Response
```

---

## ⚙️ 2. Orchestrator (LangGraph)

The system uses **LangGraph** for orchestrating both data processing and query workflows.

### 2.1 Data Processing Orchestrator

**Responsibilities:**
- Sequences agents: `ingest → embed → dedup → extract_entities → finalize`
- Performs **status polling** to ensure embeddings exist before deduplication
- Uses **asyncio + LangGraph** for parallel agent execution
- Maintains state graph to avoid race conditions
- Supports retry, failure propagation, and result caching

**Key Flows:**
- Ensures no step starts before previous confirms completion
- Validates expected rows exist at each stage
- Handles task-level isolation so one agent doesn't block others

### 2.2 Query Processing Orchestrator

**Implementation:**
```python
workflow = StateGraph(AgentState)

workflow.add_node("extract_targets", extract_targets)
workflow.add_node("run_hybrid_query", run_hybrid_query)
workflow.add_node("generate_final_answer", generate_final_answer)

workflow.set_entry_point("extract_targets")
workflow.add_edge("extract_targets", "run_hybrid_query")
workflow.add_edge("run_hybrid_query", "generate_final_answer")
workflow.add_edge("generate_final_answer", END)

app_graph = workflow.compile()
```

**Node Responsibilities:**
- **extract_targets**: Parses user query to identify companies, tickers, topics, time ranges
- **run_hybrid_query**: Executes combined keyword + vector similarity search
- **generate_final_answer**: Synthesizes retrieved articles into coherent response

---

## 🧩 3. Agents

### 3.1 Ingestion Agent

**Purpose:** Fetch and normalize raw financial news

**Responsibilities:**
- Fetches articles from configured sources (APIs, RSS, web scraping)
- Saves minimal metadata into SQL database
- Ensures idempotency—no duplicate article_ids
- Standardizes format (title, content, source, published_date)

**Output:** Raw articles in `raw_articles` table

### 3.2 Embedding Agent

**Purpose:** Generate semantic vector representations

**Responsibilities:**
- Uses LLM embeddings (Groq/OpenAI/Anthropic)
- Processes article title + content
- Stores vectors into `ArticleEmbedding` table
- Emits job-complete signals for orchestrator polling

**Output:** High-dimensional vectors for similarity search

### 3.3 Dedup Agent

**Purpose:** Identify and cluster duplicate/similar stories

**Responsibilities:**
- Reads embeddings from database
- Uses clustering algorithms (FAISS / cosine similarity)
- Groups articles covering same story
- Writes deduplicated results to **dedup/clusters.db**
- Exposes `raw_articles` table for downstream consumption

**Output:** Clustered articles with representative documents identified

### 3.4 Entity Agent

**Purpose:** Extract and link financial entities

**Responsibilities:**
- Performs Named Entity Recognition (NER)
- Links entities to knowledge base (companies, tickers, people)
- Extracts structured metadata:
  - Companies and stock tickers
  - Executives and key figures
  - Macro topics (earnings, M&A, regulatory, market sentiment)
  - Optional sentiment scoring
- Writes enriched articles to `entity_enriched_article` table

**Output:** Structured, searchable financial intelligence

### 3.5 Query Processing Agents

#### Extract Targets Agent
- Parses natural language queries
- Identifies search intent and constraints
- Extracts entity filters (companies, topics, date ranges)
- Structures query for hybrid search

#### Hybrid Query Agent
- Combines keyword search (SQL/Elasticsearch) with vector similarity
- Filters by entities and metadata
- Ranks and retrieves top-k relevant articles
- Returns scored and ordered results

#### Answer Generation Agent
- Receives query context and retrieved articles
- Uses LLM to synthesize comprehensive answer
- Cites sources and provides confidence indicators
- Formats response for user consumption

---

## 🗄️ 4. Databases

### 4.1 Main Database (SQLAlchemy + SQLite/PostgreSQL)

**Tables:**

**`raw_articles`**
- article_id (PK)
- title
- content
- source
- published_date
- url
- ingestion_timestamp

**`article_embedding`**
- id (PK)
- article_id (FK)
- embedding_vector (binary/array)
- model_name
- created_at

**`entity_enriched_article`**
- id (PK)
- article_id (FK)
- companies (JSON)
- tickers (JSON)
- executives (JSON)
- topics (JSON)
- sentiment_score
- enrichment_timestamp

### 4.2 Dedup Cluster Database

**Location:** `dedup/clusters.db`

**Tables:**

**`raw_articles`** (minimal schema)
- article_id
- cluster_id
- title
- embedding_ref

**`cluster_map`**
- cluster_id (PK)
- representative_article_id
- member_count
- similarity_threshold

**Purpose:** Isolates deduplication logic while remaining accessible to other agents

---

## 🔄 5. Data Lifecycle

### Processing Pipeline

```
1. Raw article ingested
   ↓
2. Metadata stored → main DB (raw_articles)
   ↓
3. Embedding generated → article_embedding
   ↓
4. Dedup agent clusters similar articles → clusters.db
   ↓
5. Entity agent enriches representative articles → entity_enriched_article
   ↓
6. Articles indexed for hybrid search
```

### Query Lifecycle

```
1. User submits natural language query
   ↓
2. Extract targets: parse entities & constraints
   ↓
3. Hybrid query: keyword + vector search
   ↓
4. Retrieve ranked articles with metadata
   ↓
5. Generate answer: LLM synthesis
   ↓
6. Return formatted response with citations
```

---

## 🪝 6. Error Handling & Stability

### Processing Pipeline Protections

- **Memory leak fix** in latency metrics collection
- **Race-condition-safe** entity matcher initialization
- **Retries** for unstable I/O operations
- **Task-level isolation** preventing cascade failures
- **Status polling** ensures dependencies complete before next stage
- **Validation layer** confirms expected data exists at each checkpoint

### Query Pipeline Protections

- **Timeout handling** for slow embeddings or LLM calls
- **Fallback strategies** when hybrid search returns insufficient results
- **Graceful degradation** if entity extraction fails
- **Rate limiting** on external API calls
- **Circuit breakers** for repeatedly failing components

---

## 📡 7. Execution Flow Examples

### Data Processing Example

1. Cron job triggers orchestrator at 6:00 AM
2. Ingestion fetches 150 new articles from sources
3. Embedding agent generates vectors in parallel
4. Dedup agent identifies 45 unique stories from 150 articles
5. Entity agent extracts companies, tickers, topics from 45 representatives
6. Final enriched dataset stored and indexed

### Query Example

**User Query:** "What are the latest earnings reports for tech companies?"

1. **Extract Targets**: 
   - Intent: earnings information
   - Industry: technology
   - Time: recent (last 30 days)
   
2. **Hybrid Query**:
   - Keyword filter: "earnings" + "report"
   - Vector similarity: tech company embeddings
   - Entity filter: companies tagged as "technology"
   
3. **Generate Answer**:
   - Retrieve top 10 articles
   - LLM synthesizes: "Recent tech earnings show mixed results. Apple reported..."
   - Include citations and dates

---

## 📐 8. Scalability Considerations

### Current Architecture
- Agents are stateless → horizontally scalable
- SQLite suitable for prototype/small deployments
- FAISS for in-memory vector search

### Scale-Up Path
- **Database**: Migrate to PostgreSQL/MySQL with connection pooling
- **Vector Store**: Upgrade to Pinecone, Weaviate, or Milvus
- **Search**: Integrate Elasticsearch for advanced text search
- **Caching**: Add Redis for query results and embeddings
- **Queue**: Use Celery/RabbitMQ for async agent tasks
- **Orchestration**: LangGraph supports DAG branching and parallel execution

### Performance Targets
- Ingestion: 1000+ articles/hour
- Query latency: < 2 seconds for hybrid search
- Answer generation: < 5 seconds end-to-end

---

## 🧭 9. Future Extensions

### Enhanced Intelligence
- **Sentiment Analysis**: Track market sentiment trends
- **Topic Modeling**: Automatic topic discovery and tracking
- **Contradiction Detection**: Flag conflicting reports across sources
- **Temporal Analysis**: Time-series trending of topics/entities

### Expanded Sources
- **Real-time Streaming**: WebSocket ingestion for live news
- **Social Media**: Twitter/Reddit financial discussions
- **SEC Filings**: Automated parsing of 10-K, 8-K reports
- **Earnings Calls**: Transcription and analysis

### Advanced Features
- **Multi-lingual Support**: Non-English financial news
- **Portfolio Integration**: Personalized alerts based on holdings
- **Predictive Signals**: ML models for market impact prediction
- **Collaborative Filtering**: User behavior-based recommendations

---

## 🔐 10. Security & Compliance

- API key management via environment variables
- Rate limiting on public endpoints
- Data retention policies (configurable TTL)
- Audit logging for all data access
- GDPR compliance for user data
- Financial data licensing compliance

---

## 📊 11. Monitoring & Observability

- **Metrics**: Ingestion rate, embedding latency, query performance
- **Logging**: Structured logs with correlation IDs
- **Alerts**: Pipeline failures, database errors, API limits
- **Dashboards**: Real-time system health visualization
- **Tracing**: Request flow through orchestrator graph

---

## 🚀 12. Deployment

### Development
```bash
python -m venv venv
pip install -r requirements.txt
python orchestrator/main.py
```

### Production
- Docker containers for each agent
- Kubernetes for orchestration
- Environment-specific configs
- Blue-green deployments
- Automated testing and CI/CD

---

**Document Version:** 2.0  
**Last Updated:** December 2025  

For questions or contributions, please refer to the project repository or contact the development team.
