"""
Embedding Agent (Optimized & Fixed)
----------------------------------------
This service:
- Loads new articles from ingestion DB.
- Generates embeddings using SentenceTransformers.
- Stores vectors in SQLite + FAISS.
- Supports FAISS persistence with ID mapping.
- Uses batch embedding for performance.
- Uses pathlib for clean folder structure.
- Includes detailed docstrings & comments.

Folder structure expected:
project_root/
    agents/
        embedding_agent.py (this file)
        ingestion_agent.py
    data/
        ingestion/raw_articles.db
        embeddings/vectors.db
        embeddings/faiss.index
        embeddings/faiss_id_map.json

Run:
uvicorn agents.embedding_agent:app --reload --port 8002
"""

import uuid
import json
import faiss
import logging
import numpy as np
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import select, Column, String, Text, TIMESTAMP, Integer, LargeBinary, func, Index
from sqlalchemy.pool import StaticPool

from sentence_transformers import SentenceTransformer
import uvicorn

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EmbeddingAgent")

Base = declarative_base()

# ---------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INGESTION_DB = DATA_DIR / "ingestion" / "raw_articles.db"
EMBED_DIR = DATA_DIR / "embeddings"
EMBED_DB = EMBED_DIR / "vectors.db"
FAISS_INDEX_PATH = EMBED_DIR / "faiss.index"
FAISS_ID_MAP_PATH = EMBED_DIR / "faiss_id_map.json"

EMBED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
import os
AUTO_PROCESS_ON_STARTUP = os.getenv("AUTO_PROCESS_ON_STARTUP", "false").lower() == "true"
AUTO_PROCESS_BATCH_SIZE = int(os.getenv("AUTO_PROCESS_BATCH_SIZE", "50"))

# ---------------------------------------------------------
# DB ENGINE (with connection pooling for SQLite)
# ---------------------------------------------------------
# Embedding database - stores embeddings
EMBEDDING_DATABASE_URL = f"sqlite+aiosqlite:///{EMBED_DB}"
embedding_engine = create_async_engine(
    EMBEDDING_DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
EmbeddingSessionLocal = sessionmaker(embedding_engine, class_=AsyncSession, expire_on_commit=False)

# Ingestion database - reads raw articles
INGESTION_DATABASE_URL = f"sqlite+aiosqlite:///{INGESTION_DB}"
ingestion_engine = create_async_engine(
    INGESTION_DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
IngestionSessionLocal = sessionmaker(ingestion_engine, class_=AsyncSession, expire_on_commit=False)

# ---------------------------------------------------------
# TIME HELPER
# ---------------------------------------------------------
def now():
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)

# ---------------------------------------------------------
# DATABASE MODELS
# ---------------------------------------------------------
class RawArticle(Base):
    """Raw articles inserted by ingestion agent that require embedding."""
    __tablename__ = "raw_articles"
    __table_args__ = (
        Index('idx_status_article', 'processing_status', 'article_id'),
    )

    id = Column(String, primary_key=True)
    article_id = Column(String, unique=True, index=True)
    title = Column(Text)
    content = Column(Text)
    source = Column(String)
    url = Column(String)
    category = Column(String)
    published_date = Column(TIMESTAMP(timezone=True))
    ingested_date = Column(TIMESTAMP(timezone=True))
    processing_status = Column(String, default="pending", index=True)


class ArticleEmbedding(Base):
    """Stores embedding vectors for articles."""
    __tablename__ = "article_embeddings"

    id = Column(String, primary_key=True)
    article_id = Column(String, unique=True, index=True)  # Enforce uniqueness
    vector = Column(LargeBinary)  # vector bytes
    model = Column(String)
    created_at = Column(TIMESTAMP(timezone=True), default=now)
    meta_data = Column(Text)  # Renamed from 'metadata' to avoid SQLAlchemy conflict

# ---------------------------------------------------------
# MODEL (loaded in lifespan)
# ---------------------------------------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = None
EMBED_DIM = None

# ---------------------------------------------------------
# FAISS (With Persistence for Index AND ID Mapping)
# ---------------------------------------------------------
faiss_id_map: Dict[int, str] = {}

def load_faiss_index():
    """Load FAISS index and ID mapping from disk."""
    global faiss_index, faiss_id_map
    
    if FAISS_INDEX_PATH.exists():
        logger.info("Loading existing FAISS index...")
        faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
        
        # Load ID mapping
        if FAISS_ID_MAP_PATH.exists():
            with open(FAISS_ID_MAP_PATH, 'r') as f:
                # JSON keys are strings, convert back to int
                loaded_map = json.load(f)
                faiss_id_map = {int(k): v for k, v in loaded_map.items()}
            logger.info(f"Loaded {len(faiss_id_map)} ID mappings")
        else:
            logger.warning("FAISS index exists but ID map is missing!")
    else:
        faiss_index = faiss.IndexFlatL2(EMBED_DIM)
        logger.info("Created new FAISS index")

def save_faiss_index():
    """Save FAISS index and ID mapping to disk."""
    faiss.write_index(faiss_index, str(FAISS_INDEX_PATH))
    
    # Save ID mapping as JSON
    with open(FAISS_ID_MAP_PATH, 'w') as f:
        # Convert int keys to strings for JSON
        json.dump({str(k): v for k, v in faiss_id_map.items()}, f)
    
    logger.info(f"Saved FAISS index with {faiss_index.ntotal} vectors and {len(faiss_id_map)} mappings")

# FAISS will be loaded in lifespan
faiss_index = None

# ---------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------
class EmbeddingResult(BaseModel):
    """Result summary after embedding batch."""
    processed: int
    failed: int
    pending_left: int
    skipped_existing: int = 0

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def embed_batch(texts: List[str]) -> np.ndarray:
    """Embed a batch of texts using SentenceTransformers efficiently."""
    logger.info(f"Encoding {len(texts)} texts with batch_size=32...")
    return model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=True  # Enable progress bar to see it's working
    ).astype("float32")

# ---------------------------------------------------------
# FastAPI Lifespan
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB tables, load model, and FAISS on startup."""
    global model, EMBED_DIM, faiss_index
    
    logger.info("=" * 60)
    logger.info("STARTING EMBEDDING AGENT INITIALIZATION")
    logger.info("=" * 60)
    
    # Step 1: Initialize database
    logger.info("Step 1/3: Initializing database...")
    async with embedding_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✓ Database initialized")
    
    # Step 2: Load SentenceTransformer model
    logger.info(f"Step 2/3: Loading model: {MODEL_NAME}")
    logger.info("NOTE: First run may take several minutes to download model...")
    try:
        model = SentenceTransformer(MODEL_NAME)
        EMBED_DIM = model.get_sentence_embedding_dimension()
        logger.info(f"✓ Model loaded successfully (dimension: {EMBED_DIM})")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        raise
    
    # Step 3: Load FAISS index
    logger.info("Step 3/3: Loading FAISS index...")
    try:
        load_faiss_index()
        logger.info(f"✓ FAISS loaded ({faiss_index.ntotal} vectors, {len(faiss_id_map)} mappings)")
    except Exception as e:
        logger.error(f"✗ Failed to load FAISS: {e}")
        raise
    
    logger.info("=" * 60)
    logger.info("✓ EMBEDDING AGENT READY TO ACCEPT REQUESTS")
    logger.info("=" * 60)
    
    # Optional: Auto-process on startup
    if AUTO_PROCESS_ON_STARTUP:
        logger.info(f"AUTO_PROCESS_ON_STARTUP enabled, processing {AUTO_PROCESS_BATCH_SIZE} articles...")
        try:
            # Process embeddings directly - use BOTH databases
            async with IngestionSessionLocal() as ingest_db, EmbeddingSessionLocal() as embed_db:
                result = await ingest_db.execute(
                    select(RawArticle)
                    .outerjoin(ArticleEmbedding, RawArticle.article_id == ArticleEmbedding.article_id)
                    .where(RawArticle.processing_status == "pending")
                    .where(ArticleEmbedding.article_id.is_(None))
                    .limit(AUTO_PROCESS_BATCH_SIZE)
                )
                pending = result.scalars().all()
                logger.info(f"Found {len(pending)} pending articles for auto-processing")
                
                if pending:
                    # Simple inline processing
                    contents = [a.content for a in pending]
                    vectors = embed_batch(contents)
                    
                    vectors_array = np.vstack([vectors[i] for i in range(len(vectors))])
                    start_id = faiss_index.ntotal
                    faiss_index.add(vectors_array)
                    
                    for i, article in enumerate(pending):
                        vec = vectors[i]
                        faiss_id_map[start_id + i] = article.article_id
                        
                        emb = ArticleEmbedding(
                            id=str(uuid.uuid4()),
                            article_id=article.article_id,
                            vector=vec.tobytes(),
                            model=MODEL_NAME,
                            meta_data=json.dumps({"source": article.source})
                        )
                        embed_db.add(emb)
                        article.processing_status = "processed"
                    
                    save_faiss_index()
                    await embed_db.commit()
                    await ingest_db.commit()
                    logger.info(f"✓ Auto-processed {len(pending)} articles on startup")
                else:
                    logger.info("No pending articles to auto-process")
        except Exception as e:
            logger.error(f"Auto-processing failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down: Saving FAISS index...")
    save_faiss_index()
    logger.info("✓ Shutdown complete")

app = FastAPI(
    title="Embedding Agent (Optimized)",
    version="2.0.0",
    lifespan=lifespan
)

# ---------------------------------------------------------
# ENDPOINT: View Pending Articles
# ---------------------------------------------------------
@app.get("/embed/pending")
async def get_pending(limit: int = 10):
    """
    Return up to `limit` pending articles waiting for embedding.
    Uses efficient LEFT JOIN to exclude already embedded articles.
    """
    async with IngestionSessionLocal() as ingest_db, EmbeddingSessionLocal() as embed_db:
        # Get article IDs that already have embeddings
        embedded_result = await embed_db.execute(
            select(ArticleEmbedding.article_id)
        )
        embedded_ids = {row[0] for row in embedded_result.all()}
        
        # Get pending articles from ingestion DB
        result = await ingest_db.execute(
            select(RawArticle)
            .where(RawArticle.processing_status == "pending")
            .limit(limit * 2)  # Get more to filter
        )
        all_pending = result.scalars().all()
        
        # Filter out already embedded
        articles = [a for a in all_pending if a.article_id not in embedded_ids][:limit]

    return {
        "pending": len(articles),
        "articles": [
            {"id": a.article_id, "title": a.title, "source": a.source}
            for a in articles
        ]
    }

# ---------------------------------------------------------
# MAIN PROCESSOR: Batch Embeddings with Individual Error Handling
# ---------------------------------------------------------
@app.post("/embed/process", response_model=EmbeddingResult)
async def process_embeddings(limit: int = 20):
    """
    Fetch pending articles, embed them in batches, write embeddings to DB,
    update FAISS index, and update article status.
    
    Reads from ingestion DB, writes to embedding DB.
    """
    logger.info(f"Starting embedding process (limit: {limit})...")
    
    async with IngestionSessionLocal() as ingest_db, EmbeddingSessionLocal() as embed_db:
        # Get already embedded article IDs
        logger.info("Checking for already embedded articles...")
        embedded_result = await embed_db.execute(
            select(ArticleEmbedding.article_id)
        )
        embedded_ids = {row[0] for row in embedded_result.all()}
        logger.info(f"Found {len(embedded_ids)} already embedded articles")
        
        # Query pending articles from ingestion DB
        logger.info("Querying for pending articles from ingestion DB...")
        result = await ingest_db.execute(
            select(RawArticle)
            .where(RawArticle.processing_status == "pending")
            .limit(limit * 2)  # Get more to account for already embedded
        )
        all_pending = result.scalars().all()
        
        # Filter out already embedded articles
        pending = [a for a in all_pending if a.article_id not in embedded_ids][:limit]
        
        logger.info(f"Found {len(pending)} pending articles to process (out of {len(all_pending)} total pending)")

        if not pending:
            logger.info("No pending articles found")
            return EmbeddingResult(processed=0, failed=0, pending_left=0)

        processed, failed, skipped = 0, 0, 0

        # Prepare batch data
        contents = [a.content for a in pending]
        logger.info(f"Generating embeddings for {len(contents)} articles...")
        
        # Batch embed with error handling
        try:
            import time
            start_time = time.time()
            vectors = embed_batch(contents)
            elapsed = time.time() - start_time
            logger.info(f"✓ Embeddings generated in {elapsed:.2f}s ({len(contents)/elapsed:.1f} articles/sec)")
        except Exception as e:
            logger.error(f"✗ Batch embedding failed critically: {e}")
            # Mark all as failed in ingestion DB
            for a in pending:
                a.processing_status = "failed"
            await ingest_db.commit()
            return EmbeddingResult(processed=0, failed=len(pending), pending_left=0)

        # Process each article individually for fault tolerance
        logger.info("Preparing embeddings for storage...")
        embeddings_to_add = []
        
        for idx, article in enumerate(pending):
            try:
                vec = vectors[idx]
                
                # Validate vector
                if np.isnan(vec).any() or np.isinf(vec).any():
                    raise ValueError("Invalid vector: contains NaN or Inf")

                # Prepare embedding object
                emb = ArticleEmbedding(
                    id=str(uuid.uuid4()),
                    article_id=article.article_id,
                    vector=vec.tobytes(),
                    model=MODEL_NAME,
                    meta_data=json.dumps({
                        "source": article.source,
                        "category": article.category,
                        "title": article.title[:100]  # Truncate for metadata
                    })
                )
                
                embeddings_to_add.append((vec, article.article_id, emb, article))
                
            except Exception as e:
                logger.error(f"Failed to prepare embedding for {article.article_id}: {e}")
                article.processing_status = "failed"
                failed += 1

        # Batch add to FAISS and DB (more efficient)
        if embeddings_to_add:
            logger.info(f"Adding {len(embeddings_to_add)} vectors to FAISS...")
            try:
                # Add all vectors to FAISS at once
                vectors_array = np.vstack([item[0] for item in embeddings_to_add])
                start_id = faiss_index.ntotal
                faiss_index.add(vectors_array)
                logger.info(f"✓ Vectors added to FAISS (total: {faiss_index.ntotal})")
                
                # Update ID mappings and save to both DBs
                logger.info("Updating ID mappings and databases...")
                for i, (_, article_id, emb, article) in enumerate(embeddings_to_add):
                    faiss_id_map[start_id + i] = article_id
                    embed_db.add(emb)  # Add to embedding DB
                    article.processing_status = "processed"  # Update in ingestion DB
                    processed += 1
                
                # Save FAISS index with mappings
                logger.info("Saving FAISS index to disk...")
                save_faiss_index()
                logger.info("✓ FAISS index saved")
                
            except Exception as e:
                logger.error(f"✗ Failed to add to FAISS/DB: {e}")
                # Mark as failed
                for _, _, _, article in embeddings_to_add:
                    article.processing_status = "failed"
                    failed += 1
                processed = 0

        # Commit changes to both databases
        logger.info("Committing database changes...")
        await embed_db.commit()
        await ingest_db.commit()
        logger.info("✓ Database commits successful")

        # Count remaining pending (from ingestion DB, excluding embedded)
        remaining_result = await ingest_db.execute(
            select(RawArticle)
            .where(RawArticle.processing_status == "pending")
        )
        all_remaining = remaining_result.scalars().all()
        pending_left = len([a for a in all_remaining if a.article_id not in embedded_ids]) - processed

        logger.info(f"Processing complete: {processed} processed, {failed} failed, {pending_left} remaining")

        return EmbeddingResult(
            processed=processed,
            failed=failed,
            pending_left=pending_left,
            skipped_existing=skipped
        )

# ---------------------------------------------------------
# ENDPOINT: Search Similar Articles
# ---------------------------------------------------------
@app.post("/embed/search")
async def search_similar(query: str, top_k: int = 5):
    """
    Find the top_k most similar articles to the query text.
    """
    try:
        # Embed query
        query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
        
        # Search FAISS
        distances, indices = faiss_index.search(query_vec, top_k)
        
        # Map back to article IDs
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid results
                continue
            article_id = faiss_id_map.get(int(idx))
            if article_id:
                results.append({
                    "article_id": article_id,
                    "distance": float(dist),
                    "similarity": float(1 / (1 + dist))  # Convert distance to similarity
                })
        
        return {
            "query": query,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {"error": str(e), "results": []}

# ---------------------------------------------------------
# ENDPOINT: Diagnostics
# ---------------------------------------------------------
@app.get("/embed/diagnostics")
async def diagnostics():
    """
    Comprehensive diagnostics to troubleshoot why no articles are pending.
    """
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy import text
    
    diagnostics_info = {
        "ingestion_db": {},
        "embedding_db": {},
        "files": {}
    }
    
    # Check if ingestion DB exists
    ingestion_db_exists = INGESTION_DB.exists()
    diagnostics_info["files"]["ingestion_db_exists"] = ingestion_db_exists
    diagnostics_info["files"]["ingestion_db_path"] = str(INGESTION_DB)
    
    if ingestion_db_exists:
        # Connect to ingestion DB and check articles
        try:
            ingest_engine = create_async_engine(
                f"sqlite+aiosqlite:///{INGESTION_DB}",
                echo=False,
                poolclass=StaticPool
            )
            
            async with ingest_engine.connect() as conn:
                # Count total articles
                result = await conn.execute(text("SELECT COUNT(*) FROM raw_articles"))
                total = result.scalar()
                diagnostics_info["ingestion_db"]["total_articles"] = total
                
                # Count by status
                result = await conn.execute(
                    text("SELECT processing_status, COUNT(*) FROM raw_articles GROUP BY processing_status")
                )
                status_counts = {row[0]: row[1] for row in result.fetchall()}
                diagnostics_info["ingestion_db"]["status_breakdown"] = status_counts
                
                # Sample articles
                result = await conn.execute(
                    text("SELECT article_id, title, processing_status FROM raw_articles LIMIT 5")
                )
                samples = [
                    {"article_id": row[0], "title": row[1], "status": row[2]}
                    for row in result.fetchall()
                ]
                diagnostics_info["ingestion_db"]["sample_articles"] = samples
            
            await ingest_engine.dispose()
            
        except Exception as e:
            diagnostics_info["ingestion_db"]["error"] = str(e)
    else:
        diagnostics_info["ingestion_db"]["error"] = "Ingestion database does not exist"
    
    # Check embedding DB
    async with AsyncSessionLocal() as db:
        total_embeddings = await db.execute(select(func.count()).select_from(ArticleEmbedding))
        diagnostics_info["embedding_db"]["total_embeddings"] = total_embeddings.scalar() or 0
        
        # Sample embeddings
        result = await db.execute(select(ArticleEmbedding).limit(5))
        samples = result.scalars().all()
        diagnostics_info["embedding_db"]["sample_embeddings"] = [
            {"article_id": e.article_id, "model": e.model}
            for e in samples
        ]
    
    # FAISS info
    diagnostics_info["faiss"] = {
        "index_size": faiss_index.ntotal if faiss_index else 0,
        "id_mappings": len(faiss_id_map),
        "index_file_exists": FAISS_INDEX_PATH.exists(),
        "map_file_exists": FAISS_ID_MAP_PATH.exists()
    }
    
    # Recommendations
    recommendations = []
    if not ingestion_db_exists:
        recommendations.append("⚠️ Ingestion database not found. Run the ingestion agent first.")
    elif diagnostics_info["ingestion_db"].get("total_articles", 0) == 0:
        recommendations.append("⚠️ No articles in ingestion database. Ingestion agent needs to fetch articles.")
    elif diagnostics_info["ingestion_db"].get("status_breakdown", {}).get("pending", 0) == 0:
        recommendations.append("✓ All articles already processed or no pending articles.")
    
    diagnostics_info["recommendations"] = recommendations
    
    return diagnostics_info
@app.get("/embed/stats")
async def stats():
    """Return embedding count, FAISS index size, model info."""
    async with EmbeddingSessionLocal() as embed_db, IngestionSessionLocal() as ingest_db:
        total = await embed_db.execute(select(func.count()).select_from(ArticleEmbedding))
        total_embeddings = total.scalar() or 0
        
        # Get embedded IDs
        embedded_result = await embed_db.execute(select(ArticleEmbedding.article_id))
        embedded_ids = {row[0] for row in embedded_result.all()}
        
        # Count pending from ingestion DB
        pending_result = await ingest_db.execute(
            select(RawArticle)
            .where(RawArticle.processing_status == "pending")
        )
        all_pending = pending_result.scalars().all()
        pending = len([a for a in all_pending if a.article_id not in embedded_ids])

    return {
        "total_embeddings": total_embeddings,
        "faiss_index_size": faiss_index.ntotal,
        "faiss_id_mappings": len(faiss_id_map),
        "pending_articles": pending,
        "embedding_dim": EMBED_DIM,
        "model": MODEL_NAME
    }

# ---------------------------------------------------------
# Health
# ---------------------------------------------------------
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "embedding-agent",
        "model_loaded": model is not None,
        "faiss_ready": faiss_index.ntotal >= 0
    }

# ---------------------------------------------------------
# ENDPOINT: Rebuild FAISS from DB (Recovery)
# ---------------------------------------------------------
@app.post("/embed/rebuild-faiss")
async def rebuild_faiss():
    """
    Rebuild FAISS index from existing embeddings in database.
    Useful for recovery or migration.
    """
    global faiss_index, faiss_id_map
    
    async with EmbeddingSessionLocal() as db:
        result = await db.execute(select(ArticleEmbedding))
        embeddings = result.scalars().all()
        
        if not embeddings:
            return {"message": "No embeddings found in database"}
        
        # Create new index
        faiss_index = faiss.IndexFlatL2(EMBED_DIM)
        faiss_id_map = {}
        
        # Add all embeddings
        vectors = []
        article_ids = []
        
        for emb in embeddings:
            vec = np.frombuffer(emb.vector, dtype=np.float32)
            vectors.append(vec)
            article_ids.append(emb.article_id)
        
        vectors_array = np.vstack(vectors)
        faiss_index.add(vectors_array)
        
        # Rebuild ID map
        for i, article_id in enumerate(article_ids):
            faiss_id_map[i] = article_id
        
        # Save
        save_faiss_index()
        
        return {
            "message": "FAISS index rebuilt successfully",
            "total_vectors": faiss_index.ntotal,
            "id_mappings": len(faiss_id_map)
        }

# ---------------------------------------------------------
# Run server
# ---------------------------------------------------------
if __name__ == "__main__":
    import sys
    
    # Check for --auto-process flag
    if "--auto-process" in sys.argv:
        os.environ["AUTO_PROCESS_ON_STARTUP"] = "true"
        logger.info("Auto-process enabled via command line flag")
    
    uvicorn.run(app, host="0.0.0.0", port=8002)

"""
http://localhost:8002/docs
"""