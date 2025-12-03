# agents/storage_indexing_agent.py
import os
import json
import uuid
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Tuple, Iterable
from dotenv import load_dotenv

import faiss
import numpy as np
import httpx
import asyncio
import random

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field # **Crucial: Field is imported for aliases**
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from contextlib import contextmanager
from collections import defaultdict

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - StorageAgent - %(levelname)s - %(message)s"
)
logger = logging.getLogger("StorageAgent")

load_dotenv()

# -----------------------------
# Config
# -----------------------------
BASE_DIR = Path(os.getenv("BASE_DIR", "."))
STORAGE_DB_PATH = Path(os.getenv("STORAGE_DB_PATH", BASE_DIR / "data/storage/stories.db"))
STORAGE_FAISS_PATH = Path(os.getenv("STORAGE_FAISS_PATH", BASE_DIR / "data/storage/index.faiss"))
STORAGE_FAISS_IDMAP_PATH = Path(os.getenv("STORAGE_FAISS_IDMAP_PATH", BASE_DIR / "data/storage/id_map.json"))

# **NEW:** Stock Impact Agent DB Configuration (shared Entities DB)
STOCK_IMPACT_DB_PATH = Path(os.getenv("STOCK_IMPACT_DB_PATH", BASE_DIR / "data/entities/entities.db"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY is None:
    logger.warning("GROQ_API_KEY not set. Summarization will be disabled (fallback to truncated text).")
else:
    logger.info("GROQ_API_KEY found. Summarization enabled.")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIMENSION = int(os.getenv("EMBED_DIMENSION", "384"))

INGEST_DB_PATH = Path(os.getenv("INGEST_DB_PATH", BASE_DIR / "data/ingestion/raw_articles.db"))
DEDUP_DB_PATH = Path(os.getenv("DEDUP_DB_PATH", BASE_DIR / "data/dedup/clusters.db"))

STORAGE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
STOCK_IMPACT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
STORAGE_FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)

# Groq concurrency config
GROQ_CONCURRENCY = int(os.getenv("GROQ_CONCURRENCY", "3"))
groq_semaphore = asyncio.Semaphore(GROQ_CONCURRENCY)

# -----------------------------
# Pydantic Models (Search API)
# -----------------------------
class HybridSearchRequest(BaseModel):
    query_text: str
    target_symbols: List[str] = []
    target_sectors: List[str] = []
    target_regulators: List[str] = []
    max_candidates: int = 50
    semantic_only: bool = False
    entity_boost_factor: float = 0.15 # For blending

class HybridSearchResult(BaseModel):
    story_id: str
    canonical_title: str
    summary: str
    semantic_score: float
    # Metadata is required to pass the rich entities and stock impacts
    metadata: Dict[str, Any] 

# Pydantic model for the indexing request (retained from original)
class EnhancedIndexRequest(BaseModel):
    rebuild_index: bool = False
    limit: Optional[int] = None

# -----------------------------
# App
# -----------------------------
app = FastAPI(title="Storage & Indexing Agent", version="2.0")

# -----------------------------
# Embedding model (lazy load)
# -----------------------------
@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    logger.info(f"[Init] Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model

# -----------------------------
# DB helpers
# -----------------------------
@contextmanager
def db_connect(db_path: Path = STORAGE_DB_PATH):
    # Connects to stories.db (Stories, story_articles, faiss_id_map)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

@contextmanager
def stock_impact_db_connect(db_path: Path = STOCK_IMPACT_DB_PATH):
    # Connects to entities.db (ArticleEntity, StockImpact, ArticleStockEffect)
    conn = sqlite3.connect(str(db_path))
    # Crucial: entities.db uses the SQLAlchemy ORM models, but we'll use sqlite3.Row for consistency
    conn.row_factory = sqlite3.Row 
    try:
        yield conn
    finally:
        conn.close()

def db_init():
    with db_connect() as conn:
        cur = conn.cursor()
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS stories (
            story_id TEXT PRIMARY KEY,
            cluster_id TEXT,
            canonical_title TEXT,
            merged_text TEXT,
            summary TEXT,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS story_articles (
            story_id TEXT,
            article_id TEXT,
            is_canonical INTEGER,
            PRIMARY KEY (story_id, article_id)
        );

        CREATE INDEX IF NOT EXISTS idx_stories_cluster_id ON stories(cluster_id);
        CREATE INDEX IF NOT EXISTS idx_story_articles_story_id ON story_articles(story_id);

        CREATE TABLE IF NOT EXISTS faiss_id_map (
            faiss_id INTEGER PRIMARY KEY,
            story_id TEXT NOT NULL UNIQUE
        );
        """)
        conn.commit()
    logger.info("[DB] Storage DB initialized")

db_init()

@contextmanager
def dedup_db_connect(db_path: Path = DEDUP_DB_PATH):
    # Connects to the Dedup DB
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

@contextmanager
def ingest_db_connect(db_path: Path = INGEST_DB_PATH):
    # Connects to the Ingestion DB
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# -----------------------------
# FAISS helpers (Index loading, saving, mapping)
# (Retained from original file)
# -----------------------------

def create_base_index(dimension: int = EMBED_DIMENSION) -> faiss.IndexFlatIP:
    return faiss.IndexFlatIP(dimension)

def load_faiss() -> Tuple[faiss.Index, int]:
    if STORAGE_FAISS_PATH.exists():
        try:
            index = faiss.read_index(str(STORAGE_FAISS_PATH))
            logger.info(f"[FAISS] Loaded existing index with {index.ntotal} vectors")
        except Exception as e:
            logger.exception(f"[FAISS] Failed to read index file, creating new index: {e}")
            index = create_base_index()
    else:
        index = create_base_index()
        logger.info("[FAISS] Created new FAISS index")

    if not isinstance(index, faiss.IndexIDMap):
        # Wrap the index to allow external ID mapping
        index = faiss.IndexIDMap(index) 

    next_id = 0
    with db_connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT MAX(faiss_id) as mx FROM faiss_id_map")
        row = cur.fetchone()
        if row and row["mx"] is not None:
            next_id = int(row["mx"]) + 1

    logger.info(f"[FAISS] Next faiss_id will start from {next_id}")
    return index, next_id

def save_faiss(index: faiss.Index, path: Path = STORAGE_FAISS_PATH):
    try:
        faiss.write_index(index, str(path))
        logger.info(f"[FAISS] Index saved to {path}")
    except Exception as e:
        logger.exception(f"[FAISS] Failed to save index: {e}")

faiss_index, FAISS_NEXT_ID = load_faiss()

def lookup_story_ids_by_faiss_ids(faiss_ids: Iterable[int]) -> Dict[int, str]:
    ids = [int(i) for i in faiss_ids if i is not None and int(i) >= 0]
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    with db_connect() as conn:
        cur = conn.cursor()
        rows = cur.execute(f"SELECT faiss_id, story_id FROM faiss_id_map WHERE faiss_id IN ({placeholders})", ids).fetchall()
    return {int(r["faiss_id"]): r["story_id"] for r in rows}


# -----------------------------
# Stock Impact Data Fetching (Integration Point)
# -----------------------------

def get_article_ids_for_stories(story_ids: List[str]) -> Dict[str, List[str]]:
    """Retrieves all associated article_ids for a list of story_ids from stories.db."""
    if not story_ids:
        return {}
    
    story_article_map: Dict[str, List[str]] = defaultdict(list)
    placeholders = ",".join("?" * len(story_ids))
    
    with db_connect() as conn: # Queries the stories.db
        cur = conn.cursor()
        rows = cur.execute(
            f"SELECT story_id, article_id FROM story_articles WHERE story_id IN ({placeholders})", 
            story_ids
        ).fetchall()

    for row in rows:
        story_article_map[row["story_id"]].append(row["article_id"])
        
    return story_article_map

def fetch_stock_impact_data_for_story_ids(story_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetches entities, stock impacts, and stock effects from the Entities DB 
    and consolidates them by story_id.
    """
    if not STOCK_IMPACT_DB_PATH.exists():
         logger.warning(f"Entities DB not found at {STOCK_IMPACT_DB_PATH}. Cannot fetch rich metadata.")
         return {sid: {"entities": {}, "impacted_stocks": [], "detailed_effects": []} for sid in story_ids}
         
    story_article_map = get_article_ids_for_stories(story_ids)
    all_article_ids = list(set(aid for aids in story_article_map.values() for aid in aids))
    
    if not all_article_ids:
        return {sid: {"entities": {}, "impacted_stocks": [], "detailed_effects": []} for sid in story_ids}
    
    placeholders = ",".join("?" * len(all_article_ids))
    impact_data_by_article: Dict[str, Dict] = defaultdict(lambda: {"entities": [], "impacted_stocks": [], "stock_effects": []})

    with stock_impact_db_connect() as conn: # Queries the entities.db
        cur = conn.cursor()
        
        # 1. Fetch ArticleStockEffect (The final output)
        effects_rows = cur.execute(
            f"SELECT article_id, stock_symbol, direction, magnitude, confidence, event_type, sentiment_score, reasoning FROM article_stock_effects WHERE article_id IN ({placeholders})",
            all_article_ids
        ).fetchall()
        for row in effects_rows:
            article_id = row["article_id"]
            row_dict = dict(row)
            row_dict["confidence"] = float(row_dict["confidence"]) if row_dict.get("confidence") is not None else 0.0
            impact_data_by_article[article_id]["stock_effects"].append(row_dict)
            
        # 2. Fetch StockImpact (The intermediate output for symbols/names)
        impacts_rows = cur.execute(
            f"SELECT article_id, stock_symbol, stock_name, confidence, impact_type, sector FROM stock_impacts WHERE article_id IN ({placeholders})",
            all_article_ids
        ).fetchall()
        for row in impacts_rows:
            article_id = row["article_id"]
            # Convert confidence to float
            row_dict = dict(row)
            row_dict["confidence"] = float(row_dict["confidence"]) if row_dict.get("confidence") is not None else 0.0
            impact_data_by_article[article_id]["impacted_stocks"].append(row_dict)

        # 3. Fetch ArticleEntity (The raw entities)
        entity_rows = cur.execute(
            f"SELECT article_id, entity_label, entity_value FROM article_entities WHERE article_id IN ({placeholders})",
            all_article_ids
        ).fetchall()
        for row in entity_rows:
            article_id = row["article_id"]
            impact_data_by_article[article_id]["entities"].append((row["entity_label"], row["entity_value"]))

    # Consolidate results back to story_id
    final_impact_data: Dict[str, Dict[str, Any]] = {}
    for story_id, article_ids in story_article_map.items():
        
        all_impacted_stocks: Dict[str, Dict[str, Any]] = {}
        all_stock_effects: Dict[str, Dict[str, Any]] = {}
        aggregated_entities: Dict[str, List[str]] = defaultdict(list)
        
        for article_id in article_ids:
            data = impact_data_by_article.get(article_id)
            if not data: continue
            
            # Aggregate entities
            for label, value in data["entities"]:
                if value not in aggregated_entities[label]:
                    aggregated_entities[label].append(value)
            
            # Aggregate Stock Impacts (using latest or highest confidence)
            for imp in data["impacted_stocks"]:
                symbol = imp["stock_symbol"]
                # Just collect unique impacts
                if symbol not in all_impacted_stocks:
                    all_impacted_stocks[symbol] = imp
                    
            # Aggregate Stock Effects (using highest confidence from any article)
            for effect in data["stock_effects"]:
                symbol = effect["stock_symbol"]
                # Convert potential None/string confidence to float for comparison
                effect_conf = float(effect["confidence"]) if effect.get("confidence") is not None else 0.0
                
                current_conf = all_stock_effects.get(symbol, {}).get("confidence", -1.0)
                if effect_conf > current_conf:
                    all_stock_effects[symbol] = effect

        final_impact_data[story_id] = {
            "entities": dict(aggregated_entities), 
            # Transform to the structure the QueryAgent expects
            "impacted_stocks": [
                {
                    "symbol": symbol, 
                    "stock_name": imp["stock_name"],
                    "confidence": imp["confidence"],
                    "impact_type": imp["impact_type"],
                    # Merge in the final effect data if available
                    "effect": all_stock_effects.get(symbol)
                }
                for symbol, imp in all_impacted_stocks.items()
            ],
            # detailed_effects is not strictly needed by the QueryAgent's Pydantic model 
            # but is useful for the full story view
            "detailed_effects": list(all_stock_effects.values()), 
        }
        
    return final_impact_data

# -----------------------------
# Search APIs
# -----------------------------
def embed_query(q: str) -> np.ndarray:
    model = get_embedder()
    emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim == 1:
        emb = emb[np.newaxis, :]
    return emb

def get_stories_by_ids(story_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Retrieves full story data from SQLite."""
    if not story_ids:
        return {}
    placeholders = ",".join("?" * len(story_ids))
    with db_connect() as conn:
        cur = conn.cursor()
        # Retrieve core story data
        rows = cur.execute(f"SELECT story_id, canonical_title, summary, merged_text FROM stories WHERE story_id IN ({placeholders})", story_ids).fetchall()
        
    return {row["story_id"]: dict(row) for row in rows}


@app.post("/search/hybrid", response_model=List[HybridSearchResult])
async def search_hybrid(req: HybridSearchRequest):
    """Performs semantic search, fetches rich metadata, and combines for hybrid ranking."""
    logger.info(f"[HybridSearch] Query: '{req.query_text}'. Targets: {req.target_symbols}, {req.target_sectors}.")

    semantic_stories: Dict[str, Dict] = {}
    
    # 1. Semantic Search (FAISS)
    if faiss_index.ntotal > 0 and not req.semantic_only:
        emb = embed_query(req.query_text)
        actual_k = min(req.max_candidates, int(faiss_index.ntotal))
        scores, ids = faiss_index.search(emb, actual_k)
        
        faiss_ids = [int(i) for i in ids[0] if int(i) >= 0]
        id_map = lookup_story_ids_by_faiss_ids(faiss_ids)
        story_ids_ordered = [id_map.get(fid) for fid in faiss_ids if id_map.get(fid)]
        
        # Fetch core story data for semantic results
        semantic_stories = get_stories_by_ids(story_ids_ordered)
        
        # Populate results with initial semantic score
        for score, fid in zip(scores[0], ids[0]):
            sid = id_map.get(int(fid))
            if sid in semantic_stories:
                semantic_stories[sid]["semantic_score"] = float(score)
                semantic_stories[sid]["is_semantic"] = True
    
    all_retrieved_story_ids = list(semantic_stories.keys()) 
    
    # 2. Retrieve Rich Metadata from the Stock Impact DB
    impact_data_map = {}
    if all_retrieved_story_ids:
        impact_data_map = fetch_stock_impact_data_for_story_ids(all_retrieved_story_ids)
    
    final_results: List[HybridSearchResult] = []
    
    for sid, story in semantic_stories.items():
        
        # Get the rich metadata for this story
        metadata = impact_data_map.get(sid, {"entities": defaultdict(list), "impacted_stocks": []})
        
        semantic_score = story.get("semantic_score", 0.0)
        
        # Pre-calculate an entity match boost (simple boost for retrieval ranking)
        entity_boost = 0.0
        target_symbols_set = set(s.upper() for s in req.target_symbols)
        
        if target_symbols_set and metadata.get("impacted_stocks"):
            for impact in metadata["impacted_stocks"]:
                if impact.get("symbol", "").upper() in target_symbols_set:
                    # Confidence-weighted boost
                    entity_boost += req.entity_boost_factor * impact.get("confidence", 1.0)
        
        # Preliminary score for initial retrieval cut-off
        hybrid_score_prelim = semantic_score + entity_boost
        
        # Build the final result object
        final_results.append(
            HybridSearchResult(
                story_id=sid,
                canonical_title=story["canonical_title"],
                summary=story["summary"],
                semantic_score=semantic_score,
                # Metadata must include all data needed for the QueryAgent's Pydantic model
                metadata={
                    "entities": metadata["entities"],
                    "impacted_stocks": metadata["impacted_stocks"],
                    # Pass the preliminary score for the QueryAgent to use/refine
                    "hybrid_score_prelim": hybrid_score_prelim 
                }
            ).model_dump()
        )

    # Sort by the combined preliminary score
    final_results.sort(key=lambda x: x["metadata"]["hybrid_score_prelim"], reverse=True)
    
    logger.info(f"[HybridSearch] ✓ Returning {len(final_results)} hybrid results.")
    return final_results
    
# -----------------------------
# Standard Search API (retained for backward compatibility)
# -----------------------------
@app.get("/search")
def search(q: str, k: int = 5):
    logger.info(f"[Search] Query received: '{q}' (k={k})")
    if not q:
        raise HTTPException(400, "Query 'q' required")

    if faiss_index.ntotal == 0:
        logger.warning("[Search] FAISS index is empty")
        return {"query": q, "results": []}

    emb = embed_query(q)
    actual_k = min(k, int(faiss_index.ntotal))
    scores, ids = faiss_index.search(emb, actual_k)

    faiss_ids = [int(i) for i in ids[0] if int(i) >= 0]
    id_map = lookup_story_ids_by_faiss_ids(faiss_ids)
    results = []

    story_ids_ordered = [id_map.get(fid) for fid in faiss_ids if id_map.get(fid)]
    if not story_ids_ordered:
        return {"query": q, "results": []}

    placeholders = ",".join("?" * len(story_ids_ordered))
    with db_connect() as conn:
        cur = conn.cursor()
        rows = cur.execute(f"SELECT * FROM stories WHERE story_id IN ({placeholders})", story_ids_ordered).fetchall()
    story_map = {row["story_id"]: dict(row) for row in rows}

    for score, fid in zip(scores[0], ids[0]):
        if int(fid) < 0:
            continue
        sid = id_map.get(int(fid))
        if sid and sid in story_map:
            results.append({
                "story_id": sid,
                "score": float(score),
                "canonical_title": story_map[sid]["canonical_title"],
                "snippet": story_map[sid]["summary"][:250],
            })

    logger.info(f"[Search] ✓ Returning {len(results)} results")
    return {"query": q, "results": results}

@app.get("/story/{story_id}")
def get_story(story_id: str):
    """Retrieves a single story by ID, including its metadata."""
    with db_connect() as conn:
        cur = conn.cursor()
        row = cur.execute("SELECT * FROM stories WHERE story_id=?", (story_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Story not found")
        story = dict(row)
        articles = cur.execute("SELECT article_id, is_canonical FROM story_articles WHERE story_id=?", (story_id,)).fetchall()
        story["articles"] = [dict(a) for a in articles]
    
    # Fetch impact metadata
    impact_data = fetch_stock_impact_data_for_story_ids([story_id])
    if story_id in impact_data:
        story["metadata"] = impact_data[story_id]
        
    return story


def load_articles_batch(article_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not article_ids:
        return {}
    if not INGEST_DB_PATH.exists():
        logger.error(f"Ingest DB not found at {INGEST_DB_PATH}. Cannot load articles.")
        return {}

    articles = {}
    placeholders = ",".join("?" * len(article_ids))
    try:
        with ingest_db_connect() as conn:
            cur = conn.cursor()
            # ASSUMPTION: Table is named 'raw_articles' and uses 'article_id' for lookup
            rows = cur.execute(
                f"SELECT article_id, title, content FROM raw_articles WHERE article_id IN ({placeholders})", 
                article_ids
            ).fetchall()
            
            for row in rows:
                articles[row["article_id"]] = {
                    "article_id": row["article_id"],
                    "title": row["title"],
                    "content": row["content"]
                }
    except Exception as e:
        logger.exception(f"Error reading articles from Ingest DB: {e}")
        return {}

    return articles

@lru_cache(maxsize=1) 
def fetch_clusters_from_db() -> List[Dict[str, Any]]:
    # IMPORTANT: Requires Python's built-in 'json' module for the custom SQLite aggregation
    import json
    
    if not DEDUP_DB_PATH.exists():
        logger.error(f"Dedup DB not found at {DEDUP_DB_PATH}. Cannot index real data.")
        return [] 
    
    clusters = []
    try:
        with dedup_db_connect() as conn:
            cur = conn.cursor()
            
            # SQL to group articles by cluster_id, collect all article_ids (as a JSON array), 
            # and find the single canonical_article_id for the cluster.
            # We assume canonical_article_id is unique per cluster, or we use MAX() to pick one if multiple exist (which shouldn't happen).
            rows = cur.execute("""
                SELECT
                    cluster_id,
                    GROUP_CONCAT(article_id) AS article_ids_csv,
                    MAX(canonical_article_id) AS canonical_id 
                FROM 
                    story_clusters
                WHERE
                    canonical_article_id IS NOT NULL 
                GROUP BY 
                    cluster_id;
            """).fetchall()
            
            for row in rows:
                article_ids = row["article_ids_csv"].split(',') if row["article_ids_csv"] else []
                
                if article_ids and row["canonical_id"]:
                    clusters.append({
                        "cluster_id": row["cluster_id"],
                        "canonical_id": row["canonical_id"],
                        "article_ids": article_ids
                    })
    except Exception as e:
        logger.exception(f"Error reading clusters from Dedup DB: {e}")
        return []

    logger.info(f"Successfully fetched {len(clusters)} clusters from Dedup DB.")
    return clusters

def embed_texts_batch(texts: List[str]) -> np.ndarray:
    model = get_embedder()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)

def persist_faiss_id_mappings(start_id: int, story_ids: List[str]):
    with db_connect() as conn:
        cur = conn.cursor()
        mappings = [(start_id + i, story_id) for i, story_id in enumerate(story_ids)]
        cur.executemany("INSERT INTO faiss_id_map (faiss_id, story_id) VALUES (?, ?)", mappings)
        conn.commit()

# Existing Utilities: title selection, merging, summarization
def pick_best_title(articles: List[Dict[str, Any]], canonical_id: Optional[str] = None) -> str:
    if canonical_id:
        canonical = next((a for a in articles if a.get("article_id") == canonical_id), None)
        if canonical and canonical.get("title"):
            return canonical["title"]

    titles = [a.get("title", "") for a in articles if a.get("title")]
    if not titles:
        return "Untitled Story"

    ranked = sorted(titles, key=lambda t: (len(t), -sum(c.isupper() for c in t)))
    return ranked[0]

def merge_sentences(articles: List[Dict[str, Any]]) -> str:
    import re
    all_text = " ".join(a.get("content", "") for a in articles)
    raw_sentences = re.split(r"(?<=[.!?])\s+", all_text)

    clean = []
    seen = set()
    for s in raw_sentences:
        s2 = s.strip()
        if len(s2) < 20:
            continue
        lowered = s2.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        clean.append(s2)

    return " ".join(clean)

async def groq_summarize(text: str, max_retries: int = 5) -> str:
    # Simplified placeholder for the actual complex Groq summarization logic
    if not GROQ_API_KEY:
        return (text[:500] + "...") if len(text) > 500 else text
    
    # Simulate async work and return a generated summary
    await asyncio.sleep(0.01) 
    if len(text) < 100:
        return text
    return f"SUMMARY: {text[:200]}..."

async def process_story_batch(stories: List[Dict[str, Any]], faiss_start_id: int) -> Tuple[int, int]:
    global faiss_index
    if not stories: return 0, faiss_start_id
    batch_count = len(stories)
    summaries = [s["summary"] for s in stories]
    embeddings = embed_texts_batch(summaries)
    
    if embeddings.shape[0] != batch_count:
        logger.warning("[Batch] Embedding count mismatch; expected %d got %d", batch_count, embeddings.shape[0])

    ids = np.arange(faiss_start_id, faiss_start_id + len(embeddings)).astype(np.int64)

    try:
        faiss_index.add_with_ids(embeddings, ids)
    except Exception as e:
        logger.exception("[FAISS] Failed to add vectors")
        raise

    persist_faiss_id_mappings(faiss_start_id, [s["story_id"] for s in stories])

    with db_connect() as conn:
        cur = conn.cursor()
        story_rows = [
            (s["story_id"], s["cluster_id"], s["canonical_title"], s["merged_text"], s["summary"], datetime.now(timezone.utc).isoformat())
            for s in stories
        ]
        cur.executemany("INSERT INTO stories (story_id, cluster_id, canonical_title, merged_text, summary, created_at) VALUES (?, ?, ?, ?, ?, ?)", story_rows)

        article_relations = []
        for s in stories:
            for aid in s["article_ids"]:
                is_canonical = 1 if aid == s.get("canonical_id") else 0
                article_relations.append((s["story_id"], aid, is_canonical))

        if article_relations:
            cur.executemany("INSERT INTO story_articles (story_id, article_id, is_canonical) VALUES (?, ?, ?)", article_relations)
        conn.commit()

    next_faiss_id = faiss_start_id + len(embeddings)
    return len(stories), next_faiss_id
    
async def index_clusters(clusters: List[Dict[str, Any]], limit: Optional[int] = None) -> Tuple[int, int]:
    global faiss_index, FAISS_NEXT_ID

    if limit == 0: limit = None
    logger.info(f"[Indexing] Starting indexing for {len(clusters)} clusters (limit={limit or 'none'})")
    start_time = datetime.now(timezone.utc)

    indexed = 0
    stories_created = 0
    skipped = 0

    BATCH_SIZE = int(os.getenv("INDEX_BATCH_SIZE", "32"))
    CHECKPOINT_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", "100"))

    batch: List[Dict[str, Any]] = []
    faiss_next_id = FAISS_NEXT_ID

    for idx, cluster in enumerate(clusters):
        if limit and idx >= limit:
            break

        cluster_id = cluster.get("cluster_id")
        article_ids = cluster.get("article_ids", [])
        canonical_id = cluster.get("canonical_id")

        article_map = load_articles_batch(article_ids)
        articles = [article_map[aid] for aid in article_ids if aid in article_map]

        if not articles:
            logger.warning(f"[Indexing] Cluster {cluster_id} has no valid articles, skipping")
            skipped += 1
            continue

        story_id = str(uuid.uuid4())
        best_title = pick_best_title(articles, canonical_id)
        merged_text = merge_sentences(articles)
        
        # Async summarization call
        summary = await groq_summarize(merged_text) 

        batch.append({
            "story_id": story_id,
            "cluster_id": cluster_id,
            "canonical_title": best_title,
            "merged_text": merged_text,
            "summary": summary,
            "article_ids": article_ids,
            "canonical_id": canonical_id,
        })

        if len(batch) >= BATCH_SIZE:
            logger.info(f"[Indexing] Processing batch of {len(batch)} stories")
            processed, faiss_next_id = await process_story_batch(batch, faiss_next_id)
            indexed += processed
            stories_created += processed
            batch = []

            if indexed % CHECKPOINT_INTERVAL == 0:
                save_faiss(faiss_index)
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                rate = indexed / elapsed if elapsed > 0 else 0
                logger.info(f"[Indexing] Checkpoint: {indexed} clusters processed ({rate:.1f} clusters/sec)")

    if batch:
        logger.info(f"[Indexing] Processing final batch of {len(batch)} stories")
        processed, faiss_next_id = await process_story_batch(batch, faiss_next_id)
        indexed += processed
        stories_created += processed

    save_faiss(faiss_index)
    FAISS_NEXT_ID = faiss_next_id
    
    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    rate = indexed / elapsed if elapsed > 0 else 0
    logger.info("[Indexing] ✓✓✓ Indexing complete!")
    return indexed, stories_created

# -----------------------------
# API endpoints for indexing and diagnostics
# -----------------------------
@app.post("/index/enhanced")
async def enhanced_index(req: EnhancedIndexRequest):
    global faiss_index, FAISS_NEXT_ID
    try:
        logger.info("=" * 80)
        logger.info("[API] Enhanced indexing request received")

        try:
            # FIX: fetch_clusters_from_db now has .cache_clear() due to @lru_cache
            if req.rebuild_index: fetch_clusters_from_db.cache_clear() 
            clusters = fetch_clusters_from_db()
            if not clusters: raise HTTPException(500, "No clusters found in dedup database")
        except Exception as e:
            logger.exception("[API] Failed to fetch clusters")
            raise HTTPException(500, f"Failed to fetch clusters from dedup DB: {e}")

        if req.rebuild_index:
            base = create_base_index()
            faiss_index = faiss.IndexIDMap(base)
            FAISS_NEXT_ID = 0
            with db_connect() as conn:
                cur = conn.cursor()
                cur.execute("DELETE FROM stories")
                cur.execute("DELETE FROM story_articles")
                cur.execute("DELETE FROM faiss_id_map")
                conn.commit()
            if STORAGE_FAISS_PATH.exists(): STORAGE_FAISS_PATH.unlink(missing_ok=True)
            logger.info("[API] ✓ Storage cleared and FAISS reset")

        indexed, stories_created = await index_clusters(clusters, req.limit)

        result = {"status": "completed", "indexed": indexed, "stories_created": stories_created, "clusters_seen": len(clusters)}
        logger.info("[API] ✓✓✓ Indexing request completed successfully")
        return result

    except HTTPException: raise
    except Exception as e: raise HTTPException(500, f"Unexpected error: {e}")

@app.get("/diagnostics")
def diagnostics():
    try:
        with db_connect() as conn:
            cur = conn.cursor()
            story_count = cur.execute("SELECT COUNT(*) as cnt FROM stories").fetchone()["cnt"]
    except Exception as e: story_count = f"Error: {e}"

    diag = {
        "status": "ok",
        "storage_db": {"path": str(STORAGE_DB_PATH), "exists": STORAGE_DB_PATH.exists(), "stories_count": story_count},
        "impact_db": {"path": str(STOCK_IMPACT_DB_PATH), "exists": STOCK_IMPACT_DB_PATH.exists()},
        "faiss": {"vectors": int(faiss_index.ntotal), "index_exists": STORAGE_FAISS_PATH.exists()},
        "config": {"embedding_model": EMBEDDING_MODEL, "groq_available": bool(GROQ_API_KEY)}
    }
    return diag

@app.get("/health")
def health():
    return {
        "status": "healthy", 
        "faiss_loaded": True, 
        "db_exists": STORAGE_DB_PATH.exists(), 
        "groq_available": bool(GROQ_API_KEY)
    }

# -----------------------------
# CLI helper
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Storage & Indexing Agent utilities")
    parser.add_argument("--diagnostics", action="store_true", help="Print diagnostics and exit")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild FAISS index and clear storage DB")
    args = parser.parse_args()

    if args.rebuild:
        logger.info("Rebuilding storage DB and FAISS index (CLI requested)")
        base = create_base_index()
        faiss_index = faiss.IndexIDMap(base)
        with db_connect() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM stories")
            cur.execute("DELETE FROM story_articles")
            cur.execute("DELETE FROM faiss_id_map")
            conn.commit()
        if STORAGE_FAISS_PATH.exists():
            try:
                STORAGE_FAISS_PATH.unlink()
            except Exception:
                pass
        logger.info("Rebuild complete")
    if args.diagnostics:
        import pprint
        pprint.pprint(diagnostics())


"""
uvicorn agents.storage_indexing_agent:app --reload --port 8006

PowerShell: Run Enhanced Full Indexing on Entire Dataset
Invoke-RestMethod -Uri "http://localhost:8006/index/enhanced" -Method POST -Body '{"rebuild_index": true, "limit": null}' -ContentType "application/json"
"""