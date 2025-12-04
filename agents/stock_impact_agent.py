"""
Stock Impact Analysis Agent
---------------------------

- Loads entities & stock_impacts produced by the Entity Extraction Agent.
- Runs sentiment analysis + event classification + heuristics to produce:
    article_stock_effects: directional impact, magnitude, confidence, reasoning
- Async FastAPI service with SQLite DB storage.
- Run: uvicorn agents.stock_impact_agent:app --reload --port 8005
"""

import os
import uuid
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import asyncio
from collections import defaultdict
import uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import (
    Column, String, Integer, Float, Text, TIMESTAMP, select, func, Index
)
from sqlalchemy.pool import StaticPool

# Transformers: UPDATED IMPORT
from transformers import pipeline, AutoTokenizer 

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("StockImpactAgent")

# ----------------------------
# Paths & DB setup
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ENTITIES_DIR = DATA_DIR / "entities"
ENTITIES_DB = ENTITIES_DIR / "entities.db"

MAPPINGS_DIR = DATA_DIR / "mappings"
STOCK_MAPPING_FILE = MAPPINGS_DIR / "stock_mapping.json"
SECTOR_MAPPING_FILE = MAPPINGS_DIR / "sector_mapping.json"

ENTITIES_DB_URL = f"sqlite+aiosqlite:///{ENTITIES_DB}"

# Create engine/session
engine = create_async_engine(
    ENTITIES_DB_URL,
    echo=False,
    future=True,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
Session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

# ----------------------------
# DB Models (augment existing Entities DB)
# ----------------------------
class ArticleStockEffect(Base):
    __tablename__ = "article_stock_effects"
    __table_args__ = (
        Index("ix_article_stock_effect", "article_id", "stock_symbol"),
    )

    id = Column(String, primary_key=True)
    article_id = Column(String, index=True, nullable=False)
    stock_symbol = Column(String, index=True, nullable=False)
    direction = Column(String)           # up | down | neutral
    magnitude = Column(Float)            # 0..1
    confidence = Column(Float)           # 0..1
    event_type = Column(String)          # earnings, regulatory, merger, etc.
    sentiment_score = Column(Float)      # -1..1
    reasoning = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))


# Reuse these models from entity agent (table names must match)
class ArticleEntity(Base):
    __tablename__ = "article_entities"
    id = Column(String, primary_key=True)
    article_id = Column(String, index=True)
    entity_value = Column(String)
    entity_label = Column(String)
    score = Column(String)
    span = Column(Text)
    start_pos = Column(Integer)
    end_pos = Column(Integer)
    created_at = Column(TIMESTAMP(timezone=True))


class StockImpact(Base):
    __tablename__ = "stock_impacts"
    id = Column(String, primary_key=True)
    article_id = Column(String, index=True)
    stock_symbol = Column(String, index=True)
    stock_name = Column(String)
    confidence = Column(String)
    impact_type = Column(String)
    reasoning = Column(Text)
    sector = Column(String)
    created_at = Column(TIMESTAMP(timezone=True))


# ----------------------------
# Pydantic models (API)
# ----------------------------
class RunConfig(BaseModel):
    limit: Optional[int] = 20

class RunOne(BaseModel):
    article_id: str

class StockEffectOut(BaseModel):
    article_id: str
    stock_symbol: str
    direction: str
    magnitude: float
    confidence: float
    event_type: Optional[str] = None
    sentiment_score: Optional[float] = None
    reasoning: Optional[str] = None

class RunResult(BaseModel):
    status: str
    processed: int
    effects_generated: int
    errors: int

# ----------------------------
# Models & pipelines
# ----------------------------
SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "ProsusAI/finbert")
EVENT_CLASS_MODEL = os.getenv("EVENT_CLASS_MODEL", "typeform/distilbert-base-uncased-mnli")

SENTIMENT_PIPELINE = None
SENTIMENT_TOKENIZER = None # NEW: For token-based truncation
EVENT_PIPELINE = None
SENTIMENT_LOCK = None
EVENT_LOCK = None

# OLD: MAX_CONTEXT_CHARS
# NEW: Use MAX_TOKENS for truncation (default 500 tokens)
MAX_TOKENS = int(os.getenv("STOCK_IMPACT_MAX_TOKENS", "500"))
# Keep MAX_CONTEXT_CHARS as a fallback if tokenizer fails
MAX_CONTEXT_CHARS = int(os.getenv("STOCK_IMPACT_MAX_CONTEXT", "4000")) 


EVENT_LABELS = [
    "earnings", "regulatory", "merger_acquisition", "layoff", "product_launch",
    "funding", "litigation", "macro_policy", "supply_chain", "partnership", "other"
]

# Stock mapping loader
STOCK_SYMBOL_MAP: Dict[str, Dict[str, Any]] = {}
SECTOR_STOCK_MAP: Dict[str, List[Dict[str, Any]]] = {}

def now():
    return datetime.now(timezone.utc)

def load_mappings():
    global STOCK_SYMBOL_MAP, SECTOR_STOCK_MAP
    if STOCK_MAPPING_FILE.exists():
        try:
            with open(STOCK_MAPPING_FILE, 'r') as f:
                data = json.load(f)
                STOCK_SYMBOL_MAP = data.get("mappings", {})
            logger.info(f"Loaded stock mappings: {len(STOCK_SYMBOL_MAP)}")
        except Exception as e:
            logger.error(f"Failed to load stock mappings: {e}")
    else:
        logger.warning(f"Stock mapping missing at {STOCK_MAPPING_FILE}")

    if SECTOR_MAPPING_FILE.exists():
        try:
            with open(SECTOR_MAPPING_FILE, 'r') as f:
                SECTOR_STOCK_MAP = json.load(f)
            logger.info(f"Loaded sector mappings: {len(SECTOR_STOCK_MAP)}")
        except Exception as e:
            logger.error(f"Failed to load sector mappings: {e}")
    else:
        logger.warning(f"Sector mapping missing at {SECTOR_MAPPING_FILE}")


# ----------------------------
# Heuristics & scoring
# ----------------------------
def normalize_text(t: str) -> str:
    return " ".join((t or "").strip().split())

def sentiment_to_range(label: str, score: float) -> float:
    # pipeline returns label like 'POSITIVE'/'NEGATIVE' with score 0..1
    # convert to -1..1
    if label.upper().startswith("POS"):
        return score
    else:
        return -score

def compute_direction_and_magnitude(event_type: str, sentiment: float, prior_confidence: float) -> tuple[str, float]:
    """
    Decide direction (up/down/neutral) and magnitude 0..1.
    Simple rule-based combination of event_type and sentiment.
    """
    # Base magnitude from absolute sentiment
    mag = min(1.0, max(0.0, abs(sentiment)))

    # Event-specific adjustments
    event_boost = {
        "earnings": 0.3,
        "regulatory": 0.2,
        "merger_acquisition": 0.25,
        "layoff": 0.2,
        "product_launch": 0.15,
        "funding": 0.25,
        "litigation": 0.2,
        "macro_policy": 0.2,
        "supply_chain": 0.15,
        "partnership": 0.1,
        "other": 0.0
    }
    boost = event_boost.get(event_type, 0.0)
    mag = min(1.0, mag + boost * prior_confidence)

    # Direction rules: if sentiment strongly positive -> up; strongly negative -> down
    if sentiment > 0.15:
        direction = "up"
    elif sentiment < -0.15:
        direction = "down"
    else:
        # neutral unless event implies negative (regulatory, litigation) or positive (funding)
        if event_type in ("regulatory", "litigation", "layoff"):
            direction = "down"
            # make magnitude modest
            mag = max(mag, 0.1)
        elif event_type in ("funding", "earnings", "merger_acquisition", "product_launch"):
            direction = "up"
            mag = max(mag, 0.1)
        else:
            direction = "neutral"

    # scale by prior_confidence (0..1)
    mag = mag * prior_confidence
    mag = round(float(mag), 4)
    return direction, mag

# ----------------------------
# Lifespan (startup)
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("="*60)
    logger.info("STARTING STOCK IMPACT ANALYSIS AGENT")
    logger.info("="*60)

    # Load mappings
    load_mappings()

    # Ensure DB tables exist (in same entities DB)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ready (article_stock_effects)")

    # Load HF pipelines (blocking)
    global SENTIMENT_PIPELINE, EVENT_PIPELINE, SENTIMENT_TOKENIZER # UPDATED GLOBAL DECLARATION
    try:
        logger.info(f"Loading sentiment model: {SENTIMENT_MODEL}")
        SENTIMENT_PIPELINE = pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        device=-1
        )
        # NEW: Load Tokenizer for accurate context truncation
        SENTIMENT_TOKENIZER = AutoTokenizer.from_pretrained(SENTIMENT_MODEL) 
        logger.info("✓ Sentiment pipeline and tokenizer loaded")

        logger.info(f"Loading event classification model: {EVENT_CLASS_MODEL}")
        EVENT_PIPELINE = pipeline(
        "zero-shot-classification",
        model=EVENT_CLASS_MODEL,
        device=-1
        )
        logger.info("✓ Event classification pipeline loaded")
    except Exception as e:
        logger.exception(f"Failed to load HF pipelines: {e}")
        raise

    logger.info("Stock Impact Agent ready")
    yield
    logger.info("Shutting down Stock Impact Agent...")

app = FastAPI(title="Stock Impact Analysis Agent", version="1.0.0", lifespan=lifespan)

# ----------------------------
# Core functions
# ----------------------------
async def fetch_article_stock_impacts(limit: int = 50) -> Dict[str, Dict[str, Any]]:
    """
    Fetch recent stock_impacts grouped by article_id along with cached context.
    
    OPTIMIZATION: Exclude articles that have already been processed 
    (i.e., those with records in ArticleStockEffect).
    """
    async with Session() as db:
        
        # 1. Identify article_ids that have ALREADY been processed
        # This is a list of article_ids that have an existing record in the final output table.
        processed_q = select(ArticleStockEffect.article_id).distinct()
        # Use scalars().all() to get a simple list of article_ids
        processed_article_ids = (await db.execute(processed_q)).scalars().all()

        # 2. Select recent StockImpacts, ensuring article_id is NOT in the processed list
        # We fetch the article_id based on the StockImpact table
        impacts_q = (
            select(StockImpact.article_id)
            .where(StockImpact.article_id.notin_(processed_article_ids)) # <--- NEW FILTER APPLIED HERE
            .order_by(StockImpact.created_at.desc())
            .limit(limit)
        )
        raw_article_ids = (await db.execute(impacts_q)).scalars().all()
        
        # Use dict.fromkeys to preserve order while ensuring uniqueness
        article_ids = list(dict.fromkeys(raw_article_ids)) 
        if not article_ids:
            return {}

        # 3. Fetch all StockImpact and ArticleEntity records for the remaining (unprocessed) article_ids
        impacts_res = await db.execute(
            select(StockImpact).where(StockImpact.article_id.in_(article_ids))
        )
        impacts_by_article: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for imp in impacts_res.scalars().all():
            impacts_by_article[imp.article_id].append({
                "article_id": imp.article_id,
                "stock_symbol": imp.stock_symbol,
                "stock_name": imp.stock_name,
                "stock_confidence": float(imp.confidence) if imp.confidence else 0.5,
                "impact_type": imp.impact_type,
                "reasoning": imp.reasoning,
                "sector": imp.sector
            })

        ent_res = await db.execute(
            select(ArticleEntity).where(ArticleEntity.article_id.in_(article_ids))
        )
        spans_by_article: Dict[str, List[str]] = defaultdict(list)
        for ent in ent_res.scalars().all():
            span = ent.span or ent.entity_value or ""
            if span:
                spans_by_article[ent.article_id].append(span)

        article_payload: Dict[str, Dict[str, Any]] = {}
        for aid in article_ids:
            article_impacts = impacts_by_article.get(aid, [])
            if not article_impacts:
                continue
            span_text = " ".join(spans_by_article.get(aid, []))
            article_payload[aid] = {
                "impacts": article_impacts,
                "context": normalize_text(span_text)
            }

        return article_payload


async def analyze_article_for_stock_effect(
    article_id: str,
    impacts: Optional[List[Dict[str, Any]]] = None,
    context: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    For a single article: find associated stock_impacts and produce article_stock_effects.
    """
    if not impacts:
        async with Session() as db:
            # Load stock impacts for article
            q = select(StockImpact).where(StockImpact.article_id == article_id)
            res = await db.execute(q)
            db_impacts = res.scalars().all()
            if not db_impacts:
                raise HTTPException(status_code=404, detail="No stock impacts found for article")

            impacts = [{
                "article_id": imp.article_id,
                "stock_symbol": imp.stock_symbol,
                "stock_name": imp.stock_name,
                "stock_confidence": float(imp.confidence) if imp.confidence else 0.5,
                "impact_type": imp.impact_type,
                "reasoning": imp.reasoning,
                "sector": imp.sector
            } for imp in db_impacts]

            if context is None:
                ent_q = select(ArticleEntity).where(ArticleEntity.article_id == article_id)
                ent_res = await db.execute(ent_q)
                ents = ent_res.scalars().all()
                context = " ".join([e.span or e.entity_value for e in ents]) if ents else ""

    # If context is empty fallback to using reasoning from impacts
    if not context:
        context = " ".join([(i.get("reasoning") or "") for i in impacts])

    context = normalize_text(context)
    
    # NEW: Token-based truncation
    if SENTIMENT_TOKENIZER is not None:
        # Use encode to truncate to max_tokens, then decode back to string
        # We set max_length=MAX_TOKENS and truncation=True
        encoded = SENTIMENT_TOKENIZER.encode(
            context,
            max_length=MAX_TOKENS,
            truncation=True,
            return_tensors='pt' # return as tensor
        )
        truncated_context = SENTIMENT_TOKENIZER.decode(encoded[0], skip_special_tokens=True)
        logger.debug(f"Truncated context to {encoded.size(1)} tokens.")
    else:
        # Fallback to character-based truncation
        truncated_context = context[:MAX_CONTEXT_CHARS]

    # Run sentiment and event classification synchronously via executor
    loop = asyncio.get_running_loop()
    if SENTIMENT_PIPELINE is None or EVENT_PIPELINE is None:
        raise HTTPException(status_code=503, detail="Pipelines not ready")

    global SENTIMENT_LOCK, EVENT_LOCK
    if SENTIMENT_LOCK is None:
        SENTIMENT_LOCK = asyncio.Lock()
    if EVENT_LOCK is None:
        EVENT_LOCK = asyncio.Lock()

    async with SENTIMENT_LOCK:
        sentiment_result = await loop.run_in_executor(None, SENTIMENT_PIPELINE, truncated_context)
    # sentiment_result is list like [{'label': 'POSITIVE', 'score': 0.99}]
    if isinstance(sentiment_result, list) and sentiment_result:
        sent_label = sentiment_result[0].get("label")
        sent_score = float(sentiment_result[0].get("score", 0.0))
        sentiment_val = sentiment_to_range(sent_label, sent_score)
    else:
        sent_label, sent_score, sentiment_val = "NEUTRAL", 0.0, 0.0

    # Event classification
    async with EVENT_LOCK:
        zsc_results = await loop.run_in_executor(
            None,
            lambda: EVENT_PIPELINE(truncated_context, candidate_labels=EVENT_LABELS, multi_label=False)
        )
    # zsc_results: {labels: [...], scores: [...]}
    event_type = None
    event_score = 0.5
    if isinstance(zsc_results, dict):
        labels = zsc_results.get("labels", [])
        scores = zsc_results.get("scores", [])
        if labels:
            event_type = labels[0]
            event_score = float(scores[0]) if scores else 0.5

    # Build effects for each stock impact entry
    effects = []
    for imp in impacts:
        prior_conf = float(imp.get("stock_confidence", 0.5))
        # Compute direction/magnitude
        direction, magnitude = compute_direction_and_magnitude(event_type or "other", sentiment_val, prior_conf)
        
        # UPDATED: Combined confidence (Weighted Average)
        # Weights: Prior Conf (0.4), Event Score (0.3), Absolute Sentiment Magnitude (0.3)
        event_score_norm = event_score if event_score else 0.5
        sentiment_mag_norm = abs(sentiment_val)
        combined_conf = (
            0.4 * prior_conf +
            0.3 * event_score_norm +
            0.3 * sentiment_mag_norm
        )
        combined_conf = min(1.0, max(0.0, combined_conf))


        reasoning_parts = [
            f"Extraction reasoning: {imp.get('reasoning')}" if imp.get("reasoning") else "",
            f"Event: {event_type} ({event_score:.2f})" if event_type else "",
            f"Sentiment: {sentiment_val:.3f}",
            f"Prior stock-impact-confidence: {prior_conf:.3f}"
        ]
        reasoning_text = " | ".join([r for r in reasoning_parts if r])

        effects.append({
            "id": str(uuid.uuid4()),
            "article_id": article_id,
            "stock_symbol": imp.get("stock_symbol"),
            "direction": direction,
            "magnitude": magnitude,
            "confidence": round(float(combined_conf), 4),
            "event_type": event_type,
            "sentiment_score": round(float(sentiment_val), 4),
            "reasoning": reasoning_text,
            "created_at": now()
        })

    # Persist effects
    async with Session() as db:
        for e in effects:
            ase = ArticleStockEffect(
                id=e["id"],
                article_id=e["article_id"],
                stock_symbol=e["stock_symbol"],
                direction=e["direction"],
                magnitude=e["magnitude"],
                confidence=e["confidence"],
                event_type=e["event_type"],
                sentiment_score=e["sentiment_score"],
                reasoning=e["reasoning"],
                created_at=e["created_at"]
            )
            db.add(ase)
        await db.commit()

    return effects

# ----------------------------
# API Endpoints
# ----------------------------
@app.post("/impact/run", response_model=RunResult)
async def run_bulk(cfg: RunConfig):
    """
    Run stock impact analysis for the most recent N stock_impacts (default limit).
    """
    limit = int(cfg.limit or 20)
    logger.info(f"[StockImpactAgent] Running bulk analysis (limit={limit})")
    # fetch_article_stock_impacts now filters out already processed articles.
    pending = await fetch_article_stock_impacts(limit) 
    if not pending:
        return RunResult(status="completed", processed=0, effects_generated=0, errors=0)

    processed_articles = set()
    effects_count = 0
    errors = 0

    # Process articles in parallel, but limit concurrency (to avoid memory spikes)
    sem = asyncio.Semaphore(4)

    async def worker(article_id, payload):
        nonlocal effects_count, errors
        async with sem:
            try:
                res = await analyze_article_for_stock_effect(
                    article_id,
                    impacts=payload.get("impacts"),
                    context=payload.get("context")
                )
                processed_articles.add(article_id)
                effects_count += len(res)
            except Exception as e:
                logger.exception(f"Error analyzing article {article_id}: {e}")
                errors += 1

    tasks = []
    for aid, payload in pending.items():
        tasks.append(asyncio.create_task(worker(aid, payload)))

    await asyncio.gather(*tasks)
    return RunResult(status="completed", processed=len(processed_articles), effects_generated=effects_count, errors=errors)


@app.post("/impact/article", response_model=List[StockEffectOut])
async def run_one(payload: RunOne):
    """
    Run analysis for a single article_id (will persist effects and return them).
    """
    article_id = payload.article_id
    effects = await analyze_article_for_stock_effect(article_id)
    # Convert for response
    out = []
    for e in effects:
        out.append(StockEffectOut(
            article_id=e["article_id"],
            stock_symbol=e["stock_symbol"],
            direction=e["direction"],
            magnitude=e["magnitude"],
            confidence=e["confidence"],
            event_type=e["event_type"],
            sentiment_score=e["sentiment_score"],
            reasoning=e["reasoning"]
        ))
    return out


@app.get("/impact/by_stock/{stock_symbol}")
async def get_by_stock(stock_symbol: str, limit: int = 50):
    async with Session() as db:
        q = select(ArticleStockEffect).where(ArticleStockEffect.stock_symbol == stock_symbol).order_by(ArticleStockEffect.created_at.desc()).limit(limit)
        res = await db.execute(q)
        rows = res.scalars().all()
        return {
            "stock_symbol": stock_symbol,
            "count": len(rows),
            "effects": [
                {
                    "article_id": r.article_id,
                    "direction": r.direction,
                    "magnitude": float(r.magnitude),
                    "confidence": float(r.confidence),
                    "event_type": r.event_type,
                    "sentencing": r.reasoning,
                    "created_at": r.created_at.isoformat()
                }
                for r in rows
            ]
        }


@app.get("/impact/by_article/{article_id}")
async def get_by_article(article_id: str):
    async with Session() as db:
        q = select(ArticleStockEffect).where(ArticleStockEffect.article_id == article_id)
        res = await db.execute(q)
        rows = res.scalars().all()
        if not rows:
            raise HTTPException(status_code=404, detail="No effects found for article")
        return {
            "article_id": article_id,
            "effects": [
                {
                    "stock_symbol": r.stock_symbol,
                    "direction": r.direction,
                    "magnitude": float(r.magnitude),
                    "confidence": float(r.confidence),
                    "event_type": r.event_type,
                    "sentiment_score": float(r.sentiment_score) if r.sentiment_score else None,
                    "reasoning": r.reasoning
                } for r in rows
            ]
        }


@app.get("/impact/summary")
async def summary(limit: int = 20):
    """
    Return a brief summary of the most recent effects and top impacted stocks.
    """
    async with Session() as db:
        # Most recent effects
        recent_q = select(ArticleStockEffect).order_by(ArticleStockEffect.created_at.desc()).limit(limit)
        recent_res = await db.execute(recent_q)
        recent = recent_res.scalars().all()

        # Top stocks
        top_q = select(ArticleStockEffect.stock_symbol, func.count()).group_by(ArticleStockEffect.stock_symbol).order_by(func.count().desc()).limit(10)
        top_res = await db.execute(top_q)
        top = {row[0]: row[1] for row in top_res.all()}

    return {
        "recent_count": len(recent),
        "recent": [
            {
                "article_id": r.article_id,
                "stock_symbol": r.stock_symbol,
                "direction": r.direction,
                "magnitude": float(r.magnitude),
                "confidence": float(r.confidence),
                "event_type": r.event_type
            } for r in recent
        ],
        "top_impacted_stocks": top
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy" if (SENTIMENT_PIPELINE is not None and EVENT_PIPELINE is not None) else "degraded",
        "sentiment_model": SENTIMENT_MODEL,
        "event_model": EVENT_CLASS_MODEL
    }


@app.get("/diagnostics")
async def diagnostics():
    d = {}
    async with Session() as db:
        try:
            total_effects = (await db.execute(select(func.count()).select_from(ArticleStockEffect))).scalar() or 0
            total_entities = (await db.execute(select(func.count()).select_from(ArticleEntity))).scalar() or 0
            total_stock_impacts = (await db.execute(select(func.count()).select_from(StockImpact))).scalar() or 0
            d["total_effects"] = total_effects
            d["total_entities"] = total_entities
            d["total_stock_impacts"] = total_stock_impacts
        except Exception as e:
            d["error"] = str(e)

    d["mappings_loaded"] = {"stocks": len(STOCK_SYMBOL_MAP), "sectors": len(SECTOR_STOCK_MAP)}
    return d


# ----------------------------
# Dev run
# ----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)

"""
# 1. Start the stock impact analysis service
uvicorn agents.stock_impact_agent:app --reload --port 8005

# 2. Run bulk analysis
$body = @{ limit = 5000 } | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://localhost:8005/impact/run" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
"""