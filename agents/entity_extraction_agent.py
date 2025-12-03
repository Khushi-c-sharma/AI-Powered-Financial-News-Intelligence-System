"""
Entity Extraction Agent (Financial NER - HuggingFace)
-----------------------------------------------------

- Uses a token-classification HF model fine-tuned for finance NER (default).
- Async FastAPI service with lightweight SQLite (aiosqlite) storage.
- Idempotent: skips articles already extracted.
- Optimized batch processing and proper async handling.
- Endpoints:
    POST /entities/extract            -> run extraction for N pending articles
    POST /entities/extract/article    -> run extraction for single article_id
    GET  /entities/by_article/{id}    -> list entities for an article
    GET  /entities/search             -> search entities by name
    GET  /entities/stats              -> statistics
    POST /entities/rebuild            -> rebuild entities DB (re-extract all)
- Put file under: agents/entity_extraction_agent.py
- Run: uvicorn agents.entity_extraction_agent:app --reload --port 8004
"""

import os
import uuid
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import asyncio
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import numpy as np
import uvicorn

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import (
    Column, String, Integer, Text, TIMESTAMP, select, func, Index, and_
)
from sqlalchemy.pool import StaticPool

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EntityExtractionAgent")

# ----------------------------
# Paths & DB setup
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INGESTION_DB = DATA_DIR / "ingestion" / "raw_articles.db"
ENTITIES_DIR = DATA_DIR / "entities"
ENTITIES_DB = ENTITIES_DIR / "entities.db"
MAPPINGS_DIR = DATA_DIR / "mappings"

ENTITIES_DIR.mkdir(parents=True, exist_ok=True)
MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)

# DB URLs
ENTITIES_DB_URL = f"sqlite+aiosqlite:///{ENTITIES_DB}"
INGESTION_DB_URL = f"sqlite+aiosqlite:///{INGESTION_DB}"

# Create engines / sessions with connection pooling
entities_engine = create_async_engine(
    ENTITIES_DB_URL,
    echo=False,
    future=True,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
EntitiesSession = sessionmaker(entities_engine, class_=AsyncSession, expire_on_commit=False)

ingestion_engine = create_async_engine(
    INGESTION_DB_URL,
    echo=False,
    future=True,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
IngestionSession = sessionmaker(ingestion_engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

# ----------------------------
# DB Models
# ----------------------------
class RawArticle(Base):
    """Reference to ingestion DB articles."""
    __tablename__ = "raw_articles"
    
    id = Column(String, primary_key=True)
    article_id = Column(String, unique=True, index=True)
    title = Column(Text)
    content = Column(Text)
    processing_status = Column(String, index=True)


class ArticleEntity(Base):
    """Store extracted entities for articles."""
    __tablename__ = "article_entities"
    __table_args__ = (
        Index("ix_article_entity", "article_id", "entity_value"),
        Index("ix_article_label", "article_id", "entity_label"),
    )

    id = Column(String, primary_key=True)
    article_id = Column(String, index=True, nullable=False)
    entity_value = Column(String, nullable=False, index=True)   # normalized text of entity
    entity_label = Column(String, index=True)       # label returned by model (e.g., ORG, PERSON, MONEY)
    score = Column(String)                          # model confidence as string
    span = Column(Text)                             # raw span/context
    start_pos = Column(Integer)                     # start position in text
    end_pos = Column(Integer)                       # end position in text
    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))


class ExtractionLog(Base):
    """Simple log table for extraction runs."""
    __tablename__ = "extraction_logs"
    
    id = Column(String, primary_key=True)
    started_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))
    completed_at = Column(TIMESTAMP(timezone=True))
    articles_processed = Column(Integer, default=0)
    entities_extracted = Column(Integer, default=0)
    errors = Column(Integer, default=0)
    status = Column(String, default="running")
    details = Column(Text, default="")


class ArticleExtractionStatus(Base):
    """Track which articles have been extracted to avoid reprocessing."""
    __tablename__ = "article_extraction_status"
    
    article_id = Column(String, primary_key=True, index=True)
    extracted_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))
    entity_count = Column(Integer, default=0)
    model_version = Column(String)


class StockImpact(Base):
    """Maps articles to impacted stocks with confidence scores."""
    __tablename__ = "stock_impacts"
    __table_args__ = (
        Index("ix_article_stock", "article_id", "stock_symbol"),
        Index("ix_stock_confidence", "stock_symbol", "confidence"),
    )
    
    id = Column(String, primary_key=True)
    article_id = Column(String, index=True, nullable=False)
    stock_symbol = Column(String, index=True, nullable=False)
    stock_name = Column(String)
    confidence = Column(String, nullable=False)  # confidence score as string
    impact_type = Column(String, nullable=False)  # direct, sector, regulatory, competitor
    reasoning = Column(Text)  # why this stock was mapped
    sector = Column(String, index=True)  # sector if applicable
    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))

# ----------------------------
# Pydantic models (API)
# ----------------------------
class ExtractConfig(BaseModel):
    limit: Optional[int] = 20

class ExtractOne(BaseModel):
    article_id: str

class ExtractionResult(BaseModel):
    status: str
    processed: int
    entities_extracted: int
    stocks_mapped: int = 0
    errors: int
    skipped: int = 0


class ImpactedStock(BaseModel):
    symbol: str
    name: Optional[str] = None
    confidence: float
    type: str
    sector: Optional[str] = None
    reasoning: Optional[str] = None


class ArticleAnalysis(BaseModel):
    article_id: str
    companies: List[str]
    sectors: List[str]
    people: List[str]
    events: List[str]
    impacted_stocks: List[ImpactedStock]

# ----------------------------
# NER Model Configuration (Multi-Model for Finance)
# ----------------------------
# Primary financial NER model - best for companies, sectors, financial terms
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "Jean-Baptiste/roberta-large-ner-english")

# Secondary model for events, risks, general entities
SECONDARY_MODEL = os.getenv("SECONDARY_MODEL", "dslim/bert-base-NER-uncased")

MODEL_VERSION = "2.0-multi"

logger.info(f"[EntityAgent] Primary model: {PRIMARY_MODEL}")
logger.info(f"[EntityAgent] Secondary model: {SECONDARY_MODEL}")

# Model pipelines loaded on startup
PRIMARY_PIPELINE = None
SECONDARY_PIPELINE = None

# Custom entity mapping for financial domain
FINANCIAL_ENTITY_MAP = {
    # Standard NER labels
    "ORG": "COMPANY",
    "ORGANIZATION": "COMPANY",
    "PERSON": "PERSON",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "MONEY": "FINANCIAL_VALUE",
    "PERCENT": "PERCENTAGE",
    "DATE": "DATE",
    
    # Financial-specific (from specialized models)
    "COMPANY": "COMPANY",
    "SECTOR": "SECTOR",
    "PRODUCT": "PRODUCT",
    "EVENT": "EVENT",
    "RISK": "RISK",
    "METRIC": "METRIC",
    "CURRENCY": "CURRENCY",
}

# ----------------------------
# Stock & Sector Mapping Configuration
# ----------------------------
# Load stock mapping data (company names to stock symbols)
STOCK_MAPPING_FILE = DATA_DIR / "mappings" / "stock_mapping.json"
SECTOR_MAPPING_FILE = DATA_DIR / "mappings" / "sector_mapping.json"

# Confidence levels for impact mapping
CONFIDENCE_LEVELS = {
    "direct_mention": 1.0,      # Company explicitly mentioned
    "subsidiary": 0.95,          # Subsidiary of mentioned company
    "parent_company": 0.90,      # Parent company of mentioned entity
    "sector_leader": 0.80,       # Leading company in mentioned sector
    "sector_major": 0.70,        # Major player in mentioned sector
    "sector_minor": 0.60,        # Other company in mentioned sector
    "regulatory": 0.75,          # Regulatory impact (variable based on severity)
    "competitor": 0.65,          # Competitor of mentioned company
    "supply_chain": 0.60,        # Supply chain relationship
}

# Default mappings (will be loaded from files if available)
STOCK_SYMBOL_MAP = {}
SECTOR_STOCK_MAP = {}
COMPANY_ALIASES = {}

# ----------------------------
# Helper utilities
# ----------------------------
def now():
    return datetime.now(timezone.utc)

def normalize_entity_text(text: str) -> str:
    """Normalize entity text (remove extra whitespace)."""
    return " ".join(text.strip().split())


def load_stock_mappings():
    """Load stock symbol mappings and sector data from JSON files."""
    global STOCK_SYMBOL_MAP, SECTOR_STOCK_MAP, COMPANY_ALIASES
    
    # Load stock mapping (company name -> symbol)
    if STOCK_MAPPING_FILE.exists():
        try:
            with open(STOCK_MAPPING_FILE, 'r') as f:
                data = json.load(f)
                STOCK_SYMBOL_MAP = data.get("mappings", {})
                COMPANY_ALIASES = data.get("aliases", {})
            logger.info(f"Loaded {len(STOCK_SYMBOL_MAP)} stock mappings")
        except Exception as e:
            logger.error(f"Failed to load stock mappings: {e}")
    else:
        logger.warning(f"Stock mapping file not found: {STOCK_MAPPING_FILE}")
        # Create default for Indian stocks
        create_default_mappings()
    
    # Load sector mapping (sector -> list of stocks)
    if SECTOR_MAPPING_FILE.exists():
        try:
            with open(SECTOR_MAPPING_FILE, 'r') as f:
                SECTOR_STOCK_MAP = json.load(f)
            logger.info(f"Loaded {len(SECTOR_STOCK_MAP)} sector mappings")
        except Exception as e:
            logger.error(f"Failed to load sector mappings: {e}")
    else:
        logger.warning(f"Sector mapping file not found: {SECTOR_MAPPING_FILE}")


def create_default_mappings():
    """Create default mappings for common Indian stocks."""
    global STOCK_SYMBOL_MAP, COMPANY_ALIASES, SECTOR_STOCK_MAP
    
    # Common Indian stocks
    STOCK_SYMBOL_MAP = {
        "HDFC Bank": {"symbol": "HDFCBANK", "name": "HDFC Bank Ltd", "sector": "Banking"},
        "ICICI Bank": {"symbol": "ICICIBANK", "name": "ICICI Bank Ltd", "sector": "Banking"},
        "State Bank of India": {"symbol": "SBIN", "name": "State Bank of India", "sector": "Banking"},
        "SBI": {"symbol": "SBIN", "name": "State Bank of India", "sector": "Banking"},
        "Reliance": {"symbol": "RELIANCE", "name": "Reliance Industries Ltd", "sector": "Oil & Gas"},
        "Reliance Industries": {"symbol": "RELIANCE", "name": "Reliance Industries Ltd", "sector": "Oil & Gas"},
        "TCS": {"symbol": "TCS", "name": "Tata Consultancy Services", "sector": "IT Services"},
        "Tata Consultancy Services": {"symbol": "TCS", "name": "Tata Consultancy Services", "sector": "IT Services"},
        "Infosys": {"symbol": "INFY", "name": "Infosys Ltd", "sector": "IT Services"},
        "Wipro": {"symbol": "WIPRO", "name": "Wipro Ltd", "sector": "IT Services"},
        "HCL Technologies": {"symbol": "HCLTECH", "name": "HCL Technologies Ltd", "sector": "IT Services"},
        "Tech Mahindra": {"symbol": "TECHM", "name": "Tech Mahindra Ltd", "sector": "IT Services"},
        "ITC": {"symbol": "ITC", "name": "ITC Ltd", "sector": "FMCG"},
        "Hindustan Unilever": {"symbol": "HINDUNILVR", "name": "Hindustan Unilever Ltd", "sector": "FMCG"},
        "Asian Paints": {"symbol": "ASIANPAINT", "name": "Asian Paints Ltd", "sector": "Paints"},
        "Bharti Airtel": {"symbol": "BHARTIARTL", "name": "Bharti Airtel Ltd", "sector": "Telecom"},
        "Airtel": {"symbol": "BHARTIARTL", "name": "Bharti Airtel Ltd", "sector": "Telecom"},
        "Maruti Suzuki": {"symbol": "MARUTI", "name": "Maruti Suzuki India Ltd", "sector": "Automobile"},
        "Tata Motors": {"symbol": "TATAMOTORS", "name": "Tata Motors Ltd", "sector": "Automobile"},
        "Mahindra": {"symbol": "M&M", "name": "Mahindra & Mahindra Ltd", "sector": "Automobile"},
        "Bajaj Auto": {"symbol": "BAJAJ-AUTO", "name": "Bajaj Auto Ltd", "sector": "Automobile"},
        "Axis Bank": {"symbol": "AXISBANK", "name": "Axis Bank Ltd", "sector": "Banking"},
        "Kotak Mahindra Bank": {"symbol": "KOTAKBANK", "name": "Kotak Mahindra Bank Ltd", "sector": "Banking"},
    }
    
    # Aliases
    COMPANY_ALIASES = {
        "HDFC": "HDFC Bank",
        "SBI": "State Bank of India",
        "RIL": "Reliance Industries",
        "TCS": "Tata Consultancy Services",
    }
    
    # Sector mappings
    SECTOR_STOCK_MAP = {
        "Banking": [
            {"symbol": "HDFCBANK", "name": "HDFC Bank", "tier": "leader"},
            {"symbol": "ICICIBANK", "name": "ICICI Bank", "tier": "leader"},
            {"symbol": "SBIN", "name": "State Bank of India", "tier": "leader"},
            {"symbol": "AXISBANK", "name": "Axis Bank", "tier": "major"},
            {"symbol": "KOTAKBANK", "name": "Kotak Mahindra Bank", "tier": "major"},
        ],
        "Financial Services": [
            {"symbol": "HDFCBANK", "name": "HDFC Bank", "tier": "leader"},
            {"symbol": "ICICIBANK", "name": "ICICI Bank", "tier": "major"},
            {"symbol": "BAJFINANCE", "name": "Bajaj Finance", "tier": "major"},
        ],
        "IT Services": [
            {"symbol": "TCS", "name": "TCS", "tier": "leader"},
            {"symbol": "INFY", "name": "Infosys", "tier": "leader"},
            {"symbol": "WIPRO", "name": "Wipro", "tier": "major"},
            {"symbol": "HCLTECH", "name": "HCL Tech", "tier": "major"},
            {"symbol": "TECHM", "name": "Tech Mahindra", "tier": "major"},
        ],
        "Automobile": [
            {"symbol": "MARUTI", "name": "Maruti Suzuki", "tier": "leader"},
            {"symbol": "TATAMOTORS", "name": "Tata Motors", "tier": "major"},
            {"symbol": "M&M", "name": "Mahindra & Mahindra", "tier": "major"},
            {"symbol": "BAJAJ-AUTO", "name": "Bajaj Auto", "tier": "major"},
        ],
        "Oil & Gas": [
            {"symbol": "RELIANCE", "name": "Reliance Industries", "tier": "leader"},
            {"symbol": "ONGC", "name": "ONGC", "tier": "major"},
            {"symbol": "BPCL", "name": "BPCL", "tier": "major"},
        ],
        "Telecom": [
            {"symbol": "BHARTIARTL", "name": "Bharti Airtel", "tier": "leader"},
            {"symbol": "IDEA", "name": "Vodafone Idea", "tier": "major"},
        ],
        "FMCG": [
            {"symbol": "HINDUNILVR", "name": "Hindustan Unilever", "tier": "leader"},
            {"symbol": "ITC", "name": "ITC", "tier": "major"},
            {"symbol": "NESTLEIND", "name": "Nestle India", "tier": "major"},
        ],
    }
    
    # Save to files
    try:
        with open(STOCK_MAPPING_FILE, 'w') as f:
            json.dump({
                "mappings": STOCK_SYMBOL_MAP,
                "aliases": COMPANY_ALIASES
            }, f, indent=2)
        
        with open(SECTOR_MAPPING_FILE, 'w') as f:
            json.dump(SECTOR_STOCK_MAP, f, indent=2)
        
        logger.info("Created default stock and sector mappings")
    except Exception as e:
        logger.error(f"Failed to save default mappings: {e}")


def map_company_to_stock(company_name: str) -> Optional[Dict[str, Any]]:
    """Map a company name to its stock symbol and info."""
    # Normalize company name
    normalized = normalize_entity_text(company_name)
    
    # Check aliases first
    if normalized in COMPANY_ALIASES:
        normalized = COMPANY_ALIASES[normalized]
    
    # Direct lookup
    if normalized in STOCK_SYMBOL_MAP:
        return STOCK_SYMBOL_MAP[normalized]
    
    # Fuzzy matching (case-insensitive contains)
    normalized_lower = normalized.lower()
    for company, info in STOCK_SYMBOL_MAP.items():
        if normalized_lower in company.lower() or company.lower() in normalized_lower:
            return info
    
    return None


def map_sector_to_stocks(sector: str, max_stocks: int = 5) -> List[Dict[str, Any]]:
    """Map a sector to its constituent stocks with tier information."""
    normalized_sector = normalize_entity_text(sector)
    
    # Direct lookup
    if normalized_sector in SECTOR_STOCK_MAP:
        stocks = SECTOR_STOCK_MAP[normalized_sector][:max_stocks]
        return stocks
    
    # Fuzzy matching
    normalized_lower = normalized_sector.lower()
    for sector_name, stocks in SECTOR_STOCK_MAP.items():
        if normalized_lower in sector_name.lower() or sector_name.lower() in normalized_lower:
            return stocks[:max_stocks]
    
    return []

# ----------------------------
# FastAPI Lifespan: create tables and load model
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize databases and load NER model on startup."""
    logger.info("=" * 60)
    logger.info("STARTING ENTITY EXTRACTION AGENT")
    logger.info("=" * 60)
    
    # Step 1: Load stock and sector mappings
    logger.info("Step 1/3: Loading stock and sector mappings...")
    load_stock_mappings()
    logger.info(f"✓ Loaded {len(STOCK_SYMBOL_MAP)} stock mappings, {len(SECTOR_STOCK_MAP)} sectors")
    
    # Step 2: Create entities DB tables
    logger.info("Step 2/3: Initializing entities database...")
    async with entities_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info(f"✓ Entities DB ready at {ENTITIES_DB}")

    # Step 3: Load NER models (multi-model approach)
    logger.info(f"Step 3/3: Loading NER models...")
    logger.info("NOTE: First run may take several minutes to download models...")
    
    global PRIMARY_PIPELINE, SECONDARY_PIPELINE
    try:
        # Load primary model (financial NER specialist)
        logger.info(f"Loading primary model: {PRIMARY_MODEL}")
        PRIMARY_PIPELINE = pipeline(
            "ner",
            model=PRIMARY_MODEL,
            aggregation_strategy="simple",
            device=-1  # CPU by default; set to 0 for GPU
        )
        logger.info("✓ Primary NER pipeline loaded")
        
        # Load secondary model (general NER + events)
        logger.info(f"Loading secondary model: {SECONDARY_MODEL}")
        SECONDARY_PIPELINE = pipeline(
            "ner",
            model=SECONDARY_MODEL,
            aggregation_strategy="simple",
            device=-1
        )
        logger.info("✓ Secondary NER pipeline loaded")
        
    except Exception as e:
        logger.error(f"✗ Failed to load NER models: {e}")
        raise

    logger.info("=" * 60)
    logger.info("✓ ENTITY EXTRACTION AGENT READY")
    logger.info("=" * 60)
    
    yield
    
    logger.info("Shutting down Entity Extraction Agent...")

app = FastAPI(title="Entity Extraction Agent", version="2.0.0", lifespan=lifespan)

# ----------------------------
# Core extraction logic
# ----------------------------
async def fetch_pending_articles(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Read articles from ingestion DB that haven't been extracted yet.
    NOTE: We look for ANY articles (not just 'processed' status) since embedding agent
    updates a separate DB. We filter by what hasn't been extracted.
    """
    async with IngestionSession() as ing_db, EntitiesSession() as ent_db:
        # Get already extracted article IDs
        extracted_result = await ent_db.execute(
            select(ArticleExtractionStatus.article_id)
        )
        extracted_ids = {row[0] for row in extracted_result.all()}
        
        logger.info(f"Found {len(extracted_ids)} already extracted articles")
        
        # Fetch articles from ingestion DB (any status, we'll filter by extracted_ids)
        result = await ing_db.execute(
            select(RawArticle)
            .limit(limit * 3)  # Fetch extra to account for already extracted
        )
        all_articles = result.scalars().all()
        
        # Filter out already extracted
        pending = [
            {
                "article_id": a.article_id,
                "title": a.title or "",
                "content": a.content or ""
            }
            for a in all_articles
            if a.article_id not in extracted_ids and (a.content or "").strip()
        ][:limit]
        
        return pending


async def already_extracted(article_id: str) -> bool:
    """Return True if entities already exist for this article."""
    async with EntitiesSession() as db:
        res = await db.execute(
            select(func.count())
            .select_from(ArticleExtractionStatus)
            .where(ArticleExtractionStatus.article_id == article_id)
        )
        c = res.scalar() or 0
        return c > 0


def run_ner_sync(text: str, max_length: int = 512) -> List[Dict[str, Any]]:
    """
    Run multi-model NER pipeline synchronously with chunking for long texts.
    Combines results from both primary (financial) and secondary (general) models.
    Returns list of dictionaries: {entity_group, score, word, start, end}
    """
    if PRIMARY_PIPELINE is None or SECONDARY_PIPELINE is None:
        raise RuntimeError("NER pipelines not loaded")
    
    try:
        # Truncate very long texts to avoid memory issues
        if len(text) > max_length * 10:
            logger.warning(f"Text too long ({len(text)} chars), truncating to {max_length * 10}")
            text = text[:max_length * 10]
        
        # Run both pipelines
        primary_entities = PRIMARY_PIPELINE(text)
        secondary_entities = SECONDARY_PIPELINE(text)
        
        # Combine and deduplicate entities
        entity_map = {}
        
        # Process primary model results (higher priority for financial entities)
        for ent in primary_entities:
            key = (ent.get("start"), ent.get("end"))
            label = ent.get("entity_group", "UNKNOWN")
            
            # Map to financial entity types
            mapped_label = FINANCIAL_ENTITY_MAP.get(label, label)
            
            entity_map[key] = {
                "entity_group": mapped_label,
                "score": ent.get("score", 0.0),
                "word": ent.get("word", ""),
                "start": ent.get("start"),
                "end": ent.get("end"),
                "source": "primary"
            }
        
        # Add secondary model results (only if not overlapping with primary)
        for ent in secondary_entities:
            key = (ent.get("start"), ent.get("end"))
            
            # Skip if primary model already detected this span
            if key in entity_map:
                continue
            
            # Check for overlaps with existing entities
            start, end = ent.get("start"), ent.get("end")
            overlaps = False
            for (existing_start, existing_end) in entity_map.keys():
                if (start < existing_end and end > existing_start):
                    overlaps = True
                    break
            
            if not overlaps:
                label = ent.get("entity_group", "UNKNOWN")
                mapped_label = FINANCIAL_ENTITY_MAP.get(label, label)
                
                entity_map[key] = {
                    "entity_group": mapped_label,
                    "score": ent.get("score", 0.0),
                    "word": ent.get("word", ""),
                    "start": ent.get("start"),
                    "end": ent.get("end"),
                    "source": "secondary"
                }
        
        # Convert to list and return
        return list(entity_map.values())
        
    except Exception as e:
        logger.exception(f"NER pipeline error: {e}")
        return []


async def extract_entities_for_article(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract entities for a single article, save to DB, and map to impacted stocks.
    Returns comprehensive analysis including stock impacts.
    """
    article_id = article["article_id"]
    text = (article.get("title") or "") + "\n\n" + (article.get("content") or "")

    # Skip empty content
    if not text.strip():
        logger.debug(f"Skipping empty article: {article_id}")
        return {"entities": [], "stocks": []}

    logger.info(f"Extracting entities for article: {article_id} ({len(text)} chars)")

    # Run model in executor to avoid blocking event loop
    loop = asyncio.get_running_loop()
    ner_result: List[Dict[str, Any]] = await loop.run_in_executor(
        None, run_ner_sync, text
    )

    if not ner_result:
        logger.debug(f"No entities found for article: {article_id}")

    # Organize entities by type
    companies = []
    sectors = []
    people = []
    events = []
    regulators = []
    
    async with EntitiesSession() as db:
        # Group entities to avoid duplicates
        entity_map = {}
        
        for ent in ner_result:
            try:
                entity_value = normalize_entity_text(ent.get("word", ""))
                if not entity_value:
                    continue
                
                label = ent.get("entity_group") or ent.get("entity") or "UNKNOWN"
                score = float(ent.get("score", 0.0))
                start = ent.get("start", 0)
                end = ent.get("end", 0)
                span_text = text[start:end] if start is not None and end is not None else ""

                # Create unique key for deduplication
                key = (entity_value, label)
                
                # Keep highest score if duplicate
                if key in entity_map:
                    if score > float(entity_map[key]["score"]):
                        entity_map[key] = {
                            "entity_value": entity_value,
                            "label": label,
                            "score": score,
                            "span": span_text,
                            "start": start,
                            "end": end
                        }
                else:
                    entity_map[key] = {
                        "entity_value": entity_value,
                        "label": label,
                        "score": score,
                        "span": span_text,
                        "start": start,
                        "end": end
                    }
                    
                # Categorize entities
                if label == "COMPANY":
                    companies.append(entity_value)
                elif label == "SECTOR":
                    sectors.append(entity_value)
                elif label == "PERSON":
                    people.append(entity_value)
                elif label == "EVENT":
                    events.append(entity_value)
                    
            except Exception as e:
                logger.exception(f"Failed to process entity for {article_id}: {e}")
        
        # Save entities to DB
        saved_entities = []
        for ent_data in entity_map.values():
            ae = ArticleEntity(
                id=str(uuid.uuid4()),
                article_id=article_id,
                entity_value=ent_data["entity_value"],
                entity_label=ent_data["label"],
                score=str(ent_data["score"]),
                span=ent_data["span"],
                start_pos=ent_data["start"],
                end_pos=ent_data["end"],
                created_at=now()
            )
            db.add(ae)
            saved_entities.append(ent_data)
        
        # Map to impacted stocks with confidence scores
        impacted_stocks = []
        stock_impacts_db = []
        
        # 1. Direct company mentions (confidence: 1.0)
        for company in companies:
            stock_info = map_company_to_stock(company)
            if stock_info:
                impact = {
                    "symbol": stock_info["symbol"],
                    "name": stock_info["name"],
                    "confidence": CONFIDENCE_LEVELS["direct_mention"],
                    "type": "direct",
                    "sector": stock_info.get("sector"),
                    "reasoning": f"Direct mention of {company}"
                }
                impacted_stocks.append(impact)
                
                # Save to DB
                stock_impacts_db.append(StockImpact(
                    id=str(uuid.uuid4()),
                    article_id=article_id,
                    stock_symbol=stock_info["symbol"],
                    stock_name=stock_info["name"],
                    confidence=str(CONFIDENCE_LEVELS["direct_mention"]),
                    impact_type="direct",
                    reasoning=f"Direct mention of {company}",
                    sector=stock_info.get("sector"),
                    created_at=now()
                ))
        
        # 2. Sector-wide impact (confidence: 60-80% based on tier)
        for sector in sectors:
            sector_stocks = map_sector_to_stocks(sector)
            for stock in sector_stocks:
                # Skip if already added as direct mention
                if any(s["symbol"] == stock["symbol"] for s in impacted_stocks):
                    continue
                
                # Confidence based on tier
                tier = stock.get("tier", "minor")
                if tier == "leader":
                    confidence = CONFIDENCE_LEVELS["sector_leader"]
                elif tier == "major":
                    confidence = CONFIDENCE_LEVELS["sector_major"]
                else:
                    confidence = CONFIDENCE_LEVELS["sector_minor"]
                
                impact = {
                    "symbol": stock["symbol"],
                    "name": stock["name"],
                    "confidence": confidence,
                    "type": "sector",
                    "sector": sector,
                    "reasoning": f"{tier.title()} player in {sector} sector"
                }
                impacted_stocks.append(impact)
                
                # Save to DB
                stock_impacts_db.append(StockImpact(
                    id=str(uuid.uuid4()),
                    article_id=article_id,
                    stock_symbol=stock["symbol"],
                    stock_name=stock["name"],
                    confidence=str(confidence),
                    impact_type="sector",
                    reasoning=f"{tier.title()} player in {sector} sector",
                    sector=sector,
                    created_at=now()
                ))
        
        # 3. Regulatory impact (check for regulatory keywords)
        regulatory_keywords = ["sebi", "rbi", "reserve bank", "regulator", "regulatory", "compliance"]
        has_regulatory_mention = any(keyword in text.lower() for keyword in regulatory_keywords)
        
        if has_regulatory_mention and sectors:
            # Apply regulatory impact to sector leaders only
            for sector in sectors:
                sector_stocks = map_sector_to_stocks(sector, max_stocks=2)  # Top 2 only
                for stock in sector_stocks:
                    if stock.get("tier") == "leader" and not any(s["symbol"] == stock["symbol"] for s in impacted_stocks):
                        confidence = CONFIDENCE_LEVELS["regulatory"]
                        
                        impact = {
                            "symbol": stock["symbol"],
                            "name": stock["name"],
                            "confidence": confidence,
                            "type": "regulatory",
                            "sector": sector,
                            "reasoning": f"Regulatory news affecting {sector} sector"
                        }
                        impacted_stocks.append(impact)
                        
                        stock_impacts_db.append(StockImpact(
                            id=str(uuid.uuid4()),
                            article_id=article_id,
                            stock_symbol=stock["symbol"],
                            stock_name=stock["name"],
                            confidence=str(confidence),
                            impact_type="regulatory",
                            reasoning=f"Regulatory news affecting {sector} sector",
                            sector=sector,
                            created_at=now()
                        ))
        
        # Save stock impacts
        for stock_impact in stock_impacts_db:
            db.add(stock_impact)
        
        # Mark article as extracted
        status = ArticleExtractionStatus(
            article_id=article_id,
            extracted_at=now(),
            entity_count=len(saved_entities),
            model_version=MODEL_VERSION
        )
        db.add(status)
        
        await db.commit()
        
    logger.info(
        f"✓ Extracted {len(saved_entities)} entities, "
        f"mapped to {len(impacted_stocks)} stocks for article: {article_id}"
    )
    
    return {
        "entities": saved_entities,
        "stocks": impacted_stocks,
        "companies": companies,
        "sectors": sectors,
        "people": people,
        "events": events
    }


# ----------------------------
# API Endpoints
# ----------------------------
@app.post("/entities/extract", response_model=ExtractionResult)
async def extract_bulk(cfg: ExtractConfig):
    """
    Extract entities from up to `limit` processed articles that do not yet have entities.
    Also maps entities to impacted stocks with confidence scores.
    """
    limit = int(cfg.limit or 20)
    logger.info(f"[EntityAgent] Starting bulk extraction (limit={limit})")

    # Fetch pending articles
    pending = await fetch_pending_articles(limit)
    logger.info(f"Found {len(pending)} articles pending extraction")

    if not pending:
        return ExtractionResult(
            status="completed",
            processed=0,
            entities_extracted=0,
            stocks_mapped=0,
            errors=0,
            skipped=0
        )

    processed = 0
    entities_total = 0
    stocks_total = 0
    errors = 0
    skipped = 0

    # Process each article
    for article in pending:
        try:
            if await already_extracted(article["article_id"]):
                logger.debug(f"Skipping already-extracted article: {article['article_id']}")
                skipped += 1
                continue
            
            result = await extract_entities_for_article(article)
            processed += 1
            entities_total += len(result.get("entities", []))
            stocks_total += len(result.get("stocks", []))
            
        except Exception as e:
            logger.exception(f"Extraction error for article {article.get('article_id')}: {e}")
            errors += 1

    logger.info(
        f"[EntityAgent] Bulk extraction complete. "
        f"processed={processed} entities={entities_total} stocks={stocks_total} "
        f"errors={errors} skipped={skipped}"
    )

    return ExtractionResult(
        status="completed",
        processed=processed,
        entities_extracted=entities_total,
        stocks_mapped=stocks_total,
        errors=errors,
        skipped=skipped
    )


@app.post("/entities/extract/article", response_model=ArticleAnalysis)
async def extract_single(payload: ExtractOne):
    """
    Extract entities for a single article_id and return complete analysis including impacted stocks.
    """
    article_id = payload.article_id
    
    # Check if already extracted
    if await already_extracted(article_id):
        # Return existing results
        async with EntitiesSession() as db:
            # Get entities
            entities_result = await db.execute(
                select(ArticleEntity).where(ArticleEntity.article_id == article_id)
            )
            entities = entities_result.scalars().all()
            
            # Get stock impacts
            impacts_result = await db.execute(
                select(StockImpact).where(StockImpact.article_id == article_id)
            )
            impacts = impacts_result.scalars().all()
            
            # Organize by type
            companies = [e.entity_value for e in entities if e.entity_label == "COMPANY"]
            sectors = [e.entity_value for e in entities if e.entity_label == "SECTOR"]
            people = [e.entity_value for e in entities if e.entity_label == "PERSON"]
            events = [e.entity_value for e in entities if e.entity_label == "EVENT"]
            
            impacted_stocks = [
                ImpactedStock(
                    symbol=i.stock_symbol,
                    name=i.stock_name,
                    confidence=float(i.confidence),
                    type=i.impact_type,
                    sector=i.sector,
                    reasoning=i.reasoning
                )
                for i in impacts
            ]
            
            return ArticleAnalysis(
                article_id=article_id,
                companies=companies,
                sectors=sectors,
                people=people,
                events=events,
                impacted_stocks=impacted_stocks
            )
    
    # Fetch article from ingestion DB
    async with IngestionSession() as db:
        result = await db.execute(
            select(RawArticle).where(RawArticle.article_id == article_id)
        )
        raw_article = result.scalar_one_or_none()
        
        if not raw_article:
            raise HTTPException(status_code=404, detail="Article not found in ingestion DB")
        
        article = {
            "article_id": raw_article.article_id,
            "title": raw_article.title or "",
            "content": raw_article.content or ""
        }

    # Extract entities and map stocks
    result = await extract_entities_for_article(article)
    
    # Convert to response format
    impacted_stocks = [
        ImpactedStock(
            symbol=s["symbol"],
            name=s["name"],
            confidence=s["confidence"],
            type=s["type"],
            sector=s.get("sector"),
            reasoning=s.get("reasoning")
        )
        for s in result.get("stocks", [])
    ]
    
    return ArticleAnalysis(
        article_id=article_id,
        companies=result.get("companies", []),
        sectors=result.get("sectors", []),
        people=result.get("people", []),
        events=result.get("events", []),
        impacted_stocks=impacted_stocks
    )


@app.get("/entities/by_article/{article_id}")
async def get_by_article(article_id: str):
    """Return entities for a given article_id."""
    async with EntitiesSession() as db:
        result = await db.execute(
            select(ArticleEntity)
            .where(ArticleEntity.article_id == article_id)
            .order_by(ArticleEntity.score.desc())
        )
        rows = result.scalars().all()
        
        return {
            "article_id": article_id,
            "entity_count": len(rows),
            "entities": [
                {
                    "entity": r.entity_value,
                    "label": r.entity_label,
                    "score": r.score,
                    "span": r.span,
                    "position": {"start": r.start_pos, "end": r.end_pos}
                }
                for r in rows
            ]
        }


@app.get("/entities/search")
async def search_entities(q: str, limit: int = 20, label: Optional[str] = None):
    """Search entities by exact/substring match (case-insensitive)."""
    async with EntitiesSession() as db:
        query = select(ArticleEntity).where(
            func.lower(ArticleEntity.entity_value).like(f"%{q.lower()}%")
        )
        
        if label:
            query = query.where(ArticleEntity.entity_label == label)
        
        query = query.limit(limit)
        result = await db.execute(query)
        rows = result.scalars().all()
        
        return {
            "query": q,
            "label_filter": label,
            "count": len(rows),
            "results": [
                {
                    "article_id": r.article_id,
                    "entity": r.entity_value,
                    "label": r.entity_label,
                    "score": r.score
                }
                for r in rows
            ]
        }


@app.get("/stocks/impacted/{stock_symbol}")
async def get_stock_impacts(stock_symbol: str, limit: int = 20):
    """Get all articles that impact a specific stock symbol."""
    async with EntitiesSession() as db:
        result = await db.execute(
            select(StockImpact)
            .where(StockImpact.stock_symbol == stock_symbol)
            .order_by(StockImpact.created_at.desc())
            .limit(limit)
        )
        impacts = result.scalars().all()
        
        return {
            "stock_symbol": stock_symbol,
            "total_impacts": len(impacts),
            "impacts": [
                {
                    "article_id": i.article_id,
                    "confidence": float(i.confidence),
                    "type": i.impact_type,
                    "sector": i.sector,
                    "reasoning": i.reasoning,
                    "created_at": i.created_at.isoformat()
                }
                for i in impacts
            ]
        }


@app.get("/stocks/by_article/{article_id}")
async def get_article_stock_impacts(article_id: str):
    """Get all stock impacts for a specific article."""
    async with EntitiesSession() as db:
        result = await db.execute(
            select(StockImpact)
            .where(StockImpact.article_id == article_id)
            .order_by(StockImpact.confidence.desc())
        )
        impacts = result.scalars().all()
        
        if not impacts:
            raise HTTPException(status_code=404, detail="No stock impacts found for this article")
        
        return {
            "article_id": article_id,
            "total_stocks": len(impacts),
            "impacted_stocks": [
                {
                    "symbol": i.stock_symbol,
                    "name": i.stock_name,
                    "confidence": float(i.confidence),
                    "type": i.impact_type,
                    "sector": i.sector,
                    "reasoning": i.reasoning
                }
                for i in impacts
            ]
        }


@app.get("/entities/stats")
async def stats():
    """Return counts / basic stats including stock impact data."""
    async with EntitiesSession() as db:
        total_entities = (await db.execute(
            select(func.count()).select_from(ArticleEntity)
        )).scalar() or 0
        
        total_articles = (await db.execute(
            select(func.count()).select_from(ArticleExtractionStatus)
        )).scalar() or 0
        
        total_stock_impacts = (await db.execute(
            select(func.count()).select_from(StockImpact)
        )).scalar() or 0
        
        # Entity label distribution
        label_result = await db.execute(
            select(ArticleEntity.entity_label, func.count())
            .group_by(ArticleEntity.entity_label)
            .order_by(func.count().desc())
        )
        label_distribution = {row[0]: row[1] for row in label_result.all()}
        
        # Impact type distribution
        impact_result = await db.execute(
            select(StockImpact.impact_type, func.count())
            .group_by(StockImpact.impact_type)
        )
        impact_distribution = {row[0]: row[1] for row in impact_result.all()}
        
        # Most impacted stocks
        stock_result = await db.execute(
            select(StockImpact.stock_symbol, func.count())
            .group_by(StockImpact.stock_symbol)
            .order_by(func.count().desc())
            .limit(10)
        )
        top_stocks = {row[0]: row[1] for row in stock_result.all()}
    
    return {
        "total_entities": total_entities,
        "articles_processed": total_articles,
        "total_stock_impacts": total_stock_impacts,
        "primary_model": PRIMARY_MODEL,
        "secondary_model": SECONDARY_MODEL,
        "model_version": MODEL_VERSION,
        "entity_labels": label_distribution,
        "impact_types": impact_distribution,
        "top_impacted_stocks": top_stocks,
        "supported_entity_types": list(set(FINANCIAL_ENTITY_MAP.values()))
    }


@app.post("/entities/rebuild")
async def rebuild_all(limit: Optional[int] = 0):
    """
    Re-extract entities for all articles in ingestion DB.
    WARNING: expensive. Use limit>0 to only process first N.
    """
    logger.info(f"[EntityAgent] Rebuild started (limit={limit or 'all'})")
    
    async with IngestionSession() as ing_db:
        query = select(RawArticle).where(RawArticle.processing_status == "processed")
        if limit and limit > 0:
            query = query.limit(limit)
        
        result = await ing_db.execute(query)
        articles = result.scalars().all()
        
        article_list = [
            {
                "article_id": a.article_id,
                "title": a.title or "",
                "content": a.content or ""
            }
            for a in articles
        ]

    # Clear existing data
    async with EntitiesSession() as db:
        await db.execute(ArticleEntity.__table__.delete())
        await db.execute(ArticleExtractionStatus.__table__.delete())
        await db.commit()
    
    logger.info(f"Cleared existing entities, processing {len(article_list)} articles...")

    processed = 0
    entities_total = 0
    
    for art in article_list:
        try:
            saved = await extract_entities_for_article(art)
            processed += 1
            entities_total += len(saved)
        except Exception as e:
            logger.exception(f"Rebuild failed on {art.get('article_id')}: {e}")

    logger.info(f"[EntityAgent] Rebuild completed. processed={processed} entities={entities_total}")
    return {"processed": processed, "entities_extracted": entities_total}


# ----------------------------
# Health
# ----------------------------
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if (PRIMARY_PIPELINE is not None and SECONDARY_PIPELINE is not None) else "degraded",
        "primary_model_loaded": PRIMARY_PIPELINE is not None,
        "secondary_model_loaded": SECONDARY_PIPELINE is not None,
        "primary_model": PRIMARY_MODEL,
        "secondary_model": SECONDARY_MODEL,
        "supported_entities": list(set(FINANCIAL_ENTITY_MAP.values()))
    }


@app.get("/diagnostics")
async def diagnostics():
    """Diagnostic endpoint to check article availability and processing status."""
    diagnostics = {
        "ingestion_db": {},
        "entities_db": {},
        "stock_mappings": {}
    }
    
    # Check ingestion DB
    async with IngestionSession() as db:
        try:
            # Total articles
            total = (await db.execute(
                select(func.count()).select_from(RawArticle)
            )).scalar() or 0
            
            # By status
            status_result = await db.execute(
                select(RawArticle.processing_status, func.count())
                .group_by(RawArticle.processing_status)
            )
            status_breakdown = {row[0]: row[1] for row in status_result.all()}
            
            # Sample processed articles
            sample_result = await db.execute(
                select(RawArticle)
                .limit(5)
            )
            samples = sample_result.scalars().all()
            
            diagnostics["ingestion_db"] = {
                "total_articles": total,
                "status_breakdown": status_breakdown,
                "processed_count": status_breakdown.get("processed", 0),
                "pending_count": status_breakdown.get("pending", 0),
                "sample_articles": [
                    {
                        "article_id": a.article_id,
                        "title": a.title[:100] if a.title else None,
                        "status": a.processing_status,
                        "has_content": bool((a.content or "").strip())
                    }
                    for a in samples
                ]
            }
        except Exception as e:
            diagnostics["ingestion_db"]["error"] = str(e)
    
    # Check entities DB
    async with EntitiesSession() as db:
        try:
            extracted_count = (await db.execute(
                select(func.count()).select_from(ArticleExtractionStatus)
            )).scalar() or 0
            
            entity_count = (await db.execute(
                select(func.count()).select_from(ArticleEntity)
            )).scalar() or 0
            
            stock_impact_count = (await db.execute(
                select(func.count()).select_from(StockImpact)
            )).scalar() or 0
            
            diagnostics["entities_db"] = {
                "articles_extracted": extracted_count,
                "total_entities": entity_count,
                "total_stock_impacts": stock_impact_count
            }
        except Exception as e:
            diagnostics["entities_db"]["error"] = str(e)
    
    # Stock mappings
    diagnostics["stock_mappings"] = {
        "total_stock_mappings": len(STOCK_SYMBOL_MAP),
        "total_sectors": len(SECTOR_STOCK_MAP),
        "mapping_file_exists": STOCK_MAPPING_FILE.exists(),
        "sector_file_exists": SECTOR_MAPPING_FILE.exists()
    }
    
    # Recommendations
    recommendations = []
    if diagnostics["ingestion_db"].get("total_articles", 0) == 0:
        recommendations.append("⚠️ No articles in ingestion DB. Run ingestion agent first.")
    elif diagnostics["entities_db"].get("articles_extracted", 0) == 0:
        recommendations.append("✅ Ready to extract! Run POST /entities/extract to begin.")
    else:
        extracted = diagnostics["entities_db"].get("articles_extracted", 0)
        total = diagnostics["ingestion_db"].get("total_articles", 0)
        if extracted < total:
            recommendations.append(f"ℹ️ Extracted {extracted}/{total} articles. Run POST /entities/extract to continue.")
        else:
            recommendations.append(f"✅ All {extracted} articles extracted!")
    
    diagnostics["recommendations"] = recommendations
    
    return diagnostics


# ----------------------------
# Run (dev)
# ----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)

"""
# 1. Start the entity extraction service
uvicorn agents.entity_extraction_agent:app --reload --port 8004

# 2. After startup, test with example:

# Extract entities from articles
$body = @{ limit = 10 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8004/entities/extract" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body

# Get analysis for specific article (replace with actual article_id)
$body = @{ article_id = "YOUR_ARTICLE_ID" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8004/entities/extract/article" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body

# Check which articles impact HDFCBANK stock
Invoke-RestMethod -Uri "http://localhost:8004/stocks/impacted/HDFCBANK"

# Get stock impacts for a specific article
Invoke-RestMethod -Uri "http://localhost:8004/stocks/by_article/YOUR_ARTICLE_ID"

# View stats
Invoke-RestMethod -Uri "http://localhost:8004/entities/stats"

# Health check
Invoke-RestMethod -Uri "http://localhost:8004/health"
"""