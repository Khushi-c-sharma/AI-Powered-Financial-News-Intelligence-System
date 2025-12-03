"""
Async Financial News Ingestion Agent (FastAPI)

Features:
- Async RSS ingestion using aiohttp + feedparser (non-blocking)
- Async HTTP API ingestion using aiohttp (non-blocking)
- SQLite (aiosqlite) via SQLAlchemy async engine
- Deduplication by URL and content hash
- Lightweight ingestion logging table
- FastAPI endpoints to trigger ingests and query status
- Clear docstrings and inline comments

Run (development):
    uvicorn ingestion_agent:app --reload --host 0.0.0.0 --port 8000

Dependencies (pip):
    aiohttp
    feedparser
    beautifulsoup4
    fastapi
    uvicorn[standard]
    sqlalchemy>=1.4
    aiosqlite
    pydantic

Note: ensure the project root resolution (PROJECT_ROOT) matches your layout.
"""

import uuid
import json
import hashlib
import logging
import asyncio
from typing import Optional, List
from enum import Enum
from pathlib import Path

import feedparser
import aiohttp
from bs4 import BeautifulSoup
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, Text, TIMESTAMP, Integer, select
from pydantic import BaseModel, HttpUrl
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import uvicorn

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

Base = declarative_base()

# ---------------------------------------------------------
# Time helper
# ---------------------------------------------------------
def now():
    """Return current UTC-aware datetime."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------
# ENUMS & CONFIG
# ---------------------------------------------------------
class ProcessingStatus(str, Enum):
    """Possible processing states for an article."""
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    DUPLICATE = "duplicate"


class SourceConfig:
    """
    Per-source ingestion defaults and thresholds.

    MIN_CONTENT_LENGTH: preferred minimum length for content to be considered
    substantial (Economic Times often has short teasers so value lowered).
    MIN_ABSOLUTE_LENGTH: absolute minimum allowed content length.
    """
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds between retries
    REQUEST_TIMEOUT = 15  # seconds
    MIN_CONTENT_LENGTH = 20   # ET-friendly
    MIN_ABSOLUTE_LENGTH = 5   # absolute minimum


# ---------------------------------------------------------
# PROJECT & DATABASE CONFIG (SQLite)
# ---------------------------------------------------------
# Resolve project root (folder above this file's parent -> adjust if needed)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_DIR = PROJECT_ROOT / "data" / "ingestion"
DB_DIR.mkdir(parents=True, exist_ok=True)  # ensure directory exists

DB_PATH = DB_DIR / "raw_articles.db"
DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

# Create async engine & session factory
engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


# ---------------------------------------------------------
# DATABASE MODELS
# ---------------------------------------------------------
class RawArticle(Base):
    """
    Stores raw article content and metadata ingested from various sources.
    Primary keys use UUID strings (flexible and safe across distributed ingestion).
    """
    __tablename__ = "raw_articles"

    id = Column(String, primary_key=True)  # internal UUID
    article_id = Column(String, unique=True, nullable=False, index=True)  # external/secondary id
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), index=True)
    source = Column(String, nullable=False, index=True)
    url = Column(String, unique=True, index=True)
    author = Column(String)
    published_date = Column(TIMESTAMP(timezone=True), nullable=False, index=True)
    ingested_date = Column(TIMESTAMP(timezone=True), default=now)
    language = Column(String, default="en")
    category = Column(String)
    raw_metadata = Column(Text)
    processing_status = Column(String, default=ProcessingStatus.PENDING, index=True)
    error_log = Column(Text)
    retry_count = Column(Integer, default=0)


class IngestionLog(Base):
    """
    Simple audit/log table that keeps a summary of each ingestion run.
    Useful for debugging and monitoring ingestion health.
    """
    __tablename__ = "ingestion_logs"

    id = Column(String, primary_key=True)
    source = Column(String, nullable=False)
    started_at = Column(TIMESTAMP(timezone=True), default=now)
    completed_at = Column(TIMESTAMP(timezone=True))
    articles_fetched = Column(Integer, default=0)
    articles_stored = Column(Integer, default=0)
    duplicates_found = Column(Integer, default=0)
    errors = Column(Integer, default=0)
    status = Column(String, default="running")
    error_details = Column(Text, default="")


# ---------------------------------------------------------
# Pydantic models (API request / response validation)
# ---------------------------------------------------------
class IngestionResult(BaseModel):
    """Response model summarizing an ingestion run."""
    status: str
    articles_fetched: int
    articles_stored: int
    duplicates_found: int
    errors: int
    error_details: Optional[List[str]] = None


class RSSFeedConfig(BaseModel):
    """Configuration to trigger an RSS ingest via API."""
    feed_url: HttpUrl
    source_name: str
    category: Optional[str] = "General"


class APIConfig(BaseModel):
    """Configuration to trigger an HTTP-JSON API ingest via API."""
    api_url: HttpUrl
    source_name: str
    category: Optional[str] = "General"
    api_key: Optional[str] = None


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------
def compute_content_hash(content: str) -> str:
    """Compute SHA256 hex digest for deduplication by content."""
    return hashlib.sha256((content or "").encode("utf-8")).hexdigest()


def clean_html(raw_html: str) -> str:
    """
    Remove HTML tags and return a cleaned, whitespace-normalized text string.
    Uses BeautifulSoup for HTML parsing and sanitization.
    """
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


def validate_article_data(title: str, content: str, url: str):
    """
    Validate minimal quality constraints for an article.
    - Accept short teasers (ET-style) but reject empty/invalid content and URLs.
    Returns (bool, error_message_or_None).
    """
    if not title or len(title.strip()) < 3:
        return False, "Title too short"

    content_clean = (content or "").strip()
    if len(content_clean) < SourceConfig.MIN_ABSOLUTE_LENGTH:
        return False, "Content empty or invalid"

    if len(content_clean) < SourceConfig.MIN_CONTENT_LENGTH:
        # Accept short content but emit a debug log to make it visible
        logger.debug("Short teaser accepted (%d chars): %s", len(content_clean), content_clean[:80])

    if not url or not url.startswith(("http://", "https://")):
        return False, "Invalid URL"

    return True, None


async def check_duplicate_by_url(db: AsyncSession, url: str) -> bool:
    """Return True if an article with the same URL already exists in DB."""
    result = await db.execute(select(RawArticle).where(RawArticle.url == url))
    return result.scalar_one_or_none() is not None


async def check_duplicate_by_hash(db: AsyncSession, content_hash: str) -> bool:
    """Return True if an article with the same content_hash already exists in DB."""
    result = await db.execute(select(RawArticle).where(RawArticle.content_hash == content_hash))
    return result.scalar_one_or_none() is not None


# ---------------------------------------------------------
# Async network helpers
# ---------------------------------------------------------
async def fetch_text(session: aiohttp.ClientSession, url: str, timeout: int = SourceConfig.REQUEST_TIMEOUT) -> str:
    """
    Fetch URL as text using aiohttp (async). Raises for non-2xx status.
    Returns the response body decoded as UTF-8 (best-effort).
    """
    try:
        async with session.get(url, timeout=timeout) as resp:
            resp.raise_for_status()
            b = await resp.read()
            return b.decode("utf-8", errors="ignore")
    except Exception:
        raise


async def fetch_json(session: aiohttp.ClientSession, url: str, headers: dict = None, timeout: int = SourceConfig.REQUEST_TIMEOUT):
    """
    Fetch URL and parse JSON body. Raises on HTTP error or invalid JSON.
    """
    headers = headers or {}
    try:
        async with session.get(url, headers=headers, timeout=timeout) as resp:
            resp.raise_for_status()
            return await resp.json()
    except Exception:
        raise


# ---------------------------------------------------------
# RSS ingestion (async)
# ---------------------------------------------------------
async def ingest_from_rss(feed_url: str, source_name: str, category: str, db: AsyncSession) -> IngestionResult:
    """
    Ingest articles from an RSS/Atom feed URL in a non-blocking manner:
    - fetch feed via aiohttp
    - parse with feedparser.parse(xml)
    - clean and validate content
    - deduplicate and insert into DB
    Returns an IngestionResult summarizing the run.
    """
    log_id = str(uuid.uuid4())
    log = IngestionLog(
        id=log_id,
        source=f"RSS:{source_name}",
        started_at=now(),
        articles_fetched=0,
        articles_stored=0,
        duplicates_found=0,
        errors=0,
        status="running",
        error_details=""
    )
    db.add(log)
    error_details: List[str] = []

    # Use aiohttp to fetch the RSS feed asynchronously
    async with aiohttp.ClientSession() as session:
        feed_data = None
        for attempt in range(1, SourceConfig.MAX_RETRIES + 1):
            try:
                xml = await fetch_text(session, feed_url)
                feed_data = feedparser.parse(xml)
                break
            except Exception as e:
                logger.warning("Attempt %d: failed to fetch/parse RSS %s: %s", attempt, feed_url, e)
                if attempt == SourceConfig.MAX_RETRIES:
                    # record failure and re-raise after updating log
                    log.status = "failed"
                    log.completed_at = now()
                    log.error_details = f"Failed fetching RSS after {attempt} attempts: {e}"
                    await db.commit()
                    return IngestionResult(
                        status=log.status,
                        articles_fetched=0,
                        articles_stored=0,
                        duplicates_found=0,
                        errors=1,
                        error_details=[str(e)]
                    )
                await asyncio.sleep(SourceConfig.RETRY_DELAY)

    # If parsing succeeded, process entries
    entries = getattr(feed_data, "entries", []) or []
    log.articles_fetched = len(entries)

    for entry in entries:
        try:
            url = entry.get("link", "") or entry.get("id", "")
            title = entry.get("title", "") or ""
            # prefer content -> summary -> description
            raw_content = ""
            if entry.get("content"):
                # feedparser content might be a list of dicts
                cont = entry.get("content")
                if isinstance(cont, list) and cont:
                    raw_content = cont[0].get("value", "")
                elif isinstance(cont, dict):
                    raw_content = cont.get("value", "")
            if not raw_content:
                raw_content = entry.get("summary") or entry.get("description") or ""
            summary = clean_html(raw_content)
            author = entry.get("author", "") or entry.get("author_detail", {}).get("name", "")

            # Validate
            is_valid, error_msg = validate_article_data(title, summary, url)
            if not is_valid:
                log.errors += 1
                error_details.append(f"{url}: {error_msg}")
                continue

            # Dedup checks
            if await check_duplicate_by_url(db, url):
                log.duplicates_found += 1
                continue

            content_hash = compute_content_hash(summary)
            if await check_duplicate_by_hash(db, content_hash):
                log.duplicates_found += 1
                continue

            # Published date parsing: feedparser may provide published_parsed
            published_parsed = entry.get("published_parsed") or entry.get("updated_parsed")
            if published_parsed:
                try:
                    published_at = datetime(*published_parsed[:6], tzinfo=timezone.utc)
                except Exception:
                    published_at = now()
            else:
                published_at = now()

            metadata = {
                "type": "rss",
                "feed_url": feed_url,
                "tags": entry.get("tags", [])
            }

            # Build RawArticle object
            article = RawArticle(
                id=str(uuid.uuid4()),
                article_id=str(uuid.uuid4()),
                title=title,
                content=summary,
                content_hash=content_hash,
                source=source_name,
                url=url,
                author=author,
                published_date=published_at,
                ingested_date=now(),
                language="en",
                category=category,
                raw_metadata=json.dumps(metadata)
            )

            db.add(article)
            log.articles_stored += 1

        except Exception as e:
            logger.exception("Error processing RSS entry: %s", e)
            log.errors += 1
            error_details.append(str(e))

    # Finalize log entry
    log.status = "completed"
    log.completed_at = now()
    await db.commit()

    return IngestionResult(
        status=log.status,
        articles_fetched=log.articles_fetched,
        articles_stored=log.articles_stored,
        duplicates_found=log.duplicates_found,
        errors=log.errors,
        error_details=error_details or None
    )


# ---------------------------------------------------------
# HTTP JSON API ingestion (async)
# ---------------------------------------------------------
async def ingest_from_http_api(api_url: str, source_name: str, category: str, api_key: Optional[str], db: AsyncSession):
    """
    Ingest a JSON-style HTTP API (e.g., news endpoints that return {articles: [...]})
    - uses aiohttp to fetch JSON
    - applies same validation/dedup insertion as RSS path
    """
    log_id = str(uuid.uuid4())
    log = IngestionLog(
        id=log_id,
        source=f"API:{source_name}",
        started_at=now(),
        articles_fetched=0,
        articles_stored=0,
        duplicates_found=0,
        errors=0,
        status="running",
        error_details=""
    )
    db.add(log)
    error_details: List[str] = []

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with aiohttp.ClientSession() as session:
        data = None
        for attempt in range(1, SourceConfig.MAX_RETRIES + 1):
            try:
                data = await fetch_json(session, api_url, headers=headers)
                break
            except Exception as e:
                logger.warning("Attempt %d: failed to fetch API %s: %s", attempt, api_url, e)
                if attempt == SourceConfig.MAX_RETRIES:
                    log.status = "failed"
                    log.completed_at = now()
                    log.error_details = f"Failed fetching API after {attempt} attempts: {e}"
                    await db.commit()
                    return IngestionResult(
                        status=log.status,
                        articles_fetched=0,
                        articles_stored=0,
                        duplicates_found=0,
                        errors=1,
                        error_details=[str(e)]
                    )
                await asyncio.sleep(SourceConfig.RETRY_DELAY)

    # Expecting JSON object with 'articles' key (common pattern)
    articles = (data or {}).get("articles", [])
    log.articles_fetched = len(articles)

    for item in articles:
        try:
            url = item.get("url", "")
            title = item.get("title", "") or ""
            raw_content = item.get("content") or item.get("description") or ""
            content = clean_html(raw_content)
            author = item.get("author", "") or ""

            # Validate
            is_valid, error_msg = validate_article_data(title, content, url)
            if not is_valid:
                log.errors += 1
                error_details.append(f"{url}: {error_msg}")
                continue

            # Dedup by URL & hash
            if await check_duplicate_by_url(db, url):
                log.duplicates_found += 1
                continue

            content_hash = compute_content_hash(content)
            if await check_duplicate_by_hash(db, content_hash):
                log.duplicates_found += 1
                continue

            published_at_raw = item.get("publishedAt") or item.get("published_date")
            if published_at_raw:
                try:
                    published_at = datetime.fromisoformat(published_at_raw.replace("Z", "+00:00"))
                except Exception:
                    published_at = now()
            else:
                published_at = now()

            metadata = {
                "type": "http-api",
                "api_url": api_url,
                "source_metadata": item.get("source", {})
            }

            article = RawArticle(
                id=str(uuid.uuid4()),
                article_id=str(uuid.uuid4()),
                title=title,
                content=content,
                content_hash=content_hash,
                source=source_name,
                url=url,
                author=author,
                published_date=published_at,
                ingested_date=now(),
                language="en",
                category=category,
                raw_metadata=json.dumps(metadata)
            )

            db.add(article)
            log.articles_stored += 1

        except Exception as e:
            logger.exception("Error processing API item: %s", e)
            log.errors += 1
            error_details.append(str(e))

    log.status = "completed"
    log.completed_at = now()
    await db.commit()

    return IngestionResult(
        status=log.status,
        articles_fetched=log.articles_fetched,
        articles_stored=log.articles_stored,
        duplicates_found=log.duplicates_found,
        errors=log.errors,
        error_details=error_details or None
    )


# ---------------------------------------------------------
# FASTAPI LIFESPAN: initialize DB on startup
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context: create DB tables before serving requests.
    This ensures the DB file and schema exist.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("SQLite DB initialized at %s", DB_PATH)
    yield


# ---------------------------------------------------------
# FASTAPI APP INIT + ROUTES
# ---------------------------------------------------------
app = FastAPI(
    title="Financial News Ingestion Agent",
    version="1.0.2",
    lifespan=lifespan
)


@app.post("/ingest/rss", response_model=IngestionResult)
async def trigger_rss_ingest(config: RSSFeedConfig):
    """
    Trigger an RSS ingestion run for a single feed.
    Example payload:
    {
      "feed_url": "https://example.com/rss",
      "source_name": "Example News",
      "category": "Markets"
    }
    """
    async with AsyncSessionLocal() as db:
        return await ingest_from_rss(
            str(config.feed_url),
            config.source_name,
            config.category or "General",
            db
        )


@app.post("/ingest/api", response_model=IngestionResult)
async def trigger_api_ingest(config: APIConfig):
    """
    Trigger an HTTP-JSON API ingestion run for a single endpoint.
    Example payload:
    {
      "api_url": "https://example.com/newsapi",
      "source_name": "Example API",
      "api_key": "optional"
    }
    """
    async with AsyncSessionLocal() as db:
        return await ingest_from_http_api(
            str(config.api_url),
            config.source_name,
            config.category or "General",
            config.api_key,
            db
        )


@app.get("/health")
async def health():
    """Simple healthcheck route."""
    return {"status": "healthy"}


@app.get("/stats")
async def stats():
    """
    Return quick stats: total articles, pending count.
    Note: in a production system you'd implement efficient aggregation queries
    rather than fetching full rows into memory.
    """
    async with AsyncSessionLocal() as db:
        total_rows = (await db.execute(select(RawArticle))).scalars().all()
        pending_rows = (await db.execute(
            select(RawArticle).where(RawArticle.processing_status == ProcessingStatus.PENDING)
        )).scalars().all()

        return {
            "total_articles": len(total_rows),
            "pending_processing": len(pending_rows)
        }


@app.get("/articles/pending")
async def get_pending_articles(limit: int = 10):
    """
    Fetch a small list of pending articles for downstream processors to pick up.
    Returns minimal metadata for queueing/inspection.
    """
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(RawArticle)
            .where(RawArticle.processing_status == ProcessingStatus.PENDING)
            .order_by(RawArticle.ingested_date.desc())
            .limit(limit)
        )
        articles = result.scalars().all()

        return {
            "count": len(articles),
            "articles": [
                {
                    "id": a.article_id,
                    "title": a.title,
                    "source": a.source,
                    "published_date": a.published_date.isoformat() if a.published_date else None,
                    "ingested_date": a.ingested_date.isoformat() if a.ingested_date else None
                }
                for a in articles
            ]
        }


# ---------------------------------------------------------
# MAIN (for dev run)
# ---------------------------------------------------------
if __name__ == "__main__":
    # Use uvicorn.run for convenience when launching the script directly.
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
uvicorn ingestion_agent:app --host 0.0.0.0 --port 8000 --reload

### PowerShell script to ingest all sources from sources.json
$sources = (Get-Content ".\sources.json" | ConvertFrom-Json).sources

foreach ($s in $sources) {
    Write-Host "Ingesting: $($s.name) [$($s.category)]" -ForegroundColor Cyan

    $body = @{
        feed_url    = $s.url
        source_name = $s.name
        category    = $s.category
    } | ConvertTo-Json

    Invoke-RestMethod -Uri "http://localhost:8000/ingest/rss" `
                      -Method Post `
                      -ContentType "application/json" `
                      -Body $body

    Start-Sleep -Seconds 1   # prevents overwhelming sources
}
"""
