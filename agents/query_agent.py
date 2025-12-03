# query_agent.py
import os
import logging
from typing import List, Dict, Any, Optional
import httpx # Used for making requests to the Storage Agent
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - QueryAgent - %(levelname)s - %(message)s"
)
logger = logging.getLogger("QueryAgent")

# -----------------------------
# Config
# -----------------------------
# The URL for the Storage Agent (Must match the port the agent is running on, e.g., 8006)
STORAGE_AGENT_URL = os.getenv("STORAGE_AGENT_URL", "http://localhost:8006")
HYBRID_SEARCH_ENDPOINT = f"{STORAGE_AGENT_URL}/search/hybrid"
HTTPX_TIMEOUT = 10.0 # Timeout for Storage Agent request

# -----------------------------
# Pydantic Models
# -----------------------------

class HybridQueryRequest(BaseModel):
    """Input request model for the Query Agent's main endpoint."""
    query_text: str
    target_symbols: List[str] = []
    target_sectors: List[str] = []
    
    # Ranking Configuration
    max_candidates: int = 50
    semantic_weight: float = 0.5
    impact_weight: float = 0.5

class FinalImpactResult(BaseModel):
    """Final output model for the Query Agent."""
    story_id: str
    canonical_title: str
    summary: str
    
    final_hybrid_score: float
    
    # Simplified structure to show the core impact for the user
    impact_summary: List[Dict[str, Any]]

# --- Models from Storage Agent (Duplicated for clarity) ---

class ImpactedStockMetadata(BaseModel):
    """Represents a single stock impact entry from the Storage Agent."""
    symbol: str
    stock_name: str
    confidence: float
    impact_type: str
    effect: Optional[Dict[str, Any]] # Contains direction, magnitude, etc.

class HybridSearchResult(BaseModel):
    """The shape of the result returned by the Storage Agent's /search/hybrid endpoint."""
    story_id: str
    canonical_title: str
    summary: str
    semantic_score: float
    # Metadata contains entities, impacted_stocks, and hybrid_score_prelim
    metadata: Dict[str, Any] 


# -----------------------------
# App Initialization
# -----------------------------
app = FastAPI(title="Query Orchestration Agent", version="1.0")
# Initialize async HTTP client for external service calls
http_client = httpx.AsyncClient(timeout=HTTPX_TIMEOUT)


# -----------------------------
# Hybrid Ranking Logic
# -----------------------------

def calculate_impact_score(metadata: Dict[str, Any], req: HybridQueryRequest) -> float:
    """
    Calculates a score (0.0 to 1.0) based on how strongly the story aligns 
    with the user's explicit target symbols and sectors.
    """
    target_symbols_set = {s.upper() for s in req.target_symbols}
    target_sectors_set = {s.lower() for s in req.target_sectors}
    
    total_impact_score = 0.0
    impacted_stocks = metadata.get("impacted_stocks", [])
    
    # Scoring Weights
    SYMBOL_MATCH_REWARD = 2.0
    SECTOR_MATCH_REWARD = 1.0
    
    # 1. Scoring Logic
    for stock_impact in impacted_stocks:
        symbol = stock_impact.get("symbol", "").upper()
        sector = stock_impact.get("sector", "").lower()
        confidence = stock_impact.get("confidence", 0.5)
        
        # Reward symbol match (weighted by confidence)
        if symbol in target_symbols_set:
            total_impact_score += confidence * SYMBOL_MATCH_REWARD
            
        # Reward sector match (weighted by confidence)
        if sector and sector in target_sectors_set:
            total_impact_score += confidence * SECTOR_MATCH_REWARD

    # 2. Normalization
    max_possible_score = (len(target_symbols_set) * SYMBOL_MATCH_REWARD) + \
                         (len(target_sectors_set) * SECTOR_MATCH_REWARD)
    
    if max_possible_score > 0:
        # Normalize score to 0.0-1.0 range
        return min(total_impact_score / max_possible_score, 1.0)
    
    # If no targets were specified, the impact score is 1.0 (no penalty for relevancy)
    return 1.0


def calculate_hybrid_score(semantic_score: float, impact_score: float, req: HybridQueryRequest) -> float:
    """Combines semantic and impact scores using configured weights."""
    return (semantic_score * req.semantic_weight) + (impact_score * req.impact_weight)


# -----------------------------
# Main Endpoint
# -----------------------------

@app.post("/query/impact", response_model=List[FinalImpactResult])
async def query_impact(req: HybridQueryRequest):
    """
    Orchestrates search:
    1. Calls Storage Agent for candidates (semantic + preliminary metadata).
    2. Filters candidates based on mandatory targets.
    3. Calculates final hybrid score (semantic + impact).
    4. Ranks and returns top results.
    """
    logger.info(f"[Query] Received query: '{req.query_text}'. Targets: {req.target_symbols}")

    try:
        # 1. Retrieve Candidates from Storage Agent
        storage_request_body = {
            "query_text": req.query_text,
            "target_symbols": req.target_symbols,
            "target_sectors": req.target_sectors,
            "max_candidates": req.max_candidates,
            "semantic_only": False 
        }
        
        # Use httpx to call the Storage Agent's /search/hybrid endpoint
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
            response = await client.post(HYBRID_SEARCH_ENDPOINT, json=storage_request_body)
            response.raise_for_status() 
        
        raw_candidates: List[Dict[str, Any]] = response.json()
        
    except httpx.HTTPStatusError as e:
        logger.error(f"Storage Agent returned error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(502, f"Failed to retrieve candidates from Storage Agent: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Failed to connect to Storage Agent: {e}")
        raise HTTPException(503, "Could not connect to the Storage Agent.")

    logger.info(f"[Query] Retrieved {len(raw_candidates)} candidates from Storage Agent.")

    final_results: List[FinalImpactResult] = []
    target_symbols_set = {s.upper() for s in req.target_symbols}

    for candidate in raw_candidates:
        metadata = candidate["metadata"]
        
        # 2. Mandatory Filtering (If symbols are requested, the story MUST contain at least one)
        if target_symbols_set:
            story_symbols_set = {s.get("symbol", "").upper() for s in metadata.get("impacted_stocks", [])}
            
            # If the story's symbols do not intersect with the required targets, discard it.
            if not target_symbols_set.intersection(story_symbols_set):
                continue
                
        # 3. Calculate Scores
        semantic_score = candidate["semantic_score"]
        impact_score = calculate_impact_score(metadata, req)
        final_hybrid_score = calculate_hybrid_score(semantic_score, impact_score, req)
        
        # 4. Format Impact Summary for the user
        impact_summary = []
        for stock_impact in metadata.get("impacted_stocks", []):
            symbol = stock_impact.get("symbol", "").upper()
            
            # Only include impacts that match the user's targets in the final summary
            if symbol in target_symbols_set:
                effect = stock_impact.get("effect", {})
                impact_summary.append({
                    "symbol": stock_impact["symbol"],
                    "stock_name": stock_impact["stock_name"],
                    "confidence": stock_impact["confidence"],
                    "direction": effect.get("direction"),
                    "magnitude": effect.get("magnitude")
                })
        
        # 5. Add to final results
        final_results.append(
            FinalImpactResult(
                story_id=candidate["story_id"],
                canonical_title=candidate["canonical_title"],
                summary=candidate["summary"],
                final_hybrid_score=final_hybrid_score,
                impact_summary=impact_summary
            )
        )

    # Final sorting by the comprehensive score
    final_results.sort(key=lambda x: x.final_hybrid_score, reverse=True)
    
    logger.info(f"[Query] Filtered and ranked {len(final_results)} final results.")
    return final_results


@app.get("/diagnostics")
async def diagnostics():
    """Checks connectivity to the Storage Agent."""
    storage_status = "unreachable"
    try:
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
            response = await client.get(f"{STORAGE_AGENT_URL}/health")
            response.raise_for_status()
            storage_status = response.json().get("status", "ok")
    except Exception:
        pass
        
    return {
        "status": "ok",
        "storage_agent_status": storage_status,
        "config": {
            "storage_agent_url": STORAGE_AGENT_URL
        }
    }




