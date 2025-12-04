import os
import json
import logging
from typing import List, Dict, Any, Optional, TypedDict
from datetime import datetime, timezone
import httpx
from dotenv import load_dotenv

# LangChain/LangGraph dependencies
from langgraph.graph import StateGraph, END
from groq import Groq

# FastAPI and utilities
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

load_dotenv()

# -----------------------------
# Config & Setup
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - Orchestrator - %(levelname)s - %(message)s")
logger = logging.getLogger("Orchestrator")

# Ensure environment variables are set
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QUERY_AGENT_URL = os.getenv("QUERY_AGENT_URL", "http://localhost:8007")
HYBRID_QUERY_ENDPOINT = f"{QUERY_AGENT_URL}/query/impact"

if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not found. Please set it in your environment.")

# Initialize clients (synchronous clients are fine here)
groq_client = Groq(api_key=GROQ_API_KEY)
http_client = httpx.Client(timeout=15.0) # Increased timeout slightly for API calls

# -----------------------------
# 1. Graph State Definition
# -----------------------------

class AgentState(TypedDict):
    """The state of the LangGraph workflow."""
    user_query: str
    target_symbols: List[str]
    query_text_for_search: str
    query_agent_response: List[Dict[str, Any]]
    final_answer: Optional[str]


# -----------------------------
# 2. Graph Nodes (Functions)
# -----------------------------

def extract_targets(state: AgentState) -> AgentState:
    """
    Node 1: Uses LLM to extract target symbols and a clean search query.
    """
    logger.info("Executing Node: extract_targets (LLM Call)")
    user_query = state["user_query"]
    
    prompt = f"""
    You are an expert financial query parser. Your task is to analyze the user's request and extract two things:
    1. 'symbols': A list of all relevant stock tickers/symbols mentioned.
    2. 'search_query': A concise, factual query suitable for a hybrid semantic search engine.
    
    If no specific symbols are mentioned, return an empty list for 'symbols'.
    
    User Request: "{user_query}"
    
    Respond STRICTLY in the following JSON format ONLY:
    {{
        "symbols": ["TCS", "INFY"],
        "search_query": "news impacting indian IT sector"
    }}
    """
    
    try:
        response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}], # Fix applied here
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        output = json.loads(response.choices[0].message.content)
        
        state["target_symbols"] = output.get("symbols", [])
        state["query_text_for_search"] = output.get("search_query", user_query)
        logger.info(f"Extracted Symbols: {state['target_symbols']}, Search Query: {state['query_text_for_search']}")
        
    except Exception as e:
        logger.error(f"Error during LLM extraction: {e}")
        state["target_symbols"] = []
        state["query_text_for_search"] = user_query
    
    return state


def run_hybrid_query(state: AgentState) -> AgentState:
    """
    Node 2: Calls the Query Agent.
    """
    logger.info("Executing Node: run_hybrid_query (API Call to Query Agent)")
    
    request_body = {
        "query_text": state["query_text_for_search"],
        "target_symbols": state["target_symbols"],
        "semantic_weight": 0.5, 
        "impact_weight": 0.5,
        "max_candidates": 5
    }
    
    try:
        response = http_client.post(HYBRID_QUERY_ENDPOINT, json=request_body)
        response.raise_for_status()
        
        state["query_agent_response"] = response.json()
        logger.info(f"Query Agent returned {len(state['query_agent_response'])} stories.")
        
    except httpx.HTTPStatusError as e:
        logger.error(f"Query Agent failed with status {e.response.status_code}: {e.response.text}")
        state["query_agent_response"] = []
    except httpx.RequestError as e:
        logger.error(f"Failed to connect to Query Agent at {QUERY_AGENT_URL}: {e}")
        state["query_agent_response"] = []

    return state


def generate_final_answer(state: AgentState) -> AgentState:
    """
    Node 3: Uses LLM to synthesize the results into a conversational, final answer.
    """
    logger.info("Executing Node: generate_final_answer (LLM Synthesis)")
    
    if not state["query_agent_response"]:
        state["final_answer"] = f"I could not find any relevant or impactful stories for your query: '{state['user_query']}'. The search may have been too specific or the indexed data is limited."
        return state
    
    context = []
    for story in state["query_agent_response"]:
        impact_details = json.dumps(story.get('impact_summary', {})) if story.get('impact_summary') else "No explicit impact analysis found for requested symbols."
        
        context.append(f"""
        ---
        STORY TITLE: {story['canonical_title']}
        SUMMARY: {story['summary']}
        FINAL SCORE: {story['final_hybrid_score']:.3f}
        IMPACT: {impact_details}
        ---
        """)

    context_str = "\n".join(context)

    prompt = f"""
    You are a professional financial news analyst. Your goal is to synthesize the provided news context into a single, concise, conversational answer for the user.
    
    1. Start by directly answering the user's original query.
    2. Summarize the key findings from the most highly-ranked stories.
    3. Specifically mention the directional stock impact (Positive/Negative/Neutral) and confidence level where available for the requested stocks.
    
    USER QUERY: {state["user_query"]}
    
    --- CONTEXTUAL NEWS STORIES ---
    {context_str}
    """
    
    try:
        response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You are a helpful and concise financial news synthesizer."}, 
                {"role": "user", "content": prompt}
            ],
            temperature=0.2 
        )
        
        state["final_answer"] = response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error during final answer generation: {e}")
        state["final_answer"] = "An error occurred during final synthesis, but here are the top story summaries: " + "\n".join([s['summary'] for s in state["query_agent_response"]])
        
    return state


# -----------------------------
# 3. LangGraph Assembly
# -----------------------------

def create_orchestrator():
    """Assembles and compiles the LangGraph state machine."""
    workflow = StateGraph(AgentState)

    workflow.add_node("extract_targets", extract_targets)
    workflow.add_node("run_hybrid_query", run_hybrid_query)
    workflow.add_node("generate_final_answer", generate_final_answer)

    workflow.set_entry_point("extract_targets")

    workflow.add_edge("extract_targets", "run_hybrid_query")
    workflow.add_edge("run_hybrid_query", "generate_final_answer")

    workflow.add_edge("generate_final_answer", END)

    app_graph = workflow.compile()
    logger.info("LangGraph Orchestrator compiled successfully.")
    return app_graph

# Create the runnable graph instance once at startup
orchestrator = create_orchestrator()

# -----------------------------
# 4. FastAPI Wrapper and UI
# -----------------------------
app = FastAPI(title="LangGraph Financial Orchestrator Service", version="1.0.0")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    time_taken: float
    symbols_extracted: List[str]

@app.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    """
    Endpoint to run the full LangGraph orchestration.
    """
    start_time = datetime.now(timezone.utc)
    
    initial_state = AgentState(
        user_query=request.query, 
        target_symbols=[], 
        query_text_for_search="", 
        query_agent_response=[], 
        final_answer=None
    )
    
    logger.info(f"Received query: {request.query}")
    
    # Run the synchronous graph execution
    final_state = orchestrator.invoke(initial_state)
    
    end_time = datetime.now(timezone.utc)
    time_taken = (end_time - start_time).total_seconds()

    return QueryResponse(
        answer=final_state['final_answer'] or "The orchestration failed to produce a final answer.",
        time_taken=round(time_taken, 2),
        symbols_extracted=final_state['target_symbols']
    )


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """
    Serves the simple HTML/JS interface for querying the orchestrator.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Financial News Orchestrator</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 20px; background-color: #f0f2f5; }}
            .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); }}
            h1 {{ color: #1a73e8; border-bottom: 2px solid #e0e0e0; padding-bottom: 10px; margin-top: 0; }}
            textarea, button {{ width: 100%; padding: 12px; margin-top: 10px; border-radius: 6px; box-sizing: border-box; }}
            textarea {{ height: 100px; border: 1px solid #ccc; resize: vertical; font-size: 16px; }}
            button {{ background-color: #1a73ee; color: white; border: none; cursor: pointer; font-size: 18px; font-weight: 600; transition: background-color 0.3s; }}
            button:hover:not(:disabled) {{ background-color: #1558b0; }}
            button:disabled {{ background-color: #a8c1f0; cursor: not-allowed; }}
            .response-box {{ margin-top: 25px; padding: 20px; border: 1px solid #ddd; border-radius: 6px; background-color: #fafafa; }}
            pre {{ white-space: pre-wrap; word-wrap: break-word; font-family: inherit; font-size: 14px; line-height: 1.6; color: #333; }}
            .spinner {{ border: 4px solid rgba(0, 0, 0, 0.1); border-top: 4px solid #1a73e8; border-radius: 50%; width: 20px; height: 20px; animation: spin 1s linear infinite; display: none; margin: 0 auto; }}
            @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
            .metadata {{ margin-top: 15px; font-size: 14px; color: #555; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Financial News Orchestrator Query</h1>
            <p>Ask a question about a stock and its recent market impact (e.g., "What is the market prediction for TCS following its latest earnings?").</p>
            <textarea id="queryInput" placeholder="Enter your financial query here..."></textarea>
            <button id="submitButton" onclick="submitQuery()">Run Orchestrator</button>
            <div id="loadingSpinner" class="spinner"></div>
            
            <div class="response-box">
                <h2>Analysis Result</h2>
                <div class="metadata" id="metadata"></div>
                <pre id="resultOutput">The analysis result will appear here.</pre>
            </div>
        </div>

        <script>
            async function submitQuery() {{
                const queryInput = document.getElementById('queryInput');
                const submitButton = document.getElementById('submitButton');
                const loadingSpinner = document.getElementById('loadingSpinner');
                const resultOutput = document.getElementById('resultOutput');
                const metadataOutput = document.getElementById('metadata');
                
                const userQuery = queryInput.value.trim();

                if (!userQuery) {{
                    alert('Please enter a query.');
                    return;
                }}

                // Disable button and show spinner
                submitButton.disabled = true;
                submitButton.textContent = 'Running...';
                loadingSpinner.style.display = 'block';
                resultOutput.textContent = 'Fetching and synthesizing data...';
                metadataOutput.textContent = '';


                try {{
                    const response = await fetch('/query', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json'
                        }},
                        body: JSON.stringify({{ query: userQuery }})
                    }});

                    const data = await response.json();

                    if (!response.ok) {{
                        throw new Error(data.detail || 'Unknown server error');
                    }}

                    resultOutput.textContent = data.answer;
                    metadataOutput.innerHTML = `
                        <strong>Time Taken:</strong> ${'{' + 'data.time_taken' + '}'} seconds | 
                        <strong>Symbols Extracted:</strong> ${'{' + 'data.symbols_extracted.join(", ")' + '}'}
                    `;
                }} catch (error) {{
                    console.error('Error:', error);
                    resultOutput.textContent = 'ERROR: Failed to run orchestration. Check console and ensure all agents (8005, 8006, 8007) are running.';
                    metadataOutput.textContent = 'Status: Failed';
                }} finally {{
                    // Re-enable button and hide spinner
                    submitButton.disabled = false;
                    submitButton.textContent = 'Run Orchestrator';
                    loadingSpinner.style.display = 'none';
                }}
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)