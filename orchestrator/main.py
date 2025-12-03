import os
import json
import logging
from typing import List, Dict, Any, Optional, TypedDict
import httpx
from dotenv import load_dotenv

# LangGraph and Groq dependencies
from langgraph.graph import StateGraph, END
from groq import Groq
# Note: Removed the import for SystemMessage, HumanMessage as they are no longer used.

load_dotenv()

# -----------------------------
# Config & Setup
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - Orchestrator - %(levelname)s - %(message)s")
logger = logging.getLogger("Orchestrator")

# Ensure you have your Groq API key set in your environment or .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QUERY_AGENT_URL = os.getenv("QUERY_AGENT_URL", "http://localhost:8007")
HYBRID_QUERY_ENDPOINT = f"{QUERY_AGENT_URL}/query/impact"

# Initialize clients
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not found. Please set it in your environment.")
groq_client = Groq(api_key=GROQ_API_KEY)
# Use synchronous client for simplicity in a simple graph implementation
http_client = httpx.Client(timeout=10.0) 

# -----------------------------
# 1. Graph State Definition
# -----------------------------

class AgentState(TypedDict):
    """The state of the LangGraph workflow."""
    user_query: str                                # The original user query.
    target_symbols: List[str]                      # Extracted stock tickers (e.g., ["TCS", "INFY"]).
    query_text_for_search: str                     # Refined search term for the Query Agent.
    query_agent_response: List[Dict[str, Any]]     # Results returned from the Query Agent (top stories).
    final_answer: Optional[str]                    # The conversational answer synthesized by the final LLM call.


# -----------------------------
# 2. Graph Nodes (Functions)
# -----------------------------

def extract_targets(state: AgentState) -> AgentState:
    """
    Node 1: Uses LLM to extract target symbols and a clean search query 
    from the user's natural language request.
    """
    logger.info("Executing Node: extract_targets (LLM Call)")
    user_query = state["user_query"]
    
    # Prompt the LLM to output a clean JSON object
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
            model="llama-3.3-70b-versatile", 
            # CORRECTED: Using standard dictionary with explicit 'role'
            messages=[{"role": "user", "content": prompt}], 
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        output = json.loads(response.choices[0].message.content)
        
        state["target_symbols"] = output.get("symbols", [])
        state["query_text_for_search"] = output.get("search_query", user_query)
        logger.info(f"Extracted Symbols: {state['target_symbols']}, Search Query: {state['query_text_for_search']}")
        
    except Exception as e:
        logger.error(f"Error during LLM extraction: {e}")
        # Fallback to defaults
        state["target_symbols"] = []
        state["query_text_for_search"] = user_query
    
    return state


def run_hybrid_query(state: AgentState) -> AgentState:
    """
    Node 2: Calls the Query Agent (http://localhost:8007) with the refined search parameters.
    """
    logger.info("Executing Node: run_hybrid_query (API Call to Query Agent)")
    
    request_body = {
    "query_text": state["query_text_for_search"],
    "target_symbols": state["target_symbols"],
    "semantic_weight": 0.8,  # Higher semantic weight
    "impact_weight": 0.2,    # Lower impact weight
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
        state["final_answer"] = f"I could not find any relevant or impactful stories for your query: '{state['user_query']}'."
        return state
    
    # Format the retrieved data into a single context block
    context = []
    for story in state["query_agent_response"]:
        impact_details = json.dumps(story['impact_summary']) if story['impact_summary'] else "No explicit impact analysis found for requested symbols."
        
        context.append(f"""
        ---
        STORY TITLE: {story['canonical_title']}
        SUMMARY: {story['summary']}
        FINAL SCORE: {story['final_hybrid_score']:.3f}
        IMPACT: {impact_details}
        ---
        """)

    context_str = "\n".join(context)

    # Synthesis Prompt
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
            model="llama-3.3-70b-versatile",
            # CORRECTED: Using standard dictionaries with explicit 'role'
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

    # 1. Add the nodes
    workflow.add_node("extract_targets", extract_targets)
    workflow.add_node("run_hybrid_query", run_hybrid_query)
    workflow.add_node("generate_final_answer", generate_final_answer)

    # 2. Set the entry point
    workflow.set_entry_point("extract_targets")

    # 3. Define the edges (flow)
    workflow.add_edge("extract_targets", "run_hybrid_query")
    workflow.add_edge("run_hybrid_query", "generate_final_answer")

    # 4. Set the end point
    workflow.add_edge("generate_final_answer", END)

    # 5. Compile and return the runnable graph
    app = workflow.compile()
    logger.info("LangGraph Orchestrator compiled successfully.")
    return app

# Create the runnable graph instance
orchestrator = create_orchestrator()

# -----------------------------
# 4. Example Execution
# -----------------------------

if __name__ == "__main__":
    # Example usage:
    user_input = "What are the most recent stories regarding the IT industry in India?"
    
    initial_state = AgentState(
        user_query=user_input, 
        target_symbols=[], 
        query_text_for_search="", 
        query_agent_response=[], 
        final_answer=None
    )
    
    logger.info(f"Starting orchestration for query: {user_input}")
    
    # Run the graph
    final_state = orchestrator.invoke(initial_state)
    
    print("\n" + "="*50)
    print("✨ FINAL ORCHESTRATOR RESULT ✨")
    print("="*50)
    print(f"User Query: {final_state['user_query']}")
    print(f"Extracted Symbols: {final_state['target_symbols']}")
    print(f"Final Answer:\n\n{final_state['final_answer']}")
    print("="*50 + "\n")