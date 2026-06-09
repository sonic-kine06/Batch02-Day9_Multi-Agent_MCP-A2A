import sys
import os
import json
import asyncio
from typing import Annotated, TypedDict
from dotenv import load_dotenv

# Add parent to path to use Day9's common.llm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.llm import get_llm

# Add Day08 folder to path
day08_path = os.path.join(os.path.dirname(__file__), "Day08_RAG_pipeline_cohort2")
sys.path.insert(0, day08_path)

from src.task10_generation import generate_with_citation
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

@tool
def research_drug_law(query: str) -> str:
    """Research drug law using the internal RAG system. This tool searches Vietnamese drug laws and news."""
    try:
        # Override RAG top_k to be smaller for the tool to avoid huge context length
        res = generate_with_citation(query, top_k=2)
        return res["answer"]
    except Exception as e:
        return f"Error during research: {str(e)}"

# Define state
class SupervisorState(TypedDict):
    question: str
    research_result: str
    draft_result: str
    next_action: str
    final_answer: str

# Node implementations
async def supervisor_node(state: SupervisorState) -> dict:
    print(">>> [Supervisor] Deciding next step...")
    llm = get_llm()
    prompt = f"""You are a routing supervisor.
Question: {state['question']}
Research Result: {state.get('research_result', '')}
Draft Result: {state.get('draft_result', '')}

Rules:
1. If there is no Research Result, output exactly this JSON: {{"next_action": "research"}}
2. If there is a Research Result but no Draft Result, output exactly this JSON: {{"next_action": "draft"}}
3. If there is both a Research Result and a Draft Result, output exactly this JSON: {{"next_action": "done"}}

Reply ONLY with JSON, nothing else.
"""
    result = await llm.ainvoke([HumanMessage(content=prompt)])
    try:
        raw = result.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        parsed = json.loads(raw)
        return {"next_action": parsed.get("next_action", "done")}
    except Exception as e:
        print(f"Error parsing supervisor JSON: {e}. Defaulting to done.")
        return {"next_action": "done"}

async def researcher_node(state: SupervisorState) -> dict:
    print(">>> [Researcher] Conducting research using Day 08 RAG...")
    llm = get_llm()
    prompt = "You are a Legal Researcher specializing in Vietnamese Drug Law. Use the research_drug_law tool to find answers to the user's question."
    agent = create_react_agent(model=llm, tools=[research_drug_law], prompt=prompt)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": state["question"]}]})
    return {"research_result": result["messages"][-1].content}

async def drafter_node(state: SupervisorState) -> dict:
    print(">>> [Drafter] Drafting formal response...")
    llm = get_llm()
    prompt = f"""You are a Legal Drafter. Based on the researcher's findings, draft a formal summary.
Question: {state['question']}
Research Findings: {state['research_result']}

Your draft should be well-structured, professional, and clear.
"""
    result = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"draft_result": result.content, "final_answer": result.content}

def router(state: SupervisorState) -> str:
    action = state.get("next_action", "done")
    if action == "research":
        return "researcher"
    elif action == "draft":
        return "drafter"
    else:
        return "END"

def create_graph():
    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("drafter", drafter_node)
    
    graph.set_entry_point("supervisor")
    graph.add_conditional_edges("supervisor", router, {
        "researcher": "researcher",
        "drafter": "drafter",
        "END": END
    })
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("drafter", "supervisor")
    
    return graph.compile()

async def main():
    print("==========================================================")
    print(" Day 08 Assignment: Supervisor-Workers RAG Architecture")
    print("==========================================================")
    graph = create_graph()
    question = "Hình phạt cho tội tàng trữ trái phép chất ma túy ở Việt Nam là gì?"
    print(f"Question: {question}\n")
    
    result = await graph.ainvoke({
        "question": question,
        "research_result": "",
        "draft_result": "",
        "next_action": "",
        "final_answer": ""
    })
    
    print("\n" + "="*50)
    print("FINAL ANSWER")
    print("="*50)
    print(result["final_answer"])

if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
