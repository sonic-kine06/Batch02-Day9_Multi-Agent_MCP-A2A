"""Tax Agent LangGraph definition.

Uses create_react_agent with a tax-specialised system prompt.
No tools — it answers purely from LLM knowledge.
"""

from __future__ import annotations

from langgraph.prebuilt import create_react_agent

from common.llm import get_llm

TAX_SYSTEM_PROMPT = """Bạn là chuyên gia về luật Thuế.
Hãy trả lời ngắn gọn, súc tích dưới 100 từ.
Lưu ý: câu trả lời chỉ mang tính tham khảo, người dùng cần hỏi ý kiến luật sư.
"""


def create_graph():
    """Return a compiled LangGraph create_react_agent for tax questions."""
    llm = get_llm()
    graph = create_react_agent(
        model=llm,
        tools=[],
        prompt=TAX_SYSTEM_PROMPT,
    )
    return graph