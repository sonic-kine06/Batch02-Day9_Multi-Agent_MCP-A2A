"""Shared LLM factory for all agents.

Uses OpenRouter as an OpenAI-compatible API, so any provider's model
can be selected via the OPENROUTER_MODEL env var.
"""

import os
from typing import Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration

class MockLLM(BaseChatModel):
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="[MOCKED_API_RESPONSE] Đây là câu trả lời giả lập vì không có API Key."))])
    
    @property
    def _llm_type(self) -> str:
        return "mock_llm"
    
    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

def get_llm():
    """Return a configured LLM instance for the multi-agent system."""
    if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n[WARNING] Không tìm thấy OPENROUTER_API_KEY. Đang sử dụng MockLLM để giả lập.\n")
        return MockLLM()
    
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENROUTER_API_KEY", "dummy"),
        temperature=0.3,
    )