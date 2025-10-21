"""Web search agent stub."""

from typing import Optional

from langchain.tools import DuckDuckGoSearchRun


class WebSearchAgent:
    """Wrapper around LangChain's DuckDuckGo tool."""

    def __init__(self, safe_search: str = "moderate"):
        self._search = DuckDuckGoSearchRun(safesearch=safe_search)

    def run(self, query: str, region: Optional[str] = None) -> str:
        """Execute the search query and return the top result text."""
        return self._search.run(query if region is None else f"{query} region:{region}")

