##Script function and purpose: Web Search Provider implementation using DuckDuckGo
"""
Web Search Tool

Provides web search capabilities for the ResponseGenerator RAG pipeline.
Implements the WebSearchProtocol defined in jenova.core.response.
"""

import hashlib
import structlog

from jenova.core.response import WebSearchResult, WebSearchProtocol

##Step purpose: Get module logger
logger = structlog.get_logger(__name__)


##Function purpose: Sanitize query for logging to prevent PII leakage
def sanitize_query(query: str) -> str:
    """Sanitize query for logging by hashing it."""
    return hashlib.sha256(query.encode()).hexdigest()[:16]


##Class purpose: DuckDuckGo search provider
class DuckDuckGoSearchProvider:
    """
    Web search provider using DuckDuckGo.
    
    Implements WebSearchProtocol.
    Requires 'duckduckgo-search' package.
    """
    
    def __init__(self, timeout: int = 10) -> None:
        """Initialize the search provider.
        
        Args:
            timeout: Search timeout in seconds.
        """
        self._available = False
        self._ddgs = None
        self._timeout = timeout
        
        try:
            from duckduckgo_search import DDGS
            self._ddgs = DDGS()
            self._available = True
            logger.info("web_search_provider_initialized", provider="duckduckgo", timeout=timeout)
        except ImportError:
            logger.warning(
                "web_search_dependency_missing",
                msg="duckduckgo-search package not found. Web search will be disabled.",
                fix="pip install duckduckgo-search"
            )
        except Exception as e:
            logger.error("web_search_init_failed", error=str(e))

    def is_available(self) -> bool:
        """Check if web search is available."""
        return self._available

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[WebSearchResult]:
        """
        Search the web for relevant content.

        Args:
            query: Search query string
            max_results: Maximum results to return

        Returns:
            List of WebSearchResult objects
        """
        if not self._available or not self._ddgs:
            logger.warning("web_search_unavailable_call", query_hash=sanitize_query(query))
            return []

        results: list[WebSearchResult] = []
        
        try:
            logger.debug("executing_web_search", query_hash=sanitize_query(query))
            # DDGS.text() returns a generator of dicts
            search_gen = self._ddgs.text(query, max_results=max_results, timeout=self._timeout)
            
            for r in search_gen:
                results.append(
                    WebSearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", ""),
                        snippet=r.get("body", ""),
                        source="DuckDuckGo"
                    )
                )
                
            logger.info("web_search_complete", results_count=len(results))
            
        except Exception as e:
            logger.error("web_search_execution_failed", error=str(e), query_hash=sanitize_query(query))
            
        return results

##Class purpose: Mock search provider for testing or fallback
class MockSearchProvider:
    """Mock search provider that returns static results."""
    
    def is_available(self) -> bool:
        return True
        
    def search(self, query: str, max_results: int = 5) -> list[WebSearchResult]:
        if max_results <= 0:
            return []
            
        return [
            WebSearchResult(
                title=f"Mock Result {i}",
                url=f"https://example.com/{i}",
                snippet=f"This is mock result {i} for query: {query}",
                source="Mock"
            )
            for i in range(1, min(max_results, 5) + 1)
        ]
