# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.
#
# Phase 22: Code Quality & Testing - Web Tools

"""
Web search and content retrieval tool module.

This module provides web search functionality using DuckDuckGo as the search
engine. Requires optional selenium dependencies.
"""

from typing import Any, Dict, List
from urllib.parse import quote_plus

from jenova.tools.base import BaseTool, ToolResult

# Optional selenium imports
SELENIUM_AVAILABLE = False
try:
    from selenium import webdriver
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.common.by import By
    from selenium.common.exceptions import WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    pass


class WebTools(BaseTool):
    """
    Web search and content retrieval tool.

    Provides web search functionality using DuckDuckGo search engine.
    Requires optional selenium dependencies (install with: pip install jenova-ai[web])

    Methods:
        execute: Perform web search
        search: Alias for execute
        is_available: Check if selenium dependencies are installed
    """

    def __init__(self, config: Dict[str, Any], ui_logger: Any, file_logger: Any):
        """
        Initialize web tools.

        Args:
            config: Configuration dictionary
            ui_logger: UI logger instance
            file_logger: File logger instance
        """
        super().__init__(
            name="web_tools",
            description="Web search using DuckDuckGo",
            config=config,
            ui_logger=ui_logger,
            file_logger=file_logger
        )

        self.selenium_available = SELENIUM_AVAILABLE
        self.max_results = config.get('tools', {}).get('web_search_max_results', 5)

    def execute(
        self,
        query: str,
        max_results: int = None
    ) -> ToolResult:
        """
        Perform web search using DuckDuckGo.

        Args:
            query: Search query string
            max_results: Maximum number of results to return (default: 5)

        Returns:
            ToolResult with search results or error

        Example:
            >>> tools = WebTools(config, ui_logger, file_logger)
            >>> result = tools.execute("Python programming tutorials")
            >>> if result.success:
            >>>     for item in result.data:
            >>>         print(f"{item['title']}: {item['link']}")

        Note:
            Requires selenium package. Install with: pip install jenova-ai[web]
        """
        if not self.selenium_available:
            error_msg = (
                "Web search functionality requires selenium and related packages.\n"
                "Install with: pip install jenova-ai[web]\n"
                "Or: pip install selenium webdriver-manager beautifulsoup4"
            )
            return self._create_error_result(
                error=error_msg,
                metadata={'reason': 'dependencies_missing', 'query': query}
            )

        max_results = max_results or self.max_results

        # Sanitize query for URL (SECURITY FIX: Phase 21 - LOW-3)
        safe_query = quote_plus(query)

        import os
        # Preserve CUDA environment for subprocess (geckodriver/Firefox)
        original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")

        options = Options()
        options.add_argument('--headless')
        driver = None

        try:
            # Temporarily hide CUDA from browser subprocess to prevent conflicts
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

            driver = webdriver.Firefox(options=options)
            driver.get(f"https://duckduckgo.com/html/?q={safe_query}")

            results = []
            result_elements = driver.find_elements(By.CLASS_NAME, "result")[:max_results]

            for result_elem in result_elements:
                try:
                    title_elem = result_elem.find_element(By.CLASS_NAME, "result__title")
                    link_elem = result_elem.find_element(By.CLASS_NAME, "result__url")
                    snippet_elem = result_elem.find_element(By.CLASS_NAME, "result__snippet")

                    results.append({
                        'title': title_elem.text,
                        'link': link_elem.get_attribute("href"),
                        'summary': snippet_elem.text
                    })
                except Exception as e:
                    # Skip individual result errors
                    if self.file_logger:
                        self.file_logger.log_warning(
                            f"Failed to parse search result element: {e}"
                        )
                    continue

            # Restore CUDA environment
            if original_cuda_visible:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)

            result = self._create_success_result(
                data=results,
                metadata={
                    'query': query,
                    'results_count': len(results),
                    'max_results': max_results
                }
            )

            self._log_execution({'query': query}, result)
            return result

        except WebDriverException as e:
            error_msg = f"WebDriver error: {str(e)}"
            result = self._create_error_result(
                error=error_msg,
                metadata={'query': query, 'reason': 'webdriver_error'}
            )
            self._log_execution({'query': query}, result)
            return result

        except Exception as e:
            error_msg = f"Web search failed: {str(e)}"
            result = self._create_error_result(
                error=error_msg,
                metadata={'query': query, 'exception': type(e).__name__}
            )
            self._log_execution({'query': query}, result)
            return result

        finally:
            # Ensure browser is closed
            if driver:
                try:
                    driver.quit()
                except (RuntimeError, OSError, ConnectionError) as e:
                    # Suppress cleanup errors but log for debugging
                    # Common during browser crashes or network issues
                    logger.debug(f"Browser cleanup failed (non-critical): {type(e).__name__}: {e}")

            # Restore CUDA environment
            if original_cuda_visible:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
            elif "CUDA_VISIBLE_DEVICES" in os.environ:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    def search(self, query: str, max_results: int = None) -> ToolResult:
        """
        Alias for execute() method.

        Args:
            query: Search query string
            max_results: Maximum number of results

        Returns:
            ToolResult with search results
        """
        return self.execute(query=query, max_results=max_results)

    def is_available(self) -> bool:
        """
        Check if web search is available (selenium installed).

        Returns:
            True if selenium dependencies are installed

        Example:
            >>> tools = WebTools(config, ui_logger, file_logger)
            >>> if tools.is_available():
            >>>     result = tools.search("test query")
            >>> else:
            >>>     print("Please install: pip install jenova-ai[web]")
        """
        return self.selenium_available
