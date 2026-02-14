"""
Tests for D2 Feature Sprinter Deliverables
- WIRING-005: Web Search Provider
- WIRING-007: Finetune Data Export
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jenova.core.response import WebSearchResult
from jenova.insights.manager import InsightManager
from jenova.tools.web_search import DuckDuckGoSearchProvider


class TestWebSearchProvider:
    """Test suite for Web Search Provider (WIRING-005)."""

    def test_ddg_provider_initialization_success(self) -> None:
        """Test successful initialization when package is available."""
        with patch.dict("sys.modules", {"duckduckgo_search": MagicMock()}):
            provider = DuckDuckGoSearchProvider()
            assert provider.is_available() is True

    def test_ddg_provider_initialization_failure(self) -> None:
        """Test initialization failure when package is missing."""
        # Remove the module from sys.modules to force ImportError
        with patch.dict("sys.modules", {"duckduckgo_search": None}):
            provider = DuckDuckGoSearchProvider()
            assert provider.is_available() is False

    def test_ddg_search_execution(self) -> None:
        """Test search execution returns correct results."""
        # Create a mock module with DDGS class
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = [
            {"title": "Test Result", "href": "https://example.com", "body": "Snippet"}
        ]

        mock_ddgs_cls = MagicMock(return_value=mock_ddgs_instance)
        mock_module = MagicMock()
        mock_module.DDGS = mock_ddgs_cls

        # Patch the module into existence
        with patch.dict("sys.modules", {"duckduckgo_search": mock_module}):
            provider = DuckDuckGoSearchProvider()
            results = provider.search("test query")

            assert len(results) == 1
            assert isinstance(results[0], WebSearchResult)
            assert results[0].title == "Test Result"
            assert results[0].url == "https://example.com"
            assert results[0].source == "DuckDuckGo"

            # Verify the mock was called correctly
            mock_ddgs_cls.assert_called_once_with(timeout=10)
            mock_ddgs_instance.text.assert_called_once_with("test query", max_results=5)


class TestInsightTrainingExport:
    """Test suite for Insight Training Data Export (WIRING-007)."""

    @pytest.fixture
    def mock_deps(self) -> tuple[MagicMock, MagicMock]:
        return MagicMock(), MagicMock()

    def test_training_data_append(
        self, tmp_path: Path, mock_deps: tuple[MagicMock, MagicMock]
    ) -> None:
        """Test that saving an insight appends to training data."""
        insights_root = tmp_path / "insights"
        training_file = tmp_path / "training_data.jsonl"

        graph, llm = mock_deps
        # Mock concern manager to avoid LLM calls
        with patch("jenova.insights.manager.ConcernManager") as MockCM:
            mock_cm_instance = MockCM.return_value
            mock_cm_instance.find_or_create_concern.return_value = "test_topic"

            manager = InsightManager(
                insights_root=insights_root, graph=graph, llm=llm, training_data_path=training_file
            )

            # Save an insight
            manager.save_insight(
                content="Test content for training", username="testuser", topic="test_topic"
            )

            # Verify file created
            assert training_file.exists()

            # Verify content
            with open(training_file, encoding="utf-8") as f:
                line = f.readline()
                data = json.loads(line)

                assert data["positive"] == "Test content for training"
                assert "test_topic" in data["anchor"]
                assert data["metadata"]["source"] == "insight_manager"
