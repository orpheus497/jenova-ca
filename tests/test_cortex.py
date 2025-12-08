##Script function and purpose: Unit tests for Cortex graph operations
##Tests for node management, linking, centrality calculation, and reflection

import os
import pytest
from unittest.mock import Mock
from jenova.cortex.cortex import Cortex

##Class purpose: Test suite for Cortex
class TestCortex:
    ##Function purpose: Test Cortex initialization
    def test_init(self, mock_config, mock_ui_logger, mock_file_logger, mock_llm_interface, temp_user_data_dir):
        """Test that Cortex initializes correctly."""
        cortex_root = os.path.join(temp_user_data_dir, 'cortex')
        cortex = Cortex(mock_config, mock_ui_logger, mock_file_logger, mock_llm_interface, cortex_root)
        assert cortex is not None
        assert cortex.graph is not None

    ##Function purpose: Test adding a node to the cognitive graph
    def test_add_node(self, mock_config, mock_ui_logger, mock_file_logger, mock_llm_interface, temp_user_data_dir):
        """Test adding a node to the cognitive graph."""
        cortex_root = os.path.join(temp_user_data_dir, 'cortex')
        cortex = Cortex(mock_config, mock_ui_logger, mock_file_logger, mock_llm_interface, cortex_root)
        node_id = cortex.add_node('insight', 'Test insight', 'test_user')
        assert node_id is not None
        assert node_id in cortex.graph['nodes']

    ##Function purpose: Test adding a link between nodes
    def test_add_link(self, mock_config, mock_ui_logger, mock_file_logger, mock_llm_interface, temp_user_data_dir):
        """Test adding a link between nodes."""
        cortex_root = os.path.join(temp_user_data_dir, 'cortex')
        cortex = Cortex(mock_config, mock_ui_logger, mock_file_logger, mock_llm_interface, cortex_root)
        node1_id = cortex.add_node('insight', 'Test insight 1', 'test_user')
        node2_id = cortex.add_node('insight', 'Test insight 2', 'test_user')
        cortex.add_link(node1_id, node2_id, 'related_to')
        assert len(cortex.graph['links']) > 0

    ##Function purpose: Test centrality calculation
    def test_calculate_centrality(self, mock_config, mock_ui_logger, mock_file_logger, mock_llm_interface, temp_user_data_dir):
        """Test centrality calculation for nodes."""
        cortex_root = os.path.join(temp_user_data_dir, 'cortex')
        cortex = Cortex(mock_config, mock_ui_logger, mock_file_logger, mock_llm_interface, cortex_root)
        node1_id = cortex.add_node('insight', 'Test insight 1', 'test_user')
        node2_id = cortex.add_node('insight', 'Test insight 2', 'test_user')
        cortex.add_link(node1_id, node2_id, 'related_to')
        cortex._calculate_centrality()
        assert cortex.graph['nodes'][node1_id]['metadata']['centrality'] > 0

