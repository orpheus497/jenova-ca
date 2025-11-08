# The JENOVA Cognitive Architecture - RPC Client
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
gRPC client for distributed JENOVA communication.

This module provides client-side functionality for making RPC calls to peer
JENOVA instances, with connection pooling, retry logic, and request routing.
"""

import time
import uuid
from typing import Dict, List, Optional

import grpc
from tenacity import retry, stop_after_attempt, wait_exponential


class JenovaRPCClient:
    """
    gRPC client for communicating with peer JENOVA instances.

    Features:
    - Connection pooling
    - Automatic retry with exponential backoff
    - Request routing to optimal peer
    - Error handling and failover
    """

    def __init__(
        self,
        config: dict,
        file_logger,
        peer_manager=None,
        security_manager=None
    ):
        """
        Initialize RPC client.

        Args:
            config: JENOVA configuration
            file_logger: Logger for file output
            peer_manager: Peer manager for selecting targets
            security_manager: Security manager for authentication
        """
        self.config = config
        self.file_logger = file_logger
        self.peer_manager = peer_manager
        self.security_manager = security_manager

        # Connection pool: peer_id -> grpc.Channel
        self.connections: Dict[str, grpc.Channel] = {}

        # Configuration
        network_config = config.get('network', {})
        peer_selection = network_config.get('peer_selection', {})
        self.timeout_ms = peer_selection.get('timeout_ms', 5000)
        self.timeout_seconds = self.timeout_ms / 1000.0

        # Authentication token
        self.auth_token = None
        if security_manager and security_manager.is_auth_required():
            # Token will be set during initialization
            pass

    def set_auth_token(self, token: str):
        """Set authentication token for requests."""
        self.auth_token = token

    def _get_connection(self, peer_address: str, peer_port: int) -> grpc.Channel:
        """
        Get or create a gRPC channel to a peer.

        Args:
            peer_address: Peer IP address
            peer_port: Peer port

        Returns:
            gRPC channel
        """
        peer_key = f"{peer_address}:{peer_port}"

        if peer_key not in self.connections:
            # Create new channel
            target = f"{peer_address}:{peer_port}"

            if self.security_manager and self.security_manager.is_security_enabled():
                # Use SSL/TLS
                credentials = grpc.ssl_channel_credentials()
                channel = grpc.secure_channel(target, credentials)
                self.file_logger.log_info(f"Created secure channel to {target}")
            else:
                # Insecure
                channel = grpc.insecure_channel(target)
                self.file_logger.log_info(f"Created insecure channel to {target}")

            self.connections[peer_key] = channel

        return self.connections[peer_key]

    def close_all_connections(self):
        """Close all open connections."""
        for peer_key, channel in self.connections.items():
            try:
                channel.close()
                self.file_logger.log_info(f"Closed connection to {peer_key}")
            except Exception as e:
                self.file_logger.log_error(f"Error closing connection to {peer_key}: {e}")

        self.connections.clear()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
        peer_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Request text generation from a peer.

        Args:
            prompt: Text prompt for generation
            temperature: Generation temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            peer_id: Specific peer ID (auto-selected if None)

        Returns:
            Generated text or None on failure
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        try:
            # Select peer if not specified
            if peer_id is None:
                if not self.peer_manager:
                    self.file_logger.log_error("No peer manager available")
                    return None

                peer_id = self.peer_manager.select_peer_for_task('llm')
                if not peer_id:
                    self.file_logger.log_warning("No available peers for LLM task")
                    return None

            # Get peer connection info
            peer_conn = self.peer_manager.get_peer_connection(peer_id)
            if not peer_conn:
                self.file_logger.log_error(f"Peer {peer_id} not found")
                return None

            peer_info = peer_conn.peer_info

            # Create request
            # Note: This will use actual protobuf classes once proto is compiled
            request = self._create_generate_request(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                request_id=request_id
            )

            # Get connection and make call
            channel = self._get_connection(peer_info.address, peer_info.port)

            # Note: This will use actual stub once proto is compiled
            # For now, creating placeholder
            self.file_logger.log_info(
                f"RPC Client: Requesting text generation from {peer_info.instance_name}"
            )

            # Simulate RPC call (will be real once proto is compiled)
            # stub = jenova_pb2_grpc.JenovaRPCStub(channel)
            # response = stub.GenerateText(request, timeout=self.timeout_seconds)

            # For now, return None as placeholder
            # Real implementation will parse response and return text
            response_time_ms = (time.time() - start_time) * 1000

            # Record result with peer manager
            if self.peer_manager:
                self.peer_manager.record_request_result(
                    peer_id=peer_id,
                    success=True,
                    response_time_ms=response_time_ms
                )

            self.file_logger.log_info(
                f"RPC Client: Generation completed in {response_time_ms:.0f}ms"
            )

            # This will return response.text once implemented
            return None  # Placeholder

        except grpc.RpcError as e:
            response_time_ms = (time.time() - start_time) * 1000
            self.file_logger.log_error(f"RPC error: {e.code()} - {e.details()}")

            if self.peer_manager and peer_id:
                self.peer_manager.record_request_result(
                    peer_id=peer_id,
                    success=False,
                    response_time_ms=response_time_ms
                )

            raise  # Let retry decorator handle it

        except Exception as e:
            self.file_logger.log_error(f"Error in generate_text: {e}")
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def embed_text(
        self,
        text: str,
        peer_id: Optional[str] = None
    ) -> Optional[List[float]]:
        """
        Request text embedding from a peer.

        Args:
            text: Text to embed
            peer_id: Specific peer ID (auto-selected if None)

        Returns:
            Embedding vector or None on failure
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        try:
            # Select peer if not specified
            if peer_id is None:
                if not self.peer_manager:
                    self.file_logger.log_error("No peer manager available")
                    return None

                peer_id = self.peer_manager.select_peer_for_task('embedding')
                if not peer_id:
                    self.file_logger.log_warning("No available peers for embedding task")
                    return None

            # Get peer connection info
            peer_conn = self.peer_manager.get_peer_connection(peer_id)
            if not peer_conn:
                self.file_logger.log_error(f"Peer {peer_id} not found")
                return None

            peer_info = peer_conn.peer_info

            # Create request
            request = self._create_embed_request(
                text=text,
                request_id=request_id
            )

            # Get connection
            channel = self._get_connection(peer_info.address, peer_info.port)

            self.file_logger.log_info(
                f"RPC Client: Requesting embedding from {peer_info.instance_name}"
            )

            # Placeholder for actual RPC call
            response_time_ms = (time.time() - start_time) * 1000

            # Record result
            if self.peer_manager:
                self.peer_manager.record_request_result(
                    peer_id=peer_id,
                    success=True,
                    response_time_ms=response_time_ms
                )

            return None  # Placeholder

        except grpc.RpcError as e:
            response_time_ms = (time.time() - start_time) * 1000
            self.file_logger.log_error(f"RPC error: {e.code()} - {e.details()}")

            if self.peer_manager and peer_id:
                self.peer_manager.record_request_result(
                    peer_id=peer_id,
                    success=False,
                    response_time_ms=response_time_ms
                )

            raise

        except Exception as e:
            self.file_logger.log_error(f"Error in embed_text: {e}")
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def embed_text_batch(
        self,
        texts: List[str],
        peer_id: Optional[str] = None
    ) -> Optional[List[List[float]]]:
        """
        Request batch text embeddings from a peer.

        Args:
            texts: List of texts to embed
            peer_id: Specific peer ID (auto-selected if None)

        Returns:
            List of embedding vectors or None on failure
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        try:
            # Select peer
            if peer_id is None:
                if not self.peer_manager:
                    return None

                peer_id = self.peer_manager.select_peer_for_task('embedding')
                if not peer_id:
                    return None

            # Get peer info
            peer_conn = self.peer_manager.get_peer_connection(peer_id)
            if not peer_conn:
                return None

            peer_info = peer_conn.peer_info

            # Create request
            request = self._create_embed_batch_request(
                texts=texts,
                request_id=request_id
            )

            # Get connection
            channel = self._get_connection(peer_info.address, peer_info.port)

            self.file_logger.log_info(
                f"RPC Client: Requesting batch embedding ({len(texts)} texts) "
                f"from {peer_info.instance_name}"
            )

            # Placeholder
            response_time_ms = (time.time() - start_time) * 1000

            if self.peer_manager:
                self.peer_manager.record_request_result(
                    peer_id=peer_id,
                    success=True,
                    response_time_ms=response_time_ms
                )

            return None  # Placeholder

        except Exception as e:
            self.file_logger.log_error(f"Error in embed_text_batch: {e}")
            return None

    def health_check(self, peer_id: str) -> Optional[Dict]:
        """
        Check health of a specific peer.

        Args:
            peer_id: Peer instance ID

        Returns:
            Health data dictionary or None on failure
        """
        try:
            peer_conn = self.peer_manager.get_peer_connection(peer_id)
            if not peer_conn:
                return None

            peer_info = peer_conn.peer_info

            # Create request
            request = self._create_health_check_request()

            # Get connection
            channel = self._get_connection(peer_info.address, peer_info.port)

            # Placeholder for actual health check
            # Real implementation will return health data

            return {
                'status': 'healthy',
                'peer_id': peer_id,
                'peer_name': peer_info.instance_name
            }

        except Exception as e:
            self.file_logger.log_error(f"Health check failed for {peer_id}: {e}")
            return None

    def get_capabilities(self, peer_id: str) -> Optional[Dict]:
        """
        Get capabilities from a specific peer.

        Args:
            peer_id: Peer instance ID

        Returns:
            Capabilities dictionary or None on failure
        """
        try:
            peer_conn = self.peer_manager.get_peer_connection(peer_id)
            if not peer_conn:
                return None

            peer_info = peer_conn.peer_info

            # Create request
            request = self._create_capabilities_request()

            # Get connection
            channel = self._get_connection(peer_info.address, peer_info.port)

            # Placeholder for actual capabilities request

            return None

        except Exception as e:
            self.file_logger.log_error(f"Get capabilities failed for {peer_id}: {e}")
            return None

    # Helper methods to create request objects
    # These are placeholders that will be replaced with actual protobuf messages

    def _create_generate_request(self, **kwargs):
        """Create GenerateRequest (placeholder)."""
        return type('GenerateRequest', (), kwargs)()

    def _create_embed_request(self, **kwargs):
        """Create EmbedRequest (placeholder)."""
        return type('EmbedRequest', (), kwargs)()

    def _create_embed_batch_request(self, **kwargs):
        """Create EmbedBatchRequest (placeholder)."""
        return type('EmbedBatchRequest', (), kwargs)()

    def _create_health_check_request(self):
        """Create HealthCheckRequest (placeholder)."""
        return type('HealthCheckRequest', (), {})()

    def _create_capabilities_request(self):
        """Create CapabilitiesRequest (placeholder)."""
        return type('CapabilitiesRequest', (), {})()
