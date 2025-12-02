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

# Import generated protobuf modules
try:
    from jenova.network.proto import jenova_pb2
    from jenova.network.proto import jenova_pb2_grpc
except ImportError:
    raise ImportError(
        "Protocol Buffer modules not found. Run 'python build_proto.py' to compile protos."
    )


class RPCRequestError(Exception):
    """Custom exception for RPC request failures that aren't gRPC errors."""
    
    def __init__(self, message: str, error_code: str = "FAILED_REQUEST"):
        self.message = message
        self.error_code = error_code
        super().__init__(message)


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
        self, config: dict, file_logger, peer_manager=None, security_manager=None
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
        network_config = config.get("network", {})
        peer_selection = network_config.get("peer_selection", {})
        self.timeout_ms = peer_selection.get("timeout_ms", 5000)
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
                self.file_logger.log_error(
                    f"Error closing connection to {peer_key}: {e}"
                )

        self.connections.clear()

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
        peer_id: Optional[str] = None,
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

                peer_id = self.peer_manager.select_peer_for_task("llm")
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
                request_id=request_id,
            )

            # Get connection and make call
            channel = self._get_connection(peer_info.address, peer_info.port)

            self.file_logger.log_info(
                f"RPC Client: Requesting text generation from {peer_info.instance_name}"
            )

            # Create stub and make RPC call
            stub = jenova_pb2_grpc.JenovaRPCStub(channel)

            # Add authentication metadata if available
            metadata = []
            if self.auth_token:
                metadata.append(("authorization", f"Bearer {self.auth_token}"))

            # Make the RPC call
            response = stub.GenerateText(
                request,
                timeout=self.timeout_seconds,
                metadata=metadata if metadata else None,
            )

            response_time_ms = (time.time() - start_time) * 1000

            # Check if request succeeded
            if not response.success:
                raise RPCRequestError(f"Generation failed: {response.error_message}")

            # Record result with peer manager
            if self.peer_manager:
                self.peer_manager.record_request_result(
                    peer_id=peer_id, success=True, response_time_ms=response_time_ms
                )

            self.file_logger.log_info(
                f"RPC Client: Generated {response.tokens_generated} tokens in {response_time_ms:.0f}ms"
            )

            return response.text

        except grpc.RpcError as e:
            response_time_ms = (time.time() - start_time) * 1000
            # Safely get code and details - grpc.RpcError may not have these methods
            # when it comes from actual gRPC calls, they are subclasses with these methods
            error_code = getattr(e, 'code', lambda: 'UNKNOWN')()
            error_details = getattr(e, 'details', lambda: str(e))()
            self.file_logger.log_error(f"RPC error: {error_code} - {error_details}")

            if self.peer_manager and peer_id:
                self.peer_manager.record_request_result(
                    peer_id=peer_id, success=False, response_time_ms=response_time_ms
                )

            raise  # Let retry decorator handle it

        except RPCRequestError as e:
            response_time_ms = (time.time() - start_time) * 1000
            self.file_logger.log_error(f"RPC request error: {e.error_code} - {e.message}")

            if self.peer_manager and peer_id:
                self.peer_manager.record_request_result(
                    peer_id=peer_id, success=False, response_time_ms=response_time_ms
                )

            raise  # Let retry decorator handle it

        except Exception as e:
            self.file_logger.log_error(f"Error in generate_text: {e}")
            return None

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def embed_text(
        self, text: str, peer_id: Optional[str] = None
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

                peer_id = self.peer_manager.select_peer_for_task("embedding")
                if not peer_id:
                    self.file_logger.log_warning(
                        "No available peers for embedding task"
                    )
                    return None

            # Get peer connection info
            peer_conn = self.peer_manager.get_peer_connection(peer_id)
            if not peer_conn:
                self.file_logger.log_error(f"Peer {peer_id} not found")
                return None

            peer_info = peer_conn.peer_info

            # Create request
            request = self._create_embed_request(text=text, request_id=request_id)

            # Get connection
            channel = self._get_connection(peer_info.address, peer_info.port)

            self.file_logger.log_info(
                f"RPC Client: Requesting embedding from {peer_info.instance_name}"
            )

            # Create stub and make RPC call
            stub = jenova_pb2_grpc.JenovaRPCStub(channel)

            # Add authentication metadata if available
            metadata = []
            if self.auth_token:
                metadata.append(("authorization", f"Bearer {self.auth_token}"))

            # Make the RPC call
            response = stub.EmbedText(
                request,
                timeout=self.timeout_seconds,
                metadata=metadata if metadata else None,
            )

            response_time_ms = (time.time() - start_time) * 1000

            # Check if request succeeded
            if not response.success:
                raise RPCRequestError(f"Embedding failed: {response.error_message}")

            # Record result
            if self.peer_manager:
                self.peer_manager.record_request_result(
                    peer_id=peer_id, success=True, response_time_ms=response_time_ms
                )

            return list(response.embedding)

        except grpc.RpcError as e:
            response_time_ms = (time.time() - start_time) * 1000
            error_code = getattr(e, 'code', lambda: 'UNKNOWN')()
            error_details = getattr(e, 'details', lambda: str(e))()
            self.file_logger.log_error(f"RPC error: {error_code} - {error_details}")

            if self.peer_manager and peer_id:
                self.peer_manager.record_request_result(
                    peer_id=peer_id, success=False, response_time_ms=response_time_ms
                )

            raise

        except RPCRequestError as e:
            response_time_ms = (time.time() - start_time) * 1000
            self.file_logger.log_error(f"RPC request error: {e.error_code} - {e.message}")

            if self.peer_manager and peer_id:
                self.peer_manager.record_request_result(
                    peer_id=peer_id, success=False, response_time_ms=response_time_ms
                )

            raise

        except Exception as e:
            self.file_logger.log_error(f"Error in embed_text: {e}")
            return None

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def embed_text_batch(
        self, texts: List[str], peer_id: Optional[str] = None
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

                peer_id = self.peer_manager.select_peer_for_task("embedding")
                if not peer_id:
                    return None

            # Get peer info
            peer_conn = self.peer_manager.get_peer_connection(peer_id)
            if not peer_conn:
                return None

            peer_info = peer_conn.peer_info

            # Create request
            request = self._create_embed_batch_request(
                texts=texts, request_id=request_id
            )

            # Get connection
            channel = self._get_connection(peer_info.address, peer_info.port)

            self.file_logger.log_info(
                f"RPC Client: Requesting batch embedding ({len(texts)} texts) "
                f"from {peer_info.instance_name}"
            )

            # Create stub and make RPC call
            stub = jenova_pb2_grpc.JenovaRPCStub(channel)

            # Add authentication metadata if available
            metadata = []
            if self.auth_token:
                metadata.append(("authorization", f"Bearer {self.auth_token}"))

            # Make the RPC call
            response = stub.EmbedTextBatch(
                request,
                timeout=self.timeout_seconds,
                metadata=metadata if metadata else None,
            )

            response_time_ms = (time.time() - start_time) * 1000

            # Check if request succeeded
            if not response.success:
                raise RPCRequestError(f"Batch embedding failed: {response.error_message}")

            if self.peer_manager:
                self.peer_manager.record_request_result(
                    peer_id=peer_id, success=True, response_time_ms=response_time_ms
                )

            # Extract embeddings from response
            return [list(emb_result.embedding) for emb_result in response.embeddings]

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

            # Create stub and make RPC call
            stub = jenova_pb2_grpc.JenovaRPCStub(channel)

            # Add authentication metadata if available
            metadata = []
            if self.auth_token:
                metadata.append(("authorization", f"Bearer {self.auth_token}"))

            # Make the RPC call
            response = stub.HealthCheck(
                request,
                timeout=self.timeout_seconds,
                metadata=metadata if metadata else None,
            )

            return {
                "status": response.status,
                "cpu_percent": response.cpu_percent,
                "memory_percent": response.memory_percent,
                "gpu_memory_percent": response.gpu_memory_percent,
                "active_requests": response.active_requests,
                "total_requests_served": response.total_requests_served,
                "uptime_seconds": response.uptime_seconds,
                "peer_id": peer_id,
                "peer_name": peer_info.instance_name,
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

            # Create stub and make RPC call
            stub = jenova_pb2_grpc.JenovaRPCStub(channel)

            # Add authentication metadata if available
            metadata = []
            if self.auth_token:
                metadata.append(("authorization", f"Bearer {self.auth_token}"))

            # Make the RPC call
            response = stub.GetCapabilities(
                request,
                timeout=self.timeout_seconds,
                metadata=metadata if metadata else None,
            )

            return {
                "share_llm": response.share_llm,
                "share_embeddings": response.share_embeddings,
                "share_memory": response.share_memory,
                "max_concurrent_requests": response.max_concurrent_requests,
                "supports_streaming": response.supports_streaming,
                "version": response.version,
            }

        except Exception as e:
            self.file_logger.log_error(f"Get capabilities failed for {peer_id}: {e}")
            return None

    # Helper methods to create request objects using real protobuf messages

    def _create_generate_request(self, **kwargs):
        """Create GenerateRequest using protobuf."""
        return jenova_pb2.GenerateRequest(
            prompt=kwargs.get("prompt", ""),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.95),
            max_tokens=kwargs.get("max_tokens", 512),
            request_id=kwargs.get("request_id", ""),
        )

    def _create_embed_request(self, **kwargs):
        """Create EmbedRequest using protobuf."""
        return jenova_pb2.EmbedRequest(
            text=kwargs.get("text", ""), request_id=kwargs.get("request_id", "")
        )

    def _create_embed_batch_request(self, **kwargs):
        """Create EmbedBatchRequest using protobuf."""
        return jenova_pb2.EmbedBatchRequest(
            texts=kwargs.get("texts", []), request_id=kwargs.get("request_id", "")
        )

    def _create_health_check_request(self):
        """Create HealthCheckRequest using protobuf."""
        return jenova_pb2.HealthCheckRequest()

    def _create_capabilities_request(self):
        """Create CapabilitiesRequest using protobuf."""
        return jenova_pb2.CapabilitiesRequest()
