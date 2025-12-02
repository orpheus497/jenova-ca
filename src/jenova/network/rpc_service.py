# The JENOVA Cognitive Architecture - RPC Service
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
gRPC service implementation for distributed JENOVA.

This module implements the server-side gRPC service that allows peers to
request LLM inference, embeddings, and memory search operations.
"""

import platform
import time
from concurrent import futures
from typing import Optional

import grpc
import psutil

# Import generated protobuf modules
try:
    from jenova.network.proto import jenova_pb2
    from jenova.network.proto import jenova_pb2_grpc
except ImportError:
    raise ImportError(
        "Protocol Buffer modules not found. Run 'python build_proto.py' to compile protos."
    )


class JenovaRPCServicer(jenova_pb2_grpc.JenovaRPCServicer):
    """
    gRPC service implementation for JENOVA distributed operations.

    This servicer exposes local resources (LLM, embeddings, memory) to peers
    via gRPC, enabling distributed computing across the LAN.
    """

    def __init__(
        self,
        config: dict,
        file_logger,
        llm_interface=None,
        embedding_manager=None,
        memory_search=None,
        health_monitor=None,
    ):
        """
        Initialize RPC servicer.

        Args:
            config: JENOVA configuration
            file_logger: Logger for file output
            llm_interface: LLM interface for text generation
            embedding_manager: Embedding model manager
            memory_search: Memory search instance
            health_monitor: Health monitor for status reporting
        """
        self.config = config
        self.file_logger = file_logger
        self.llm_interface = llm_interface
        self.embedding_manager = embedding_manager
        self.memory_search = memory_search
        self.health_monitor = health_monitor

        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.start_time = time.time()

        # Resource sharing settings
        network_config = config.get("network", {})
        resource_sharing = network_config.get("resource_sharing", {})
        self.share_llm = resource_sharing.get("share_llm", True)
        self.share_embeddings = resource_sharing.get("share_embeddings", True)
        self.share_memory = resource_sharing.get("share_memory", False)
        self.max_concurrent = resource_sharing.get("max_concurrent_requests", 5)

        self.file_logger.log_info(
            f"RPC servicer initialized (share_llm={self.share_llm}, "
            f"share_embeddings={self.share_embeddings}, "
            f"share_memory={self.share_memory})"
        )

    def GenerateText(self, request, context):
        """
        Handle text generation request from peer.

        Args:
            request: GenerateRequest protobuf message
            context: gRPC context

        Returns:
            GenerateResponse protobuf message
        """
        self.total_requests += 1
        start_time = time.time()

        try:
            # Check if LLM sharing is enabled
            if not self.share_llm:
                self.failed_requests += 1
                return self._create_generate_response(
                    request_id=request.request_id,
                    success=False,
                    error_message="LLM sharing is disabled on this instance",
                )

            # Check if LLM interface is available
            if not self.llm_interface:
                self.failed_requests += 1
                return self._create_generate_response(
                    request_id=request.request_id,
                    success=False,
                    error_message="LLM interface not available",
                )

            # Generate text
            self.file_logger.log_info(
                f"RPC: Generating text for peer request {request.request_id}"
            )

            result = self.llm_interface.generate(
                prompt=request.prompt,
                temperature=request.temperature or 0.7,
                top_p=request.top_p or 0.95,
                max_tokens=request.max_tokens or 512,
            )

            generation_time = time.time() - start_time
            self.successful_requests += 1

            self.file_logger.log_info(
                f"RPC: Generated {len(result)} chars in {generation_time:.2f}s"
            )

            return self._create_generate_response(
                text=result,
                tokens_generated=len(result.split()),  # Rough estimate
                generation_time_seconds=generation_time,
                model_name=self.config.get("model", {}).get("model_path", "unknown"),
                request_id=request.request_id,
                success=True,
            )

        except Exception as e:
            self.failed_requests += 1
            self.file_logger.log_error(f"RPC: Text generation failed: {e}")
            return self._create_generate_response(
                request_id=request.request_id, success=False, error_message=str(e)
            )

    def GenerateTextStream(self, request, context):
        """
        Handle streaming text generation request.

        Args:
            request: GenerateRequest protobuf message
            context: gRPC context

        Yields:
            GenerateStreamResponse protobuf messages
        """
        # Note: Streaming implementation would require LLM interface to support streaming
        # For now, fall back to non-streaming
        self.file_logger.log_warning("RPC: Streaming not yet implemented, using batch")
        response = self.GenerateText(request, context)

        # Convert to stream format
        if response.success:
            # Yield tokens one by one
            tokens = response.text.split()
            for i, token in enumerate(tokens):
                yield self._create_stream_response(
                    token=token + " ",
                    is_final=(i == len(tokens) - 1),
                    request_id=request.request_id,
                )

    def EmbedText(self, request, context):
        """
        Handle text embedding request.

        Args:
            request: EmbedRequest protobuf message
            context: gRPC context

        Returns:
            EmbedResponse protobuf message
        """
        self.total_requests += 1
        start_time = time.time()

        try:
            # Check if embedding sharing is enabled
            if not self.share_embeddings:
                self.failed_requests += 1
                return self._create_embed_response(
                    request_id=request.request_id,
                    success=False,
                    error_message="Embedding sharing is disabled on this instance",
                )

            # Check if embedding manager is available
            if not self.embedding_manager or not self.embedding_manager.embedding_model:
                self.failed_requests += 1
                return self._create_embed_response(
                    request_id=request.request_id,
                    success=False,
                    error_message="Embedding model not available",
                )

            # Generate embedding
            self.file_logger.log_info(
                f"RPC: Generating embedding for peer request {request.request_id}"
            )

            embedding = self.embedding_manager.embedding_model.encode(
                request.text, show_progress_bar=False
            )

            embedding_time = time.time() - start_time
            self.successful_requests += 1

            return self._create_embed_response(
                embedding=embedding.tolist(),
                dimension=len(embedding),
                embedding_time_seconds=embedding_time,
                request_id=request.request_id,
                success=True,
            )

        except Exception as e:
            self.failed_requests += 1
            self.file_logger.log_error(f"RPC: Embedding generation failed: {e}")
            return self._create_embed_response(
                request_id=request.request_id, success=False, error_message=str(e)
            )

    def EmbedTextBatch(self, request, context):
        """
        Handle batch text embedding request.

        Args:
            request: EmbedBatchRequest protobuf message
            context: gRPC context

        Returns:
            EmbedBatchResponse protobuf message
        """
        self.total_requests += 1
        start_time = time.time()

        try:
            if not self.share_embeddings:
                self.failed_requests += 1
                return self._create_embed_batch_response(
                    request_id=request.request_id,
                    success=False,
                    error_message="Embedding sharing is disabled",
                )

            if not self.embedding_manager or not self.embedding_manager.embedding_model:
                self.failed_requests += 1
                return self._create_embed_batch_response(
                    request_id=request.request_id,
                    success=False,
                    error_message="Embedding model not available",
                )

            # Generate batch embeddings
            self.file_logger.log_info(
                f"RPC: Generating {len(request.texts)} embeddings for peer"
            )

            embeddings = self.embedding_manager.embedding_model.encode(
                list(request.texts), show_progress_bar=False
            )

            embedding_time = time.time() - start_time
            self.successful_requests += 1

            # Convert to embedding results
            results = [
                self._create_embedding_result(emb.tolist(), idx)
                for idx, emb in enumerate(embeddings)
            ]

            return self._create_embed_batch_response(
                embeddings=results,
                total_time_seconds=embedding_time,
                request_id=request.request_id,
                success=True,
            )

        except Exception as e:
            self.failed_requests += 1
            self.file_logger.log_error(f"RPC: Batch embedding failed: {e}")
            return self._create_embed_batch_response(
                request_id=request.request_id, success=False, error_message=str(e)
            )

    def SearchMemory(self, request, context):
        """
        Handle memory search request.

        Args:
            request: MemorySearchRequest protobuf message
            context: gRPC context

        Returns:
            MemorySearchResponse protobuf message
        """
        self.total_requests += 1
        start_time = time.time()

        try:
            # Check if memory sharing is enabled
            if not self.share_memory:
                self.failed_requests += 1
                return self._create_memory_search_response(
                    request_id=request.request_id,
                    success=False,
                    error_message="Memory sharing is disabled (privacy setting)",
                )

            # Note: Memory search implementation would require username context
            # For privacy, this is intentionally restrictive
            self.file_logger.log_warning(
                "RPC: Memory search requested but not implemented (privacy)"
            )
            self.failed_requests += 1

            return self._create_memory_search_response(
                request_id=request.request_id,
                success=False,
                error_message="Memory search not available for privacy reasons",
            )

        except Exception as e:
            self.failed_requests += 1
            self.file_logger.log_error(f"RPC: Memory search failed: {e}")
            return self._create_memory_search_response(
                request_id=request.request_id, success=False, error_message=str(e)
            )

    def SearchMemoryFederated(self, request, context):
        """
        Handle federated memory search (only embeddings, no content).

        Args:
            request: MemorySearchRequest protobuf message
            context: gRPC context

        Returns:
            MemorySearchResponse protobuf message
        """
        # Similar to SearchMemory but only returns embeddings
        # This preserves privacy while allowing distributed search
        return self.SearchMemory(request, context)

    def HealthCheck(self, request, context):
        """
        Handle health check request.

        Args:
            request: HealthCheckRequest protobuf message
            context: gRPC context

        Returns:
            HealthCheckResponse protobuf message
        """
        try:
            health_data = {
                "status": "healthy",
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "gpu_memory_percent": 0.0,
                "active_requests": 0,
                "total_requests_served": self.total_requests,
                "uptime_seconds": time.time() - self.start_time,
            }

            # Get real health data if monitor available
            if self.health_monitor:
                snapshot = self.health_monitor.get_health_snapshot()
                health_data.update(
                    {
                        "status": snapshot.status.value,
                        "cpu_percent": snapshot.cpu_percent,
                        "memory_percent": snapshot.memory_percent,
                        "gpu_memory_percent": (
                            (
                                snapshot.gpu_memory_used_mb
                                / snapshot.gpu_memory_total_mb
                                * 100
                            )
                            if snapshot.gpu_memory_total_mb
                            else 0.0
                        ),
                    }
                )

            return self._create_health_response(**health_data)

        except Exception as e:
            self.file_logger.log_error(f"RPC: Health check failed: {e}")
            return self._create_health_response(
                status="unhealthy", uptime_seconds=time.time() - self.start_time
            )

    def GetCapabilities(self, request, context):
        """
        Handle capabilities request.

        Args:
            request: CapabilitiesRequest protobuf message
            context: gRPC context

        Returns:
            CapabilitiesResponse protobuf message
        """
        # Build and return capabilities information
        # This will be implemented fully when proto is compiled
        return self._create_capabilities_response()

    def GetMetrics(self, request, context):
        """
        Handle metrics request.

        Args:
            request: MetricsRequest protobuf message
            context: gRPC context

        Returns:
            MetricsResponse protobuf message
        """
        return self._create_metrics_response()

    # Helper methods to create response objects using real protobuf messages

    def _create_generate_response(self, **kwargs):
        """Create GenerateResponse using protobuf."""
        return jenova_pb2.GenerateResponse(
            text=kwargs.get("text", ""),
            tokens_generated=kwargs.get("tokens_generated", 0),
            generation_time_seconds=kwargs.get("generation_time_seconds", 0.0),
            model_name=kwargs.get("model_name", ""),
            request_id=kwargs.get("request_id", ""),
            success=kwargs.get("success", False),
            error_message=kwargs.get("error_message", ""),
        )

    def _create_stream_response(self, **kwargs):
        """Create GenerateStreamResponse using protobuf."""
        return jenova_pb2.GenerateStreamResponse(
            token=kwargs.get("token", ""),
            is_final=kwargs.get("is_final", False),
            request_id=kwargs.get("request_id", ""),
        )

    def _create_embed_response(self, **kwargs):
        """Create EmbedResponse using protobuf."""
        return jenova_pb2.EmbedResponse(
            embedding=kwargs.get("embedding", []),
            dimension=kwargs.get("dimension", 0),
            embedding_time_seconds=kwargs.get("embedding_time_seconds", 0.0),
            request_id=kwargs.get("request_id", ""),
            success=kwargs.get("success", False),
            error_message=kwargs.get("error_message", ""),
        )

    def _create_embed_batch_response(self, **kwargs):
        """Create EmbedBatchResponse using protobuf."""
        return jenova_pb2.EmbedBatchResponse(
            embeddings=kwargs.get("embeddings", []),
            total_time_seconds=kwargs.get("total_time_seconds", 0.0),
            request_id=kwargs.get("request_id", ""),
            success=kwargs.get("success", False),
            error_message=kwargs.get("error_message", ""),
        )

    def _create_embedding_result(self, embedding, index):
        """Create EmbeddingResult using protobuf."""
        return jenova_pb2.EmbeddingResult(embedding=embedding, index=index)

    def _create_memory_search_response(self, **kwargs):
        """Create MemorySearchResponse using protobuf."""
        return jenova_pb2.MemorySearchResponse(
            results=kwargs.get("results", []),
            total_results=kwargs.get("total_results", 0),
            search_time_seconds=kwargs.get("search_time_seconds", 0.0),
            request_id=kwargs.get("request_id", ""),
            success=kwargs.get("success", False),
            error_message=kwargs.get("error_message", ""),
        )

    def _create_health_response(self, **kwargs):
        """Create HealthCheckResponse using protobuf."""
        return jenova_pb2.HealthCheckResponse(
            status=kwargs.get("status", "unknown"),
            cpu_percent=kwargs.get("cpu_percent", 0.0),
            memory_percent=kwargs.get("memory_percent", 0.0),
            gpu_memory_percent=kwargs.get("gpu_memory_percent", 0.0),
            active_requests=kwargs.get("active_requests", 0),
            total_requests_served=kwargs.get("total_requests_served", 0),
            uptime_seconds=kwargs.get("uptime_seconds", 0.0),
        )

    def _create_capabilities_response(self):
        """Create CapabilitiesResponse using protobuf."""
        # Build ResourceSharing message
        sharing = jenova_pb2.ResourceSharing(
            share_llm=self.share_llm,
            share_embeddings=self.share_embeddings,
            share_memory=self.share_memory,
            max_concurrent_requests=self.max_concurrent,
            current_load_percent=0,  # Would need actual load tracking
        )
        
        # Build HardwareCapabilities message
        try:
            cpu_count = psutil.cpu_count(logical=False) or 1
            cpu_threads = psutil.cpu_count(logical=True) or 1
            memory = psutil.virtual_memory()
            total_ram_mb = memory.total // (1024 * 1024)
            available_ram_mb = memory.available // (1024 * 1024)
        except Exception:
            cpu_count = 1
            cpu_threads = 1
            total_ram_mb = 0
            available_ram_mb = 0
        
        hardware = jenova_pb2.HardwareCapabilities(
            cpu_cores=cpu_count,
            cpu_threads=cpu_threads,
            total_ram_mb=total_ram_mb,
            available_ram_mb=available_ram_mb,
            platform=platform.system().lower(),
            architecture=platform.machine(),
        )
        
        # Get model info if available
        models = []
        model_config = self.config.get("model", {})
        if model_config:
            gpu_layers_config = model_config.get("gpu_layers", 0)
            gpu_layers = gpu_layers_config if isinstance(gpu_layers_config, int) else 0
            
            models.append(jenova_pb2.ModelInfo(
                model_name=model_config.get("model_path", "unknown"),
                model_type="llm",
                context_size=model_config.get("context_size", 4096),
                gpu_layers=gpu_layers,
                available=True,
            ))
            models.append(jenova_pb2.ModelInfo(
                model_name=model_config.get("embedding_model", "all-MiniLM-L6-v2"),
                model_type="embedding",
                available=True,
            ))
        
        return jenova_pb2.CapabilitiesResponse(
            instance_id=self.config.get("instance_id", "unknown"),
            instance_name=self.config.get("instance_name", "jenova-instance"),
            hardware=hardware,
            models=models,
            sharing=sharing,
            jenova_version="6.0.0",
            protocol_version="1.0",
        )

    def _create_metrics_response(self):
        """Create MetricsResponse using protobuf."""
        return jenova_pb2.MetricsResponse(
            total_requests=self.total_requests,
            successful_requests=self.successful_requests,
            failed_requests=self.failed_requests,
            uptime_seconds=time.time() - self.start_time,
        )


class JenovaRPCServer:
    """
    gRPC server for distributed JENOVA operations.

    Manages the lifecycle of the gRPC server and coordinates with the
    discovery service to advertise availability.
    """

    def __init__(
        self,
        config: dict,
        file_logger,
        servicer: JenovaRPCServicer,
        security_manager=None,
        port: int = 50051,
    ):
        """
        Initialize RPC server.

        Args:
            config: JENOVA configuration
            file_logger: Logger for file output
            servicer: RPC servicer implementation
            security_manager: Security manager for SSL/TLS
            port: Port to listen on
        """
        self.config = config
        self.file_logger = file_logger
        self.servicer = servicer
        self.security_manager = security_manager
        self.port = port

        self.server: Optional[grpc.Server] = None

        # Configuration
        network_config = config.get("network", {})
        resource_sharing = network_config.get("resource_sharing", {})
        self.max_workers = resource_sharing.get("max_concurrent_requests", 5)

    def start(self):
        """Start the gRPC server."""
        try:
            self.file_logger.log_info(f"Starting gRPC server on port {self.port}...")

            # Create server
            self.server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=self.max_workers)
            )

            # Add servicer to server using generated gRPC code
            jenova_pb2_grpc.add_JenovaRPCServicer_to_server(self.servicer, self.server)

            # Bind port
            if self.security_manager and self.security_manager.is_security_enabled():
                # Use SSL/TLS
                private_key, cert_chain = self.security_manager.get_ssl_credentials()
                credentials = grpc.ssl_server_credentials([(private_key, cert_chain)])
                self.server.add_secure_port(f"[::]:{self.port}", credentials)
                self.file_logger.log_info("gRPC server using SSL/TLS encryption")
            else:
                # Insecure (for testing or private networks)
                self.server.add_insecure_port(f"[::]:{self.port}")
                self.file_logger.log_warning(
                    "gRPC server running WITHOUT encryption (not recommended)"
                )

            # Start server
            self.server.start()

            self.file_logger.log_info(
                f"gRPC server started successfully on port {self.port} "
                f"(max_workers={self.max_workers})"
            )

        except Exception as e:
            self.file_logger.log_error(f"Failed to start gRPC server: {e}")
            raise

    def stop(self, grace_period: int = 5):
        """
        Stop the gRPC server.

        Args:
            grace_period: Seconds to wait for pending requests
        """
        if self.server:
            self.file_logger.log_info("Stopping gRPC server...")
            self.server.stop(grace_period)
            self.file_logger.log_info("gRPC server stopped")

    def wait_for_termination(self):
        """Block until server terminates."""
        if self.server:
            self.server.wait_for_termination()
