# The JENOVA Cognitive Architecture - Distributed LLM Interface
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Distributed LLM interface for multi-instance inference.

This module extends the standard LLM interface to support distributed generation
across multiple JENOVA instances on the LAN, enabling:
- Load balancing across peers
- Parallel generation with voting
- Automatic failover
- Performance optimization through peer selection
"""

import threading
import time
from enum import Enum
from typing import List, Optional, Tuple

from jenova.llm.llm_interface import LLMInterface


class DistributionStrategy(Enum):
    """Strategies for distributing LLM requests."""
    LOCAL_FIRST = "local_first"  # Try local, fallback to peers
    LOAD_BALANCED = "load_balanced"  # Distribute across peers by load
    FASTEST_PEER = "fastest"  # Route to fastest peer
    PARALLEL_VOTING = "parallel_voting"  # Generate on multiple peers, vote on best
    ROUND_ROBIN = "round_robin"  # Simple round-robin distribution


class DistributedLLMInterface:
    """
    Distributed LLM interface for load-balanced inference across peers.

    This class wraps a local LLM interface and can distribute generation
    requests to peer JENOVA instances based on configurable strategies.

    Features:
    - Multiple distribution strategies
    - Automatic failover
    - Parallel generation with consensus
    - Performance tracking
    """

    def __init__(
        self,
        config: dict,
        file_logger,
        ui_logger=None,
        local_llm_interface: Optional[LLMInterface] = None,
        rpc_client=None,
        peer_manager=None,
        network_metrics=None
    ):
        """
        Initialize distributed LLM interface.

        Args:
            config: JENOVA configuration
            file_logger: Logger for file output
            ui_logger: Optional UI logger
            local_llm_interface: Local LLM interface
            rpc_client: RPC client for peer communication
            peer_manager: Peer manager for peer selection
            network_metrics: Network metrics collector
        """
        self.config = config
        self.file_logger = file_logger
        self.ui_logger = ui_logger
        self.local_llm = local_llm_interface
        self.rpc_client = rpc_client
        self.peer_manager = peer_manager
        self.network_metrics = network_metrics

        # Configuration
        network_config = config.get('network', {})
        self.network_enabled = network_config.get('enabled', False)

        peer_selection = network_config.get('peer_selection', {})
        strategy_name = peer_selection.get('strategy', 'local_first')
        self.strategy = DistributionStrategy(strategy_name)

        # Round-robin counter
        self.round_robin_counter = 0
        self.round_robin_lock = threading.Lock()

        # Statistics
        self.local_generations = 0
        self.distributed_generations = 0
        self.failed_generations = 0

        self.file_logger.log_info(
            f"Distributed LLM interface initialized (strategy: {self.strategy.value}, "
            f"network_enabled: {self.network_enabled})"
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
        force_local: bool = False,
        force_distributed: bool = False
    ) -> str:
        """
        Generate text using local or distributed LLM.

        Args:
            prompt: Text prompt
            temperature: Generation temperature
            top_p: Top-p sampling
            max_tokens: Maximum tokens to generate
            force_local: Force local generation only
            force_distributed: Force distributed generation only

        Returns:
            Generated text
        """
        # If network is disabled or forced local, use local LLM
        if not self.network_enabled or force_local:
            return self._generate_local(prompt, temperature, top_p, max_tokens)

        # If forced distributed, skip local option
        if force_distributed:
            result = self._generate_distributed(prompt, temperature, top_p, max_tokens)
            if result:
                return result
            else:
                # Fallback to local if distributed fails
                self.file_logger.log_warning(
                    "Distributed generation failed, falling back to local"
                )
                return self._generate_local(prompt, temperature, top_p, max_tokens)

        # Apply strategy
        if self.strategy == DistributionStrategy.LOCAL_FIRST:
            return self._strategy_local_first(prompt, temperature, top_p, max_tokens)

        elif self.strategy == DistributionStrategy.LOAD_BALANCED:
            return self._strategy_load_balanced(prompt, temperature, top_p, max_tokens)

        elif self.strategy == DistributionStrategy.FASTEST_PEER:
            return self._strategy_fastest_peer(prompt, temperature, top_p, max_tokens)

        elif self.strategy == DistributionStrategy.PARALLEL_VOTING:
            return self._strategy_parallel_voting(prompt, temperature, top_p, max_tokens)

        elif self.strategy == DistributionStrategy.ROUND_ROBIN:
            return self._strategy_round_robin(prompt, temperature, top_p, max_tokens)

        else:
            # Default to local-first
            return self._strategy_local_first(prompt, temperature, top_p, max_tokens)

    def _generate_local(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int
    ) -> str:
        """Generate using local LLM."""
        if not self.local_llm:
            raise RuntimeError("Local LLM not available")

        self.local_generations += 1
        return self.local_llm.generate(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )

    def _generate_distributed(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        peer_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate using a peer LLM.

        Args:
            prompt: Text prompt
            temperature: Generation temperature
            top_p: Top-p sampling
            max_tokens: Maximum tokens
            peer_id: Specific peer ID (auto-selected if None)

        Returns:
            Generated text or None on failure
        """
        if not self.rpc_client:
            self.file_logger.log_warning("RPC client not available")
            return None

        start_time = time.time()

        try:
            result = self.rpc_client.generate_text(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                peer_id=peer_id
            )

            if result:
                self.distributed_generations += 1

                # Record metrics
                if self.network_metrics and peer_id:
                    peer_conn = self.peer_manager.get_peer_connection(peer_id)
                    if peer_conn:
                        latency_ms = (time.time() - start_time) * 1000
                        self.network_metrics.record_request(
                            peer_id=peer_id,
                            peer_name=peer_conn.peer_info.instance_name,
                            request_type='llm',
                            latency_ms=latency_ms,
                            success=True
                        )

                return result
            else:
                self.failed_generations += 1
                return None

        except Exception as e:
            self.failed_generations += 1
            self.file_logger.log_error(f"Distributed generation failed: {e}")

            # Record failure
            if self.network_metrics and peer_id:
                peer_conn = self.peer_manager.get_peer_connection(peer_id)
                if peer_conn:
                    latency_ms = (time.time() - start_time) * 1000
                    self.network_metrics.record_request(
                        peer_id=peer_id,
                        peer_name=peer_conn.peer_info.instance_name,
                        request_type='llm',
                        latency_ms=latency_ms,
                        success=False
                    )

            return None

    def _strategy_local_first(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int
    ) -> str:
        """LOCAL_FIRST strategy: Try local, fallback to distributed."""
        try:
            return self._generate_local(prompt, temperature, top_p, max_tokens)
        except Exception as e:
            self.file_logger.log_warning(f"Local generation failed: {e}, trying distributed")
            result = self._generate_distributed(prompt, temperature, top_p, max_tokens)
            if result:
                return result
            else:
                raise RuntimeError("Both local and distributed generation failed")

    def _strategy_load_balanced(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int
    ) -> str:
        """LOAD_BALANCED strategy: Select least loaded instance (local or peer)."""
        if not self.peer_manager:
            return self._generate_local(prompt, temperature, top_p, max_tokens)

        # Get available peers
        connected_peers = self.peer_manager.get_connected_peer_count()

        if connected_peers == 0:
            # No peers, use local
            return self._generate_local(prompt, temperature, top_p, max_tokens)

        # Select best peer
        peer_id = self.peer_manager.select_peer_for_task('llm')

        if peer_id:
            # Try peer first
            result = self._generate_distributed(
                prompt, temperature, top_p, max_tokens, peer_id=peer_id
            )
            if result:
                return result

        # Fallback to local
        return self._generate_local(prompt, temperature, top_p, max_tokens)

    def _strategy_fastest_peer(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int
    ) -> str:
        """FASTEST_PEER strategy: Route to peer with best latency."""
        if not self.peer_manager:
            return self._generate_local(prompt, temperature, top_p, max_tokens)

        # Select fastest peer
        peer_id = self.peer_manager.select_peer_for_task('llm')

        if peer_id:
            result = self._generate_distributed(
                prompt, temperature, top_p, max_tokens, peer_id=peer_id
            )
            if result:
                return result

        # Fallback to local
        return self._generate_local(prompt, temperature, top_p, max_tokens)

    def _strategy_parallel_voting(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int
    ) -> str:
        """
        PARALLEL_VOTING strategy: Generate on multiple instances and vote.

        This strategy:
        1. Generates on local + 2 peers in parallel
        2. Uses voting/consensus to select best result
        3. Provides higher quality but slower
        """
        if not self.peer_manager:
            return self._generate_local(prompt, temperature, top_p, max_tokens)

        # Get up to 2 peers
        all_peers = self.peer_manager.get_all_peers()
        available_peers = [
            p.peer_info.instance_id for p in all_peers
            if p.status.value in ['connected', 'degraded'] and
            p.capabilities and p.capabilities.share_llm
        ]

        if len(available_peers) < 2:
            # Not enough peers for voting, fallback to local-first
            return self._strategy_local_first(prompt, temperature, top_p, max_tokens)

        results = []
        threads = []

        # Generate locally
        def generate_local_thread():
            try:
                result = self._generate_local(prompt, temperature, top_p, max_tokens)
                results.append(('local', result))
            except Exception as e:
                self.file_logger.log_error(f"Parallel local generation failed: {e}")

        # Generate on peers
        def generate_peer_thread(peer_id):
            try:
                result = self._generate_distributed(
                    prompt, temperature, top_p, max_tokens, peer_id=peer_id
                )
                if result:
                    results.append((peer_id, result))
            except Exception as e:
                self.file_logger.log_error(f"Parallel peer generation failed: {e}")

        # Start threads
        local_thread = threading.Thread(target=generate_local_thread)
        local_thread.start()
        threads.append(local_thread)

        for peer_id in available_peers[:2]:
            peer_thread = threading.Thread(target=generate_peer_thread, args=(peer_id,))
            peer_thread.start()
            threads.append(peer_thread)

        # Wait for all to complete (with timeout)
        for thread in threads:
            thread.join(timeout=60)  # 60s timeout per thread

        # Vote on results (for now, just take the longest response as "best")
        # In a production system, this could use semantic similarity, perplexity, etc.
        if results:
            best_result = max(results, key=lambda x: len(x[1]))
            self.file_logger.log_info(
                f"Parallel voting complete: {len(results)} results, "
                f"selected from {best_result[0]}"
            )
            return best_result[1]
        else:
            raise RuntimeError("All parallel generations failed")

    def _strategy_round_robin(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int
    ) -> str:
        """ROUND_ROBIN strategy: Distribute requests evenly."""
        if not self.peer_manager:
            return self._generate_local(prompt, temperature, top_p, max_tokens)

        with self.round_robin_lock:
            # Get all available instances (local + peers)
            available_peers = [
                p.peer_info.instance_id for p in self.peer_manager.get_all_peers()
                if p.status.value in ['connected', 'degraded'] and
                p.capabilities and p.capabilities.share_llm
            ]

            if not available_peers:
                # No peers, use local
                return self._generate_local(prompt, temperature, top_p, max_tokens)

            # Add "local" as an option
            instances = ['local'] + available_peers

            # Select next instance
            selected_idx = self.round_robin_counter % len(instances)
            self.round_robin_counter += 1
            selected = instances[selected_idx]

        # Generate
        if selected == 'local':
            return self._generate_local(prompt, temperature, top_p, max_tokens)
        else:
            result = self._generate_distributed(
                prompt, temperature, top_p, max_tokens, peer_id=selected
            )
            if result:
                return result
            else:
                # Fallback to local
                return self._generate_local(prompt, temperature, top_p, max_tokens)

    def get_stats(self) -> dict:
        """Get generation statistics."""
        total = self.local_generations + self.distributed_generations + self.failed_generations
        return {
            'local_generations': self.local_generations,
            'distributed_generations': self.distributed_generations,
            'failed_generations': self.failed_generations,
            'total_generations': total,
            'distributed_percentage': (
                self.distributed_generations / total * 100
                if total > 0 else 0.0
            ),
            'failure_rate': (
                self.failed_generations / total * 100
                if total > 0 else 0.0
            ),
            'strategy': self.strategy.value,
            'network_enabled': self.network_enabled
        }

    def close(self):
        """Close resources."""
        if self.local_llm:
            self.local_llm.close()
