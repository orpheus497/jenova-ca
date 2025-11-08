# The JENOVA Cognitive Architecture - Network Metrics
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Network performance metrics and monitoring for distributed JENOVA.

Tracks latency, bandwidth, request distribution, and peer performance
to provide visibility into distributed operations.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Dict, List, Optional


@dataclass
class NetworkMetricSnapshot:
    """Snapshot of network metrics at a point in time."""
    timestamp: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    active_peers: int
    bytes_sent: int = 0
    bytes_received: int = 0


@dataclass
class PeerMetrics:
    """Metrics for a specific peer."""
    peer_id: str
    peer_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_request_time: float = 0.0
    bytes_sent: int = 0
    bytes_received: int = 0


class NetworkMetricsCollector:
    """
    Collects and aggregates metrics for distributed JENOVA operations.

    Tracks:
    - Request latency (per peer and aggregate)
    - Success/failure rates
    - Load distribution
    - Bandwidth usage
    - Historical trends
    """

    def __init__(self, file_logger, history_size: int = 1000):
        """
        Initialize metrics collector.

        Args:
            file_logger: Logger for file output
            history_size: Number of historical snapshots to retain
        """
        self.file_logger = file_logger
        self.history_size = history_size

        # Peer-specific metrics
        self.peer_metrics: Dict[str, PeerMetrics] = {}
        self.lock = RLock()

        # Aggregate metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

        # Request type breakdown
        self.requests_by_type: Dict[str, int] = defaultdict(int)

        # Historical snapshots
        self.history: deque = deque(maxlen=history_size)

        # Start time
        self.start_time = time.time()

    def record_request(
        self,
        peer_id: str,
        peer_name: str,
        request_type: str,
        latency_ms: float,
        success: bool,
        bytes_sent: int = 0,
        bytes_received: int = 0
    ):
        """
        Record a network request.

        Args:
            peer_id: Peer instance ID
            peer_name: Peer name
            request_type: Type of request (llm, embedding, memory, etc.)
            latency_ms: Request latency in milliseconds
            success: Whether request succeeded
            bytes_sent: Bytes sent in request
            bytes_received: Bytes received in response
        """
        with self.lock:
            # Get or create peer metrics
            if peer_id not in self.peer_metrics:
                self.peer_metrics[peer_id] = PeerMetrics(
                    peer_id=peer_id,
                    peer_name=peer_name
                )

            peer = self.peer_metrics[peer_id]

            # Update peer metrics
            peer.total_requests += 1
            if success:
                peer.successful_requests += 1
            else:
                peer.failed_requests += 1

            peer.response_times.append(latency_ms)
            peer.last_request_time = time.time()
            peer.bytes_sent += bytes_sent
            peer.bytes_received += bytes_received

            # Update aggregate metrics
            self.total_requests += 1
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1

            # Update request type breakdown
            self.requests_by_type[request_type] += 1

    def get_peer_metrics(self, peer_id: str) -> Optional[PeerMetrics]:
        """Get metrics for a specific peer."""
        with self.lock:
            return self.peer_metrics.get(peer_id)

    def get_all_peer_metrics(self) -> Dict[str, PeerMetrics]:
        """Get metrics for all peers."""
        with self.lock:
            return dict(self.peer_metrics)

    def get_aggregate_metrics(self) -> Dict:
        """Get aggregated metrics across all peers."""
        with self.lock:
            # Collect all response times
            all_response_times = []
            active_peers = 0

            for peer in self.peer_metrics.values():
                all_response_times.extend(peer.response_times)
                if peer.last_request_time > time.time() - 300:  # Active in last 5 min
                    active_peers += 1

            # Calculate percentiles
            if all_response_times:
                sorted_times = sorted(all_response_times)
                avg_latency = sum(sorted_times) / len(sorted_times)
                p95_latency = sorted_times[int(len(sorted_times) * 0.95)]
                p99_latency = sorted_times[int(len(sorted_times) * 0.99)]
            else:
                avg_latency = p95_latency = p99_latency = 0.0

            return {
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'success_rate': (
                    self.successful_requests / self.total_requests * 100
                    if self.total_requests > 0 else 0.0
                ),
                'avg_latency_ms': avg_latency,
                'p95_latency_ms': p95_latency,
                'p99_latency_ms': p99_latency,
                'active_peers': active_peers,
                'requests_by_type': dict(self.requests_by_type),
                'uptime_seconds': time.time() - self.start_time
            }

    def take_snapshot(self) -> NetworkMetricSnapshot:
        """Take a snapshot of current metrics."""
        metrics = self.get_aggregate_metrics()

        snapshot = NetworkMetricSnapshot(
            timestamp=time.time(),
            total_requests=metrics['total_requests'],
            successful_requests=metrics['successful_requests'],
            failed_requests=metrics['failed_requests'],
            avg_latency_ms=metrics['avg_latency_ms'],
            p95_latency_ms=metrics['p95_latency_ms'],
            p99_latency_ms=metrics['p99_latency_ms'],
            active_peers=metrics['active_peers']
        )

        with self.lock:
            self.history.append(snapshot)

        return snapshot

    def get_historical_snapshots(self, count: Optional[int] = None) -> List[NetworkMetricSnapshot]:
        """
        Get historical metric snapshots.

        Args:
            count: Number of recent snapshots to return (None = all)

        Returns:
            List of snapshots, newest first
        """
        with self.lock:
            if count is None:
                return list(reversed(self.history))
            else:
                return list(reversed(self.history))[:count]

    def get_peer_performance_summary(self) -> List[Dict]:
        """
        Get performance summary for all peers.

        Returns:
            List of peer performance dictionaries, sorted by total requests
        """
        with self.lock:
            summary = []

            for peer_id, peer in self.peer_metrics.items():
                if peer.response_times:
                    sorted_times = sorted(peer.response_times)
                    avg_latency = sum(sorted_times) / len(sorted_times)
                    p95_latency = sorted_times[int(len(sorted_times) * 0.95)]
                else:
                    avg_latency = p95_latency = 0.0

                summary.append({
                    'peer_id': peer_id,
                    'peer_name': peer.peer_name,
                    'total_requests': peer.total_requests,
                    'success_rate': (
                        peer.successful_requests / peer.total_requests * 100
                        if peer.total_requests > 0 else 0.0
                    ),
                    'avg_latency_ms': avg_latency,
                    'p95_latency_ms': p95_latency,
                    'last_request_seconds_ago': time.time() - peer.last_request_time,
                    'bytes_sent': peer.bytes_sent,
                    'bytes_received': peer.bytes_received
                })

            # Sort by total requests (descending)
            summary.sort(key=lambda x: x['total_requests'], reverse=True)

            return summary

    def get_load_distribution(self) -> Dict[str, float]:
        """
        Get load distribution across peers.

        Returns:
            Dictionary mapping peer_name to percentage of total requests
        """
        with self.lock:
            if self.total_requests == 0:
                return {}

            distribution = {}
            for peer in self.peer_metrics.values():
                percentage = (peer.total_requests / self.total_requests) * 100
                distribution[peer.peer_name] = percentage

            return distribution

    def log_summary(self, top_n: int = 10):
        """
        Log a summary of network metrics.

        Args:
            top_n: Number of top peers to include in summary
        """
        try:
            metrics = self.get_aggregate_metrics()

            self.file_logger.log_info("=== Network Metrics Summary ===")
            self.file_logger.log_info(
                f"Total Requests: {metrics['total_requests']} "
                f"(Success: {metrics['successful_requests']}, "
                f"Failed: {metrics['failed_requests']}, "
                f"Rate: {metrics['success_rate']:.1f}%)"
            )
            self.file_logger.log_info(
                f"Latency: Avg {metrics['avg_latency_ms']:.1f}ms, "
                f"P95 {metrics['p95_latency_ms']:.1f}ms, "
                f"P99 {metrics['p99_latency_ms']:.1f}ms"
            )
            self.file_logger.log_info(
                f"Active Peers: {metrics['active_peers']}"
            )

            # Log request type breakdown
            if metrics['requests_by_type']:
                self.file_logger.log_info("Request Types:")
                for req_type, count in sorted(
                    metrics['requests_by_type'].items(),
                    key=lambda x: x[1],
                    reverse=True
                ):
                    self.file_logger.log_info(f"  {req_type}: {count}")

            # Log top peers
            peer_summary = self.get_peer_performance_summary()
            if peer_summary:
                self.file_logger.log_info(f"Top {top_n} Peers:")
                for peer in peer_summary[:top_n]:
                    self.file_logger.log_info(
                        f"  {peer['peer_name']}: {peer['total_requests']} requests, "
                        f"{peer['success_rate']:.1f}% success, "
                        f"{peer['avg_latency_ms']:.1f}ms avg latency"
                    )

        except Exception as e:
            self.file_logger.log_error(f"Error logging network metrics summary: {e}")

    def reset(self):
        """Reset all metrics."""
        with self.lock:
            self.peer_metrics.clear()
            self.total_requests = 0
            self.successful_requests = 0
            self.failed_requests = 0
            self.requests_by_type.clear()
            self.history.clear()
            self.start_time = time.time()
            self.file_logger.log_info("Network metrics reset")
