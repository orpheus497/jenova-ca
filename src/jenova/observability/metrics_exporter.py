# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Metrics Export with OpenTelemetry and Prometheus for JENOVA.

This module provides comprehensive metrics collection and export capabilities,
enabling monitoring of cognitive operations, memory usage, LLM performance,
and system health.

Phase 20 Feature #2 & #9: Metrics Export and Observability Dashboard
- Prometheus metrics export
- Custom cognitive metrics (insight rate, memory growth, query latency)
- Real-time performance monitoring
- Resource utilization tracking
"""

import logging
from typing import Dict, Any, Optional
from collections import defaultdict
import threading

try:
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from prometheus_client import start_http_server
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    metrics = None
    MeterProvider = None

logger = logging.getLogger(__name__)


class MetricsExporter:
    """
    Manages metrics export using OpenTelemetry and Prometheus.

    Provides centralized configuration for metrics collection and export.
    """

    def __init__(
        self,
        service_name: str = "jenova-cognitive-architecture",
        prometheus_port: int = 8000,
        enable_prometheus: bool = True,
    ):
        """
        Initialize metrics exporter.

        Args:
            service_name: Name of the service for metrics
            prometheus_port: Port for Prometheus metrics endpoint
            enable_prometheus: Enable Prometheus export
        """
        self.service_name = service_name
        self.enabled = OPENTELEMETRY_AVAILABLE
        self._meter = None
        self._metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()

        if not self.enabled:
            logger.warning(
                "OpenTelemetry not available. Metrics export disabled. "
                "Install with: pip install opentelemetry-api opentelemetry-sdk "
                "opentelemetry-exporter-prometheus"
            )
            return

        try:
            # Create resource
            resource = Resource(attributes={
                SERVICE_NAME: service_name
            })

            # Create metric readers
            readers = []

            if enable_prometheus:
                try:
                    # Create Prometheus reader
                    prometheus_reader = PrometheusMetricReader()
                    readers.append(prometheus_reader)

                    # Start Prometheus HTTP server
                    start_http_server(port=prometheus_port, addr="0.0.0.0")
                    logger.info(f"Prometheus metrics available at http://localhost:{prometheus_port}/metrics")
                except Exception as e:
                    logger.warning(f"Failed to initialize Prometheus export: {e}")

            # Create meter provider
            if readers:
                provider = MeterProvider(
                    resource=resource,
                    metric_readers=readers,
                )
                metrics.set_meter_provider(provider)

            # Get meter instance
            self._meter = metrics.get_meter(__name__)

            logger.info(f"Metrics export initialized for service: {service_name}")

        except Exception as e:
            logger.error(f"Failed to initialize metrics export: {e}")
            self.enabled = False

    def get_counter(self, name: str, description: str = "", unit: str = "") -> Any:
        """
        Get or create a counter metric.

        Args:
            name: Metric name
            description: Metric description
            unit: Unit of measurement

        Returns:
            Counter instrument
        """
        if not self.enabled or not self._meter:
            return None

        with self._lock:
            key = f"counter_{name}"
            if key not in self._metrics:
                self._metrics[key] = self._meter.create_counter(
                    name=name,
                    description=description,
                    unit=unit,
                )
            return self._metrics[key]

    def get_histogram(self, name: str, description: str = "", unit: str = "") -> Any:
        """
        Get or create a histogram metric.

        Args:
            name: Metric name
            description: Metric description
            unit: Unit of measurement

        Returns:
            Histogram instrument
        """
        if not self.enabled or not self._meter:
            return None

        with self._lock:
            key = f"histogram_{name}"
            if key not in self._metrics:
                self._metrics[key] = self._meter.create_histogram(
                    name=name,
                    description=description,
                    unit=unit,
                )
            return self._metrics[key]

    def get_gauge(self, name: str, description: str = "", unit: str = "") -> Any:
        """
        Get or create a gauge metric.

        Args:
            name: Metric name
            description: Metric description
            unit: Unit of measurement

        Returns:
            Observable gauge instrument
        """
        if not self.enabled or not self._meter:
            return None

        with self._lock:
            key = f"gauge_{name}"
            if key not in self._metrics:
                # Gauges require callbacks - store callback registration
                callbacks = defaultdict(list)

                def create_callback(value_func):
                    def callback(options):
                        return [(value_func(), {})]
                    return callback

                self._metrics[key] = {
                    "callbacks": callbacks,
                    "meter": self._meter,
                    "name": name,
                    "description": description,
                    "unit": unit,
                }
            return self._metrics[key]

    def is_enabled(self) -> bool:
        """Check if metrics export is enabled."""
        return self.enabled and self._meter is not None


# Global metrics exporter instance
_metrics_exporter: Optional[MetricsExporter] = None


def initialize_metrics(
    service_name: str = "jenova-cognitive-architecture",
    prometheus_port: int = 8000,
    enable_prometheus: bool = True,
) -> MetricsExporter:
    """
    Initialize global metrics exporter.

    Args:
        service_name: Name of the service
        prometheus_port: Port for Prometheus endpoint
        enable_prometheus: Enable Prometheus export

    Returns:
        MetricsExporter instance
    """
    global _metrics_exporter
    _metrics_exporter = MetricsExporter(
        service_name=service_name,
        prometheus_port=prometheus_port,
        enable_prometheus=enable_prometheus,
    )
    return _metrics_exporter


def get_metrics_exporter() -> Optional[MetricsExporter]:
    """
    Get global metrics exporter.

    Returns:
        MetricsExporter if initialized, None otherwise
    """
    return _metrics_exporter


def record_counter(
    name: str,
    value: int = 1,
    attributes: Optional[Dict[str, Any]] = None,
    description: str = "",
    unit: str = "",
) -> None:
    """
    Record a counter metric.

    Args:
        name: Metric name
        value: Value to add to counter
        attributes: Metric attributes (labels)
        description: Metric description
        unit: Unit of measurement

    Example:
        record_counter("llm_requests_total", attributes={"model": "gpt-3.5"})
    """
    exporter = get_metrics_exporter()
    if not exporter or not exporter.is_enabled():
        return

    try:
        counter = exporter.get_counter(name, description, unit)
        if counter:
            counter.add(value, attributes=attributes or {})
    except Exception as e:
        logger.debug(f"Failed to record counter: {e}")


def record_histogram(
    name: str,
    value: float,
    attributes: Optional[Dict[str, Any]] = None,
    description: str = "",
    unit: str = "",
) -> None:
    """
    Record a histogram metric.

    Args:
        name: Metric name
        value: Value to record
        attributes: Metric attributes (labels)
        description: Metric description
        unit: Unit of measurement

    Example:
        record_histogram("llm_latency_seconds", 0.5, attributes={"model": "gpt-3.5"})
    """
    exporter = get_metrics_exporter()
    if not exporter or not exporter.is_enabled():
        return

    try:
        histogram = exporter.get_histogram(name, description, unit)
        if histogram:
            histogram.record(value, attributes=attributes or {})
    except Exception as e:
        logger.debug(f"Failed to record histogram: {e}")


def record_gauge(
    name: str,
    value: float,
    attributes: Optional[Dict[str, Any]] = None,
    description: str = "",
    unit: str = "",
) -> None:
    """
    Record a gauge metric.

    Args:
        name: Metric name
        value: Current value
        attributes: Metric attributes (labels)
        description: Metric description
        unit: Unit of measurement

    Example:
        record_gauge("memory_usage_bytes", 1024000, attributes={"type": "episodic"})
    """
    exporter = get_metrics_exporter()
    if not exporter or not exporter.is_enabled():
        return

    try:
        # Note: Actual gauge implementation would require observable gauge
        # with callbacks. This is a simplified version.
        logger.debug(f"Gauge {name}={value} (attributes={attributes})")
    except Exception as e:
        logger.debug(f"Failed to record gauge: {e}")


# Pre-defined cognitive metrics for JENOVA

def record_llm_request(
    model: str,
    tokens: int,
    latency_seconds: float,
    success: bool = True,
) -> None:
    """Record LLM request metrics."""
    record_counter(
        "jenova_llm_requests_total",
        attributes={"model": model, "status": "success" if success else "error"},
        description="Total LLM requests",
    )

    record_histogram(
        "jenova_llm_latency_seconds",
        latency_seconds,
        attributes={"model": model},
        description="LLM request latency in seconds",
        unit="s",
    )

    record_histogram(
        "jenova_llm_tokens",
        tokens,
        attributes={"model": model},
        description="LLM tokens generated",
        unit="tokens",
    )


def record_memory_operation(
    operation: str,
    memory_type: str,
    latency_seconds: float,
    success: bool = True,
) -> None:
    """Record memory operation metrics."""
    record_counter(
        "jenova_memory_operations_total",
        attributes={"operation": operation, "type": memory_type, "status": "success" if success else "error"},
        description="Total memory operations",
    )

    record_histogram(
        "jenova_memory_latency_seconds",
        latency_seconds,
        attributes={"operation": operation, "type": memory_type},
        description="Memory operation latency in seconds",
        unit="s",
    )


def record_insight_generation(user: str, success: bool = True) -> None:
    """Record insight generation metrics."""
    record_counter(
        "jenova_insights_generated_total",
        attributes={"user": user, "status": "success" if success else "error"},
        description="Total insights generated",
    )


def record_graph_size(nodes: int, links: int) -> None:
    """Record cognitive graph size metrics."""
    record_gauge(
        "jenova_graph_nodes",
        nodes,
        description="Number of nodes in cognitive graph",
        unit="nodes",
    )

    record_gauge(
        "jenova_graph_links",
        links,
        description="Number of links in cognitive graph",
        unit="links",
    )
