# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Distributed Tracing with OpenTelemetry for JENOVA.

This module provides comprehensive distributed tracing capabilities using OpenTelemetry,
enabling full observability across all cognitive operations, LLM calls, memory searches,
and distributed peer requests.

Phase 20 Feature #2: Distributed Tracing with OpenTelemetry
- Automatic span creation for all major operations
- Trace context propagation across RPC calls
- Integration with Jaeger/Zipkin for visualization
- Custom span attributes for cognitive operations
- Performance monitoring and bottleneck identification
"""

import logging
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
from enum import Enum
from functools import wraps

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.trace import Status, StatusCode, Span
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    trace = None
    TracerProvider = None
    Span = None
    Status = None
    StatusCode = None

logger = logging.getLogger(__name__)


class SpanStatus(Enum):
    """Span status codes."""
    OK = "ok"
    ERROR = "error"
    UNSET = "unset"


class TracingManager:
    """
    Manages distributed tracing using OpenTelemetry.

    Provides centralized configuration and lifecycle management for tracing.
    """

    def __init__(
        self,
        service_name: str = "jenova-cognitive-architecture",
        jaeger_host: Optional[str] = None,
        jaeger_port: int = 6831,
        console_export: bool = False,
    ):
        """
        Initialize tracing manager.

        Args:
            service_name: Name of the service for tracing
            jaeger_host: Jaeger agent host (None to disable Jaeger export)
            jaeger_port: Jaeger agent port
            console_export: Enable console export for debugging
        """
        self.service_name = service_name
        self.enabled = OPENTELEMETRY_AVAILABLE
        self._tracer = None

        if not self.enabled:
            logger.warning(
                "OpenTelemetry not available. Tracing disabled. "
                "Install with: pip install opentelemetry-api opentelemetry-sdk"
            )
            return

        try:
            # Create resource with service name
            resource = Resource(attributes={
                SERVICE_NAME: service_name
            })

            # Create tracer provider
            provider = TracerProvider(resource=resource)

            # Add console exporter if requested
            if console_export:
                console_processor = BatchSpanProcessor(ConsoleSpanExporter())
                provider.add_span_processor(console_processor)
                logger.info("Console span export enabled")

            # Add Jaeger exporter if host specified
            if jaeger_host:
                try:
                    jaeger_exporter = JaegerExporter(
                        agent_host_name=jaeger_host,
                        agent_port=jaeger_port,
                    )
                    jaeger_processor = BatchSpanProcessor(jaeger_exporter)
                    provider.add_span_processor(jaeger_processor)
                    logger.info(f"Jaeger export enabled: {jaeger_host}:{jaeger_port}")
                except Exception as e:
                    logger.warning(f"Failed to initialize Jaeger exporter: {e}")

            # Set global tracer provider
            trace.set_tracer_provider(provider)

            # Get tracer instance
            self._tracer = trace.get_tracer(__name__)

            logger.info(f"Tracing initialized for service: {service_name}")

        except Exception as e:
            logger.error(f"Failed to initialize tracing: {e}")
            self.enabled = False

    @property
    def tracer(self):
        """Get the tracer instance."""
        return self._tracer if self.enabled else None

    def is_enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self.enabled and self._tracer is not None


# Global tracing manager instance
_tracing_manager: Optional[TracingManager] = None


def initialize_tracing(
    service_name: str = "jenova-cognitive-architecture",
    jaeger_host: Optional[str] = None,
    jaeger_port: int = 6831,
    console_export: bool = False,
) -> TracingManager:
    """
    Initialize global tracing manager.

    Args:
        service_name: Name of the service
        jaeger_host: Jaeger agent host (None to disable)
        jaeger_port: Jaeger agent port
        console_export: Enable console export

    Returns:
        TracingManager instance
    """
    global _tracing_manager
    _tracing_manager = TracingManager(
        service_name=service_name,
        jaeger_host=jaeger_host,
        jaeger_port=jaeger_port,
        console_export=console_export,
    )
    return _tracing_manager


def get_tracing_manager() -> Optional[TracingManager]:
    """
    Get global tracing manager.

    Returns:
        TracingManager if initialized, None otherwise
    """
    return _tracing_manager


@contextmanager
def create_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    kind: Optional[str] = None,
):
    """
    Create a tracing span as a context manager.

    Args:
        name: Span name
        attributes: Optional span attributes
        kind: Span kind (e.g., 'client', 'server', 'internal')

    Yields:
        Span object if tracing enabled, None otherwise

    Example:
        with create_span("llm_generation", {"model": "gpt-3.5"}) as span:
            result = llm.generate(prompt)
            set_span_attribute(span, "tokens", len(result))
    """
    manager = get_tracing_manager()

    if not manager or not manager.is_enabled():
        # Tracing disabled, yield None
        yield None
        return

    tracer = manager.tracer
    if not tracer:
        yield None
        return

    # Map kind string to SpanKind enum
    span_kind = None
    if kind:
        kind_map = {
            "client": trace.SpanKind.CLIENT,
            "server": trace.SpanKind.SERVER,
            "internal": trace.SpanKind.INTERNAL,
            "producer": trace.SpanKind.PRODUCER,
            "consumer": trace.SpanKind.CONSUMER,
        }
        span_kind = kind_map.get(kind.lower())

    with tracer.start_as_current_span(name, kind=span_kind) as span:
        # Set attributes if provided
        if attributes:
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, value)

        yield span


def trace_function(
    span_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
):
    """
    Decorator to automatically trace a function.

    Args:
        span_name: Custom span name (default: function name)
        attributes: Static attributes to add to span

    Example:
        @trace_function(attributes={"component": "memory"})
        def search_memory(query: str) -> List[Result]:
            return results
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = span_name or f"{func.__module__}.{func.__qualname__}"

            with create_span(name, attributes=attributes) as span:
                try:
                    result = func(*args, **kwargs)
                    if span:
                        set_span_status(span, SpanStatus.OK)
                    return result
                except Exception as e:
                    if span:
                        set_span_status(span, SpanStatus.ERROR)
                        set_span_attribute(span, "error.message", str(e))
                        set_span_attribute(span, "error.type", type(e).__name__)
                    raise

        return wrapper
    return decorator


def get_current_span() -> Optional[Span]:
    """
    Get the currently active span.

    Returns:
        Active span if available, None otherwise
    """
    if not OPENTELEMETRY_AVAILABLE:
        return None

    try:
        return trace.get_current_span()
    except Exception:
        return None


def set_span_attribute(span: Optional[Span], key: str, value: Any) -> None:
    """
    Set an attribute on a span.

    Args:
        span: Span to modify (can be None)
        key: Attribute key
        value: Attribute value
    """
    if span and OPENTELEMETRY_AVAILABLE:
        try:
            span.set_attribute(key, value)
        except Exception as e:
            logger.debug(f"Failed to set span attribute: {e}")


def set_span_status(span: Optional[Span], status: SpanStatus) -> None:
    """
    Set the status of a span.

    Args:
        span: Span to modify (can be None)
        status: Status to set
    """
    if not span or not OPENTELEMETRY_AVAILABLE:
        return

    try:
        if status == SpanStatus.OK:
            span.set_status(Status(StatusCode.OK))
        elif status == SpanStatus.ERROR:
            span.set_status(Status(StatusCode.ERROR))
        else:
            span.set_status(Status(StatusCode.UNSET))
    except Exception as e:
        logger.debug(f"Failed to set span status: {e}")


def add_span_event(
    span: Optional[Span],
    name: str,
    attributes: Optional[Dict[str, Any]] = None
) -> None:
    """
    Add an event to a span.

    Args:
        span: Span to modify (can be None)
        name: Event name
        attributes: Event attributes
    """
    if not span or not OPENTELEMETRY_AVAILABLE:
        return

    try:
        span.add_event(name, attributes=attributes or {})
    except Exception as e:
        logger.debug(f"Failed to add span event: {e}")
