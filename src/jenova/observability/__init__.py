# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Observability module for JENOVA Cognitive Architecture.

Provides distributed tracing, metrics export, and monitoring capabilities
using OpenTelemetry.

Phase 20 Feature #2: Distributed Tracing with OpenTelemetry
- Automatic span creation for all major operations
- Trace context propagation across RPC calls
- Integration with Jaeger/Zipkin for visualization
- Custom span attributes for cognitive operations
"""

from jenova.observability.tracing import (
    TracingManager,
    create_span,
    get_current_span,
    set_span_attribute,
    set_span_status,
    SpanStatus,
)

from jenova.observability.metrics_exporter import (
    MetricsExporter,
    record_counter,
    record_histogram,
    record_gauge,
)

__all__ = [
    # Tracing
    "TracingManager",
    "create_span",
    "get_current_span",
    "set_span_attribute",
    "set_span_status",
    "SpanStatus",
    # Metrics
    "MetricsExporter",
    "record_counter",
    "record_histogram",
    "record_gauge",
]
