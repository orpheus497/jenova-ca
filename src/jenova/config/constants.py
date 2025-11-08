# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.
#
# Phase 21: Full Project Remediation & Modernization
# Configuration Constants - Eliminates all magic numbers from codebase

"""
Configuration constants for The JENOVA Cognitive Architecture.

This module centralizes all magic numbers, thresholds, and configuration values
that were previously scattered throughout the codebase. This improves maintainability
and makes configuration changes easier.

All constants are organized by module/feature for easy navigation.
"""

from typing import Final

# ============================================================================
# APPLICATION LIFECYCLE CONSTANTS
# ============================================================================

# Progress bar percentages for startup sequence
PROGRESS_LOADING_CONFIG: Final[int] = 10
PROGRESS_INIT_INFRASTRUCTURE: Final[int] = 20
PROGRESS_CHECKING_HEALTH: Final[int] = 30
PROGRESS_LOADING_MODEL: Final[int] = 40
PROGRESS_LOADING_EMBEDDINGS: Final[int] = 50
PROGRESS_INIT_MEMORY: Final[int] = 60
PROGRESS_INIT_COGNITIVE: Final[int] = 70
PROGRESS_INIT_NETWORK: Final[int] = 80
PROGRESS_INIT_CLI_TOOLS: Final[int] = 90
PROGRESS_COMPLETE: Final[int] = 100

# ============================================================================
# TIMEOUT CONSTANTS (seconds)
# ============================================================================

# Command execution timeouts
DEFAULT_COMMAND_TIMEOUT_SECONDS: Final[int] = 30
LONG_COMMAND_TIMEOUT_SECONDS: Final[int] = 120
MODEL_LOAD_TIMEOUT_SECONDS: Final[int] = 300

# LLM generation timeouts
DEFAULT_LLM_TIMEOUT_SECONDS: Final[int] = 240
PLANNING_TIMEOUT_SECONDS: Final[int] = 120
RAG_GENERATION_TIMEOUT_SECONDS: Final[int] = 240
RERANK_TIMEOUT_SECONDS: Final[int] = 30

# Network timeouts
RPC_TIMEOUT_MILLISECONDS: Final[int] = 5000
PEER_HEALTH_CHECK_TIMEOUT_SECONDS: Final[int] = 5
DISCOVERY_SERVICE_TTL_SECONDS: Final[int] = 60

# ============================================================================
# FILE SIZE LIMITS (bytes)
# ============================================================================

# File validation limits
MAX_FILE_SIZE_MB: Final[int] = 100
MAX_FILE_SIZE_BYTES: Final[int] = MAX_FILE_SIZE_MB * 1024 * 1024

# JSON parsing limits
MAX_JSON_FILE_SIZE_MB: Final[int] = 100
MAX_JSON_FILE_SIZE_BYTES: Final[int] = MAX_JSON_FILE_SIZE_MB * 1024 * 1024
MAX_JSON_STRING_SIZE_MB: Final[int] = 10
MAX_JSON_STRING_SIZE_BYTES: Final[int] = MAX_JSON_STRING_SIZE_MB * 1024 * 1024
MAX_JSON_DEPTH: Final[int] = 100

# Backup size limits
MAX_BACKUP_SIZE_MB: Final[int] = 500
MAX_BACKUP_SIZE_BYTES: Final[int] = MAX_BACKUP_SIZE_MB * 1024 * 1024

# Input validation limits
MAX_INPUT_LENGTH_KB: Final[int] = 100
MAX_INPUT_LENGTH_BYTES: Final[int] = MAX_INPUT_LENGTH_KB * 1024
MAX_PATH_LENGTH: Final[int] = 4096

# ============================================================================
# MODEL CONFIGURATION CONSTANTS
# ============================================================================

# GPU layer configuration
GPU_LAYERS_ALL: Final[int] = -1
GPU_LAYERS_CPU_ONLY: Final[int] = 0
GPU_LAYERS_AUTO: Final[str] = "auto"

# Default model parameters
DEFAULT_THREADS: Final[int] = -1  # Auto-detect
DEFAULT_N_BATCH: Final[int] = 256
DEFAULT_CONTEXT_SIZE: Final[int] = 4096
DEFAULT_MAX_TOKENS: Final[int] = 512
DEFAULT_TEMPERATURE: Final[float] = 0.7
DEFAULT_TOP_P: Final[float] = 0.95

# Hardware tiers for auto-detection
HARDWARE_TIER_LOW_VRAM_GB: Final[int] = 4
HARDWARE_TIER_MEDIUM_VRAM_GB: Final[int] = 8
HARDWARE_TIER_HIGH_VRAM_GB: Final[int] = 12

# ============================================================================
# MEMORY SYSTEM CONSTANTS
# ============================================================================

# Memory search result counts
SEMANTIC_N_RESULTS_DEFAULT: Final[int] = 5
EPISODIC_N_RESULTS_DEFAULT: Final[int] = 3
PROCEDURAL_N_RESULTS_DEFAULT: Final[int] = 3
INSIGHT_N_RESULTS_DEFAULT: Final[int] = 5

# Reflection intervals (conversation turns)
DEFAULT_REFLECTION_INTERVAL: Final[int] = 3
DEFAULT_INSIGHT_INTERVAL: Final[int] = 5
DEFAULT_ASSUMPTION_INTERVAL: Final[int] = 7
DEFAULT_VERIFY_ASSUMPTION_INTERVAL: Final[int] = 8
DEFAULT_DEEP_REFLECTION_INTERVAL: Final[int] = 10
DEFAULT_REORGANIZE_INSIGHTS_INTERVAL: Final[int] = 10
DEFAULT_PROCESS_DOCUMENTS_INTERVAL: Final[int] = 15

# Cortex graph configuration
CORTEX_PRUNE_INTERVAL_CYCLES: Final[int] = 10
CORTEX_MAX_AGE_DAYS: Final[int] = 30
CORTEX_MIN_CENTRALITY: Final[float] = 0.1

# Default relationship weights
RELATIONSHIP_WEIGHT_ELABORATES_ON: Final[float] = 1.5
RELATIONSHIP_WEIGHT_CONFLICTS_WITH: Final[float] = 2.0
RELATIONSHIP_WEIGHT_RELATED_TO: Final[float] = 1.0
RELATIONSHIP_WEIGHT_DEVELOPS: Final[float] = 1.5
RELATIONSHIP_WEIGHT_SUMMARIZES: Final[float] = 1.2

# RAG system caching
DEFAULT_RAG_CACHE_SIZE: Final[int] = 100

# ============================================================================
# COMPRESSION & DEDUPLICATION CONSTANTS
# ============================================================================

# Compression tiers (days)
COMPRESSION_HOT_TIER_DAYS: Final[int] = 7
COMPRESSION_WARM_TIER_DAYS: Final[int] = 30
COMPRESSION_COLD_TIER_DAYS: Final[int] = 90

# Compression algorithms
COMPRESSION_LZ4_LEVEL: Final[int] = 0  # Fast
COMPRESSION_ZSTD_WARM_LEVEL: Final[int] = 3  # Balanced
COMPRESSION_ZSTD_COLD_LEVEL: Final[int] = 19  # Maximum

# Deduplication
DEDUPLICATION_HASH_SIZE_BYTES: Final[int] = 32  # 256 bits

# ============================================================================
# SECURITY CONSTANTS
# ============================================================================

# Cryptography parameters
PBKDF2_ITERATIONS: Final[int] = 600000  # OWASP 2024 recommendation
SALT_SIZE_BYTES: Final[int] = 32  # 256 bits
RSA_KEY_SIZE_BITS: Final[int] = 2048
RSA_PUBLIC_EXPONENT: Final[int] = 65537

# Scrypt parameters (for network security)
SCRYPT_N: Final[int] = 2**14  # 16,384 (CPU/memory cost)
SCRYPT_R: Final[int] = 8      # Block size
SCRYPT_P: Final[int] = 1      # Parallelization
SCRYPT_KEY_LENGTH: Final[int] = 32

# JWT token validity
JWT_TOKEN_VALIDITY_SECONDS: Final[int] = 3600  # 1 hour

# Rate limiting (token bucket)
RATE_LIMIT_DEFAULT_CAPACITY: Final[int] = 100
RATE_LIMIT_DEFAULT_REFILL_RATE: Final[float] = 1.0  # ops/second

# Prompt sanitizer
MAX_PROMPT_LENGTH_KB: Final[int] = 50
MAX_PROMPT_LENGTH_BYTES: Final[int] = MAX_PROMPT_LENGTH_KB * 1024

# File permissions
SECURE_FILE_PERMISSIONS: Final[int] = 0o600  # Owner read/write only
SECURE_DIR_PERMISSIONS: Final[int] = 0o700   # Owner rwx only

# ============================================================================
# NETWORK & DISTRIBUTED COMPUTING CONSTANTS
# ============================================================================

# gRPC configuration
DEFAULT_GRPC_PORT: Final[int] = 50051
MAX_CONCURRENT_PEER_REQUESTS: Final[int] = 5

# Peer selection strategies
STRATEGY_LOCAL_FIRST: Final[str] = "local_first"
STRATEGY_LOAD_BALANCED: Final[str] = "load_balanced"
STRATEGY_FASTEST: Final[str] = "fastest"
STRATEGY_PARALLEL_VOTING: Final[str] = "parallel_voting"
STRATEGY_ROUND_ROBIN: Final[str] = "round_robin"

# ============================================================================
# UI & DISPLAY CONSTANTS
# ============================================================================

# Health display refresh rate
HEALTH_DISPLAY_REFRESH_SECONDS: Final[float] = 2.0
HEALTH_DISPLAY_WARNING_CPU_PERCENT: Final[float] = 80.0
HEALTH_DISPLAY_CRITICAL_CPU_PERCENT: Final[float] = 95.0
HEALTH_DISPLAY_WARNING_MEMORY_PERCENT: Final[float] = 80.0
HEALTH_DISPLAY_CRITICAL_MEMORY_PERCENT: Final[float] = 95.0

# Progress bar configuration
PROGRESS_BAR_WIDTH: Final[int] = 40
PROGRESS_BAR_FILL_CHAR: Final[str] = "█"
PROGRESS_BAR_EMPTY_CHAR: Final[str] = "░"

# Terminal display limits
MAX_DISPLAY_LINES: Final[int] = 1000
TRUNCATE_LONG_OUTPUT_LINES: Final[int] = 500

# ============================================================================
# CODE ANALYSIS CONSTANTS
# ============================================================================

# Complexity thresholds
CYCLOMATIC_COMPLEXITY_WARNING: Final[int] = 10
CYCLOMATIC_COMPLEXITY_CRITICAL: Final[int] = 20
MAX_FUNCTION_LENGTH_LINES: Final[int] = 100
MAX_CLASS_LENGTH_LINES: Final[int] = 500
MAX_PARAMETER_COUNT: Final[int] = 5

# Maintainability index grades
MAINTAINABILITY_A: Final[float] = 20.0
MAINTAINABILITY_B: Final[float] = 10.0
MAINTAINABILITY_C: Final[float] = 0.0

# ============================================================================
# ORCHESTRATION & TASK MANAGEMENT CONSTANTS
# ============================================================================

# Background task limits
MAX_BACKGROUND_TASKS: Final[int] = 10
TASK_OUTPUT_MAX_SIZE_LINES: Final[int] = 10000
TASK_CLEANUP_AFTER_HOURS: Final[int] = 24

# Checkpoint intervals
DEFAULT_CHECKPOINT_INTERVAL_STEPS: Final[int] = 10

# ============================================================================
# OBSERVABILITY CONSTANTS
# ============================================================================

# Metrics export
PROMETHEUS_EXPORT_PORT: Final[int] = 8000
METRICS_COLLECTION_INTERVAL_SECONDS: Final[int] = 60

# Tracing
JAEGER_AGENT_HOST: Final[str] = "localhost"
JAEGER_AGENT_PORT: Final[int] = 6831

# ============================================================================
# TESTING CONSTANTS
# ============================================================================

# Test timeouts
TEST_SHORT_TIMEOUT_SECONDS: Final[float] = 1.0
TEST_MEDIUM_TIMEOUT_SECONDS: Final[float] = 5.0
TEST_LONG_TIMEOUT_SECONDS: Final[float] = 30.0

# Test data sizes
TEST_SMALL_DATA_SIZE: Final[int] = 100
TEST_MEDIUM_DATA_SIZE: Final[int] = 1000
TEST_LARGE_DATA_SIZE: Final[int] = 10000

# ============================================================================
# FEATURE-SPECIFIC CONSTANTS
# ============================================================================

# Adaptive Context Window (Feature 1)
CONTEXT_WINDOW_MIN_PRIORITY_SCORE: Final[float] = 0.3
CONTEXT_WINDOW_COMPRESSION_THRESHOLD: Final[float] = 0.8

# Emotional Intelligence (Feature 2)
EMOTION_DETECTION_CONFIDENCE_THRESHOLD: Final[float] = 0.6
EMOTION_VECTOR_DIMENSIONS: Final[int] = 6  # joy, sadness, anger, fear, surprise, disgust

# Self-Optimization (Feature 3)
OPTIMIZATION_MIN_SAMPLES: Final[int] = 10
OPTIMIZATION_RUN_INTERVAL_HOURS: Final[int] = 24
BAYESIAN_OPTIMIZATION_ITERATIONS: Final[int] = 50

# Knowledge Graph Visualization (Feature 4)
GRAPH_VIZ_SERVER_PORT: Final[int] = 8080
GRAPH_VIZ_MAX_NODES_DISPLAY: Final[int] = 500

# Plugin Architecture (Feature 6)
PLUGIN_TIMEOUT_SECONDS: Final[int] = 30
MAX_PLUGINS_LOADED: Final[int] = 20

# Voice Interface (Feature 8)
VOICE_SAMPLE_RATE_HZ: Final[int] = 16000
VOICE_RECORDING_TIMEOUT_SECONDS: Final[int] = 30
VOICE_SILENCE_THRESHOLD_DB: Final[float] = -40.0

# ============================================================================
# ENVIRONMENT & PATH CONSTANTS
# ============================================================================

# Default paths (relative to user home)
DEFAULT_USER_DATA_ROOT: Final[str] = "~/.jenova-ai"
DEFAULT_MODELS_DIR_SYSTEM: Final[str] = "/usr/local/share/models"
DEFAULT_MODELS_DIR_LOCAL: Final[str] = "./models"
DEFAULT_CERTS_DIR: Final[str] = "~/.jenova-ai/certs"
DEFAULT_PLUGINS_DIR: Final[str] = "~/.jenova-ai/plugins"
DEFAULT_LOGS_DIR: Final[str] = "~/.jenova-ai/logs"
DEFAULT_BACKUPS_DIR: Final[str] = "~/.jenova-ai/backups"

# ============================================================================
# VERSION CONSTANTS
# ============================================================================

# Current version
JENOVA_VERSION: Final[str] = "5.3.0"
JENOVA_VERSION_MAJOR: Final[int] = 5
JENOVA_VERSION_MINOR: Final[int] = 3
JENOVA_VERSION_PATCH: Final[int] = 0

# Minimum compatible version for distributed peers
MIN_COMPATIBLE_VERSION: Final[str] = "5.0.0"

# ============================================================================
# MISCELLANEOUS CONSTANTS
# ============================================================================

# Retry configuration
DEFAULT_RETRY_COUNT: Final[int] = 3
DEFAULT_RETRY_BACKOFF_SECONDS: Final[float] = 2.0
MAX_RETRY_BACKOFF_SECONDS: Final[float] = 60.0

# Batch sizes
DEFAULT_BATCH_SIZE: Final[int] = 32
LARGE_BATCH_SIZE: Final[int] = 128

# Buffer sizes
DEFAULT_BUFFER_SIZE_KB: Final[int] = 64
LARGE_BUFFER_SIZE_KB: Final[int] = 256

# String truncation
MAX_LOG_MESSAGE_LENGTH: Final[int] = 1000
MAX_ERROR_MESSAGE_LENGTH: Final[int] = 500
TRUNCATION_SUFFIX: Final[str] = "..."

# Boolean flags
ENABLE_DEBUG_MODE: Final[bool] = False
ENABLE_VERBOSE_LOGGING: Final[bool] = False
ENABLE_PERFORMANCE_PROFILING: Final[bool] = False
