# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.
#
# Phase 21: Full Project Remediation & Modernization
# Core Application Module - Structured application initialization and lifecycle

"""
Core application module for The JENOVA Cognitive Architecture.

This module provides the foundational application architecture, replacing the
monolithic 793-line main() function with a clean, testable structure based on
dependency injection and component lifecycle management.

Key Components:
    - Application: Main application class
    - ApplicationBootstrapper: Handles phased initialization
    - DependencyContainer: Manages component dependencies
    - ComponentLifecycle: Manages component lifecycles

Example:
    >>> from jenova.core import Application
    >>> app = Application()
    >>> app.run()
"""

from jenova.core.application import Application
from jenova.core.bootstrap import ApplicationBootstrapper
from jenova.core.container import DependencyContainer
from jenova.core.lifecycle import ComponentLifecycle, LifecyclePhase

__all__ = [
    "Application",
    "ApplicationBootstrapper",
    "DependencyContainer",
    "ComponentLifecycle",
    "LifecyclePhase",
]
