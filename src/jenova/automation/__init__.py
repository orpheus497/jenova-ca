# The JENOVA Cognitive Architecture - Automation Module
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Workflow automation and custom commands module for JENOVA.

Provides custom command management, event-driven hooks system,
template processing, and predefined workflow library.
"""

from jenova.automation.custom_commands import CustomCommandManager
from jenova.automation.hooks_system import HooksSystem
from jenova.automation.template_engine import TemplateEngine
from jenova.automation.workflow_library import WorkflowLibrary

__all__ = ['CustomCommandManager', 'HooksSystem', 'TemplateEngine', 'WorkflowLibrary']

__version__ = '5.2.0'
