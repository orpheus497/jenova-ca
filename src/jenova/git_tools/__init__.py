# The JENOVA Cognitive Architecture - Git Tools Module
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Git workflow automation module for JENOVA.

Provides comprehensive Git integration including commit generation,
branch management, diff analysis, and workflow automation.
"""

from jenova.git_tools.git_interface import GitInterface
from jenova.git_tools.commit_assistant import CommitAssistant
from jenova.git_tools.diff_analyzer import DiffAnalyzer
from jenova.git_tools.hooks_manager import HooksManager
from jenova.git_tools.branch_manager import BranchManager

__all__ = [
    'GitInterface',
    'CommitAssistant',
    'DiffAnalyzer',
    'HooksManager',
    'BranchManager',
]

__version__ = '5.2.0'
