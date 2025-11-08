# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""This module is responsible for disabling the telemetry of ChromaDB."""

from unittest.mock import patch


def apply_telemetry_patch():
    """Surgically disables noisy and unwanted telemetry from ChromaDB."""
    try:
        target = "posthog.capture"

        def dummy_capture(*args, **kwargs):
            pass

        patcher = patch(target, new=dummy_capture)
        patcher.start()
        print("[Telemetry Fix] ChromaDB telemetry has been permanently disabled.")
    except (ModuleNotFoundError, AttributeError):
        print("[Telemetry Fix] Telemetry module not found. No patch needed.")
