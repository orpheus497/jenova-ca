##Script function and purpose: Telemetry Fix for The JENOVA Cognitive Architecture
##This module disables ChromaDB telemetry to reduce noise in logs

from unittest.mock import patch

##Function purpose: Surgically disable ChromaDB telemetry using mock patching
def apply_telemetry_patch():
    """Surgically disables noisy and unwanted telemetry from ChromaDB."""
    try:
        target = 'posthog.capture'
        def dummy_capture(*args, **kwargs):
            pass
        patcher = patch(target, new=dummy_capture)
        patcher.start()
        print("[Telemetry Fix] ChromaDB telemetry has been permanently disabled.")
    except (ModuleNotFoundError, AttributeError):
        print("[Telemetry Fix] Telemetry module not found. No patch needed.")