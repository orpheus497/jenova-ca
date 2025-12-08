from unittest.mock import patch

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