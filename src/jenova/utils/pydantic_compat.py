##Module purpose: Pydantic compatibility shim for ChromaDB
##ChromaDB versions may try to import BaseSettings from pydantic, but Pydantic v2 moved it to pydantic-settings
##This module ensures compatibility by making BaseSettings available from pydantic before any ChromaDB imports

##Block purpose: Apply Pydantic compatibility fix before any ChromaDB imports
try:
    import pydantic
    import pydantic_settings
    
    # Store original __getattr__ if it exists
    original_getattr = getattr(pydantic, '__getattr__', None)
    
    # Create a patched __getattr__ that handles BaseSettings
    def patched_getattr(name):
        if name == 'BaseSettings':
            return pydantic_settings.BaseSettings
        # Call original __getattr__ for other attributes
        if original_getattr:
            return original_getattr(name)
        raise AttributeError(f"module 'pydantic' has no attribute '{name}'")
    
    # Replace __getattr__ in the module
    pydantic.__getattr__ = patched_getattr
    
    # Also add BaseSettings directly to the module for direct attribute access
    pydantic.BaseSettings = pydantic_settings.BaseSettings
    
except (ImportError, AttributeError):
    pass  # If pydantic-settings isn't available, let the error occur naturally

##Block purpose: Patch ChromaDB Settings to handle None values for optional configuration fields
##ChromaDB 1.3.5+ has strict validation that requires strings, but these fields should be optional
try:
    import os
    # Set default values for ChromaDB Settings fields that may be None
    # These environment variables will be used by ChromaDB's Settings class
    chromadb_env_defaults = {
        'CHROMA_SERVER_HOST': '',
        'CHROMA_SERVER_HTTP_PORT': '',
        'CHROMA_SERVER_GRPC_PORT': '',
        'CLICKHOUSE_HOST': '',
        'CLICKHOUSE_PORT': '',
    }
    for key, default_value in chromadb_env_defaults.items():
        if key not in os.environ:
            os.environ[key] = default_value
except Exception:
    pass  # If environment patching fails, continue anyway
