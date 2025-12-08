##Module purpose: Pydantic compatibility shim for ChromaDB
##ChromaDB versions may try to import BaseSettings from pydantic, but Pydantic v2 moved it to pydantic-settings
##This module ensures compatibility by making BaseSettings available from pydantic before any ChromaDB imports

##Block purpose: Apply Pydantic compatibility fix before any ChromaDB imports
try:
    import pydantic
    import pydantic_settings
    
    # Add BaseSettings directly to the module for direct attribute access
    # This handles both direct imports and __getattr__ calls
    pydantic.BaseSettings = pydantic_settings.BaseSettings
    
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
    
    # Also patch _getattr_migration if it exists (Pydantic v2)
    if hasattr(pydantic, '_getattr_migration'):
        original_getattr_migration = pydantic._getattr_migration
        
        def patched_getattr_migration(attr_name):
            if attr_name == 'BaseSettings':
                return pydantic_settings.BaseSettings
            return original_getattr_migration(attr_name)
        
        pydantic._getattr_migration = patched_getattr_migration
    
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

##Block purpose: Create ChromaDB client compatibility wrapper
##Different ChromaDB versions use different APIs for creating persistent clients
def create_chromadb_client(path: str):
    """
    Create a ChromaDB client with compatibility for different API versions.
    
    Args:
        path: Directory path where the database will be stored
        
    Returns:
        ChromaDB client instance
    """
    try:
        import chromadb
        # Try PersistentClient first (ChromaDB 0.3.23+)
        if hasattr(chromadb, 'PersistentClient'):
            return chromadb.PersistentClient(path=path)
        # Fallback to Client with Settings (older versions)
        elif hasattr(chromadb, 'Client'):
            from chromadb.config import Settings
            return chromadb.Client(settings=Settings(persist_directory=path))
        else:
            raise AttributeError("ChromaDB client API not found")
    except Exception as e:
        raise RuntimeError(f"Failed to create ChromaDB client: {e}") from e
