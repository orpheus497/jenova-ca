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
        ChromaDB client instance with patched create_collection method
    """
    try:
        import chromadb
        # Try PersistentClient first (ChromaDB 0.3.23+)
        if hasattr(chromadb, 'PersistentClient'):
            client = chromadb.PersistentClient(path=path)
        # Fallback to Client with Settings (older versions)
        elif hasattr(chromadb, 'Client'):
            from chromadb.config import Settings
            client = chromadb.Client(settings=Settings(persist_directory=path))
        else:
            raise AttributeError("ChromaDB client API not found")
        
        # Patch create_collection to ensure _embedding_function is set in private attrs
        original_create_collection = client.create_collection
        
        def patched_create_collection(*args, **kwargs):
            collection = original_create_collection(*args, **kwargs)
            # Ensure _embedding_function is set in __pydantic_private__
            if hasattr(collection, '__pydantic_private__'):
                private_attrs = collection.__pydantic_private__
                embedding_func = kwargs.get('embedding_function')
                if embedding_func is not None:
                    private_attrs['_embedding_function'] = embedding_func
                    # Also set embedding_function if it exists as a field
                    if 'embedding_function' not in private_attrs:
                        private_attrs['embedding_function'] = embedding_func
                elif '_embedding_function' not in private_attrs:
                    # Set to None if not provided
                    private_attrs['_embedding_function'] = None
            return collection
        
        client.create_collection = patched_create_collection
        
        # Also patch get_collection to ensure _embedding_function exists
        original_get_collection = client.get_collection
        
        def patched_get_collection(*args, **kwargs):
            collection = original_get_collection(*args, **kwargs)
            # Ensure _embedding_function exists in __pydantic_private__
            if hasattr(collection, '__pydantic_private__'):
                private_attrs = collection.__pydantic_private__
                if '_embedding_function' not in private_attrs:
                    # Try to get it from the collection's attributes
                    try:
                        if hasattr(collection, 'embedding_function'):
                            private_attrs['_embedding_function'] = collection.embedding_function
                        else:
                            private_attrs['_embedding_function'] = None
                    except Exception:
                        private_attrs['_embedding_function'] = None
            return collection
        
        client.get_collection = patched_get_collection
        
        return client
    except Exception as e:
        raise RuntimeError(f"Failed to create ChromaDB client: {e}") from e

##Block purpose: Helper function to ensure _embedding_function exists in Collection's private attributes
##This prevents AttributeError when ChromaDB tries to access _embedding_function internally
def _ensure_embedding_function(collection):
    """Ensure _embedding_function exists in collection's __pydantic_private__ dictionary."""
    if hasattr(collection, '__pydantic_private__'):
        private_attrs = collection.__pydantic_private__
        if '_embedding_function' not in private_attrs:
            try:
                if hasattr(collection, 'embedding_function'):
                    private_attrs['_embedding_function'] = collection.embedding_function
                else:
                    private_attrs['_embedding_function'] = None
            except Exception:
                private_attrs['_embedding_function'] = None

##Block purpose: Patch ChromaDB Collection class to handle missing _embedding_function gracefully
##Pydantic v2 raises AttributeError when accessing missing private attributes, but ChromaDB expects None
try:
    import chromadb
    from chromadb.api.models.Collection import Collection
    
    # Patch Collection's _validate_embedding_set to handle missing _embedding_function
    if hasattr(Collection, '_validate_embedding_set'):
        original_validate = Collection._validate_embedding_set
        
        def patched_validate_embedding_set(self, ids, embeddings, metadatas, documents):
            _ensure_embedding_function(self)
            return original_validate(self, ids, embeddings, metadatas, documents)
        
        Collection._validate_embedding_set = patched_validate_embedding_set
    
    # Patch Collection methods that might access _embedding_function internally
    # List of methods to patch - comprehensive coverage for all operations
    methods_to_patch = ['add', 'query', 'get', 'count', 'update', 'delete', 'upsert', 'modify', 'peek']
    
    for method_name in methods_to_patch:
        if hasattr(Collection, method_name):
            original_method = getattr(Collection, method_name)
            
            # Create a closure for each method to avoid variable capture issues
            # Use a lambda with default argument to capture the original method properly
            def create_patcher(original):
                def patched_method(self, *args, **kwargs):
                    _ensure_embedding_function(self)
                    return original(self, *args, **kwargs)
                return patched_method
            
            setattr(Collection, method_name, create_patcher(original_method))
except (ImportError, AttributeError):
    pass  # If ChromaDB Collection class isn't available, continue anyway

##Block purpose: Ensure ChromaDB collection has embedding function properly set
##Some ChromaDB versions/APIs may not properly set embedding function on existing collections
def get_or_create_collection_with_embedding(client, name: str, embedding_function):
    """
    Get or create a ChromaDB collection, ensuring it has the embedding function properly set.
    
    This function handles compatibility issues where existing collections might not have
    the embedding function attribute set, which causes AttributeError when trying to add documents.
    
    Args:
        client: ChromaDB client instance
        name: Name of the collection
        embedding_function: Embedding function to use
        
    Returns:
        ChromaDB collection instance with embedding function properly set
    """
    try:
        # Try to get existing collection first
        try:
            collection = client.get_collection(name=name)
            # Check if collection has embedding function attribute properly set
            # Use a safer method that doesn't trigger Pydantic's __getattr__
            has_embedding_func = False
            try:
                # Try to access via __pydantic_private__ first (Pydantic v2)
                if hasattr(collection, '__pydantic_private__'):
                    private_attrs = collection.__pydantic_private__
                    if '_embedding_function' in private_attrs and private_attrs['_embedding_function'] is not None:
                        has_embedding_func = True
                    elif 'embedding_function' in private_attrs and private_attrs['embedding_function'] is not None:
                        has_embedding_func = True
                # Fallback: try direct attribute access
                if not has_embedding_func:
                    if hasattr(collection, '_embedding_function'):
                        try:
                            ef = getattr(collection, '_embedding_function')
                            if ef is not None:
                                has_embedding_func = True
                        except (AttributeError, KeyError):
                            pass
                    elif hasattr(collection, 'embedding_function'):
                        try:
                            ef = getattr(collection, 'embedding_function')
                            if ef is not None:
                                has_embedding_func = True
                        except (AttributeError, KeyError):
                            pass
            except Exception:
                has_embedding_func = False
            
            if has_embedding_func:
                # Collection has embedding function, return it
                return collection
            else:
                # Collection exists but doesn't have embedding function properly set
                # We need to recreate it with the embedding function
                # First, backup the data
                try:
                    data = collection.get(include=["documents", "metadatas", "ids"])
                    # Delete the old collection
                    client.delete_collection(name=name)
                    # Create new collection with embedding function
                    collection = client.create_collection(name=name, embedding_function=embedding_function)
                    # Restore data if any
                    if data.get('ids'):
                        collection.add(
                            ids=data['ids'],
                            documents=data['documents'],
                            metadatas=data['metadatas']
                        )
                    return collection
                except Exception as e:
                    # If migration fails, delete and create fresh collection
                    try:
                        client.delete_collection(name=name)
                    except Exception:
                        pass
                    return client.create_collection(name=name, embedding_function=embedding_function)
        except Exception:
            # Collection doesn't exist, create it with embedding function
            return client.create_collection(name=name, embedding_function=embedding_function)
    except Exception as e:
        # Fallback to get_or_create_collection if create_collection fails
        try:
            return client.get_or_create_collection(name=name, embedding_function=embedding_function)
        except Exception:
            raise RuntimeError(f"Failed to get or create collection '{name}': {e}") from e
