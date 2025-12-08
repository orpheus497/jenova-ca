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
        
        # Patch create_collection to ensure all private attributes are set
        original_create_collection = client.create_collection
        
        def make_create_collection_patcher(client_instance):
            def patched_create_collection(*args, **kwargs):
                collection = original_create_collection(*args, **kwargs)
                # Ensure all private attributes are set in __pydantic_private__
                if hasattr(collection, '__pydantic_private__'):
                    private_attrs = collection.__pydantic_private__
                    
                    # Set _embedding_function
                    embedding_func = kwargs.get('embedding_function')
                    if embedding_func is not None:
                        private_attrs['_embedding_function'] = embedding_func
                        if 'embedding_function' not in private_attrs:
                            private_attrs['embedding_function'] = embedding_func
                    elif '_embedding_function' not in private_attrs:
                        private_attrs['_embedding_function'] = None
                    
                    # Set _client (the client that created this collection)
                    if '_client' not in private_attrs:
                        private_attrs['_client'] = client_instance
                    
                    # Set _name
                    name = kwargs.get('name') or (args[0] if args else None)
                    if name and '_name' not in private_attrs:
                        private_attrs['_name'] = name
                    
                    # Ensure all other attributes exist
                    _ensure_collection_attributes(collection)
                return collection
            return patched_create_collection
        
        client.create_collection = make_create_collection_patcher(client)
        
        # Also patch get_collection to ensure all private attributes exist
        original_get_collection = client.get_collection
        
        def make_get_collection_patcher(client_instance):
            def patched_get_collection(*args, **kwargs):
                collection = original_get_collection(*args, **kwargs)
                # Ensure all private attributes exist in __pydantic_private__
                if hasattr(collection, '__pydantic_private__'):
                    private_attrs = collection.__pydantic_private__
                    
                    # Set _client (the client that retrieved this collection)
                    if '_client' not in private_attrs:
                        private_attrs['_client'] = client_instance
                    
                    # Ensure all other attributes exist
                    _ensure_collection_attributes(collection)
                return collection
            return patched_get_collection
        
        client.get_collection = make_get_collection_patcher(client)
        
        return client
    except Exception as e:
        raise RuntimeError(f"Failed to create ChromaDB client: {e}") from e

##Block purpose: Helper function to ensure all Collection private attributes exist
##This prevents AttributeError when ChromaDB tries to access private attributes internally
##Pydantic v2 requires all private attributes to be explicitly set in __pydantic_private__
def _ensure_collection_attributes(collection):
    """
    Ensure all private attributes that ChromaDB Collection might access exist in __pydantic_private__.
    
    ChromaDB Collection accesses these private attributes internally:
    - _embedding_function: The embedding function for the collection
    - _client: The ChromaDB client instance
    - _name: The collection name
    - _id: The collection ID
    
    This function ensures all of these exist before any operation.
    """
    if hasattr(collection, '__pydantic_private__'):
        private_attrs = collection.__pydantic_private__
        
        # Ensure _embedding_function exists
        if '_embedding_function' not in private_attrs:
            try:
                if hasattr(collection, 'embedding_function'):
                    private_attrs['_embedding_function'] = collection.embedding_function
                else:
                    private_attrs['_embedding_function'] = None
            except Exception:
                private_attrs['_embedding_function'] = None
        
        # Ensure _client exists (critical - used by add, query, get, etc.)
        if '_client' not in private_attrs:
            try:
                # Try to get it from public attributes or model fields
                if hasattr(collection, 'client'):
                    private_attrs['_client'] = collection.client
                elif hasattr(collection, '_client'):
                    # Try direct access (might work if it's a model field)
                    try:
                        private_attrs['_client'] = object.__getattribute__(collection, '_client')
                    except AttributeError:
                        # If it doesn't exist, we need to get it from the collection's creation context
                        # Collections are created with a client, so we should be able to find it
                        private_attrs['_client'] = None
                else:
                    private_attrs['_client'] = None
            except Exception:
                private_attrs['_client'] = None
        
        # Ensure _name exists
        if '_name' not in private_attrs:
            try:
                if hasattr(collection, 'name'):
                    private_attrs['_name'] = collection.name
                elif hasattr(collection, '_name'):
                    try:
                        private_attrs['_name'] = object.__getattribute__(collection, '_name')
                    except AttributeError:
                        private_attrs['_name'] = None
                else:
                    private_attrs['_name'] = None
            except Exception:
                private_attrs['_name'] = None
        
        # Ensure _id exists
        if '_id' not in private_attrs:
            try:
                if hasattr(collection, 'id'):
                    private_attrs['_id'] = collection.id
                elif hasattr(collection, '_id'):
                    try:
                        private_attrs['_id'] = object.__getattribute__(collection, '_id')
                    except AttributeError:
                        private_attrs['_id'] = None
                else:
                    private_attrs['_id'] = None
            except Exception:
                private_attrs['_id'] = None

##Block purpose: Patch ChromaDB Collection class to handle missing _embedding_function gracefully
##Pydantic v2 raises AttributeError when accessing missing private attributes, but ChromaDB expects None
try:
    import chromadb
    from chromadb.api.models.Collection import Collection
    
    # Patch Collection's _validate_embedding_set to handle missing _embedding_function
    if hasattr(Collection, '_validate_embedding_set'):
        original_validate = Collection._validate_embedding_set
        
        def patched_validate_embedding_set(self, ids, embeddings, metadatas, documents):
            _ensure_collection_attributes(self)
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
                    _ensure_collection_attributes(self)
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
