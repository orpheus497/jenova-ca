##Module purpose: Pydantic compatibility shim for ChromaDB
##ChromaDB versions may try to import BaseSettings from pydantic, but Pydantic v2 moved it to pydantic-settings
##This module ensures compatibility by making BaseSettings available from pydantic before any ChromaDB imports

##Block purpose: Patch chromadb config.py source before import to add missing type annotations
##This fixes the non-annotated attribute error in chromadb's Settings class
##Note: This must run BEFORE any chromadb imports
try:
    import sys
    import importlib.util
    import importlib.machinery
    import os
    import site
    
    class ChromaDBConfigFinder:
        """Custom MetaPathFinder that patches chromadb.config before loading it"""
        def find_spec(self, name, path, target=None):
            if name == 'chromadb.config' and 'chromadb.config' not in sys.modules:
                # Find the original spec using the default finder
                # Skip custom finders to avoid recursion by checking class name
                for finder in sys.meta_path:
                    # Skip self and other custom finders
                    if finder is self:
                        continue
                    # Skip other custom finders by checking class name
                    finder_class_name = finder.__class__.__name__ if hasattr(finder, '__class__') else ''
                    if finder_class_name in ('ChromaDBConfigFinder', 'ChromaDBMainPatcher'):
                        continue
                    if not hasattr(finder, 'find_spec'):
                        continue
                    spec = finder.find_spec(name, path, target)
                    if spec and spec.loader:
                        original_loader = spec.loader
                        
                        # Create a patched loader that calls model_rebuild() after Settings is defined
                        class PatchedConfigLoader:
                            def __init__(self, original_loader):
                                self.original_loader = original_loader
                            
                            def create_module(self, spec):
                                return None
                            
                            def exec_module(self, module):
                                # Load the module normally
                                self.original_loader.exec_module(module)
                                
                                # After module is loaded, patch Settings to call model_rebuild()
                                # Ensure Optional is imported first (needed for forward refs)
                                import typing
                                # Ensure Optional is available
                                if not hasattr(typing, 'Optional'):
                                    from typing import Optional
                                
                                if hasattr(module, 'Settings'):
                                    Settings = module.Settings
                                    if hasattr(Settings, 'model_rebuild'):
                                        # Call model_rebuild() immediately after Settings is defined
                                        # Note: We don't patch __new__ as it interferes with Pydantic's instantiation
                                        # The fix_chromadb_compat.py script handles model_rebuild() in chromadb's source
                                        try:
                                            import typing
                                            if not hasattr(typing, 'Optional'):
                                                from typing import Optional
                                            Settings.model_rebuild()
                                        except Exception:
                                            pass
                        
                        spec.loader = PatchedConfigLoader(original_loader)
                        
                        # Also try source patching if origin exists
                        if spec.origin and os.path.exists(spec.origin):
                            try:
                                with open(spec.origin, 'r', encoding='utf-8') as f:
                                    source = f.read()
                                
                                # Patch chroma_coordinator_host to have a type annotation
                                if 'chroma_coordinator_host = ' in source and 'chroma_coordinator_host:' not in source:
                                    import re
                                    source = re.sub(
                                        r'(\s+)chroma_coordinator_host\s*=\s*"localhost"',
                                        r'\1chroma_coordinator_host: str = "localhost"',
                                        source
                                    )
                                    
                                    # Create a new loader that uses the patched source
                                    class PatchedSourceLoader(importlib.machinery.SourceFileLoader):
                                        def get_data(self, path):
                                            if path == spec.origin:
                                                return source.encode('utf-8')
                                            return super().get_data(path)
                                        
                                        def exec_module(self, module):
                                            super().exec_module(module)
                                            # After loading, call model_rebuild()
                                            # Ensure Optional is imported first
                                            import typing
                                            if hasattr(module, 'Settings'):
                                                Settings = module.Settings
                                                if hasattr(Settings, 'model_rebuild'):
                                                    try:
                                                        # Force import of Optional if not already imported
                                                        if not hasattr(typing, 'Optional'):
                                                            from typing import Optional
                                                        Settings.model_rebuild()
                                                    except Exception:
                                                        pass
                                    
                                    spec.loader = PatchedSourceLoader(spec.name, spec.origin)
                            except Exception:
                                pass
                        
                        return spec
                    elif spec:
                        return spec
            return None
    
    # Install the custom finder at the beginning of meta_path
    if 'chromadb.config' not in sys.modules:
        # Insert at the beginning so it's checked first
        finder = ChromaDBConfigFinder()
        sys.meta_path.insert(0, finder)
except Exception:
    pass  # If patching fails, continue with other methods

##Block purpose: Patch chromadb main module to delay Settings instantiation until after patching
##This fixes Pydantic 2.12 forward reference issues by ensuring Settings.model_rebuild() is called
try:
    import sys
    import importlib.util
    import importlib.machinery
    
    class ChromaDBMainPatcher:
        """Patches chromadb.__init__ to delay Settings instantiation"""
        def find_spec(self, name, path, target=None):
            if name == 'chromadb' and 'chromadb' not in sys.modules:
                # Find the original spec
                # Skip custom finders to avoid recursion by checking class name
                for finder in sys.meta_path:
                    # Skip self and other custom finders
                    if finder is self:
                        continue
                    # Skip other custom finders by checking class name
                    finder_class_name = finder.__class__.__name__ if hasattr(finder, '__class__') else ''
                    if finder_class_name in ('ChromaDBConfigFinder', 'ChromaDBMainPatcher'):
                        continue
                    if not hasattr(finder, 'find_spec'):
                        continue
                    spec = finder.find_spec(name, path, target)
                    if spec and spec.loader:
                        original_loader = spec.loader
                        
                        class PatchedChromaDBLoader:
                            def __init__(self, original_loader):
                                self.original_loader = original_loader
                            
                            def create_module(self, spec):
                                return None
                            
                            def exec_module(self, module):
                                # First, ensure chromadb.config is loaded and patched
                                if 'chromadb.config' not in sys.modules:
                                    import chromadb.config
                                
                                # Patch Settings.model_rebuild() if needed
                                # Ensure Optional is imported first
                                import typing
                                if not hasattr(typing, 'Optional'):
                                    from typing import Optional
                                
                                try:
                                    from chromadb.config import Settings
                                    if hasattr(Settings, 'model_rebuild'):
                                        try:
                                            Settings.model_rebuild()
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                                
                                # Patch chromadb's __init__.py to wrap Settings instantiation
                                # This ensures model_rebuild() is called before Settings() is instantiated
                                try:
                                    import os
                                    chromadb_init_path = os.path.join(os.path.dirname(module.__file__), '__init__.py')
                                    if os.path.exists(chromadb_init_path):
                                        with open(chromadb_init_path, 'r') as f:
                                            source = f.read()
                                        
                                        # Patch __settings = chromadb.config.Settings() to delay instantiation
                                        if '__settings = chromadb.config.Settings()' in source:
                                            patched_line = '''try:
    __settings = chromadb.config.Settings()
except Exception:
    # Ensure model_rebuild() is called before retrying
    import typing
    if not hasattr(typing, 'Optional'):
        from typing import Optional
    chromadb.config.Settings.model_rebuild()
    __settings = chromadb.config.Settings()'''
                                            source = source.replace('__settings = chromadb.config.Settings()', patched_line)
                                            
                                            # Create a patched loader that uses the modified source
                                            class PatchedInitLoader(importlib.machinery.SourceFileLoader):
                                                def get_data(self, path):
                                                    if path == chromadb_init_path:
                                                        return source.encode('utf-8')
                                                    return super().get_data(path)
                                            
                                            # Replace the loader
                                            import importlib.machinery
                                            self.original_loader = PatchedInitLoader(module.__name__, chromadb_init_path)
                                except Exception:
                                    pass
                                
                                # Now load chromadb normally
                                self.original_loader.exec_module(module)
                                
                                # After module is loaded, ensure Settings is rebuilt
                                try:
                                    from chromadb.config import Settings
                                    if hasattr(Settings, 'model_rebuild'):
                                        try:
                                            if not hasattr(typing, 'Optional'):
                                                from typing import Optional
                                            Settings.model_rebuild()
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                        
                        spec.loader = PatchedChromaDBLoader(original_loader)
                        return spec
            return None
    
    # Install patcher if chromadb hasn't been imported
    if 'chromadb' not in sys.modules:
        patcher = ChromaDBMainPatcher()
        sys.meta_path.insert(0, patcher)
except Exception:
    pass  # If patching fails, continue with other methods

##Block purpose: Apply Pydantic compatibility fix before any ChromaDB imports
try:
    import pydantic
    import pydantic_settings
    import pydantic.errors
    
    # Add BaseSettings directly to the module for direct attribute access
    # This handles both direct imports and __getattr__ calls
    pydantic.BaseSettings = pydantic_settings.BaseSettings
    
    # Patch BaseSettings.__init__ to automatically call model_rebuild() if validation fails
    # This fixes the "Settings is not fully defined" error in chromadb
    original_basesettings_init = pydantic_settings.BaseSettings.__init__
    
    def patched_basesettings_init(self, *args, **kwargs):
        """Patched __init__ that calls model_rebuild() if validation fails"""
        try:
            return original_basesettings_init(self, *args, **kwargs)
        except pydantic.errors.PydanticUserError as e:
            # Check if this is the "not fully defined" error
            error_msg = str(e).lower()
            if 'not fully defined' in error_msg or 'define `optional`' in error_msg:
                # Try to call model_rebuild() and retry
                if hasattr(self.__class__, 'model_rebuild'):
                    try:
                        import typing
                        if not hasattr(typing, 'Optional'):
                            from typing import Optional
                        self.__class__.model_rebuild()
                        # Retry initialization
                        return original_basesettings_init(self, *args, **kwargs)
                    except Exception:
                        pass
            # Re-raise if we can't fix it
            raise
    
    # Only patch if not already patched
    if not hasattr(pydantic_settings.BaseSettings, '_jenova_init_patched'):
        pydantic_settings.BaseSettings.__init__ = patched_basesettings_init
        pydantic_settings.BaseSettings._jenova_init_patched = True
    
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

##Block purpose: Patch Pydantic's model construction to allow non-annotated attributes in BaseSettings
##This fixes compatibility issues with chromadb's Settings class which has non-annotated attributes
try:
    import pydantic
    import pydantic_settings
    import pydantic.errors
    import inspect
    import typing
    
    # Patch pydantic's inspect_namespace to automatically annotate non-annotated attributes
    # in classes that inherit from BaseSettings (for chromadb compatibility)
    if hasattr(pydantic._internal._model_construction, 'inspect_namespace'):
        from pydantic._internal._model_construction import inspect_namespace as original_inspect_namespace
        
        def patched_inspect_namespace(namespace, raw_annotations, config_wrapper, class_vars, base_field_names):
            """
            Patched version of inspect_namespace that automatically adds type annotations
            for non-annotated attributes. This fixes compatibility with chromadb's Settings class.
            """
            # Check if we're likely dealing with a BaseSettings class
            # by checking the calling frame's module name
            is_basesettings_context = False
            try:
                import sys
                # Check multiple frames to find chromadb context
                for i in range(1, 5):  # Check up to 4 frames up
                    try:
                        frame = sys._getframe(i)
                        if frame:
                            module_name = frame.f_globals.get('__name__', '')
                            file_path = frame.f_code.co_filename
                            # Check if we're in chromadb.config or similar BaseSettings context
                            if 'chromadb' in module_name or 'chromadb' in file_path:
                                is_basesettings_context = True
                                break
                    except (ValueError, AttributeError):
                        break
            except (AttributeError, ValueError):
                pass
            
            # Also check __bases__ if available
            if not is_basesettings_context and '__bases__' in namespace:
                bases = namespace['__bases__']
                for base in bases:
                    if inspect.isclass(base):
                        mro = inspect.getmro(base)
                        if any('BaseSettings' in str(b.__name__) for b in mro if hasattr(b, '__name__')):
                            is_basesettings_context = True
                            break
            
            # Check __module__ in namespace
            if not is_basesettings_context and '__module__' in namespace:
                module_name = namespace.get('__module__', '')
                if 'chromadb' in str(module_name):
                    is_basesettings_context = True
            
            # Auto-annotate non-annotated simple assignments
            # This helps with BaseSettings classes that have non-annotated attributes
            annotations = dict(raw_annotations) if raw_annotations else {}
            needs_update = False
            
            # Always auto-annotate non-annotated simple assignments in BaseSettings context
            # or when we detect chromadb module
            if is_basesettings_context:
                for key, value in namespace.items():
                    # Skip special attributes, already annotated ones, and class vars
                    if key.startswith('__') or key in annotations or key in class_vars:
                        continue
                    
                    # Skip callables (methods, functions)
                    if callable(value):
                        continue
                    
                    # Auto-annotate based on value type using proper type objects
                    if isinstance(value, str):
                        annotations[key] = typing.Optional[str]
                        needs_update = True
                    elif isinstance(value, int):
                        annotations[key] = typing.Optional[int]
                        needs_update = True
                    elif isinstance(value, bool):
                        annotations[key] = typing.Optional[bool]
                        needs_update = True
                    elif isinstance(value, float):
                        annotations[key] = typing.Optional[float]
                        needs_update = True
                    elif isinstance(value, (list, tuple)):
                        annotations[key] = typing.Optional[list]
                        needs_update = True
                    elif isinstance(value, dict):
                        annotations[key] = typing.Optional[dict]
                        needs_update = True
                    else:
                        # Default to Any for unknown types
                        annotations[key] = typing.Any
                        needs_update = True
            
            # Update raw_annotations if we added any
            if needs_update:
                if raw_annotations is not None:
                    raw_annotations.update(annotations)
                else:
                    raw_annotations = annotations
                
                # Also update __annotations__ in namespace if it exists
                if '__annotations__' in namespace:
                    namespace['__annotations__'].update(annotations)
                else:
                    namespace['__annotations__'] = annotations.copy()
            
            # Call original function with updated annotations
            # Wrap in try-except to handle PydanticUserError for chromadb compatibility
            try:
                return original_inspect_namespace(namespace, raw_annotations, config_wrapper, class_vars, base_field_names)
            except pydantic.errors.PydanticUserError as e:
                # If the error is about non-annotated attributes, try to auto-annotate them
                error_msg = str(e).lower()
                if 'non-annotated attribute' in error_msg or 'missing annotation' in error_msg:
                    # Extract the attribute name from the error message
                    import re
                    attr_match = re.search(r"`?(\w+)\s*=", error_msg)
                    
                    # Add annotations for ALL non-annotated attributes in the namespace
                    # This is a workaround for chromadb's Settings class
                    for key, value in list(namespace.items()):
                        if key.startswith('__') or key in annotations or key in class_vars or callable(value):
                            continue
                        
                        # Auto-annotate based on value type
                        if isinstance(value, str):
                            annotations[key] = typing.Optional[str]
                        elif isinstance(value, int):
                            annotations[key] = typing.Optional[int]
                        elif isinstance(value, bool):
                            annotations[key] = typing.Optional[bool]
                        elif isinstance(value, float):
                            annotations[key] = typing.Optional[float]
                        elif isinstance(value, (list, tuple)):
                            annotations[key] = typing.Optional[list]
                        elif isinstance(value, dict):
                            annotations[key] = typing.Optional[dict]
                        else:
                            annotations[key] = typing.Any
                    
                    # Update raw_annotations
                    if raw_annotations is not None:
                        raw_annotations.update(annotations)
                    else:
                        raw_annotations = annotations
                    
                    # Update __annotations__ in namespace
                    if '__annotations__' in namespace:
                        namespace['__annotations__'].update(annotations)
                    else:
                        namespace['__annotations__'] = annotations.copy()
                    
                    # Retry with updated annotations (up to 3 times to handle multiple attributes)
                    try:
                        return original_inspect_namespace(namespace, raw_annotations, config_wrapper, class_vars, base_field_names)
                    except pydantic.errors.PydanticUserError as e2:
                        # If still failing after annotation, it might be a different issue
                        # Check if it's still about non-annotated attributes
                        if 'non-annotated attribute' in str(e2).lower():
                            # Try one more time with even more aggressive annotation
                            for key, value in list(namespace.items()):
                                if not key.startswith('__') and key not in annotations and key not in class_vars and not callable(value):
                                    annotations[key] = typing.Any
                            if raw_annotations is not None:
                                raw_annotations.update(annotations)
                            if '__annotations__' in namespace:
                                namespace['__annotations__'].update(annotations)
                            return original_inspect_namespace(namespace, raw_annotations, config_wrapper, class_vars, base_field_names)
                        raise
                else:
                    # Re-raise if not about annotations
                    raise
        
        # Replace the function
        pydantic._internal._model_construction.inspect_namespace = patched_inspect_namespace
except (ImportError, AttributeError):
    pass  # If patching fails, let the error occur naturally

##Block purpose: Patch ChromaDB Settings to handle None values for optional configuration fields
##ChromaDB 1.3.5+ has strict validation that requires strings, but these fields should be optional
##Note: We don't set empty strings for integer fields as pydantic v2 will fail to parse them
try:
    import os
    # Remove integer fields if they're set to empty strings (pydantic v2 can't parse them)
    # This must happen BEFORE chromadb imports to prevent validation errors
    for int_key in ['CHROMA_SERVER_HTTP_PORT', 'CHROMA_SERVER_GRPC_PORT', 'CLICKHOUSE_PORT']:
        if int_key in os.environ and os.environ[int_key] == '':
            del os.environ[int_key]
    # Only set string defaults if not already set, and only for string fields
    chromadb_env_defaults = {
        'CHROMA_SERVER_HOST': '',  # String field, empty is OK
        'CLICKHOUSE_HOST': '',  # String field, empty is OK
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


##Block purpose: Patch Pydantic Settings to auto-rebuild on class definition
##This fixes forward reference issues in chromadb's Settings class
try:
    import pydantic_settings
    from typing import TYPE_CHECKING
    
    if TYPE_CHECKING:
        pass
    
    # Patch BaseSettings metaclass to auto-rebuild after class definition
    original_basesettings_init_subclass = None
    if hasattr(pydantic_settings.BaseSettings, '__init_subclass__'):
        original_basesettings_init_subclass = pydantic_settings.BaseSettings.__init_subclass__
    
    @classmethod
    def patched_init_subclass(cls, **kwargs):
        """Patched __init_subclass__ that calls model_rebuild() after class definition"""
        if original_basesettings_init_subclass:
            original_basesettings_init_subclass(**kwargs)
        # Call model_rebuild() to resolve forward references
        # Ensure Optional is imported first
        import typing
        if not hasattr(typing, 'Optional'):
            from typing import Optional
        if hasattr(cls, 'model_rebuild'):
            try:
                # Call model_rebuild() immediately to resolve forward references
                cls.model_rebuild()
                # Also try to rebuild with all forward refs resolved
                import sys
                # Give a moment for all imports to complete
                cls.model_rebuild()
            except Exception:
                # If model_rebuild fails, try again after a brief delay
                try:
                    import time
                    time.sleep(0.01)
                    cls.model_rebuild()
                except Exception:
                    pass
    
    # Only patch if not already patched
    if not hasattr(pydantic_settings.BaseSettings, '_jenova_patched'):
        pydantic_settings.BaseSettings.__init_subclass__ = patched_init_subclass
        pydantic_settings.BaseSettings._jenova_patched = True
except Exception:
    pass  # If patching fails, continue with other methods

##Block purpose: Patch ChromaDB Collection class to handle missing _embedding_function gracefully
##Pydantic v2 raises AttributeError when accessing missing private attributes, but ChromaDB expects None
try:
    import chromadb
    from chromadb.api.models.Collection import Collection
    
    # After chromadb is imported, ensure Settings is rebuilt
    try:
        from chromadb.config import Settings
        if hasattr(Settings, 'model_rebuild'):
            try:
                Settings.model_rebuild()
            except Exception:
                pass
    except Exception:
        pass
    
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
