##Script function and purpose: Pre-import compatibility patches for Python 3.14
##Fix: Monkey-patch Pydantic V1 to handle ChromaDB Settings class (BH-2026-02-11T02:12:00Z)
##Date: 2026-02-11T02:12:55Z
"""
Pre-import compatibility layer for Python 3.14 + Pydantic V1.

This module MUST be imported before any code that uses ChromaDB.
It monkey-patches Pydantic V1's type inference to work with Python 3.14.
"""

##Refactor: Alphabetized stdlib imports per PEP 8 (D3-2026-02-11T07:30:05Z)
import logging
import sys


##Function purpose: Monkey-patch Pydantic V1 for Python 3.14 type inference compatibility
def patch_pydantic_v1_for_py314() -> None:
    """Patch Pydantic V1 to handle Python 3.14 type inference issues."""
    ##Condition purpose: Only patch if Python 3.14+
    if sys.version_info < (3, 14):
        return  # Not needed for Python < 3.14

    try:
        ##Step purpose: Import Pydantic V1 components
        ##Refactor: Alphabetized pydantic imports (D3-2026-02-11T07:30:05Z)
        from pydantic.v1 import fields
        from pydantic.v1.errors import ConfigError
        from pydantic.v1.fields import ModelField

        ##Step purpose: Store original _set_default_and_type method
        original_set_default_and_type = ModelField._set_default_and_type

        ##Fix: Wrap the problematic method to handle ChromaDB Settings attributes
        ##Refactor: Added return type annotation (D3-2026-02-11T07:30:05Z)
        def patched_set_default_and_type(self) -> None:
            """Patched version that handles ChromaDB's Settings attributes with type inference issues."""
            ##Condition purpose: Check if this is a ChromaDB Settings attribute with undefined type
            if hasattr(self, 'outer_type_') and str(self.outer_type_) == 'PydanticUndefined':
                ##Fix: Manually infer types for Optional attributes with defaults
                import types
                from typing import Optional, Union, get_args, get_origin

                ##Step purpose: Try to infer from field_info
                if hasattr(self, 'field_info') and hasattr(self.field_info, 'annotation'):
                    annotation = self.field_info.annotation
                    if annotation is not None:
                        ##Condition purpose: Handle Optional[T] = None pattern
                        origin = get_origin(annotation)
                        ##Fix: Correctly detect Union/Optional types (D3-2026-02-11T07:03:00Z)
                        if origin is Union or (hasattr(types, 'UnionType') and origin is types.UnionType):
                            args = get_args(annotation)
                            if args and isinstance(args[0], type):
                                self.type_ = args[0]
                                self.outer_type_ = annotation
                                self.shape = fields.SHAPE_SINGLETON
                                self.required = False
                                self.allow_none = True
                                return

            ##Step purpose: Call original method for all other attributes
            try:
                return original_set_default_and_type(self)
            except ConfigError as e:
                ##Error purpose: If original fails, try generic fallback for Optional types
                ##Refactor: Narrowed to ConfigError, removed redundant imports (D3-2026-02-11T07:03:00Z)
                if "unable to infer type" in str(e):
                    ##Fix: Last resort - assume Optional[str] for string-like attributes
                    from typing import Optional
                    if self.default is None:
                        self.type_ = str
                        self.outer_type_ = Optional[str]
                        self.shape = fields.SHAPE_SINGLETON
                        self.required = False
                        self.allow_none = True
                        return
                raise

        ##Action purpose: Replace the method with our patched version
        ModelField._set_default_and_type = patched_set_default_and_type

    except (ImportError, AttributeError):
        ##Error purpose: Log but don't crash if patch fails
        ##Refactor: Use stdlib logger with narrow exceptions (D3-2026-02-11T07:03:00Z)
        logging.getLogger(__name__).warning(
            "Failed to apply Pydantic V1 patch for Python 3.14 compatibility",
            exc_info=True
        )


##Action purpose: Apply patch immediately when module is imported
patch_pydantic_v1_for_py314()
