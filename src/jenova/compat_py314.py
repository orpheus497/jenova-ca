##Script function and purpose: Pre-import compatibility patches for Python 3.14
##Fix: Monkey-patch Pydantic V1 to handle ChromaDB Settings class (BH-2026-02-11T02:12:00Z)
##Date: 2026-02-11T02:12:55Z
"""
Pre-import compatibility layer for Python 3.14 + Pydantic V1.

This module MUST be imported before any code that uses ChromaDB.
It monkey-patches Pydantic V1's type inference to work with Python 3.14.
"""

import sys


def patch_pydantic_v1_for_py314():
    """Patch Pydantic V1 to handle Python 3.14 type inference issues."""
    ##Condition purpose: Only patch if Python 3.14+
    if sys.version_info < (3, 14):
        return  # Not needed for Python < 3.14
    
    try:
        ##Step purpose: Import Pydantic V1 components
        from pydantic.v1 import fields
        from pydantic.v1.fields import ModelField
        
        ##Step purpose: Store original _set_default_and_type method
        original_set_default_and_type = ModelField._set_default_and_type
        
        ##Fix: Wrap the problematic method to handle ChromaDB Settings attributes
        def patched_set_default_and_type(self):
            """Patched version that handles ChromaDB's Settings attributes with type inference issues."""
            ##Condition purpose: Check if this is a ChromaDB Settings attribute with undefined type
            if hasattr(self, 'outer_type_') and str(self.outer_type_) == 'PydanticUndefined':
                ##Fix: Manually infer types for Optional attributes with defaults
                from typing import Optional, get_origin, get_args
                import sys
                
                ##Step purpose: Try to infer from field_info
                if hasattr(self, 'field_info') and hasattr(self.field_info, 'annotation'):
                    annotation = self.field_info.annotation
                    if annotation is not None:
                        ##Condition purpose: Handle Optional[T] = None pattern
                        if get_origin(annotation) in (type(Optional[int]), type(int | None)):
                            args = get_args(annotation)
                            if args and args[0] in (int, str, bool):
                                self.type_ = args[0]
                                self.outer_type_ = annotation
                                self.shape = fields.SHAPE_SINGLETON
                                self.required = False
                                self.allow_none = True
                                return
            
            ##Step purpose: Call original method for all other attributes
            try:
                return original_set_default_and_type(self)
            except Exception as e:
                ##Error purpose: If original fails, try generic fallback for Optional types
                if "unable to infer type" in str(e):
                    import sys
                    from typing import Optional
                    ##Fix: Last resort - assume Optional[str] for string-like attributes
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
        
    except Exception as e:
        ##Error purpose: Log but don't crash if patch fails
        print(f"⚠ Warning: Failed to apply Pydantic V1 patch: {e}", file=sys.stderr)


##Action purpose: Apply patch immediately when module is imported
patch_pydantic_v1_for_py314()
