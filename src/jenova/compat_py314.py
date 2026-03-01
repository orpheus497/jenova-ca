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
import warnings


##Function purpose: Monkey-patch Pydantic V1 for Python 3.14 type inference compatibility
def patch_pydantic_v1_for_py314() -> None:
    """Patch Pydantic V1 to handle Python 3.14 type inference issues."""
    ##Condition purpose: Only patch if Python 3.14+
    if sys.version_info < (3, 14):
        return  # Not needed for Python < 3.14

    try:
        ##Step purpose: Import Pydantic V1 components
        ##Refactor: Alphabetized pydantic imports (D3-2026-02-11T07:30:05Z)
        ##Fix: Suppress expected UserWarning from pydantic.v1 on Python 3.14 (D3-2026-02-15T06:44:08Z)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Core Pydantic V1", category=UserWarning)
            from pydantic.v1 import fields
            from pydantic.v1.errors import ConfigError
            from pydantic.v1.fields import ModelField, Undefined

        ##Step purpose: Store original _set_default_and_type method
        original_set_default_and_type = ModelField._set_default_and_type

        ##Fix: Wrap the problematic method to handle ChromaDB Settings attributes
        ##Refactor: Added return type annotation (D3-2026-02-11T07:30:05Z)
        def patched_set_default_and_type(self) -> None:
            """Patched version that handles ChromaDB's Settings attributes with type inference issues."""
            ##Condition purpose: Check if this is a ChromaDB Settings attribute with undefined type
            ##Refactor: Use sentinel identity check instead of string comparison (D3-2026-02-11T08:22:24Z)
            if hasattr(self, "outer_type_") and self.outer_type_ is Undefined:
                ##Fix: Manually infer types for Optional attributes with defaults
                import types
                from typing import Union, get_args, get_origin

                ##Step purpose: Try to infer from field_info
                if hasattr(self, "field_info") and hasattr(self.field_info, "annotation"):
                    annotation = self.field_info.annotation
                    if annotation is not None:
                        ##Condition purpose: Handle Optional[T] = None pattern
                        origin = get_origin(annotation)
                        ##Fix: Correctly detect Union/Optional types (D3-2026-02-11T07:03:00Z)
                        if origin is Union or (
                            hasattr(types, "UnionType") and origin is types.UnionType
                        ):
                            args = get_args(annotation)
                            non_none_args = [
                                arg for arg in args
                                if isinstance(arg, type) and arg is not type(None)
                            ]
                            if non_none_args:
                                self.type_ = non_none_args[0]
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
                    ##Refactor: Log at debug level (expected for ChromaDB on Python 3.14) (D3-2026-02-11T08:39:11Z)
                    logger = logging.getLogger(__name__)
                    logger.debug(
                        "Applying Pydantic type fallback (expected for ChromaDB on Python 3.14): "
                        "field=%s, fallback=Optional[str]",
                        getattr(self, "name", "<unknown>"),
                    )
                    ##Fix: Last resort - assume Optional[str] for string-like attributes
                    if self.default is None:
                        self.type_ = str
                        self.outer_type_ = str | None
                        self.shape = fields.SHAPE_SINGLETON
                        self.required = False
                        self.allow_none = True
                        return
                raise

        ##Action purpose: Replace the method with our patched version
        ModelField._set_default_and_type = patched_set_default_and_type

    except ImportError as e:
        ##Error purpose: Log warning if Pydantic V1 not available (expected in some environments)
        ##Refactor: Separate ImportError (warning) from AttributeError (error) (D3-2026-02-11T08:22:24Z)
        logging.getLogger(__name__).warning(
            "Pydantic V1 not available, skipping Python 3.14 compatibility patch: %s",
            str(e),
            exc_info=True,
        )
    except AttributeError as e:
        ##Error purpose: Log error and re-raise if ModelField API changed (breaking change)
        ##Refactor: AttributeError now fails fast to surface API changes (D3-2026-02-11T08:22:24Z)
        logger = logging.getLogger(__name__)
        logger.error(
            "CRITICAL: Pydantic V1 ModelField._set_default_and_type is missing or changed. "
            "This indicates a breaking internal API change in Pydantic. "
            "Error: %s",
            str(e),
            exc_info=True,
        )
        raise RuntimeError(
            f"Failed to patch Pydantic V1: ModelField._set_default_and_type missing or changed: {e}"
        ) from e


##Action purpose: Apply patch immediately when module is imported
patch_pydantic_v1_for_py314()
