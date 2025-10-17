"""
Utility functions for sanitizing data before passing to external systems.
"""


def sanitize_metadata(metadata: dict) -> dict:
    """
    Remove None values from a metadata dictionary.
    
    ChromaDB does not accept None as a value for metadata fields.
    This function creates a new dictionary with all key-value pairs
    where the value is None removed.
    
    Args:
        metadata: Dictionary that may contain None values
        
    Returns:
        New dictionary with None values removed
    """
    return {k: v for k, v in metadata.items() if v is not None}
