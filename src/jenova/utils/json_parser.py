##Script function and purpose: JSON Parser Utility for The JENOVA Cognitive Architecture
##This module extracts JSON objects from LLM responses that may contain extra text

import json
from typing import Any, Union

##Sentinel purpose: Distinguish between "no default provided" and "explicit None"
_UNSET = object()

##Function purpose: Extract JSON object or array from string, even if embedded in other text
def extract_json(json_str: str, default: Any = _UNSET) -> Union[dict, list, Any]:
    """
    Extracts a JSON object or array from a string, even if it's embedded in other text.
    
    Args:
        json_str: String containing JSON, possibly with surrounding text
        default: Value to return if extraction fails (if _UNSET, raises exception)
        
    Returns:
        Parsed JSON data (dict or list), or default value on failure
        
    Raises:
        ValueError: If no valid JSON found and default is _UNSET
    """
    ##Block purpose: Handle markdown-wrapped JSON
    if '```json' in json_str:
        json_str = json_str.split('```json')[1].split('```')[0].strip()
    
    ##Block purpose: Try direct JSON parsing first
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    ##Block purpose: Find the start of JSON object/array
    start_brace = json_str.find('{')
    start_bracket = json_str.find('[')

    if start_brace == -1 and start_bracket == -1:
        if default is not _UNSET:
            return default
        raise ValueError("No JSON object or array found in the string.")

    if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
        start = start_brace
        start_char = '{'
        end_char = '}'
    else:
        start = start_bracket
        start_char = '['
        end_char = ']'

    ##Block purpose: Find the matching end brace/bracket with string context awareness
    count = 0
    end = -1
    in_string = False
    escape_next = False
    
    for i in range(start, len(json_str)):
        char = json_str[i]
        
        ##Block purpose: Handle escape sequences inside strings
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\' and in_string:
            escape_next = True
            continue
        
        ##Block purpose: Track string context to ignore braces inside quoted strings
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        ##Block purpose: Only count structural chars when not inside a string
        if not in_string:
            if char == start_char:
                count += 1
            elif char == end_char:
                count -= 1
            
            if count == 0:
                end = i + 1
                break
    
    if end == -1:
        if default is not _UNSET:
            return default
        raise ValueError("Invalid JSON structure.")

    try:
        return json.loads(json_str[start:end])
    except json.JSONDecodeError as e:
        if default is not _UNSET:
            return default
        raise ValueError(f"Invalid JSON structure: {e}")
