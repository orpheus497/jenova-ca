##Script function and purpose: JSON Parser Utility for The JENOVA Cognitive Architecture
##This module extracts JSON objects from LLM responses that may contain extra text

import json

##Function purpose: Extract JSON object or array from string, even if embedded in other text
def extract_json(json_str: str) -> dict | list:
    """Extracts a JSON object or array from a string, even if it's embedded in other text."""
    # Common case: JSON is wrapped in markdown backticks
    if '```json' in json_str:
        json_str = json_str.split('```json')[1].split('```')[0].strip()
    
    # Sometimes the LLM just returns the JSON without the markdown
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Find the start of the JSON object/array
        start_brace = json_str.find('{')
        start_bracket = json_str.find('[')

        if start_brace == -1 and start_bracket == -1:
            raise ValueError("No JSON object or array found in the string.")

        if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
            start = start_brace
            end_char = '}'
        else:
            start = start_bracket
            end_char = ']'

        # Find the matching end brace/bracket
        count = 0
        end = -1
        for i in range(start, len(json_str)):
            if json_str[i] == json_str[start]:
                count += 1
            elif json_str[i] == end_char:
                count -= 1
            
            if count == 0:
                end = i + 1
                break
        
        if end == -1:
            raise ValueError("Invalid JSON structure.")

        try:
            return json.loads(json_str[start:end])
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON structure: {e}")
