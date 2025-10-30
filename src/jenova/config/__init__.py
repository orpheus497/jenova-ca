# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Configuration module for The JENOVA Cognitive Architecture.
Handles loading, validation, and migration of configuration files.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import ValidationError

from jenova.config.config_schema import JenovaConfig


def load_configuration(ui_logger=None, file_logger=None) -> Dict[str, Any]:
    """
    Load and validate configuration from YAML files.

    Args:
        ui_logger: Optional UI logger for user messages
        file_logger: Optional file logger for detailed logging

    Returns:
        Validated configuration dictionary

    Raises:
        RuntimeError: If configuration cannot be loaded or is invalid
    """
    # Find config directory
    config_dir = Path(__file__).parent
    main_config_path = config_dir / "main_config.yaml"
    persona_config_path = config_dir / "persona.yaml"

    if not main_config_path.exists():
        raise RuntimeError(f"Configuration file not found: {main_config_path}")

    # Load YAML files
    try:
        with open(main_config_path, 'r') as f:
            main_config_data = yaml.safe_load(f) or {}
    except Exception as e:
        raise RuntimeError(f"Failed to parse main_config.yaml: {e}")

    # Load persona config if exists
    persona_data = {}
    if persona_config_path.exists():
        try:
            with open(persona_config_path, 'r') as f:
                persona_data = yaml.safe_load(f) or {}
        except Exception as e:
            if ui_logger:
                ui_logger.system_message(
                    f"Warning: Could not load persona.yaml: {e}. Using defaults."
                )
            if file_logger:
                file_logger.log_warning(f"Failed to parse persona.yaml: {e}")

    # Merge persona into main config
    if persona_data:
        main_config_data['persona'] = persona_data

    # Validate with Pydantic
    try:
        validated_config = JenovaConfig(**main_config_data)

        if ui_logger:
            ui_logger.system_message("✓ Configuration validated successfully")
        if file_logger:
            file_logger.log_info("Configuration loaded and validated")

        # Convert back to dict for backward compatibility
        config_dict = validated_config.model_dump()

        return config_dict

    except ValidationError as e:
        error_msg = "Configuration validation failed:\n"
        for error in e.errors():
            field = " -> ".join(str(x) for x in error['loc'])
            message = error['msg']
            error_msg += f"  • {field}: {message}\n"

        if ui_logger:
            ui_logger.error(error_msg)
        if file_logger:
            file_logger.log_error(error_msg)

        raise RuntimeError(error_msg)


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration without loading from files.
    Useful for testing or initialization.

    Returns:
        Default configuration dictionary
    """
    default_config = JenovaConfig()
    return default_config.model_dump()


__all__ = ['load_configuration', 'get_default_config', 'JenovaConfig']
