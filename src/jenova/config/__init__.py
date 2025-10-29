# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""This module is responsible for loading the configuration for the JENOVA Cognitive Architecture.
"""

from pathlib import Path

import yaml


def load_configuration(ui_logger=None, file_logger=None):
    """Loads and merges configuration from YAML files."""
    try:
        if ui_logger:
            ui_logger.info(">> Loading configuration...")

        # Assuming this file is at src/jenova/config/__init__.py
        main_config_path = Path(__file__).parent / 'main_config.yaml'
        with open(main_config_path, 'r', encoding='utf-8') as f:
            main_config = yaml.safe_load(f)

        persona_config_path = Path(__file__).parent / 'persona.yaml'
        with open(persona_config_path, 'r', encoding='utf-8') as f:
            persona_config = yaml.safe_load(f)

        # Merge configurations
        merged_config = main_config
        merged_config['persona'] = persona_config

        if ui_logger:
            ui_logger.info(">> Configuration loaded successfully.")
        if file_logger:
            file_logger.log_info("Configuration loaded successfully.")

        return merged_config

    except FileNotFoundError as e:
        error_message = f"Configuration file not found: {e}. Please ensure main_config.yaml and persona.yaml are in the config directory."
        if ui_logger:
            ui_logger.system_message(error_message)
        if file_logger:
            file_logger.log_error(error_message)
        raise
    except yaml.YAMLError as e:
        error_message = f"Error parsing YAML configuration file: {e}"
        if ui_logger:
            ui_logger.system_message(error_message)
        if file_logger:
            file_logger.log_error(error_message)
        raise
    except Exception as e:
        error_message = f"An unexpected error occurred while loading configuration: {e}"
        if ui_logger:
            ui_logger.system_message(error_message)
        if file_logger:
            file_logger.log_error(error_message)
        raise
