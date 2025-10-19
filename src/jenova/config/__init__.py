import yaml
from pathlib import Path

def load_configuration(ui_logger, file_logger):
    """Loads and merges configuration from YAML files."""
    try:
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
        
        ui_logger.info(">> Configuration loaded successfully.")
        file_logger.log_info("Configuration loaded successfully.")
        
        return merged_config

    except FileNotFoundError as e:
        error_message = f"Configuration file not found: {e}. Please ensure main_config.yaml and persona.yaml are in the config directory."
        ui_logger.system_message(error_message)
        file_logger.log_error(error_message)
        raise
    except yaml.YAMLError as e:
        error_message = f"Error parsing YAML configuration file: {e}"
        ui_logger.system_message(error_message)
        file_logger.log_error(error_message)
        raise
    except Exception as e:
        error_message = f"An unexpected error occurred while loading configuration: {e}"
        ui_logger.system_message(error_message)
        file_logger.log_error(error_message)
        raise