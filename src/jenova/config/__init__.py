##Script function and purpose: Config package initialization - exposes load_config function
"""Configuration loading for JENOVA."""

from jenova.config.models import JenovaConfig, load_config

__all__ = ["JenovaConfig", "load_config"]
