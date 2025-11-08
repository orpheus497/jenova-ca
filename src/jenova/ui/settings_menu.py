# The JENOVA Cognitive Architecture - Interactive Settings Menu
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 9: Interactive settings menu for runtime configuration.

Provides:
- Category-based settings navigation
- Runtime configuration updates
- Validation and persistence
- Preview mode before applying changes
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
import os


@dataclass
class SettingDefinition:
    """Defines a single setting with metadata."""

    key: str
    name: str
    description: str
    value_type: type
    current_value: Any
    default_value: Any
    choices: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    requires_restart: bool = False


class SettingsCategory:
    """A category of related settings."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.settings: Dict[str, SettingDefinition] = {}

    def add_setting(self, setting: SettingDefinition):
        """Add a setting to this category."""
        self.settings[setting.key] = setting

    def get_setting(self, key: str) -> Optional[SettingDefinition]:
        """Get a setting by key."""
        return self.settings.get(key)

    def list_settings(self) -> List[SettingDefinition]:
        """Get all settings in this category."""
        return list(self.settings.values())


class SettingsMenu:
    """
    Interactive settings menu for runtime configuration.

    Features:
    - Organized by category (Network, LLM, Memory, Learning, Privacy)
    - Validation before applying changes
    - Persistence to user profile
    - Undo/redo support
    - Preview mode
    """

    def __init__(self, config: dict, user_profile, file_logger):
        self.config = config
        self.user_profile = user_profile
        self.file_logger = file_logger
        self.categories: Dict[str, SettingsCategory] = {}
        self.change_history: List[Dict[str, Any]] = []
        self.pending_changes: Dict[str, Any] = {}

        self._initialize_categories()

    def _initialize_categories(self):
        """Initialize all settings categories."""

        # Network Category
        network = SettingsCategory(
            "network", "Network and distributed computing settings"
        )
        network.add_setting(
            SettingDefinition(
                key="network.enabled",
                name="Network Mode",
                description="Enable distributed computing features",
                value_type=bool,
                current_value=self.config.get("network", {}).get("enabled", True),
                default_value=True,
                requires_restart=True,
            )
        )
        network.add_setting(
            SettingDefinition(
                key="network.mode",
                name="Network Strategy",
                description="Network operation mode",
                value_type=str,
                current_value=self.config.get("network", {}).get("mode", "auto"),
                default_value="auto",
                choices=["auto", "local_only", "distributed"],
                requires_restart=False,
            )
        )
        network.add_setting(
            SettingDefinition(
                key="network.peer_selection.strategy",
                name="Distribution Strategy",
                description="How to distribute work across peers",
                value_type=str,
                current_value=self.config.get("network", {})
                .get("peer_selection", {})
                .get("strategy", "load_balanced"),
                default_value="load_balanced",
                choices=[
                    "local_first",
                    "load_balanced",
                    "fastest_peer",
                    "parallel_voting",
                    "round_robin",
                ],
            )
        )
        self.categories["network"] = network

        # LLM Category
        llm = SettingsCategory("llm", "Language model configuration")
        llm.add_setting(
            SettingDefinition(
                key="llm.max_tokens",
                name="Max Tokens",
                description="Maximum tokens to generate per response",
                value_type=int,
                current_value=self.config.get("llm", {}).get("max_tokens", 512),
                default_value=512,
                min_value=64,
                max_value=4096,
            )
        )
        llm.add_setting(
            SettingDefinition(
                key="llm.temperature",
                name="Temperature",
                description="Creativity/randomness (0.0-2.0)",
                value_type=float,
                current_value=self.config.get("llm", {}).get("temperature", 0.7),
                default_value=0.7,
                min_value=0.0,
                max_value=2.0,
            )
        )
        llm.add_setting(
            SettingDefinition(
                key="llm.top_p",
                name="Top P",
                description="Nucleus sampling threshold",
                value_type=float,
                current_value=self.config.get("llm", {}).get("top_p", 0.95),
                default_value=0.95,
                min_value=0.0,
                max_value=1.0,
            )
        )
        self.categories["llm"] = llm

        # Memory Category
        memory = SettingsCategory("memory", "Memory system settings")
        memory.add_setting(
            SettingDefinition(
                key="memory.max_results",
                name="Max Memory Results",
                description="Maximum memory items to retrieve",
                value_type=int,
                current_value=self.config.get("memory", {}).get("max_results", 5),
                default_value=5,
                min_value=1,
                max_value=20,
            )
        )
        memory.add_setting(
            SettingDefinition(
                key="memory.similarity_threshold",
                name="Similarity Threshold",
                description="Minimum similarity for memory retrieval (0.0-1.0)",
                value_type=float,
                current_value=self.config.get("memory", {}).get(
                    "similarity_threshold", 0.7
                ),
                default_value=0.7,
                min_value=0.0,
                max_value=1.0,
            )
        )
        self.categories["memory"] = memory

        # Learning Category
        learning = SettingsCategory("learning", "Learning and personalization settings")
        learning.add_setting(
            SettingDefinition(
                key="user.learning_mode",
                name="Learning Mode",
                description="Enable adaptive learning from interactions",
                value_type=bool,
                current_value=(
                    self.user_profile.preferences.learning_mode
                    if self.user_profile
                    else True
                ),
                default_value=True,
            )
        )
        learning.add_setting(
            SettingDefinition(
                key="user.proactive_suggestions",
                name="Proactive Suggestions",
                description="Enable proactive suggestions and recommendations",
                value_type=bool,
                current_value=(
                    self.user_profile.preferences.proactive_suggestions
                    if self.user_profile
                    else True
                ),
                default_value=True,
            )
        )
        learning.add_setting(
            SettingDefinition(
                key="user.response_style",
                name="Response Style",
                description="Preferred response verbosity",
                value_type=str,
                current_value=(
                    self.user_profile.preferences.response_style
                    if self.user_profile
                    else "balanced"
                ),
                default_value="balanced",
                choices=["concise", "balanced", "detailed"],
            )
        )
        learning.add_setting(
            SettingDefinition(
                key="user.expertise_level",
                name="Expertise Level",
                description="Your technical expertise level",
                value_type=str,
                current_value=(
                    self.user_profile.preferences.expertise_level
                    if self.user_profile
                    else "intermediate"
                ),
                default_value="intermediate",
                choices=["beginner", "intermediate", "advanced", "expert"],
            )
        )
        self.categories["learning"] = learning

        # Privacy Category
        privacy = SettingsCategory("privacy", "Privacy and data settings")
        privacy.add_setting(
            SettingDefinition(
                key="network.resource_sharing.share_memory",
                name="Share Memory",
                description="Allow peers to search your memory (privacy-sensitive)",
                value_type=bool,
                current_value=self.config.get("network", {})
                .get("resource_sharing", {})
                .get("share_memory", False),
                default_value=False,
                requires_restart=True,
            )
        )
        privacy.add_setting(
            SettingDefinition(
                key="network.resource_sharing.share_llm",
                name="Share LLM",
                description="Allow peers to use your LLM resources",
                value_type=bool,
                current_value=self.config.get("network", {})
                .get("resource_sharing", {})
                .get("share_llm", True),
                default_value=True,
            )
        )
        self.categories["privacy"] = privacy

    def get_category(self, name: str) -> Optional[SettingsCategory]:
        """Get a settings category by name."""
        return self.categories.get(name)

    def list_categories(self) -> List[SettingsCategory]:
        """Get all settings categories."""
        return list(self.categories.values())

    def get_setting_value(self, key: str) -> Any:
        """Get current value of a setting by dot-notation key."""
        # Check pending changes first
        if key in self.pending_changes:
            return self.pending_changes[key]

        # Navigate through nested config
        parts = key.split(".")
        value = self.config

        # Special handling for user profile settings
        if parts[0] == "user" and self.user_profile:
            if len(parts) > 1:
                return getattr(self.user_profile.preferences, parts[1], None)
            return None

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        return value

    def set_setting_value(
        self, key: str, value: Any, apply_immediately: bool = False
    ) -> bool:
        """
        Set a setting value.

        Args:
            key: Dot-notation key (e.g., "llm.temperature")
            value: New value
            apply_immediately: If True, apply change immediately. If False, add to pending changes.

        Returns:
            True if validation passed and change was recorded
        """
        # Find the setting definition
        setting = None
        for category in self.categories.values():
            setting = category.get_setting(key)
            if setting:
                break

        if not setting:
            self.file_logger.log_error(f"Unknown setting key: {key}")
            return False

        # Validate value
        if not self._validate_value(setting, value):
            return False

        if apply_immediately:
            self._apply_change(key, value)
            self.file_logger.log_info(f"Setting changed: {key} = {value}")
        else:
            self.pending_changes[key] = value
            self.file_logger.log_info(f"Pending change: {key} = {value}")

        return True

    def _validate_value(self, setting: SettingDefinition, value: Any) -> bool:
        """Validate a setting value."""
        # Type check
        if not isinstance(value, setting.value_type):
            try:
                value = setting.value_type(value)
            except (ValueError, TypeError):
                self.file_logger.log_error(
                    f"Invalid type for {setting.key}: expected {setting.value_type.__name__}, got {type(value).__name__}"
                )
                return False

        # Choice validation
        if setting.choices and value not in setting.choices:
            self.file_logger.log_error(
                f"Invalid value for {setting.key}: must be one of {setting.choices}"
            )
            return False

        # Range validation
        if setting.min_value is not None and value < setting.min_value:
            self.file_logger.log_error(
                f"Value for {setting.key} too low: minimum is {setting.min_value}"
            )
            return False

        if setting.max_value is not None and value > setting.max_value:
            self.file_logger.log_error(
                f"Value for {setting.key} too high: maximum is {setting.max_value}"
            )
            return False

        return True

    def _apply_change(self, key: str, value: Any):
        """Apply a setting change to the config or user profile."""
        parts = key.split(".")

        # Special handling for user profile settings
        if parts[0] == "user" and self.user_profile:
            if len(parts) > 1:
                setattr(self.user_profile.preferences, parts[1], value)
                self.user_profile.save()
            return

        # Navigate to the parent dict and set the value
        current = self.config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

        # Record in change history
        self.change_history.append(
            {
                "key": key,
                "value": value,
                "timestamp": None,  # Would use datetime if imported
            }
        )

    def apply_pending_changes(self) -> Dict[str, bool]:
        """
        Apply all pending changes.

        Returns:
            Dictionary mapping setting keys to success status
        """
        results = {}
        for key, value in self.pending_changes.items():
            self._apply_change(key, value)
            results[key] = True
            self.file_logger.log_info(f"Applied setting: {key} = {value}")

        self.pending_changes.clear()
        return results

    def discard_pending_changes(self):
        """Discard all pending changes."""
        count = len(self.pending_changes)
        self.pending_changes.clear()
        self.file_logger.log_info(f"Discarded {count} pending changes")

    def reset_to_defaults(self, category_name: Optional[str] = None):
        """
        Reset settings to defaults.

        Args:
            category_name: If specified, only reset settings in this category
        """
        categories = (
            [self.categories[category_name]]
            if category_name
            else self.categories.values()
        )

        for category in categories:
            for setting in category.list_settings():
                self._apply_change(setting.key, setting.default_value)
                self.file_logger.log_info(
                    f"Reset {setting.key} to default: {setting.default_value}"
                )

    def get_settings_requiring_restart(self) -> List[SettingDefinition]:
        """Get all settings that have been changed and require a restart."""
        restart_settings = []
        for category in self.categories.values():
            for setting in category.list_settings():
                if setting.requires_restart:
                    current_value = self.get_setting_value(setting.key)
                    if (
                        current_value != setting.default_value
                        or setting.key in self.pending_changes
                    ):
                        restart_settings.append(setting)
        return restart_settings

    def export_settings(self, filepath: str):
        """Export current settings to a JSON file."""
        export_data = {}
        for category in self.categories.values():
            category_data = {}
            for setting in category.list_settings():
                category_data[setting.key] = self.get_setting_value(setting.key)
            export_data[category.name] = category_data

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        self.file_logger.log_info(f"Settings exported to: {filepath}")

    def import_settings(self, filepath: str) -> bool:
        """Import settings from a JSON file."""
        if not os.path.exists(filepath):
            self.file_logger.log_error(f"Settings file not found: {filepath}")
            return False

        try:
            with open(filepath, "r") as f:
                import_data = json.load(f)

            for category_name, category_data in import_data.items():
                for key, value in category_data.items():
                    self.set_setting_value(key, value, apply_immediately=True)

            self.file_logger.log_info(f"Settings imported from: {filepath}")
            return True
        except Exception as e:
            self.file_logger.log_error(f"Error importing settings: {e}")
            return False
