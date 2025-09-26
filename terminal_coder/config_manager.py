"""
Advanced Configuration and Settings Management System
Comprehensive settings management with profiles, validation, and encryption
"""

import os
import json
import yaml
import toml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
import logging
import shutil
from cryptography.fernet import Fernet
import base64
import hashlib


class ConfigFormat(Enum):
    """Configuration file formats"""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"


class SettingType(Enum):
    """Setting data types"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    PATH = "path"
    SECRET = "secret"  # Encrypted storage


@dataclass
class SettingDefinition:
    """Definition of a configuration setting"""
    key: str
    name: str
    description: str
    setting_type: SettingType
    default_value: Any
    required: bool = False
    category: str = "general"
    validator: Optional[Callable] = None
    choices: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    secret: bool = False  # Whether to encrypt this setting
    restart_required: bool = False  # Whether changing this requires restart


@dataclass
class ConfigProfile:
    """Configuration profile"""
    name: str
    description: str
    settings: Dict[str, Any]
    created_at: datetime
    modified_at: datetime
    active: bool = False


class ConfigValidator:
    """Configuration validation utilities"""

    @staticmethod
    def validate_setting(definition: SettingDefinition, value: Any) -> tuple[bool, str]:
        """Validate a setting value against its definition"""
        try:
            # Check required
            if definition.required and value is None:
                return False, f"{definition.name} is required"

            # Skip validation for None values if not required
            if value is None and not definition.required:
                return True, ""

            # Type validation
            if not ConfigValidator._validate_type(definition.setting_type, value):
                return False, f"{definition.name} must be of type {definition.setting_type.value}"

            # Choice validation
            if definition.choices and value not in definition.choices:
                return False, f"{definition.name} must be one of: {definition.choices}"

            # Range validation
            if definition.setting_type in [SettingType.INTEGER, SettingType.FLOAT]:
                if definition.min_value is not None and value < definition.min_value:
                    return False, f"{definition.name} must be >= {definition.min_value}"
                if definition.max_value is not None and value > definition.max_value:
                    return False, f"{definition.name} must be <= {definition.max_value}"

            # Path validation
            if definition.setting_type == SettingType.PATH:
                path = Path(value)
                if not path.exists() and definition.required:
                    return False, f"Path {value} does not exist"

            # Custom validator
            if definition.validator:
                result = definition.validator(value)
                if isinstance(result, tuple):
                    valid, message = result
                    if not valid:
                        return False, message
                elif not result:
                    return False, f"Custom validation failed for {definition.name}"

            return True, ""

        except Exception as e:
            return False, f"Validation error for {definition.name}: {str(e)}"

    @staticmethod
    def _validate_type(setting_type: SettingType, value: Any) -> bool:
        """Validate value type"""
        type_mapping = {
            SettingType.STRING: str,
            SettingType.INTEGER: int,
            SettingType.FLOAT: (int, float),
            SettingType.BOOLEAN: bool,
            SettingType.LIST: list,
            SettingType.DICT: dict,
            SettingType.PATH: str,
            SettingType.SECRET: str
        }

        expected_type = type_mapping.get(setting_type)
        if expected_type:
            return isinstance(value, expected_type)
        return True


class EncryptionManager:
    """Manages encryption for sensitive settings"""

    def __init__(self, key: bytes = None):
        if key:
            self.key = key
        else:
            # Generate key from system info (for consistent encryption)
            system_info = f"{os.environ.get('USERNAME', 'user')}_{os.environ.get('COMPUTERNAME', 'host')}"
            key_hash = hashlib.sha256(system_info.encode()).digest()
            self.key = base64.urlsafe_b64encode(key_hash[:32])

        self.cipher = Fernet(self.key)

    def encrypt(self, value: str) -> str:
        """Encrypt a string value"""
        if isinstance(value, str):
            return self.cipher.encrypt(value.encode()).decode()
        return str(value)

    def decrypt(self, encrypted_value: str) -> str:
        """Decrypt an encrypted string"""
        try:
            return self.cipher.decrypt(encrypted_value.encode()).decode()
        except Exception:
            # Return as-is if decryption fails (might be unencrypted)
            return encrypted_value


class AdvancedConfigManager:
    """Advanced configuration management system"""

    def __init__(self, config_dir: str = None):
        self.config_dir = Path(config_dir or Path.home() / ".terminal_coder")
        self.config_dir.mkdir(exist_ok=True)

        self.config_file = self.config_dir / "config.json"
        self.profiles_file = self.config_dir / "profiles.json"
        self.settings_schema_file = self.config_dir / "settings_schema.json"

        self.logger = logging.getLogger(__name__)
        self.encryption_manager = EncryptionManager()

        # Initialize default settings schema
        self.settings_definitions: Dict[str, SettingDefinition] = {}
        self._initialize_default_settings()

        # Load configuration
        self.config = self._load_config()
        self.profiles = self._load_profiles()
        self.current_profile = self._get_active_profile()

        # Configuration change callbacks
        self._change_callbacks: Dict[str, List[Callable]] = {}

    def _initialize_default_settings(self):
        """Initialize default settings schema"""
        default_settings = [
            # General Settings
            SettingDefinition(
                key="app.name",
                name="Application Name",
                description="Name of the application",
                setting_type=SettingType.STRING,
                default_value="Terminal Coder",
                category="general"
            ),
            SettingDefinition(
                key="app.version",
                name="Application Version",
                description="Current application version",
                setting_type=SettingType.STRING,
                default_value="1.0.0",
                category="general"
            ),
            SettingDefinition(
                key="workspace.directory",
                name="Workspace Directory",
                description="Default workspace directory for projects",
                setting_type=SettingType.PATH,
                default_value=str(Path.home() / "terminal_coder_workspace"),
                category="workspace",
                required=True
            ),
            SettingDefinition(
                key="workspace.auto_save",
                name="Auto Save",
                description="Automatically save work at intervals",
                setting_type=SettingType.BOOLEAN,
                default_value=True,
                category="workspace"
            ),
            SettingDefinition(
                key="workspace.auto_save_interval",
                name="Auto Save Interval",
                description="Auto save interval in seconds",
                setting_type=SettingType.INTEGER,
                default_value=300,
                category="workspace",
                min_value=30,
                max_value=3600
            ),

            # AI Settings
            SettingDefinition(
                key="ai.default_provider",
                name="Default AI Provider",
                description="Default AI provider to use",
                setting_type=SettingType.STRING,
                default_value="openai",
                category="ai",
                choices=["openai", "anthropic", "google", "cohere"]
            ),
            SettingDefinition(
                key="ai.default_model",
                name="Default AI Model",
                description="Default AI model to use",
                setting_type=SettingType.STRING,
                default_value="gpt-4",
                category="ai"
            ),
            SettingDefinition(
                key="ai.max_tokens",
                name="Max Tokens",
                description="Maximum tokens for AI responses",
                setting_type=SettingType.INTEGER,
                default_value=4000,
                category="ai",
                min_value=100,
                max_value=128000
            ),
            SettingDefinition(
                key="ai.temperature",
                name="Temperature",
                description="AI response creativity (0.0-2.0)",
                setting_type=SettingType.FLOAT,
                default_value=0.7,
                category="ai",
                min_value=0.0,
                max_value=2.0
            ),

            # API Keys (encrypted)
            SettingDefinition(
                key="api.openai_key",
                name="OpenAI API Key",
                description="API key for OpenAI services",
                setting_type=SettingType.SECRET,
                default_value="",
                category="api_keys",
                secret=True
            ),
            SettingDefinition(
                key="api.anthropic_key",
                name="Anthropic API Key",
                description="API key for Anthropic services",
                setting_type=SettingType.SECRET,
                default_value="",
                category="api_keys",
                secret=True
            ),
            SettingDefinition(
                key="api.google_key",
                name="Google API Key",
                description="API key for Google services",
                setting_type=SettingType.SECRET,
                default_value="",
                category="api_keys",
                secret=True
            ),
            SettingDefinition(
                key="api.cohere_key",
                name="Cohere API Key",
                description="API key for Cohere services",
                setting_type=SettingType.SECRET,
                default_value="",
                category="api_keys",
                secret=True
            ),

            # UI Settings
            SettingDefinition(
                key="ui.theme",
                name="Theme",
                description="Application color theme",
                setting_type=SettingType.STRING,
                default_value="dark",
                category="ui",
                choices=["light", "dark", "auto"]
            ),
            SettingDefinition(
                key="ui.show_line_numbers",
                name="Show Line Numbers",
                description="Show line numbers in code editor",
                setting_type=SettingType.BOOLEAN,
                default_value=True,
                category="ui"
            ),
            SettingDefinition(
                key="ui.syntax_highlighting",
                name="Syntax Highlighting",
                description="Enable syntax highlighting",
                setting_type=SettingType.BOOLEAN,
                default_value=True,
                category="ui"
            ),
            SettingDefinition(
                key="ui.font_size",
                name="Font Size",
                description="Font size for the interface",
                setting_type=SettingType.INTEGER,
                default_value=12,
                category="ui",
                min_value=8,
                max_value=24
            ),

            # Performance Settings
            SettingDefinition(
                key="performance.max_concurrent_requests",
                name="Max Concurrent Requests",
                description="Maximum concurrent API requests",
                setting_type=SettingType.INTEGER,
                default_value=5,
                category="performance",
                min_value=1,
                max_value=20
            ),
            SettingDefinition(
                key="performance.request_timeout",
                name="Request Timeout",
                description="API request timeout in seconds",
                setting_type=SettingType.INTEGER,
                default_value=30,
                category="performance",
                min_value=5,
                max_value=300
            ),
            SettingDefinition(
                key="performance.cache_enabled",
                name="Enable Caching",
                description="Enable response caching",
                setting_type=SettingType.BOOLEAN,
                default_value=True,
                category="performance"
            ),

            # Security Settings
            SettingDefinition(
                key="security.encrypt_config",
                name="Encrypt Configuration",
                description="Encrypt sensitive configuration data",
                setting_type=SettingType.BOOLEAN,
                default_value=True,
                category="security",
                restart_required=True
            ),
            SettingDefinition(
                key="security.log_level",
                name="Logging Level",
                description="Application logging level",
                setting_type=SettingType.STRING,
                default_value="INFO",
                category="security",
                choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            ),

            # Feature Toggles
            SettingDefinition(
                key="features.code_completion",
                name="Code Completion",
                description="Enable AI-powered code completion",
                setting_type=SettingType.BOOLEAN,
                default_value=True,
                category="features"
            ),
            SettingDefinition(
                key="features.error_analysis",
                name="Error Analysis",
                description="Enable intelligent error analysis",
                setting_type=SettingType.BOOLEAN,
                default_value=True,
                category="features"
            ),
            SettingDefinition(
                key="features.security_scanning",
                name="Security Scanning",
                description="Enable security vulnerability scanning",
                setting_type=SettingType.BOOLEAN,
                default_value=True,
                category="features"
            ),
            SettingDefinition(
                key="features.performance_monitoring",
                name="Performance Monitoring",
                description="Enable performance monitoring",
                setting_type=SettingType.BOOLEAN,
                default_value=True,
                category="features"
            )
        ]

        for setting in default_settings:
            self.settings_definitions[setting.key] = setting

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if not self.config_file.exists():
            return self._get_default_config()

        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)

            # Decrypt sensitive values
            decrypted_config = {}
            for key, value in config.items():
                setting_def = self.settings_definitions.get(key)
                if setting_def and setting_def.secret:
                    decrypted_config[key] = self.encryption_manager.decrypt(value)
                else:
                    decrypted_config[key] = value

            return decrypted_config

        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        default_config = {}
        for key, setting_def in self.settings_definitions.items():
            default_config[key] = setting_def.default_value
        return default_config

    def _save_config(self):
        """Save configuration to file"""
        try:
            # Encrypt sensitive values
            encrypted_config = {}
            for key, value in self.config.items():
                setting_def = self.settings_definitions.get(key)
                if setting_def and setting_def.secret and value:
                    encrypted_config[key] = self.encryption_manager.encrypt(str(value))
                else:
                    encrypted_config[key] = value

            with open(self.config_file, 'w') as f:
                json.dump(encrypted_config, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Error saving config: {e}")

    def _load_profiles(self) -> List[ConfigProfile]:
        """Load configuration profiles"""
        if not self.profiles_file.exists():
            return []

        try:
            with open(self.profiles_file, 'r') as f:
                data = json.load(f)

            profiles = []
            for profile_data in data:
                profile_data['created_at'] = datetime.fromisoformat(profile_data['created_at'])
                profile_data['modified_at'] = datetime.fromisoformat(profile_data['modified_at'])
                profiles.append(ConfigProfile(**profile_data))

            return profiles

        except Exception as e:
            self.logger.error(f"Error loading profiles: {e}")
            return []

    def _save_profiles(self):
        """Save configuration profiles"""
        try:
            data = []
            for profile in self.profiles:
                profile_dict = asdict(profile)
                profile_dict['created_at'] = profile.created_at.isoformat()
                profile_dict['modified_at'] = profile.modified_at.isoformat()
                data.append(profile_dict)

            with open(self.profiles_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Error saving profiles: {e}")

    def _get_active_profile(self) -> Optional[ConfigProfile]:
        """Get currently active profile"""
        for profile in self.profiles:
            if profile.active:
                return profile
        return None

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        # First check current profile
        if self.current_profile and key in self.current_profile.settings:
            return self.current_profile.settings[key]

        # Then check main config
        return self.config.get(key, default)

    def set(self, key: str, value: Any, validate: bool = True) -> bool:
        """Set configuration value"""
        # Validate if requested
        if validate and key in self.settings_definitions:
            setting_def = self.settings_definitions[key]
            valid, error = ConfigValidator.validate_setting(setting_def, value)
            if not valid:
                self.logger.error(f"Validation error for {key}: {error}")
                return False

        # Set in current profile if active
        if self.current_profile:
            self.current_profile.settings[key] = value
            self.current_profile.modified_at = datetime.now()
            self._save_profiles()
        else:
            # Set in main config
            self.config[key] = value
            self._save_config()

        # Trigger callbacks
        self._trigger_callbacks(key, value)
        return True

    def get_category_settings(self, category: str) -> Dict[str, Any]:
        """Get all settings for a category"""
        settings = {}
        for key, setting_def in self.settings_definitions.items():
            if setting_def.category == category:
                settings[key] = self.get(key)
        return settings

    def get_categories(self) -> List[str]:
        """Get all setting categories"""
        categories = set()
        for setting_def in self.settings_definitions.values():
            categories.add(setting_def.category)
        return sorted(list(categories))

    def get_setting_definition(self, key: str) -> Optional[SettingDefinition]:
        """Get setting definition"""
        return self.settings_definitions.get(key)

    def add_setting_definition(self, setting_def: SettingDefinition):
        """Add custom setting definition"""
        self.settings_definitions[setting_def.key] = setting_def
        # Set default value if not already set
        if self.get(setting_def.key) is None:
            self.set(setting_def.key, setting_def.default_value, validate=False)

    def create_profile(self, name: str, description: str,
                      base_settings: Dict[str, Any] = None) -> ConfigProfile:
        """Create new configuration profile"""
        profile = ConfigProfile(
            name=name,
            description=description,
            settings=base_settings or {},
            created_at=datetime.now(),
            modified_at=datetime.now()
        )

        self.profiles.append(profile)
        self._save_profiles()
        return profile

    def activate_profile(self, name: str) -> bool:
        """Activate a configuration profile"""
        profile = next((p for p in self.profiles if p.name == name), None)
        if not profile:
            return False

        # Deactivate current profile
        if self.current_profile:
            self.current_profile.active = False

        # Activate new profile
        profile.active = True
        self.current_profile = profile
        self._save_profiles()

        # Trigger callbacks for all changed settings
        for key, value in profile.settings.items():
            self._trigger_callbacks(key, value)

        return True

    def deactivate_profile(self):
        """Deactivate current profile"""
        if self.current_profile:
            self.current_profile.active = False
            self.current_profile = None
            self._save_profiles()

    def delete_profile(self, name: str) -> bool:
        """Delete a configuration profile"""
        profile = next((p for p in self.profiles if p.name == name), None)
        if not profile:
            return False

        if profile.active:
            self.deactivate_profile()

        self.profiles = [p for p in self.profiles if p.name != name]
        self._save_profiles()
        return True

    def list_profiles(self) -> List[ConfigProfile]:
        """List all profiles"""
        return self.profiles.copy()

    def export_config(self, filepath: str, format: ConfigFormat = ConfigFormat.JSON,
                     include_secrets: bool = False):
        """Export configuration to file"""
        config_data = self.config.copy()

        # Remove secrets if not including them
        if not include_secrets:
            for key, setting_def in self.settings_definitions.items():
                if setting_def.secret and key in config_data:
                    config_data[key] = "***REDACTED***"

        # Add metadata
        export_data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "version": self.get("app.version", "1.0.0"),
                "format": format.value
            },
            "settings": config_data,
            "profiles": [asdict(p) for p in self.profiles] if self.profiles else []
        }

        try:
            filepath = Path(filepath)
            if format == ConfigFormat.JSON:
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            elif format == ConfigFormat.YAML:
                import yaml
                with open(filepath, 'w') as f:
                    yaml.dump(export_data, f, default_flow_style=False)
            elif format == ConfigFormat.TOML:
                import toml
                with open(filepath, 'w') as f:
                    toml.dump(export_data, f)

        except Exception as e:
            self.logger.error(f"Error exporting config: {e}")
            raise

    def import_config(self, filepath: str, merge: bool = True) -> bool:
        """Import configuration from file"""
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileNotFoundError(f"Config file not found: {filepath}")

            # Detect format from extension
            if filepath.suffix.lower() == '.yaml' or filepath.suffix.lower() == '.yml':
                with open(filepath, 'r') as f:
                    data = yaml.safe_load(f)
            elif filepath.suffix.lower() == '.toml':
                with open(filepath, 'r') as f:
                    data = toml.load(f)
            else:  # Default to JSON
                with open(filepath, 'r') as f:
                    data = json.load(f)

            # Import settings
            imported_settings = data.get('settings', {})
            if merge:
                # Merge with existing config
                for key, value in imported_settings.items():
                    if self.set(key, value):
                        self.logger.info(f"Imported setting: {key}")
            else:
                # Replace entire config
                self.config = imported_settings
                self._save_config()

            # Import profiles if present
            if 'profiles' in data:
                for profile_data in data['profiles']:
                    if 'created_at' in profile_data:
                        profile_data['created_at'] = datetime.fromisoformat(profile_data['created_at'])
                    if 'modified_at' in profile_data:
                        profile_data['modified_at'] = datetime.fromisoformat(profile_data['modified_at'])

                    profile = ConfigProfile(**profile_data)
                    # Check if profile already exists
                    existing = next((p for p in self.profiles if p.name == profile.name), None)
                    if existing:
                        # Update existing profile
                        existing.settings.update(profile.settings)
                        existing.modified_at = datetime.now()
                    else:
                        # Add new profile
                        self.profiles.append(profile)

                self._save_profiles()

            return True

        except Exception as e:
            self.logger.error(f"Error importing config: {e}")
            return False

    def reset_to_defaults(self, category: str = None):
        """Reset settings to default values"""
        if category:
            # Reset specific category
            for key, setting_def in self.settings_definitions.items():
                if setting_def.category == category:
                    self.set(key, setting_def.default_value, validate=False)
        else:
            # Reset all settings
            self.config = self._get_default_config()
            self._save_config()

    def backup_config(self, backup_dir: str = None) -> str:
        """Create backup of current configuration"""
        if not backup_dir:
            backup_dir = self.config_dir / "backups"

        backup_dir = Path(backup_dir)
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"config_backup_{timestamp}.json"

        try:
            # Create backup
            shutil.copy2(self.config_file, backup_file)

            # Also backup profiles
            if self.profiles_file.exists():
                profiles_backup = backup_dir / f"profiles_backup_{timestamp}.json"
                shutil.copy2(self.profiles_file, profiles_backup)

            return str(backup_file)

        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
            raise

    def restore_config(self, backup_file: str) -> bool:
        """Restore configuration from backup"""
        try:
            backup_path = Path(backup_file)
            if not backup_path.exists():
                return False

            # Restore main config
            shutil.copy2(backup_path, self.config_file)
            self.config = self._load_config()

            # Try to restore profiles
            profiles_backup = backup_path.parent / f"profiles_backup_{backup_path.stem.split('_')[-1]}.json"
            if profiles_backup.exists():
                shutil.copy2(profiles_backup, self.profiles_file)
                self.profiles = self._load_profiles()

            return True

        except Exception as e:
            self.logger.error(f"Error restoring config: {e}")
            return False

    def add_change_callback(self, key: str, callback: Callable):
        """Add callback for configuration changes"""
        if key not in self._change_callbacks:
            self._change_callbacks[key] = []
        self._change_callbacks[key].append(callback)

    def _trigger_callbacks(self, key: str, value: Any):
        """Trigger callbacks for configuration change"""
        if key in self._change_callbacks:
            for callback in self._change_callbacks[key]:
                try:
                    callback(key, value)
                except Exception as e:
                    self.logger.error(f"Error in config callback: {e}")

    def validate_all(self) -> Dict[str, List[str]]:
        """Validate all current settings"""
        errors = {}

        for key, value in self.config.items():
            if key in self.settings_definitions:
                setting_def = self.settings_definitions[key]
                valid, error = ConfigValidator.validate_setting(setting_def, value)
                if not valid:
                    category = setting_def.category
                    if category not in errors:
                        errors[category] = []
                    errors[category].append(f"{setting_def.name}: {error}")

        return errors

    def get_settings_summary(self) -> Dict[str, Any]:
        """Get comprehensive settings summary"""
        categories = {}
        total_settings = 0
        invalid_settings = 0

        for key, setting_def in self.settings_definitions.items():
            category = setting_def.category
            if category not in categories:
                categories[category] = {
                    "count": 0,
                    "settings": []
                }

            current_value = self.get(key)
            valid, error = ConfigValidator.validate_setting(setting_def, current_value)

            setting_info = {
                "key": key,
                "name": setting_def.name,
                "type": setting_def.setting_type.value,
                "current_value": current_value if not setting_def.secret else "***",
                "default_value": setting_def.default_value if not setting_def.secret else "***",
                "valid": valid,
                "error": error if not valid else None,
                "required": setting_def.required,
                "secret": setting_def.secret
            }

            categories[category]["settings"].append(setting_info)
            categories[category]["count"] += 1
            total_settings += 1

            if not valid:
                invalid_settings += 1

        return {
            "total_settings": total_settings,
            "invalid_settings": invalid_settings,
            "categories": categories,
            "active_profile": self.current_profile.name if self.current_profile else None,
            "total_profiles": len(self.profiles)
        }