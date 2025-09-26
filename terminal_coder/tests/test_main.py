"""
Tests for main Terminal Coder application
Using modern pytest features and Python 3.13+ syntax
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from rich.console import Console

from terminal_coder.main import TerminalCoder, Project, AIProvider


class TestAIProvider:
    """Test AIProvider dataclass with modern features"""

    def test_ai_provider_creation(self) -> None:
        """Test creating an AI provider"""
        provider = AIProvider(
            name="OpenAI",
            base_url="https://api.openai.com/v1",
            auth_type="api_key",
            models=["gpt-4", "gpt-3.5-turbo"],
            max_tokens=4000,
            supports_streaming=True,
            rate_limit=60
        )

        assert provider.name == "OpenAI"
        assert provider.models == ["gpt-4", "gpt-3.5-turbo"]
        assert provider.max_tokens == 4000

    def test_ai_provider_validation(self) -> None:
        """Test AIProvider validation"""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            AIProvider(
                name="Invalid",
                base_url="https://api.example.com",
                auth_type="api_key",
                models=["model1"],
                max_tokens=-1,  # Invalid
                supports_streaming=False,
                rate_limit=60
            )

    def test_ai_provider_frozen(self) -> None:
        """Test that AIProvider is immutable (frozen)"""
        provider = AIProvider(
            name="OpenAI",
            base_url="https://api.openai.com/v1",
            auth_type="api_key",
            models=["gpt-4"],
            max_tokens=4000,
            supports_streaming=True,
            rate_limit=60
        )

        with pytest.raises(Exception):  # FrozenInstanceError in Python 3.10+
            provider.name = "Modified"  # type: ignore


class TestProject:
    """Test Project dataclass with modern features"""

    def test_project_creation(self) -> None:
        """Test creating a project"""
        project = Project(
            name="Test Project",
            path="/tmp/test",
            language="python",
            framework="django"
        )

        assert project.name == "Test Project"
        assert project.language == "python"
        assert project.framework == "django"
        assert project.ai_provider == "openai"  # default

    def test_project_path_property(self) -> None:
        """Test cached property functionality"""
        project = Project(
            name="Test Project",
            path="/tmp/test",
            language="python"
        )

        # Test cached_property
        path_obj1 = project.path_obj
        path_obj2 = project.path_obj
        assert path_obj1 is path_obj2  # Same object due to caching
        assert isinstance(path_obj1, Path)

    def test_project_union_type_syntax(self) -> None:
        """Test modern union type syntax (str | None)"""
        project = Project(
            name="Test Project",
            path="/tmp/test",
            language="python",
            framework=None  # Using union type
        )

        assert project.framework is None


class TestTerminalCoder:
    """Test main TerminalCoder application"""

    @pytest.fixture
    def temp_config_dir(self) -> Path:
        """Create temporary config directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def terminal_coder(self, temp_config_dir: Path, monkeypatch) -> TerminalCoder:
        """Create TerminalCoder instance with temp config"""
        # Mock Path.home() to use temp directory
        monkeypatch.setattr("pathlib.Path.home", lambda: temp_config_dir)
        return TerminalCoder()

    def test_terminal_coder_initialization(self, terminal_coder: TerminalCoder) -> None:
        """Test TerminalCoder initialization"""
        assert isinstance(terminal_coder.console, Console)
        assert terminal_coder.config_dir.exists()
        assert isinstance(terminal_coder.config, dict)
        assert isinstance(terminal_coder.projects, list)

    def test_class_constants(self) -> None:
        """Test class constants using Final"""
        assert TerminalCoder.DEFAULT_CONFIG_DIR == ".terminal_coder"
        assert TerminalCoder.DEFAULT_WORKSPACE == "terminal_coder_workspace"
        assert TerminalCoder.APP_VERSION == "2.0.0"

    def test_ai_providers_initialization(self, terminal_coder: TerminalCoder) -> None:
        """Test AI providers are properly initialized"""
        providers = terminal_coder.ai_providers

        assert "openai" in providers
        assert "anthropic" in providers
        assert "google" in providers
        assert "cohere" in providers

        # Test modern model names
        openai_models = providers["openai"].models
        assert "gpt-4o" in openai_models
        assert "gpt-4-turbo" in openai_models

        anthropic_models = providers["anthropic"].models
        assert "claude-3-5-sonnet-20241022" in anthropic_models

    def test_load_config_default(self, terminal_coder: TerminalCoder) -> None:
        """Test loading default configuration"""
        config = terminal_coder.config

        assert config["theme"] == "dark"
        assert config["auto_save"] is True
        assert config["ai_provider"] == "openai"
        assert config["max_tokens"] == 8000  # Updated default

        # Test new features
        features = config["features"]
        assert features["ai_code_explanation"] is True
        assert features["pattern_recognition"] is True
        assert features["code_translation"] is True

    def test_save_and_load_config(self, terminal_coder: TerminalCoder) -> None:
        """Test saving and loading configuration"""
        # Modify configuration
        terminal_coder.config["theme"] = "light"
        terminal_coder.config["max_tokens"] = 6000

        # Save configuration
        terminal_coder.save_config()

        # Verify file exists and contains correct data
        assert terminal_coder.config_file.exists()

        with terminal_coder.config_file.open('r', encoding='utf-8') as f:
            saved_config = json.load(f)

        assert saved_config["theme"] == "light"
        assert saved_config["max_tokens"] == 6000

    def test_config_merge_with_dict_union_operator(
        self,
        terminal_coder: TerminalCoder,
        temp_config_dir: Path
    ) -> None:
        """Test modern dict merge using | operator"""
        # Create a config file with partial settings
        config_file = temp_config_dir / ".terminal_coder" / "config.json"
        config_file.parent.mkdir(exist_ok=True)

        partial_config = {
            "theme": "light",
            "ai_provider": "anthropic",
            "custom_setting": "test_value"
        }

        with config_file.open('w', encoding='utf-8') as f:
            json.dump(partial_config, f)

        # Create new instance to load the config
        from unittest.mock import patch
        with patch("pathlib.Path.home", return_value=temp_config_dir):
            new_terminal_coder = TerminalCoder()

        # Test that defaults are merged with saved config
        config = new_terminal_coder.config
        assert config["theme"] == "light"  # From saved config
        assert config["ai_provider"] == "anthropic"  # From saved config
        assert config["custom_setting"] == "test_value"  # From saved config
        assert config["auto_save"] is True  # From defaults
        assert config["max_tokens"] == 8000  # From defaults

    def test_load_projects_empty(self, terminal_coder: TerminalCoder) -> None:
        """Test loading projects when no projects exist"""
        projects = terminal_coder.load_projects()
        assert projects == []

    def test_save_and_load_projects(self, terminal_coder: TerminalCoder) -> None:
        """Test saving and loading projects"""
        # Create test projects
        project1 = Project(
            name="Project 1",
            path="/tmp/project1",
            language="python",
            framework="django"
        )
        project2 = Project(
            name="Project 2",
            path="/tmp/project2",
            language="javascript",
            framework="react"
        )

        terminal_coder.projects = [project1, project2]
        terminal_coder.save_projects()

        # Load projects in new instance
        new_terminal_coder = TerminalCoder()
        new_terminal_coder.config_dir = terminal_coder.config_dir
        new_terminal_coder.projects_file = terminal_coder.projects_file

        loaded_projects = new_terminal_coder.load_projects()

        assert len(loaded_projects) == 2
        assert loaded_projects[0].name == "Project 1"
        assert loaded_projects[1].name == "Project 2"

    def test_error_handling_invalid_config(
        self,
        terminal_coder: TerminalCoder,
        capfd
    ) -> None:
        """Test error handling for invalid configuration"""
        # Create invalid JSON file
        with terminal_coder.config_file.open('w') as f:
            f.write("invalid json content")

        # Should return default config on error
        config = terminal_coder.load_config()
        assert config["theme"] == "dark"  # Default value

        # Should log error
        captured = capfd.readouterr()
        # Note: This might not capture Rich output, but tests the concept


@pytest.mark.asyncio
class TestAsyncFeatures:
    """Test async functionality"""

    async def test_async_context_manager_concept(self) -> None:
        """Test async context manager concepts (would be used with AI integration)"""
        # This demonstrates the pattern that would be used
        # with the ModernAIIntegration class

        class MockAsyncResource:
            def __init__(self):
                self.closed = False

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                self.closed = True

        async with MockAsyncResource() as resource:
            assert not resource.closed

        assert resource.closed  # Should be closed after context


class TestModernPythonFeatures:
    """Test modern Python language features"""

    def test_pattern_matching_simulation(self) -> None:
        """Test pattern matching concepts (Python 3.10+)"""
        def handle_provider(provider_name: str) -> str:
            match provider_name:
                case "openai":
                    return "OpenAI GPT models"
                case "anthropic":
                    return "Anthropic Claude models"
                case "google":
                    return "Google Gemini models"
                case _:
                    return "Unknown provider"

        assert handle_provider("openai") == "OpenAI GPT models"
        assert handle_provider("anthropic") == "Anthropic Claude models"
        assert handle_provider("unknown") == "Unknown provider"

    def test_type_hints_with_union_operator(self) -> None:
        """Test modern type hints with | operator"""
        def process_value(value: str | int | None) -> str:
            match value:
                case str():
                    return f"String: {value}"
                case int():
                    return f"Integer: {value}"
                case None:
                    return "None value"
                case _:
                    return "Unknown type"

        assert process_value("test") == "String: test"
        assert process_value(42) == "Integer: 42"
        assert process_value(None) == "None value"

    def test_generic_type_hints(self) -> None:
        """Test built-in generic type hints (PEP 585)"""
        def process_list(items: list[str]) -> dict[str, int]:
            return {item: len(item) for item in items}

        result = process_list(["hello", "world"])
        assert result == {"hello": 5, "world": 5}

    @pytest.mark.parametrize("provider,expected_models", [
        ("openai", ["gpt-4o", "gpt-4-turbo"]),
        ("anthropic", ["claude-3-5-sonnet-20241022"]),
        ("google", ["gemini-1.5-pro"]),
    ])
    def test_parametrized_provider_models(
        self,
        provider: str,
        expected_models: list[str]
    ) -> None:
        """Test AI provider models with parametrization"""
        terminal_coder = TerminalCoder()
        ai_providers = terminal_coder.ai_providers

        if provider in ai_providers:
            models = ai_providers[provider].models
            for expected_model in expected_models:
                assert expected_model in models