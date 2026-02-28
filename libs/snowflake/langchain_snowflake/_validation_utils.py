"""Centralized validation utilities for langchain-snowflake package.

This module provides standardized validation patterns to reduce code duplication
and ensure consistent validation across the codebase.
"""

import os
from typing import Any, Dict, List, Optional

from ._error_handling import SnowflakeErrorHandler


class SnowflakeValidationUtils:
    """Centralized validation utilities for Snowflake integrations."""

    @staticmethod
    def validate_non_empty_string(value: Any, field_name: str) -> str:
        """Validate that a value is a non-empty string.

        Args:
            value: Value to validate
            field_name: Name of the field for error messages

        Returns:
            Validated and stripped string

        Raises:
            ValueError: If validation fails
        """
        try:
            if not value or not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            return value.strip()
        except Exception as e:
            SnowflakeErrorHandler.log_and_raise(e, f"validate {field_name}")

    @staticmethod
    def validate_required_env_vars(required_vars: List[str]) -> Dict[str, str]:
        """Validate that required environment variables are set.

        Args:
            required_vars: List of required environment variable names

        Returns:
            Dictionary mapping variable names to their values

        Raises:
            ValueError: If any required variables are missing
        """
        try:
            env_values = {}
            for var in required_vars:
                value = os.getenv(var)
                if not value:
                    raise ValueError(f"Missing required environment variable: {var}")
                env_values[var] = value
            return env_values
        except Exception as e:
            SnowflakeErrorHandler.log_and_raise(e, "validate required environment variables")

    @staticmethod
    def validate_connection_params(connection_params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate connection parameters for Snowflake session creation.

        Args:
            connection_params: Dictionary of connection parameters

        Returns:
            Validated connection parameters

        Raises:
            ValueError: If validation fails
        """
        try:
            required_params = ["account", "user"]

            # Check for required parameters
            for param in required_params:
                if param not in connection_params:
                    raise ValueError(f"Missing required connection parameter: {param}")

            # Validate account format
            account = connection_params["account"]
            if not isinstance(account, str) or not account.strip():
                raise ValueError("Account must be a non-empty string")

            # Validate user
            user = connection_params["user"]
            if not isinstance(user, str) or not user.strip():
                raise ValueError("User must be a non-empty string")

            # Ensure we have at least one authentication method
            auth_methods = ["password", "private_key", "private_key_path", "token"]
            if not any(param in connection_params for param in auth_methods):
                raise ValueError("At least one authentication method must be provided")

            return connection_params

        except Exception as e:
            SnowflakeErrorHandler.log_and_raise(e, "validate connection parameters")

    @staticmethod
    def validate_model_name(model: str) -> str:
        """Validate Snowflake model name for security and format.

        Args:
            model: Model name to validate

        Returns:
            Validated model name

        Raises:
            ValueError: If model name is invalid
        """
        try:
            if not model or not isinstance(model, str):
                raise ValueError("Model name must be a non-empty string")

            # Remove any potential SQL injection characters
            model = model.strip()
            if "'" in model or '"' in model or ";" in model:
                raise ValueError("Model name contains invalid characters")

            return model

        except Exception as e:
            SnowflakeErrorHandler.log_and_raise(e, "validate model name")

    @staticmethod
    def validate_service_name(service_name: str) -> tuple[str, str, str]:
        """Validate and parse fully qualified service name.

        Args:
            service_name: Fully qualified service name (database.schema.service)

        Returns:
            Tuple of (database, schema, service)

        Raises:
            ValueError: If service name format is invalid
        """
        try:
            parts = service_name.split(".")
            if len(parts) != 3:
                raise ValueError(f"Service name must be fully qualified (database.schema.service): {service_name}")
            return parts[0], parts[1], parts[2]

        except Exception as e:
            SnowflakeErrorHandler.log_and_raise(e, "validate service name format")

    @staticmethod
    def validate_url_scheme(url: str, expected_scheme: str) -> None:
        """Validate URL scheme.

        Args:
            url: URL to validate
            expected_scheme: Expected URL scheme

        Raises:
            ValueError: If URL scheme is invalid
        """
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            if parsed.scheme != expected_scheme:
                raise ValueError(f"URL must start with '{expected_scheme}://'")

        except Exception as e:
            SnowflakeErrorHandler.log_and_raise(e, f"validate URL scheme for {expected_scheme}")

    @staticmethod
    def validate_optional_env_vars(optional_vars: List[str]) -> Dict[str, Optional[str]]:
        """Get optional environment variables.

        Args:
            optional_vars: List of optional environment variable names

        Returns:
            Dictionary mapping variable names to their values (None if not set)
        """
        env_values = {}
        for var in optional_vars:
            env_values[var] = os.getenv(var)
        return env_values

    @staticmethod
    def validate_auth_requirements(
        account: Optional[str], user: Optional[str], auth_type: str, additional_required: Optional[List[str]] = None
    ) -> None:
        """Validate authentication requirements.

        Args:
            account: Snowflake account
            user: Snowflake user
            auth_type: Type of authentication (for error messages)
            additional_required: Additional required fields

        Raises:
            ValueError: If authentication requirements are not met
        """
        try:
            if not all([account, user]):
                raise ValueError(
                    f"{auth_type} authentication requires environment variables: SNOWFLAKE_ACCOUNT and SNOWFLAKE_USER"
                )

            if additional_required:
                for field in additional_required:
                    if not field:
                        raise ValueError(f"{auth_type} authentication requires additional field: {field}")

        except Exception as e:
            SnowflakeErrorHandler.log_and_raise(e, f"validate {auth_type} authentication requirements")

    @staticmethod
    def validate_package_dependency(package_name: str, install_command: str) -> None:
        """Validate that a required package is available.

        Args:
            package_name: Name of the package to check
            install_command: Installation command for error message

        Raises:
            ValueError: If package is not available
        """
        try:
            __import__(package_name)
        except ImportError:
            error_msg = f"{package_name} package is required. Install with: {install_command}"
            SnowflakeErrorHandler.log_and_raise(ValueError(error_msg), f"validate {package_name} package dependency")

    @staticmethod
    def validate_file_exists(file_path: str, file_description: str) -> None:
        """Validate that a file exists.

        Args:
            file_path: Path to the file
            file_description: Description of the file for error messages

        Raises:
            ValueError: If file does not exist
        """
        try:
            if not os.path.exists(file_path):
                raise ValueError(f"{file_description} not found: {file_path}")
        except Exception as e:
            SnowflakeErrorHandler.log_and_raise(e, f"validate {file_description} existence")
