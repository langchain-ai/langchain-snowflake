import os
import re
from importlib import metadata
from typing import Optional
from urllib.parse import parse_qs, urlparse

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from snowflake.snowpark import Session

from ._connection.session_manager import SnowflakeSessionManager
from ._error_handling import SnowflakeErrorHandler
from ._validation_utils import SnowflakeValidationUtils

# Agents - Snowflake Cortex Agents
from .agents import (
    # Schemas
    AgentCreateInput,
    AgentUpdateInput,
    EnhancedAgentInput,
    FeedbackInput,
    FeedbackOutput,
    SnowflakeCortexAgent,
    ThreadUpdateInput,
)

# Chat Models - Cortex Complete
from .chat_models import ChatSnowflake

# Document Formatters - RAG Utilities
from .formatters import (
    format_cortex_search_documents,
)

# MCP Integration
from .mcp_integration import (
    MCPToolWrapper,
    bind_mcp_tools,
    create_langchain_tool_from_mcp,
    filter_compatible_mcp_tools,
)

# Retrievers - Cortex Search
from .retrievers import SnowflakeCortexSearchRetriever

# Tools - Cortex Functions
from .tools import (
    CortexCompleteTool,
    CortexSentimentTool,
    CortexSummarizerTool,
    CortexTranslatorTool,
    SnowflakeCortexAnalyst,
    SnowflakeQueryTool,
)


def create_session_from_env() -> Session:
    """Create a Snowflake session from environment variables.

    Expected environment variables:
    - SNOWFLAKE_ACCOUNT: Account identifier
    - SNOWFLAKE_USER: Username
    - SNOWFLAKE_PASSWORD: Password
    - SNOWFLAKE_WAREHOUSE: Warehouse (optional)
    - SNOWFLAKE_DATABASE: Database (optional)
    - SNOWFLAKE_SCHEMA: Schema (optional)

    Returns:
        Configured Snowflake session

    Raises:
        ValueError: If required environment variables are missing
    """
    connection_params = {}

    # Use centralized validation utilities
    required_vars = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"]
    optional_vars = ["SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA"]

    # Validate required environment variables
    env_values = SnowflakeValidationUtils.validate_required_env_vars(required_vars)
    for var, value in env_values.items():
        connection_params[var.lower().replace("snowflake_", "")] = value

    # Get optional environment variables
    optional_values = SnowflakeValidationUtils.validate_optional_env_vars(optional_vars)
    for var, value in optional_values.items():
        if value:
            connection_params[var.lower().replace("snowflake_", "")] = value

    return SnowflakeSessionManager.create_session(connection_params)


def create_session_from_connection_string() -> Session:
    """Create a Snowflake session from a connection string in environment variables.

    Reads connection string from SNOWFLAKE_CONNECTION_STRING environment variable.
    Supports environment variable substitution using ${VARIABLE_NAME} syntax.

    Required environment variable:
    - SNOWFLAKE_CONNECTION_STRING: Connection string with format
      snowflake://user:password@account/database/schema?warehouse=warehouse_name

    Example connection string with environment variable substitution:
    SNOWFLAKE_CONNECTION_STRING="snowflake://${SNOWFLAKE_USER}:${SNOWFLAKE_PASSWORD}@${SNOWFLAKE_ACCOUNT}/${SNOWFLAKE_DATABASE}/${SNOWFLAKE_SCHEMA}?warehouse=${SNOWFLAKE_WAREHOUSE}"

    Returns:
        Configured Snowflake session

    Raises:
        ValueError: If connection string format is invalid or environment variable
            missing
    """
    try:
        # Get connection string from environment variable
        connection_string = os.getenv("SNOWFLAKE_CONNECTION_STRING")
        if not connection_string:
            raise ValueError(
                "Connection string authentication requires environment variable: SNOWFLAKE_CONNECTION_STRING"
            )

        # Substitute environment variables in the connection string
        # Pattern: ${VAR_NAME} or $VAR_NAME
        def replace_env_var(match):
            var_name = match.group(1) if match.group(1) else match.group(2)
            value = os.getenv(var_name)
            if value is None:
                raise ValueError(f"Environment variable {var_name} referenced in connection string but not set")
            return value

        # Replace ${VAR} and $VAR patterns
        connection_string = re.sub(r"\$\{([^}]+)\}|\$([A-Z_][A-Z0-9_]*)", replace_env_var, connection_string)

        parsed = urlparse(connection_string)

        # Use centralized validation utilities
        SnowflakeValidationUtils.validate_url_scheme(connection_string, "snowflake")

        connection_params = {
            "account": parsed.hostname,
            "user": parsed.username,
            "password": parsed.password,
        }

        # Extract database and schema from path
        path_parts = [p for p in parsed.path.split("/") if p]
        if len(path_parts) >= 1:
            connection_params["database"] = path_parts[0]
        if len(path_parts) >= 2:
            connection_params["schema"] = path_parts[1]

        # Extract warehouse and other parameters from query string
        query_params = parse_qs(parsed.query)
        for key, values in query_params.items():
            if values:
                connection_params[key] = values[0]

        return SnowflakeSessionManager.create_session(connection_params)

    except Exception as e:
        SnowflakeErrorHandler.log_and_raise(e, "create session from connection string")


def create_session_from_pat() -> Session:
    """Create a Snowflake session using Personal Access Token (PAT) from
    environment variables.

    Reads all credentials from environment variables:
    - SNOWFLAKE_ACCOUNT: Account identifier (required)
    - SNOWFLAKE_USER: Username (required)
    - SNOWFLAKE_PAT: Personal Access Token (required)
    - SNOWFLAKE_WAREHOUSE: Warehouse (optional)
    - SNOWFLAKE_DATABASE: Database (optional)
    - SNOWFLAKE_SCHEMA: Schema (optional)

    Returns:
        Configured Snowflake session

    Raises:
        ValueError: If required environment variables are missing
    """
    # Read all credentials from environment variables only
    # Use centralized validation utilities
    required_vars = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PAT"]
    optional_vars = ["SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA"]

    # Validate required environment variables
    env_values = SnowflakeValidationUtils.validate_required_env_vars(required_vars)
    account = env_values["SNOWFLAKE_ACCOUNT"]
    user = env_values["SNOWFLAKE_USER"]
    token = env_values["SNOWFLAKE_PAT"]

    # Validate authentication requirements
    SnowflakeValidationUtils.validate_auth_requirements(account, user, "PAT", additional_required=[token])

    # Get optional environment variables
    optional_values = SnowflakeValidationUtils.validate_optional_env_vars(optional_vars)
    warehouse = optional_values.get("SNOWFLAKE_WAREHOUSE")
    database = optional_values.get("SNOWFLAKE_DATABASE")
    schema = optional_values.get("SNOWFLAKE_SCHEMA")

    connection_params = {
        "account": account,
        "user": user,
        "password": token,  # PAT is used as password
    }

    # Add optional parameters
    if warehouse:
        connection_params["warehouse"] = warehouse
    if database:
        connection_params["database"] = database
    if schema:
        connection_params["schema"] = schema

    return SnowflakeSessionManager.create_session(connection_params)


def create_session_from_key_pair() -> Session:
    """Create a Snowflake session using RSA key pair authentication from
    environment variables.

    Reads all credentials from environment variables:
    - SNOWFLAKE_ACCOUNT: Account identifier (required)
    - SNOWFLAKE_USER: Username (required)
    - SNOWFLAKE_PRIVATE_KEY_PATH: Path to private key file (required)
    - SNOWFLAKE_PRIVATE_KEY_PASSPHRASE: Passphrase for encrypted private key (optional)
    - SNOWFLAKE_WAREHOUSE: Warehouse (optional)
    - SNOWFLAKE_DATABASE: Database (optional)
    - SNOWFLAKE_SCHEMA: Schema (optional)

    Returns:
        Configured Snowflake session

    Raises:
        ValueError: If required environment variables are missing or key is invalid
    """
    # Use centralized validation utilities
    required_vars = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PRIVATE_KEY_PATH"]
    optional_vars = [
        "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE",
        "SNOWFLAKE_WAREHOUSE",
        "SNOWFLAKE_DATABASE",
        "SNOWFLAKE_SCHEMA",
    ]

    # Validate required environment variables
    env_values = SnowflakeValidationUtils.validate_required_env_vars(required_vars)
    account = env_values["SNOWFLAKE_ACCOUNT"]
    user = env_values["SNOWFLAKE_USER"]
    private_key_path = env_values["SNOWFLAKE_PRIVATE_KEY_PATH"]

    # Validate authentication requirements
    SnowflakeValidationUtils.validate_auth_requirements(
        account, user, "Key pair", additional_required=[private_key_path]
    )

    # Validate file exists
    SnowflakeValidationUtils.validate_file_exists(private_key_path, "Private key file")

    # Get optional environment variables
    optional_values = SnowflakeValidationUtils.validate_optional_env_vars(optional_vars)
    private_key_passphrase = optional_values.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")
    warehouse = optional_values.get("SNOWFLAKE_WAREHOUSE")
    database = optional_values.get("SNOWFLAKE_DATABASE")
    schema = optional_values.get("SNOWFLAKE_SCHEMA")

    # Validate cryptography package dependency
    SnowflakeValidationUtils.validate_package_dependency("cryptography", "pip install cryptography")

    try:
        # Load private key from file
        with open(private_key_path, "rb") as key_file:
            private_key_data = key_file.read()

        # Parse private key
        passphrase_bytes = private_key_passphrase.encode("utf-8") if private_key_passphrase else None
        private_key = serialization.load_pem_private_key(
            private_key_data, password=passphrase_bytes, backend=default_backend()
        )

        # Get DER-encoded private key
        private_key_der = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        connection_params = {
            "account": account,
            "user": user,
            "private_key": private_key_der,
        }

        # Add optional parameters
        if warehouse:
            connection_params["warehouse"] = warehouse
        if database:
            connection_params["database"] = database
        if schema:
            connection_params["schema"] = schema

        return SnowflakeSessionManager.create_session(connection_params)

    except Exception as e:
        SnowflakeErrorHandler.log_and_raise(e, "create session with key pair")


def get_default_session() -> Optional[Session]:
    """Get a default Snowflake session with smart fallback strategy.

    Tries to create a session in the following priority order:
    1. Key pair authentication (most secure)
    2. PAT authentication (recommended for production)
    3. Password authentication (development/legacy)
    4. Connection string authentication

    All methods read from the same SNOWFLAKE_* environment variables.

    Returns:
        Snowflake session or None if no valid credentials found
    """
    # Try key pair authentication first (most secure)
    try:
        return create_session_from_key_pair()
    except ValueError as e:
        SnowflakeErrorHandler.log_error("key pair authentication attempt", e)

    # Try PAT authentication (recommended for production)
    try:
        return create_session_from_pat()
    except ValueError as e:
        SnowflakeErrorHandler.log_error("PAT authentication attempt", e)

    # Try password authentication (development/legacy)
    try:
        return create_session_from_env()
    except ValueError as e:
        SnowflakeErrorHandler.log_error("password authentication attempt", e)

    # Try connection string as last resort
    try:
        return create_session_from_connection_string()
    except ValueError as e:
        SnowflakeErrorHandler.log_error("connection string authentication attempt", e)

    return None


try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    # Chat Models
    "ChatSnowflake",
    # Retrievers
    "SnowflakeCortexSearchRetriever",
    # Tools
    "SnowflakeQueryTool",
    "CortexSentimentTool",
    "CortexSummarizerTool",
    "CortexTranslatorTool",
    "CortexCompleteTool",
    "SnowflakeCortexAnalyst",
    # Snowflake Managed Cortex Agent
    "SnowflakeCortexAgent",
    # Agent schemas
    "AgentCreateInput",
    "AgentUpdateInput",
    "EnhancedAgentInput",
    "ThreadUpdateInput",
    "FeedbackInput",
    "FeedbackOutput",
    # MCP Integration
    "MCPToolWrapper",
    "create_langchain_tool_from_mcp",
    "filter_compatible_mcp_tools",
    "bind_mcp_tools",
    # Document Formatters
    "format_cortex_search_documents",
    # Authentication utilities
    "create_session_from_env",
    "create_session_from_connection_string",
    "create_session_from_pat",
    "create_session_from_key_pair",
    "get_default_session",
    # Version
    "__version__",
]
