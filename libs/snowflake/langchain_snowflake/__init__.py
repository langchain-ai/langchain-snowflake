# Note: Agent and workflow examples are now available in docs/examples/
# - docs/examples/snowflake_agents.ipynb for basic agent patterns
# - docs/examples/langgraph_workflows.ipynb for advanced LangGraph workflows
# Authentication utilities
import os
from importlib import metadata
from typing import Dict, Optional

from pydantic import SecretStr
from snowflake.snowpark import Session

# Chat Models - Cortex Complete
from .chat_models import ChatSnowflake

# Document Formatters - RAG Utilities
from .formatters import (
    format_cortex_search_documents,
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

    # Required parameters
    required_vars = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"]
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            raise ValueError(f"Missing required environment variable: {var}")
        connection_params[var.lower().replace("snowflake_", "")] = value

    # Optional parameters
    optional_vars = ["SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA"]
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            connection_params[var.lower().replace("snowflake_", "")] = value

    return Session.builder.configs(connection_params).create()


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
        ValueError: If connection string format is invalid or environment variable missing
    """
    import re
    from urllib.parse import parse_qs, urlparse

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
            raise ValueError(
                f"Environment variable {var_name} referenced in connection string but not set"
            )
        return value

    # Replace ${VAR} and $VAR patterns
    connection_string = re.sub(
        r"\$\{([^}]+)\}|\$([A-Z_][A-Z0-9_]*)", replace_env_var, connection_string
    )

    try:
        parsed = urlparse(connection_string)

        if parsed.scheme != "snowflake":
            raise ValueError("Connection string must start with 'snowflake://'")

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

        return Session.builder.configs(connection_params).create()

    except Exception as e:
        raise ValueError(f"Invalid connection string format: {e}")


def create_session_from_pat() -> Session:
    """Create a Snowflake session using Personal Access Token (PAT) from environment variables.

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
    account = os.getenv("SNOWFLAKE_ACCOUNT")
    user = os.getenv("SNOWFLAKE_USER")
    token = os.getenv("SNOWFLAKE_PAT")
    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
    database = os.getenv("SNOWFLAKE_DATABASE")
    schema = os.getenv("SNOWFLAKE_SCHEMA")

    if not all([account, user, token]):
        raise ValueError(
            "PAT authentication requires environment variables: SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, and SNOWFLAKE_PAT"
        )

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

    try:
        return Session.builder.configs(connection_params).create()
    except Exception as e:
        raise ValueError(f"Failed to create session with PAT: {e}")


def create_session_from_key_pair() -> Session:
    """Create a Snowflake session using RSA key pair authentication from environment variables.

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
    # Read all credentials from environment variables only
    account = os.getenv("SNOWFLAKE_ACCOUNT")
    user = os.getenv("SNOWFLAKE_USER")
    private_key_path = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH")
    private_key_passphrase = os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")
    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
    database = os.getenv("SNOWFLAKE_DATABASE")
    schema = os.getenv("SNOWFLAKE_SCHEMA")

    if not all([account, user]):
        raise ValueError(
            "Key pair authentication requires environment variables: SNOWFLAKE_ACCOUNT and SNOWFLAKE_USER"
        )

    if not private_key_path:
        raise ValueError(
            "Key pair authentication requires environment variable: SNOWFLAKE_PRIVATE_KEY_PATH"
        )

    try:
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization

        # Load private key from file
        with open(private_key_path, "rb") as key_file:
            private_key_data = key_file.read()

        # Parse private key
        passphrase_bytes = (
            private_key_passphrase.encode("utf-8") if private_key_passphrase else None
        )
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

        return Session.builder.configs(connection_params).create()

    except ImportError:
        raise ValueError(
            "cryptography package is required for key pair authentication. "
            "Install with: pip install cryptography"
        )
    except FileNotFoundError:
        raise ValueError(f"Private key file not found: {private_key_path}")
    except Exception as e:
        raise ValueError(f"Failed to create session with key pair: {e}")


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
    except ValueError:
        pass

    # Try PAT authentication (recommended for production)
    try:
        return create_session_from_pat()
    except ValueError:
        pass

    # Try password authentication (development/legacy)
    try:
        return create_session_from_env()
    except ValueError:
        pass

    # Try connection string as last resort
    try:
        return create_session_from_connection_string()
    except ValueError:
        pass

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
    # Document Formatters
    "format_cortex_search_documents",
    # Note: Agents and workflows moved to docs/examples/ following LangChain partner package standards
    # Authentication utilities
    "create_session_from_env",
    "create_session_from_connection_string",
    "create_session_from_pat",
    "create_session_from_key_pair",
    "get_default_session",
    # Version
    "__version__",
]
