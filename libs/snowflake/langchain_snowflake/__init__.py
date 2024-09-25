from importlib import metadata

from langchain_snowflake.retrievers import (
    CortexSearchRetriever,
    CortexSearchRetrieverError,
)

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "CortexSearchRetriever",
    "CortexSearchRetrieverError",
    "__version__",
]