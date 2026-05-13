import os
from functools import cache

from langfuse import Langfuse

DEFAULT_FLUSH_AT = 200
DEFAULT_FLUSH_INTERVAL_S = 5.0


@cache
def init_langfuse_client() -> Langfuse:
    """Construct (or return) the process-wide Langfuse client with our flush settings.

    Must be the first Langfuse construction in the process — if anything calls
    bare `Langfuse()` or `get_client()` first, the SDK's resource-manager
    singleton locks in default flush values and ours are silently ignored.
    """
    flush_at = int(os.environ.get("LANGFUSE_FLUSH_AT") or DEFAULT_FLUSH_AT)
    flush_interval = float(
        os.environ.get("LANGFUSE_FLUSH_INTERVAL") or DEFAULT_FLUSH_INTERVAL_S
    )
    return Langfuse(flush_at=flush_at, flush_interval=flush_interval)
