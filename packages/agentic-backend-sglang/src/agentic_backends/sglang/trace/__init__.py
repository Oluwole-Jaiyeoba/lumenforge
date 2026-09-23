"""In-server SGLang tracing.  Import ``install`` lazily: it touches SGLang internals."""


def install() -> None:
    """Explicit entry point for the trace hooks (``sitecustomize`` calls this)."""

    from .patch import install_sglang_kv_trace

    install_sglang_kv_trace()
