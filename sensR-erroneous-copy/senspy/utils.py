from __future__ import annotations

try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError
except ImportError:  # pragma: no cover
    from importlib_metadata import version as _pkg_version, PackageNotFoundError  # type: ignore

__all__ = ["has_jax", "version"]

__version__ = "0.0.1"


def has_jax() -> bool:
    """Return True if JAX is importable."""
    try:
        import jax  # type: ignore
    except Exception:  # pragma: no cover - environment dependent
        return False
    else:
        return True


def version() -> str:
    """Return the installed senspy version."""
    try:
        return _pkg_version("senspy")
    except PackageNotFoundError:  # pragma: no cover - not installed
        return __version__
