#!/usr/bin/env python3
"""
scripts/generate_docs.py
=========================
Auto-generate API reference Markdown from nesy package docstrings.

Walks every public module under ``nesy/``, extracts class and function
docstrings, and writes Markdown files into ``docs/api_reference/``.

Usage:
    python scripts/generate_docs.py            # writes into docs/api_reference/
    python scripts/generate_docs.py -o build/  # custom output dir
"""
import argparse
import importlib
import inspect
import logging
import os
import pkgutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────
def _heading(text: str, level: int = 1) -> str:
    return f"{'#' * level} {text}\n"


def _code_block(code: str, lang: str = "python") -> str:
    return f"```{lang}\n{code}\n```\n"


def _format_signature(obj) -> str:
    """Return a clean signature string or empty string if not callable."""
    try:
        sig = inspect.signature(obj)
        return str(sig)
    except (ValueError, TypeError):
        return ""


def _collect_module_docs(module) -> str:
    """Produce Markdown for one module's public API."""
    lines: list[str] = []
    module_name = module.__name__

    # Module-level docstring
    if module.__doc__:
        lines.append(module.__doc__.strip())
        lines.append("")

    members = inspect.getmembers(module)
    # Filter to objects defined in *this* module
    own = [
        (name, obj)
        for name, obj in members
        if not name.startswith("_")
        and getattr(obj, "__module__", None) == module_name
    ]

    classes = [(n, o) for n, o in own if inspect.isclass(o)]
    functions = [(n, o) for n, o in own if inspect.isfunction(o)]

    for name, cls in classes:
        sig = _format_signature(cls)
        lines.append(_heading(f"class `{name}{sig}`", 3))
        if cls.__doc__:
            lines.append(cls.__doc__.strip())
            lines.append("")

        # Public methods
        for mname, mobj in inspect.getmembers(cls, predicate=inspect.isfunction):
            if mname.startswith("_") and mname != "__init__":
                continue
            msig = _format_signature(mobj)
            lines.append(_heading(f"`{mname}{msig}`", 4))
            if mobj.__doc__:
                lines.append(mobj.__doc__.strip())
                lines.append("")

    for name, func in functions:
        sig = _format_signature(func)
        lines.append(_heading(f"`{name}{sig}`", 3))
        if func.__doc__:
            lines.append(func.__doc__.strip())
            lines.append("")

    return "\n".join(lines)


# ── walker ───────────────────────────────────────────────────────────────
def walk_package(package_name: str) -> dict[str, str]:
    """Walk *package_name* recursively; return {module_name: markdown}."""
    result: dict[str, str] = {}
    pkg = importlib.import_module(package_name)
    pkg_path = getattr(pkg, "__path__", None)
    if pkg_path is None:
        return result

    for importer, modname, ispkg in pkgutil.walk_packages(
        pkg_path, prefix=package_name + "."
    ):
        try:
            mod = importlib.import_module(modname)
        except Exception as exc:
            logger.warning("Skipping %s: %s", modname, exc)
            continue
        md = _collect_module_docs(mod)
        if md.strip():
            result[modname] = md

    return result


# ── main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate NeSy-Core API docs")
    parser.add_argument(
        "-o", "--output-dir",
        default="docs/api_reference",
        help="Output directory for Markdown files (default: docs/api_reference/)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Walking nesy package tree …")
    docs = walk_package("nesy")

    for modname, md in docs.items():
        # nesy.symbolic.engine  → symbolic_engine.md
        short = modname.replace("nesy.", "").replace(".", "_") + ".md"
        target = out / short
        target.write_text(f"# {modname}\n\n{md}\n", encoding="utf-8")
        logger.info("Wrote %s", target)

    logger.info("Done – %d modules documented.", len(docs))


if __name__ == "__main__":
    main()
