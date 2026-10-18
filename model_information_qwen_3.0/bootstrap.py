"""Run this versioned directory through the legal Python package alias.

Examples: ``bootstrap.py launcher --help``, ``bootstrap.py cli chat --text ...``,
and ``bootstrap.py unittest discover -s model_information_qwen_3.0/tests -v``.
Registering the package only imports its empty initializer, never a model.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import runpy
import sys


PACKAGE_NAME = "model_information_qwen_v3"
PACKAGE_ROOT = Path(__file__).resolve().parent


def register_package():
    """Make relative imports resolve to this directory without renaming it."""
    project = str(PACKAGE_ROOT.parent)
    if project not in sys.path:
        sys.path.insert(0, project)
    existing = sys.modules.get(PACKAGE_NAME)
    initializer = PACKAGE_ROOT / "__init__.py"
    if existing is not None:
        if Path(existing.__file__).resolve() != initializer:
            raise ImportError(f"{PACKAGE_NAME} already refers to a different directory")
        return existing
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME, initializer, submodule_search_locations=[str(PACKAGE_ROOT)])
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot register package at {PACKAGE_ROOT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[PACKAGE_NAME]
        raise
    return module


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print("Usage: bootstrap.py {cli|launcher|server|pi_client|tools.NAME|unittest|tests/FILE.py} [arguments]")
        return 0
    register_package()
    target, *arguments = args
    if target == "-c":
        if not arguments:
            raise SystemExit("bootstrap.py -c requires Python source")
        source, *arguments = arguments
        sys.argv = ["-c", *arguments]
        exec(compile(source, "<bootstrap -c>", "exec"), {"__name__": "__main__"})
        return 0
    if target == "unittest":
        module = "unittest"
    elif target == "cli":
        # cli.py exposes main(); the package entry is what invokes it.
        module = PACKAGE_NAME + ".__main__"
    else:
        if target.endswith(".py"):
            path = Path(target).expanduser()
            if not path.is_absolute():
                path = PACKAGE_ROOT / path
            path = path.resolve()
            if not path.is_relative_to(PACKAGE_ROOT) or not path.is_file():
                raise SystemExit("The requested Python entry must be inside this version directory")
            target = ".".join(path.relative_to(PACKAGE_ROOT).with_suffix("").parts)
        module = target if target.startswith(PACKAGE_NAME + ".") else PACKAGE_NAME + "." + target
    sys.argv = [target, *arguments]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
