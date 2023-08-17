from __future__ import annotations

import ast
import importlib
import os
import pkgutil
from types import ModuleType

from zetta_utils import log

logger = log.get_logger("zetta_utils")


class NameOverlapError(Exception):
    """Custom exception for name overlaps in modules."""


def get_defined_objects(filepath: str) -> list[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=filepath)
        return [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        ]


def handle_name_overlap(
    module_name: str,
    filepath: str,
    name: str,
    cache: dict[str, ModuleType],
    package_module: ModuleType,
) -> list[str]:
    imports: set[str] = set()
    seen_objects: dict[int, list[str]] = {}
    module = cache.get(module_name)
    if module is None:
        return list(imports)
    obj = getattr(module, name, None)
    if not name.startswith("_") and (callable(obj) or isinstance(obj, type)):
        obj_address = id(obj)
        if module_name == package_module.__name__:
            seen_modules = seen_objects.setdefault(obj_address, [])
            if len(seen_modules) == 1 and seen_modules[0] == filepath:
                return list(imports)
            seen_modules.append(filepath)
            conflicting_module = seen_modules[0]
            raise NameOverlapError(
                f"Name overlap detected: {name}. Defined in {filepath} and {conflicting_module}."
            )
        imports.add(f"from {module.__name__} import {name}")
    return list(imports)


def get_all_objects(package: str, root_package: str, module: ModuleType | None = None) -> str:
    cache: dict[str, ModuleType] = {}
    imports: set[str] = set()
    try:
        package_module = importlib.import_module(package) if module is None else module
    except ImportError:
        logger.warning(f"Failed to import {package}.")
        return "\n".join(imports)
    if package_module.__file__:
        package_dir = os.path.dirname(package_module.__file__)
        for _, module_name, _ in pkgutil.walk_packages(
            [package_dir], prefix=f"{package_module.__name__}."
        ):
            if module_name == "zetta_utils.everything":
                continue
            try:
                module = cache.setdefault(module_name, importlib.import_module(module_name))
            except ImportError:
                continue
            if (
                module.__package__
                and module.__package__.startswith(root_package)
                and module.__file__
            ):
                filepath = module.__file__
                defined_objects = get_defined_objects(filepath)
                for name in dir(module):
                    if name in defined_objects:
                        imports.update(
                            handle_name_overlap(module_name, filepath, name, cache, package_module)
                        )
            if hasattr(module, "__path__"):
                imports.update(get_all_objects(package, root_package, module).split("\n"))
    return "\n".join(imports)


if __name__ == "__main__":
    logger.info("Generating import statements...")
    IMPORT_STATEMENTS = get_all_objects("zetta_utils", "zetta_utils")
    logger.info("Import statements generated successfully.")
    OUTPUT_FILE = "api.py"
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write("# pylint: disable=unused-import\n\n")
        file.write(IMPORT_STATEMENTS + "\n\n")
        file.write("set_verbosity(\"INFO\")\n")
        file.write("configure_logger()\n")
