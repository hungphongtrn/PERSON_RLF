import importlib


def parse_module_str(module: str):
    from_modules, imported = module.rsplit(".", 1)
    get_module = getattr(importlib.import_module(from_modules), imported)
    return get_module
