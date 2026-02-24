"""Calculator auto-registration registry."""

from collections import OrderedDict

REGISTRY = OrderedDict()


def register(calc_id, *, description, short_desc, group="expression",
             examples=None, i18n=None):
    """Decorator to register a calculator function with its metadata."""
    if examples is None:
        examples = []
    if i18n is None:
        i18n = {}

    def decorator(func):
        REGISTRY[calc_id] = {
            "function": func,
            "description": description,
            "short_desc": short_desc,
            "group": group,
            "examples": examples,
            "i18n": i18n,
        }
        return func

    return decorator
