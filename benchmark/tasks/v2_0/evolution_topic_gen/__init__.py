"""
Evolution Topic Gen Task Module
"""

from .config import EVOLUTION_TOPIC_GEN_CONFIG
def __getattr__(name):
    if name == "utils":
        from importlib import import_module
        return import_module(".utils", __name__)
    raise AttributeError(name)

__all__ = [
    "EVOLUTION_TOPIC_GEN_CONFIG",
    "utils",
]
