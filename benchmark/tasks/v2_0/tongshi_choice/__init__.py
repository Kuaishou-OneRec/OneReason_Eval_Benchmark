"""
Tongshi Choice (通识选择题) Task Module
"""

from .config import TONGSHI_CHOICE_CONFIG
def __getattr__(name):
    if name == "utils":
        from importlib import import_module
        return import_module(".utils", __name__)
    raise AttributeError(name)

__all__ = [
    "TONGSHI_CHOICE_CONFIG",
    "utils",
]
