"""OneReason Eval Benchmark public interfaces."""
from importlib import import_module
__version__ = "0.1.0"
__all__ = ["Benchmark", "Generator", "GenerationRunner"]
_MODULES = {"Benchmark": ".benchmark", "Generator": ".base_generator", "GenerationRunner": ".generation_runner"}
def __getattr__(name):
    if name in _MODULES:
        return getattr(import_module(_MODULES[name], __name__), name)
    raise AttributeError(name)
