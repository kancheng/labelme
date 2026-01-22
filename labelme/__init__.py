import importlib.metadata
import logging

__appname__ = "Labelme"

# Semantic Versioning 2.0.0: https://semver.org/
# 1. MAJOR version when you make incompatible API changes;
# 2. MINOR version when you add functionality in a backwards-compatible manner;
# 3. PATCH version when you make backwards-compatible bug fixes.
# e.g., 1.0.0a0, 1.0.0a1, 1.0.0b0, 1.0.0rc0, 1.0.0, 1.0.0.post0
__version__ = importlib.metadata.version("labelme")

# XXX: has to be imported before PyQt5 to load dlls in order on Windows
# https://github.com/wkentaro/labelme/issues/1564
try:
    import onnxruntime
except (ImportError, OSError, RuntimeError) as e:
    # If onnxruntime fails to load (e.g., missing DLLs on Windows),
    # log a warning but continue. AI features may not work, but other
    # labelme functionality should still be available.
    import warnings
    warnings.warn(
        f"Failed to import onnxruntime: {e}. "
        "AI-assisted annotation features may not be available. "
        "If you need these features, please ensure Visual C++ Redistributable "
        "is installed and onnxruntime is properly configured.",
        ImportWarning,
        stacklevel=2,
    )
    onnxruntime = None  # type: ignore[assignment]

from labelme import testing
from labelme import utils
from labelme._label_file import LabelFile
