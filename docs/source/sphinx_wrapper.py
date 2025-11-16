#!/usr/bin/env python
"""
Wrapper script for sphinx-build that imports tensorstore/neuroglancer first
to prevent absl timezone initialization errors.
"""

# Print immediately to stderr before ANY imports
import sys
print("=" * 80, file=sys.stderr, flush=True)
print("DEBUG: Sphinx wrapper script started - VERY FIRST LINE", file=sys.stderr, flush=True)
print(f"DEBUG: Python version: {sys.version}", file=sys.stderr, flush=True)
print(f"DEBUG: sys.argv: {sys.argv}", file=sys.stderr, flush=True)
print("=" * 80, file=sys.stderr, flush=True)

import os

# Set environment variable
os.environ.setdefault('TZ', 'UTC')
print("DEBUG: Set TZ environment variable", file=sys.stderr, flush=True)

# Check if anything has already imported tensorstore
print(f"DEBUG: 'tensorstore' in sys.modules: {'tensorstore' in sys.modules}", file=sys.stderr, flush=True)
print(f"DEBUG: 'neuroglancer' in sys.modules: {'neuroglancer' in sys.modules}", file=sys.stderr, flush=True)

# Check numpy and ml_dtypes versions before importing tensorstore
print("DEBUG: Checking package versions...", file=sys.stderr, flush=True)
try:
    import numpy as np
    print(f"DEBUG: numpy version: {np.__version__}", file=sys.stderr, flush=True)
    print(f"DEBUG: numpy.__file__: {np.__file__}", file=sys.stderr, flush=True)
except ImportError as e:
    print(f"DEBUG: numpy not available: {e}", file=sys.stderr, flush=True)

try:
    import ml_dtypes
    print(f"DEBUG: ml_dtypes version: {ml_dtypes.__version__}", file=sys.stderr, flush=True)
    print(f"DEBUG: ml_dtypes.__file__: {ml_dtypes.__file__}", file=sys.stderr, flush=True)
except ImportError as e:
    print(f"DEBUG: ml_dtypes not available: {e}", file=sys.stderr, flush=True)
except AttributeError:
    print(f"DEBUG: ml_dtypes has no __version__", file=sys.stderr, flush=True)

# Import tensorstore FIRST
print("DEBUG: About to import tensorstore...", file=sys.stderr, flush=True)
try:
    import tensorstore
    print("DEBUG: tensorstore module imported", file=sys.stderr, flush=True)
    print(f"DEBUG: tensorstore.__file__: {tensorstore.__file__}", file=sys.stderr, flush=True)
    try:
        print(f"DEBUG: tensorstore version: {tensorstore.__version__}", file=sys.stderr, flush=True)
    except AttributeError:
        print("DEBUG: tensorstore has no __version__", file=sys.stderr, flush=True)

    # Force load C++ extension
    print("DEBUG: About to access tensorstore._tensorstore...", file=sys.stderr, flush=True)
    _ = tensorstore._tensorstore
    print("DEBUG: tensorstore._tensorstore accessed successfully", file=sys.stderr, flush=True)
except (ImportError, AttributeError, RecursionError) as e:
    print(f"DEBUG: Tensorstore import/access failed: {e}", file=sys.stderr, flush=True)
    print(f"DEBUG: Error type: {type(e).__name__}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)

# Import neuroglancer
print("DEBUG: About to import neuroglancer...", file=sys.stderr, flush=True)
try:
    import neuroglancer
    print("DEBUG: neuroglancer imported successfully", file=sys.stderr, flush=True)
    print(f"DEBUG: neuroglancer.__file__: {neuroglancer.__file__}", file=sys.stderr, flush=True)
except (ImportError, RecursionError) as e:
    print(f"DEBUG: Neuroglancer import failed: {e}", file=sys.stderr, flush=True)
    print(f"DEBUG: Error type: {type(e).__name__}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)

# Now run sphinx-build
print("DEBUG: About to import sphinx.cmd.build...", file=sys.stderr, flush=True)
try:
    from sphinx.cmd.build import main
    print("DEBUG: sphinx.cmd.build.main imported successfully", file=sys.stderr, flush=True)
except ImportError as e:
    print(f"DEBUG: Sphinx import failed: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    print("DEBUG: Running sphinx main with args:", sys.argv[1:], file=sys.stderr, flush=True)
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as e:
        print(f"DEBUG: Sphinx main failed with exception: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
