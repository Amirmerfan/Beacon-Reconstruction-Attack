import sys
import numpy as np
import scipy
import matplotlib
import pandas

try:
    import torch
    torch_version = torch.__version__
except ImportError:
    torch_version = "Torch not installed or import failed"

argparse_version = "Builtin (part of Python standard library)"

print("Python Version:", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
print("numpy Version:", np.__version__)
print("scipy Version:", scipy.__version__)
print("matplotlib Version:", matplotlib.__version__)
print("torch Version:", torch_version)
print("pandas Version:", pandas.__version__)
print("argparse Version:", argparse_version)
