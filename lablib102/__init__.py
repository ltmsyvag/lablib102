from .core import *

## 配合 toml 文件中的设置后, __version__直接就是 githash
from importlib.metadata import version
__version__ = version('lablib102')