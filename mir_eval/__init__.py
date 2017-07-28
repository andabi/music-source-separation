#!/usr/bin/env python
"""Top-level module for mir_eval"""

# Import all submodules (for each task)
from . import beat
from . import chord
from . import io
from . import onset
from . import segment
from . import separation
from . import util
from . import sonify
from . import melody
from . import multipitch
from . import pattern
from . import tempo
from . import hierarchy
from . import transcription
from . import key

__version__ = '0.4'
