"""Teacher and student model wrappers for the evaluation harness."""

from .pillar0 import Pillar0
from .tangerine_vit import TangerineVit
from .ctclip_zero_shot import CTClipZeroShot
from .curia1 import Curia1
from .curia2 import Curia2
from .student import MuCoDiStudent
from .random_features import RandomFeatures3D

from . import pillar0
from . import tangerine_vit
from . import ctclip_zero_shot
from . import curia1
from . import curia2
from . import student
from . import random_features

__all__ = [
    "Pillar0", "TangerineVit", "CTClipZeroShot", "Curia1", "Curia2",
    "MuCoDiStudent", "RandomFeatures3D",
]
