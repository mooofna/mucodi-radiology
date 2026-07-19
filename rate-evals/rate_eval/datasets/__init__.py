"""Dataset classes for the RATE evaluation pipeline."""

from .nifti import NiftiCTDataset
from ._core.lidc_base import LIDCBaseDataset

# shared per-wrapper NIfTI recipes (chest-CT cohorts)
from .nifti_recipes import NiftiForCTClip, NiftiForTangerine, NiftiForPillar0, NiftiForCuria

# bespoke CT-RATE loaders (real-HU metadata rescale, per wrapper)
from .ctrate.for_pillar0 import CTRateForPillar0
from .ctrate.for_tangerine import CTRateForTangerine
from .ctrate.for_ctclip import CTRateForCTClip
from .ctrate.for_curia import CTRateForCuria

# bespoke RAD-ChestCT loaders (NPZ, per wrapper)
from .radchestct.for_ctclip import RADChestCTForCTClip
from .radchestct.for_tangerine import RADChestCTForTangerine
from .radchestct.for_pillar0 import RADChestCTForPillar0

# STOIC2021 (.mha) loaders
from .stoic2021.for_ctclip import STOIC2021ForCTClip
from .stoic2021.for_pillar0 import STOIC2021ForPillar0
from .stoic2021.for_curia import STOIC2021ForCuria
from .stoic2021.for_mucodi_student import STOIC2021ForMuCoDiStudent

__all__ = [
    "NiftiCTDataset",
    "LIDCBaseDataset",
    "NiftiForCTClip",
    "NiftiForTangerine",
    "NiftiForPillar0",
    "NiftiForCuria",
    "CTRateForPillar0",
    "CTRateForTangerine",
    "CTRateForCTClip",
    "CTRateForCuria",
    "RADChestCTForCTClip",
    "RADChestCTForTangerine",
    "RADChestCTForPillar0",
    "STOIC2021ForCTClip",
    "STOIC2021ForPillar0",
    "STOIC2021ForCuria",
    "STOIC2021ForMuCoDiStudent",
]
