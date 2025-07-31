import perceval as pcvl

from dataclasses import dataclass
from typing import List


@dataclass
class FeatureMapParams:
    circuit: pcvl.Circuit
    input_state: str
    min_detect_photon: int
