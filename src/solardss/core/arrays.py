"""Numpy array aliases used in public and internal signatures."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]
IntArray: TypeAlias = npt.NDArray[np.int16]
AnyArray: TypeAlias = npt.NDArray[np.generic]
