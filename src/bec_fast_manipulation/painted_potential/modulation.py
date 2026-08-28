"""Normalized modulation functions for painted optical potentials."""

from __future__ import annotations

import numpy as np


def harmonic_painting_modulation(phase):
    """Return the normalized harmonic-painting position for a cycle phase.

    ``phase`` is periodic with period one. During the first half cycle the
    returned value moves monotonically from -1 to +1; during the second half it
    returns from +1 to -1. The branch is the real inverse of
    ``f - f**3 / 3`` on ``[-1, 1]``.
    """
    phase_array = np.asarray(phase, dtype=float)
    if not np.all(np.isfinite(phase_array)):
        raise ValueError("phase must contain only finite values.")

    normalized_phase = np.mod(phase_array, 1.0)
    first_half = normalized_phase < 0.5
    half_phase = np.where(first_half, 2.0 * normalized_phase, 2.0 * (normalized_phase - 0.5))
    target = np.where(first_half, -2.0 / 3.0 + (4.0 / 3.0) * half_phase, 2.0 / 3.0 - (4.0 / 3.0) * half_phase)
    result = 2.0 * np.sin(np.arcsin(1.5 * target) / 3.0)
    if phase_array.ndim == 0:
        return float(result)
    return result
