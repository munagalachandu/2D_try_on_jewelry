# ---------------------------------------------------------------------------
# smoother.py — Exponential Moving Average smoothing for jewelry placement
# ---------------------------------------------------------------------------

from config import SMOOTHING_ALPHA


class PositionSmoother:
    """
    Applies Exponential Moving Average (EMA) smoothing to (x, y, scale)
    for a single jewelry piece, preventing jittery placement.
    """

    def __init__(self, alpha: float = SMOOTHING_ALPHA):
        self.alpha = alpha
        self._x: float | None = None
        self._y: float | None = None
        self._scale: float | None = None

    def smooth(self, x: float, y: float, scale: float) -> tuple[float, float, float]:
        """Return EMA-smoothed (x, y, scale). First call seeds the state."""
        if self._x is None:
            self._x, self._y, self._scale = x, y, scale
        else:
            a = self.alpha
            self._x     = a * x     + (1 - a) * self._x
            self._y     = a * y     + (1 - a) * self._y
            self._scale = a * scale + (1 - a) * self._scale
        return self._x, self._y, self._scale

    def reset(self):
        """Call this when a face is lost so stale state is discarded."""
        self._x = self._y = self._scale = None
