# tests/test_smoother.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smoother import PositionSmoother


def test_first_call_seeds_state():
    s = PositionSmoother(alpha=0.5)
    x, y, sc = s.smooth(100, 200, 30)
    assert x == 100 and y == 200 and sc == 30


def test_ema_converges_to_target():
    s = PositionSmoother(alpha=0.5)
    # Drive toward (0, 0, 0) from (100, 100, 100)
    s.smooth(100, 100, 100)
    for _ in range(40):
        x, y, sc = s.smooth(0, 0, 0)
    assert x < 0.01 and y < 0.01 and sc < 0.01


def test_reset_clears_state():
    s = PositionSmoother(alpha=0.5)
    s.smooth(50, 50, 50)
    s.reset()
    # After reset, next call should seed directly
    x, y, sc = s.smooth(10, 20, 5)
    assert x == 10 and y == 20 and sc == 5


def test_alpha_1_is_passthrough():
    s = PositionSmoother(alpha=1.0)
    s.smooth(999, 999, 999)
    x, y, sc = s.smooth(42, 43, 44)
    assert x == 42 and y == 43 and sc == 44


def test_alpha_0_holds_initial_value():
    s = PositionSmoother(alpha=0.0)
    s.smooth(10, 20, 30)
    x, y, sc = s.smooth(99, 99, 99)
    # alpha=0 means no update, stays at seed
    assert x == 10 and y == 20 and sc == 30
