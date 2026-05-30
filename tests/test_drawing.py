"""
Tests for the air drawing engine.
"""

import numpy as np

from core.effects.drawing import DrawingEngine


def test_drawing_engine_draws_visible_stroke():
    engine = DrawingEngine(120, 80)

    engine.start_drawing(10, 10)
    engine.draw_at(80, 40)
    engine.stop_drawing()

    frame = engine.get_frame()

    assert np.count_nonzero(frame) > 0


def test_drawing_engine_eraser_removes_stroke_pixels():
    engine = DrawingEngine(120, 80)

    engine.set_brush_size(12)
    engine.start_drawing(10, 20)
    engine.draw_at(100, 20)
    engine.stop_drawing()

    before = np.count_nonzero(engine.get_frame())

    engine.toggle_eraser()
    engine.start_drawing(10, 20)
    engine.draw_at(100, 20)
    engine.stop_drawing()

    after = np.count_nonzero(engine.get_frame())

    assert after < before
