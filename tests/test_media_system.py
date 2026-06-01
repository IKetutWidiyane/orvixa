"""
Tests for floating media loading and rendering.
"""

import cv2
import numpy as np

from core.rendering.renderer import Renderer
from core.workspace.manager import VirtualWorkspace
from media.image.image_manager import ImageManager


def test_image_media_loads_and_renders_pixels(tmp_path):
    image_path = tmp_path / "sample.png"
    sample = np.zeros((40, 60, 3), dtype=np.uint8)
    sample[:] = (0, 0, 255)
    cv2.imwrite(str(image_path), sample)

    manager = ImageManager()
    image_obj = manager.load_image("image_1", str(image_path), 0, 0)

    assert image_obj is not None
    assert image_obj.media_type == "image"

    workspace = VirtualWorkspace(width=800, height=600)
    assert workspace.add_object(image_obj)

    renderer = Renderer(800, 600)
    renderer.clear()
    renderer.render_object(image_obj, workspace)

    center_pixel = renderer.frame[300, 400]
    assert center_pixel[2] > 200
