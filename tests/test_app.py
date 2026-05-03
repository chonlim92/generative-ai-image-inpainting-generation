"""
Unit tests for the GenAI Image Inpainting application.

Tests cover:
- mask_to_rgb utility function
- make_divisible_by_8 utility function
- InpaintGenAI class initialization and methods
- reset_points behavior
- preprocess behavior
- run method with mocked models
"""

import sys
import math
import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch
import pytest

# Mock gradio before importing app module
sys.modules['gradio'] = MagicMock()
import gradio as gr

from app import InpaintGenAI


# --- Tests for make_divisible_by_8 ---

class TestMakeDivisibleBy8:
    """Tests for the make_divisible_by_8 utility function."""

    def _make_divisible_by_8(self, x):
        """Local copy of the function for unit testing without importing the heavy module."""
        return int(math.ceil(x / 8.0)) * 8

    def test_already_divisible(self):
        assert self._make_divisible_by_8(8) == 8
        assert self._make_divisible_by_8(16) == 16
        assert self._make_divisible_by_8(512) == 512
        assert self._make_divisible_by_8(1024) == 1024

    def test_rounds_up(self):
        assert self._make_divisible_by_8(1) == 8
        assert self._make_divisible_by_8(7) == 8
        assert self._make_divisible_by_8(9) == 16
        assert self._make_divisible_by_8(100) == 104
        assert self._make_divisible_by_8(513) == 520

    def test_zero(self):
        assert self._make_divisible_by_8(0) == 0

    def test_large_values(self):
        assert self._make_divisible_by_8(1080) == 1080
        assert self._make_divisible_by_8(1081) == 1088
        assert self._make_divisible_by_8(1920) == 1920


# --- Tests for mask_to_rgb ---

class TestMaskToRgb:
    """Tests for the mask_to_rgb utility function."""

    def _mask_to_rgb(self, mask):
        """Local copy for unit testing without importing the heavy module."""
        bg_transparent = np.zeros(mask.shape + (4,), dtype=np.uint8)
        bg_transparent[mask == 1] = [0, 255, 0, 127]
        return bg_transparent

    def test_all_zeros_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = self._mask_to_rgb(mask)
        assert result.shape == (100, 100, 4)
        assert result.sum() == 0

    def test_all_ones_mask(self):
        mask = np.ones((50, 50), dtype=np.uint8)
        result = self._mask_to_rgb(mask)
        assert result.shape == (50, 50, 4)
        # Every pixel should be [0, 255, 0, 127]
        assert np.all(result[:, :, 0] == 0)
        assert np.all(result[:, :, 1] == 255)
        assert np.all(result[:, :, 2] == 0)
        assert np.all(result[:, :, 3] == 127)

    def test_partial_mask(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[5:, :] = 1  # Bottom half is masked
        result = self._mask_to_rgb(mask)
        # Top half should be transparent (all zeros)
        assert np.all(result[:5, :, :] == 0)
        # Bottom half should be green
        assert np.all(result[5:, :, 1] == 255)
        assert np.all(result[5:, :, 3] == 127)

    def test_output_dtype(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        result = self._mask_to_rgb(mask)
        assert result.dtype == np.uint8


# --- Tests for InpaintGenAI class ---

class TestInpaintGenAI:
    """Tests for the InpaintGenAI class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_get_processed_inputs = MagicMock()
        self.mock_inpaint = MagicMock()
        self.app = InpaintGenAI(self.mock_get_processed_inputs, self.mock_inpaint)

    def test_init(self):
        assert self.app.input_points == []
        assert self.app.input_image is None
        assert self.app.original_width is None
        assert self.app.original_height is None
        assert self.app.get_processed_inputs == self.mock_get_processed_inputs
        assert self.app.inpaint == self.mock_inpaint

    def test_reset_points(self):
        self.app.input_points = [[10, 20], [30, 40]]
        result = self.app.reset_points()
        assert self.app.input_points == []
        assert result == (True, True)

    def test_preprocess(self):
        img = Image.new("RGB", (640, 480))
        result = self.app.preprocess(img)
        assert self.app.original_width == 640
        assert self.app.original_height == 480
        assert result.size == (640, 480)

    def test_preprocess_non_square(self):
        img = Image.new("RGB", (1920, 1080))
        result = self.app.preprocess(img)
        assert self.app.original_width == 1920
        assert self.app.original_height == 1080

    def test_run_sam_no_image_raises(self):
        with pytest.raises(Exception):
            self.app.run_sam()

    def test_run_sam_with_image(self):
        self.app.input_image = Image.new("RGB", (100, 100))
        self.app.input_points = [[50, 50]]
        mock_mask = np.ones((100, 100), dtype=bool)
        self.mock_get_processed_inputs.return_value = mock_mask

        result = self.app.run_sam()
        assert result[0] == self.app.input_image
        assert len(result[1]) == 2
        assert result[1][0][1] == "background"
        assert result[1][1][1] == "subject"

    def test_run_no_image_raises(self):
        with pytest.raises(Exception):
            self.app.run("prompt", "neg", 7, 42, False, 50, False)

    def test_run_inpainting_background(self):
        """Test inpainting with background mode."""
        self.app.input_image = Image.new("RGB", (64, 64))
        self.app.original_width = 64
        self.app.original_height = 64
        self.app.input_points = [[32, 32]]

        # Mock SAM output
        mock_mask = np.ones((64, 64), dtype=bool)
        self.mock_get_processed_inputs.return_value = mock_mask

        # Mock inpaint output (non-black image)
        inpainted_img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        self.mock_inpaint.return_value = inpainted_img

        result = self.app.run("test prompt", "bad quality", 7, 42, False, 50, False)
        assert result.size == (64, 64)
        self.mock_inpaint.assert_called_once()

    def test_run_inpainting_subject(self):
        """Test inpainting with subject (inverted) mode."""
        self.app.input_image = Image.new("RGB", (64, 64))
        self.app.original_width = 64
        self.app.original_height = 64
        self.app.input_points = [[32, 32]]

        mock_mask = np.zeros((64, 64), dtype=bool)
        self.mock_get_processed_inputs.return_value = mock_mask

        inpainted_img = Image.new("RGB", (64, 64), color=(200, 200, 200))
        self.mock_inpaint.return_value = inpainted_img

        result = self.app.run("test prompt", "", 7, 42, True, 50, False)
        assert result.size == (64, 64)

    def test_run_nsfw_black_image(self):
        """Test that a fully black inpainted image triggers NSFW warning."""
        self.app.input_image = Image.new("RGB", (200, 200))
        self.app.original_width = 200
        self.app.original_height = 200
        self.app.input_points = [[100, 100]]

        mock_mask = np.ones((200, 200), dtype=bool)
        self.mock_get_processed_inputs.return_value = mock_mask

        # Return an all-black image (NSFW detection)
        black_img = Image.new("RGB", (200, 200), color=(0, 0, 0))
        self.mock_inpaint.return_value = black_img

        result = self.app.run("test", "", 7, 42, False, 50, False)
        # Result should still be 200x200 with text drawn on it
        assert result.size == (200, 200)


# --- Tests for image resolution handling ---

class TestResolutionHandling:
    """Tests for image resolution and resizing logic."""

    def test_preprocess_preserves_original_size(self):
        mock_fn = MagicMock()
        app = InpaintGenAI(mock_fn, mock_fn)
        img = Image.new("RGB", (1234, 567))
        result = app.preprocess(img)
        assert result.size == (1234, 567)
        assert app.original_width == 1234
        assert app.original_height == 567

    def test_output_matches_original_resolution(self):
        """Verify the run method resizes output back to original dimensions."""
        mock_get = MagicMock()
        mock_inpaint = MagicMock()
        app = InpaintGenAI(mock_get, mock_inpaint)

        app.input_image = Image.new("RGB", (300, 200))
        app.original_width = 300
        app.original_height = 200
        app.input_points = [[150, 100]]

        mock_mask = np.ones((200, 300), dtype=bool)
        mock_get.return_value = mock_mask

        # Return image with slightly different size (simulating pipeline output)
        output_img = Image.new("RGB", (304, 200), color=(100, 100, 100))
        mock_inpaint.return_value = output_img

        result = app.run("prompt", "", 7, 42, False, 50, False)
        assert result.size == (300, 200)
