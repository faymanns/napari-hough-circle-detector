import numpy as np
import pytest

from napari_hough_circle_detector import CircleDetector


@pytest.fixture
def example_img():
    img = np.zeros((200, 200), dtype=np.uint8)
    xx, yy = np.meshgrid(np.arange(200), np.arange(200))
    for x0, y0, r in (
        (20, 70, 10),
        (80, 120, 30),
        (150, 50, 40),
        (170, 170, 20),
    ):
        rr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
        img[rr < r] = 255
    return img


def test_circle_detector_widget(make_napari_viewer, example_img):
    viewer = make_napari_viewer()
    widget = CircleDetector(viewer)

    assert widget._image_layer_combo.value is None

    # Add an image
    layer_name = "Test"
    viewer.add_image(example_img, name=layer_name)

    # Select the new layer
    widget._image_layer_combo.reset_choices()
    widget._image_layer_combo.value = viewer.layers[layer_name]

    # Median filter
    widget._median_filter_slider.value = 3
    filtered = viewer.layers[f"{layer_name}_median_filtered"]
    assert filtered.data.shape == viewer.layers[layer_name].data.shape

    points_layer = viewer.layers[f"{layer_name}_circles"]
    _validate_detected_circles(points_layer, widget)


def _validate_detected_circles(layer, widget):
    points = layer.data

    # Check minimum distance
    dx = points[:, 0][None, :] - points[:, 0][:, None]
    dy = points[:, 1][None, :] - points[:, 1][:, None]
    dist = np.sqrt(dx**2 + dy**2)[np.triu_indices(dx.shape[0], k=1)]
    assert np.all(dist > widget._min_dist_slider.value)

    # Check minimum radius
    assert np.all(layer.size > widget._min_radius_slider.value)

    # Check maximum radius
    assert np.all(layer.size < widget._max_radius_slider.value)
