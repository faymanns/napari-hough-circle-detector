import numpy as np

from napari_hough_circle_detector import make_sample_data


def test_sample_data():
    sample_data = make_sample_data()

    # Check the return type
    assert isinstance(sample_data, list)

    # Check that the list entry conforms to the LayerDataTuple
    assert isinstance(sample_data[0], tuple)
    assert isinstance(sample_data[0][0], np.ndarray)
    assert isinstance(sample_data[0][1], dict)

    # Check that image is 2D and non-empty
    assert sample_data[0][0].ndim == 2
    assert sample_data[0][0].min() > 0
