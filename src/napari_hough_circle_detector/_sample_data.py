"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""

from __future__ import annotations

import numpy


def _add_circle(img, center, radius):
    """
    Draw a white (255) circle on the provided image.

    Parameters
    ----------
    img : numpy.ndarray
        2D array representing the image.
    cetner : array-like
        The x and y coordinates for the center
        of the circle.
    radius : float
        The radius of the circle.

    Returns
    -------
    img : numpy.ndarray
        2D image array with additional circle.
    """
    height, width = img.shape
    y, x = numpy.meshgrid(
        numpy.arange(height), numpy.arange(width), indexing="ij"
    )
    y = y - center[0]
    x = x - center[1]
    r = numpy.sqrt(y**2 + x**2)
    mask = r <= radius
    img[mask] = 255
    return img


def make_sample_data():
    """
    Generates a sample image with circles.

    Returns
    -------
    List of tuples consiting of the data and a dictionary
    holding the meta data of the sample image.
    """
    data = numpy.ones((512, 512), dtype=numpy.uint8) * 125
    data = _add_circle(data, (100, 100), 50)
    data = _add_circle(data, (120, 230), 70)
    data = _add_circle(data, (400, 80), 20)
    data = _add_circle(data, (300, 300), 100)
    data = _add_circle(data, (200, 400), 30)

    return [(data, {"name": "Sample circles"})]
