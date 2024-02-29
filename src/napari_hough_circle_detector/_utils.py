import cv2
import napari
import numpy as np


def _extract_img_data(
    img: napari.layers.Image, view_only: bool = False
) -> np.ndarray:
    """
    Extracts the img data from a napari image layer and converts it
    to gray scale if the image is RGB.
    """
    if img.multiscale:
        raise ValueError(
            "napari-hough-circle-detector does not support multiscale data"
        )
    img_data = img._data_view if view_only else img.data
    if img.rgb:
        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
    return img_data
